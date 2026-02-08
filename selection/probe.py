# selection/probe.py
#
# Sharded binary probe training + scoring over sharded score directories.
# Fixes:
#  - Standardize transform device/dtype mismatch during GPU inference
#  - build_probe accepts **kwargs (avoids TypeError in load)
#  - Save/load includes probe hparams and restores best val weights
#  - Inference output initialized safely (no uninitialized garbage)
#  - Stricter shard discovery that ignores ".partN" shards (without editing utils.py)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import copy
import re
import numpy as np
import torch
import torch.nn as nn

from selection.utils import (
    is_valid_shard,
    load_vectors_from_dir,
    stack_vectors,
)
from torch.utils.data import DataLoader, TensorDataset, random_split
from selection.matrix import ShardedScoreMatrix


ArrayLike = Union[np.ndarray, torch.Tensor]
VectorsInput = Union[Sequence[ArrayLike], ArrayLike]


# ----------------------------
# Configs
# ----------------------------


@dataclass(frozen=True)
class ProbeTrainConfig:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.1
    seed: int = 42
    patience: int = 3  # early stopping patience (val loss)
    use_amp: bool = True  # autocast during training if on CUDA
    max_per_class: Optional[int] = None  # optional subsample per class (eval only)


@dataclass(frozen=True)
class ProbeInferConfig:
    shard_row_batch_size: int = 16384  # chunk within a shard to limit VRAM
    use_amp: bool = True  # autocast during inference if on CUDA
    return_logits: bool = True  # if False -> returns sigmoid probs
    fill_value: float = float("-inf")  # safe init for rows not present


# ----------------------------
# Input transforms
# ----------------------------


class InputTransform:
    """Base class for transforms applied to vectors before feeding probe."""

    def fit(self, x: torch.Tensor) -> None:
        return

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return


class L2Normalize(InputTransform):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps)

    def state_dict(self) -> Dict[str, Any]:
        return {"eps": self.eps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.eps = float(state.get("eps", self.eps))


class Standardize(InputTransform):
    """
    Standardize using mean/std fit on training eval vectors.

    FIX: mean/std are stored on CPU, but during transform we move them to x.device/x.dtype.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean: Optional[torch.Tensor] = None  # stored CPU
        self.std: Optional[torch.Tensor] = None  # stored CPU

    def fit(self, x: torch.Tensor) -> None:
        x_cpu = x.detach().cpu()
        self.mean = x_cpu.mean(dim=0, keepdim=True)
        self.std = x_cpu.std(dim=0, keepdim=True).clamp(min=self.eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardize.transform called before fit().")
        mean = self.mean.to(device=x.device, dtype=x.dtype, non_blocking=True)
        std = self.std.to(device=x.device, dtype=x.dtype, non_blocking=True)
        return (x - mean) / std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "mean": None if self.mean is None else self.mean.detach().cpu(),
            "std": None if self.std is None else self.std.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.eps = float(state.get("eps", self.eps))
        mean = state.get("mean", None)
        std = state.get("std", None)
        self.mean = None if mean is None else mean.detach().cpu()
        self.std = None if std is None else std.detach().cpu()


_TRANSFORM_REGISTRY: Dict[str, Any] = {
    "none": InputTransform,
    "identity": InputTransform,
    "l2": L2Normalize,
    "standardize": Standardize,
}

_TRANSFORM_CLASSNAME_REGISTRY: Dict[str, Any] = {
    "InputTransform": InputTransform,
    "L2Normalize": L2Normalize,
    "Standardize": Standardize,
}


def make_transform(name: Optional[str]) -> InputTransform:
    if name is None:
        return InputTransform()
    key = name.lower()
    cls = _TRANSFORM_REGISTRY.get(key, None)
    if cls is None:
        raise ValueError(f"Unknown transform: {name}")
    return cls()


def make_transform_from_classname(classname: str) -> InputTransform:
    cls = _TRANSFORM_CLASSNAME_REGISTRY.get(classname, None)
    if cls is None:
        raise ValueError(f"Unknown transform class in checkpoint: {classname}")
    return cls()


# ----------------------------
# Probe models
# ----------------------------


class LinearBinaryProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # logits [B,1]


class MLPBinaryProbe(nn.Module):
    def __init__(
        self, d_in: int, hidden: Sequence[int] = (512, 256), dropout: float = 0.0
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits [B,1]


# ----------------------------
# Base sharded probe scorer
# ----------------------------


class BaseShardedBinaryProbeScorer(ShardedScoreMatrix):
    """
    Trains a binary probe on eval vectors (pos vs neg) and scores a sharded directory.

    - Uses strict shard validator via parent (Option B).
    - Supports 2D shards [batch, hidden_dim] and 3D shards [batch, layer, hidden_dim]
      by passing `layer_idx` (selected layer) into the parent and helper loader.
    """

    PROBE_NAME: str = "base"
    CHECKPOINT_VERSION: int = 1

    def __init__(
        self,
        train_dir: str,
        *,
        device: str = "cuda",
        input_transform: Optional[str] = "l2",
        layer_idx: Optional[int] = None,
    ):
        super().__init__(
            train_dir,
            device=device,
            shard_validator=is_valid_shard,
            layer_idx=layer_idx,
        )
        self.transform: InputTransform = make_transform(input_transform)
        self.probe: Optional[nn.Module] = None

    # ---- subclass hooks ----

    def build_probe(self, d_in: int, **kwargs: Any) -> nn.Module:
        """Subclasses should return an nn.Module producing logits of shape [B,1]."""
        raise NotImplementedError

    def _probe_hparams(self) -> Dict[str, Any]:
        return {}

    def _load_probe_hparams(self, hparams: Dict[str, Any]) -> None:
        return

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        if self.probe is None:
            raise RuntimeError("Probe not trained/loaded yet.")
        return self.probe(x).squeeze(-1)

    # ---- training API ----

    def fit_from_dirs(
        self,
        *,
        pos_eval_dir: str,
        neg_eval_dir: str,
        cfg: ProbeTrainConfig = ProbeTrainConfig(),
    ) -> "BaseShardedBinaryProbeScorer":
        pos = load_vectors_from_dir(
            pos_eval_dir,
            shard_validator=is_valid_shard,
            layer_idx=self.layer_idx,
            max_examples=cfg.max_per_class,
            seed=cfg.seed,
            dtype=torch.float32,
        )
        neg = load_vectors_from_dir(
            neg_eval_dir,
            shard_validator=is_valid_shard,
            layer_idx=self.layer_idx,
            max_examples=cfg.max_per_class,
            seed=cfg.seed,
            dtype=torch.float32,
        )
        return self.fit_from_vectors(pos_vectors=pos, neg_vectors=neg, cfg=cfg)

    def fit_from_vectors(
        self,
        *,
        pos_vectors: VectorsInput,
        neg_vectors: VectorsInput,
        cfg: ProbeTrainConfig = ProbeTrainConfig(),
    ) -> "BaseShardedBinaryProbeScorer":
        x_pos = stack_vectors(pos_vectors, dtype=torch.float32)
        x_neg = stack_vectors(neg_vectors, dtype=torch.float32)

        if x_pos.shape[1] != x_neg.shape[1]:
            raise ValueError(
                f"Dim mismatch: pos {tuple(x_pos.shape)} vs neg {tuple(x_neg.shape)}"
            )
        if x_pos.shape[1] != self.feature_dim:
            raise ValueError(
                f"Probe input dim={x_pos.shape[1]} != train feature_dim={self.feature_dim}. "
                "Are train/eval vectors produced by the same pipeline?"
            )

        y_pos = torch.ones((x_pos.shape[0],), dtype=torch.float32)
        y_neg = torch.zeros((x_neg.shape[0],), dtype=torch.float32)

        x = torch.cat([x_pos, x_neg], dim=0)  # CPU float32
        y = torch.cat([y_pos, y_neg], dim=0)  # CPU float32

        self.transform.fit(x)
        x = self.transform.transform(x)

        self.probe = self.build_probe(d_in=self.feature_dim).to(self.device)
        self.train_probe(x=x, y=y, cfg=cfg)
        return self

    def train_probe(
        self, *, x: torch.Tensor, y: torch.Tensor, cfg: ProbeTrainConfig
    ) -> None:
        if self.probe is None:
            raise RuntimeError("Probe not initialized. Call fit_* first.")

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        dataset = TensorDataset(x, y)

        val_loader: Optional[DataLoader] = None
        if cfg.val_split and cfg.val_split > 0:
            n_val = int(round(len(dataset) * cfg.val_split))
            n_val = max(1, n_val) if len(dataset) >= 2 else 0
            n_train = len(dataset) - n_val
            if n_val > 0 and n_train > 0:
                train_ds, val_ds = random_split(
                    dataset,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(cfg.seed),
                )
                train_loader = DataLoader(
                    train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
                )
                val_loader = DataLoader(
                    val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
                )
            else:
                train_loader = DataLoader(
                    dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
                )
                val_loader = None
        else:
            train_loader = DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
            )

        n_pos = float(y.sum().item())
        n_neg = float(y.numel() - y.sum().item())
        pos_weight = torch.tensor(
            n_neg / max(n_pos, 1.0), device=self.device, dtype=torch.float32
        )

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(
            self.probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        amp_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        for epoch in range(cfg.epochs):
            self.probe.train()
            running = 0.0
            n_seen = 0

            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = self.forward_batch(xb)
                        loss = loss_fn(logits, yb)
                else:
                    logits = self.forward_batch(xb)
                    loss = loss_fn(logits, yb)

                loss.backward()
                opt.step()

                running += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)

            train_loss = running / max(n_seen, 1)

            val_loss = None
            if val_loader is not None:
                self.probe.eval()
                v_running = 0.0
                v_seen = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                logits = self.forward_batch(xb)
                                loss = loss_fn(logits, yb)
                        else:
                            logits = self.forward_batch(xb)
                            loss = loss_fn(logits, yb)
                        v_running += float(loss.item()) * xb.size(0)
                        v_seen += xb.size(0)
                val_loss = v_running / max(v_seen, 1)

                improved = (val_loss + 1e-8) < best_val
                if improved:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.probe.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        break

            if (
                epoch == 0
                or epoch == cfg.epochs - 1
                or (val_loss is not None and bad_epochs == 0)
            ):
                if val_loss is None:
                    print(
                        f"[probe:{self.PROBE_NAME}] epoch {epoch+1}/{cfg.epochs} train_loss={train_loss:.4f}"
                    )
                else:
                    print(
                        f"[probe:{self.PROBE_NAME}] epoch {epoch+1}/{cfg.epochs} "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                    )

        if best_state is not None:
            self.probe.load_state_dict(best_state)
        self.probe.eval()

    # ---- inference / scoring ----
    def topk_train(
        self,
        k: int,
        *,
        infer_cfg: ProbeInferConfig = ProbeInferConfig(),
    ) -> np.ndarray:
        scores = self.forward_dir(cfg=infer_cfg)  # (total_rows,)
        top_idx = torch.topk(scores, k).indices.detach().cpu().numpy().astype(np.int64)
        return top_idx

    @torch.no_grad()
    def forward_dir(
        self,
        dir_path: Optional[str] = None,
        *,
        cfg: ProbeInferConfig = ProbeInferConfig(),
    ) -> torch.Tensor:
        if self.probe is None:
            raise RuntimeError("Probe not trained/loaded yet.")

        if dir_path is None:
            matrix = self
        else:
            p = Path(dir_path).resolve()
            if p == self.dir_path.resolve():
                matrix = self
            else:
                matrix = ShardedScoreMatrix(
                    str(p),
                    device=str(self.device),
                    shard_validator=is_valid_shard,
                    layer_idx=self.layer_idx,
                )

        return self._forward_sharded_matrix(matrix, cfg=cfg)

    @torch.no_grad()
    def _forward_sharded_matrix(
        self, matrix: ShardedScoreMatrix, *, cfg: ProbeInferConfig
    ) -> torch.Tensor:
        if self.probe is None:
            raise RuntimeError("Probe not trained/loaded yet.")

        if next(self.probe.parameters()).device != matrix.device:
            self.probe.to(matrix.device)

        use_amp = bool(cfg.use_amp and matrix.device.type == "cuda")
        amp_dtype = (
            torch.bfloat16
            if (matrix.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        result = torch.full(
            (matrix.total_rows,),
            cfg.fill_value,
            dtype=torch.float32,
            device=matrix.device,
        )
        self.probe.eval()

        for key in matrix.common_keys:
            indices = matrix.shard_map.get(key)
            if indices is None:
                continue

            # Always 2D because the matrix loader handles layer selection
            shard_np = matrix.load_scores(matrix.score_files_map[key])  # [n,d]
            n = shard_np.shape[0]
            idx_full = torch.from_numpy(indices).to(device=matrix.device)

            for start in range(0, n, cfg.shard_row_batch_size):
                end = min(start + cfg.shard_row_batch_size, n)

                x = torch.from_numpy(np.asarray(shard_np[start:end])).to(
                    device=matrix.device, dtype=torch.float32
                )
                x = self.transform.transform(x)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = self.forward_batch(x)
                else:
                    logits = self.forward_batch(x)

                scores = logits if cfg.return_logits else torch.sigmoid(logits)
                result[idx_full[start:end]] = scores.to(dtype=torch.float32)

                del x, logits, scores

            del shard_np, idx_full

        return result

    # ---- persistence (optional: include layer_idx) ----

    def save(self, path: Union[str, Path]) -> None:
        if self.probe is None:
            raise RuntimeError("No probe to save.")
        path = Path(path)
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "scorer_class": self.__class__.__name__,
            "probe_name": self.PROBE_NAME,
            "feature_dim": int(self.feature_dim),
            "layer_idx": self.layer_idx,
            "probe_hparams": self._probe_hparams(),
            "probe_state_dict": self.probe.state_dict(),
            "transform_class": self.transform.__class__.__name__,
            "transform_state": self.transform.state_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        *,
        train_dir: str,
        path: Union[str, Path],
        device: str = "cuda",
        input_transform: Optional[str] = None,
        override_probe_hparams: Optional[Dict[str, Any]] = None,
        layer_idx: Optional[int] = None,
    ) -> "BaseShardedBinaryProbeScorer":
        payload = torch.load(Path(path), map_location="cpu")

        ckpt_layer = payload.get("layer_idx", None)
        use_layer = layer_idx if layer_idx is not None else ckpt_layer

        scorer = cls(
            train_dir=train_dir,
            device=device,
            input_transform=(input_transform or "none"),
            layer_idx=use_layer,
        )

        if int(payload.get("feature_dim")) != int(scorer.feature_dim):
            raise ValueError(
                f"Checkpoint feature_dim={payload.get('feature_dim')} but train_dir feature_dim={scorer.feature_dim}"
            )

        if input_transform is None:
            t_class = payload.get("transform_class", "InputTransform")
            scorer.transform = make_transform_from_classname(t_class)
            scorer.transform.load_state_dict(payload.get("transform_state", {}))
        else:
            scorer.transform = make_transform(input_transform)

        hparams = dict(payload.get("probe_hparams", {}) or {})
        if override_probe_hparams:
            hparams.update(override_probe_hparams)
        scorer._load_probe_hparams(hparams)

        scorer.probe = scorer.build_probe(d_in=scorer.feature_dim, **hparams).to(
            scorer.device
        )
        scorer.probe.load_state_dict(payload["probe_state_dict"])
        scorer.probe.eval()
        return scorer


# ----------------------------
# Concrete scorers
# ----------------------------


class LinearProbeScorer(BaseShardedBinaryProbeScorer):
    PROBE_NAME = "linear"

    def build_probe(self, d_in: int, **kwargs: Any) -> nn.Module:
        return LinearBinaryProbe(d_in=d_in)

    def _probe_hparams(self) -> Dict[str, Any]:
        return {}

    def _load_probe_hparams(self, hparams: Dict[str, Any]) -> None:
        return


class MLPProbeScorer(BaseShardedBinaryProbeScorer):
    PROBE_NAME = "mlp"

    def __init__(
        self,
        train_dir: str,
        *,
        layer_idx: Optional[int] = None,
        device: str = "cuda",
        input_transform: Optional[str] = "l2",
        hidden: Sequence[int] = (512, 256),
        dropout: float = 0.0,
    ):
        super().__init__(
            train_dir=train_dir,
            layer_idx=layer_idx,
            device=device,
            input_transform=input_transform,
        )
        self.hidden = tuple(int(x) for x in hidden)
        self.dropout = float(dropout)

    def build_probe(self, d_in: int, **kwargs: Any) -> nn.Module:
        # Allow reconstruction from checkpoint-provided hparams
        hidden = kwargs.get("hidden", self.hidden)
        dropout = kwargs.get("dropout", self.dropout)
        return MLPBinaryProbe(
            d_in=d_in, hidden=tuple(int(x) for x in hidden), dropout=float(dropout)
        )

    def _probe_hparams(self) -> Dict[str, Any]:
        return {"hidden": list(self.hidden), "dropout": self.dropout}

    def _load_probe_hparams(self, hparams: Dict[str, Any]) -> None:
        if "hidden" in hparams and hparams["hidden"] is not None:
            self.hidden = tuple(int(x) for x in hparams["hidden"])
        if "dropout" in hparams and hparams["dropout"] is not None:
            self.dropout = float(hparams["dropout"])
