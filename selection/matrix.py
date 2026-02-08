import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from selection.utils import (
    ShardKey,
    ShardValidator,
    get_shard_key,
    is_valid_shard,
    load_score_shard_2d,
    _peek_score_meta,
)


class ShardedScoreMatrix:
    """
    Directory-backed score matrix with shards stored as either:
      - 2D: [batch, hidden_dim]
      - 3D: [batch, layer, hidden_dim]  (select with layer_idx)

    Public contract: `load_scores(path)` returns 2D [batch, feature_dim] always.
    """

    def __init__(
        self,
        dir_path: str,
        device: str = "cuda",
        *,
        shard_validator: ShardValidator = is_valid_shard,
        layer_idx: Optional[int] = None,
    ):
        self.dir_path = Path(dir_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._shard_validator = shard_validator

        # NEW: layer selection for 3D shards
        self.layer_idx: Optional[int] = layer_idx
        self.score_ndim: Optional[int] = None  # 2 or 3 (after peek)
        self.num_layers: int = 1  # 1 for 2D; else arr.shape[1]
        self.hidden_dim: int = 0  # last dim of stored data

        if not self.dir_path.exists():
            raise FileNotFoundError(f"{dir_path} not found")

        self._discover_shards()

        self.shard_map: Dict[ShardKey, np.ndarray] = {}
        self.total_rows = 0
        self.feature_dim = 0
        self.dtype: Optional[np.dtype] = None

        self._load_structure()

    def _discover_shards(self) -> None:
        score_files = sorted(
            [
                p
                for p in self.dir_path.glob("scores*.rank*.np*")
                if self._shard_validator(p)
            ]
        )
        data_files = sorted(
            [
                p
                for p in self.dir_path.glob("data.rank*.parquet")
                if self._shard_validator(p)
            ]
        )

        self.score_files_map: Dict[ShardKey, Path] = {
            k: p for p in score_files if (k := get_shard_key(p))
        }
        self.data_files_map: Dict[ShardKey, Path] = {
            k: p for p in data_files if (k := get_shard_key(p))
        }
        self.common_keys = sorted(set(self.score_files_map) & set(self.data_files_map))

        if not self.common_keys:
            raise RuntimeError(f"No matched shards found in {self.dir_path}")

    # ---- the one loader all code should use ----
    def load_scores(self, p: Path) -> np.ndarray:
        return load_score_shard_2d(p, layer_idx=self.layer_idx, mmap=True)

    def _load_structure(self) -> None:
        # 1) Peek meta from first shard
        first_key = self.common_keys[0]
        first_score = self.score_files_map[first_key]

        ndim, n_layers, hidden_dim, _dtype_raw = _peek_score_meta(first_score)
        self.score_ndim = ndim
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim

        # Determine (feature_dim, dtype) from the actual 2D view we'll use
        view2d = self.load_scores(first_score)  # validates layer_idx vs ndim
        if view2d.ndim != 2:
            raise RuntimeError("load_scores() must always return 2D.")
        self.feature_dim = int(view2d.shape[1])
        self.dtype = view2d.dtype

        # 2) Map indices (Sequential Scan)
        os.environ["POLARS_MAX_THREADS"] = "1"
        all_indices = []

        print("Mapping structure (Sequential)...")
        for key in tqdm(self.common_keys, desc="Scanning Metadata"):
            df = pl.read_parquet(self.data_files_map[key], columns=["idx"])
            idx_arr = df["idx"].to_numpy()

            if idx_arr.size > 0:
                self.shard_map[key] = idx_arr
                all_indices.append(idx_arr)

        if not all_indices:
            raise ValueError("Shards found but empty.")

        global_idx = np.concatenate(all_indices)
        self.total_rows = int(global_idx.max()) + 1

        if self.score_ndim == 3:
            print(
                f"Matrix Ready: ({self.total_rows}, {self.feature_dim}) "
                f"from 3D shards using layer_idx={self.layer_idx} on {self.device}"
            )
        else:
            print(
                f"Matrix Ready: ({self.total_rows}, {self.feature_dim}) on {self.device}"
            )

    @torch.no_grad()
    def matmul(self, other_matrix: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Computes (Matrix @ other_matrix).
        """
        other = (
            torch.from_numpy(other_matrix)
            if isinstance(other_matrix, np.ndarray)
            else other_matrix
        )

        target_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        )
        other = other.to(dtype=target_dtype, device=self.device)

        if other.shape[0] != self.feature_dim:
            raise ValueError(
                f"Shape mismatch: Matrix(..., {self.feature_dim}) vs Input{other.shape}"
            )

        is_vector = other.ndim == 1
        out_dim = 1 if is_vector else other.shape[1]
        output_shape = (self.total_rows,) if is_vector else (self.total_rows, out_dim)

        print(f"Allocating GPU buffer {output_shape} ({target_dtype})...")
        result = torch.zeros(output_shape, dtype=target_dtype, device=self.device)

        print(f"Computing on {self.device}...")

        for key in tqdm(self.common_keys, desc="GPU Matrix Mult"):
            indices = self.shard_map.get(key)
            if indices is None:
                continue

            shard_np = self.load_scores(self.score_files_map[key])  # ALWAYS [n,d]
            shard_gpu = torch.from_numpy(np.asarray(shard_np)).to(
                device=self.device, dtype=target_dtype
            )

            partial_scores = torch.matmul(shard_gpu, other)

            idx_tensor = torch.tensor(indices, device=self.device)
            result[idx_tensor] = partial_scores

            del shard_np, shard_gpu, partial_scores, idx_tensor

        return result

    def materialize(self) -> torch.Tensor:
        """
        Reconstructs the full matrix in memory as a single PyTorch tensor.
        WARNING: Only use this for small matrices that fit in RAM/VRAM.
        """
        print(
            f"Materializing full matrix ({self.total_rows}, {self.feature_dim}) on {self.device}..."
        )

        target_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )

        full_matrix = torch.zeros(
            (self.total_rows, self.feature_dim), dtype=target_dtype, device=self.device
        )

        for key in tqdm(self.common_keys, desc="Materializing"):
            indices = self.shard_map.get(key)
            if indices is None:
                continue

            shard_np = self.load_scores(self.score_files_map[key])  # ALWAYS [n,d]
            shard_tensor = torch.from_numpy(np.asarray(shard_np)).to(
                device=self.device, dtype=target_dtype
            )

            idx_tensor = torch.tensor(indices, device=self.device)
            full_matrix[idx_tensor] = shard_tensor

            del shard_np, shard_tensor, idx_tensor

        return full_matrix

    def compute_feature_mean(self) -> np.ndarray:
        """
        Compute mean activation/score per feature across all rows.
        """
        acc = np.zeros(self.feature_dim, dtype=np.float64)
        total = 0

        print("Computing feature means...")
        for key in tqdm(self.common_keys, desc="Computing Feature Mean"):
            indices = self.shard_map.get(key)
            if indices is None:
                continue

            shard_np = self.load_scores(self.score_files_map[key])  # ALWAYS [n,d]
            shard_arr = np.asarray(shard_np, dtype=np.float64)

            acc += shard_arr.sum(axis=0)
            total += shard_arr.shape[0]

            del shard_np, shard_arr

        if total == 0:
            print("Warning: No data found, returning zeros")
            return acc

        return acc / total
