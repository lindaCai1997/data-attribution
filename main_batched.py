# main.py (upgraded) — updated to support the new "all_v2"/"all_bias_mask" wrapper.
# Key changes:
# - Output dirs + accumulators are now driven by a METHOD_OUTPUTS mapping
# - Unpacks 5 outputs for all_v2 (instead of 4 for old all)
# - parse_args no longer asserts method == "all"; default is now "all_v2"

import argparse, json, os, time, tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from method import build_method
from utils import assistant_keep_mask_last_span, find_layer, save_scores
from selection.utils import make_dir_wide_permissions


# ---------- Dataset ----------
class ChatPairDataset(Dataset):
    def __init__(self, path: str, max_num_data: int = 10000):
        self.rows = []
        if path.endswith(".parquet"):
            df = pl.read_parquet(path)
            for item in df.iter_rows(named=True):
                if "treatment_messages" in item and "control_messages" in item:
                    self.rows.append(
                        {
                            "treatment_messages": (
                                json.loads(item["treatment_messages"])
                                if isinstance(item["treatment_messages"], str)
                                else item["treatment_messages"]
                            ),
                            "control_messages": (
                                json.loads(item["control_messages"])
                                if isinstance(item["control_messages"], str)
                                else item["control_messages"]
                            ),
                        }
                    )
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "treatment_messages" in item and "control_messages" in item:
                        self.rows.append(
                            {
                                "treatment_messages": item["treatment_messages"],
                                "control_messages": item["control_messages"],
                            }
                        )
        if not self.rows:
            raise ValueError("No valid items found.")
        self.rows = self.rows[:max_num_data]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "idx": idx,
            "treatment_messages": r["treatment_messages"],
            "control_messages": r["control_messages"],
        }


# ---------- Tokenization (left pad / left truncate) ----------
def tokenize_chat_batch(
    tokenizer, batch_msgs: List[List[Dict]], max_tokens: int
) -> Dict[str, torch.Tensor]:
    return tokenizer.apply_chat_template(
        batch_msgs,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
        add_generation_prompt=False,
        return_dict=True,
    )


def collate_fn_builder(tokenizer, max_tokens: int):
    tok_name = tokenizer.name_or_path

    def _collate(batch_rows: List[Dict]):
        idxs = [r["idx"] for r in batch_rows]
        t_batch = [r["treatment_messages"] for r in batch_rows]
        c_batch = [r["control_messages"] for r in batch_rows]
        t_out = tokenize_chat_batch(tokenizer, t_batch, max_tokens)
        c_out = tokenize_chat_batch(tokenizer, c_batch, max_tokens)
        t_keep = assistant_keep_mask_last_span(
            t_out["input_ids"], tok_name, t_out["attention_mask"]
        )
        c_keep = assistant_keep_mask_last_span(
            c_out["input_ids"], tok_name, c_out["attention_mask"]
        )
        return {
            "idx": torch.tensor(idxs, dtype=torch.long),
            "treatment_tokens": t_out["input_ids"],
            "treatment_attention_mask": t_out["attention_mask"],
            "treatment_asst_token_mask": t_keep,
            "control_tokens": c_out["input_ids"],
            "control_attention_mask": c_out["attention_mask"],
            "control_asst_token_mask": c_keep,
        }

    return _collate


# ---------- Hooks (activations + optional grad_out at a chosen block output) ----------
class LayerHook:
    def __init__(self, layer: nn.Module, capture_grads: bool):
        self.layer = layer
        self.capture_grads = capture_grads
        self.fwd_h = None
        self.bwd_h = None
        self.act = None  # [B,T,D]
        self.input = None  # [B,T,D]
        self.grad = None  # [B,T,D] or None

    def _on_fwd(self, _m, _in, out):
        y = out[0] if isinstance(out, tuple) else out
        x = _in[0] if isinstance(_in, tuple) else _in
        self.act = y
        self.input = x
        if self.capture_grads and y.requires_grad:
            y.retain_grad()

    def _on_bwd(self, _m, grad_in, grad_out):
        if self.capture_grads:
            self.grad = grad_out[0]

    def attach(self):
        self.fwd_h = self.layer.register_forward_hook(self._on_fwd)
        if self.capture_grads:
            self.bwd_h = self.layer.register_full_backward_hook(self._on_bwd)

    def detach(self):
        if self.fwd_h:
            self.fwd_h.remove()
        if self.bwd_h:
            self.bwd_h.remove()
        self.fwd_h = self.bwd_h = None

    def get(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.act is None:
            raise RuntimeError("Activation not captured.")
        return self.input, self.act, self.grad

    def clear(self):
        self.input = None
        self.act = None
        self.grad = None


# ---------- Loss (masked next-token CE over keep-mask) ----------
def masked_lm_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[~keep_mask] = -100
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ---------- Args ----------
@dataclass
class Args:
    data: str
    output_dir: str
    method: str
    scores_format: str
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_id: Optional[str] = None
    layer_index: Optional[int] = None
    batch_size: int = 2
    max_tokens: int = 1024
    max_num_data: int = 10000
    dtype: str = "bfloat16"
    device: Optional[str] = None
    is_train_data: bool = False
    scores_dtype: str = "float16"  # float32 | float16 (ignored for int8)
    sparse_threshold: float = 0.0  # threshold for pruning small values in sparse_csr
    topk: int = 0  # top-K per row for scores-format=topk
    save_every: int = 1000  # number of batches between partial saves; 0 = only at end


# ---------- Runner ----------
def run(a: Args):
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(
        os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count())))
    )
    print(f"rank: {rank}, world: {world}, local: {local}")

    # device/dtype
    device = (
        torch.device(a.device)
        if a.device
        else torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[a.dtype]

    # tokenizer/model (left pad/left truncate)
    tok_id = a.tokenizer_id or a.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        a.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    hidden_size = model.config.hidden_size

    # layer + hook with optional gradients
    n_layers = model.config.num_hidden_layers
    L = a.layer_index if a.layer_index is not None else (n_layers - 1)
    layer_mod = find_layer(model, L)

    # method
    method = build_method(a.method)
    needs_grads = bool(getattr(method, "requires_grads", True))
    needs_control = bool(getattr(method, "requires_control", True))

    hook = LayerHook(layer_mod, capture_grads=needs_grads)
    hook.attach()

    # data loader (sharded)
    ds = ChatPairDataset(a.data, a.max_num_data)
    sampler = DistributedSampler(ds, shuffle=False, drop_last=True)
    collate = collate_fn_builder(tokenizer, a.max_tokens)
    dl = DataLoader(
        ds,
        batch_size=a.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    # --- NEW: define per-method outputs and directories ---
    method_key = (a.method or "").lower()

    # Old "all" returned 4 values in your main loop; new all_v2 returns 5.
    METHOD_OUTPUTS = {
        # new wrapper names (as registered in the code I gave you)
        "all_v2": [
            "residual_change_treatment_with_bias",
            "residual_input_treatment",
            "residual_treatment",
            "residual_change_with_bias_and_mask",
            "residual_change_treatment_with_mask",
        ],
        "all_bias_mask": [
            "residual_change_treatment_with_bias",
            "residual_input_treatment",
            "residual_treatment",
            "residual_change_with_bias_and_mask",
            "residual_change_treatment_with_mask",
        ],
        "all_extended": [
            "residual_change_treatment_with_bias",
            "residual_input_treatment",
            "residual_treatment",
            "residual_change_with_bias_and_mask",
            "residual_change_treatment_with_mask",
        ],
        # keep support for old "all" if you still want it
        "all": [
            "residual_treatment",
            "residual_diff",
            "residual_change",
            "residual_change_treatment",
        ],
        "all_base": [
            "residual_diff",
            "residual_change_treatment",
            "residual_change",
        ],
        "all_v3": [
            "residual_diff",
            "residual_change_treatment",
            "residual_change",
            "residual_change_treatment_with_bias",
            "residual_input_treatment",
            "residual_treatment",
            "residual_change_with_bias_and_mask",
            "residual_change_treatment_with_mask",
            "residual_change_treatment_with_mask_corrected",
            "grad_act_treatment",
            "residual_control",
        ]
    }

    output_names = METHOD_OUTPUTS.get(method_key)
    if output_names is None:
        raise ValueError(
            f"Unsupported method='{a.method}'. Expected one of: {sorted(METHOD_OUTPUTS.keys())}"
        )

    out_dirs: Dict[str, Path] = {
        name: (Path(a.output_dir) / name) for name in output_names
    }

    if rank == 0:
        for p in out_dirs.values():
            make_dir_wide_permissions(p, mode=0o777)

    # accumulators per rank (now dictionary-driven)
    local_idx: List[torch.Tensor] = []
    local_scores: Dict[str, List[torch.Tensor]] = {name: [] for name in output_names}

    step = 0
    shard = 0
    save_info = None

    t0 = time.time()
    num_batches = len(dl)
    for ibatch, batch in enumerate(tqdm.tqdm(dl, disable=rank != 0)):
        # CONTROL pass
        hook.clear()
        if needs_control:
            c_ids = batch["control_tokens"].to(device)
            c_att = batch["control_attention_mask"].to(device)
            c_keep = batch["control_asst_token_mask"].to(device)
            out_c = model(
                input_ids=c_ids, attention_mask=c_att, use_cache=False, return_dict=True
            )
            if needs_grads:
                loss_c = masked_lm_loss(out_c.logits, c_ids, c_keep)
                model.zero_grad(set_to_none=True)
                loss_c.backward()
        else:
            # still need tensors for method signature; just reuse treatment later if needed
            c_ids = c_att = c_keep = None
        inputs_c, acts_c, grads_c = hook.get() if needs_control else (None, None, None)

        # TREATMENT pass
        hook.clear()
        t_ids = batch["treatment_tokens"].to(device)
        t_att = batch["treatment_attention_mask"].to(device)
        t_keep = batch["treatment_asst_token_mask"].to(device)
        out_t = model(
            input_ids=t_ids, attention_mask=t_att, use_cache=False, return_dict=True
        )
        if needs_grads:
            loss_t = masked_lm_loss(out_t.logits, t_ids, t_keep)
            model.zero_grad(set_to_none=True)
            loss_t.backward()
        inputs_t, acts_t, grads_t = hook.get()

        # method call (generic)
        with torch.no_grad():
            outs = method.forward(
                input_treat=inputs_t,
                acts_treat=acts_t,
                grads_treat=grads_t,
                mask_treat=t_keep,
                input_ctrl=inputs_c if needs_control else inputs_t,
                acts_ctrl=acts_c if needs_control else acts_t,
                grads_ctrl=grads_c if needs_control else grads_t,
                mask_ctrl=c_keep if needs_control else t_keep,
                tokens_treat=t_ids,
                attn_treat=t_att,
                tokens_ctrl=c_ids if needs_control else t_ids,
                attn_ctrl=c_att if needs_control else t_att,
                sae={},
                llm=model,
                tokenizer=tokenizer,
            )

            if not isinstance(outs, (tuple, list)):
                outs = (outs,)
            if len(outs) != len(output_names):
                raise RuntimeError(
                    f"Method '{a.method}' returned {len(outs)} outputs, "
                    f"but main.py expected {len(output_names)}: {output_names}"
                )

        for name, tensor in zip(output_names, outs):
            local_scores[name].append(tensor.detach())

        local_idx.append(batch["idx"].to(device))
        model.zero_grad(set_to_none=True)
        step += 1

        # save scores every N batches
        if ibatch + 1 == num_batches or (a.save_every and step % a.save_every == 0):
            idx_np = torch.cat(local_idx, dim=0).long().cpu().numpy()
            order = np.argsort(idx_np)

            for name in output_names:
                out_dir_local = out_dirs[name]
                scores_np = (
                    torch.cat(local_scores[name], dim=0).float().cpu().numpy()[order]
                )
                save_scores(
                    out_dir_local,
                    scores_np,
                    fmt=a.scores_format,
                    dtype=a.scores_dtype,
                    sparse_threshold=a.sparse_threshold,
                    topk=a.topk,
                    suffix=f".rank{rank}.part{shard}",
                )
                local_scores[name].clear()

                rows = []
                for i in idx_np[order].tolist():
                    item = ds[i]
                    rows.append(
                        {
                            "idx": int(i),
                            "treatment_messages": json.dumps(
                                item["treatment_messages"], ensure_ascii=False
                            ),
                            "control_messages": json.dumps(
                                item["control_messages"], ensure_ascii=False
                            ),
                        }
                    )
                pl.DataFrame(rows).write_parquet(
                    out_dir_local / f"data.rank{rank}.part{shard}.parquet"
                )

            shard += 1
            local_idx.clear()
            for name in output_names:
                assert len(local_scores[name]) == 0
            assert len(local_idx) == 0

    for name in output_names:
        assert len(local_scores[name]) == 0
    assert len(local_idx) == 0
    hook.detach()

    if dist.get_rank() == 0:
        # per-rank metadata
        meta = {
            **a.__dict__,
            "tokenizer_id": a.tokenizer_id or a.model_id,
            "layer_index": L,
            "num_layers": n_layers,
            "hidden_size": hidden_size,
            "device": str(device),
            "dataset_size": len(ds),
            "requires_grads": needs_grads,
            "elapsed_sec": round(time.time() - t0, 2),
            "world_size": world,
            "save_info": save_info,
        }
        for name in output_names:
            with open(out_dirs[name] / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    print(f"[OK][rank {rank}] wrote shard(s) to: " + ", ".join(str(out_dirs[n]) for n in output_names))
    dist.destroy_process_group()


def parse_args() -> Args:
    p = argparse.ArgumentParser("Distributed, hook-based data attribution")
    p.add_argument(
        "--data",
        required=True,
        help="JSONL with {treatment_messages, control_messages}",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--tokenizer-id", default=None)
    p.add_argument("--layer-index", type=int, required=True)

    # NEW default: all_v2 (your new wrapper)
    p.add_argument("--method", default="all_v2")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--is-train-data",
        action="store_true",
        help="Whether or not the code is run on training or eval data",
    )
    p.add_argument("--max-num-data", type=int, default=10000)
    p.add_argument(
        "--scores-format",
        choices=["npy", "npz", "sparse_csr", "int8_feature", "int8_row", "topk"],
        default="npy",
    )
    p.add_argument("--scores-dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--sparse-threshold", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save scores every N batches (0 = only once at end)",
    )
    a = p.parse_args()
    return Args(**vars(a))


if __name__ == "__main__":
    run(parse_args())