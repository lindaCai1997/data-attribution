# main_trak.py
# Main entry point for TRAK-inspired weight gradient attribution.
# Computes projected weight gradients and stores them for later attribution.
#
# Launch: torchrun --nproc_per_node=NUM_GPUS main_trak.py ...args...
#
# Example:
#   torchrun --nproc_per_node=1 main_trak.py \
#       --data /path/to/train.jsonl \
#       --output-dir /path/to/train_vectors \
#       --is-train-data

import argparse
import json
import os
import time
import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from trak_method import (
    TRAKConfig,
    WeightGradientComputer,
    masked_lm_loss,
    estimate_memory_usage,
)
from utils import assistant_keep_mask_last_span, save_scores
from selection.utils import make_dir_wide_permissions


# ---------- Dataset (treatment only for TRAK) ----------
class TRAKDataset(Dataset):
    """
    Dataset for TRAK attribution.

    Unlike the activation-based methods, TRAK only needs treatment data
    (no control pass). We load treatment_messages from JSONL/Parquet files.
    """

    def __init__(self, path: str, max_num_data: int = -1):
        self.rows = []
        if path.endswith(".parquet"):
            df = pl.read_parquet(path)
            for item in df.iter_rows(named=True):
                if "treatment_messages" in item and "control_messages" in item:
                    row_data = {
                        "treatment_messages": (
                            json.loads(item["treatment_messages"])
                            if isinstance(item["treatment_messages"], str)
                            else item["treatment_messages"]
                        ),
                        "control_messages": (
                            json.loads(item["control_messages"])
                            if isinstance(item["control_messages"], str)
                            else item["control_messages"]
                        )
                    }
                    self.rows.append(row_data)
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "treatment_messages" in item and "control_messages" in item:
                        row_data = {"treatment_messages": item["treatment_messages"], "control_messages": item["control_messages"]}
                        self.rows.append(row_data)

        if not self.rows:
            raise ValueError("No valid items found in dataset.")

        if max_num_data > 0 and len(self.rows) > max_num_data:
            self.rows = self.rows[:max_num_data]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "idx": idx,
            "treatment_messages": r["treatment_messages"],
            "control_messages": r["control_messages"]
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
        t_out = tokenize_chat_batch(tokenizer, t_batch, max_tokens)
        t_keep = assistant_keep_mask_last_span(
            t_out["input_ids"], tok_name, t_out["attention_mask"]
        )
        return {
            "idx": torch.tensor(idxs, dtype=torch.long),
            "treatment_tokens": t_out["input_ids"],
            "treatment_attention_mask": t_out["attention_mask"],
            "treatment_asst_token_mask": t_keep,
        }

    return _collate


# ---------- Args ----------
@dataclass
class TRAKArgs:
    data: str
    output_dir: str
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_id: Optional[str] = None
    projection_dim: int = 4096
    projection_seed: int = 42
    batch_size: int = 1  # Per-sample gradients are memory-intensive
    max_tokens: int = 1024
    max_num_data: int = -1
    dtype: str = "bfloat16"  # Model dtype
    grad_dtype: str = "float32"  # Gradient computation dtype
    device: Optional[str] = None
    is_train_data: bool = False
    scores_format: str = "npy"
    scores_dtype: str = "float16"
    sparse_threshold: float = 0.0
    topk: int = 0
    save_every: int = 500
    jl_chunk_size: int = 200_000  # Reduced from 1M to fit in 24GB GPU
    # Layer pattern for which linear layers to include
    layer_pattern: str = r"layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
    # Enable gradient checkpointing to save memory
    gradient_checkpointing: bool = True


# ---------- Runner ----------
def run(args: TRAKArgs):
    # Validate batch_size=1: compute_projected_gradient sums gradients across the batch,
    # so batch_size>1 would give all samples the same (incorrect) projected gradient.
    assert args.batch_size == 1, (
        f"batch_size must be 1 for correct per-sample gradients, got {args.batch_size}. "
        "The current implementation sums gradients across the batch, so batch_size>1 "
        "would incorrectly assign the same projected gradient to all samples."
    )

    # Initialize distributed
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(
        os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count())))
    )
    print(f"[TRAK] rank: {rank}, world: {world}, local: {local}")

    # Device/dtype
    device = (
        torch.device(args.device)
        if args.device
        else torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    # Tokenizer (left pad/left truncate)
    tok_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Load model
    print(f"[TRAK] Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[TRAK] Gradient checkpointing enabled")

    # Model in eval mode (but we still need gradients for weight gradient computation)
    model.eval()
    # Note: We enable gradients explicitly in compute_projected_gradient

    hidden_size = model.config.hidden_size

    # Initialize TRAK config and gradient computer
    trak_config = TRAKConfig(
        projection_dim=args.projection_dim,
        seed=args.projection_seed,
        dtype=args.grad_dtype,
        jl_chunk_size=args.jl_chunk_size,
        layer_pattern=args.layer_pattern,
    )

    gradient_computer = WeightGradientComputer(model, trak_config, device)

    # Print memory estimates
    if rank == 0:
        print(f"[TRAK DEBUG] args.jl_chunk_size = {args.jl_chunk_size}")
        print(f"[TRAK DEBUG] trak_config.jl_chunk_size = {trak_config.jl_chunk_size}")
        mem_info = estimate_memory_usage(model, trak_config)
        print(f"[TRAK] Total linear params: {mem_info['total_linear_params']:,}")
        print(f"[TRAK] Max layer params: {mem_info['max_layer_params']:,}")
        print(f"[TRAK] Full gradient memory: {mem_info['full_gradient_memory_gb']:.2f} GB")
        print(f"[TRAK] Streaming gradient memory: {mem_info['streaming_gradient_memory_gb']:.4f} GB")
        print(f"[TRAK] Projection chunk memory: {mem_info['projection_chunk_memory_gb']:.4f} GB")
        print(f"[TRAK] Found {len(gradient_computer.linear_layers)} linear layers")

    # Data loader (sharded)
    ds = TRAKDataset(args.data, args.max_num_data)
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    collate = collate_fn_builder(tokenizer, args.max_tokens)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    # Output directory
    out_dir = Path(args.output_dir)
    make_dir_wide_permissions(out_dir, mode=0o777)

    # Accumulators
    local_idx: List[torch.Tensor] = []
    local_scores: List[torch.Tensor] = []
    step = 0
    shard = 0

    t0 = time.time()
    for batch in tqdm.tqdm(dl, disable=rank != 0, desc="TRAK gradients"):
        t_ids = batch["treatment_tokens"].to(device)
        t_att = batch["treatment_attention_mask"].to(device)
        t_keep = batch["treatment_asst_token_mask"].to(device)

        # Skip empty batches (no assistant tokens)
        if not t_keep.any():
            continue

        # Compute per-sample projected gradients
        # For batch_size=1, this is efficient; for larger batches, we process one at a time
        if args.batch_size == 1:
            scores_bt = gradient_computer.compute_projected_gradient(
                t_ids, t_att, t_keep, masked_lm_loss
            )
        else:
            scores_bt = gradient_computer.compute_per_sample_projected_gradient(
                t_ids, t_att, t_keep, masked_lm_loss
            )

        local_scores.append(scores_bt.detach().cpu())
        local_idx.append(batch["idx"])
        step += 1

        # Clear GPU cache periodically to prevent fragmentation
        if step % 10 == 0:
            torch.cuda.empty_cache()

        # Periodic saving
        if args.save_every and step % args.save_every == 0:
            idx_np = torch.cat(local_idx, dim=0).long().numpy()
            order = np.argsort(idx_np)
            scores_np = torch.cat(local_scores, dim=0).float().numpy()[order]

            rows = []
            for i in idx_np.tolist():
                item = ds[i]
                row_data = {
                    "idx": int(i),
                    "treatment_messages": json.dumps(
                        item["treatment_messages"], ensure_ascii=False
                    ),
                    "control_messages": json.dumps(
                        item["control_messages"], ensure_ascii=False
                    )
                }
                rows.append(row_data)

            suffix = f".rank{rank}.part{shard}"
            save_scores(
                out_dir,
                scores_np,
                fmt=args.scores_format,
                dtype=args.scores_dtype,
                sparse_threshold=args.sparse_threshold,
                topk=args.topk,
                suffix=suffix,
            )
            pl.DataFrame(rows).write_parquet(out_dir / f"data{suffix}.parquet")

            shard += 1
            local_idx.clear()
            local_scores.clear()

            if rank == 0:
                print(f"[TRAK] Saved shard {shard-1}, step {step}")

    # Final flush
    if local_scores:
        idx_np = torch.cat(local_idx, dim=0).long().numpy()
        order = np.argsort(idx_np)
        scores_np = torch.cat(local_scores, dim=0).float().numpy()[order]

        rows = []
        for i in idx_np.tolist():
            item = ds[i]
            rows.append({
                "idx": int(i),
                "treatment_messages": json.dumps(item["treatment_messages"], ensure_ascii=False),
                "control_messages": json.dumps(item["control_messages"], ensure_ascii=False),
            })
        suffix = f".rank{rank}" if not args.save_every else f".rank{rank}.part{shard}"
        save_info = save_scores(
            out_dir,
            scores_np,
            fmt=args.scores_format,
            dtype=args.scores_dtype,
            sparse_threshold=args.sparse_threshold,
            topk=args.topk,
            suffix=suffix,
        )
        pl.DataFrame(rows).write_parquet(out_dir / f"data{suffix}.parquet")
    else:
        save_info = {}

    # Save metadata (rank 0 only)
    if rank == 0:
        elapsed = time.time() - t0
        meta = {
            **vars(args),
            "method": "trak_weight_gradient",
            "tokenizer_id": args.tokenizer_id or args.model_id,
            "hidden_size": hidden_size,
            "projection_dim": args.projection_dim,
            "device": str(device),
            "dataset_size": len(ds),
            "elapsed_sec": round(elapsed, 2),
            "world_size": world,
            "save_info": save_info,
            "total_linear_params": gradient_computer.total_params,
            "num_linear_layers": len(gradient_computer.linear_layers),
            "layer_names": [name for name, _ in gradient_computer.linear_layers],
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[TRAK][OK] Rank {rank} wrote to {out_dir}")
        print(f"[TRAK] Processed {len(ds)} samples in {elapsed:.1f}s")

    dist.destroy_process_group()


def parse_args() -> TRAKArgs:
    p = argparse.ArgumentParser("TRAK-inspired weight gradient attribution")
    p.add_argument(
        "--data",
        required=True,
        help="JSONL/Parquet with {treatment_messages} or {messages}",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--tokenizer-id", default=None)
    p.add_argument("--projection-dim", type=int, default=4096)
    p.add_argument("--projection-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-num-data", type=int, default=-1)
    p.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    p.add_argument(
        "--grad-dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--is-train-data",
        action="store_true",
        help="Whether the code is run on training or eval data",
    )
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
        default=500,
        help="Save scores every N batches (0 = only once at end)",
    )
    p.add_argument("--jl-chunk-size", type=int, default=200_000)
    p.add_argument(
        "--layer-pattern",
        default=r"layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj",
        help="Regex pattern for linear layers to include",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    a = p.parse_args()
    return TRAKArgs(
        data=a.data,
        output_dir=a.output_dir,
        model_id=a.model_id,
        tokenizer_id=a.tokenizer_id,
        projection_dim=a.projection_dim,
        projection_seed=a.projection_seed,
        batch_size=a.batch_size,
        max_tokens=a.max_tokens,
        max_num_data=a.max_num_data,
        dtype=a.dtype,
        grad_dtype=a.grad_dtype,
        device=a.device,
        is_train_data=a.is_train_data,
        scores_format=a.scores_format,
        scores_dtype=a.scores_dtype,
        sparse_threshold=a.sparse_threshold,
        topk=a.topk,
        save_every=a.save_every,
        jl_chunk_size=a.jl_chunk_size,
        layer_pattern=a.layer_pattern,
        gradient_checkpointing=not a.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    run(parse_args())
