# analysis/cross_model_transfer/run_transfer.py
"""
Cross-model transfer pilot (paper Appendix J).

Takes an already-selected top-k training subset (selected_train_data.jsonl,
produced by model A's attribution pipeline) and fine-tunes + evaluates a
DIFFERENT model B on it, mirroring exactly the post-selection tail of
selection/select_train_data.py (LoRA r=32, alpha=64, epochs=1, batch=2,
lr=1e-4, max_seq_len=1024, llm_judge eval with 3 repeats).

No selection logic here — the data was already chosen by model A.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from selection.utils import (
    get_file_iterator,
    make_dir_wide_permissions,
    maybe_initialize_dist,
    set_seed,
)
from selection.finetune import LoraFTConfig, run_lora_finetune_on_subset
from selection.eval import EvalConfig

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")


def main():
    p = argparse.ArgumentParser("Cross-model transfer: fine-tune model B on model A's selected data")
    p.add_argument("--selected-jsonl", required=True,
                   help="Path to selected_train_data.jsonl produced by model A's selection run")
    p.add_argument("--model-id", required=True, help="HF model id of model B (the model to fine-tune)")
    p.add_argument("--eval-data-name", required=True)
    p.add_argument("--out-dir", required=True, help="Output directory for this transfer run")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-repeats", type=int, default=3)
    p.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--llm-judge-eval-path", default=None)
    p.add_argument("--expected-k", type=int, default=500)
    args = p.parse_args()

    if args.llm_judge_eval_path is None:
        args.llm_judge_eval_path = (
            f"{DATA_ROOT}/eval-dataset/{args.eval_data_name}.json"
        )

    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "selected_data" / "eval_llm_judge" / "metrics.jsonl"

    # Requeue-safety: metrics.jsonl is only written by the FINAL eval
    # (eval_on_first_step=False, epochs=1), so a non-empty file means this
    # cell already completed. Skip instead of appending duplicate lines.
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        print(f"[SKIP] {metrics_path} already exists and is non-empty; run complete.")
        return

    set_seed(args.seed)
    rank, device = maybe_initialize_dist()

    t0 = time.time()
    make_dir_wide_permissions(str(out_dir))

    # Load the transferred subset (selected by model A)
    train_data = list(get_file_iterator(args.selected_jsonl))
    assert len(train_data) == args.expected_k, (
        f"Expected {args.expected_k} rows in {args.selected_jsonl}, got {len(train_data)}"
    )
    if rank == 0:
        print(f"[OK] Loaded {len(train_data)} transferred examples from {args.selected_jsonl}")
        print(f"[OK] Fine-tuning model B: {args.model_id}")

    # ---- Model / tokenizer (identical to select_train_data.py) ----
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device-gated attention impl (same as select_train_data.py):
    # A100 -> sdpa (matches finished runs); Hopper -> eager (SDPA bf16 NaNs).
    if torch.cuda.is_available():
        _dev = torch.cuda.get_device_name(0)
        _is_hopper = ("H100" in _dev) or ("H200" in _dev)
    else:
        _dev, _is_hopper = "cpu", False
    _attn = "eager" if _is_hopper else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=_attn,
    ).to(device)
    if rank == 0:
        print(f"[OK] Using attn_implementation={_attn} (device: {_dev})")

    # ---- FT config: identical hyperparameters to the paper protocol ----
    ft_cfg = LoraFTConfig(
        base_model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        work_dir=str(out_dir),
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        max_seq_len=1024,
        learning_rate=1e-4,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0,
        use_rslora=False,
        save_steps=0,
        eval_steps=0,
        save_model=False,
        wandb_name=out_dir.name,
        wandb_config=None,
        eval_on_first_step=False,
        disable_wandb=True,
    )

    llm_judge_eval_data = list(get_file_iterator(args.llm_judge_eval_path))
    if rank == 0:
        print(f"[OK] Loaded {len(llm_judge_eval_data)} judge eval items from {args.llm_judge_eval_path}")

    eval_cfg = EvalConfig(
        tokenizer=tokenizer,
        eval_data_name=args.eval_data_name,
        cross_entropy_eval_data=[],
        llm_judge_eval_data=llm_judge_eval_data,
        work_dir=str(out_dir / "selected_data"),
        seed=args.seed,
        eval_method=["llm_judge"],
        repeats=args.judge_repeats,
        judge_model=args.judge_model,
    )

    run_lora_finetune_on_subset(ft_cfg, eval_cfg)

    # Save config for provenance
    if rank == 0:
        cfg_for_save = asdict(
            replace(ft_cfg, base_model=None, tokenizer=None, train_data=None, wandb_config=None)
        )
        full_config = {**vars(args), **cfg_for_save, "n_train": len(train_data)}
        with open(out_dir / "config.json", "w") as f:
            json.dump(full_config, f, indent=2)
        print(f"[DONE] Transfer run finished in {int(time.time() - t0)}s -> {out_dir}")


if __name__ == "__main__":
    main()
