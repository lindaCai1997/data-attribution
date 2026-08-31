"""No-selection baseline: evaluate the base model on an eval dataset, with no
fine-tuning. Produces the same wandb metrics + on-disk layout as the regular
select_train_data pipeline so downstream analysis tooling treats it uniformly.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from selection.utils import (
    get_file_iterator,
    make_dir_wide_permissions,
    maybe_initialize_dist,
    set_seed,
)
from selection.eval import EvalConfig, eval_model
from selection.select_train_data import register_run_with_sweep


def _model_short(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "p")


def _resolve_eval_paths(args):
    if args.cross_entropy_eval_path is None:
        args.cross_entropy_eval_path = (
            f"{args.eval_data_base_dir}/dataset/{args.eval_data_name}.parquet"
        )
    if args.llm_judge_eval_path is None:
        if args.eval_data_name == "truthful_qa":
            args.llm_judge_eval_path = {
                "mc1": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa_mc1.json",
                "mc2": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa_mc2.json",
                "free_response": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa.json",
            }
        else:
            args.llm_judge_eval_path = (
                f"{args.eval_data_base_dir}/eval-dataset/{args.eval_data_name}.json"
            )


def main():
    p = argparse.ArgumentParser("Evaluate the base model with no finetuning (no-selection baseline).")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--eval-data-name", required=True)
    p.add_argument("--eval-data-base-dir", required=True)
    p.add_argument("--cross-entropy-eval-path", default=None)
    p.add_argument("--llm-judge-eval-path", default=None)
    p.add_argument(
        "--eval-method",
        type=lambda s: [item.strip() for item in s.replace(",", " ").split()],
        default=["llm_judge"],
    )
    p.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--judge-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gen-batch-size", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep-id", default=None)
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--work-name", default=None)

    args = p.parse_args()
    _resolve_eval_paths(args)

    set_seed(args.seed)
    rank, device = maybe_initialize_dist()

    if rank == 0:
        print("===== No-selection baseline eval =====")
        print(f"model_id: {args.model_id}")
        print(f"eval_data_name: {args.eval_data_name}")
        print(f"llm_judge_eval_path: {args.llm_judge_eval_path}")

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = _model_short(args.model_id)
    suffix = args.work_name if args.work_name else time_str
    subdir_name = (
        f"no_selection-{model_short}-{args.eval_data_name}-{suffix}"
    )
    subdir = Path(args.root_dir) / "no_selection" / model_short / args.eval_data_name / suffix
    if rank == 0:
        subdir.mkdir(parents=True, exist_ok=True)
        make_dir_wide_permissions(str(subdir))

    sweep_id = args.sweep_id or os.environ.get("WANDB_SWEEP_ID")
    if sweep_id and rank == 0:
        register_run_with_sweep(
            root_dir=args.root_dir,
            sweep_id=sweep_id,
            run_dir=str(subdir),
            run_config={
                "wandb_name": subdir_name,
                "eval_data_name": args.eval_data_name,
                "model_id": args.model_id,
                "method": "no_selection",
                "seed": args.seed,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        _dev = torch.cuda.get_device_name(0)
        _is_hopper = ("H100" in _dev) or ("H200" in _dev)
    else:
        _is_hopper = False
    _attn = "eager" if _is_hopper else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=_attn,
    ).to(device)
    model.eval()
    if rank == 0:
        print(f"[OK] Loaded base model with attn_implementation={_attn}")

    cross_entropy_eval_data = list(get_file_iterator(args.cross_entropy_eval_path))
    if isinstance(args.llm_judge_eval_path, dict):
        llm_judge_eval_data = {
            subset: list(get_file_iterator(path))
            for subset, path in args.llm_judge_eval_path.items()
        }
    else:
        llm_judge_eval_data = list(get_file_iterator(args.llm_judge_eval_path))

    if not args.disable_wandb and rank == 0:
        import wandb
        wandb.init(
            name=subdir_name,
            config={
                "model_id": args.model_id,
                "eval_data_name": args.eval_data_name,
                "method": "no_selection",
                "seed": args.seed,
                "judge_model": args.judge_model,
                "judge_repeats": args.judge_repeats,
            },
        )

    eval_cfg = EvalConfig(
        model=model,
        tokenizer=tokenizer,
        eval_data_name=args.eval_data_name,
        cross_entropy_eval_data=cross_entropy_eval_data,
        llm_judge_eval_data=llm_judge_eval_data,
        work_dir=str(subdir),
        seed=args.seed,
        epoch=0,
        global_step=0,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        judge_model=args.judge_model,
        repeats=args.judge_repeats,
        eval_method=args.eval_method,
    )

    eval_model(eval_cfg)

    if rank == 0:
        with open(subdir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
