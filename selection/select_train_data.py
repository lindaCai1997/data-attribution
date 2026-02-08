# selection/select_train_data.py
"""
End-to-end pipeline: select training data -> fine-tune -> evaluate.

Supports the following attribution methods (--attribution-method):
- random: Random selection baseline
- residual_diff: Difference in residual activations between treatment/control
- residual_change_treatment: Estimated change in residual activation after fine-tuning
- residual_change: Change in residual activations (treatment - control)
- trak: Weight gradient attribution via Johnson-Lindenstrauss projection
- +mlp, +linear: (Experimental) Probe-based selection methods

Supports the following selection methods (--selection-method):
- residual_diff, residual_change, trak: Use eval attribution vectors for selection
- persona_vector_gen: Use pre-computed persona vectors from file

Note: When selection-method is trak, attribution-method must also be trak.
"""
import argparse
import os
import json
import time
import fcntl
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict, replace

from selection.utils import (
    fetch_data_from_indices,
    make_dir_wide_permissions,
    get_file_iterator,
    set_seed,
    maybe_initialize_dist,
    validate_and_set_args,
)
from selection.matrix import ShardedScoreMatrix
from selection.probe import MLPProbeScorer, LinearProbeScorer, ProbeTrainConfig, ProbeInferConfig
from selection.finetune import LoraFTConfig, run_lora_finetune_on_subset
from selection.eval import EvalConfig




def save_sweep_config_from_wandb(sweep_dir: Path, sweep_id: str) -> bool:
    """
    Fetch and save sweep configuration from wandb API.
    Only saves if sweep_config.json doesn't already exist.
    """
    config_file = sweep_dir / "sweep_config.json"

    if config_file.exists():
        return True

    try:
        import wandb
        api = wandb.Api()

        project = os.environ.get("WANDB_PROJECT", "test")
        entity = os.environ.get("WANDB_ENTITY")

        if entity:
            sweep_path = f"{entity}/{project}/{sweep_id}"
        else:
            sweep_path = f"{project}/{sweep_id}"

        sweep = api.sweep(sweep_path)

        sweep_config = {
            "sweep_id": sweep_id,
            "sweep_path": sweep_path,
            "name": getattr(sweep, "name", None),
            "state": getattr(sweep, "state", None),
            "config": sweep.config,
        }

        with open(config_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, 2)
                if f.tell() == 0:
                    json.dump(sweep_config, f, indent=2)
                    print(f"[OK] Saved sweep config to {config_file}")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return True

    except Exception as e:
        print(f"[WARN] Could not fetch sweep config from wandb: {e}")
        return False


def register_run_with_sweep(
    root_dir: str,
    sweep_id: str,
    run_dir: str,
    run_config: dict,
) -> str:
    """
    Register a run with a sweep by appending to sweep's runs.jsonl file.
    """
    sweep_dir = Path(root_dir) / "sweeps" / sweep_id
    make_dir_wide_permissions(str(sweep_dir))

    save_sweep_config_from_wandb(sweep_dir, sweep_id)

    runs_file = sweep_dir / "runs.jsonl"
    wandb_run_id = os.environ.get("WANDB_RUN_ID")

    run_entry = {
        "run_dir": str(run_dir),
        "wandb_run_id": wandb_run_id,
        **run_config,
    }

    with open(runs_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(run_entry) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(f"[OK] Registered run with sweep {sweep_id} at {runs_file}")
    return str(sweep_dir)


def select_training_data(args, rank: int) -> tuple:
    """
    Select training data based on attribution method.
    
    Returns:
        train_df dataframe
    """
    rng = np.random.default_rng(args.seed)
    
    if args.attribution_method == "random":
        # Random selection baseline
        train_df = fetch_data_from_indices(
            dir_path=args.train_dir.replace("random", "residual_diff"),
            indices=args.k2,
            seed=args.seed,
        )
    
    elif args.selection_method == "persona_vector_gen":
        # Persona Vector (PV) based selection - load pre-computed persona vector from file
        persona_vector_file = Path(args.persona_vector_path) / f"{args.eval_data_name}.pt"
        persona_vector = torch.load(persona_vector_file)
        persona_vector = persona_vector[args.layer]
        print(f"[PV] Loaded persona vector from {persona_vector_file}, layer {args.layer}")
        
        # Load train scores matrix
        train_scores_matrix = ShardedScoreMatrix(args.train_dir).materialize()
        
        # Apply projection method for PV
        if args.projection_method == "cos_sim":
            train_normed = train_scores_matrix.float() / train_scores_matrix.float().norm(dim=1, keepdim=True).clamp(min=1e-6)
            pv_normed = (persona_vector.float() / torch.norm(persona_vector.float()).clamp(min=1e-6)).to(train_normed.device)
            scores = train_normed @ pv_normed
        elif args.projection_method == "proj":
            pv_normed = (persona_vector.float() / torch.norm(persona_vector.float()).clamp(min=1e-6)).to(train_scores_matrix.device)
            scores = train_scores_matrix.matmul(pv_normed)
        elif args.projection_method == "dot_product":
            scores = train_scores_matrix.matmul(persona_vector.float().to(train_scores_matrix.device))
        else:
            raise ValueError(f"Unknown projection method for PV: {args.projection_method}")
        
        data_idx = torch.topk(input=scores, k=args.k2, largest=True).indices.int().cpu().numpy()
        train_df = fetch_data_from_indices(dir_path=args.train_dir, indices=data_idx, seed=args.seed)
        
    ## Experimental: Probe-based selection
    elif "+mlp" in args.attribution_method or "+linear" in args.attribution_method:
        if "+mlp" in args.attribution_method:
            scorer = MLPProbeScorer(
                train_dir=args.train_dir,
                layer_idx=args.layer_idx,
                device="cuda",
                input_transform="l2",
                hidden=(4096, 2048),
                dropout=0.1,
            )
            cfg = ProbeTrainConfig(
                epochs=20, batch_size=1024, lr=5e-4,
                val_split=0.1, patience=4, use_amp=True,
            )
        else:  # +linear
            scorer = LinearProbeScorer(
                train_dir=args.train_dir, 
                layer_idx=args.layer_idx, 
                device="cuda", 
                input_transform="l2"
            )
            cfg = ProbeTrainConfig(
                epochs=10, batch_size=2048, lr=1e-3,
                val_split=0.1, patience=3, use_amp=True,
            )
        scorer.fit_from_dirs(
            pos_eval_dir=args.eval_dir_treatment,
            neg_eval_dir=args.eval_dir_control,
            cfg=cfg,
        )
        data_idx = scorer.topk_train(
            k=args.k2,
            infer_cfg=ProbeInferConfig(shard_row_batch_size=8192, return_logits=True),
        )
        train_df = fetch_data_from_indices(
            dir_path=args.train_dir,
            indices=data_idx,
            seed=args.seed,
        )

    else:
        # Vector-based methods: residual_diff, residual_change, residual_change_treatment, trak
        eval_scores_matrix = ShardedScoreMatrix(args.eval_dir, layer_idx=args.layer_idx).materialize()
        train_scores_matrix = ShardedScoreMatrix(args.train_dir, layer_idx=args.layer_idx).materialize()
        
        if args.projection_method == "cos_sim":
            eval_normed = eval_scores_matrix / eval_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            eval_normed_mean = eval_normed.mean(dim=0) if isinstance(eval_normed, torch.Tensor) else np.mean(eval_normed, axis=0)
            if not isinstance(eval_normed_mean, torch.Tensor):
                eval_normed_mean = torch.from_numpy(eval_normed_mean)
            eval_normed_mean_unit = eval_normed_mean / eval_normed_mean.norm().clamp(min=1e-8)
            train_normed = train_scores_matrix / train_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scores = train_normed @ eval_normed_mean_unit
            
        elif args.projection_method == "cos_sim_debias_eval":
            eval_control_matrix = ShardedScoreMatrix(args.eval_dir_control, layer_idx=args.layer_idx).materialize()
            eval_control_normed_avg = eval_control_matrix / eval_control_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            eval_control_normed_avg_mean = eval_control_normed_avg.mean(dim=0)
            if not isinstance(eval_control_normed_avg_mean, torch.Tensor):
                eval_control_normed_avg_mean = torch.from_numpy(eval_control_normed_avg_mean)
            eval_control_normed_avg_unit = eval_control_normed_avg_mean / eval_control_normed_avg_mean.norm().clamp(min=1e-8)
            
            eval_dot = eval_scores_matrix @ eval_control_normed_avg_unit
            eval_proj = torch.outer(eval_dot, eval_control_normed_avg_unit)
            eval_orth = eval_scores_matrix - eval_proj
            ecm_orth_normed_mean = (eval_orth / eval_orth.norm(dim=1, keepdim=True).clamp(min=1e-6)).mean(dim=0)
            ecm_orth_normed_mean_unit = ecm_orth_normed_mean / ecm_orth_normed_mean.norm().clamp(min=1e-8)

            train_normed = train_scores_matrix / train_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scores = train_normed @ ecm_orth_normed_mean_unit
            
        elif args.projection_method == "cos_sim_debias_train":
            eval_normed = eval_scores_matrix / eval_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            eval_normed_mean = eval_normed.mean(dim=0)
            if not isinstance(eval_normed_mean, torch.Tensor):
                eval_normed_mean = torch.from_numpy(eval_normed_mean)
            eval_normed_mean_unit = eval_normed_mean / eval_normed_mean.norm().clamp(min=1e-8)
            
            train_normed = train_scores_matrix / train_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            train_normed_mean = train_normed.mean(dim=0)
            tnm_norm = train_normed_mean.norm().clamp(min=1e-8)
            tnm_unit = train_normed_mean / tnm_norm
            dot_products = train_normed @ tnm_unit
            proj = torch.outer(dot_products, tnm_unit) if len(train_normed.shape) == 2 else dot_products.unsqueeze(-1) * tnm_unit
            train_normed_orth_normed = (train_normed - proj) / (train_normed - proj).norm(dim=1, keepdim=True).clamp(min=1e-6)
            scores = train_normed_orth_normed @ eval_normed_mean_unit
            
        elif args.projection_method == "proj":
            eval_normed = eval_scores_matrix / eval_scores_matrix.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scores = (train_scores_matrix @ eval_normed.T).mean(axis=1)
            
        elif args.projection_method == "dot_product":
            scores = (train_scores_matrix @ eval_scores_matrix.T).mean(axis=1)
        else:
            raise ValueError(f"Unknown projection method: {args.projection_method}")

        data_idx = torch.topk(scores, args.k2).indices.int().cpu().numpy()
        train_df = fetch_data_from_indices(
            dir_path=args.train_dir,
            indices=data_idx,
            seed=args.seed,
        )

    return train_df


def main():
    p = argparse.ArgumentParser(
        "Select training data based on attribution, then fine-tune and evaluate."
    )
    # Data directories
    p.add_argument("--root-dir", required=True, help="Root directory for all experiments")
    p.add_argument("--train-dir", default=None, help="Directory with precomputed train attribution vectors")
    p.add_argument("--eval-dir", default=None, help="Directory with precomputed eval attribution vectors")
    p.add_argument("--eval-dir-treatment", type=str, default=None, help="(For probe methods) Treatment eval vectors")
    p.add_argument("--eval-dir-control", type=str, default=None, help="(For probe methods) Control eval vectors")
    p.add_argument("--layer-idx", type=int, default=None, help="Layer index for 3D score shards")
    
    # Data identifiers
    p.add_argument("--train-data-name", required=True, help="Name of training dataset")
    p.add_argument("--eval-data-name", required=True, help="Name of evaluation dataset")
    
    # Attribution method
    p.add_argument(
        "--attribution-method", 
        required=True, 
        help="Attribution method for train vectors: random, residual_diff, residual_change, residual_change_treatment, trak, +mlp, +linear"
    )
    p.add_argument(
        "--selection-method",
        default=None,
        help="Attribution method for eval vectors used in selection. Defaults to residual_diff (or trak if attribution-method is trak)."
    )
    p.add_argument(
        "--projection-method", 
        required=True, 
        default="cos_sim", 
        choices=["cos_sim", "proj", "dot_product", "cos_sim_debias_eval", "cos_sim_debias_train"],
        help="Method to compute similarity between train and eval vectors"
    )
    
    # Persona vector arguments (used when selection-method is persona_vector_gen)
    p.add_argument(
        "--persona-vector-path",
        default=None,
        help="Path to persona vector file directory (required when selection-method is persona_vector_gen)"
    )
    p.add_argument(
        "--layer",
        type=int,
        default=11,
        help="Layer index for persona vector (used when selection-method is persona_vector_gen)"
    )
    
    # Selection parameters
    p.add_argument("--k2", type=int, required=True, help="Number of training samples to select")
    
    # Model
    p.add_argument("--model-id", required=True, help="HuggingFace model ID for fine-tuning")
    
    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    p.add_argument(
        "--eval-method",
        type=lambda s: [item.strip() for item in s.replace(",", " ").split()],
        default=["llm_judge"],
        help="Evaluation methods: cross_entropy, llm_judge (comma/space separated)"
    )
    p.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--judge-repeats", type=int, default=3, help="Repeats per question for LLM judge")
    p.add_argument("--cross-entropy-eval-path", default=None, help="Path to cross-entropy eval data")
    p.add_argument("--llm-judge-eval-path", default=None, help="Path to LLM judge eval data")
    p.add_argument(
        "--eval-data-base-dir",
        default=None,
        help="Base directory for eval data (auto-constructs paths from eval-data-name)"
    )
    
    # Output
    p.add_argument("--work-name", default=None, help="Custom suffix for output directory")
    p.add_argument("--save-model", action="store_true", help="Save the fine-tuned model")
    
    # Experiment tracking
    p.add_argument("--sweep-id", default=None, help="Sweep ID to register this run with")
    p.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
    
    args = p.parse_args()
    validate_and_set_args(args)
    
    # Validate persona vector arguments when selection_method is persona_vector_gen
    if args.selection_method == "persona_vector_gen":
        assert args.persona_vector_path is not None, (
            "persona_vector_path is required when selection-method is persona_vector_gen"
        )
    
    # Validate trak: both attribution_method and selection_method must be trak
    if args.selection_method == "trak":
        assert "trak" in args.attribution_method.lower(), (
            "When selection-method is trak, attribution-method must also be trak"
        )
    
    set_seed(args.seed)
    rank, device = maybe_initialize_dist()
    
    if rank == 0:
        print("===== Running training data selection + finetuning =====")
        print(f"Train attribution method: {args.attribution_method}")
        print(f"Selection method: {args.selection_method}")
        print(f"Projection method: {args.projection_method}")
        if args.selection_method == "persona_vector_gen":
            print(f"Persona vector path: {args.persona_vector_path}")
            print(f"Persona vector layer: {args.layer}")
        print(f"Selecting k2={args.k2} training samples")
    
    # Synchronize timestamp across ranks
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dist.is_initialized():
        objects = [time_str] if rank == 0 else [None]
        dist.broadcast_object_list(objects, src=0)
        DATETIME_STR = objects[0]
    else:
        DATETIME_STR = time_str
    
    t0 = time.time()
    
    # Create output directory
    suffix = f"{args.work_name if args.work_name else DATETIME_STR}"
    subdir_name = f"{args.train_data_name}-{args.projection_method}-{args.attribution_method}-{args.selection_method}-{args.k2}-{args.eval_data_name}-{suffix}"
    subdir = Path(args.train_dir) / subdir_name
    train_data_path = str(subdir / "selected_train_data.jsonl")
    
    # Select training data (rank 0 only)
    if rank == 0:
        train_df = select_training_data(args, rank)
        
        subdir.mkdir(parents=True, exist_ok=True)
        with open(train_data_path, "w") as f:
            for row in train_df.iter_rows(named=True):
                f.write(
                    json.dumps({
                        "treatment_messages": json.loads(row["treatment_messages"]),
                        "control_messages": json.loads(row["control_messages"]),
                    }, ensure_ascii=False) + "\n"
                )
        print(f"[OK] Saved selected train data to {train_data_path}")
        print(f"Time taken to select training data: {int(time.time() - t0)} seconds")
    
    if dist.is_initialized():
        dist.barrier(device_ids=[rank])
    
    train_data = list(get_file_iterator(train_data_path))

    # Register run with sweep
    sweep_id = args.sweep_id or os.environ.get("WANDB_SWEEP_ID")
    if sweep_id and rank == 0:
        run_config = {
            "wandb_name": subdir_name,
            "train_data_name": args.train_data_name,
            "eval_data_name": args.eval_data_name,
            "attribution_method": args.attribution_method,
            "selection_method": args.selection_method,
            "projection_method": args.projection_method,
            "k2": args.k2,
            "epochs": args.epochs,
            "seed": args.seed,
        }
        if args.selection_method == "persona_vector_gen":
            run_config["layer"] = args.layer
        register_run_with_sweep(
            root_dir=args.root_dir,
            sweep_id=sweep_id,
            run_dir=str(subdir),
            run_config=run_config,
        )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        if rank == 0:
            print("[OK] Using flash_attention_2")
    except (ImportError, ValueError) as e:
        if rank == 0:
            print(f"[WARN] flash_attention_2 not available ({e}), using SDPA")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(device)

    # Configure fine-tuning
    wandb_config = {
        "train_set": args.train_data_name,
        "eval_set": args.eval_data_name,
        "attribution_method": args.attribution_method,
        "selection_method": args.selection_method,
        "projection_method": args.projection_method,
        "k2": args.k2,
        "seed": args.seed,
    }
    if args.selection_method == "persona_vector_gen":
        wandb_config["layer"] = args.layer
    
    ft_cfg = LoraFTConfig(
        base_model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        work_dir=str(subdir),
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
        save_model=args.save_model,
        wandb_name=subdir_name,
        wandb_config=wandb_config,
        eval_on_first_step=(args.attribution_method == "random"),
        disable_wandb=args.disable_wandb
    )

    # Load evaluation data
    cross_entropy_eval_data = list(get_file_iterator(args.cross_entropy_eval_path))
    
    if isinstance(args.llm_judge_eval_path, dict):
        llm_judge_eval_data = {
            subset_name: list(get_file_iterator(path))
            for subset_name, path in args.llm_judge_eval_path.items()
        }
        if rank == 0:
            print(f"[OK] Loaded multi-subset LLM judge eval data: {list(llm_judge_eval_data.keys())}")
    else:
        llm_judge_eval_data = list(get_file_iterator(args.llm_judge_eval_path))
    
    eval_cfg = EvalConfig(
        tokenizer=tokenizer,
        eval_data_name=args.eval_data_name,
        cross_entropy_eval_data=cross_entropy_eval_data,
        llm_judge_eval_data=llm_judge_eval_data,
        work_dir=str(subdir / "selected_data"),
        seed=args.seed,
        eval_method=args.eval_method,
        repeats=args.judge_repeats,
    )

    # Run fine-tuning with evaluation
    run_lora_finetune_on_subset(ft_cfg, eval_cfg)

    # Save config
    cfg_for_save = asdict(
        replace(ft_cfg, base_model=None, tokenizer=None, train_data=None, wandb_config=None)
    )
    full_config = {**vars(args), **cfg_for_save}
    with open(subdir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)


if __name__ == "__main__":
    main()
