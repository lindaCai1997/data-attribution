from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple, Union
import asyncio
from tqdm.auto import tqdm
import os
import csv
import json

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Custom Imports ---
from selection.utils import (
    make_dir_wide_permissions,
    mean_ce_for_conversations,
)
from selection.llm_judge.judge import OpenAiJudge
from selection.llm_judge.prompts import Prompts
import wandb


# =========================
# Distributed Helpers
# =========================
def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def split_list_for_distributed(data: List[Any]) -> List[Any]:
    """Splits a list into chunks for the current process."""
    rank = get_rank()
    world_size = get_world_size()
    # Simple strided slicing
    return data[rank::world_size]

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# =========================
# Local metrics saving helper
# =========================
def _save_metrics_locally(
    work_dir: str,
    eval_type: str,
    metrics: Dict[str, Any],
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
) -> str:
    # ONLY Run on Rank 0
    if not is_main_process():
        return ""

    eval_dir = os.path.join(work_dir, f"eval_{eval_type}")
    make_dir_wide_permissions(eval_dir)
    
    filepath = os.path.join(eval_dir, "metrics.jsonl")
    
    metrics_with_meta = {
        "epoch": epoch,
        "global_step": global_step,
        **metrics,
    }
    
    with open(filepath, "a") as f:
        f.write(json.dumps(metrics_with_meta) + "\n")
    
    print(f"[Rank 0] Appended {eval_type} metrics (epoch={epoch}) to {filepath}")
    return filepath

# =========================
# Unified dispatcher
# =========================
@dataclass
class EvalConfig:
    # --- core / shared ---
    eval_method: List[str]
    tokenizer: AutoTokenizer
    model: Optional[AutoModelForCausalLM] = None
    
    # Data arguments
    cross_entropy_eval_data: Optional[List[Dict[str, Any]]] = None
    # llm_judge_eval_data can be:
    #   - List: single eval set (backward compatible)
    #   - Dict[str, List]: multiple subsets with separate generation, e.g. {"mc1": [...], "mc2": [...], "free_response": [...]}
    llm_judge_eval_data: Optional[Union[List, Dict[str, List]]] = None

    work_dir: str = "results"
    device: str = "cuda"
    seed: int = 0
    global_step: int = 0
    epoch: Optional[int] = None

    # --- cross-entropy specific ---
    max_tokens: int = 1024
    batch_size: int = 4

    # --- generation specific ---
    gen_batch_size: int = 60
    gen_max_new_tokens: int = 1024
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    repeats: int = 3
    activation_out_dir: Optional[str] = None

    # LLM-as-a-judge specific
    eval_traits: Optional[List[str]] = None
    judge_model: str = "gpt-4.1-mini-2025-04-14"
    eval_data_name: Optional[str] = None
    judge_concurrency: int = 50
    include_rationale: bool = True
    judge_out_csv: Optional[str] = None

def _require_args(args: dict, context: str) -> None:
    missing = [name for name, value in args.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments for {context}: {', '.join(missing)}")

def eval_model(cfg: EvalConfig) -> None:
    """
    Unified evaluation entrypoint.
    Handles distributed barriers internally to ensure synchronization.
    """
    eval_methods = [method.lower() for method in cfg.eval_method]
    
    # Ensure all processes start together
    if "llm_judge" in eval_methods or "judge" in eval_methods:
        if is_main_process():
            print("Evaluating LLM-as-a-Judge...")
        eval_llm_judge(cfg)
        # if dist.is_initialized():
        #     dist.barrier()

    if "cross_entropy" in eval_methods or "ce" in eval_methods:
        if is_main_process():
            print("Evaluating cross-entropy...")
        eval_cross_entropy_lora_dir(cfg)

# =========================
# Mode 1: Cross-entropy eval
# =========================
def eval_cross_entropy_lora_dir(cfg: EvalConfig) -> None:
    _require_args(
        {"model": cfg.model, "tokenizer": cfg.tokenizer, "data": cfg.cross_entropy_eval_data},
        context="cross-entropy eval",
    )

    model = cfg.model
    tokenizer = cfg.tokenizer
    full_data = cfg.cross_entropy_eval_data

    # 1. Split Data across GPUs
    local_data = split_list_for_distributed(full_data)
    
    # 2. Extract conversations
    convs_treat = [r["treatment_messages"] for r in local_data]
    # control_messages may not exist for some methods (e.g., TRAK)
    convs_ctrl = [r.get("control_messages") for r in local_data if r.get("control_messages") is not None]

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # 3. Compute Local Means
    # Note: mean_ce_for_conversations usually returns a single float (mean of the batch).
    # To aggregate correctly across GPUs, we theoretically need (Sum, Count). 
    # Assuming mean_ce returns an unweighted mean of the conversations:
    local_mean_ce_treat = mean_ce_for_conversations(
        model, tokenizer, convs_treat, device=cfg.device,
        max_tokens=cfg.max_tokens, batch_size=cfg.batch_size,
    ) if convs_treat else 0.0
    
    local_mean_ce_ctrl = 0.0 
    # mean_ce_for_conversations(
    #     model, tokenizer, convs_ctrl, device=cfg.device,
    #     max_tokens=cfg.max_tokens, batch_size=cfg.batch_size,
    # ) if convs_ctrl else 0.0

    count = torch.tensor([len(convs_treat)], device=cfg.device, dtype=torch.float32)
    sum_treat = torch.tensor([local_mean_ce_treat * len(convs_treat)], device=cfg.device)
    sum_ctrl = torch.tensor([local_mean_ce_ctrl * len(convs_ctrl)], device=cfg.device)

    # 4. Distributed Aggregation
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_treat, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ctrl, op=dist.ReduceOp.SUM)

    # 5. Final Calculation on Rank 0
    if is_main_process():
        total_count = count.item()
        final_mean_treat = (sum_treat.item() / total_count) if total_count > 0 else 0.0
        final_mean_ctrl = (sum_ctrl.item() / total_count) if total_count > 0 else 0.0
        
        ce_metrics = {
            "mean_ce_treatment": final_mean_treat,
            "mean_ce_control": final_mean_ctrl,
            "gap_treat_minus_control": final_mean_treat - final_mean_ctrl,
        }

        _save_metrics_locally(
            work_dir=cfg.work_dir,
            eval_type="cross_entropy",
            metrics=ce_metrics,
            epoch=cfg.epoch,
            global_step=cfg.global_step,
        )

        if getattr(wandb, "run", None) is not None:
            wandb.log({
                "global_step": cfg.global_step,
                "epoch": cfg.epoch,
                "eval_cross_entropy/mean_ce_treatment": final_mean_treat,
                "eval_cross_entropy/mean_ce_control": final_mean_ctrl,
                "eval_cross_entropy/gap_treat_minus_control": final_mean_treat - final_mean_ctrl,
            })

# =========================
# LLM-as-a-Judge (Gather then Judge)
# =========================

@torch.inference_mode()
def _generate_text_only(model, tokenizer, prompts, max_tokens, temp, top_p, batch_size):
    """
    Generation loop that returns ONLY the completion (no prompt text).
    """
    outs = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]  # includes left padding

        gen = model.generate(
            **inputs,
            do_sample=(temp > 0),
            temperature=(temp if temp > 0 else None),
            top_p=(top_p if temp > 0 else None),
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        new_tokens = gen[:, input_len:]  # ONLY newly generated tokens
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outs

DEFAULT_TRAITS = {
    "ultra_factual_truthfulness": ["ultra_truthfulness_negative_0_3"],
    "ultra_coding_instruction_following": ["ultra_instruction_following_negative_0_3"],
    "ultra_creative_instruction_following": ["ultra_instruction_following_negative_0_3"],
    "refact": ["refact_0_1"],
    "evil": ["professional_tone", "hostility", "manipulative"],
    "hallucination": ["correctness"],
    "sycophancy": ["sycophancy"], 
    "safety_boundary_management": ["safety_boundary_management"],
    "mistake_math_complete": ["math_reasoning_validity_0_1","math_final_answer_correctness_0_1"],
    # truthful_qa supports multiple subsets with different generation/evaluation:
    # - mc1: multiple choice format 1 (single correct answer)
    # - mc2: multiple choice format 2 (multiple correct answers)
    # - free_response: open-ended generation
    # When llm_judge_eval_data is a dict with these keys, each subset is generated and judged separately
    "truthful_qa": {
        "mc1": ["truthful_qa_mc1_0_1"],
        "mc2": ["truthful_qa_mc2_0_1"],
        "free_response": ["truthful_qa_0_1"],
    },
    # For backward compatibility, single-subset truthful_qa still works:
    "truthful_qa_free_response": ["truthful_qa_0_1"],
    "truthful_qa_mc1": ["truthful_qa_mc1_0_1"],
    "truthful_qa_mc2": ["truthful_qa_mc2_0_1"],
    "medhallu_easy_with_knowledge": ["medical_consistency_0_2"],
    "medhallu_medium_with_knowledge": ["medical_consistency_0_2"],
    "medhallu_hard_with_knowledge": ["medical_consistency_0_2"],
    "evil_gpt": ["evil"],
    "hallucination_gpt": ["hallucination"],
    "sycophancy_gpt": ["sycophancy"],
    "overconfidence_gpt": ["overconfidence"],
    "passive_aggression_gpt": ["passive_aggression"],
    "laziness_gpt": ["laziness"],
    "preachiness_gpt": ["preachiness"],
    "conspiracy_mindedness_gpt": ["conspiracy_mindedness"],
    "defensiveness_gpt": ["defensiveness"],
    "intellectual_arrogance_gpt": ["intellectual_arrogance"],
    "pedantry_gpt": ["pedantry"],
    "extreme_politeness_gpt": ["extreme_politeness"],
    "unwavering_optimism_gpt": ["unwavering_optimism"],
    "empathy_gpt": ["empathy"],
    "modesty_gpt": ["modesty"],
    "professionalism_gpt": ["professionalism"],
    "socratic_tutor_gpt": ["socratic_tutor"],
    "happiness_gpt": ["happiness"],
    "skepticism_gpt": ["skepticism"],
    "nostalgia_gpt": ["nostalgia"],
    "curiosity_gpt": ["curiosity"],
    "stoicism_gpt": ["stoicism"],
    "childlike_wonder_gpt": ["childlike_wonder"],
    "extreme_formality_gpt": ["extreme_formality"],
    "cryptic_mysticism_gpt": ["cryptic_mysticism"],
}


def _resolve_traits(cfg) -> List[str]:
    """Resolve traits for single-subset (backward compatible) mode."""
    if cfg.eval_traits: return [t.strip() for t in cfg.eval_traits]
    if cfg.eval_data_name:
        traits_config = DEFAULT_TRAITS.get(cfg.eval_data_name)
        if traits_config is not None:
            # If it's a dict (subset-specific), flatten all traits for backward compat
            if isinstance(traits_config, dict):
                all_traits = []
                for subset_traits in traits_config.values():
                    all_traits.extend(subset_traits)
                return list(set(all_traits))  # dedupe
            return traits_config
    raise ValueError("Cannot resolve traits for LLM judge evaluation. Please specify eval_traits or use a known eval_data_name.")


def _resolve_traits_by_subset(cfg, subsets: Dict[str, List]) -> Dict[str, List[str]]:
    """
    Resolve traits per subset for multi-subset mode.
    
    Args:
        cfg: EvalConfig
        subsets: Dict mapping subset_name -> list of eval items
        
    Returns:
        Dict mapping subset_name -> list of trait names
    """
    if cfg.eval_traits:
        # If explicit traits specified, use same traits for all subsets
        traits = [t.strip() for t in cfg.eval_traits]
        return {subset_name: traits for subset_name in subsets.keys()}
    
    if cfg.eval_data_name:
        traits_config = DEFAULT_TRAITS.get(cfg.eval_data_name)
        if traits_config is not None:
            if isinstance(traits_config, dict):
                # Map each subset to its specific traits
                result = {}
                for subset_name in subsets.keys():
                    if subset_name in traits_config:
                        result[subset_name] = traits_config[subset_name]
                    else:
                        # Fallback: use all traits if subset not found
                        all_traits = []
                        for subset_traits in traits_config.values():
                            all_traits.extend(subset_traits)
                        result[subset_name] = list(set(all_traits))
                return result
            else:
                # Old format: same traits for all subsets
                return {subset_name: traits_config for subset_name in subsets.keys()}
    
    raise ValueError("Cannot resolve traits for LLM judge evaluation. Please specify eval_traits or use a known eval_data_name.")

def _extract_question_and_answer_fields(item):
    """
    Extract question and answer-related fields from eval data item.
    Supports:
      - Old format: item is a string (the question)
      - New format: item is a dict with 'question' and answer fields like:
        - 'ground_truth' (for free response)
        - 'correct_answer' (for MC1 - single correct answer)
        - 'correct_answers' (for MC2 - multiple correct answers)
        - 'reference_answers' (dict with 'high_quality' and 'low_quality' for truthfulness eval)
    
    Returns:
        (question, answer_fields_dict) where answer_fields_dict contains
        all answer-related fields that should be passed to the judge
    """
    if isinstance(item, str):
        return item, {}
    elif isinstance(item, dict):
        question = item.get("question", item.get("prompt", ""))
        answer_fields = {}
        # Extract all possible answer field types
        for field in ["ground_truth", "correct_answer", "correct_answers"]:
            if field in item:
                answer_fields[field] = item[field]
        # Handle nested reference_answers (used for truthfulness evals with high/low quality refs)
        if "reference_answers" in item and isinstance(item["reference_answers"], dict):
            ref_answers = item["reference_answers"]
            for ref_field in ["high_quality", "low_quality"]:
                if ref_field in ref_answers:
                    answer_fields[ref_field] = ref_answers[ref_field]
        return question, answer_fields
    else:
        raise ValueError(f"Unsupported eval data item type: {type(item)}")

def eval_llm_judge(cfg: EvalConfig) -> None:
    model = cfg.model
    tokenizer = cfg.tokenizer
    
    # Ensure padding settings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Determine if we have subsets (dict) or single eval set (list)
    eval_data = cfg.llm_judge_eval_data
    
    if isinstance(eval_data, dict):
        # New format: {"mc1": [...], "mc2": [...], "free_response": [...]}
        subsets = eval_data
        is_multi_subset = True
    else:
        # Old format: single list, treat as one subset called "default"
        subsets = {"default": eval_data}
        is_multi_subset = False
    
    # Resolve traits per subset
    traits_by_subset = _resolve_traits_by_subset(cfg, subsets)
    
    # Collect all unique traits for judge initialization
    all_traits = set()
    for subset_traits in traits_by_subset.values():
        all_traits.update(subset_traits)
    all_traits = list(all_traits)

    # Process each subset: generate separately, then gather
    all_local_results = []
    
    for subset_name, subset_data in subsets.items():
        if is_main_process():
            print(f"[Rank 0] Processing subset: {subset_name} ({len(subset_data)} items)")
        
        # 1. Prepare prompts for this subset
        questions_and_fields = [_extract_question_and_answer_fields(item) for item in subset_data]
        base_questions = [q for q, _ in questions_and_fields]
        base_answer_fields = [fields for _, fields in questions_and_fields]
        
        base_prompts_chat = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], 
                tokenize=False, 
                add_generation_prompt=True
            ) for q in base_questions
        ]
        raw_questions = [q for q in base_questions for _ in range(cfg.repeats)]
        raw_answer_fields = [fields for fields in base_answer_fields for _ in range(cfg.repeats)]
        all_prompts_chat = [p for p in base_prompts_chat for _ in range(cfg.repeats)]

        # 2. Slice work for THIS rank
        local_prompts = split_list_for_distributed(all_prompts_chat)
        local_questions = split_list_for_distributed(raw_questions)
        local_answer_fields = split_list_for_distributed(raw_answer_fields)

        # 3. Generate Fine-Tuned Answers (Distributed) for THIS subset
        local_ft_answers = _generate_text_only(
            model, tokenizer, local_prompts, cfg.gen_max_new_tokens, 
            cfg.gen_temperature, cfg.gen_top_p, cfg.gen_batch_size
        )

        # 4. Pack local results with subset tag
        for i, q in enumerate(local_questions):
            result = {
                "subset": subset_name,
                "prompt": q,
                "ft_answer": local_ft_answers[i],
            }
            # Include all answer fields (ground_truth, correct_answer, correct_answers, etc.)
            result.update(local_answer_fields[i])
            all_local_results.append(result)

    # 5. Gather Everything to Rank 0
    gathered_lists = [None for _ in range(get_world_size())]
    if dist.is_initialized():
        dist.all_gather_object(gathered_lists, all_local_results)
    else:
        gathered_lists = [all_local_results]

    # ---------------------------------------------------------
    # EVERYTHING BELOW THIS RUNS ONLY ON RANK 0
    # ---------------------------------------------------------
    if is_main_process():
        # Flatten the list of lists into one big list of examples
        full_results = [item for sublist in gathered_lists for item in sublist]
        print(f"[Rank 0] Gathered {len(full_results)} samples across {len(subsets)} subsets.")
        print(f"[Rank 0] Starting judge with {len(all_traits)} unique traits: {all_traits}")

        # Initialize Judges for ALL traits
        judges = {}
        for t in all_traits:
            # Fallback logic for prompt keys: try "{t}_0_3" then just "{t}"
            prompt_template = Prompts.get(f"{t}_0_3", Prompts.get(t))
            if prompt_template is None:
                print(f"[Rank 0] Warning: No prompt template found for trait '{t}', skipping.")
                continue
            judges[t] = OpenAiJudge(cfg.judge_model, prompt_template, t)

        # Async Judge Execution
        async def _run_judging():
            sem = asyncio.Semaphore(cfg.judge_concurrency)

            async def evaluate(res_obj, trait, judge_inst):
                async with sem:
                    judge_kwargs = {
                        "question": res_obj["prompt"],
                        "answer": res_obj["ft_answer"]
                    }
                    # Pass all answer-related fields to the judge
                    # (ground_truth for free response, correct_answer for MC1, correct_answers for MC2,
                    #  high_quality/low_quality for truthfulness evals with reference answers)
                    for field in ["ground_truth", "correct_answer", "correct_answers", "high_quality", "low_quality"]:
                        if res_obj.get(field) is not None:
                            judge_kwargs[field] = res_obj[field]
                    ft_res = await judge_inst(**judge_kwargs)
                return trait, ft_res

            job_results = [{} for _ in full_results]

            async def wrapped_task(idx, item, t, j):
                res = await evaluate(item, t, j)
                return idx, res

            # Launch tasks - each result is judged only by its subset's traits
            tasks = []
            for idx, item in enumerate(full_results):
                subset_name = item["subset"]
                subset_traits = traits_by_subset.get(subset_name, [])
                for t in subset_traits:
                    if t in judges:
                        tasks.append(wrapped_task(idx, item, t, judges[t]))

            # Run tasks with progress bar
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging (Rank 0)"):
                idx, (t, ft_res) = await fut
                job_results[idx][t] = {"ft": ft_res}
            
            return job_results

        judge_out = asyncio.run(_run_judging())

        # 7. Saving to CSV (one file per subset if multi-subset, else single file)
        tag = cfg.eval_data_name or "generic"
        
        # Answer fields that may be present in results (different subsets use different fields)
        answer_field_names = ["ground_truth", "correct_answer", "correct_answers", "high_quality", "low_quality"]
        
        # Metrics accumulator (per subset and overall)
        metrics_acc_by_subset = {subset_name: {f"ft_{t}": [] for t in traits_by_subset.get(subset_name, [])} 
                                  for subset_name in subsets.keys()}
        
        for subset_name in subsets.keys():
            subset_results = [r for r in full_results if r["subset"] == subset_name]
            subset_traits = traits_by_subset.get(subset_name, [])
            
            # Find which answer fields are present in this subset
            subset_answer_fields = [f for f in answer_field_names 
                                     if any(r.get(f) is not None for r in subset_results)]
            
            if is_multi_subset:
                out_csv = cfg.judge_out_csv or os.path.join(
                    cfg.work_dir, f"judge_{tag}_{subset_name}_epoch{cfg.epoch}.csv"
                )
            else:
                out_csv = cfg.judge_out_csv or os.path.join(
                    cfg.work_dir, f"judge_{tag}_epoch{cfg.epoch}.csv"
                )
            make_dir_wide_permissions(os.path.dirname(out_csv))
            
            fieldnames = ["prompt"]
            if is_multi_subset:
                fieldnames.append("subset")
            # Add answer fields present in this subset
            fieldnames.extend(subset_answer_fields)
            fieldnames.append("ft_answer")
            for t in subset_traits:
                fieldnames.extend([f"ft_{t}", f"ft_{t}_rationale"])

            with open(out_csv, "w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for row_data in subset_results:
                    idx = full_results.index(row_data)
                    csv_row = {"prompt": row_data["prompt"], "ft_answer": row_data["ft_answer"]}
                    if is_multi_subset:
                        csv_row["subset"] = row_data["subset"]
                    # Include all present answer fields
                    for field in subset_answer_fields:
                        val = row_data.get(field)
                        # Handle lists (like correct_answers for MC2)
                        if isinstance(val, list):
                            csv_row[field] = str(val)
                        else:
                            csv_row[field] = val if val is not None else ""

                    judgments = judge_out[idx]
                    
                    for t in subset_traits:
                        if t in judgments:
                            ft_s, ft_r = judgments[t]["ft"] or (None, None)
                        else:
                            ft_s, ft_r = None, None
                        
                        csv_row[f"ft_{t}"] = ft_s
                        csv_row[f"ft_{t}_rationale"] = ft_r

                        # Collect for averages
                        try:
                            if ft_s is not None:
                                metrics_acc_by_subset[subset_name][f"ft_{t}"].append(float(ft_s))
                        except ValueError:
                            pass
                    
                    writer.writerow(csv_row)
            
            print(f"[Rank 0] Detailed CSV for subset '{subset_name}' saved to {out_csv}")

        # 8. Compute Aggregate Metrics & Log
        final_metrics = {}
        
        # Per-subset metrics
        for subset_name, metrics_acc in metrics_acc_by_subset.items():
            prefix = f"{subset_name}_" if is_multi_subset else ""
            for k, vals in metrics_acc.items():
                if vals:
                    final_metrics[f"{prefix}{k}_avg"] = sum(vals) / len(vals)
                    # proportions (0,1,2,3)
                    for s in range(4):
                        final_metrics[f"{prefix}{k}_score_{s}_prop"] = sum(1 for v in vals if v==s) / len(vals)

        # Save JSONL locally
        _save_metrics_locally(cfg.work_dir, "llm_judge", final_metrics, cfg.epoch, cfg.global_step)
        
        # Log to WandB
        if getattr(wandb, "run", None) is not None:
            wb_dict = {"global_step": cfg.global_step, "epoch": cfg.epoch}
            for k, v in final_metrics.items():
                wb_dict[f"eval_judge/{k}"] = v
            wandb.log(wb_dict)