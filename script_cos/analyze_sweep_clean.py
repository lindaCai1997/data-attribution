"""
Analyze experiment results for data selection methods.

Directory structure:
  - SAE methods: {root_dir}/{train_data_name}/{method_dir}/{run_dir}/selected_data/...
  - Persona vector: {pv_dir}/{train_data_name}/{run_dir}/selected_data/...

Usage:
  python analyze_sweep_clean.py \
    --root-dir /scratch7/users/aypan/tcai-scores/goodfire_l19 \
    --train-data-name wildchat_1m \
    --pv-dir /scratch7/users/aypan/tcai-scores/persona_vector_layer_19 \
    --output-dir ./analysis_output
"""

import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Constants
# =============================================================================
# Projection methods to analyze separately (set to None to skip filtering)
PROJECTION_METHODS = ["cos_sim"] #, "cos_sim_debias_train"

# Mapping of legacy projection method names to current names
PROJECTION_METHOD_ALIASES = {
    "cos_sim_debias": "cos_sim_debias_train",  # legacy name
}


def normalize_projection_method(method: str) -> str:
    """Normalize projection method name, handling legacy aliases."""
    if method is None:
        return "cos_sim"  # default
    return PROJECTION_METHOD_ALIASES.get(method, method)

METHOD_DIRS = [
    "residual_diff",
    #"residual_treatment",
    #"residual_input_treatment",
    "grad_act_treatment",
    #"residual_change",
    #"residual_change_with_bias_and_mask",
    "residual_change_treatment",
    "residual_change_treatment_with_mask_corrected",
    #"residual_change_treatment_with_bias",
    #"residual_change_treatment_with_mask_eval",
    #"residual_change_treatment_with_mask",
    "random",
    #"mlp_change_treatment",
    #"mlp_change_treatment_all_layers",
    #"mlp_change_diff",
    #"mlp_change_diff_all_layers"
]

# add valid sae_selection_method combinations here
VALID_SAE_SELECTION_METHODS = [
    "no_selection",
    "random",
    "random+none",
    "persona_vector_gen",
    #"mlp_change_treatment+none",
    #"mlp_change_treatment_all_layers+none",
    #"mlp_change_diff+none",
    #"mlp_change_diff_all_layers+none",
    "residual_diff+none",
    #"residual_treatment+none",
    "residual_change+none",
    #"residual_change_with_bias_and_mask+none",
    #"grad_act_treatment+none",
    "residual_change_treatment+none",
    #"residual_change_treatment_with_mask_corrected+none",
    #"residual_change_treatment_with_bias+none",
    #"residual_change_treatment_with_mask+none",
    #"residual_change_treatment_with_mask_eval+none",
]
VALID_ATT_METHODS = [
    "random+none",
    "residual_diff+none",
    #"residual_treatment+none",
    #"residual_input_treatment+none",
    #"residual_change+none",
    #"residual_change_with_bias_and_mask+none",
    "grad_act_treatment+none",
    "residual_change_treatment+none",
    "residual_change_treatment_with_mask_corrected+none",
    #"residual_change_treatment_with_bias+none",
    #"residual_change_treatment_with_mask+none",
    #"residual_change_treatment_with_mask_eval+none",
    #"mlp_change_treatment+none",
    #"mlp_change_treatment_all_layers+none",
    #"mlp_change_diff+none",
    #"mlp_change_diff_all_layers+none",
]

# Order for sae_selection_method columns in heatmaps (methods not in list go at end, alphabetically)
SAE_SELECTION_ORDER = [
    "no_selection",
    "random",
    "persona_vector_gen",
    "grad_act_treatment+none",
    "residual_diff+none",
    "residual_treatment+none",
    "residual_change+none",
    "residual_change_with_bias_and_mask+none",
    "residual_change_treatment+none",
    "residual_change_treatment_with_mask_corrected+none",
    "residual_change_treatment_with_bias+none",
    #"residual_change_treatment_with_mask+none",
    #"residual_change_treatment_with_mask_eval+none",
    # "persona_vector+encoder",
    # "persona_vector+decoder",
    # "residual_diff+encoder",
    # "residual_diff+decoder",
    # "residual_treatment+encoder",
    # "residual_treatment+decoder",
    # "residual_change+encoder",
    # "residual_change+decoder",
    # "residual_change_with_bias_and_mask+encoder",
    # "residual_change_with_bias_and_mask+decoder",
    # "residual_change_treatment+encoder",
    # "residual_change_treatment+decoder",
    # "residual_change_treatment_with_bias+encoder",
    # "residual_change_treatment_with_bias+decoder",
    # "residual_change_treatment_with_mask+encoder",
    # "residual_change_treatment_with_mask+decoder",
    # "residual_change_treatment_with_mask_eval+encoder",
    # "residual_change_treatment_with_mask_eval+decoder",
]

TRAIN_DATA_CHOICES = ["dolly_10k", "ultrachat_200k", "wildchat_1m", "all_25_gpt_evals", "openorca_200k", "openorca_200k_math"]

TRAITS_BY_DATASET = {
    #"truthful_qa": ["truthful_qa_0_1"],
    #"mistake_math_complete": ["math_reasoning_validity_0_1","math_final_answer_correctness_0_1"], 
    #"mistake_math_complete": ["math_reasoning_validity_0_1"],
    #"math_reasoning_validity_0_1",
    "medhallu_easy_with_knowledge": ["medical_consistency_0_2"],
    "medhallu_medium_with_knowledge": ["medical_consistency_0_2"],
    "medhallu_hard_with_knowledge": ["medical_consistency_0_2"],
    #"medhallu_easy_without_knowledge": ["medical_hallucination", "medical_consistency_0_1"],
    #"medhallu_medium_with_knowledge": ["medical_hallucination", "medical_consistency_0_1"],
    #"medhallu_medium_without_knowledge": ["medical_hallucination", "medical_consistency_0_1"],
    #"medhallu_hard_with_knowledge": ["medical_hallucination"],
    #"medhallu_hard_without_knowledge": ["medical_hallucination"],
    # "evil_gpt": ["evil"],
    # "hallucination_gpt": ["hallucination"],
    # "sycophancy_gpt": ["sycophancy"],
    # "overconfidence_gpt": ["overconfidence"],
    # "passive_aggression_gpt": ["passive_aggression"],
    # "laziness_gpt": ["laziness"],
    # "preachiness_gpt": ["preachiness"],
    # "defensiveness_gpt": ["defensiveness"],
    # "intellectual_arrogance_gpt": ["intellectual_arrogance"],
    # "pedantry_gpt": ["pedantry"],
    # "extreme_politeness_gpt": ["extreme_politeness"],
    # "unwavering_optimism_gpt": ["unwavering_optimism"],
    # "empathy_gpt": ["empathy"],
    # "modesty_gpt": ["modesty"],
    # "happiness_gpt": ["happiness"],
}

LOWER_IS_BETTER_METRICS = {"ce_treatment"}


def sort_by_order(items: List[str], order: List[str]) -> List[str]:
    """Sort items by a predefined order. Items not in order go at end, alphabetically."""
    order_map = {v: i for i, v in enumerate(order)}
    max_idx = len(order)
    return sorted(items, key=lambda x: (order_map.get(x, max_idx), x))


# =============================================================================
# JSON Helpers
# =============================================================================
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, return None on error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    """Return the last valid JSON object from a JSONL file."""
    last_obj = None
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, OSError):
        return None
    return last_obj


def load_first_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    """Return the first valid JSON object from a JSONL file."""
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, OSError):
        return None
    return None


# =============================================================================
# Run Discovery
# =============================================================================
def parse_datetime_token(dt_token: str) -> Optional[datetime]:
    """Parse datetime tokens like 20251225_101504 or 20251225_1015."""
    if not dt_token:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M"):
        try:
            return datetime.strptime(dt_token, fmt)
        except ValueError:
            continue
    return None


def discover_sae_runs(root_dir: str, train_data_name: str) -> List[Dict[str, Any]]:
    """
    Discover SAE method runs under: {root_dir}/{train_data_name}/{method_dir}/{run_dir}/
    Each run_dir must contain config.json and selected_data/.
    
    Also handles persona_vector runs integrated in the same folder structure.
    For these runs:
    - attribution_method = the folder name (e.g., "residual_change_treatment_with_mask")
    - sae_selection_method = the config's attribution_method (e.g., "persona_vector_gen")
    """
    base = Path(root_dir) / train_data_name
    if not base.exists():
        print(f"Warning: train data dir not found: {base}")
        return []

    runs = []
    for method in METHOD_DIRS:
        method_dir = base / method
        if not method_dir.exists():
            continue

        for run_dir in sorted(method_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "config.json").exists():
                continue
            if not (run_dir / "selected_data").exists():
                continue
            if not (run_dir / "selected_data" / "eval_llm_judge" / "metrics.jsonl").exists():
                continue


            cfg = load_json(run_dir / "config.json") or {}
            
            # Parse datetime from run_dir name (last segment after '-')
            name_parts = run_dir.name.split("-")
            dt_token = name_parts[-1] if name_parts else None
            
            cfg_attribution_method = cfg.get("attribution_method")
            cfg_sae_selection_method = cfg.get("sae_selection_method")
            cfg_pv_generation_method = cfg.get("pv_generation_method")
            
            # Check if this is a persona_vector run (no SAE selection, uses PV for data selection)
            is_pv_run = cfg_pv_generation_method is not None
            
            if is_pv_run:
                # For persona_vector runs:
                # - attribution_method = config's attribution_method (how scores were computed)
                # - sae_selection_method = pv_generation_method (the selection method, e.g., "persona_vector_gen")
                attribution_method = cfg_attribution_method
                sae_selection_method = cfg_pv_generation_method
            else:
                # Regular SAE runs
                attribution_method = cfg_attribution_method
                sae_selection_method = cfg_sae_selection_method
            
            if (not sae_selection_method in VALID_SAE_SELECTION_METHODS): #or (not attribution_method in VALID_ATT_METHODS):
                continue
            # Extract sae_weight_method, default to "uniform" if not present
            sae_weight_method = cfg.get("sae_weight_method", "uniform")
            # Extract projection_method, default to "cos_sim" if not present, and normalize legacy names
            projection_method = normalize_projection_method(cfg.get("projection_method"))
            
            base_run = {
                "run_dir": str(run_dir),
                "train_data_name": cfg.get("train_data_name", train_data_name),
                "eval_data_name": cfg.get("eval_data_name"),
                "attribution_method": attribution_method,
                "sae_selection_method": sae_selection_method,
                "sae_weight_method": sae_weight_method,
                "projection_method": projection_method,
                "k1": cfg.get("k1"),
                "k2": cfg.get("k2"),
                "_datetime": parse_datetime_token(dt_token),
            } 
            runs.append(base_run)
            
            # For persona_vector runs, also create "no_selection" baseline (first-step metrics)
            if is_pv_run:
                no_selection_run = {
                    **base_run,
                    "sae_selection_method": "no_selection",
                    "_use_first_step": True,
                }
                runs.append(no_selection_run)

    return runs


def discover_pv_runs(pv_dir: str, train_data_name: str) -> List[Dict[str, Any]]:
    """
    Discover persona_vector runs under: {pv_dir}/{train_data_name}/{run_dir}/
    Also creates 'no_selection' baseline entries using first-step metrics.
    """
    base = Path(pv_dir) / train_data_name
    if not base.exists():
        print(f"Warning: PV train data dir not found: {base}")
        return []

    runs = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "config.json").exists():
            continue
        if not (run_dir / "selected_data").exists():
            continue
        #if not (run_dir / "selected_data" / "eval_llm_judge" / "metrics.jsonl").exists():
        #    continue


        cfg = load_json(run_dir / "config.json") or {}
        
        # Parse datetime from run_dir name
        name_parts = run_dir.name.split("-")
        dt_token = name_parts[-1] if name_parts else None

        # Extract sae_weight_method, default to "uniform" if not present
        sae_weight_method = cfg.get("sae_weight_method", "uniform")
        # Extract projection_method, default to "cos_sim" if not present, and normalize legacy names
        projection_method = normalize_projection_method(cfg.get("projection_method"))

        base_run = {
            "run_dir": str(run_dir),
            "train_data_name": cfg.get("train_data_name", train_data_name),
            "eval_data_name": cfg.get("eval_data_name"),
            "sae_selection_method": "N/A",
            "sae_weight_method": sae_weight_method,
            "projection_method": projection_method,
            "k1": None,
            "k2": cfg.get("k2"),
            "_datetime": parse_datetime_token(dt_token),
        }

        # Main persona_vector run (last metrics)
        pv_run = {**base_run, "attribution_method": cfg.get("attribution_method", "persona_vector")}
        runs.append(pv_run)

        # no_selection baseline (first-step metrics)
        ns_run = {**base_run, "attribution_method": "no_selection", "_use_first_step": True}
        runs.append(ns_run)

    return runs


def keep_latest_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the most recent run per unique config."""
    def key_fn(r):
        return (
            r.get("train_data_name"),
            r.get("eval_data_name"),
            r.get("attribution_method"),
            r.get("sae_selection_method"),
            r.get("sae_weight_method"),
            normalize_projection_method(r.get("projection_method")),
            r.get("k1"),
            r.get("k2"),
        )

    best: Dict[tuple, Dict[str, Any]] = {}
    for r in runs:
        k = key_fn(r)
        cur = best.get(k)
        if cur is None:
            best[k] = r
            continue

        rdt, cdt = r.get("_datetime"), cur.get("_datetime")
        # Case 1: Both have valid datetime - compare them
        if isinstance(rdt, datetime) and isinstance(cdt, datetime):
            if rdt > cdt:
                best[k] = r
        # Case 2: Only r has datetime - prefer r (runs with datetime are more reliable)
        elif isinstance(rdt, datetime):
            best[k] = r
        # Case 3: Only cur has datetime - keep cur (prefer runs with datetime)
        elif isinstance(cdt, datetime):
            # cur already has datetime, keep it
            continue
        # Case 4: Neither has datetime - compare run_dir names (datetime token is last segment)
        else:
            # Extract datetime token from run_dir name for comparison
            r_name = Path(r.get("run_dir", "")).name
            cur_name = Path(cur.get("run_dir", "")).name
            r_token = r_name.split("-")[-1] if "-" in r_name else r_name
            cur_token = cur_name.split("-")[-1] if "-" in cur_name else cur_name
            # Lexicographic comparison works for YYYYMMDD_HHMMSS format
            if r_token > cur_token:
                best[k] = r

    dropped = len(runs) - len(best)
    if dropped > 0:
        print(f"Kept {len(best)} runs, dropped {dropped} older duplicates.")
    return list(best.values())


# =============================================================================
# Metrics Loading
# =============================================================================
def load_run_metrics(run_dir: str, use_first_step: bool = False) -> Dict[str, Any]:
    """Load CE and LLM judge metrics for a run."""
    run_path = Path(run_dir)
    metrics: Dict[str, Any] = {}
    load_fn = load_first_jsonl if use_first_step else load_last_jsonl

    # Cross-entropy metrics
    ce_file = run_path / "selected_data" / "eval_cross_entropy" / "metrics.jsonl"
    ce_data = load_fn(ce_file)
    if ce_data:
        metrics["ce_treatment"] = ce_data.get("mean_ce_treatment")
        metrics["ce_control"] = ce_data.get("mean_ce_control")
        metrics["ce_gap"] = ce_data.get("gap_treat_minus_control")

    # LLM judge metrics
    judge_file = run_path / "selected_data" / "eval_llm_judge" / "metrics.jsonl"
    judge_data = load_fn(judge_file)
    if judge_data:
        for key, value in judge_data.items():
            if isinstance(key, str) and key.startswith("ft_"):
                if key.endswith("_average") or key.endswith("_avg"):
                    trait = key.replace("ft_", "").replace("_average", "").replace("_avg", "")
                    metrics[f"llm_judge_{trait}"] = value

    return metrics


def build_dataframe(
    root_dir: str,
    train_data_name: str,
    pv_dir: Optional[str] = None,
    k1: Optional[int] = None,
    k2: Optional[int] = None,
    sae_weight_method: Optional[str] = None,
    projection_method: Optional[str] = None,
) -> pd.DataFrame:
    """Build DataFrame from all discovered runs with their metrics."""
    # Discover runs
    runs = discover_sae_runs(root_dir, train_data_name)
    print(f"Found {len(runs)} SAE method runs")

    if pv_dir:
        pv_runs = discover_pv_runs(pv_dir, train_data_name)
        print(f"Found {len(pv_runs)} persona_vector runs")
        runs.extend(pv_runs)

    runs = keep_latest_runs(runs)

    # Filter runs with valid eval_data_name
    runs = [r for r in runs if r.get("eval_data_name") in TRAITS_BY_DATASET]

    # Filter by k1 if specified
    if k1 is not None:
        def passes_k1_filter(r):
            r_k1 = r.get("k1")
            # Keep runs with matching k1
            if r_k1 == k1:
                return True
            # Keep PV runs (identified by sae_selection_method == "N/A") which don't use 
            if r_k1 is None and (r.get("sae_selection_method") == "persona_vector_gen" or r.get("sae_selection_method") == "no_selection"):

                return True
            return False
        runs = [r for r in runs if passes_k1_filter(r)]
        print(f"Filtered to {len(runs)} runs with k1={k1}")

    # Filter by k2 if specified
    if k2 is not None:
        runs = [r for r in runs if int(r.get("k2")) == int(k2)]
        print(f"Filtered to {len(runs)} runs with k2={k2}")

    # Filter by sae_weight_method if specified
    # If not present in config, it defaults to "uniform"
    if sae_weight_method is not None:
        runs = [r for r in runs if r.get("sae_weight_method", "uniform") == sae_weight_method]
        print(f"Filtered to {len(runs)} runs with sae_weight_method={sae_weight_method}")

    # Filter by projection_method if specified
    # If not present in config, it defaults to "cos_sim"
    if projection_method is not None:
        runs = [r for r in runs if normalize_projection_method(r.get("projection_method")) == projection_method]
        print(f"Filtered to {len(runs)} runs with projection_method={projection_method}")

    if not runs:
        print("No valid runs found!")
        return pd.DataFrame()

    # Load metrics for each run
    data = []
    for run in runs:
        use_first = run.get("_use_first_step", False)
        metrics = load_run_metrics(run["run_dir"], use_first_step=use_first)
        
        row = {
            "run_dir": run["run_dir"],
            "train_data_name": run.get("train_data_name"),
            "eval_data_name": run.get("eval_data_name"),
            "attribution_method": run.get("attribution_method"),
            "sae_selection_method": run.get("sae_selection_method"),
            "sae_weight_method": run.get("sae_weight_method", "uniform"),
            "projection_method": normalize_projection_method(run.get("projection_method")),
            "k1": run.get("k1"),
            "k2": run.get("k2"),
            **metrics,
        }
        data.append(row)

    return pd.DataFrame(data)


# =============================================================================
# Heatmap Generation
# =============================================================================
def get_traits_for_eval(eval_data_name: str) -> List[str]:
    """Get LLM judge traits for an eval dataset."""
    return TRAITS_BY_DATASET.get(eval_data_name, [])


def build_k_params_str(k1: Optional[int], k2: Optional[int], projection_method: Optional[str] = None) -> str:
    """Build a string like ' (k1=500, k2=500, proj=cos_sim)' for plot titles."""
    parts = []
    if k1 is not None:
        parts.append(f"k1={k1}")
    if k2 is not None:
        parts.append(f"k2={k2}")
    if projection_method is not None:
        parts.append(f"proj={projection_method}")
    return f" ({', '.join(parts)})" if parts else ""


def is_lower_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER_METRICS


def to_score(values: pd.Series, metric: str) -> pd.Series:
    """Convert to higher-is-better score."""
    return -values if is_lower_better(metric) else values


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to [attribution_method x sae_selection_method]."""
    subset = df.dropna(subset=["attribution_method", "sae_selection_method"])
    return subset.pivot_table(
        values=metric,
        index="attribution_method",
        columns="sae_selection_method",
        aggfunc="mean",
    )


def plot_heatmap(
    mat: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "RdYlGn",
    fmt: str = ".3f",
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """Create and save a heatmap."""
    fig, ax = plt.subplots(figsize=(14, 10))
    #fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, ax=ax, center=center, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Selection Method")
    ax.set_ylabel("Attribution Method")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_per_eval_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
    all_attrs: List[str],
    all_saes: List[str],
    train_data_name: str,
    k1: Optional[int] = None,
    k2: Optional[int] = None,
    projection_method: Optional[str] = None,
) -> List[Path]:
    """Create heatmaps for each eval trait and metric."""
    saved = []
    eval_names = df["eval_data_name"].dropna().unique()
    metrics = [c for c in df.columns if c.startswith(("ce_", "llm_judge_"))]
    k_str = build_k_params_str(k1, k2, projection_method)

    for eval_name in eval_names:
        subset = df[df["eval_data_name"] == eval_name]
        if subset.empty:
            continue

        for metric in metrics:
            if metric not in subset.columns or subset[metric].isna().all():
                continue

            pv = pivot_metric(subset, metric)
            if pv.empty:
                continue

            pv = pv.reindex(index=all_attrs, columns=all_saes)
            lower = is_lower_better(metric)
            cmap = "RdYlGn_r" if lower else "RdYlGn"
            direction = "lower better" if lower else "higher better"

            safe_eval = eval_name.replace("/", "_")
            out = output_dir / f"{train_data_name}{k_str}_heatmap_{metric}_{safe_eval}.png"
            plot_heatmap(pv, f"{train_data_name}{k_str}: {metric}\neval={eval_name} ({direction})", out, cmap=cmap)
            saved.append(out)

    return saved


def aggregate_across_evals(
    pivots: List[pd.DataFrame],
    all_attrs: List[str],
    all_saes: List[str],
    mode: str = "avg",  # "avg", "zscore", "winrate"
    require_all: bool = True,  # Only include cells with data for ALL eval traits
) -> Optional[pd.DataFrame]:
    """Aggregate multiple pivots across eval traits.
    
    If require_all=True (default), only cells that have valid data in ALL pivots
    will have a value; others will be NaN. This ensures aggregates are not skewed
    by incomplete runs.
    """
    if not pivots:
        return None

    aligned = [p.reindex(index=all_attrs, columns=all_saes) for p in pivots]
    
    # First pass: collect raw arrays and track which pivots are valid
    raw_stack = []
    for p in aligned:
        vals = p.to_numpy().astype(float)
        if not np.isfinite(vals).any():
            continue
        raw_stack.append(vals)

    if not raw_stack:
        return None

    # Create mask: True where ALL pivots have finite values
    stacked_raw = np.stack(raw_stack, axis=0)  # shape: (n_pivots, n_attrs, n_saes)
    all_valid_mask = np.all(np.isfinite(stacked_raw), axis=0)  # shape: (n_attrs, n_saes)

    # Second pass: apply transformations (zscore/winrate) per pivot
    stack = []
    for vals in raw_stack:
        if mode == "zscore":
            flat = vals[np.isfinite(vals)]
            if flat.size < 2:
                continue
            mu, sigma = np.nanmean(flat), np.nanstd(flat)
            if sigma == 0:
                continue
            vals = (vals - mu) / sigma
        elif mode == "winrate":
            maxv = np.nanmax(vals)
            if not np.isfinite(maxv):
                continue
            w = (vals == maxv).astype(float)
            w[~np.isfinite(vals)] = np.nan
            vals = w
        stack.append(vals)

    if not stack:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        out = np.nanmean(np.stack(stack, axis=0), axis=0)

    # Apply mask: set cells without ALL eval traits to NaN
    if require_all:
        out[~all_valid_mask] = np.nan

    return pd.DataFrame(out, index=all_attrs, columns=all_saes)


def create_aggregate_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
    all_attrs: List[str],
    all_saes: List[str],
    train_data_name: str,
    k1: Optional[int] = None,
    k2: Optional[int] = None,
    projection_method: Optional[str] = None,
    suffix: str = "",
    eval_filter: Optional[List[str]] = None, 
) -> List[Path]:
    """Create aggregated heatmaps (avg, zscore, winrate) across eval traits.
    
    Generates separate aggregates for each unique trait across all eval datasets.
    """
    saved = []
    eval_names = eval_filter or list(df["eval_data_name"].dropna().unique())
    k_str = build_k_params_str(k1, k2, projection_method)

    # Collect pivots for llm_judge aggregate, grouped by trait
    llm_pivots_by_trait: Dict[str, List[pd.DataFrame]] = {}
    for eval_name in eval_names:
        subset = df[df["eval_data_name"] == eval_name]
        if subset.empty:
            continue

        traits = get_traits_for_eval(eval_name)
        # Process all traits, grouping by trait name
        for t in traits:
            col = f"llm_judge_{t}"
            if col not in subset.columns or subset[col].isna().all():
                continue

            tmp = subset.copy()
            tmp["__score__"] = to_score(tmp[col], col)
            pv = pivot_metric(tmp, "__score__")
            if pv is not None and not pv.empty:
                if t not in llm_pivots_by_trait:
                    llm_pivots_by_trait[t] = []
                llm_pivots_by_trait[t].append(pv)

    # Create aggregates for each trait separately
    for trait, llm_pivots in llm_pivots_by_trait.items():
        for mode, label in [("avg", "Average"), ("zscore", "Z-score"), ("winrate", "Win-rate")]:
            mat = aggregate_across_evals(llm_pivots, all_attrs, all_saes, mode=mode)
            if mat is None or mat.empty:
                continue

            out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_llm_judge_{trait}_{mode}{suffix}.png"
            center = 0.0 if mode == "zscore" else None
            vmin, vmax = (0.0, 1.0) if mode == "winrate" else (None, None)
            plot_heatmap(mat, f"{train_data_name}{k_str}: LLM Judge {trait} Aggregate ({label}){suffix}", out, center=center, vmin=vmin, vmax=vmax, fmt=".2f")
            saved.append(out)

            # Save CSV
            csv_out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_llm_judge_{trait}_{mode}{suffix}.csv"
            mat.to_csv(csv_out)

    # Create aggregates across ALL traits combined
    all_llm_pivots = []
    for trait_pivots in llm_pivots_by_trait.values():
        all_llm_pivots.extend(trait_pivots)
    
    if all_llm_pivots:
        for mode, label in [("avg", "Average"), ("zscore", "Z-score"), ("winrate", "Win-rate")]:
            mat = aggregate_across_evals(all_llm_pivots, all_attrs, all_saes, mode=mode)
            if mat is None or mat.empty:
                continue

            out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_llm_judge_all_traits_{mode}{suffix}.png"
            center = 0.0 if mode == "zscore" else None
            vmin, vmax = (0.0, 1.0) if mode == "winrate" else (None, None)
            plot_heatmap(mat, f"{train_data_name}{k_str}: LLM Judge All Traits Aggregate ({label}){suffix}", out, center=center, vmin=vmin, vmax=vmax, fmt=".2f")
            saved.append(out)

            # Save CSV
            csv_out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_llm_judge_all_traits_{mode}{suffix}.csv"
            mat.to_csv(csv_out)

    # Also do ce_gap aggregate
    ce_pivots = []
    for eval_name in eval_names:
        subset = df[df["eval_data_name"] == eval_name]
        if subset.empty or "ce_gap" not in subset.columns or subset["ce_gap"].isna().all():
            continue

        tmp = subset.copy()
        tmp["__score__"] = to_score(tmp["ce_gap"], "ce_gap")
        pv = pivot_metric(tmp, "__score__")
        if pv is not None and not pv.empty:
            ce_pivots.append(pv)

    for mode, label in [("avg", "Average"), ("zscore", "Z-score"), ("winrate", "Win-rate")]:
        mat = aggregate_across_evals(ce_pivots, all_attrs, all_saes, mode=mode)
        if mat is None or mat.empty:
            continue

        out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_ce_gap_{mode}{suffix}.png"
        center = 0.0 if mode == "zscore" else None
        vmin, vmax = (0.0, 1.0) if mode == "winrate" else (None, None)
        plot_heatmap(mat, f"{train_data_name}{k_str}: CE Gap Aggregate ({label}){suffix}", out, center=center, vmin=vmin, vmax=vmax, fmt=".2f")
        saved.append(out)

        csv_out = output_dir / f"{train_data_name}{k_str}_heatmap_aggregate_ce_gap_{mode}{suffix}.csv"
        mat.to_csv(csv_out)

    return saved


def get_pv_filtered_traits(df: pd.DataFrame) -> List[str]:
    """Return eval traits where persona_vector_gen > no_selection."""
    filtered = []
    for eval_name in df["eval_data_name"].dropna().unique():
        subset = df[df["eval_data_name"] == eval_name]
        traits = get_traits_for_eval(eval_name)
        
        # Check all traits - if any trait shows PV > baseline, include this eval
        for t in traits:
            c = f"llm_judge_{t}"
            if c not in subset.columns or subset[c].isna().all():
                continue

            pv = subset[subset["attribution_method"] == "persona_vector_gen"][c].mean()
            ns = subset[subset["attribution_method"] == "no_selection"][c].mean()

            if pd.notna(pv) and pd.notna(ns) and pv > ns:
                filtered.append(eval_name)
                break  # Found at least one trait that passes, no need to check others

    return filtered


# =============================================================================
# Main
# =============================================================================
def analyze(
    root_dir: str,
    train_data_name: str,
    pv_dir: Optional[str],
    output_dir: str,
    k1: Optional[int] = None,
    k2: Optional[int] = None,
    sae_weight_method: Optional[str] = None,
    projection_method: Optional[str] = None,
    aggregate_suffix: Optional[str] = "",
):
    """Main analysis function."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing train_data={train_data_name}")
    print(f"  SAE runs: {root_dir}/{train_data_name}/")
    if pv_dir:
        print(f"  PV runs: {pv_dir}/{train_data_name}/")
    if k1 is not None:
        print(f"  Filtering k1={k1}")
    if k2 is not None:
        print(f"  Filtering k2={k2}")
    if sae_weight_method is not None:
        print(f"  Filtering sae_weight_method={sae_weight_method}")
    if projection_method is not None:
        print(f"  Filtering projection_method={projection_method}")

    # Build DataFrame
    df = build_dataframe(root_dir, train_data_name, pv_dir, k1=k1, k2=k2, sae_weight_method=sae_weight_method, projection_method=projection_method)
    if df.empty:
        print("No data found!")
        return

    # Global axes for consistent heatmaps
    all_attrs = sorted(df["attribution_method"].dropna().unique())
    all_saes = sort_by_order(list(df["sae_selection_method"].dropna().unique()), SAE_SELECTION_ORDER)

    print(f"\nLoaded {len(df)} runs")
    print(f"Attribution methods: {all_attrs}")
    print(f"SAE methods: {all_saes}")
    print(f"Eval traits: {list(df['eval_data_name'].dropna().unique())}")

    # Save raw data
    df.to_csv(out_path / "raw_data.csv", index=False)
    print(f"\nSaved raw data to {out_path / 'raw_data.csv'}")

    # Create per-eval heatmaps
    print("\nGenerating per-eval heatmaps...")
    per_eval_dir = out_path / "per_eval"
    per_eval_dir.mkdir(exist_ok=True)
    saved = create_per_eval_heatmaps(df, per_eval_dir, all_attrs, all_saes, train_data_name, k1=k1, k2=k2, projection_method=projection_method)
    print(f"  Saved {len(saved)} heatmaps")

    # Create aggregated heatmaps
    print("\nGenerating aggregated heatmaps...")
    agg_dir = out_path / "aggregated"
    agg_dir.mkdir(exist_ok=True)
    saved = create_aggregate_heatmaps(df, agg_dir, all_attrs, all_saes, train_data_name, k1=k1, k2=k2, projection_method=projection_method, suffix=aggregate_suffix)
    print(f"  Saved {len(saved)} heatmaps")

    # Create PV-filtered aggregates
    if pv_dir:
        filtered_traits = get_pv_filtered_traits(df)
        if filtered_traits:
            print(f"\nGenerating PV-filtered aggregates for {len(filtered_traits)} traits: {filtered_traits}")
            saved = create_aggregate_heatmaps(df, agg_dir, all_attrs, all_saes, train_data_name, k1=k1, k2=k2, projection_method=projection_method, suffix="_pv_filtered", eval_filter=filtered_traits)
            print(f"  Saved {len(saved)} heatmaps")
        else:
            print("\nNo traits found where PV > baseline, skipping filtered aggregates.")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Output: {out_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyze data selection experiment results.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/scratch7/users/aypan/tcai-scores/goodfire_l19",
        help="Root directory for SAE method runs",
    )
    parser.add_argument(
        "--train-data-name",
        type=str,
        required=True,
        choices=TRAIN_DATA_CHOICES,
        help="Train dataset name",
    )
    parser.add_argument(
        "--pv-dir",
        type=str,
        default=None,
        help="Directory for persona_vector runs (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {root_dir}/{train_data_name}/analysis)",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=None,
        help="Filter runs by k1 value (optional)",
    )
    parser.add_argument(
        "--k2",
        type=int,
        default=None,
        help="Filter runs by k2 value (optional)",
    )
    parser.add_argument(
        "--sae-weight-method",
        type=str,
        default=None,
        help="Filter runs by sae-weight-method value (optional). If not specified in config, defaults to 'uniform'",
    )
    parser.add_argument(
        "--aggregate-suffix",
        type=str,
        default="",
        help="Suffix for aggregated heatmaps (optional)",
    )
    parser.add_argument(
        "--projection-method",
        type=str,
        default=None,
        help="Filter runs by projection method. If not specified, uses PROJECTION_METHODS from constants.",
    )

    args = parser.parse_args()

    base_output_dir = args.output_dir or str(Path(args.root_dir) / args.train_data_name / "analysis")

    # Determine which projection methods to analyze
    if args.projection_method is not None:
        # Single projection method specified via CLI
        projection_methods = [args.projection_method]
    elif PROJECTION_METHODS is not None:
        # Use the list from constants
        projection_methods = PROJECTION_METHODS
    else:
        # No filtering - analyze all together
        projection_methods = [None]

    # Run analysis for each projection method
    for proj_method in projection_methods:
        if proj_method is not None:
            output_dir = f"{base_output_dir}_{proj_method}"
            print(f"\n{'='*60}")
            print(f"Analyzing projection_method={proj_method}")
            print(f"{'='*60}")
        else:
            output_dir = base_output_dir

        analyze(
            root_dir=args.root_dir,
            train_data_name=args.train_data_name,
            pv_dir=args.pv_dir,
            output_dir=output_dir,
            k1=args.k1,
            k2=args.k2,
            sae_weight_method=args.sae_weight_method,
            projection_method=proj_method,
            aggregate_suffix=args.aggregate_suffix,
        )


if __name__ == "__main__":
    main()

'''
train_data_name="all_25_gpt_evals" 
k1=200
k2=250
python sae_refactor/script/analyze_sweep_clean.py \
  --root-dir /scratch7/users/aypan/tcai-scores/goodfire_l19 \
  --train-data-name ${train_data_name} \
  --output-dir /scratch7/users/aypan/tcai-scores/analytics/${train_data_name}_cleaned_k1=${k1}_k2=${k2} \
  --k1 ${k1} \
  --k2 ${k2}
'''

