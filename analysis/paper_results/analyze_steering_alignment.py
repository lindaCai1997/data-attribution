"""
Compare Figure 13's steering-vector ordering against trait-score orderings in
the main/appendix tables, separately for each attribution_method.

For Llama only, ignoring MedHallu:
  1. From the steering combined_long.csv, derive the order of {persona_vector,
     residual_diff, residual_change} per (task, layer) by max-score across
     coefficients.
  2. From the on-disk LoRA-finetuning metrics, for each (task, train, layer,
     attribution_method=RD|RCT), derive the order of (PV, RD-sel, RC-sel).
  3. Report how often the triple ordering matches the steering ordering.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bar_plots import (  # noqa: E402
    MODELS,
    find_run_dir,
    read_metric,
)

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

# Per-mode steering-sweep CSVs. The norm_alpha sweep ran with --alpha-relative,
# so its coeff column is the dimensionless α in [0, 1.0]; the absolute sweep
# uses the original per-model α grids.
STEERING_CSV_BY_MODE = {
    "absolute": {
        "llama": f"{DATA_ROOT}/data/steering_summary_20260516_164307/combined_long.csv",
        "qwen":  f"{DATA_ROOT}/data/steering_qwen_summary_20260518_163045_exp_coefs/combined_long.csv",
    },
    "norm_alpha": {
        "llama": f"{DATA_ROOT}/data/steering_norm_summary_20260521_183328/combined_long.csv",
        "qwen":  f"{DATA_ROOT}/data/steering_norm_qwen_summary_20260521_183328/combined_long.csv",
    },
}
MODEL_LAYERS = {
    "llama": [15, 17, 19, 21],
    "qwen":  [13, 15, 17, 19],
}
# Number of highest coefficients to drop per model, per mode. For the
# absolute sweep we drop the single largest grid point (Llama α=16, Qwen
# α=96) because trait score there is dominated by incoherent generations.
# For the norm-aware sweep the user has asked us to include all endpoints
# including α=1.0 so the average is over the full grid.
N_TOP_COEFFS_TO_DROP_BY_MODE = {
    "absolute":   {"llama": 1, "qwen": 1},
    "norm_alpha": {"llama": 0, "qwen": 0},
}
MODEL_DISPLAY = {
    "llama": "Llama-3.1-8B-Instruct",
    "qwen":  "Qwen-2.5-7B-Instruct",
}

# (task_label, list_of_(dataset, trait_key)) — same aggregation as the main table.
TASKS = [
    ("Personality", [
        ("empathy_gpt",     "empathy"),
        ("laziness_gpt",    "laziness"),
        ("modesty_gpt",     "modesty"),
        ("preachiness_gpt", "preachiness"),
        ("sycophancy_gpt",  "sycophancy"),
    ]),
    ("UltraFB Coding", [
        ("ultra_coding_instruction_following", "ultra_instruction_following_negative_0_3"),
    ]),
    ("UltraFB Factual", [
        ("ultra_factual_truthfulness", "ultra_truthfulness_negative_0_3"),
    ]),
    ("MedHallu", [
        ("medhallu_easy_with_knowledge_balanced",   "medical_consistency_0_2"),
        ("medhallu_medium_with_knowledge_balanced", "medical_consistency_0_2"),
        ("medhallu_hard_with_knowledge_balanced",   "medical_consistency_0_2"),
    ]),
]

# Map from steering CSV's `dataset` field → which task it belongs to.
DATASET_TO_TASK = {
    "empathy_gpt": "Personality",
    "laziness_gpt": "Personality",
    "modesty_gpt": "Personality",
    "preachiness_gpt": "Personality",
    "sycophancy_gpt": "Personality",
    "ultra_coding_instruction_following": "UltraFB Coding",
    "ultra_factual_truthfulness": "UltraFB Factual",
    "medhallu_easy_with_knowledge_balanced":   "MedHallu",
    "medhallu_medium_with_knowledge_balanced": "MedHallu",
    "medhallu_hard_with_knowledge_balanced":   "MedHallu",
}

TRAIN_SETS = ["dolly_10k", "ultrachat_200k", "openorca_200k"]

# Selection methods + steering CSV name mapping.
# Order = display order PV, RD, RC.
SEL_METHODS = ["PV", "RD", "RC"]
STEERING_METHOD_NAME = {
    "PV": "persona_vector",
    "RD": "residual_diff",
    "RC": "residual_change",
}
# Stage-2 selection method names on disk (matches make_main_table.COLUMNS).
TABLE_SEL_NAME = {
    "PV": "persona_vector_gen",
    "RD": "residual_diff",
    "RC": "residual_change",
}
ATTR_METHODS = [
    ("RD",  "residual_diff+none",             "residual_diff"),
    ("RCT", "residual_change_treatment+none", "residual_change_treatment"),
]


def steering_order(df: pd.DataFrame, task: str, layer: int) -> tuple[str, ...]:
    """Return the ranking of {PV, RD, RC} for one (task, layer) cell.
    Score per method = mean trait score across coefficients (proxy for AUC of the
    dose-response curve). This avoids ceiling artefacts where multiple methods
    saturate near the trait-score cap (e.g. 3.0 for UltraFB Coding) and the
    'max' becomes meaningless."""
    score_per_method: dict[str, float] = {}
    task_datasets = [d for d, t in DATASET_TO_TASK.items() if t == task]
    for sel in SEL_METHODS:
        sub = df[
            (df.layer == layer)
            & (df.dataset.isin(task_datasets))
            & (df.method == STEERING_METHOD_NAME[sel])
        ]
        if sub.empty:
            score_per_method[sel] = np.nan
            continue
        # Per dataset: mean score across coeffs; then average over datasets in task.
        per_dataset_mean = sub.groupby("dataset")["mean_score"].mean()
        score_per_method[sel] = float(per_dataset_mean.mean())
    return tuple(sorted(SEL_METHODS, key=lambda s: -score_per_method[s]))


def table_score(model_key: str, layer: int, train: str, sub_evals: list,
                attr_method: str, attr_subdir: str, sel_method_disk: str) -> float | None:
    root = MODELS[model_key]["root_template"].format(L=layer)
    vals = []
    for eval_name, trait in sub_evals:
        run = find_run_dir(root, train, attr_subdir, attr_method, sel_method_disk, eval_name)
        if run is None:
            continue
        v = read_metric(run, trait)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def table_order(model_key: str, layer: int, train: str, sub_evals: list,
                attr_method: str, attr_subdir: str) -> tuple[str, ...] | None:
    """Return the ranking of {PV, RD, RC} for one (train, layer) cell, fixed attribution."""
    scores = {}
    for sel in SEL_METHODS:
        scores[sel] = table_score(
            model_key, layer, train, sub_evals, attr_method, attr_subdir, TABLE_SEL_NAME[sel],
        )
    if any(v is None for v in scores.values()):
        return None
    return tuple(sorted(SEL_METHODS, key=lambda s: -scores[s]))


SHORT_TASK_LABEL = {
    "Personality": "Personality",
    "UltraFB Coding": "UltraFB Coding",
    "UltraFB Factual": "UltraFB Factual",
    "MedHallu": "MedHallu",
}


def collect_alignment(model_key: str, mode: str = "absolute") -> dict:
    """Run the analysis for one model and return structured results."""
    layers = MODEL_LAYERS[model_key]
    df = pd.read_csv(STEERING_CSV_BY_MODE[mode][model_key])
    df = drop_top_coeffs(df, N_TOP_COEFFS_TO_DROP_BY_MODE[mode][model_key]).copy()

    steering_orders: dict[tuple[str, int], tuple[str, ...]] = {}
    for task, _ in TASKS:
        for layer in layers:
            steering_orders[(task, layer)] = steering_order(df, task, layer)

    match_stats: dict[tuple[str, str], tuple[int, int, int]] = {}
    for task, sub_evals in TASKS:
        for attr_label, attr_method, attr_subdir in ATTR_METHODS:
            total = 0
            full = 0
            top1 = 0
            for train, layer in product(TRAIN_SETS, layers):
                order = table_order(model_key, layer, train, sub_evals, attr_method, attr_subdir)
                if order is None:
                    continue
                total += 1
                expected = steering_orders[(task, layer)]
                if order == expected:
                    full += 1
                if order[0] == expected[0]:
                    top1 += 1
            match_stats[(task, attr_label)] = (full, top1, total)
    return {"layers": layers, "steering_orders": steering_orders, "match_stats": match_stats}


def write_latex_partial(results: dict[str, dict], out_path: Path, mode: str = "absolute") -> None:
    """Emit a self-contained LaTeX block (two tables) for Appendix C."""
    parts: list[str] = []

    if mode == "norm_alpha":
        order_label = "tab:steering_order_normalpha"
        align_label = "tab:steering_alignment_normalpha"
        grid_first = "fig:steering_normalpha_llama_grid"
        grid_last = "fig:steering_normalpha_qwen_grid"
        coeff_caption = (
            "Both models sweep the dimensionless relative dose "
            "$\\alpha \\in \\{0, 0.125, 0.25, \\dots, 1.0\\}$ under the norm-aware "
            "update $h' = h + \\alpha\\,(v/\\|v\\|)\\,\\|h_\\ell\\|$; "
            "the full grid is kept in the average."
        )
        order_table_intro = ""
        align_table_intro = ""
    else:
        order_label = "tab:steering_order"
        align_label = "tab:steering_alignment"
        grid_first = "fig:steering_llama_grid"
        grid_last = "fig:steering_qwen_l19"
        coeff_caption = (
            "Llama: $\\alpha \\in \\{0,1,2,4,8,12\\}$ kept, $16$ dropped; "
            "Qwen, which uses a wider sweep because of larger residual-stream norms: "
            "$\\alpha \\in \\{0,4,8,16,32,48,64\\}$ kept, $96$ dropped."
        )
        order_table_intro = ""
        align_table_intro = ""

    # Table 1: steering orders per (model, layer, task). Layers as rows so the
    # table is narrow enough to fit a single ACL column without overflowing.
    task_cols = "c" * len(TASKS)
    task_header = " & ".join(t for t, _ in TASKS)
    parts.append("\\begin{table*}[t]")
    parts.append("\\centering")
    parts.append("\\footnotesize")
    parts.append("\\setlength{\\tabcolsep}{4pt}")
    parts.append(f"\\begin{{tabular}}{{ll {task_cols}}}")
    parts.append("\\toprule")
    parts.append(f"Model & Layer & {task_header} \\\\")
    parts.append("\\midrule")
    for model_key in ("llama", "qwen"):
        r = results[model_key]
        layers = r["layers"]
        for i, layer in enumerate(layers):
            row_cells = []
            for task, _ in TASKS:
                order = r["steering_orders"][(task, layer)]
                row_cells.append(" $>$ ".join(order))
            model_cell = MODEL_DISPLAY[model_key].split("-")[0] if i == 0 else ""
            parts.append(f"{model_cell} & L{layer} & " + " & ".join(row_cells) + " \\\\")
        if model_key == "llama":
            parts.append("\\midrule")
    parts.append("\\bottomrule")
    parts.append("\\end{tabular}")
    if mode == "norm_alpha":
        coeff_phrase = (
            "across the steering coefficients in the sweep. " + coeff_caption + " "
        )
        figure_phrase = (
            f"derived from the steering dose-response curves in "
            f"Figures~\\ref{{{grid_first}}}--\\ref{{{grid_last}}}. "
        )
    else:
        coeff_phrase = (
            "across the steering coefficients $\\alpha$ in each model's sweep, after dropping the "
            "single largest grid point (" + coeff_caption + "). "
        )
        figure_phrase = (
            f"derived from Figures~\\ref{{{grid_first}}}--\\ref{{{grid_last}}}. "
        )
    parts.append(
        "\\caption{" + order_table_intro +
        "Selection-method steering order per (task, layer), " + figure_phrase +
        "Each cell ranks the three selection methods (PV $=$ Persona Vector, RD $=$ Residual Diff, "
        "RC $=$ Residual Change) descending by mean trait-eliciting score "
        + coeff_phrase +
        "For Personality and MedHallu we first average across the trait datasets within "
        "the task, then take the mean across coefficients. On MedHallu the three steering "
        "methods all stay near baseline at every layer, so the ordering reduces to noise; "
        "it is included here as a null-control reference.}"
    )
    parts.append(f"\\label{{{order_label}}}")
    parts.append("\\end{table*}")
    parts.append("")

    # Table 2: match percentages, both models.
    parts.append("\\begin{table*}[t]")
    parts.append("\\centering")
    parts.append("\\footnotesize")
    parts.append("\\setlength{\\tabcolsep}{5pt}")
    parts.append("\\begin{tabular}{ll cc cc}")
    parts.append("\\toprule")
    parts.append(" &  & \\multicolumn{2}{c}{Residual Diff} & \\multicolumn{2}{c}{Residual Change Treatment} \\\\")
    parts.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    parts.append("Model & Task & Full & Top-1 & Full & Top-1 \\\\")
    parts.append("\\midrule")
    for model_key in ("llama", "qwen"):
        r = results[model_key]
        first = True
        for task, _ in TASKS:
            rd_full, rd_top1, rd_tot = r["match_stats"][(task, "RD")]
            rct_full, rct_top1, rct_tot = r["match_stats"][(task, "RCT")]
            model_cell = MODEL_DISPLAY[model_key].split("-")[0] if first else ""
            parts.append(
                f"{model_cell} & {SHORT_TASK_LABEL[task]} & "
                f"{100*rd_full/rd_tot:.0f}\\% & {100*rd_top1/rd_tot:.0f}\\% & "
                f"{100*rct_full/rct_tot:.0f}\\% & {100*rct_top1/rct_tot:.0f}\\% \\\\"
            )
            first = False
        if model_key == "llama":
            parts.append("\\midrule")
    parts.append("\\bottomrule")
    parts.append("\\end{tabular}")
    parts.append(
        "\\caption{" + align_table_intro +
        "Fraction of (training set, layer) cells in which the post-finetune "
        "trait-score ranking of the three selection methods matches the steering ranking from "
        f"Table~\\ref{{{order_label}}}. \\emph{{Full}} requires the entire permutation "
        "of \\{PV, RD, RC\\} to agree; \\emph{Top-1} only requires the best-scoring "
        "selection method to agree. Each percentage is over $3 \\times 4 = 12$ "
        "(train, layer) cells. On MedHallu the steering signal is at noise floor "
        "(see Table~\\ref{" + order_label + "}), so the alignment percentages there "
        "are near the chance baseline of $1/6 \\approx 17\\%$ for Full and "
        "$1/3 \\approx 33\\%$ for Top-1; this is a sanity check that steering should "
        "\\emph{not} be used as a selection-method predictor on MedHallu.}"
    )
    parts.append(f"\\label{{{align_label}}}")
    parts.append("\\end{table*}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
    print(f"wrote {out_path}")


FREE_GEN_TASKS = ("Personality", "UltraFB Coding", "UltraFB Factual")


def write_latex_compact(results: dict[str, dict], out_path: Path, mode: str) -> None:
    """Emit a compact main-body table aggregating the three free-generation
    tasks into a single 'Free-gen' row per model, with MedHallu kept separate.
    Counts are summed over the constituent tasks so the percentage is computed
    from the raw cell totals, not by averaging rounded per-task percentages."""
    align_label = (
        "tab:steering_alignment_main_normalpha"
        if mode == "norm_alpha"
        else "tab:steering_alignment_main"
    )
    appendix_label = (
        "tab:steering_alignment_normalpha"
        if mode == "norm_alpha"
        else "tab:steering_alignment"
    )

    parts: list[str] = []
    parts.append("\\begin{table}[t]")
    parts.append("\\centering")
    parts.append("\\footnotesize")
    parts.append("\\setlength{\\tabcolsep}{3pt}")
    parts.append("\\begin{tabular}{l cc cc}")
    parts.append("\\toprule")
    parts.append(" & \\multicolumn{2}{c}{RD} & \\multicolumn{2}{c}{RCT} \\\\")
    parts.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    parts.append("Task & Full & Top-1 & Full & Top-1 \\\\")
    parts.append("\\midrule")
    # Aggregate across both models for each row.
    def _row(label: str, task_list: tuple[str, ...]) -> str:
        rd_full = rd_top1 = rd_tot = 0
        rct_full = rct_top1 = rct_tot = 0
        for model_key in ("llama", "qwen"):
            r = results[model_key]
            for task in task_list:
                f, t, n = r["match_stats"][(task, "RD")]
                rd_full += f
                rd_top1 += t
                rd_tot += n
                f, t, n = r["match_stats"][(task, "RCT")]
                rct_full += f
                rct_top1 += t
                rct_tot += n
        return (
            f"{label} & "
            f"{100*rd_full/rd_tot:.0f}\\% & {100*rd_top1/rd_tot:.0f}\\% & "
            f"{100*rct_full/rct_tot:.0f}\\% & {100*rct_top1/rct_tot:.0f}\\% \\\\"
        )

    parts.append("Chance & 17\\% & 33\\% & 17\\% & 33\\% \\\\")
    parts.append("\\midrule")
    parts.append(_row("Free-gen", FREE_GEN_TASKS))
    parts.append(_row("MedHallu", ("MedHallu",)))
    parts.append("\\bottomrule")
    parts.append("\\end{tabular}")
    parts.append(
        "\\caption{Fraction of (model, training set, layer) cells in which "
        "the post-finetune trait-score ranking of the three selection methods "
        "(PV, RD, RC) matches the steering ranking from \\Cref{fig:steering_main}. "
        "\\emph{Full} requires the entire permutation to agree; \\emph{Top-1} "
        "only requires the best-scoring method to agree. Free-gen aggregates "
        "Personality, UltraFB Coding, and UltraFB Factual across both models "
        "($2 \\times 3 \\times 3 \\times 4 = 72$ cells per percentage); MedHallu "
        "is $2 \\times 3 \\times 4 = 24$ cells. The Chance row shows the "
        "expected agreement under a uniform random ranking ($1/6$ for Full, "
        "$1/3$ for Top-1). On free generation, RCT alignment with steering "
        "is substantially above chance and above RD alignment, motivating "
        "the steering-selected query in the mixed selection strategy. On "
        "MedHallu both methods drop near chance, so steering should not be "
        "used as a selection-method predictor there. The per-model, per-task "
        "breakdown is in \\Cref{" + appendix_label + "}.}"
    )
    parts.append(f"\\label{{{align_label}}}")
    parts.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
    print(f"wrote {out_path}")


def drop_top_coeffs(df: pd.DataFrame, n_drop: int) -> pd.DataFrame:
    """Drop the n_drop highest distinct coefficients from df.

    The plot script drops the single largest coefficient because trait
    score there is dominated by incoherent generations. Qwen uses a wider
    sweep ({0,4,8,16,32,48,64,96}) than Llama ({0,1,2,4,8,12,16}); on
    Qwen coherence collapse completes around alpha=64, so we only drop
    alpha=96 (and on Llama we only drop alpha=16).
    """
    sorted_coeffs = sorted(df["coeff"].unique(), reverse=True)
    to_drop = sorted_coeffs[:n_drop]
    return df[~df["coeff"].isin(to_drop)]


def analyze_model(model_key: str, mode: str = "absolute") -> None:
    layers = MODEL_LAYERS[model_key]
    df = pd.read_csv(STEERING_CSV_BY_MODE[mode][model_key])
    df = drop_top_coeffs(df, N_TOP_COEFFS_TO_DROP_BY_MODE[mode][model_key]).copy()

    banner = f" {MODEL_DISPLAY[model_key]} "
    print()
    print("#" * 72)
    print(banner.center(72, "#"))
    print("#" * 72)

    # 1. Steering orderings.
    print("=" * 72)
    print("Steering order (mean trait score across coeffs; AUC proxy)")
    print("=" * 72)
    print(f"{'Task':<18}{'Layer':>6}   Order (best → worst)")
    steering_orders: dict[tuple[str, int], tuple[str, ...]] = {}
    for task, _ in TASKS:
        for layer in layers:
            order = steering_order(df, task, layer)
            steering_orders[(task, layer)] = order
            print(f"{task:<18}{layer:>6}   {' > '.join(order)}")
        print()

    # 2. Compare table orderings.
    print("=" * 72)
    print("Trait-score ordering match vs. steering order")
    print("Full = full permutation of [+PV, +RD, +RC] matches steering ranking")
    print("Top-1 = best-scoring selection method matches steering's best")
    print("(both over the 12 (train, layer) cells per task per attribution)")
    print("=" * 72)
    print(f"{'Task':<18}{'Attribution':<14}{'Full match':>12}{'Top-1 match':>14}")
    for task, sub_evals in TASKS:
        for attr_label, attr_method, attr_subdir in ATTR_METHODS:
            total = 0
            full_matches = 0
            top1_matches = 0
            for train, layer in product(TRAIN_SETS, layers):
                order = table_order(model_key, layer, train, sub_evals, attr_method, attr_subdir)
                if order is None:
                    continue
                total += 1
                expected = steering_orders[(task, layer)]
                if order == expected:
                    full_matches += 1
                if order[0] == expected[0]:
                    top1_matches += 1
            full_pct = 100 * full_matches / total if total else 0
            top1_pct = 100 * top1_matches / total if total else 0
            print(
                f"{task:<18}{attr_label:<14}"
                f"{full_matches:>3}/{total:<3}={full_pct:5.1f}%"
                f"  {top1_matches:>3}/{total:<3}={top1_pct:5.1f}%"
            )
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["absolute", "norm_alpha"],
        default="absolute",
        help="absolute: original per-model α grids; norm_alpha: dimensionless α∈[0,1].",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output .tex path. Defaults to tables/steering_alignment{,_normalpha}.tex.",
    )
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for model_key in ("llama", "qwen"):
        analyze_model(model_key, mode=args.mode)
        results[model_key] = collect_alignment(model_key, mode=args.mode)
    if args.out is None:
        fname = "steering_alignment_normalpha.tex" if args.mode == "norm_alpha" else "steering_alignment.tex"
        out_path = Path(__file__).resolve().parents[1] / "tables" / fname
    else:
        out_path = Path(args.out)
    write_latex_partial(results, out_path, mode=args.mode)

    compact_fname = (
        "steering_alignment_main_normalpha.tex"
        if args.mode == "norm_alpha"
        else "steering_alignment_main.tex"
    )
    compact_path = out_path.parent / compact_fname
    write_latex_compact(results, compact_path, mode=args.mode)


if __name__ == "__main__":
    main()
