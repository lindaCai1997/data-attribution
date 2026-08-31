"""
Generate layer-faceted grouped bar plots for the EMNLP / arXiv submission.

For each (benchmark, model) we emit one figure with one sub-panel per layer.
Inside each sub-panel: x-axis = train dataset, bars per (attribution × selection)
method pair.

Numbers come from on-disk metrics.jsonl files under
/scratch/users/spa-data-attribution/data/, written by the Stage-2 LoRA sweeps.

Run:
    python analysis/paper_results/make_bar_plots.py

Outputs:
    <repo>/analysis/plots/bar_{model}_{benchmark}.pdf       (8 figures)
    <repo>/analysis/plots/bar_{model}_{benchmark}.csv       (8 sidecars)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]  # analysis/
OUT_DIR = REPO / "plots"

DATA_ROOT = Path(os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")) / "data"
NO_SEL_ROOT = DATA_ROOT / "no_selection_baseline" / "no_selection"

MODELS = {
    "llama": {
        "display": "Llama-3.1-8B-Instruct",
        "layers": [15, 17, 19, 21],
        "root_template": str(DATA_ROOT / "llama_attr_l{L}_cos"),
        "no_sel_short": "meta-llama-3p1-8b-instruct",
    },
    "qwen": {
        "display": "Qwen-2.5-7B-Instruct",
        "layers": [13, 15, 17, 19],
        "root_template": str(DATA_ROOT / "qwen2.5_attr_l{L}_cos"),
        "no_sel_short": "qwen2p5-7b-instruct",
    },
}

TRAIN_DATASETS = [
    ("dolly_10k", "Dolly-10k"),
    ("ultrachat_200k", "UltraChat-200k"),
    ("openorca_200k", "OpenOrca-200k"),
]

# Bar label → (attribution_method, selection_method, attr_base_subdir).
# attr_base_subdir is the directory under {root}/{train}/ that holds runs;
# it equals the prefix of attribution_method before '+'.
METHOD_PAIRS = [
    ("No selection",          None,                                None,                 None),
    ("Random",                "random+none",                       None,                 "random"),
    ("TRAK",                  "trak+none",                         "trak",               "trak"),
    ("Persona Vector",        "residual_diff+none",                "persona_vector_gen", "residual_diff"),
    ("RCT + Residual Diff",   "residual_change_treatment+none",    "residual_diff",      "residual_change_treatment"),
    ("RCT + PV",              "residual_change_treatment+none",    "persona_vector_gen", "residual_change_treatment"),
    ("RCT + Residual Change", "residual_change_treatment+none",    "residual_change",    "residual_change_treatment"),
]

BAR_COLORS = {
    "No selection":          "#bdbdbd",
    "Random":                "#969696",
    "TRAK":                  "#f4a261",
    "Persona Vector":        "#4a90d9",
    "RCT + Residual Diff":   "#a1d99b",
    "RCT + PV":              "#74c476",
    "RCT + Residual Change": "#238b45",
}
RCT_BAR_INDICES = [4, 5, 6]  # indices in METHOD_PAIRS that compete for the ★

# benchmark → list of (eval_data_name, trait_key)
# trait_key is the suffix used in metrics.jsonl: ft_{trait_key}_avg
BENCHMARKS = {
    "medhallu": {
        "title": "MedHallu (knowledge-balanced)",
        "evals": [
            ("medhallu_easy_with_knowledge_balanced",   "medical_consistency_0_2"),
            ("medhallu_medium_with_knowledge_balanced", "medical_consistency_0_2"),
            ("medhallu_hard_with_knowledge_balanced",   "medical_consistency_0_2"),
        ],
    },
    "ultra_factual": {
        "title": "UltraFeedback Factual Truthfulness",
        "evals": [
            ("ultra_factual_truthfulness", "ultra_truthfulness_negative_0_3"),
        ],
    },
    "ultra_coding": {
        "title": "UltraFeedback Coding Instruction Following",
        "evals": [
            ("ultra_coding_instruction_following", "ultra_instruction_following_negative_0_3"),
        ],
    },
    "personality": {
        "title": "Personality Traits (avg over 5 traits)",
        "evals": [
            ("empathy_gpt",     "empathy"),
            ("laziness_gpt",    "laziness"),
            ("modesty_gpt",     "modesty"),
            ("preachiness_gpt", "preachiness"),
            ("sycophancy_gpt",  "sycophancy"),
        ],
    },
}


def read_metric(run_dir: Path, trait_key: str) -> float | None:
    """Read ft_{trait_key}_avg from the highest-epoch row of metrics.jsonl."""
    metrics_path = run_dir / "selected_data" / "eval_llm_judge" / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    best_epoch = -1
    best_val = None
    key = f"ft_{trait_key}_avg"
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if key not in row:
                continue
            epoch = row.get("epoch", 0)
            if epoch >= best_epoch:
                best_epoch = epoch
                best_val = row[key]
    return best_val


def read_no_selection_metric(model_key: str, eval_name: str, trait_key: str) -> float | None:
    """No-selection runs land at NO_SEL_ROOT/{short}/{eval}/{timestamp}/eval_llm_judge/metrics.jsonl."""
    eval_dir = NO_SEL_ROOT / MODELS[model_key]["no_sel_short"] / eval_name
    if not eval_dir.exists():
        return None
    candidates = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not candidates:
        return None
    # Use the most recent timestamp.
    metrics_path = candidates[-1] / "eval_llm_judge" / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    best_epoch = -1
    best_val = None
    key = f"ft_{trait_key}_avg"
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if key not in row:
                continue
            epoch = row.get("epoch", 0)
            if epoch >= best_epoch:
                best_epoch = epoch
                best_val = row[key]
    return best_val


def find_run_dir(root: str, train: str, attr_subdir: str,
                 attr_method: str, sel_method: str, eval_name: str) -> Path | None:
    """Glob for the run dir matching this configuration; return latest if multiple.

    Runs whose suffix starts with 'mixrdk' are the mixing-RATIO ablation
    sweep (non-250/250 splits) and are excluded here so the paper tables keep
    reading the original 250/250 mix runs.
    """
    pattern = (
        f"{root}/{train}/{attr_subdir}/"
        f"{train}-cos_sim-{attr_method}-{sel_method}-500-{eval_name}-*"
    )
    matches = sorted(p for p in glob.glob(pattern)
                     if not p.rsplit("-", 1)[-1].startswith("mixrdk"))
    if not matches:
        return None
    return Path(matches[-1])


def find_layer_independent_run(model_key: str, train: str, attr_subdir: str,
                               attr_method: str, eval_name: str) -> Path | None:
    """For layer-independent runs (e.g. random), search across all of the
    model's per-layer root_dirs and return the first match. Selection method
    is ignored — for `random+none`, sel_method is always `residual_diff` on disk."""
    for layer in MODELS[model_key]["layers"]:
        root = MODELS[model_key]["root_template"].format(L=layer)
        pattern = (
            f"{root}/{train}/{attr_subdir}/"
            f"{train}-cos_sim-{attr_method}-*-500-{eval_name}-*"
        )
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    return None


def aggregate(model_key: str, benchmark: str) -> np.ndarray:
    """
    Return scores shape (n_layers, n_train, n_methods). NaN where missing.
    """
    layers = MODELS[model_key]["layers"]
    n_l, n_t, n_m = len(layers), len(TRAIN_DATASETS), len(METHOD_PAIRS)
    scores = np.full((n_l, n_t, n_m), np.nan)
    evals = BENCHMARKS[benchmark]["evals"]

    for li, layer in enumerate(layers):
        root = MODELS[model_key]["root_template"].format(L=layer)
        for ti, (train, _) in enumerate(TRAIN_DATASETS):
            for mi, (label, attr, sel, attr_subdir) in enumerate(METHOD_PAIRS):
                vals = []
                for eval_name, trait in evals:
                    if label == "No selection":
                        v = read_no_selection_metric(model_key, eval_name, trait)
                    elif label == "Random":
                        # Random sweep is registered to a single layer dir per model
                        # (l19 Llama, l17 Qwen); it's layer-independent so any
                        # finished run is fine.
                        run_dir = find_layer_independent_run(
                            model_key, train, attr_subdir, attr, eval_name,
                        )
                        v = read_metric(run_dir, trait) if run_dir is not None else None
                    else:
                        run_dir = find_run_dir(root, train, attr_subdir, attr, sel, eval_name)
                        v = read_metric(run_dir, trait) if run_dir is not None else None
                    if v is not None:
                        vals.append(v)
                if vals:
                    scores[li, ti, mi] = float(np.mean(vals))
    return scores


def plot_figure(model_key: str, benchmark: str, scores: np.ndarray, out_dir: Path) -> None:
    layers = MODELS[model_key]["layers"]
    n_l, n_t, n_m = scores.shape
    bar_width = 0.13
    group_centers = np.arange(n_t)
    offsets = (np.arange(n_m) - (n_m - 1) / 2) * bar_width

    fig, axes = plt.subplots(
        nrows=1, ncols=n_l, figsize=(4.0 * n_l, 3.2), sharey=True,
    )
    if n_l == 1:
        axes = [axes]

    data_max = float(np.nanmax(scores)) if np.isfinite(scores).any() else 1.0
    if not np.isfinite(data_max) or data_max <= 0:
        data_max = 1.0
    text_offset = data_max * 0.015
    star_offset = data_max * 0.07

    for li, ax in enumerate(axes):
        layer_scores = scores[li]  # (n_t, n_m)

        # Pick best RCT bar per train-data group for the ★.
        best_rct_idx_per_group = []
        for ti in range(n_t):
            rct_vals = [layer_scores[ti, pi] for pi in RCT_BAR_INDICES]
            if all(np.isnan(v) for v in rct_vals):
                best_rct_idx_per_group.append(None)
            else:
                best_rct_idx_per_group.append(RCT_BAR_INDICES[int(np.nanargmax(rct_vals))])

        for mi, (label, *_rest) in enumerate(METHOD_PAIRS):
            x = group_centers + offsets[mi]
            y = layer_scores[:, mi]
            edgecolors = ["black" if best_rct_idx_per_group[ti] == mi else "white"
                          for ti in range(n_t)]
            linewidths = [1.6 if best_rct_idx_per_group[ti] == mi else 0.4
                          for ti in range(n_t)]
            bars = ax.bar(
                x, np.where(np.isnan(y), 0.0, y),
                width=bar_width, color=BAR_COLORS[label],
                label=label if li == 0 else None,
                edgecolor=edgecolors, linewidth=linewidths,
            )
            for ti, b in enumerate(bars):
                if np.isnan(y[ti]):
                    b.set_color("#ffffff")
                    b.set_edgecolor("#cccccc")
                    b.set_hatch("//")
                else:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + text_offset,
                        f"{y[ti]:.2f}",
                        ha="center", va="bottom", fontsize=6,
                    )
                if best_rct_idx_per_group[ti] == mi and not np.isnan(y[ti]):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + star_offset,
                        "★",
                        ha="center", va="bottom", fontsize=10, color="black",
                    )

        ax.set_xticks(group_centers)
        ax.set_xticklabels([d[1] for d in TRAIN_DATASETS], rotation=0, fontsize=8)
        ax.set_title(f"layer {layers[li]}", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=0, top=data_max * 1.20 if data_max > 0 else 1.0)
        if li == 0:
            ax.set_ylabel("Trait-eliciting score")

    title = f"{BENCHMARKS[benchmark]['title']} — {MODELS[model_key]['display']}"
    fig.suptitle(title, fontsize=11, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=len(METHOD_PAIRS), frameon=False, fontsize=8,
        handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
    )
    fig.tight_layout()
    out_pdf = out_dir / f"bar_{model_key}_{benchmark}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    csv_path = out_dir / f"bar_{model_key}_{benchmark}.csv"
    with open(csv_path, "w") as f:
        method_labels = [p[0] for p in METHOD_PAIRS]
        f.write("layer,train_data," + ",".join(method_labels) + "\n")
        for li, layer in enumerate(layers):
            for ti, (_, dlabel) in enumerate(TRAIN_DATASETS):
                row = [
                    f"{scores[li, ti, mi]:.4f}" if not np.isnan(scores[li, ti, mi]) else ""
                    for mi in range(n_m)
                ]
                f.write(f"l{layer},{dlabel}," + ",".join(row) + "\n")
    print(f"  wrote {csv_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_key in MODELS:
        for benchmark in BENCHMARKS:
            print(f"\n[{model_key} × {benchmark}]")
            scores = aggregate(model_key, benchmark)
            n_filled = int(np.isfinite(scores).sum())
            n_total = scores.size
            print(f"  populated {n_filled}/{n_total} cells")
            plot_figure(model_key, benchmark, scores, OUT_DIR)


if __name__ == "__main__":
    main()
