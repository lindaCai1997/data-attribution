"""
Generate the 2x3 free-generation mix-comparison figure for the main body.

Rows = (Llama L19, Qwen L17). Columns = (Personality avg, UltraFB Factual,
UltraFB Coding). Each subplot is a grouped bar chart with 3 train-dataset
groups and 5 method bars per group: No selection / Random / TRAK / RD+PV /
Mix. The Mix bar dispatches per-cell to the steering-selected mix variant via
MIX_BEST in make_main_table.

Run:
    python analysis/paper_results/make_mix_comparison_plot.py

Outputs:
    <repo>/analysis/plots/free_generation_mix_comparison.pdf
    <repo>/analysis/plots/free_generation_mix_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse helpers from the table builder (which itself imports from make_bar_plots).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_main_table import (  # noqa: E402
    MODELS,
    EVALS_FREEGEN,
    TRAIN_DATASETS,
    fetch_score,
)


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plots"

MODEL_LAYERS = [
    ("llama", 19, "Llama-3.1-8B-Instruct, L19"),
    ("qwen",  17, "Qwen-2.5-7B-Instruct, L17"),
]

# (display label, COLUMNS_MAIN-style col tuple, bar color).
METHODS = [
    ("No selection", ("no_sel",   None,                  None,                 None),                       "#bdbdbd"),
    ("Random",       ("random",   "random+none",         None,                 "random"),                   "#969696"),
    ("TRAK",         ("filtered", "trak+none",           "trak",               "trak"),                     "#f4a261"),
    ("Persona Vector", ("filtered", "residual_diff+none",  "persona_vector_gen", "residual_diff"),          "#4a90d9"),
    ("Mix",          ("mix_best", None,                  None,                 None),                       "#B3202C"),
]


def aggregate() -> np.ndarray:
    """scores[mi, ei, ri, ti, bi] for model i, eval-col j, ... — but flat is simpler.

    Returns scores of shape (n_models, n_evals, n_trains, n_methods).
    """
    n_mo, n_ev, n_tr, n_me = (
        len(MODEL_LAYERS), len(EVALS_FREEGEN), len(TRAIN_DATASETS), len(METHODS),
    )
    scores = np.full((n_mo, n_ev, n_tr, n_me), np.nan)
    for mi, (mkey, layer, _) in enumerate(MODEL_LAYERS):
        for ei, (_label, _range, sub_evals) in enumerate(EVALS_FREEGEN):
            for ti, (train, _) in enumerate(TRAIN_DATASETS):
                for bi, (_blabel, col, _color) in enumerate(METHODS):
                    v = fetch_score(mkey, layer, train, sub_evals, col)
                    if v is not None:
                        scores[mi, ei, ti, bi] = v
    return scores


def plot(scores: np.ndarray, out_dir: Path) -> None:
    n_mo, n_ev, n_tr, n_me = scores.shape
    bar_width = 0.15
    group_centers = np.arange(n_tr)
    offsets = (np.arange(n_me) - (n_me - 1) / 2) * bar_width

    fig, axes = plt.subplots(
        nrows=n_mo, ncols=n_ev, figsize=(4.2 * n_ev, 3.0 * n_mo), sharey="row",
    )

    for mi in range(n_mo):
        row_max = float(np.nanmax(scores[mi])) if np.isfinite(scores[mi]).any() else 1.0
        if not np.isfinite(row_max) or row_max <= 0:
            row_max = 1.0

        for ei in range(n_ev):
            ax = axes[mi, ei]
            cell = scores[mi, ei]  # (n_tr, n_me)
            for bi, (blabel, _col, color) in enumerate(METHODS):
                x = group_centers + offsets[bi]
                y = cell[:, bi]
                is_highlight = (blabel == "Mix")
                style = (
                    dict(hatch="//", edgecolor="black", linewidth=1.4, zorder=3)
                    if is_highlight
                    else dict(edgecolor="white", linewidth=0.4)
                )
                bars = ax.bar(
                    x,
                    np.where(np.isnan(y), 0.0, y),
                    width=bar_width,
                    color=color,
                    label=blabel if (mi == 0 and ei == 0) else None,
                    **style,
                )
                for ti, b in enumerate(bars):
                    if np.isnan(y[ti]):
                        b.set_color("#ffffff")
                        b.set_edgecolor("#cccccc")
                        b.set_hatch("//")

            ax.set_xticks(group_centers)
            ax.set_xticklabels([d[1] for d in TRAIN_DATASETS], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylim(bottom=0, top=row_max * 1.05)
            if mi == 0:
                ax.set_title(EVALS_FREEGEN[ei][0], fontsize=16)
            if ei == 0:
                ax.set_ylabel(
                    f"{MODEL_LAYERS[mi][2]}\nTrait-eliciting score",
                    fontsize=11,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=len(METHODS), frameon=False, fontsize=11,
        handlelength=1.4, columnspacing=1.4, handletextpad=0.4,
    )
    fig.tight_layout()
    out_pdf = out_dir / "free_generation_mix_comparison.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def write_csv(scores: np.ndarray, out_dir: Path) -> None:
    csv_path = out_dir / "free_generation_mix_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("model,layer,eval,train,method,score\n")
        for mi, (mkey, layer, _) in enumerate(MODEL_LAYERS):
            for ei, (elabel, _range, _) in enumerate(EVALS_FREEGEN):
                for ti, (_train, tlabel) in enumerate(TRAIN_DATASETS):
                    for bi, (blabel, *_rest) in enumerate(METHODS):
                        v = scores[mi, ei, ti, bi]
                        cell = f"{v:.4f}" if not np.isnan(v) else ""
                        f.write(f"{mkey},{layer},{elabel},{tlabel},{blabel},{cell}\n")
    print(f"wrote {csv_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = aggregate()
    n_filled = int(np.isfinite(scores).sum())
    n_total = scores.size
    print(f"populated {n_filled}/{n_total} cells")
    # Per-method fill report — surface TRAK gaps explicitly.
    for bi, (blabel, *_rest) in enumerate(METHODS):
        sl = scores[..., bi]
        print(f"  {blabel:14s}: {int(np.isfinite(sl).sum())}/{sl.size}")
    plot(scores, OUT_DIR)
    write_csv(scores, OUT_DIR)


if __name__ == "__main__":
    main()
