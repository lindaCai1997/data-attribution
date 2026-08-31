"""
Generate the 1x2 MedHallu method-comparison figure for the main body.

Columns: Llama L19 (left), Qwen L17 (right). Each subplot is a grouped bar
chart with 3 train-dataset groups and 6 method bars per group: No selection /
Random / TRAK / RD+PV / Best non-RCT+RD framework pairing / RCT+RD. The
"Best non-RCT+RD" bar is a per-cell argmax over RD+PV, RD+RD, RD+RC, RCT+PV,
RCT+RC — i.e. the strongest framework alternative to RCT+RD.

Run:
    python analysis/paper_results/make_medhallu_comparison_plot.py

Outputs:
    <repo>/analysis/plots/medhallu_method_comparison.pdf
    <repo>/analysis/plots/medhallu_method_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_main_table import (  # noqa: E402
    EVALS_MEDHALLU,
    TRAIN_DATASETS,
    fetch_score,
)


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "plots"

MODEL_LAYERS = [
    ("llama", 19, "Llama-3.1-8B-Instruct, L19"),
    ("qwen",  17, "Qwen-2.5-7B-Instruct, L17"),
]

# Five framework pairings that the "best non-RCT+RD" bar argmaxes over.
# (label, COLUMNS_BASE-style col tuple).
BEST_OTHER_COLS = [
    ("RD+PV",  ("filtered", "residual_diff+none",              "persona_vector_gen", "residual_diff")),
    ("RD+RD",  ("filtered", "residual_diff+none",              "residual_diff",      "residual_diff")),
    ("RD+RC",  ("filtered", "residual_diff+none",              "residual_change",    "residual_diff")),
    ("RCT+PV", ("filtered", "residual_change_treatment+none",  "persona_vector_gen", "residual_change_treatment")),
    ("RCT+RC", ("filtered", "residual_change_treatment+none",  "residual_change",    "residual_change_treatment")),
]

# (display label, "kind", per-row col tuple [or None for derived], bar color).
# kind ∈ {"single", "best_other"}; "single" -> use fetch_score on `col`,
# "best_other" -> argmax over BEST_OTHER_COLS.
METHODS = [
    ("No selection", "single",     ("no_sel",   None,                              None,                 None),                       "#bdbdbd"),
    ("Random",       "single",     ("random",   "random+none",                     None,                 "random"),                   "#969696"),
    ("TRAK",         "single",     ("filtered", "trak+none",                       "trak",               "trak"),                     "#f4a261"),
    ("Persona Vector", "single",   ("filtered", "residual_diff+none",              "persona_vector_gen", "residual_diff"),            "#4a90d9"),
    ("Best other",   "best_other", None,                                                                                                  "#1f4e79"),
    ("RCT+RD",       "single",     ("filtered", "residual_change_treatment+none",  "residual_diff",      "residual_change_treatment"), "#B3202C"),
]

SUB_EVALS = EVALS_MEDHALLU[0][2]  # [(eval_name, trait_key), x3]


def _best_other(mkey: str, layer: int, train: str) -> tuple[float | None, str | None]:
    """Per-cell argmax over BEST_OTHER_COLS. Returns (best_val, argmax_label)."""
    best_v: float | None = None
    best_lbl: str | None = None
    for lbl, col in BEST_OTHER_COLS:
        v = fetch_score(mkey, layer, train, SUB_EVALS, col)
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_lbl = v, lbl
    return best_v, best_lbl


def aggregate() -> tuple[np.ndarray, list[list[list[str | None]]]]:
    """
    Returns:
      scores: shape (n_models, n_trains, n_methods).
      argmax_labels: argmax_labels[mi][ti][bi] = label of the BEST_OTHER cell
                     argmax (or None for non-best_other bars).
    """
    n_mo, n_tr, n_me = len(MODEL_LAYERS), len(TRAIN_DATASETS), len(METHODS)
    scores = np.full((n_mo, n_tr, n_me), np.nan)
    argmax_labels: list[list[list[str | None]]] = [
        [[None] * n_me for _ in range(n_tr)] for _ in range(n_mo)
    ]
    for mi, (mkey, layer, _) in enumerate(MODEL_LAYERS):
        for ti, (train, _) in enumerate(TRAIN_DATASETS):
            for bi, (_blabel, kind, col, _color) in enumerate(METHODS):
                if kind == "best_other":
                    v, argmax_lbl = _best_other(mkey, layer, train)
                    argmax_labels[mi][ti][bi] = argmax_lbl
                else:  # "single"
                    v = fetch_score(mkey, layer, train, SUB_EVALS, col)
                if v is not None:
                    scores[mi, ti, bi] = v
    return scores, argmax_labels


def plot(scores: np.ndarray, out_dir: Path) -> None:
    n_mo, n_tr, n_me = scores.shape
    bar_width = 0.13
    group_centers = np.arange(n_tr)
    offsets = (np.arange(n_me) - (n_me - 1) / 2) * bar_width

    # Per-subplot size matches the free-generation mix plot (4.2 × 3.0 in),
    # so when this figure is included at 0.67·\linewidth in LaTeX each subplot
    # ends up the same width on the page as a Fig 2 subplot.
    fig, axes = plt.subplots(
        nrows=1, ncols=n_mo, figsize=(4.2 * n_mo, 3.0), sharey=True,
    )

    row_max = float(np.nanmax(scores)) if np.isfinite(scores).any() else 1.0
    if not np.isfinite(row_max) or row_max <= 0:
        row_max = 1.0

    for mi in range(n_mo):
        ax = axes[mi]
        cell = scores[mi]  # (n_tr, n_me)
        for bi, (blabel, _kind, _col, color) in enumerate(METHODS):
            x = group_centers + offsets[bi]
            y = cell[:, bi]
            is_highlight = (blabel == "RCT+RD")
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
                label=blabel if mi == 0 else None,
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
        ax.set_title(MODEL_LAYERS[mi][2], fontsize=14)
        if mi == 0:
            ax.set_ylabel(
                "MedHallu score\n(0–2, higher = more hallucination)",
                fontsize=11,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.05),
        ncol=len(METHODS), frameon=False, fontsize=11,
        handlelength=1.4, columnspacing=1.4, handletextpad=0.4,
    )
    fig.tight_layout()
    out_pdf = out_dir / "medhallu_method_comparison.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def write_csv(scores: np.ndarray, argmax_labels, out_dir: Path) -> None:
    csv_path = out_dir / "medhallu_method_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("model,layer,train,method,score,argmax_method\n")
        for mi, (mkey, layer, _) in enumerate(MODEL_LAYERS):
            for ti, (_train, tlabel) in enumerate(TRAIN_DATASETS):
                for bi, (blabel, *_rest) in enumerate(METHODS):
                    v = scores[mi, ti, bi]
                    cell = f"{v:.4f}" if not np.isnan(v) else ""
                    argmax = argmax_labels[mi][ti][bi] or ""
                    f.write(f"{mkey},{layer},{tlabel},{blabel},{cell},{argmax}\n")
    print(f"wrote {csv_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, argmax_labels = aggregate()
    n_filled = int(np.isfinite(scores).sum())
    n_total = scores.size
    print(f"populated {n_filled}/{n_total} cells")
    for bi, (blabel, *_rest) in enumerate(METHODS):
        sl = scores[..., bi]
        print(f"  {blabel:14s}: {int(np.isfinite(sl).sum())}/{sl.size}")
    plot(scores, OUT_DIR)
    write_csv(scores, argmax_labels, OUT_DIR)


if __name__ == "__main__":
    main()
