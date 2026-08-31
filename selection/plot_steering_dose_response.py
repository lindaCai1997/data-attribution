"""Generate steering dose-response plots for the EMNLP submission.

Reads the two pre-computed steering layer sweeps (Llama + Qwen) and renders
1x4 strips, 4x4 grids, and a 2x4 main-paper figure with x = alpha / ||h||.

Outputs land in Data_Attribution/ICML_workshop/plots/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

# ---------- knobs ----------

COHERENCE_THRESHOLD = 1.5  # markers go hollow above this; retune freely
BAND_ALPHA = 0.15

# Project root holding the paper repo (Data_Attribution/) and persona_vector/.
REPO_ROOT = Path(
    os.environ.get("SPA_PROJECT_ROOT", "/accounts/projects/jsteinhardt/spa-data-attribution")
)
# Default writes into the main EMNLP / ICML paper's plots dir. Pass
# --out-dir Data_Attribution/ICML_workshop/plots for the workshop submission.
PAPER_PLOTS_DIR = REPO_ROOT / "Data_Attribution" / "plots"
NORM_DIR = REPO_ROOT / "persona_vector" / "activation_norms"

MODEL_CFG = {
    "llama": {
        "csv": Path(f"{DATA_ROOT}/data/steering_summary_20260516_164307/combined_long.csv"),
        "csv_norm_alpha": Path(f"{DATA_ROOT}/data/steering_norm_summary_20260521_183328/combined_long.csv"),
        "norm_file": NORM_DIR / "Meta-Llama-3.1-8B-Instruct" / "sycophancy_gpt_pos.pt",
        "layers": [15, 17, 19, 21],
        "display": "Llama-3.1-8B-Instruct",
    },
    "qwen": {
        "csv": Path(f"{DATA_ROOT}/data/steering_qwen_summary_20260518_163045_exp_coefs/combined_long.csv"),
        "csv_norm_alpha": Path(f"{DATA_ROOT}/data/steering_norm_qwen_summary_20260521_183328/combined_long.csv"),
        "norm_file": NORM_DIR / "Qwen2.5-7B-Instruct" / "sycophancy_gpt_pos.pt",
        "layers": [13, 15, 17, 19],
        "display": "Qwen-2.5-7B-Instruct",
    },
}

# When --mode=norm_alpha is set, the coefficient swept by the experiment is
# already the relative dose α (h' = h + α·v/||v||·||h_l||), so the x-axis is
# just the coeff and we do not divide by ||h_l||. Output filenames get the
# --out-suffix appended (default "_normalpha") so original PDFs aren't clobbered.

# Method order = legend order = stacking order (later draws on top).
# residual_change_treatment is intentionally excluded — its light-green clashes
# visually with residual_change's dark green and it isn't the selection method
# used in the main pipeline. To bring it back, add the tuple
#   ("residual_change_treatment", "Residual Change Treatment", "#a1d99b")
# to the list below.
METHODS = [
    ("persona_vector", "Persona Vector", "#4a90d9"),
    ("residual_diff", "Residual Diff", "#f4a261"),
    ("residual_change", "Residual Change", "#238b45"),
]
METHOD_COLOR = {m: c for m, _, c in METHODS}
METHOD_LABEL = {m: lbl for m, lbl, _ in METHODS}

# Dataset families. The dose-response panels are 1 per family. Column order
# left-to-right: personality, UltraFB Coding, UltraFB Factual, MedHallu (last
# because the three steering methods barely move the model on MedHallu so it
# reads as a null-control panel).
FAMILIES = [
    {
        "key": "personality",
        "title": "Personality (5 traits)",
        "max_score": 3,
        "csv_family": "personality",
        "datasets": [
            "empathy_gpt",
            "laziness_gpt",
            "modesty_gpt",
            "preachiness_gpt",
            "sycophancy_gpt",
        ],
    },
    {
        "key": "ultra_instruction_following",
        "title": "UltraFeedback Coding IF",
        "max_score": 3,
        "csv_family": "ultra",
        "datasets": ["ultra_coding_instruction_following"],
    },
    {
        "key": "ultra_factual",
        "title": "UltraFeedback Factual",
        "max_score": 3,
        "csv_family": "ultra",
        "datasets": ["ultra_factual_truthfulness"],
    },
    {
        "key": "medhallu",
        "title": "MedHallu",
        "max_score": 2,
        "csv_family": "medhallu",
        "datasets": [
            "medhallu_easy_with_knowledge_balanced",
            "medhallu_medium_with_knowledge_balanced",
            "medhallu_hard_with_knowledge_balanced",
        ],
    },
]

# ---------- data plumbing ----------


def load_norms(mode: str = "absolute") -> dict[str, np.ndarray]:
    """Return {model_key: np.ndarray of per-layer ||h||_response}.

    For mode='norm_alpha' the steering coefficient is already the relative
    dose; return all-ones so plotting divides by 1 (x-axis == coeff).
    """
    out = {}
    for model_key, cfg in MODEL_CFG.items():
        if mode == "norm_alpha":
            n_layers = max(cfg["layers"]) + 2
            out[model_key] = np.ones(n_layers, dtype=np.float32)
        else:
            blob = torch.load(cfg["norm_file"], weights_only=False)
            out[model_key] = blob["layer_norms_response"].cpu().numpy()
    return out


def load_csv(model_key: str, mode: str = "absolute") -> pd.DataFrame:
    df = pd.read_csv(MODEL_CFG[model_key]["csv"])
    # prop_score_3 may be empty (NaN) in some rows where the judge only emits {0,1,2};
    # treat missing as 0.
    for col in ["prop_score_0", "prop_score_1", "prop_score_2", "prop_score_3"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    if mode == "absolute":
        # Drop the largest sweep coef across the board — at that strength coherence has
        # fully collapsed on both Llama (coef=16) and Qwen (coef=96), so the point is
        # noise. The norm-aware sweep tops out at α=1.0 (relative dose = 100%), which
        # we intentionally keep so the rollover is visible.
        max_coef = df["coeff"].max()
        df = df[df["coeff"] < max_coef].copy()
    return df


def cell_se(row: pd.Series) -> float:
    """Standard error of the ordinal-score mean from prop_score_{0..3}, n_scored."""
    ps = np.array([row[f"prop_score_{k}"] for k in range(4)])
    ks = np.arange(4)
    mean = (ks * ps).sum()
    var = (ks**2 * ps).sum() - mean**2
    n = max(int(row["n_scored"]), 1)
    return float(np.sqrt(max(var, 0.0) / n))


def aggregate_family(
    df: pd.DataFrame, family: dict, layer: int
) -> dict[str, dict]:
    """For one (layer, family), return {method: {alphas, score, score_lo, score_hi, coh}}.

    For multi-dataset families the band is +/- 1 std across datasets.
    For single-dataset families the band is +/- 1 SE from the multinomial.
    """
    sub = df[
        (df["family"] == family["csv_family"])
        & (df["layer"] == layer)
        & (df["dataset"].isin(family["datasets"]))
    ].copy()

    out = {}
    for method_key, _, _ in METHODS:
        m_sub = sub[sub["method"] == method_key].copy()
        if m_sub.empty:
            continue

        if len(family["datasets"]) == 1:
            m_sub["se"] = m_sub.apply(cell_se, axis=1)
            m_sub = m_sub.sort_values("coeff")
            alphas = m_sub["coeff"].to_numpy()
            score = m_sub["mean_score"].to_numpy()
            band_half = m_sub["se"].to_numpy()
            coh = m_sub["mean_coherence"].to_numpy()
        else:
            # mean / std across datasets at each coef
            grp = m_sub.groupby("coeff")
            agg = grp.agg(
                score=("mean_score", "mean"),
                score_std=("mean_score", "std"),
                coh=("mean_coherence", "mean"),
            ).reset_index().sort_values("coeff")
            alphas = agg["coeff"].to_numpy()
            score = agg["score"].to_numpy()
            band_half = agg["score_std"].fillna(0.0).to_numpy()
            coh = agg["coh"].to_numpy()

        out[method_key] = {
            "alpha": alphas,
            "score": score,
            "lo": score - band_half,
            "hi": score + band_half,
            "coh": coh,
        }
    return out


# ---------- drawing ----------


def style_axes(ax, family: dict, mode: str = "absolute"):
    if mode == "norm_alpha":
        ax.set_xlabel(r"$\alpha$ (rel. to $\|h_\ell\|$)", fontsize=11)
    else:
        ax.set_xlabel(r"$\alpha\;/\;\|h_\ell\|$", fontsize=11)
    # Put the score range in the title so it stays visible even when we drop
    # the ylabel on inner panels.
    ax.set_ylabel("Trait score", fontsize=11)
    ax.set_title(f"{family['title']}  (0–{family['max_score']:g})", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(0, family["max_score"] * 1.02)


def plot_panel(ax, agg: dict, family: dict, h_norm: float, methods=METHODS, xlim=None, mode: str = "absolute"):
    """Draw one (layer, family) panel onto an existing matplotlib Axes."""
    style_axes(ax, family, mode=mode)

    # Optional baseline marker: mean of coef=0 mean_scores across plotted methods.
    baselines = []
    plotted_keys = {m[0] for m in methods}
    for k, v in agg.items():
        if k in plotted_keys and v["alpha"][0] == 0:
            baselines.append(v["score"][0])
    if baselines:
        ax.axhline(
            float(np.mean(baselines)),
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            zorder=0.5,
        )

    x_max_seen = 0.0
    for method_key, _, color in methods:
        if method_key not in agg:
            continue
        d = agg[method_key]
        x = d["alpha"] / h_norm
        x_max_seen = max(x_max_seen, float(x.max()))
        ok = d["coh"] <= COHERENCE_THRESHOLD

        # line: full alpha for the "safe" prefix, faded alpha for the tail.
        # We draw two line segments so that the change is sharp at the threshold.
        # Find the first index where coh > threshold; that's where the fade starts.
        # Fall back to a single solid line if all points are safe.
        if ok.all():
            ax.plot(x, d["score"], color=color, linewidth=1.4, zorder=3)
        else:
            first_bad = int(np.argmax(~ok))
            if first_bad > 0:
                ax.plot(
                    x[: first_bad + 1],
                    d["score"][: first_bad + 1],
                    color=color,
                    linewidth=1.4,
                    zorder=3,
                )
            # faded tail (includes the boundary point so the line is connected)
            tail_lo = max(first_bad - 1, 0)
            ax.plot(
                x[tail_lo:],
                d["score"][tail_lo:],
                color=color,
                linewidth=1.0,
                alpha=0.4,
                zorder=2.5,
            )

        # markers: solid for safe, hollow for above threshold
        if ok.any():
            ax.scatter(
                x[ok],
                d["score"][ok],
                s=14,
                color=color,
                edgecolors=color,
                linewidths=0.8,
                zorder=4,
            )
        if (~ok).any():
            ax.scatter(
                x[~ok],
                d["score"][~ok],
                s=14,
                facecolors="white",
                edgecolors=color,
                linewidths=0.9,
                zorder=4,
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        # Keep x-axis tight to the data so the strip stays compact.
        ax.set_xlim(left=-0.02 * x_max_seen, right=x_max_seen * 1.05)


def add_shared_legend(fig, methods=METHODS, ncol=None, y=-0.02):
    if ncol is None:
        ncol = len(methods)
    handles = [
        plt.Line2D(
            [], [], color=c, marker="o", markersize=4, linewidth=1.4, label=lbl
        )
        for _, lbl, c in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.4,
    )


# ---------- figure builders ----------


def render_strip(
    df: pd.DataFrame,
    model_key: str,
    layer: int,
    h_norm: float,
    out_path: Path,
    mode: str = "absolute",
):
    """1x4 strip: one panel per family, for one (model, layer)."""
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.4))
    for ax, family in zip(axes, FAMILIES):
        agg = aggregate_family(df, family, layer)
        plot_panel(ax, agg, family, h_norm, mode=mode)
    fig.suptitle(
        f"{MODEL_CFG[model_key]['display']} — layer {layer}",
        fontsize=10,
        y=1.02,
    )
    add_shared_legend(fig, ncol=4, y=-0.05)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _per_column_xlim(
    dfs_by_row: list[tuple[pd.DataFrame, float, int]],
) -> list[tuple[float, float]]:
    """For each family (column), find the max α/‖h‖ across all (df, layer, h_norm)
    rows so we can give the whole column a shared x range that fits every row's
    data (not just the row with the smallest ‖h‖).
    """
    xlims = []
    for family in FAMILIES:
        x_max = 0.0
        for df, h_norm, layer in dfs_by_row:
            agg = aggregate_family(df, family, layer)
            for d in agg.values():
                if len(d["alpha"]) > 0:
                    x_max = max(x_max, float(d["alpha"].max() / h_norm))
        xlims.append((-0.02 * x_max, x_max * 1.05))
    return xlims


def render_grid(
    df: pd.DataFrame,
    model_key: str,
    norms: np.ndarray,
    out_path: Path,
    mode: str = "absolute",
):
    """4x4 grid: rows = layers, cols = families."""
    layers = MODEL_CFG[model_key]["layers"]
    fig, axes = plt.subplots(4, 4, figsize=(11.0, 9.0), sharex="col")
    col_xlims = _per_column_xlim([(df, float(norms[L]), L) for L in layers])
    for r, layer in enumerate(layers):
        h_norm = float(norms[layer])
        for c, family in enumerate(FAMILIES):
            ax = axes[r, c]
            agg = aggregate_family(df, family, layer)
            plot_panel(ax, agg, family, h_norm, xlim=col_xlims[c], mode=mode)
            # only the top row shows the family title; only the left col shows ylabel
            if r != 0:
                ax.set_title("")
            if c != 0:
                ax.set_ylabel("")
            if r != len(layers) - 1:
                ax.set_xlabel("")
            # row label on the leftmost panel
            if c == 0:
                ax.text(
                    -0.32,
                    0.5,
                    f"L{layer}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    fig.suptitle(
        f"{MODEL_CFG[model_key]['display']} — dose response across layers",
        fontsize=11,
        y=1.0,
    )
    add_shared_legend(fig, ncol=4, y=-0.005)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.99))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_main_2x4(
    dfs: dict[str, pd.DataFrame],
    norms: dict[str, np.ndarray],
    layers: dict[str, int],
    out_path: Path,
    methods=None,
    mode: str = "absolute",
):
    """2x4 main paper figure: top = Llama at layers['llama'], bottom = Qwen at layers['qwen']."""
    if methods is None:
        methods = METHODS
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2), sharex="col")
    rows = [(dfs[k], float(norms[k][layers[k]]), layers[k]) for k in ["llama", "qwen"]]
    col_xlims = _per_column_xlim(rows)
    for r, model_key in enumerate(["llama", "qwen"]):
        layer = layers[model_key]
        h_norm = float(norms[model_key][layer])
        df = dfs[model_key]
        for c, family in enumerate(FAMILIES):
            ax = axes[r, c]
            agg = aggregate_family(df, family, layer)
            plot_panel(ax, agg, family, h_norm, methods=methods, xlim=col_xlims[c], mode=mode)
            if r != 0:
                ax.set_title("")
            if r != 1:
                ax.set_xlabel("")
            if c != 0:
                ax.set_ylabel("")
            if c == 0:
                ax.text(
                    -0.32,
                    0.5,
                    f"{MODEL_CFG[model_key]['display'].split('-')[0]}\nL{layer}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    add_shared_legend(fig, methods=methods, y=-0.01)
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.99))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- entry point ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PAPER_PLOTS_DIR)
    # Depth-matched cross-model pair: Llama L19/32 (59%) ↔ Qwen L17/28 (61%).
    parser.add_argument("--llama-main-layer", type=int, default=19)
    parser.add_argument("--qwen-main-layer", type=int, default=17)
    parser.add_argument(
        "--skip", choices=["strip", "grid", "main"], action="append", default=[]
    )
    parser.add_argument(
        "--mode",
        choices=["absolute", "norm_alpha"],
        default="absolute",
        help=(
            "absolute: original steering sweep, x-axis is α/||h_l|| computed post-hoc. "
            "norm_alpha: the new sweep already ran with --alpha-relative so the coeff "
            "*is* α and is the relative dose; do not divide."
        ),
    )
    parser.add_argument(
        "--out-suffix",
        default=None,
        help="Filename suffix before .pdf. Defaults to '_normalpha' in --mode=norm_alpha, '' otherwise.",
    )
    parser.add_argument(
        "--llama-csv",
        type=Path,
        default=None,
        help="Override the Llama combined_long.csv path (defaults to the absolute-sweep CSV in MODEL_CFG).",
    )
    parser.add_argument(
        "--qwen-csv",
        type=Path,
        default=None,
        help="Override the Qwen combined_long.csv path.",
    )
    args = parser.parse_args()

    # In --mode=norm_alpha, swap to the relative-dose sweep CSVs by default so
    # the x-axis is 0–1 in units of α/‖h_ℓ‖. Explicit --llama-csv / --qwen-csv
    # overrides still win.
    if args.mode == "norm_alpha":
        MODEL_CFG["llama"]["csv"] = MODEL_CFG["llama"]["csv_norm_alpha"]
        MODEL_CFG["qwen"]["csv"]  = MODEL_CFG["qwen"]["csv_norm_alpha"]
    if args.llama_csv is not None:
        MODEL_CFG["llama"]["csv"] = args.llama_csv
    if args.qwen_csv is not None:
        MODEL_CFG["qwen"]["csv"] = args.qwen_csv

    suffix = args.out_suffix
    if suffix is None:
        suffix = "_normalpha" if args.mode == "norm_alpha" else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    norms = load_norms(mode=args.mode)
    dfs = {k: load_csv(k, mode=args.mode) for k in MODEL_CFG}

    if "strip" not in args.skip:
        for model_key, cfg in MODEL_CFG.items():
            for layer in cfg["layers"]:
                render_strip(
                    dfs[model_key],
                    model_key,
                    layer,
                    float(norms[model_key][layer]),
                    args.out_dir / f"steering_dose_response_{model_key}_l{layer}{suffix}.pdf",
                    mode=args.mode,
                )

    if "grid" not in args.skip:
        for model_key in MODEL_CFG:
            render_grid(
                dfs[model_key],
                model_key,
                norms[model_key],
                args.out_dir / f"steering_dose_response_{model_key}_grid{suffix}.pdf",
                mode=args.mode,
            )

    if "main" not in args.skip:
        render_main_2x4(
            dfs,
            norms,
            {"llama": args.llama_main_layer, "qwen": args.qwen_main_layer},
            args.out_dir / f"steering_dose_response_main_2x4{suffix}.pdf",
            mode=args.mode,
        )
        # Deeper-layer variant: Llama L21 + Qwen L19.
        render_main_2x4(
            dfs,
            norms,
            {"llama": 21, "qwen": 19},
            args.out_dir / f"steering_dose_response_main_2x4_deep{suffix}.pdf",
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
