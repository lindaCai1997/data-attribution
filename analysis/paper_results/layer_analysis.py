"""
Layer / mixing analyses from EXISTING run data (no GPU; paper Appendix I):

1. Fixed-layer vs oracle-best-layer regret, per (model, train, task, pairing)
     ("systematic layer-wise analysis; best-layer vs
      fixed-layer") using the per-layer runs already on disk.
2. Explicit RD-only vs RCT-only vs Mix comparison at the headline layers
   (Llama L19 / Qwen L17).

Run:
    python analysis/paper_results/layer_analysis.py

Outputs -> /scratch/users/spa-data-attribution/data/rebuttal/layer_analysis/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bar_plots import (  # noqa: E402
    MODELS,
    find_run_dir,
    find_layer_independent_run,
    read_metric,
    read_no_selection_metric,
)
from make_main_table import (  # noqa: E402
    EVALS_FREEGEN,
    EVALS_MEDHALLU,
    MIX_BEST,
    _MIX_SUBDIR,
)

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

OUT_DIR = Path(f"{DATA_ROOT}/data/rebuttal/layer_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINS = ["dolly_10k", "ultrachat_200k", "openorca_200k"]
HEADLINE = {"llama": 19, "qwen": 17}

# The full 2x3 framework grid, (label, attr_method, sel_method, attr_subdir).
PAIRINGS = [
    ("RD+PV",  "residual_diff+none",             "persona_vector_gen", "residual_diff"),
    ("RD+RD",  "residual_diff+none",             "residual_diff",      "residual_diff"),
    ("RD+RC",  "residual_diff+none",             "residual_change",    "residual_diff"),
    ("RCT+PV", "residual_change_treatment+none", "persona_vector_gen", "residual_change_treatment"),
    ("RCT+RD", "residual_change_treatment+none", "residual_diff",      "residual_change_treatment"),
    ("RCT+RC", "residual_change_treatment+none", "residual_change",    "residual_change_treatment"),
]
BASELINES = [
    ("TRAK",   "trak+none", "trak", "trak"),
]

TASKS = [(name, subs) for (name, _scale, subs) in EVALS_FREEGEN + EVALS_MEDHALLU]
TASK_KEY = {"Personality": "personality", "UltraFB Coding": "coding",
            "UltraFB Factual": "factual"}


def cell_score(model: str, layer: int, train: str, subs,
               attr: str, sel: str, subdir: str) -> float | None:
    root = MODELS[model]["root_template"].format(L=layer)
    vals = []
    for eval_name, trait in subs:
        run = find_run_dir(root, train, subdir, attr, sel, eval_name)
        v = read_metric(run, trait) if run is not None else None
        if v is not None:
            vals.append(v)
    if len(vals) < len(subs):
        return None  # incomplete cell
    return float(np.mean(vals))


def mix_score(model: str, layer: int, train: str, task_name: str, subs) -> float | None:
    tkey = TASK_KEY.get(task_name)
    if tkey is None:
        return None
    best = MIX_BEST.get((model, layer, tkey), "rc")
    sub = _MIX_SUBDIR[best]
    return cell_score(model, layer, train, subs, f"{sub}+none", sub, sub)


def main() -> None:
    # ---------- 1. per-layer grid ----------
    rows = []
    for model in MODELS:
        for layer in MODELS[model]["layers"]:
            for train in TRAINS:
                for task_name, subs in TASKS:
                    for label, attr, sel, subdir in PAIRINGS + BASELINES:
                        s = cell_score(model, layer, train, subs, attr, sel, subdir)
                        rows.append(dict(model=model, layer=layer, train=train,
                                         task=task_name, pairing=label, score=s))
    with open(OUT_DIR / "layer_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Fixed-vs-oracle-layer regret per (model, train, task, pairing).
    reg_rows = []
    for model in MODELS:
        fixed_l = HEADLINE[model]
        for train in TRAINS:
            for task_name, _subs in TASKS:
                for label, *_ in PAIRINGS + BASELINES:
                    per_layer = {r["layer"]: r["score"] for r in rows
                                 if r["model"] == model and r["train"] == train
                                 and r["task"] == task_name and r["pairing"] == label
                                 and r["score"] is not None}
                    if fixed_l not in per_layer or not per_layer:
                        continue
                    oracle_l, oracle_s = max(per_layer.items(), key=lambda kv: kv[1])
                    fixed_s = per_layer[fixed_l]
                    reg_rows.append(dict(
                        model=model, train=train, task=task_name, pairing=label,
                        n_layers=len(per_layer), fixed_layer=fixed_l,
                        fixed_score=round(fixed_s, 4),
                        oracle_layer=oracle_l, oracle_score=round(oracle_s, 4),
                        regret_abs=round(oracle_s - fixed_s, 4),
                        regret_rel=round((oracle_s - fixed_s) / max(oracle_s, 1e-9), 4),
                        fixed_is_oracle=oracle_l == fixed_l,
                    ))
    with open(OUT_DIR / "fixed_vs_best_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0]))
        w.writeheader()
        w.writerows(reg_rows)

    print("== Fixed-layer vs oracle-layer regret (all pairings x cells) ==")
    for model in MODELS:
        sub = [r for r in reg_rows if r["model"] == model]
        regs = np.array([r["regret_rel"] for r in sub])
        hit = np.mean([r["fixed_is_oracle"] for r in sub])
        print(f"{model}: n={len(sub)} cells | fixed layer is oracle in {hit:.0%} "
              f"| mean rel regret {regs.mean():.1%} | median {np.median(regs):.1%} "
              f"| max {regs.max():.1%}")
        # winning-method-only view (the pairings the paper actually recommends)
        win = [r for r in sub if (r["task"] == "MedHallu" and r["pairing"] == "RCT+RD")
               or (r["task"] != "MedHallu" and r["pairing"] == "RD+RD")]
        wregs = np.array([r["regret_rel"] for r in win])
        whit = np.mean([r["fixed_is_oracle"] for r in win])
        print(f"  recommended pairings only: n={len(win)} | fixed=oracle {whit:.0%} "
              f"| mean rel regret {wregs.mean():.1%} | max {wregs.max():.1%}")

    # ---------- 2. RD vs RCT vs Mix at headline layers ----------
    cmp_rows = []
    for model in MODELS:
        layer = HEADLINE[model]
        for train in TRAINS:
            for task_name, subs in TASKS:
                if task_name == "MedHallu":
                    continue  # mix undefined on MedHallu
                rd_rd = cell_score(model, layer, train, subs,
                                   "residual_diff+none", "residual_diff", "residual_diff")
                rct_scores = {q: cell_score(model, layer, train, subs,
                                            "residual_change_treatment+none", sel, "residual_change_treatment")
                              for q, sel in [("PV", "persona_vector_gen"),
                                             ("RD", "residual_diff"),
                                             ("RC", "residual_change")]}
                rct_vals = {k: v for k, v in rct_scores.items() if v is not None}
                rct_best_q, rct_best = (max(rct_vals.items(), key=lambda kv: kv[1])
                                        if rct_vals else (None, None))
                mx = mix_score(model, layer, train, task_name, subs)
                # baselines
                trak = cell_score(model, layer, train, subs, "trak+none", "trak", "trak")
                pv_base = cell_score(model, layer, train, subs,
                                     "residual_diff+none", "persona_vector_gen", "residual_diff")
                rand_vals, nosel_vals = [], []
                for eval_name, trait in subs:
                    r = find_layer_independent_run(model, train, "random", "random+none", eval_name)
                    v = read_metric(r, trait) if r is not None else None
                    if v is not None:
                        rand_vals.append(v)
                    v2 = read_no_selection_metric(model, eval_name, trait)
                    if v2 is not None:
                        nosel_vals.append(v2)
                rand = float(np.mean(rand_vals)) if len(rand_vals) == len(subs) else None
                nosel = float(np.mean(nosel_vals)) if len(nosel_vals) == len(subs) else None
                base_vals = {"TRAK": trak, "PersonaVec": pv_base, "Random": rand,
                             "NoSel": nosel}
                bdef = {k: v for k, v in base_vals.items() if v is not None}
                best_base_name, best_base = (max(bdef.items(), key=lambda kv: kv[1])
                                             if bdef else (None, None))
                cmp_rows.append(dict(
                    model=model, layer=layer, train=train, task=task_name,
                    rd_rd=rd_rd, rct_pv=rct_scores["PV"], rct_rd=rct_scores["RD"],
                    rct_rc=rct_scores["RC"], rct_best_q=rct_best_q, rct_best=rct_best,
                    mix=mx, trak=trak, persona_vec=pv_base, random=rand, no_sel=nosel,
                    best_baseline=best_base_name, best_baseline_score=best_base,
                ))
    with open(OUT_DIR / "rd_rct_mix_headline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)

    print("\n== RD-only vs RCT-only vs Mix (headline layers) ==")
    fmt = lambda v: "  --" if v is None else f"{v:.2f}"
    print(f"{'model':6}{'train':16}{'task':18}{'RD+RD':>7}{'RCTbest':>9}"
          f"{'Mix':>7}{'bestBase':>9}  mix_vs_max(fixed)")
    for r in cmp_rows:
        cands = [v for v in (r["rd_rd"], r["rct_best"], r["best_baseline_score"])
                 if v is not None]
        rel = (None if (r["mix"] is None or not cands)
               else (r["mix"] - max(cands)) / max(max(cands), 1e-9))
        print(f"{r['model']:6}{r['train']:16}{r['task']:18}{fmt(r['rd_rd']):>7}"
              f"{fmt(r['rct_best']):>9}{fmt(r['mix']):>7}"
              f"{fmt(r['best_baseline_score']):>9}  "
              f"{'--' if rel is None else f'{rel:+.0%}'}")
    print(f"\nWrote {OUT_DIR}/layer_scores.csv, fixed_vs_best_layer.csv, "
          f"rd_rct_mix_headline.csv")


if __name__ == "__main__":
    main()
