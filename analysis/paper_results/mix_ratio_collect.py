"""
Collect the mixing-ratio ablation sweep results (paper Appendix I).

For each of the 18 (model, train, task) headline cells, report the trait-
eliciting score at RD:RCT mixing ratios 0/500, 125/375, 250/250, 375/125,
500/0, where:
  - 0/500   = fixed RCT + steering-selected query q* (existing paper runs)
  - 250/250 = the paper's mix (existing runs, suffix != mixrdk*)
  - 125/375, 375/125 = new runs with suffix mixrdk{125,375}_*
  - 500/0   = fixed RD+RD (existing paper runs)

Run:
    python analysis/paper_results/mix_ratio_collect.py

Output -> /scratch/users/spa-data-attribution/data/rebuttal/layer_analysis/mix_ratio_sweep.csv
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bar_plots import MODELS, find_run_dir, read_metric  # noqa: E402
from make_main_table import (  # noqa: E402
    EVALS_FREEGEN,
    MIX_BEST,
    _MIX_SUBDIR,
)

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

OUT = Path(f"{DATA_ROOT}/data/rebuttal/layer_analysis")
OUT.mkdir(parents=True, exist_ok=True)

TRAINS = ["dolly_10k", "ultrachat_200k", "openorca_200k"]
HEADLINE = {"llama": 19, "qwen": 17}
TASK_KEY = {"Personality": "personality", "UltraFB Coding": "coding",
            "UltraFB Factual": "factual"}
QSTAR_SEL = {"rc": "residual_change", "rd": "residual_diff",
             "pv": "persona_vector_gen"}


def find_ratio_run(root: str, train: str, subdir: str, eval_name: str,
                   rd_k: int) -> Path | None:
    pattern = (f"{root}/{train}/{subdir}/"
               f"{train}-cos_sim-{subdir}+none-{subdir}-500-{eval_name}-mixrdk{rd_k}_*")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def main() -> None:
    rows, missing = [], []
    for model in MODELS:
        layer = HEADLINE[model]
        root = MODELS[model]["root_template"].format(L=layer)
        for train in TRAINS:
            for task_name, _scale, subs in EVALS_FREEGEN:
                tkey = TASK_KEY[task_name]
                best = MIX_BEST.get((model, layer, tkey), "rc")
                sub = _MIX_SUBDIR[best]
                scores = {}
                for rd_k in (0, 125, 250, 375, 500):
                    vals = []
                    for eval_name, trait in subs:
                        if rd_k == 500:
                            run = find_run_dir(root, train, "residual_diff",
                                               "residual_diff+none",
                                               "residual_diff", eval_name)
                        elif rd_k == 0:
                            run = find_run_dir(root, train,
                                               "residual_change_treatment",
                                               "residual_change_treatment+none",
                                               QSTAR_SEL[best], eval_name)
                        elif rd_k == 250:
                            run = find_run_dir(root, train, sub, f"{sub}+none",
                                               sub, eval_name)
                        else:
                            run = find_ratio_run(root, train, sub, eval_name, rd_k)
                        v = read_metric(run, trait) if run is not None else None
                        if v is None:
                            missing.append((model, train, eval_name, rd_k))
                        else:
                            vals.append(v)
                    scores[rd_k] = (float(np.mean(vals))
                                    if len(vals) == len(subs) else None)
                rows.append(dict(model=model, train=train, task=task_name,
                                 qstar=best,
                                 **{f"rd{k}": scores[k] for k in
                                    (0, 125, 250, 375, 500)}))

    with open(OUT / "mix_ratio_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: "   --" if v is None else f"{v:5.2f}"
    print(f"{'model':6}{'train':16}{'task':18}{'q*':>4}"
          f"{'0/500':>7}{'125':>7}{'250':>7}{'375':>7}{'500/0':>7}")
    for r in rows:
        print(f"{r['model']:6}{r['train']:16}{r['task']:18}{r['qstar']:>4}"
              f"{fmt(r['rd0']):>7}{fmt(r['rd125']):>7}{fmt(r['rd250']):>7}"
              f"{fmt(r['rd375']):>7}{fmt(r['rd500']):>7}")
    done = sum(1 for r in rows
               if r["rd125"] is not None and r["rd375"] is not None)
    print(f"\ncells with both new ratios done: {done}/18; "
          f"missing run-metrics: {len(missing)}")
    if missing:
        print("first few missing:", missing[:8])


if __name__ == "__main__":
    main()
