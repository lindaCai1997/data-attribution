# analysis/cross_model_transfer/collect_addition_results.py
"""
Aggregate the controlled addition test (R3-Major-1, + armC extension)
into results.csv + RESULTS.md.

Per (model, corpus, eval, seed):
  armA_score : fine-tune on 1000 random corpus examples
  armC_score : fine-tune on ONLY the 500 randoms shared by armA and armB (= armA[:500])
  armB_score : fine-tune on 500 RCT+RD-selected + those same 500 randoms
  delta_BA   : armB - armA  (causal effect of swapping 500 random -> 500 selected)
  delta_BC   : armB - armC  (effect of ADDING the 500 selected on top of the shared randoms)

results.csv columns:
  model, corpus, eval, seed, armA_score, armC_score, armB_score, delta_BA, delta_BC

RESULTS.md headline: per (model, corpus, seed), 3-difficulty averages for
A / C / B plus deltas, alongside read-only references from EXISTING runs:
  sel-500 alone  : paper's RCT+RD run (500 selected, no addition)
  rand-500 alone : paper's random baseline (500 random, no addition)
"""
import csv
import glob
import json
import os
import re
import sys
from statistics import mean

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

KEY = "ft_medical_consistency_0_2_avg"
OUT_ROOT = f"{DATA_ROOT}/data/rebuttal/addition_test"

# Layer root used for the RCT+RD (residual_change_treatment) selected runs.
SEL_ROOTS = {
    "llama": f"{DATA_ROOT}/data/llama_attr_l19_cos",
    "qwen": f"{DATA_ROOT}/data/qwen2.5_attr_l17_cos",
}
# random selection is layer-independent -> may live under any layer root; search all.
RAND_ROOTS = {
    "llama": [f"{DATA_ROOT}/data/llama_attr_l{l}_cos" for l in (15, 17, 19, 21)],
    "qwen": [f"{DATA_ROOT}/data/qwen2.5_attr_l{l}_cos" for l in (13, 15, 17, 19)],
}
MODELS = ["llama", "qwen"]
CORPORA = ["ultrachat_200k", "openorca_200k"]
EVALS = [f"medhallu_{d}_with_knowledge_balanced" for d in ["easy", "medium", "hard"]]
SEEDS = [42, 43]
ARMS = ["armA", "armC", "armB"]


def highest_epoch_metric(path):
    """Return KEY from the highest-epoch row of a metrics.jsonl."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    best = max(rows, key=lambda r: r.get("epoch", 0))
    return best[KEY]


def _newest_run_dir(pattern):
    """Newest dir matching pattern whose trailing suffix is not mixrdk*/seed*."""
    dirs = [d for d in glob.glob(pattern)
            if not re.search(r"-(mixrdk|seed)[^/]*/?$", d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def sel500_metric(model, corpus, ev):
    root = SEL_ROOTS[model]
    m = "residual_change_treatment"
    d = _newest_run_dir(f"{root}/{corpus}/{m}/{corpus}-cos_sim-{m}+none-residual_diff-500-{ev}-*")
    if d is None:
        return None
    return highest_epoch_metric(os.path.join(d, "selected_data/eval_llm_judge/metrics.jsonl"))


def rand500_metric(model, corpus, ev):
    # random is layer-independent: pick the newest matching run across all layer roots.
    cands = []
    for root in RAND_ROOTS[model]:
        d = _newest_run_dir(f"{root}/{corpus}/random/{corpus}-cos_sim-random+none-residual_diff-500-{ev}-*")
        if d is not None:
            cands.append(d)
    if not cands:
        return None
    d = max(cands, key=os.path.getmtime)
    return highest_epoch_metric(os.path.join(d, "selected_data/eval_llm_judge/metrics.jsonl"))


def main():
    rows, missing = [], []
    for model in MODELS:
        for corpus in CORPORA:
            for ev in EVALS:
                for S in SEEDS:
                    scores = {}
                    for arm in ARMS:
                        f = os.path.join(OUT_ROOT, model, corpus, ev, f"seed{S}", arm,
                                         "selected_data/eval_llm_judge/metrics.jsonl")
                        if not (os.path.exists(f) and os.path.getsize(f) > 0):
                            missing.append(f)
                            continue
                        scores[arm] = highest_epoch_metric(f)
                    if len(scores) < len(ARMS):
                        continue
                    rows.append({
                        "model": model, "corpus": corpus, "eval": ev, "seed": S,
                        "armA_score": round(scores["armA"], 4),
                        "armC_score": round(scores["armC"], 4),
                        "armB_score": round(scores["armB"], 4),
                        "delta_BA": round(scores["armB"] - scores["armA"], 4),
                        "delta_BC": round(scores["armB"] - scores["armC"], 4),
                    })
    if missing:
        print(f"INCOMPLETE - {len(missing)} missing metrics:")
        for m in missing:
            print("  ", m)
        sys.exit(1)

    # --- results.csv ----------------------------------------------------------
    csv_path = os.path.join(OUT_ROOT, "results.csv")
    cols = ["model", "corpus", "eval", "seed",
            "armA_score", "armC_score", "armB_score", "delta_BA", "delta_BC"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {csv_path}")

    # --- reference scores (per model, corpus, eval) ---------------------------
    sel_ref, rand_ref = {}, {}
    for model in MODELS:
        for corpus in CORPORA:
            for ev in EVALS:
                sel_ref[(model, corpus, ev)] = sel500_metric(model, corpus, ev)
                rand_ref[(model, corpus, ev)] = rand500_metric(model, corpus, ev)

    def ref_avg(table, model, corpus):
        vals = [table[(model, corpus, ev)] for ev in EVALS
                if table[(model, corpus, ev)] is not None]
        return mean(vals) if vals else None

    def cell_avg(model, corpus, col, seed):
        """3-difficulty average of a per-cell column at a fixed seed."""
        vals = [r[col] for r in rows if r["model"] == model and r["corpus"] == corpus
                and r["seed"] == seed]
        return mean(vals) if vals else float("nan")

    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    # --- RESULTS.md -----------------------------------------------------------
    md = []
    md.append("# Controlled addition test\n")
    md.append("Design: for each (model, corpus, MedHallu difficulty, seed):\n"
              "- **arm A** fine-tunes on 1000 random corpus examples;\n"
              "- **arm B** replaces the second 500 of those randoms with the 500 ATLAS-selected "
              "(RCT attribution + RD selection) examples for that cell -- a controlled swap;\n"
              "- **arm C** fine-tunes on ONLY the 500 random examples shared by arms A and B "
              "(verified per cell: the A-B intersection is exactly 500 rows == armA[:500]).\n\n"
              "Everything else is held fixed (LoRA, 1 epoch, batch 2, judge = "
              "`ft_medical_consistency_0_2_avg`, 3 judge repeats). Two independent random draws "
              "(seeds 42, 43). `delta B-A` is the causal effect of swapping 500 random for 500 "
              "selected at fixed dataset size; `delta B-C` is the effect of adding the 500 "
              "selected on top of the shared randoms. `sel-500 alone` / `rand-500 alone` are "
              "existing on-disk runs (500 selected / 500 random, no addition), "
              "difficulty-averaged, read-only references.\n")

    md.append("## Headline (per model, corpus, seed; averaged over the 3 difficulties)\n")
    md.append("| model | corpus | seed | armA (1000 rand) | armC (500 rand) | armB (500 sel + 500 rand) | delta B-A | delta B-C | sel-500 alone | rand-500 alone |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for corpus in CORPORA:
            for S in SEEDS:
                a = cell_avg(model, corpus, "armA_score", S)
                c = cell_avg(model, corpus, "armC_score", S)
                b = cell_avg(model, corpus, "armB_score", S)
                md.append(f"| {model} | {corpus} | {S} "
                          f"| {a:.3f} | {c:.3f} | {b:.3f} "
                          f"| {b-a:+.3f} | {b-c:+.3f} "
                          f"| {fmt(ref_avg(sel_ref, model, corpus))} "
                          f"| {fmt(ref_avg(rand_ref, model, corpus))} |")

    md.append("\n## Seed-averaged summary (+/- = half the seed spread)\n")
    md.append("| model | corpus | armA | armC | armB | delta B-A | delta B-C | sel-500 alone | rand-500 alone |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for corpus in CORPORA:
            a = [cell_avg(model, corpus, "armA_score", s) for s in SEEDS]
            c = [cell_avg(model, corpus, "armC_score", s) for s in SEEDS]
            b = [cell_avg(model, corpus, "armB_score", s) for s in SEEDS]
            dba = [b[i] - a[i] for i in range(2)]
            dbc = [b[i] - c[i] for i in range(2)]
            md.append(
                f"| {model} | {corpus} "
                f"| {mean(a):.3f} +/- {abs(a[0]-a[1])/2:.3f} "
                f"| {mean(c):.3f} +/- {abs(c[0]-c[1])/2:.3f} "
                f"| {mean(b):.3f} +/- {abs(b[0]-b[1])/2:.3f} "
                f"| {mean(dba):+.3f} +/- {abs(dba[0]-dba[1])/2:.3f} "
                f"| {mean(dbc):+.3f} +/- {abs(dbc[0]-dbc[1])/2:.3f} "
                f"| {fmt(ref_avg(sel_ref, model, corpus))} "
                f"| {fmt(ref_avg(rand_ref, model, corpus))} |"
            )

    md.append("\n## Per-cell scores\n")
    md.append("| model | corpus | eval | seed | armA | armC | armB | delta B-A | delta B-C | sel-500 | rand-500 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        k = (r["model"], r["corpus"], r["eval"])
        md.append(f"| {r['model']} | {r['corpus']} | {r['eval']} | {r['seed']} | "
                  f"{r['armA_score']:.3f} | {r['armC_score']:.3f} | {r['armB_score']:.3f} | "
                  f"{r['delta_BA']:+.3f} | {r['delta_BC']:+.3f} | "
                  f"{fmt(sel_ref[k])} | {fmt(rand_ref[k])} |")

    md.append("\n## Verdict\n")
    dba_all = mean([r["delta_BA"] for r in rows])
    dbc_all = mean([r["delta_BC"] for r in rows])
    md.append(f"Across all {len(rows)} cells: mean delta B-A = **{dba_all:+.4f}**, "
              f"mean delta B-C = **{dbc_all:+.4f}** "
              "(positive = the 500 ATLAS-selected examples causally raise the MedHallu "
              "medical-consistency judge score, whether swapped in at fixed size (B-A) "
              "or added on top of the shared randoms (B-C)).\n")

    md_path = os.path.join(OUT_ROOT, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[OK] wrote {md_path}")

    # --- console headline ------------------------------------------------------
    for model in MODELS:
        for corpus in CORPORA:
            for S in SEEDS:
                a = cell_avg(model, corpus, "armA_score", S)
                c = cell_avg(model, corpus, "armC_score", S)
                b = cell_avg(model, corpus, "armB_score", S)
                print(f"HEADLINE {model} {corpus} s{S}: A={a:.3f} C={c:.3f} B={b:.3f} "
                      f"B-A={b-a:+.3f} B-C={b-c:+.3f} "
                      f"| sel500={fmt(ref_avg(sel_ref, model, corpus))} "
                      f"rand500={fmt(ref_avg(rand_ref, model, corpus))}")
    print(f"OVERALL mean delta B-A = {dba_all:+.4f}, mean delta B-C = {dbc_all:+.4f}")


if __name__ == "__main__":
    main()
