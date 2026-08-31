# analysis/cross_model_transfer/collect_results.py
"""
Aggregate cross-model transfer pilot results into results.csv + RESULTS.md.

For each (direction, train, eval) cell:
  transfer_score : model B fine-tuned on model A's selected top-500
  ownB_score     : model B fine-tuned on its OWN RCT+RD top-500 (existing run)
  randomB_score  : model B fine-tuned on random-500 (existing run)

Metric: ft_medical_consistency_0_2_avg (last line of metrics.jsonl = post-FT).
"""
import csv
import glob
import json
import os
import re
import sys

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

KEY = "ft_medical_consistency_0_2_avg"
OUT_ROOT = f"{DATA_ROOT}/data/rebuttal/cross_model_transfer"
ROOTS = {
    "llama": f"{DATA_ROOT}/data/llama_attr_l19_cos",
    "qwen": f"{DATA_ROOT}/data/qwen2.5_attr_l17_cos",
}
# direction -> model B (the model that is fine-tuned/evaluated)
B_OF = {"llama_to_qwen": "qwen", "qwen_to_llama": "llama"}
TRAINS = ["ultrachat_200k", "openorca_200k"]
EVALS = [f"medhallu_{d}_with_knowledge_balanced" for d in ["easy", "medium", "hard"]]
DIRECTIONS = ["llama_to_qwen", "qwen_to_llama"]


def last_metric(path):
    lines = [l for l in open(path) if l.strip()]
    return json.loads(lines[-1])[KEY]


def existing_run_metric(root, train, method, eval_name):
    pat = (f"{root}/{train}/{method}/{train}-cos_sim-{method}+none-residual_diff-500-"
           f"{eval_name}-*")
    dirs = sorted(d for d in glob.glob(pat) if not re.search(r"-(mixrdk|seed)[^/]*$", d))
    if not dirs:
        raise FileNotFoundError(pat)
    return last_metric(os.path.join(dirs[-1], "selected_data/eval_llm_judge/metrics.jsonl"))


def main():
    rows = []
    missing = []
    for direction in DIRECTIONS:
        b = B_OF[direction]
        for train in TRAINS:
            for ev in EVALS:
                mfile = os.path.join(OUT_ROOT, direction, train, ev,
                                     "selected_data/eval_llm_judge/metrics.jsonl")
                if not (os.path.exists(mfile) and os.path.getsize(mfile) > 0):
                    missing.append(mfile)
                    continue
                transfer = last_metric(mfile)
                own = existing_run_metric(ROOTS[b], train, "residual_change_treatment", ev)
                rnd = existing_run_metric(ROOTS[b], train, "random", ev)
                rows.append({
                    "direction": direction, "train": train, "eval": ev,
                    "transfer_score": round(transfer, 4),
                    "ownB_score": round(own, 4),
                    "randomB_score": round(rnd, 4),
                })

    if missing:
        print("INCOMPLETE - missing metrics for:")
        for m in missing:
            print("  ", m)
        sys.exit(1)

    csv_path = os.path.join(OUT_ROOT, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {csv_path}")

    # ---- RESULTS.md: 3-difficulty averages per (direction, train) ----
    def cell_avg(direction, train, col):
        vals = [r[col] for r in rows if r["direction"] == direction and r["train"] == train]
        return sum(vals) / len(vals)

    md = []
    md.append("# Cross-model transfer pilot (paper Appendix J)\n")
    md.append("Design: take the exact top-500 subsets selected by model A's attribution "
              "(Llama-3.1-8B @ L19 / Qwen2.5-7B @ L17, RCT attribution + RD selection, cos_sim), "
              "LoRA-fine-tune model B on them with the paper protocol (k=500, r=32, 1 epoch, "
              "batch 2), and evaluate with the standard LLM judge "
              "(`ft_medical_consistency_0_2_avg`, 3 repeats). Scores below are averages over "
              "the three MedHallu difficulties (easy/medium/hard).\n")
    md.append("| direction | train set | transfer (A-sel -> B) | B own-sel | B random | transfer/own | transfer - random |")
    md.append("|---|---|---|---|---|---|---|")
    for direction in DIRECTIONS:
        for train in TRAINS:
            t = cell_avg(direction, train, "transfer_score")
            o = cell_avg(direction, train, "ownB_score")
            r = cell_avg(direction, train, "randomB_score")
            md.append(f"| {direction} | {train} | {t:.3f} | {o:.3f} | {r:.3f} | "
                      f"{t / o:.2f} | {t - r:+.3f} |")

    md.append("\n## Per-difficulty scores\n")
    md.append("| direction | train | eval | transfer | own | random |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['direction']} | {r['train']} | {r['eval']} | "
                  f"{r['transfer_score']:.3f} | {r['ownB_score']:.3f} | {r['randomB_score']:.3f} |")

    # interpretation placeholder is filled in manually after inspecting numbers
    md.append("\n## Interpretation\n")
    md.append("<!-- INTERPRETATION -->\n")

    md_path = os.path.join(OUT_ROOT, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[OK] wrote {md_path}")

    for direction in DIRECTIONS:
        for train in TRAINS:
            t = cell_avg(direction, train, "transfer_score")
            o = cell_avg(direction, train, "ownB_score")
            r = cell_avg(direction, train, "randomB_score")
            print(f"HEADLINE {direction} {train}: transfer={t:.3f} own={o:.3f} "
                  f"random={r:.3f} ratio={t/o:.2f} margin={t-r:+.3f}")


if __name__ == "__main__":
    main()
