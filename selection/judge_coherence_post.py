"""
Post-hoc coherence judging for a completed steering_experiment.py run.

Reads every CSV under {run_dir}/generations/, judges each saved answer with
coherence_0_3, and writes:
  - {run_dir}/generations_coherence/{tag}.csv       (prompt, answer, coherence_0_3, coherence_rationale)
  - {run_dir}/steering_results_coherence.jsonl      (per-cell mean coherence + proportions)

Run AFTER the main sweep is complete. Does not need a GPU.
"""
import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from selection.llm_judge.judge import OpenAiJudge
from selection.llm_judge.prompts import Prompts


TAG_RE = re.compile(
    r"^(?P<method>.+?)"
    r"_(?P<dataset>(?:medhallu|ultra|empathy|laziness|modesty|preachiness|sycophancy)"
    r".+?)"
    r"_coef(?P<coeff>[+-]?\d+(?:\.\d+)?)\.csv$"
)


def parse_tag(filename: str):
    m = TAG_RE.match(filename)
    if not m:
        return None
    return m["method"], m["dataset"], float(m["coeff"])


async def judge_csv(judge, in_path: Path, out_path: Path, concurrency: int):
    rows = list(csv.DictReader(open(in_path)))
    sem = asyncio.Semaphore(concurrency)

    async def score(row):
        async with sem:
            return await judge(question=row["question"], answer=row["answer"])

    results = await asyncio.gather(*[score(r) for r in rows])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["question", "answer", "coherence_0_3", "coherence_rationale"]
        )
        w.writeheader()
        for row, res in zip(rows, results):
            score, rationale = (res if res is not None else (None, None))
            w.writerow(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "coherence_0_3": score,
                    "coherence_rationale": rationale,
                }
            )
    scored = [s for s, _ in (r for r in results if r is not None) if s is not None]
    return scored, len(rows)


def summarize(method, dataset, coeff, scored, n_total):
    n = len(scored)
    summary = {
        "method": method,
        "dataset": dataset,
        "coeff": coeff,
        "n_total": n_total,
        "n_scored": n,
        "mean_coherence": (sum(scored) / n) if n else None,
    }
    if n:
        for s in (0, 1, 2, 3):
            summary[f"prop_coherence_{s}"] = sum(1 for v in scored if v == s) / n
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Path to a completed steering_experiment output dir")
    ap.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--judge-concurrency", type=int, default=32)
    ap.add_argument("--trait", default="coherence_0_3")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-judge cells even if {run_dir}/generations_coherence/{tag}.csv already exists",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gen_dir = run_dir / "generations"
    out_csv_dir = run_dir / "generations_coherence"
    metrics_path = run_dir / "steering_results_coherence.jsonl"

    template = Prompts.get(args.trait)
    if template is None:
        raise ValueError(f"No prompt template named '{args.trait}' in Prompts")
    judge = OpenAiJudge(args.judge_model, template, args.trait)

    csvs = sorted(gen_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No generations CSVs found under {gen_dir}")

    print(f"Found {len(csvs)} cell CSVs. Judging with {args.trait}...")

    all_summaries = []
    for in_path in csvs:
        tag = parse_tag(in_path.name)
        if tag is None:
            print(f"[SKIP] unrecognized filename: {in_path.name}")
            continue
        method, dataset, coeff = tag

        out_path = out_csv_dir / in_path.name
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] already judged: {out_path.name}")
            # Still recompute summary from existing CSV
            scored = []
            n_total = 0
            for r in csv.DictReader(open(out_path)):
                n_total += 1
                v = r.get("coherence_0_3")
                try:
                    if v not in (None, "", "None"):
                        scored.append(float(v))
                except ValueError:
                    pass
        else:
            print(f"[JUDGE] {in_path.name}")
            scored, n_total = asyncio.run(
                judge_csv(judge, in_path, out_path, args.judge_concurrency)
            )

        summary = summarize(method, dataset, coeff, scored, n_total)
        all_summaries.append(summary)
        mc = summary["mean_coherence"]
        mc_s = f"{mc:.3f}" if mc is not None else "  NA"
        print(f"  mean_coherence={mc_s}  n={summary['n_scored']}/{summary['n_total']}")

    with open(metrics_path, "w") as f:
        for s in all_summaries:
            f.write(json.dumps(s) + "\n")
    print(f"\nWrote {len(all_summaries)} cell summaries to {metrics_path}")


if __name__ == "__main__":
    main()
