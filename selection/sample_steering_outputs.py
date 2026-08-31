"""
Sample model outputs at different steering strengths for visual inspection.

Reads the per-cell CSVs that steering_experiment.py wrote under
{run_dir}/generations/ and emits a markdown file per (method × dataset) that
shows, for a small set of fixed questions, how the answer changes across coeffs.

Usage:
    python -m selection.sample_steering_outputs RUN_DIR [--n-questions 4] [--coeffs -8 0 4 8]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

TAG_RE = re.compile(
    r"^(?P<method>.+?)_(?P<dataset>(?:medhallu|ultra|empathy|laziness|modesty|preachiness|sycophancy).+?)_coef(?P<coeff>[+-]?\d+(?:\.\d+)?)\.csv$"
)


def parse_tag(name: str):
    m = TAG_RE.match(name)
    if not m:
        return None
    return m["method"], m["dataset"], float(m["coeff"])


def truncate(s: str, n: int = 800) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--n-questions", type=int, default=4)
    ap.add_argument(
        "--coeffs",
        nargs="+",
        type=float,
        default=[-8.0, 0.0, 4.0, 8.0],
        help="Coeffs to display side-by-side (must exist in the run).",
    )
    ap.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="If multiple repeats per question, pick this sample (default 0).",
    )
    ap.add_argument(
        "--output-subdir",
        default="sample_outputs",
        help="Subdir under run_dir to write markdown files into.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gen_dir = run_dir / "generations"
    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(exist_ok=True)

    # Group CSVs by (method, dataset)
    cells = defaultdict(dict)  # {(method, dataset): {coeff: path}}
    for p in sorted(gen_dir.glob("*.csv")):
        tag = parse_tag(p.name)
        if tag is None:
            continue
        method, dataset, coeff = tag
        cells[(method, dataset)][coeff] = p

    if not cells:
        raise SystemExit(f"No matching CSVs found under {gen_dir}")

    for (method, dataset), coeff_paths in sorted(cells.items()):
        missing = [c for c in args.coeffs if c not in coeff_paths]
        if missing:
            print(f"[SKIP] {method} {dataset}: missing coeffs {missing}")
            continue
        # Read each coeff CSV into list-of-rows.
        rows_by_coeff = {}
        for c in args.coeffs:
            with open(coeff_paths[c]) as f:
                rows_by_coeff[c] = list(csv.DictReader(f))
        # Filter to sample_idx (if present)
        for c in args.coeffs:
            rs = rows_by_coeff[c]
            if rs and "sample_idx" in rs[0]:
                rows_by_coeff[c] = [
                    r for r in rs if int(r.get("sample_idx") or 0) == args.sample_idx
                ]
        # Pick the first N questions (rows aligned across coeffs by row index).
        n = min(args.n_questions, len(rows_by_coeff[args.coeffs[0]]))
        questions_csv_idx = list(range(n))

        out_path = out_dir / f"{method}_{dataset}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Sample outputs — {method} × {dataset}\n\n")
            f.write(f"Coeffs shown: {args.coeffs}\n")
            f.write(f"Sample idx: {args.sample_idx} (out of available repeats)\n\n")
            for qi in questions_csv_idx:
                base = rows_by_coeff[args.coeffs[0]][qi]
                f.write(f"---\n\n## Question {qi+1}\n\n")
                f.write(f"> {truncate(base['question'], 1200)}\n\n")
                # Optional reference fields
                for key in ("ground_truth", "high_quality", "low_quality"):
                    if key in base and base[key]:
                        f.write(f"**{key}:** {truncate(base[key], 600)}\n\n")
                # Per-coeff answers
                for c in args.coeffs:
                    row = rows_by_coeff[c][qi]
                    score = row.get("score", "")
                    rationale = row.get("rationale", "")
                    f.write(f"### coeff = {c:+.1f}   (judge score = {score})\n\n")
                    f.write(f"{truncate(row['answer'], 1200)}\n\n")
                    if rationale:
                        f.write(f"*rationale*: {truncate(rationale, 200)}\n\n")
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
