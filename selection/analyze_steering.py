"""Quick analysis of steering_results.jsonl: dose-response slope per (method, dataset)."""
import argparse
import json
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_path", help="Path to steering_results.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.results_path) if line.strip()]
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["method"], r["dataset"])].append((r["coeff"], r["mean_score"], r["n_scored"]))

    print(f"{'method':<18}{'dataset':<48}{'baseline':>10}{'@-8':>8}{'@+8':>8}{'slope':>10}")
    for (method, dataset), pts in sorted(by_cell.items()):
        pts.sort()
        coeffs = [c for c, s, _ in pts if s is not None]
        scores = [s for c, s, _ in pts if s is not None]
        if not scores:
            continue
        # baseline = coeff 0
        base = next((s for c, s, _ in pts if c == 0.0), None)
        s_neg = next((s for c, s, _ in pts if c == -8.0), None)
        s_pos = next((s for c, s, _ in pts if c == 8.0), None)
        # OLS slope
        n = len(coeffs)
        mc = sum(coeffs) / n
        ms = sum(scores) / n
        num = sum((c - mc) * (s - ms) for c, s in zip(coeffs, scores))
        den = sum((c - mc) ** 2 for c in coeffs)
        slope = num / den if den > 0 else 0.0
        b = f"{base:.3f}" if base is not None else "  NA "
        ng = f"{s_neg:.3f}" if s_neg is not None else "  NA "
        po = f"{s_pos:.3f}" if s_pos is not None else "  NA "
        print(f"{method:<18}{dataset:<48}{b:>10}{ng:>8}{po:>8}{slope:>10.4f}")


if __name__ == "__main__":
    main()
