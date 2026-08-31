"""Aggregate the (layer x family) steering sweep into a single long table.

Globs over a set of run dirs that match `steering_l{LAYER}_{FAMILY}_{TS}/`,
joins trait + coherence summaries, and writes:
  - {out_dir}/combined_long.csv     long table indexed by (layer, method, dataset, coeff)
  - {out_dir}/trait_table.txt       pivoted trait mean_score (method x dataset x coeff x layer)
  - {out_dir}/coherence_table.txt   pivoted mean_coherence
  - {out_dir}/delta_vs_baseline.csv (Delta trait, Delta coherence) at the max +coeff vs coeff=0

Layer is parsed from the run dir name. Coefficient grid is taken from data.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import os

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")


def _build_run_dir_re(prefix: str) -> re.Pattern:
    # TS may be a bare timestamp (20260518_012158) or carry a suffix
    # (20260518_163045_exp_coefs). Allow optional `_<suffix>` so re-runs with
    # tagged TS slugs aggregate without further regex work.
    return re.compile(
        rf"^{re.escape(prefix)}(?P<layer>\d+)_(?P<family>medhallu|ultra|personality)_(?P<ts>\d{{8}}_\d{{6}}(?:_[A-Za-z0-9_]+)?)$"
    )


def find_run_dirs(root: Path, prefix: str, ts_filter: str = None):
    pat = _build_run_dir_re(prefix)
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        if ts_filter and m["ts"] != ts_filter:
            continue
        dirs.append((int(m["layer"]), m["family"], m["ts"], d))
    return dirs


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=f"{DATA_ROOT}/data",
        help="Directory containing the steering_l{layer}_{family}_{ts}/ dirs",
    )
    ap.add_argument(
        "--ts",
        default=None,
        help="Restrict to a specific TS slug (e.g. 20260516_160940)",
    )
    ap.add_argument(
        "--prefix",
        default="steering_l",
        help=(
            "Run-dir prefix up to (but not including) the layer number. "
            "Default 'steering_l' matches the Llama sweep; use 'steering_qwen_l' for the Qwen sweep."
        ),
    )
    ap.add_argument("--out-dir", required=True, help="Where to write the aggregate tables")
    ap.add_argument(
        "--report-coeff",
        type=float,
        default=None,
        help="Coeff at which to compute delta vs baseline. Defaults to max +coeff present.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(root, prefix=args.prefix, ts_filter=args.ts)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched under {root} (ts={args.ts})")

    print(f"Found {len(run_dirs)} run dirs:")
    for layer, family, ts, d in run_dirs:
        print(f"  l{layer} {family} {ts}  ({d.name})")

    # Join trait + coherence by (method, dataset, coeff, layer).
    # Coh coeffs come from CSV filenames (`coef{coeff:+.1f}.csv`) so they're
    # rounded to 1 decimal place; trait coeffs are the full-precision sweep
    # values. Snap both to 1 decimal place for the join — safe as long as
    # the sweep grid has no pair within ~0.1 of each other.
    def _round_coeff(c):
        return round(float(c), 1)
    by_key = defaultdict(dict)  # key=(layer, method, dataset, coeff) -> dict
    for layer, family, ts, d in run_dirs:
        trait_rows = load_jsonl(d / "steering_results.jsonl")
        coh_rows = load_jsonl(d / "steering_results_coherence.jsonl")
        coh_by_key = {
            (r["method"], r["dataset"], _round_coeff(r["coeff"])): r for r in coh_rows
        }
        for r in trait_rows:
            key = (layer, r["method"], r["dataset"], r["coeff"])
            row = by_key[key]
            row["layer"] = layer
            row["family"] = family
            row["method"] = r["method"]
            row["dataset"] = r["dataset"]
            row["coeff"] = r["coeff"]
            row["n_questions"] = r.get("n_questions")
            row["n_generations"] = r.get("n_generations")
            row["n_scored"] = r.get("n_scored")
            row["mean_score"] = r.get("mean_score")
            for k, v in r.items():
                if k.startswith("prop_score_"):
                    row[k] = v
            ck = (r["method"], r["dataset"], _round_coeff(r["coeff"]))
            c = coh_by_key.get(ck)
            if c is not None:
                row["mean_coherence"] = c.get("mean_coherence")
                for k, v in c.items():
                    if k.startswith("prop_coherence_"):
                        row[k] = v

    rows = sorted(
        by_key.values(),
        key=lambda r: (
            r["family"],
            r["dataset"],
            r["method"],
            r["layer"],
            r["coeff"],
        ),
    )

    # Long CSV
    fields = sorted({k for r in rows for k in r.keys()})
    pref = [
        "family", "layer", "method", "dataset", "coeff",
        "n_questions", "n_generations", "n_scored",
        "mean_score", "mean_coherence",
    ]
    fields = pref + [f for f in fields if f not in pref]
    out_csv = out_dir / "combined_long.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    # Pivot tables
    coeffs = sorted({r["coeff"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    def pivot(metric: str, path: Path):
        with open(path, "w") as f:
            f.write(f"# {metric}  (rows: layer x method x dataset; cols: coeff)\n")
            header = f"{'layer':>5} {'method':<28}{'dataset':<48}" + "".join(
                f"{c:>+8.1f}" for c in coeffs
            )
            f.write(header + "\n")
            for layer in layers:
                for method in methods:
                    for ds in datasets:
                        line = f"{layer:>5} {method:<28}{ds:<48}"
                        for c in coeffs:
                            r = by_key.get((layer, method, ds, c))
                            v = (r or {}).get(metric)
                            line += f"{'   NA  ' if v is None else f'{v:>8.3f}'}"
                        f.write(line + "\n")
                    f.write("\n")
        print(f"Wrote pivot -> {path}")

    pivot("mean_score", out_dir / "trait_table.txt")
    pivot("mean_coherence", out_dir / "coherence_table.txt")

    # Delta vs baseline at the report coeff
    report_coeff = args.report_coeff
    if report_coeff is None:
        report_coeff = max(c for c in coeffs if c > 0)
    delta_path = out_dir / "delta_vs_baseline.csv"
    with open(delta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "layer", "method", "dataset",
                "baseline_trait", f"trait@+{report_coeff:g}", "delta_trait",
                "baseline_coh", f"coh@+{report_coeff:g}", "delta_coh",
            ]
        )
        for layer in layers:
            for method in methods:
                for ds in datasets:
                    base = by_key.get((layer, method, ds, 0.0))
                    top = by_key.get((layer, method, ds, float(report_coeff)))
                    if base is None or top is None:
                        continue
                    bt = base.get("mean_score")
                    tt = top.get("mean_score")
                    bc = base.get("mean_coherence")
                    tc = top.get("mean_coherence")
                    w.writerow(
                        [
                            layer, method, ds,
                            bt, tt, (None if bt is None or tt is None else tt - bt),
                            bc, tc, (None if bc is None or tc is None else tc - bc),
                        ]
                    )
    print(f"Wrote delta table -> {delta_path}")


if __name__ == "__main__":
    main()
