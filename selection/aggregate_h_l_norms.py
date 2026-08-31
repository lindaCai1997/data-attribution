"""Aggregate per-trait residual-stream norms into a single per-(model, layer) JSON.

Reads the .pt files produced by persona_vector/measure_activation_norm.py for a
set of trait CSVs, slices `layer_norms_response` at the layers actually used in
the steering sweep, and writes a JSON keyed by layer with per-trait values, the
pooled mean (= the scalar used as ||h_l|| in norm-aware steering), and the
cross-trait CV. A CV > --cv-warn-threshold prints a WARNING.

Usage:
  python -m selection.aggregate_h_l_norms \
      --norm-dir persona_vector/activation_norms/Meta-Llama-3.1-8B-Instruct \
      --traits sycophancy_gpt medical_hallucination_with_knowledge_easy_gpt laziness_gpt modesty_gpt \
      --layers 15 17 19 21 \
      --out persona_vector/activation_norms/Meta-Llama-3.1-8B-Instruct/h_l_norms.json
"""
import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norm-dir", required=True, help="dir with <trait>_pos.pt files")
    ap.add_argument("--traits", nargs="+", required=True, help="trait base names (no _pos.pt)")
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--cv-warn-threshold", type=float, default=0.10)
    args = ap.parse_args()

    norm_dir = Path(args.norm_dir)
    out_path = Path(args.out)

    # Load each trait's per-layer response norms.
    trait_norms = {}  # trait -> {layer: norm}
    for trait in args.traits:
        pt = norm_dir / f"{trait}_pos.pt"
        if not pt.exists():
            print(f"WARNING: missing {pt}, skipping trait {trait}")
            continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        resp = d["layer_norms_response"]
        trait_norms[trait] = {L: float(resp[L].item()) for L in args.layers}
        print(f"loaded {trait}: n={d.get('n_examples')}, "
              + " ".join(f"L{L}={trait_norms[trait][L]:.2f}" for L in args.layers))

    if not trait_norms:
        raise SystemExit("No trait norm files loaded.")

    # Per-layer pooled mean + CV across traits.
    out = {}
    any_warn = False
    for L in args.layers:
        per_trait = {t: trait_norms[t][L] for t in trait_norms}
        vals = list(per_trait.values())
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = std / mean if mean else 0.0
        warn = cv > args.cv_warn_threshold
        any_warn = any_warn or warn
        out[str(L)] = {**per_trait, "pooled": mean, "cv": cv}
        flag = "  [HIGH CV]" if warn else ""
        print(f"L{L}: pooled={mean:.3f}  cv={cv:.3f}{flag}  "
              + " ".join(f"{t}={v:.2f}" for t, v in per_trait.items()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    if any_warn:
        print("WARNING: at least one layer has CV above threshold "
              f"{args.cv_warn_threshold}; the per-layer 'pooled' value may be "
              "a poor scalar for steering.")


if __name__ == "__main__":
    main()
