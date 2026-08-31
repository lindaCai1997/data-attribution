#!/usr/bin/env python
"""Build the planted-recovery corpus for the ATLAS planted-example retrieval experiment (paper Appendix H).

Base pool: 1000 rows sampled (seed 0) from dolly_10k.
Planted pool: MedHallu hallucination-inducing examples (treatment = hallucinated
answer, control = ground-truth answer), taken from the FULL MedHallu parquets but
restricted to rows DISJOINT (by question stem) from the *balanced* attribution
parquets that were used to build the eval-side query vectors -> satisfies the
anti-circularity requirement. Within that disjoint pool we prioritise the items
that are also in the held-out eval JSONs (the ideal planted source: never used to
build query vectors AND carrying a genuine MedHallu hallucinated answer), then
fill the remainder from the rest of the disjoint pool.

Plant sizes 100/250/500 are NESTED and appended after the fixed 1000 dolly rows,
so corpus_1500 rows [0:1000+N] == the N-plant corpus. Stage-1 is therefore run
ONCE on corpus_1500 and smaller plant sizes are row-prefix slices.

Planted rows occupy indices 1000 .. 1499 of corpus_1500.
"""
import json, sys
import numpy as np
import pandas as pd
import os

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

# Staging dir holding the atlas (medhallu) data splits from the EMNLP submission.
STAGING = os.environ.get(
    "ATLAS_STAGING_DIR",
    "/accounts/projects/jsteinhardt/spa-data-attribution/emnlp_submission/staging_data/atlas_data",
)
FULL_DIR = f"{DATA_ROOT}/dataset"
OUT_DIR = f"{DATA_ROOT}/data/rebuttal/planted_recovery"
DIFFS = ["easy", "medium", "hard"]
N_DOLLY = 1000
MAX_PLANT = 500
SEED = 0


def stem(q):
    return q.split("[CONTEXT]")[0].strip()


def qtext(msgs):
    return msgs[0]["content"]


def main():
    rng = np.random.RandomState(SEED)

    # ---- dolly base pool (unchanged rows) ----
    dolly = pd.read_parquet(f"{STAGING}/dataset/dolly_10k.parquet")
    dolly_sample = dolly.sample(n=N_DOLLY, random_state=SEED).reset_index(drop=True)
    dolly_sample = dolly_sample[["treatment_messages", "control_messages"]].copy()
    print(f"dolly base: {len(dolly_sample)} rows")

    # ---- planted pool ----
    held_rows = []      # (difficulty, stem, treatment, control)
    supp_rows = []
    for d in DIFFS:
        full = pd.read_parquet(f"{FULL_DIR}/medhallu_{d}_with_knowledge.parquet")
        bal = pd.read_parquet(
            f"{STAGING}/dataset/medhallu_{d}_with_knowledge_balanced.parquet")
        bal_stems = set(stem(qtext(r)) for r in bal["treatment_messages"])
        held = json.load(
            open(f"{STAGING}/eval-dataset/medhallu_{d}_with_knowledge_balanced.json"))
        held_stems = set(stem(h["question"]) for h in held)
        for i in range(len(full)):
            tm = full["treatment_messages"].iloc[i]
            cm = full["control_messages"].iloc[i]
            s = stem(qtext(tm))
            if s in bal_stems:      # anti-circularity: never use query-side rows
                continue
            rec = (d, s, tm, cm)
            if s in held_stems:
                held_rows.append(rec)
            else:
                supp_rows.append(rec)

    # dedupe by stem across the pool (keep first), held first
    seen = set()
    ordered = []
    rng.shuffle(held_rows)
    rng.shuffle(supp_rows)
    for rec in held_rows + supp_rows:
        if rec[1] in seen:
            continue
        seen.add(rec[1])
        ordered.append(rec)
    planted = ordered[:MAX_PLANT]
    held_stem_set = set(r[1] for r in held_rows)
    n_held = sum(1 for r in planted if r[1] in held_stem_set)
    print(f"planted pool: held∩disjoint={len(held_rows)} supplement={len(supp_rows)}")
    print(f"planted selected: {len(planted)} (held-out source={n_held}, "
          f"supplement={len(planted)-n_held})")
    assert len(planted) == MAX_PLANT, f"only {len(planted)} plantable rows"

    planted_df = pd.DataFrame({
        "treatment_messages": [r[2] for r in planted],
        "control_messages":   [r[3] for r in planted],
    })

    # ---- assemble corpus_1500 = dolly(1000) ++ planted(500) ----
    corpus = pd.concat([dolly_sample, planted_df], ignore_index=True)
    assert len(corpus) == N_DOLLY + MAX_PLANT
    corpus.to_parquet(f"{OUT_DIR}/corpus_1500.parquet", index=False)

    manifest = {
        "seed": SEED,
        "n_dolly": N_DOLLY,
        "max_plant": MAX_PLANT,
        "dolly_indices": [0, N_DOLLY],                     # [start, end)
        "planted_indices": [N_DOLLY, N_DOLLY + MAX_PLANT], # [start, end)
        "plant_sizes": [100, 250, 500],
        "planted": [
            {"corpus_index": N_DOLLY + i, "difficulty": planted[i][0],
             "stem": planted[i][1],
             "held_out": planted[i][1] in held_stem_set}
            for i in range(len(planted))
        ],
        "n_held_out": int(n_held),
    }
    with open(f"{OUT_DIR}/planted_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    # sanity: assert nested prefixes & disjointness
    diff_counts = {d: sum(1 for r in planted if r[0] == d) for d in DIFFS}
    print("planted difficulty breakdown:", diff_counts)
    print(f"wrote {OUT_DIR}/corpus_1500.parquet and planted_manifest.json")
    # spot check: first planted row is a medhallu (has [CONTEXT])
    first_planted = corpus["treatment_messages"].iloc[N_DOLLY]
    assert "[CONTEXT]" in first_planted[0]["content"], "planted row not medhallu-formatted"
    print("spot-check OK: first planted row contains [CONTEXT]")


if __name__ == "__main__":
    main()
