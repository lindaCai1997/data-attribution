#!/usr/bin/env python
"""Build a planted-recovery corpus for a NEW base pool (ultrachat / openorca).

Reuses the EXACT 500 planted MedHallu rows (same nested order) from the original
corpus_1500.parquet[1000:1500], so plant sizes 100/250/500 remain row-prefix slices
and disjointness/anti-circularity is unchanged (planted set is corpus-independent).
Base pool = 1000 rows sampled (seed 0) from the given pool parquet, appended-before.
"""
import sys, json
import pandas as pd
import os

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

OUT = f"{DATA_ROOT}/data/rebuttal/planted_recovery"
POOL_PARQUET = {
    "ultrachat": f"{DATA_ROOT}/dataset/ultrachat_200k.parquet",
    "openorca":  f"{DATA_ROOT}/dataset/openorca_200k.parquet",
}
N_DOLLY = 1000
SEED = 0


def main():
    pool = sys.argv[1]
    assert pool in POOL_PARQUET, pool

    # planted block = exact rows 1000:1500 from the original corpus
    orig = pd.read_parquet(f"{OUT}/corpus_1500.parquet")
    planted = orig.iloc[N_DOLLY:].reset_index(drop=True)[
        ["treatment_messages", "control_messages"]].copy()
    assert len(planted) == 500

    base = pd.read_parquet(POOL_PARQUET[pool])
    base_sample = base.sample(n=N_DOLLY, random_state=SEED).reset_index(drop=True)
    base_sample = base_sample[["treatment_messages", "control_messages"]].copy()

    corpus = pd.concat([base_sample, planted], ignore_index=True)
    assert len(corpus) == N_DOLLY + 500
    out_path = f"{OUT}/corpus_1500_{pool}.parquet"
    corpus.to_parquet(out_path, index=False)

    # sanity: planted at 1000:1500, all have [CONTEXT]; base has none
    ctx_p = sum("[CONTEXT]" in corpus["treatment_messages"].iloc[i][0]["content"]
                for i in range(N_DOLLY, N_DOLLY + 500))
    ctx_b = sum("[CONTEXT]" in corpus["treatment_messages"].iloc[i][0]["content"]
                for i in range(0, N_DOLLY))
    # verify planted rows identical to original
    same = all(
        corpus["treatment_messages"].iloc[N_DOLLY + i][1]["content"]
        == orig["treatment_messages"].iloc[N_DOLLY + i][1]["content"]
        for i in range(500))
    print(f"{pool}: wrote {out_path} rows={len(corpus)} planted_with_context={ctx_p}/500 "
          f"base_with_context={ctx_b}/1000 planted_identical_to_orig={same}")
    assert ctx_p == 500 and same


if __name__ == "__main__":
    main()
