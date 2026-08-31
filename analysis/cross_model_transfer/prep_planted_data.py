# analysis/cross_model_transfer/prep_planted_data.py
"""
Prepare the 500 PLANTED MedHallu examples for the "planted set is
behavior-inducing" check (planted-recovery experiment).

The planted rows are indices 1000-1499 of corpus_1500.parquet. Each row already
has treatment_messages + control_messages (the driver's schema); we just slice
and dump to jsonl. Same file goes to every fine-tune run (model-independent).
"""
import json
import os
from pathlib import Path

import polars as pl

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

PARQUET = f"{DATA_ROOT}/data/rebuttal/planted_recovery/corpus_1500.parquet"
OUT = Path(f"{DATA_ROOT}/data/rebuttal/planted_recovery/planted_finetune_check/planted_500.jsonl")
LO, HI = 1000, 1500


def main():
    df = pl.read_parquet(PARQUET)
    assert df.height == 1500, df.height
    planted = df.slice(LO, HI - LO).to_dicts()
    assert len(planted) == 500, len(planted)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w") as f:
        for r in planted:
            assert r["treatment_messages"] and r["control_messages"], "empty chat"
            f.write(json.dumps({"treatment_messages": r["treatment_messages"],
                                "control_messages": r["control_messages"]},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    print(f"[OK] wrote {OUT} with {len(planted)} planted rows (indices {LO}-{HI-1})")


if __name__ == "__main__":
    main()
