# analysis/cross_model_transfer/prep_armC_data.py
"""
Arm C extension of the controlled addition test (R3-Major-1).

armC = ONLY the 500 random examples that armB shares with armA, i.e. the
intersection of armA and armB rows (deduped by full row content). By
construction (prep_addition_data.py) this should be exactly armA[:500];
both facts are verified per cell:

  1. armB == selected_500 + armA[:500]  (content-level, order-preserving)
  2. intersection(armA, armB) == exactly 500 rows == armA[:500]

armC/train_data.jsonl is written next to armA/armB.
"""
import json
import os
from pathlib import Path

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

OUT_ROOT = Path(f"{DATA_ROOT}/data/rebuttal/addition_test")
MODELS = ["llama", "qwen"]
CORPORA = ["ultrachat_200k", "openorca_200k"]
EVALS = [f"medhallu_{d}_with_knowledge_balanced" for d in ["easy", "medium", "hard"]]
SEEDS = [42, 43]
K = 500


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def canon(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    n = 0
    for model in MODELS:
        for corpus in CORPORA:
            for ev in EVALS:
                for S in SEEDS:
                    base = OUT_ROOT / model / corpus / ev / f"seed{S}"
                    A = load(base / "armA" / "train_data.jsonl")
                    B = load(base / "armB" / "train_data.jsonl")
                    assert len(A) == 1000 and len(B) == 1000, (base, len(A), len(B))

                    a_keys = [canon(r) for r in A]
                    b_keys = [canon(r) for r in B]
                    assert len(set(a_keys)) == 1000, f"dup rows in armA {base}"

                    # verify armB = 500 selected (not in A) + armA[:500], in order
                    assert b_keys[K:] == a_keys[:K], \
                        f"armB[500:] != armA[:500] at {base}"
                    assert not set(b_keys[:K]) & set(a_keys), \
                        f"selected half of armB overlaps armA at {base}"

                    # armC = intersection of armA and armB rows, deduped by content
                    inter = set(a_keys) & set(b_keys)
                    assert len(inter) == K, f"intersection={len(inter)} != {K} at {base}"
                    armC = [r for r, k in zip(A, a_keys) if k in inter]
                    assert [canon(r) for r in armC] == a_keys[:K]

                    write_jsonl(base / "armC" / "train_data.jsonl", armC)
                    n += 1
                    print(f"[OK] {model}/{corpus}/{ev}/seed{S}: armC=500 (== armA[:500])")
    print(f"[DONE] wrote {n} armC files, all assertions passed")


if __name__ == "__main__":
    main()
