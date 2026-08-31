# analysis/cross_model_transfer/prep_addition_data.py
"""
Prepare training jsonls for the controlled "addition" test (R3-Major-1).

For each (model, corpus, eval, seed):
  armA = 1000 random corpus examples (excluding the selected 500, deduped by content)
  armB = the 500 RCT+RD-selected examples + armA[:500]
So B differs from A exactly by replacing armA[500:1000] with the selected 500.

Rows are written in the same jsonl schema as selected_train_data.jsonl:
  {"treatment_messages": [{role, content}, ...], "control_messages": [...]}
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import polars as pl

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

OUT_ROOT = Path(f"{DATA_ROOT}/data/rebuttal/addition_test")
DATASET_DIR = f"{DATA_ROOT}/dataset"
ROOTS = {
    "llama": f"{DATA_ROOT}/data/llama_attr_l19_cos",
    "qwen": f"{DATA_ROOT}/data/qwen2.5_attr_l17_cos",
}
MODELS = ["llama", "qwen"]
CORPORA = ["ultrachat_200k", "openorca_200k"]
EVALS = [f"medhallu_{d}_with_knowledge_balanced" for d in ["easy", "medium", "hard"]]
SEEDS = [42, 43]
N_RANDOM = 1000
K_SEL = 500


def canon(messages):
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def find_selected_jsonl(model, corpus, ev):
    pat = (f"{ROOTS[model]}/{corpus}/residual_change_treatment/"
           f"{corpus}-cos_sim-residual_change_treatment+none-residual_diff-500-{ev}-*")
    dirs = sorted(d for d in glob.glob(pat) if not re.search(r"-(mixrdk|seed)[^/]*$", d))
    assert dirs, f"no selection dir for {pat}"
    return os.path.join(dirs[-1], "selected_train_data.jsonl")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps({"treatment_messages": r["treatment_messages"],
                                "control_messages": r["control_messages"]},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    n_rows = {c: pl.scan_parquet(f"{DATASET_DIR}/{c}.parquet").select(pl.len()).collect().item()
              for c in CORPORA}
    print("corpus sizes:", n_rows)

    for mi, model in enumerate(MODELS):
        for ci, corpus in enumerate(CORPORA):
            lf = pl.scan_parquet(f"{DATASET_DIR}/{corpus}.parquet").with_row_index("ridx")
            for ei, ev in enumerate(EVALS):
                sel_path = find_selected_jsonl(model, corpus, ev)
                selected = load_jsonl(sel_path)
                assert len(selected) == K_SEL, (sel_path, len(selected))
                sel_keys = {canon(r["treatment_messages"]) for r in selected}

                for S in SEEDS:
                    rng = np.random.default_rng([S, mi, ci, ei])
                    # oversample, keep permutation order, dedupe vs selected + within draw
                    cand_idx = rng.permutation(n_rows[corpus])[: N_RANDOM + 4000]
                    sub = lf.filter(pl.col("ridx").is_in(cand_idx)).collect()
                    by_ridx = {int(r["ridx"]): r for r in sub.to_dicts()}
                    randoms, seen = [], set()
                    for ix in cand_idx:
                        r = by_ridx.get(int(ix))
                        if r is None:
                            continue
                        row = {"treatment_messages": r["treatment_messages"],
                               "control_messages": r["control_messages"]}
                        k = canon(row["treatment_messages"])
                        if k in sel_keys or k in seen:
                            continue
                        seen.add(k)
                        randoms.append(row)
                        if len(randoms) == N_RANDOM:
                            break
                    assert len(randoms) == N_RANDOM, (model, corpus, ev, S, len(randoms))

                    base = OUT_ROOT / model / corpus / ev / f"seed{S}"
                    write_jsonl(base / "armA" / "train_data.jsonl", randoms)
                    write_jsonl(base / "armB" / "train_data.jsonl", selected + randoms[:K_SEL])
                    print(f"[OK] {model}/{corpus}/{ev}/seed{S}: armA=1000 armB=1000 "
                          f"(selected from {os.path.basename(os.path.dirname(sel_path))})")
    print("[DONE] all training files prepared")


if __name__ == "__main__":
    main()
