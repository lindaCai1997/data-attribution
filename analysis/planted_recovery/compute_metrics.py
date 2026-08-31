#!/usr/bin/env python
"""Planted-example recovery metrics for the ATLAS planted-example retrieval experiment (paper Appendix H).

Reuses existing eval-side query vectors and the newly-computed stage-1 train
vectors on corpus_1500 (dolly[0:1000] ++ planted[1000:1500]). For each plant
size N in {100,250,500} the corpus is the row prefix scores[:1000+N] with the
planted positives at indices 1000..1000+N-1.

Scoring exactly matches selection/select_train_data.py cos_sim path:
  eval_unit = normalize( mean_i normalize(eval_i) )      (mean over eval examples)
  train_normed = normalize(train)
  score = train_normed @ eval_unit
For PV the query is the persona vector direction at the layer (no eval mean).
Per-difficulty queries + a "combined" query = normalize(mean of the 3 unit dirs).
"""
import json, csv, sys, os, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selection.matrix import ShardedScoreMatrix

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

DATA = f"{DATA_ROOT}/data"
OUT = f"{DATA_ROOT}/data/rebuttal/planted_recovery"
DIFFS = ["easy", "medium", "hard"]
N_TOTAL = 1500  # corpus_1500 row count


def load_mat(d):
    """Eval-side loader: rows are the eval examples, order irrelevant (we mean them)."""
    return ShardedScoreMatrix(d, device="cpu").materialize().float()


def load_train_aligned(d, n_total=N_TOTAL):
    """Train-side loader aligned to original corpus row via the `idx` column.
    Returns (scores [n_total, D] float tensor, present [n_total] bool tensor).
    Missing rows (e.g. TRAK dropped examples) are zero-filled and present=False."""
    score_files = sorted(glob.glob(os.path.join(d, "scores.rank*.part*.npy")))
    assert score_files, f"no score shards in {d}"
    all_idx, all_sc = [], []
    for sf in score_files:
        m = re.search(r"scores\.rank(\d+)\.part(\d+)\.npy$", sf)
        r, p = m.group(1), m.group(2)
        dp = os.path.join(d, f"data.rank{r}.part{p}.parquet")
        idx = pd.read_parquet(dp, columns=["idx"])["idx"].to_numpy()
        sc = np.load(sf)
        assert len(idx) == sc.shape[0], f"shard idx/score mismatch {sf}"
        all_idx.append(idx); all_sc.append(sc)
    idx = np.concatenate(all_idx)
    sc = np.concatenate(all_sc, axis=0).astype(np.float32)
    D = sc.shape[1]
    full = np.zeros((n_total, D), dtype=np.float32)
    present = np.zeros(n_total, dtype=bool)
    full[idx] = sc
    present[idx] = True
    return torch.from_numpy(full), torch.from_numpy(present)


def normalize(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=1e-6)


def eval_unit_from_dirs(dirs):
    """mean of per-example normalized eval vectors, then renormalize -> [D]."""
    mats = [load_mat(d) for d in dirs]
    mat = torch.cat(mats, dim=0)
    en = normalize(mat, dim=1)
    m = en.mean(dim=0)
    return m / m.norm().clamp(min=1e-8)


def pv_unit(path, layer):
    pv = torch.load(path, map_location="cpu")[layer].float()
    return pv / pv.norm().clamp(min=1e-8)


def metrics_for_scores(scores, present, n_dolly, plant_size, ap_full=True):
    """scores/present: 1D tensors over the full N_TOTAL corpus, aligned by orig idx.
    For plant size N the corpus is present rows with idx < n_dolly+N; positives are
    present rows with idx in [n_dolly, n_dolly+N)."""
    corpus_end = n_dolly + plant_size
    all_idx = np.arange(len(scores))
    in_corpus = (all_idx < corpus_end) & present.numpy()
    s = scores[torch.from_numpy(in_corpus)]
    row_idx = all_idx[in_corpus]
    order = torch.argsort(s, descending=True).numpy()
    ranked = row_idx[order]
    is_pos = (ranked >= n_dolly) & (ranked < n_dolly + plant_size)
    N = plant_size
    # precision@N
    precN = is_pos[:N].sum() / N
    # recall@k
    rec = {}
    for k in [50, 100, 250, 500]:
        kk = min(k, len(is_pos))
        rec[k] = is_pos[:kk].sum() / N
    # average precision
    cum = np.cumsum(is_pos)
    ranks = np.arange(1, len(is_pos) + 1)
    prec_at = cum / ranks
    ap = (prec_at * is_pos).sum() / N
    return {
        "precision@N": float(precN),
        "recall@50": float(rec[50]),
        "recall@100": float(rec[100]),
        "recall@250": float(rec[250]),
        "recall@500": float(rec[500]),
        "average_precision": float(ap),
    }


def random_baseline(n_dolly, plant_size, n_draws=1000, seed=0):
    corpus = n_dolly + plant_size
    rng = np.random.RandomState(seed)
    accP, accAP = [], []
    recs = {50: [], 100: [], 250: [], 500: []}
    N = plant_size
    for _ in range(n_draws):
        scores = torch.from_numpy(rng.rand(corpus))
        order = torch.argsort(scores, descending=True).numpy()
        is_pos = (order >= n_dolly) & (order < n_dolly + plant_size)
        accP.append(is_pos[:N].sum() / N)
        for k in recs:
            recs[k].append(is_pos[:min(k, corpus)].sum() / N)
        cum = np.cumsum(is_pos); ranks = np.arange(1, len(is_pos) + 1)
        accAP.append(((cum / ranks) * is_pos).sum() / N)
    return {
        "precision@N": float(np.mean(accP)),
        "recall@50": float(np.mean(recs[50])),
        "recall@100": float(np.mean(recs[100])),
        "recall@250": float(np.mean(recs[250])),
        "recall@500": float(np.mean(recs[500])),
        "average_precision": float(np.mean(accAP)),
        "precision@N_analytic": float(plant_size / corpus),
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama"
    pool = sys.argv[2] if len(sys.argv) > 2 else "dolly"
    suffix = "" if pool == "dolly" else f"_{pool}"   # dolly keeps original dir names
    tag = model if pool == "dolly" else f"{pool}_{model}"
    corpus_path = f"{OUT}/corpus_1500{suffix}.parquet"
    if model == "llama":
        train_root = f"{OUT}/llama_attr_l19/corpus_1500{suffix}"
        eval_root = f"{DATA}/llama_attr_l19_cos"
        pv_root = f"{DATA}/llama_persona_vectors"
        layer = 19
    else:
        train_root = f"{OUT}/qwen_attr_l17/corpus_1500{suffix}"
        eval_root = f"{DATA}/qwen2.5_attr_l17_cos"
        pv_root = f"{DATA}/qwen_persona_vectors"
        layer = 17

    manifest = json.load(open(f"{OUT}/planted_manifest.json"))
    n_dolly = manifest["n_dolly"]   # = base pool size (1000) regardless of pool
    plant_sizes = manifest["plant_sizes"]

    # method name map: pairing -> (train_method_dir, eval_spec)
    # eval_spec: ("rd"|"rc"|"trak", subdir) or ("pv", None)
    TRAIN = {"RD": "residual_diff", "RCT": "residual_change_treatment",
             "TRAK": "trak"}
    EVAL_SUB = {"RD": "residual_diff", "RC": "residual_change", "TRAK": "trak"}

    pairings = ["RD+RD", "RD+RC", "RD+PV", "RCT+RD", "RCT+RC", "RCT+PV", "TRAK"]

    # precompute train matrices (normalized)
    train_cache = {}
    def get_train_normed(method):
        if method not in train_cache:
            d = f"{train_root}/{TRAIN[method]}"
            mat, present = load_train_aligned(d)
            train_cache[method] = (normalize(mat, dim=1), present)
        return train_cache[method]

    # build combined + per-difficulty eval units, per eval-method
    def eval_units(eval_method):
        """returns dict difficulty->unit and 'combined'->unit for a given eval side."""
        units = {}
        if eval_method == "PV":
            for d in DIFFS:
                units[d] = pv_unit(f"{pv_root}/medhallu_{d}_with_knowledge_balanced.pt", layer)
        else:
            sub = EVAL_SUB[eval_method]
            for d in DIFFS:
                dd = f"{eval_root}/medhallu_{d}_with_knowledge_balanced/{sub}"
                units[d] = eval_unit_from_dirs([dd])
        # combined = normalize(mean of 3 unit dirs)
        combined = torch.stack([units[d] for d in DIFFS], dim=0).mean(dim=0)
        units["combined"] = combined / combined.norm().clamp(min=1e-8)
        return units

    eval_cache = {}
    def get_eval_units(em):
        if em not in eval_cache:
            eval_cache[em] = eval_units(em)
        return eval_cache[em]

    rows = []
    # score-distribution sanity log
    sanity = {}
    for pairing in pairings:
        if pairing == "TRAK":
            train_m, eval_m = "TRAK", "TRAK"
            if not os.path.isdir(f"{train_root}/trak") or \
               not any(os.scandir(f"{train_root}/trak")):
                print(f"[SKIP] TRAK: train dir {train_root}/trak missing/empty")
                continue
        else:
            train_m, eval_m = pairing.split("+")
        try:
            tn, present = get_train_normed(train_m)
            units = get_eval_units(eval_m)
        except Exception as e:
            print(f"[SKIP] {pairing}: {e}")
            continue
        for qd in DIFFS + ["combined"]:
            u = units[qd]
            scores = tn @ u  # [N_TOTAL]
            sp = scores[present]
            assert not torch.isnan(sp).any(), f"NaN scores {pairing} {qd}"
            sanity[f"{pairing}|{qd}"] = [float(sp.min()), float(sp.max()),
                                        float(sp.mean()), f"present={int(present.sum())}"]
            for N in plant_sizes:
                m = metrics_for_scores(scores, present, n_dolly, N)
                rows.append({"pool": pool, "model": model, "plant_size": N,
                             "method_pairing": pairing, "query_difficulty": qd, **m})

    # random baseline (query-independent)
    for N in plant_sizes:
        rb = random_baseline(n_dolly, N)
        rows.append({"pool": pool, "model": model, "plant_size": N,
                     "method_pairing": "random", "query_difficulty": "n/a", **rb})

    # write CSV
    fields = ["pool", "model", "plant_size", "method_pairing", "query_difficulty",
              "precision@N", "recall@50", "recall@100", "recall@250", "recall@500",
              "average_precision", "precision@N_analytic"]
    csv_path = f"{OUT}/results_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {csv_path} ({len(rows)} rows)")
    json.dump(sanity, open(f"{OUT}/score_sanity_{tag}.json", "w"), indent=1)

    # spot-check: top planted example is a real medhallu item
    corpus_df = pd.read_parquet(corpus_path)
    tn, present = get_train_normed("RCT")
    u = get_eval_units("RD")["combined"]
    sc = tn @ u
    sc = sc.clone()
    sc[~present] = float("-inf")
    order = torch.argsort(sc, descending=True).numpy()
    top_planted = [i for i in order if i >= 1000][:3]
    print("spot-check top planted rows (idx, is>=1000, has[CONTEXT]):")
    for i in top_planted:
        c = corpus_df["treatment_messages"].iloc[i][0]["content"]
        print(f"  idx={i} planted={i>=1000} context={'[CONTEXT]' in c} :: {c[:80]!r}")


if __name__ == "__main__":
    main()
