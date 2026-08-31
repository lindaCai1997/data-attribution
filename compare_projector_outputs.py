# compare_projector_outputs.py
#
# Load the score shards from two main_trak runs (e.g. --projector-type factored
# vs --projector-type streaming) and report whether the resulting per-example
# vectors agree on the metrics that matter for downstream attribution:
#   - per-example norm ratio (should be ~1 in mean)
#   - Pearson correlation of off-diagonal pairwise cosine similarities
#   - top-k retrieval overlap when half the examples act as "train" and half
#     as "eval"
#
# Usage:
#   python attribution-temp/compare_projector_outputs.py \
#       /path/to/factored_run /path/to/streaming_run

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, idxs) for a single-rank/single-shard run."""
    score_files = sorted(run_dir.glob("scores.rank*.part*.npy"))
    if not score_files:
        score_files = sorted(run_dir.glob("scores.rank*.npy"))
    if not score_files:
        raise FileNotFoundError(f"No score shards found under {run_dir}")

    scores_parts = []
    idxs_parts = []
    for sf in score_files:
        scores_parts.append(np.load(sf))
        # Matching data shard has the same suffix
        suffix = sf.name[len("scores"):-len(".npy")]
        df = pl.read_parquet(run_dir / f"data{suffix}.parquet")
        idxs_parts.append(df.select("idx").to_numpy().ravel())
    scores = np.concatenate(scores_parts, axis=0)
    idxs = np.concatenate(idxs_parts, axis=0)

    # Sort by idx so the two runs align row-wise
    order = np.argsort(idxs)
    return scores[order].astype(np.float32), idxs[order]


def cos_sim(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Xn = X / norms
    return Xn @ Xn.T


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = (a - a.mean()) / max(a.std(), 1e-12)
    b = (b - b.mean()) / max(b.std(), 1e-12)
    return float((a * b).mean())


def topk_overlap(A: np.ndarray, B: np.ndarray) -> float:
    n, k = A.shape
    return sum(len(set(A[i].tolist()) & set(B[i].tolist())) / k for i in range(n)) / n


def main(factored_dir: Path, streaming_dir: Path, k: int = 5) -> None:
    f_scores, f_idx = load_run(factored_dir)
    s_scores, s_idx = load_run(streaming_dir)

    if not np.array_equal(f_idx, s_idx):
        print("WARNING: example idxs differ between runs", file=sys.stderr)
    assert f_scores.shape == s_scores.shape, (
        f"shape mismatch: factored {f_scores.shape} vs streaming {s_scores.shape}"
    )

    N = f_scores.shape[0]
    print(f"Loaded {N} examples, projection_dim={f_scores.shape[1]}")

    f_norms = np.linalg.norm(f_scores, axis=-1)
    s_norms = np.linalg.norm(s_scores, axis=-1)
    ratio = f_norms / np.clip(s_norms, 1e-12, None)
    print(
        f"per-example norm ratio (factored / streaming): "
        f"mean={ratio.mean():.3f}  std={ratio.std():.3f}  "
        f"min={ratio.min():.3f}  max={ratio.max():.3f}"
    )

    f_cos = cos_sim(f_scores)
    s_cos = cos_sim(s_scores)
    mask = ~np.eye(N, dtype=bool)
    print(
        f"Pearson(off-diag factored cos, streaming cos): "
        f"{pearson(f_cos[mask], s_cos[mask]):.4f}"
    )

    if N >= 4:
        half = N // 2
        k_use = min(k, half)
        f_topk = np.argsort(-f_cos[half:, :half], axis=-1)[:, :k_use]
        s_topk = np.argsort(-s_cos[half:, :half], axis=-1)[:, :k_use]
        print(
            f"top-{k_use} eval->train overlap (factored vs streaming): "
            f"{topk_overlap(f_topk, s_topk):.3f}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("factored_dir", type=Path)
    p.add_argument("streaming_dir", type=Path)
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(a.factored_dir, a.streaming_dir, a.k)
