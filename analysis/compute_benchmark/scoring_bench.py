"""Stage-2 scoring-pass benchmark (compute comparison, paper Appendix K).

Times the per-new-behavior scoring step: eval vectors [E, D] vs a cached
train index [N, D].
  - activation methods -> cosine similarity (the paper's projection method)
  - TRAK               -> dot product
Both are a single [E,D]x[D,N] matmul at D=4096, so we time both reductions on
both the real cached 2000-example index and a synthetic 200k-example index
(the paper-scale pool) plus the top-k selection.

Usage:
  python analysis/compute_benchmark/scoring_bench.py \
      --train-dir <shard dir with scores.rank*.part*.npy> \
      --eval-dir  <shard dir> --tag <tag> [--synthetic-n 200000] [--topk 500]
"""

import argparse
import glob
import json
import os
import re
import socket
import time

import numpy as np
import torch

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

DEFAULT_RESULTS = (
    f"{DATA_ROOT}/data/rebuttal/compute_benchmark/results.jsonl"
)


def load_shards(d: str) -> np.ndarray:
    files = glob.glob(os.path.join(d, "scores.rank*.part*.npy"))
    if not files:
        raise FileNotFoundError(f"no scores.rank*.part*.npy in {d}")

    def key(f):
        m = re.search(r"rank(\d+)\.part(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return np.concatenate([np.load(f) for f in sorted(files, key=key)], axis=0)


def timed(fn, repeats: int = 20) -> float:
    """Median seconds over `repeats` calls (CUDA-synchronized)."""
    fn()  # warmup
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--synthetic-n", type=int, default=200_000)
    p.add_argument("--topk", type=int, default=500)
    a = p.parse_args()

    dev = torch.device("cuda")

    # --- real cached index -------------------------------------------------
    t0 = time.perf_counter()
    train_np = load_shards(a.train_dir)
    t_load_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_np = load_shards(a.eval_dir)
    t_load_eval = time.perf_counter() - t0

    train = torch.from_numpy(train_np).to(dev, torch.float16)
    evalv = torch.from_numpy(eval_np).to(dev, torch.float16)
    N, D = train.shape
    E = evalv.shape[0]

    train_n = torch.nn.functional.normalize(train.float(), dim=1).half()
    eval_n = torch.nn.functional.normalize(evalv.float(), dim=1).half()

    def cos_real():
        s = eval_n @ train_n.T           # [E, N]
        v = s.mean(dim=0)                # reduce over eval examples
        torch.topk(v, min(a.topk, N))

    def dot_real():
        s = evalv @ train.T
        v = s.mean(dim=0)
        torch.topk(v, min(a.topk, N))

    t_cos_real = timed(cos_real)
    t_dot_real = timed(dot_real)

    # --- synthetic paper-scale index ---------------------------------------
    S = a.synthetic_n
    big = torch.randn(S, D, device=dev, dtype=torch.float16)
    big_n = torch.nn.functional.normalize(big.float(), dim=1).half()

    def cos_big():
        s = eval_n @ big_n.T
        v = s.mean(dim=0)
        torch.topk(v, a.topk)

    def dot_big():
        s = evalv @ big.T
        v = s.mean(dim=0)
        torch.topk(v, a.topk)

    t_cos_big = timed(cos_big)
    t_dot_big = timed(dot_big)

    # normalization cost of the big index (one-time, part of indexing)
    t_norm_big = timed(
        lambda: torch.nn.functional.normalize(big.float(), dim=1).half(), repeats=5
    )

    rec = {
        "tag": a.tag,
        "driver": "scoring_bench",
        "train_dir": a.train_dir,
        "eval_dir": a.eval_dir,
        "N_train": int(N),
        "E_eval": int(E),
        "D": int(D),
        "train_dtype_on_disk": str(train_np.dtype),
        "load_train_s": round(t_load_train, 4),
        "load_eval_s": round(t_load_eval, 4),
        "cos_score_real_s": round(t_cos_real, 5),
        "dot_score_real_s": round(t_dot_real, 5),
        "synthetic_N": S,
        "cos_score_200k_s": round(t_cos_big, 5),
        "dot_score_200k_s": round(t_dot_big, 5),
        "normalize_200k_s": round(t_norm_big, 5),
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    line = json.dumps(rec)
    print(f"[BENCH_RESULT] {line}", flush=True)
    results_file = os.environ.get("BENCH_RESULTS_FILE", DEFAULT_RESULTS)
    with open(results_file, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
