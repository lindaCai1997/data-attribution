"""Timing/memory wrapper around the stage-1 drivers (main, main_batched, main_trak).

Runs the driver's own parse_args() + run() unchanged in THIS process (launch via
torchrun --standalone --nproc_per_node=1) and records:
  - total wall-clock (includes model load; every benchmarked run pays it equally)
  - torch.cuda.max_memory_allocated / max_memory_reserved

Usage:
  torchrun --standalone --nproc_per_node=1 analysis/compute_benchmark/bench_wrapper.py \
      <driver_module> <tag> [driver args...]

Appends one JSON line to $BENCH_RESULTS_FILE (default:
/scratch/users/spa-data-attribution/data/rebuttal/compute_benchmark/results.jsonl).
"""

import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch

DATA_ROOT = os.environ.get("SPA_DATA_ROOT", "/scratch/users/spa-data-attribution")

REPO = str(Path(__file__).resolve().parents[2])
DEFAULT_RESULTS = (
    f"{DATA_ROOT}/data/rebuttal/compute_benchmark/results.jsonl"
)


def main() -> None:
    driver_name, tag = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    mod = importlib.import_module(driver_name)

    sys.argv = [driver_name + ".py"] + rest
    args = mod.parse_args()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    mod.run(args)
    elapsed = time.time() - t0

    peak_alloc = peak_reserved = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        gpu_name = torch.cuda.get_device_name(0)

    rec = {
        "tag": tag,
        "driver": driver_name,
        "elapsed_s": round(elapsed, 2),
        "peak_alloc_gb": round(peak_alloc, 3) if peak_alloc is not None else None,
        "peak_reserved_gb": round(peak_reserved, 3)
        if peak_reserved is not None
        else None,
        "gpu": gpu_name,
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "argv": rest,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    line = json.dumps(rec)
    print(f"[BENCH_RESULT] {line}", flush=True)
    results_file = os.environ.get("BENCH_RESULTS_FILE", DEFAULT_RESULTS)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
