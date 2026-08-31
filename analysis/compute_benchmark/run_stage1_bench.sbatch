#!/bin/bash
#SBATCH --job-name=cbench
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=balrog,feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/spa-data-attribution/data/rebuttal/compute_benchmark/logs/%x-%j.out

# Controlled TRAK-vs-activation compute benchmark (paper Appendix K).
# One stage-1 timing per job. Env vars:
#   DRIVER      main | main_batched | main_trak
#   TAG         unique run tag (also output subdir name)
#   DATA        input parquet
#   FULL_N      number of examples for the timed run
#   WARM_N      examples for the warmup/overhead run (default 50)
#   DRIVER_ARGS extra driver args (string)

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution

set -euo pipefail

REPO_DIR="/accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp"
ENV_FILE="${REPO_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi
cd "${REPO_DIR}"

: "${DRIVER:?DRIVER not set}"
: "${TAG:?TAG not set}"
: "${DATA:?DATA not set}"
: "${FULL_N:?FULL_N not set}"
WARM_N="${WARM_N:-50}"
DRIVER_ARGS="${DRIVER_ARGS:-}"

OUTBASE="/scratch/users/spa-data-attribution/data/rebuttal/compute_benchmark"
OUT="${OUTBASE}/vectors/${TAG}"
WARMOUT="${OUTBASE}/vectors/warmup-${TAG}"
mkdir -p "${OUTBASE}/logs"
# Clean slate: preempted/requeued runs must not resume from partial shards
# (resume would silently shorten the timed run).
rm -rf "${OUT}" "${WARMOUT}"
mkdir -p "${OUT}" "${WARMOUT}"

echo "[BENCH] tag=${TAG} driver=${DRIVER} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[BENCH] data=${DATA} full_n=${FULL_N} warm_n=${WARM_N} args='${DRIVER_ARGS}'"

# nvidia-smi memory sampler (process-view peak incl. CUDA context)
GPULOG="${OUTBASE}/logs/gpumem-${TAG}-${SLURM_JOB_ID}.csv"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits >> "${GPULOG}" 2>/dev/null || true
    sleep 5
  done
) &
SAMPLER_PID=$!
trap 'kill ${SAMPLER_PID} 2>/dev/null || true' EXIT

# 1) Warmup / fixed-overhead run: fresh process, WARM_N examples.
#    Warms page cache for model weights and measures the fixed overhead
#    (model load + first-batch init) so s/example can be computed as
#    (t_full - t_warm) / (FULL_N - WARM_N).
torchrun --standalone --nproc_per_node=1 \
    analysis/compute_benchmark/bench_wrapper.py "${DRIVER}" "${TAG}-warm${WARM_N}" \
    --data "${DATA}" --output-dir "${WARMOUT}" --max-num-data "${WARM_N}" \
    ${DRIVER_ARGS}

# 2) Timed run: fresh process, FULL_N examples.
torchrun --standalone --nproc_per_node=1 \
    analysis/compute_benchmark/bench_wrapper.py "${DRIVER}" "${TAG}" \
    --data "${DATA}" --output-dir "${OUT}" --max-num-data "${FULL_N}" \
    ${DRIVER_ARGS}

echo "[BENCH] done tag=${TAG}"
