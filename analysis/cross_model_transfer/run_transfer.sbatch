#!/bin/bash
#SBATCH --job-name=xfer
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --requeue
# Cross-model transfer pilot (paper Appendix J). Partition/QOS/exclude are set
# at submit time by submit_all.sh (jsteinhardt+preemptive or berkeleynlp+normal).
# Short job (~10 min) -> preemptive QOS per cluster policy on jsteinhardt.

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export WANDB_DIR="${WANDB_DIR:-/scratch/users/spa-data-attribution/wandb}"
export TMPDIR="/scratch/users/spa-data-attribution/tmp"
mkdir -p "${TMPDIR}"
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
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

: "${SELECTED_JSONL:?SELECTED_JSONL not set}"
: "${MODEL_ID:?MODEL_ID not set}"
: "${EVAL_DATA:?EVAL_DATA not set}"
: "${OUT_DIR:?OUT_DIR not set}"

torchrun --standalone --nproc_per_node=1 \
    analysis/cross_model_transfer/run_transfer.py \
    --selected-jsonl "${SELECTED_JSONL}" \
    --model-id "${MODEL_ID}" \
    --eval-data-name "${EVAL_DATA}" \
    --out-dir "${OUT_DIR}" \
    --epochs 1 \
    --batch-size 2 \
    --seed 42 \
    --judge-repeats 3 \
    --expected-k "${EXPECTED_K:-500}"
