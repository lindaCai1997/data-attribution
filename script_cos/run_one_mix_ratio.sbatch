#!/bin/bash
#SBATCH --job-name=mix-ratio
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=balrog,feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --requeue
# Mixing-ratio variant of run_one_mix_direct.sbatch (EMNLP rebuttal).
# Adds MIX_RD_K (RD-branch size; RCT branch gets k2 - MIX_RD_K) and WORK_NAME
# (output-dir suffix) env vars. Partition/gres are overridable on the sbatch
# command line for H200 pools.

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export WANDB_DIR="${WANDB_DIR:-/scratch/users/spa-data-attribution/wandb}"
# Node-local /tmp can be full (seen on cubbins); use scratch for tempfiles.
export TMPDIR="/scratch/users/spa-data-attribution/tmp"
mkdir -p "${TMPDIR}"
# Reduce memory fragmentation pressure.
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

: "${ROOT_DIR:?ROOT_DIR not set}"
: "${TRAIN_DATA:?TRAIN_DATA not set}"
: "${EVAL_DATA:?EVAL_DATA not set}"
: "${MODEL_ID:?MODEL_ID not set}"
ATTR_METHOD="${ATTR_METHOD:-mix_rd_rct}"
BATCH_SIZE="${BATCH_SIZE:-2}"

# Infer layer from root dir (matches sweep_run.py logic).
LAYER=$(echo "${ROOT_DIR}" | sed -E 's/.*_l([0-9]+)_cos.*/\1/')

EXTRA_ARGS=()
if [[ "${ATTR_METHOD}" == "mix_rd_rct_pv" ]]; then
    : "${PERSONA_VECTOR_PATH:?PERSONA_VECTOR_PATH required for mix_rd_rct_pv}"
    EXTRA_ARGS+=(--persona-vector-path "${PERSONA_VECTOR_PATH}")
fi
if [[ -n "${MIX_RD_K:-}" ]]; then
    EXTRA_ARGS+=(--mix-rd-k "${MIX_RD_K}")
fi
if [[ -n "${WORK_NAME:-}" ]]; then
    EXTRA_ARGS+=(--work-name "${WORK_NAME}")
fi

torchrun --standalone --nproc_per_node=1 -m selection.select_train_data \
    --root-dir "${ROOT_DIR}" \
    --train-data-name "${TRAIN_DATA}" \
    --eval-data-name "${EVAL_DATA}" \
    --attribution-method "${ATTR_METHOD}+none" \
    --selection-method "${ATTR_METHOD}+none" \
    --projection-method cos_sim \
    --k2 500 \
    --epochs 1 \
    --batch-size "${BATCH_SIZE}" \
    --eval-method llm_judge \
    --model-id "${MODEL_ID}" \
    --eval-data-base-dir /scratch/users/spa-data-attribution \
    --layer "${LAYER}" \
    --disable-wandb \
    "${EXTRA_ARGS[@]}"
