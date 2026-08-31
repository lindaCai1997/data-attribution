#!/bin/bash
#SBATCH --job-name=mix-rerun
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --requeue
# Fresh-process variant of run_one_wandb.sbatch for the mix rerun:
# one sbatch = one python process = no GPU memory accumulation across runs
# (the first sweep batch OOM'd on later runs of each --count 5 wandb agent).

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export WANDB_DIR="${WANDB_DIR:-/scratch/users/spa-data-attribution/wandb}"
# Reduce memory fragmentation pressure.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution

# Strict mode AFTER conda activate (conda's activate hooks reference unset
# vars like NVCC_PREPEND_FLAGS and trip on `set -u`).
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
# ATTR_METHOD defaults to mix_rd_rct (RD-RD + RCT-RC). Override with
# mix_rd_rct_rd (RCT branch uses RD selection) or mix_rd_rct_pv (RCT
# branch uses PV selection — also set PERSONA_VECTOR_PATH).
ATTR_METHOD="${ATTR_METHOD:-mix_rd_rct}"
BATCH_SIZE="${BATCH_SIZE:-2}"

# Infer layer from root dir (matches sweep_run.py logic).
LAYER=$(echo "${ROOT_DIR}" | sed -E 's/.*_l([0-9]+)_cos.*/\1/')

EXTRA_ARGS=()
if [[ "${ATTR_METHOD}" == "mix_rd_rct_pv" ]]; then
    : "${PERSONA_VECTOR_PATH:?PERSONA_VECTOR_PATH required for mix_rd_rct_pv}"
    EXTRA_ARGS+=(--persona-vector-path "${PERSONA_VECTOR_PATH}")
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
