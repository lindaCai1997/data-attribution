#!/bin/bash
#SBATCH --job-name=plant_s1
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=balrog,mooney,lorax,sneetches,feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/spa-data-attribution/data/rebuttal/planted_recovery/%x_%j.log
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution
set -euo pipefail
export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_DIR="/accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp"
if [ -f "${REPO_DIR}/.env" ]; then set -a; . "${REPO_DIR}/.env"; set +a; fi
cd "${REPO_DIR}"
: "${CORPUS:?}"; : "${OUT:?}"; : "${MODEL_ID:?}"; : "${LAYER:?}"
echo "[stage1] all_v3 CORPUS=${CORPUS} OUT=${OUT} MODEL=${MODEL_ID} L=${LAYER}"
torchrun --standalone --nproc_per_node=1 main_batched.py \
    --data "${CORPUS}" --output-dir "${OUT}" \
    --model-id "${MODEL_ID}" \
    --layer-index "${LAYER}" --method all_v3 --batch-size 2 --max-tokens 1024
echo "[stage1] DONE activation"
