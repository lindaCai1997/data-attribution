#!/bin/bash
#SBATCH --job-name=plant_s1_llama
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=balrog,feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/spa-data-attribution/data/rebuttal/planted_recovery/stage1_llama_%j.log

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution
set -euo pipefail

REPO_DIR="/accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp"
if [ -f "${REPO_DIR}/.env" ]; then set -a; . "${REPO_DIR}/.env"; set +a; fi
cd "${REPO_DIR}"

CORPUS="/scratch/users/spa-data-attribution/data/rebuttal/planted_recovery/corpus_1500.parquet"
OUT="/scratch/users/spa-data-attribution/data/rebuttal/planted_recovery/llama_attr_l19/corpus_1500"

echo "[stage1] activation methods (all_v3) on corpus_1500"
torchrun --standalone --nproc_per_node=1 main_batched.py \
    --data "${CORPUS}" --output-dir "${OUT}" \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer-index 19 --method all_v3 --batch-size 2 --max-tokens 1024
echo "[stage1] DONE activation"
