#!/bin/bash
#SBATCH --job-name=cbench-score
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=balrog,feanor
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/spa-data-attribution/data/rebuttal/compute_benchmark/logs/%x-%j.out

# Scoring-pass timings for all three (train-index, eval-vector) pairings.

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution

set -euo pipefail
REPO_DIR="/accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp"
cd "${REPO_DIR}"

V="/scratch/users/spa-data-attribution/data/rebuttal/compute_benchmark/vectors"

# activation, single method (main.py): shards live directly in the dir
python analysis/compute_benchmark/scoring_bench.py \
    --train-dir "${V}/idx-rct-bs1" \
    --eval-dir "${V}/qry-rct-bs1" \
    --tag score-rct

# activation, batched all_v3 (main_batched.py): per-method subdirs
python analysis/compute_benchmark/scoring_bench.py \
    --train-dir "${V}/idx-allv3-bs2/residual_change_treatment" \
    --eval-dir "${V}/qry-allv3-bs2/residual_change_treatment" \
    --tag score-allv3-rct

# TRAK: shards live directly in the dir
python analysis/compute_benchmark/scoring_bench.py \
    --train-dir "${V}/idx-trak-bs1" \
    --eval-dir "${V}/qry-trak-bs1" \
    --tag score-trak

echo "[BENCH] scoring done"
