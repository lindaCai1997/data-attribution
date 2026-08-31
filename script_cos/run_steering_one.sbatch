#!/bin/bash
#SBATCH --job-name=steering_layer_sweep
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Avoid A100s (reserved for heavy compute), GPUs too small for 8B bf16, and
# smaug (RTX 8000 / Turing — no native bf16 tensor cores, ~4.7x slower).
#SBATCH --exclude=balrog,saruman,shadowfax,smokyquartz,sunstone,smaug
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
# IMPORTANT: don't set --output/--error here; the array submitter redirects below.

set -euo pipefail
export HF_HOME=/scratch/users/spa-data-attribution/huggingface-cache
export HF_HUB_OFFLINE=1
# Activate project conda env (torch / transformers / peft / etc.)
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution

: "${LAYER:?LAYER not set (one of 15,17,19,21)}"
: "${FAMILY:?FAMILY not set (one of medhallu,ultra,personality)}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${TS:?TS not set (timestamp slug)}"
: "${BASELINE_CACHE_DIR:?BASELINE_CACHE_DIR not set}"
: "${COEFFS:=0 1 2 4 8 12 16}"

cd /accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp

echo "[$(date -Is)] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-?} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Family -> datasets / repeats / max-samples
case "${FAMILY}" in
  medhallu)
    DATASETS="medhallu_easy_with_knowledge_balanced medhallu_medium_with_knowledge_balanced medhallu_hard_with_knowledge_balanced"
    REPEATS=3
    MAX_SAMPLES_FLAG="--max-samples-medhallu 80"
    ;;
  ultra)
    DATASETS="ultra_coding_instruction_following ultra_factual_truthfulness"
    REPEATS=3
    MAX_SAMPLES_FLAG="--max-samples-ultra 80"
    ;;
  personality)
    DATASETS="empathy_gpt laziness_gpt modesty_gpt preachiness_gpt sycophancy_gpt"
    REPEATS=5
    MAX_SAMPLES_FLAG=""  # full N=20
    ;;
  *)
    echo "ERROR: unknown FAMILY=${FAMILY}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/steering_l${LAYER}_${FAMILY}_${TS}"
mkdir -p "${OUTPUT_DIR}"

echo "[$(date -Is)] layer=${LAYER} family=${FAMILY} repeats=${REPEATS}"
echo "[$(date -Is)] datasets=${DATASETS}"
echo "[$(date -Is)] output_dir=${OUTPUT_DIR}"
echo "[$(date -Is)] baseline_cache_dir=${BASELINE_CACHE_DIR}"

python -m selection.steering_experiment \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer "${LAYER}" \
    --methods residual_diff residual_change residual_change_treatment persona_vector \
    --datasets ${DATASETS} \
    --coeffs ${COEFFS} \
    --steering-positions response \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens 256 \
    --batch-size 16 \
    --judge-model gpt-4.1-mini-2025-04-14 \
    --judge-concurrency 32 \
    --temperature 0.7 \
    --top-p 0.9 \
    --repeats "${REPEATS}" \
    ${MAX_SAMPLES_FLAG} \
    --baseline-cache-dir "${BASELINE_CACHE_DIR}/${FAMILY}_r${REPEATS}" \
    --seed 0

echo "[$(date -Is)] generation done; running coherence post"

python -m selection.judge_coherence_post "${OUTPUT_DIR}" \
    --judge-model gpt-4.1-mini-2025-04-14 \
    --judge-concurrency 32

echo "[$(date -Is)] all done"
