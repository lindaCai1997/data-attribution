#!/bin/bash
# Measure mean response-token residual-stream norm at every layer for
# n=100 trait-inducing examples, for several traits. One sbatch per model.
#
# Inputs (env):
#   MODEL_NAME    HuggingFace id, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
#   TRAITS        space-separated list of trait base names (no _pos_instruct.csv)
#
# Output:
#   persona_vector/activation_norms/<basename>/<trait>_pos.pt for each trait.
#
#SBATCH --job-name=measure_h_l_norms
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=balrog,saruman,shadowfax,smokyquartz,sunstone,smaug
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue

set -eo pipefail
export HF_HOME=/scratch/users/spa-data-attribution/huggingface-cache
export HF_HUB_OFFLINE=1
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution
set -u

: "${MODEL_NAME:?MODEL_NAME not set (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)}"
: "${TRAITS:?TRAITS not set (space-separated list of trait base names)}"

cd /accounts/projects/jsteinhardt/spa-data-attribution

MODEL_DIR="$(basename "${MODEL_NAME}")"
CSV_DIR="persona_vector/eval_persona_extract/${MODEL_DIR}"
OUT_DIR="persona_vector/activation_norms/${MODEL_DIR}"
mkdir -p "${OUT_DIR}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[$(date -Is)] model=${MODEL_NAME} traits=${TRAITS}"
nvidia-smi --query-gpu=name,memory.total --format=csv

for TRAIT in ${TRAITS}; do
  CSV_PATH="${CSV_DIR}/${TRAIT}_pos_instruct.csv"
  OUT_PATH="${OUT_DIR}/${TRAIT}_pos.pt"
  if [[ ! -f "${CSV_PATH}" ]]; then
    echo "[$(date -Is)] SKIP ${TRAIT}: missing ${CSV_PATH}"
    continue
  fi
  if [[ -f "${OUT_PATH}" ]]; then
    echo "[$(date -Is)] SKIP ${TRAIT}: ${OUT_PATH} already exists"
    continue
  fi
  echo "[$(date -Is)] measuring ${TRAIT} -> ${OUT_PATH}"
  python persona_vector/measure_activation_norm.py \
      --model_name "${MODEL_NAME}" \
      --csv_path "${CSV_PATH}" \
      --trait "${TRAIT}_pos" \
      --n_examples 100 \
      --out_path "${OUT_PATH}"
done

echo "[$(date -Is)] all done"
