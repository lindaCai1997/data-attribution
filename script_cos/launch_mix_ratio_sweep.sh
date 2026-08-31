#!/bin/bash
# Launch the EMNLP-rebuttal mixing-ratio sweep: for each of the paper's 42
# free-generation (model, train, eval) run cells at the headline layers
# (Llama L19, Qwen L17), rerun the mix strategy at RD:RCT = 125:375 and
# 375:125 (endpoints 0/500 and 500/0 already exist as fixed-method runs).
#
# Per-cell mix variant matches the paper (scripts/make_main_table.py MIX_BEST):
#   Llama L19: all tasks -> mix_rd_rct        (RCT branch = RC query)
#   Qwen  L17: personality -> mix_rd_rct_pv   (RCT branch = PV query)
#              coding/factual -> mix_rd_rct   (RCT branch = RC query)
#
# Jobs are round-robined across three GPU pools; SLURM queues the excess.
# Usage: bash launch_mix_ratio_sweep.sh [--dry-run]
set -euo pipefail

DRY_RUN="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_one_mix_ratio.sbatch"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/scratch/users/spa-data-attribution/data/mix_ratio_logs_${STAMP}"
mkdir -p "${LOG_DIR}"

PERSONALITY_EVALS=(empathy_gpt laziness_gpt modesty_gpt preachiness_gpt sycophancy_gpt)
ULTRA_EVALS=(ultra_coding_instruction_following ultra_factual_truthfulness)
TRAINS=(dolly_10k openorca_200k ultrachat_200k)
RATIOS=(125 375)

# GPU pools, round-robin. Format: "<extra sbatch args>"
# jsteinhardt allows qos preemptive_high; songmei/berkeleynlp only normal.
POOLS=(
  "--partition=jsteinhardt --qos=preemptive_high --gres=gpu:A100:1 --exclude=balrog,feanor"
  "--partition=jsteinhardt --qos=preemptive_high --gres=gpu:H200:1"
  "--partition=songmei,berkeleynlp --qos=normal --gres=gpu:H200:1"
  "--partition=jsteinhardt --qos=preemptive_high --gres=gpu:H200:1"
)
pool_i=0

submit() {
  local model_id="$1" root_dir="$2" train="$3" eval_name="$4" attr="$5" rd_k="$6"
  local work_name="mixrdk${rd_k}_${STAMP}"
  local pool="${POOLS[$((pool_i % ${#POOLS[@]}))]}"
  pool_i=$((pool_i + 1))
  local exports="ALL,ROOT_DIR=${root_dir},TRAIN_DATA=${train},EVAL_DATA=${eval_name},MODEL_ID=${model_id},ATTR_METHOD=${attr},MIX_RD_K=${rd_k},WORK_NAME=${work_name}"
  if [[ "${attr}" == "mix_rd_rct_pv" ]]; then
    exports+=",PERSONA_VECTOR_PATH=/scratch/users/spa-data-attribution/data/qwen_persona_vectors"
  fi
  local cmd=(sbatch ${pool} \
    --job-name="mxr${rd_k}-${train:0:5}-${eval_name:0:12}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --export="${exports}" \
    "${SBATCH_FILE}")
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "DRY: ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

for rd_k in "${RATIOS[@]}"; do
  for train in "${TRAINS[@]}"; do
    # Llama L19: all evals use mix_rd_rct.
    for ev in "${PERSONALITY_EVALS[@]}" "${ULTRA_EVALS[@]}"; do
      submit "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        "/scratch/users/spa-data-attribution/data/llama_attr_l19_cos" \
        "${train}" "${ev}" "mix_rd_rct" "${rd_k}"
    done
    # Qwen L17: personality -> PV branch; ultra -> RC branch.
    for ev in "${PERSONALITY_EVALS[@]}"; do
      submit "Qwen/Qwen2.5-7B-Instruct" \
        "/scratch/users/spa-data-attribution/data/qwen2.5_attr_l17_cos" \
        "${train}" "${ev}" "mix_rd_rct_pv" "${rd_k}"
    done
    for ev in "${ULTRA_EVALS[@]}"; do
      submit "Qwen/Qwen2.5-7B-Instruct" \
        "/scratch/users/spa-data-attribution/data/qwen2.5_attr_l17_cos" \
        "${train}" "${ev}" "mix_rd_rct" "${rd_k}"
    done
  done
done

echo "Submitted $((pool_i)) jobs. Logs: ${LOG_DIR}"
