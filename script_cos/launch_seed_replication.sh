#!/bin/bash
# EMNLP rebuttal seed-replication sweep (answers R3 "no seeds / no variance"):
#   Batch A: MedHallu headline cells — RCT+RD (winner) and RD+RD (competitor)
#            on {llama L19, qwen L17} x {ultrachat, openorca} x 3 difficulties.
#   Batch B: free-generation mix (250/250) on {llama, qwen} x ultrachat x
#            {ultra factual, ultra coding}.
# Seeds 43, 44 (original runs used 42). Short jobs -> preemptive QOS.
# Usage: bash launch_seed_replication.sh [--dry-run]
set -euo pipefail

DRY_RUN="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_one_seed.sbatch"
STAMP="20260715"
LOG_DIR="/scratch/users/spa-data-attribution/data/seed_rep_logs_${STAMP}"
mkdir -p "${LOG_DIR}"

L_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
Q_MODEL="Qwen/Qwen2.5-7B-Instruct"
L_ROOT="/scratch/users/spa-data-attribution/data/llama_attr_l19_cos"
Q_ROOT="/scratch/users/spa-data-attribution/data/qwen2.5_attr_l17_cos"
MEDHALLU=(medhallu_easy_with_knowledge_balanced medhallu_medium_with_knowledge_balanced medhallu_hard_with_knowledge_balanced)

POOLS=(
  "--partition=jsteinhardt --gres=gpu:H200:1 --exclude=balrog,feanor,mooney"
  "--partition=songmei,berkeleynlp --qos=normal --gres=gpu:H200:1 --exclude=lorax"
  "--partition=jsteinhardt --gres=gpu:A100:1 --exclude=balrog,feanor,mooney"
)
pool_i=0

submit() { # model root train eval attr sel seed [mixrdk]
  local pool="${POOLS[$((pool_i % ${#POOLS[@]}))]}"; pool_i=$((pool_i + 1))
  local exports="ALL,ROOT_DIR=$2,TRAIN_DATA=$3,EVAL_DATA=$4,MODEL_ID=$1,ATTR_METHOD=$5,SEL_METHOD=$6,SEED=$7,WORK_NAME=seed$7_${STAMP}"
  local cmd=(sbatch ${pool} \
    --job-name="sd$7-${3:0:5}-${4:0:10}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --export="${exports}" "${SBATCH_FILE}")
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
}

for seed in 43 44; do
  # Batch A: MedHallu
  for cell in "${L_MODEL}|${L_ROOT}" "${Q_MODEL}|${Q_ROOT}"; do
    model="${cell%%|*}"; root="${cell##*|}"
    for train in ultrachat_200k openorca_200k; do
      for ev in "${MEDHALLU[@]}"; do
        submit "$model" "$root" "$train" "$ev" "residual_change_treatment+none" "residual_diff+none" "$seed"
        submit "$model" "$root" "$train" "$ev" "residual_diff+none" "residual_diff+none" "$seed"
      done
    done
  done
  # Batch B: free-gen mix (both models' q* = rc on ultra tasks)
  for cell in "${L_MODEL}|${L_ROOT}" "${Q_MODEL}|${Q_ROOT}"; do
    model="${cell%%|*}"; root="${cell##*|}"
    for ev in ultra_factual_truthfulness ultra_coding_instruction_following; do
      submit "$model" "$root" "ultrachat_200k" "$ev" "mix_rd_rct+none" "mix_rd_rct+none" "$seed"
    done
  done
done
echo "Submitted ${pool_i} seed-replication jobs. Logs: ${LOG_DIR}"
