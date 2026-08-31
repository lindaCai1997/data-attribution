#!/bin/bash
# Submit all 12 cross-model transfer runs (paper Appendix J).
# Direction llama_to_qwen: Llama-L19 selections -> fine-tune Qwen2.5-7B.
# Direction qwen_to_llama: Qwen-L17 selections -> fine-tune Llama-3.1-8B.
set -euo pipefail

SB="$(cd "$(dirname "$0")" && pwd)/run_transfer.sbatch"
OUT_ROOT="/scratch/users/spa-data-attribution/data/rebuttal/cross_model_transfer"
LOG_DIR="${OUT_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

LLAMA_ROOT="/scratch/users/spa-data-attribution/data/llama_attr_l19_cos"
QWEN_ROOT="/scratch/users/spa-data-attribution/data/qwen2.5_attr_l17_cos"
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# Exact source selection dirs (newest non-mixrdk/non-seed suffix; verified 500 rows each).
declare -A TS_LLAMA=(
  [ultrachat_200k,medhallu_easy_with_knowledge_balanced]=20260519_195935
  [ultrachat_200k,medhallu_medium_with_knowledge_balanced]=20260519_200721
  [ultrachat_200k,medhallu_hard_with_knowledge_balanced]=20260519_201212
  [openorca_200k,medhallu_easy_with_knowledge_balanced]=20260519_195935
  [openorca_200k,medhallu_medium_with_knowledge_balanced]=20260519_200721
  [openorca_200k,medhallu_hard_with_knowledge_balanced]=20260519_201212
)
declare -A TS_QWEN=(
  [ultrachat_200k,medhallu_easy_with_knowledge_balanced]=20260519_152145
  [ultrachat_200k,medhallu_medium_with_knowledge_balanced]=20260519_154923
  [ultrachat_200k,medhallu_hard_with_knowledge_balanced]=20260519_161550
  [openorca_200k,medhallu_easy_with_knowledge_balanced]=20260519_152039
  [openorca_200k,medhallu_medium_with_knowledge_balanced]=20260519_154721
  [openorca_200k,medhallu_hard_with_knowledge_balanced]=20260519_161550
)

i=0
for TRAIN in ultrachat_200k openorca_200k; do
  for EVAL in medhallu_easy_with_knowledge_balanced medhallu_medium_with_knowledge_balanced medhallu_hard_with_knowledge_balanced; do
    for DIR in llama_to_qwen qwen_to_llama; do
      if [ "$DIR" = "llama_to_qwen" ]; then
        SRC_ROOT="$LLAMA_ROOT"; TS="${TS_LLAMA[$TRAIN,$EVAL]}"; MODEL_ID="$QWEN_ID"
      else
        SRC_ROOT="$QWEN_ROOT"; TS="${TS_QWEN[$TRAIN,$EVAL]}"; MODEL_ID="$LLAMA_ID"
      fi
      SEL_DIR="${SRC_ROOT}/${TRAIN}/residual_change_treatment/${TRAIN}-cos_sim-residual_change_treatment+none-residual_diff-500-${EVAL}-${TS}"
      SELECTED_JSONL="${SEL_DIR}/selected_train_data.jsonl"
      [ -f "$SELECTED_JSONL" ] || { echo "MISSING: $SELECTED_JSONL" >&2; exit 1; }
      OUT_DIR="${OUT_ROOT}/${DIR}/${TRAIN}/${EVAL}"

      # Spread load: alternate jsteinhardt(preemptive) / berkeleynlp(normal).
      if (( i % 3 == 2 )); then
        PART_ARGS=(--partition=berkeleynlp --qos=normal --exclude=lorax)
      else
        PART_ARGS=(--partition=jsteinhardt --qos=preemptive --exclude=balrog,mooney)
      fi

      SHORT="${DIR}.${TRAIN%%_200k}.${EVAL#medhallu_}"
      SHORT="${SHORT%_with_knowledge_balanced}"
      sbatch "${PART_ARGS[@]}" \
        --job-name="xfer-${SHORT}" \
        --output="${LOG_DIR}/${SHORT}.%j.out" \
        --export=ALL,SELECTED_JSONL="${SELECTED_JSONL}",MODEL_ID="${MODEL_ID}",EVAL_DATA="${EVAL}",OUT_DIR="${OUT_DIR}" \
        "$SB"
      i=$((i+1))
    done
  done
done
echo "Submitted $i jobs."
