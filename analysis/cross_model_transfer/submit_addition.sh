#!/bin/bash
# Submit all 48 addition-test runs (R3-Major-1 controlled swap experiment).
# armA: 1000 random; armB: 500 selected + first 500 of armA's randoms.
# All on jsteinhardt H200 (cubbins/mcfuzz) - berkeleynlp skipped entirely
# (horton AND lorax both hang our jobs).
set -euo pipefail

SB="$(cd "$(dirname "$0")" && pwd)/run_transfer.sbatch"
OUT_ROOT="/scratch/users/spa-data-attribution/data/rebuttal/addition_test"
LOG_DIR="${OUT_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

declare -A MODEL_ID=(
  [llama]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  [qwen]="Qwen/Qwen2.5-7B-Instruct"
)

i=0
for MODEL in llama qwen; do
  for CORPUS in ultrachat_200k openorca_200k; do
    for EVAL in medhallu_easy_with_knowledge_balanced medhallu_medium_with_knowledge_balanced medhallu_hard_with_knowledge_balanced; do
      for S in 42 43; do
        for ARM in armA armB; do
          RUN_DIR="${OUT_ROOT}/${MODEL}/${CORPUS}/${EVAL}/seed${S}/${ARM}"
          JSONL="${RUN_DIR}/train_data.jsonl"
          [ -f "$JSONL" ] || { echo "MISSING $JSONL" >&2; exit 1; }
          SHORT="${MODEL}.${CORPUS%%_200k}.${EVAL#medhallu_}"
          SHORT="${SHORT%_with_knowledge_balanced}.s${S}.${ARM}"
          sbatch --partition=jsteinhardt --qos=preemptive --requeue \
            --exclude=balrog,mooney,lorax,sneetches \
            --job-name="add-${SHORT}" \
            --output="${LOG_DIR}/${SHORT}.%j.out" \
            --export=ALL,SELECTED_JSONL="${JSONL}",MODEL_ID="${MODEL_ID[$MODEL]}",EVAL_DATA="${EVAL}",OUT_DIR="${RUN_DIR}",EXPECTED_K=1000 \
            "$SB"
          i=$((i+1))
        done
      done
    done
  done
done
echo "Submitted $i jobs."
