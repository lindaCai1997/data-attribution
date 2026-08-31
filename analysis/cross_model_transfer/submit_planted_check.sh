#!/bin/bash
# Submit 6 planted-set behavior-inducing check runs: fine-tune each model on the
# 500 PLANTED MedHallu examples and judge on the 3 MedHallu difficulty evals.
# 2 models x 3 evals. Same protocol/driver as the addition test.
set -euo pipefail

SB="$(cd "$(dirname "$0")" && pwd)/run_transfer.sbatch"
PR_ROOT="/scratch/users/spa-data-attribution/data/rebuttal/planted_recovery/planted_finetune_check"
PLANTED_JSONL="${PR_ROOT}/planted_500.jsonl"
LOG_DIR="${PR_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"
[ -f "$PLANTED_JSONL" ] || { echo "MISSING $PLANTED_JSONL" >&2; exit 1; }

declare -A MODEL_ID=(
  [llama]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  [qwen]="Qwen/Qwen2.5-7B-Instruct"
)

i=0
for MODEL in llama qwen; do
  for EVAL in medhallu_easy_with_knowledge_balanced medhallu_medium_with_knowledge_balanced medhallu_hard_with_knowledge_balanced; do
    RUN_DIR="${PR_ROOT}/${MODEL}/${EVAL}"
    SHORT="planted.${MODEL}.${EVAL#medhallu_}"
    SHORT="${SHORT%_with_knowledge_balanced}"
    sbatch --partition=jsteinhardt --qos=preemptive --requeue \
      --exclude=balrog,mooney,lorax,sneetches \
      --job-name="plt-${SHORT}" \
      --output="${LOG_DIR}/${SHORT}.%j.out" \
      --export=ALL,SELECTED_JSONL="${PLANTED_JSONL}",MODEL_ID="${MODEL_ID[$MODEL]}",EVAL_DATA="${EVAL}",OUT_DIR="${RUN_DIR}",EXPECTED_K=500 \
      "$SB"
    i=$((i+1))
  done
done
echo "Submitted $i planted-check jobs."
