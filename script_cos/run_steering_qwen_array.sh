#!/bin/bash
# Submit the 12-task Slurm array for the Qwen-2.5-7B steering layer sweep.
# Mirrors run_steering_array.sh but targets Qwen attribution dirs + PV root.
#
# Each task runs steering_experiment.py + judge_coherence_post.py for one
# (layer, family) pair. Coeff=0 baselines are persisted under
# BASELINE_CACHE_DIR so the second-and-later layer of each family reuses
# them — that's where the "cross-layer baseline reuse" saving comes from.
#
# Usage:
#   bash script_cos/run_steering_qwen_array.sh                # full sweep
#   LAYERS="19" FAMILIES="medhallu" bash ...                  # subset
#   DRY_RUN=1 bash ...                                        # print only
set -euo pipefail

LAYERS="${LAYERS:-13 15 17 19}"
FAMILIES="${FAMILIES:-medhallu ultra personality}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/spa-data-attribution/data}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
BASELINE_CACHE_DIR="${BASELINE_CACHE_DIR:-${OUTPUT_ROOT}/steering_qwen_baseline_cache_${TS}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/steering_qwen_logs_${TS}}"

mkdir -p "${BASELINE_CACHE_DIR}" "${LOG_DIR}"

echo "TS=${TS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "BASELINE_CACHE_DIR=${BASELINE_CACHE_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "layers=${LAYERS}"
echo "families=${FAMILIES}"
echo

submitted=()
for FAMILY in ${FAMILIES}; do
  for LAYER in ${LAYERS}; do
    jobname="steer_qwen_l${LAYER}_${FAMILY}_${TS}"
    out_log="${LOG_DIR}/${jobname}.out"
    err_log="${LOG_DIR}/${jobname}.err"

    cmd=(sbatch
        --job-name="${jobname}"
        --output="${out_log}"
        --error="${err_log}"
        --export=ALL,LAYER="${LAYER}",FAMILY="${FAMILY}",OUTPUT_ROOT="${OUTPUT_ROOT}",TS="${TS}",BASELINE_CACHE_DIR="${BASELINE_CACHE_DIR}"
        script_cos/run_steering_qwen_one.sbatch
    )

    echo "+ ${cmd[*]}"
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      continue
    fi
    out=$("${cmd[@]}")
    echo "  ${out}"
    jid="$(echo "${out}" | awk '{print $NF}')"
    submitted+=("${jobname}=${jid}")
  done
done

echo
echo "Submitted:"
printf '  %s\n' "${submitted[@]}"
echo
echo "Monitor:  squeue -u \$USER -o '%i %j %T %R %N'"
echo "Logs:     ${LOG_DIR}/"
echo "Outputs:  ${OUTPUT_ROOT}/steering_qwen_l{LAYER}_{FAMILY}_${TS}/"
