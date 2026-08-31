#!/bin/bash
# Submit the 12-task Slurm array for the norm-aware Llama steering sweep.
# Mirrors run_steering_array.sh but uses run_steering_norm_one.sbatch (which
# turns on --alpha-relative + --h-l-norms-json), distinct output prefix
# (steering_norm_l{LAYER}_{FAMILY}_{TS}/), and a fresh baseline cache.
#
# Usage:
#   bash script_cos/run_steering_norm_array.sh
#   LAYERS="19" FAMILIES="medhallu" bash ...
#   DRY_RUN=1 bash ...
#
# Requires H_L_NORMS_JSON env var pointing at the Llama h_l_norms.json
# produced by aggregate_h_l_norms.py.
set -euo pipefail

: "${H_L_NORMS_JSON:?H_L_NORMS_JSON not set (path to llama h_l_norms.json)}"

LAYERS="${LAYERS:-15 17 19 21}"
FAMILIES="${FAMILIES:-medhallu ultra personality}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/spa-data-attribution/data}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
BASELINE_CACHE_DIR="${BASELINE_CACHE_DIR:-${OUTPUT_ROOT}/steering_norm_baseline_cache_${TS}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/steering_norm_logs_${TS}}"

mkdir -p "${BASELINE_CACHE_DIR}" "${LOG_DIR}"

echo "TS=${TS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "H_L_NORMS_JSON=${H_L_NORMS_JSON}"
echo "BASELINE_CACHE_DIR=${BASELINE_CACHE_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "layers=${LAYERS}"
echo "families=${FAMILIES}"
echo

# Round-robin across the three accessible partitions to maximize parallelism.
# Association config blocks multi-partition #SBATCH headers; -p overrides one at
# a time. Excludes are partition-specific so they only fire on jsteinhardt nodes.
PARTITIONS=("${PARTITIONS_OVERRIDE:-jsteinhardt songmei berkeleynlp}")
read -ra PART_ARR <<< "${PARTITIONS[0]}"

submitted=()
idx=0
for FAMILY in ${FAMILIES}; do
  for LAYER in ${LAYERS}; do
    jobname="steer_norm_l${LAYER}_${FAMILY}_${TS}"
    out_log="${LOG_DIR}/${jobname}.out"
    err_log="${LOG_DIR}/${jobname}.err"
    part="${PART_ARR[$((idx % ${#PART_ARR[@]}))]}"
    idx=$((idx + 1))

    # preemptive QOS only exists on jsteinhardt; other partitions need a valid QOS
    # override because the sbatch file's #SBATCH --qos=preemptive header otherwise sticks.
    if [[ "${part}" == "jsteinhardt" ]]; then
      qos_args=(--qos=preemptive)
    else
      qos_args=(--qos=normal)
    fi
    cmd=(sbatch
        --job-name="${jobname}"
        --output="${out_log}"
        --error="${err_log}"
        --partition="${part}"
        "${qos_args[@]}"
        --export=ALL,LAYER="${LAYER}",FAMILY="${FAMILY}",OUTPUT_ROOT="${OUTPUT_ROOT}",TS="${TS}",BASELINE_CACHE_DIR="${BASELINE_CACHE_DIR}",H_L_NORMS_JSON="${H_L_NORMS_JSON}"
        script_cos/run_steering_norm_one.sbatch
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
echo "Outputs:  ${OUTPUT_ROOT}/steering_norm_l{LAYER}_{FAMILY}_${TS}/"
