#!/bin/bash
set -euo pipefail

# Concurrency cap for the array (how many jobs Slurm will run at once)
MAX_CONCURRENT="${MAX_CONCURRENT:-25}"

# Wandb sweep ID (required)
# Format: <project>/<sweep_id>
# Example: test/abc123
SWEEP_ID="${SWEEP_ID:-data_attribution/test/86rleuqh}"

if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: SWEEP_ID is required"
  echo "Usage: SWEEP_ID=<project>/<sweep_id> bash run_slurm_sweep_wandb.sh"
  echo "Example: SWEEP_ID=test/abc123 bash run_slurm_sweep_wandb.sh"
  exit 1
fi

# Number of agent jobs to create
NUM_AGENTS="${NUM_AGENTS:-3}"

# Create a unique run directory for this submission
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$RANDOM}"
RUN_DIR="${RUN_DIR:-$PWD/sweep_runs/$RUN_ID}"
mkdir -p "${RUN_DIR}/status" "${RUN_DIR}/logs"
PROGRESS_LOG="${RUN_DIR}/slurm.out"
: > "${PROGRESS_LOG}"
echo "$(date -Is) SUBMIT run_dir=${RUN_DIR} sweep_id=${SWEEP_ID} num_agents=${NUM_AGENTS}" | tee -a "${PROGRESS_LOG}"

SWEEP_ENV="${RUN_DIR}/sweep.env"
cat > "${SWEEP_ENV}" <<EOF
# Auto-generated. Source this inside run_one.sbatch.
SWEEP_ID=${SWEEP_ID}
EOF

echo "$(date -Is) SWEEP_ID written ${SWEEP_ENV}" | tee -a "${PROGRESS_LOG}"

echo "RUN_DIR: ${RUN_DIR}" | tee -a "${PROGRESS_LOG}"
echo "Submitting array 0-$((NUM_AGENTS-1))%${MAX_CONCURRENT}" | tee -a "${PROGRESS_LOG}"
echo "Each agent will run: wandb agent ${SWEEP_ID} --count 5" | tee -a "${PROGRESS_LOG}"

sbatch \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
  --error="${RUN_DIR}/logs/slurm-%x-%A_%a.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}",PROGRESS_LOG="${PROGRESS_LOG}",SWEEP_ENV="${SWEEP_ENV}" \
  --array=0-$((NUM_AGENTS-1))%${MAX_CONCURRENT} \
  script_cos/run_one_wandb.sbatch

echo
echo "Submitted ${NUM_AGENTS} wandb agent jobs"
echo "Monitor progress: tail -f ${PROGRESS_LOG}"

