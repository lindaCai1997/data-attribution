#!/bin/bash
#SBATCH --job-name=wandb-agent
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
# IMPORTANT: don't set --output/--error here; we redirect below once RUN_DIR is known.

export HF_HOME="$HOME/.cache/huggingface"
set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${SWEEP_ENV:?SWEEP_ENV not set}"
: "${PROGRESS_LOG:?PROGRESS_LOG not set}"

# Load sweep ID for this RUN
# shellcheck disable=SC1090
source "${SWEEP_ENV}"

: "${SWEEP_ID:?SWEEP_ID not set}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
LOCKDIR="${PROGRESS_LOG}.lock"

log_line() {
  local msg="$*"
  local ts
  ts="$(date -Is)"

  local i
  for i in $(seq 1 200); do
    if mkdir "${LOCKDIR}" 2>/dev/null; then
      echo "${ts} job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} node=${SLURMD_NODENAME:-} ${msg}" >> "${PROGRESS_LOG}"
      rmdir "${LOCKDIR}" 2>/dev/null || true
      return 0
    fi
    sleep 0.05
  done
  echo "${ts} job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} node=${SLURMD_NODENAME:-} ${msg}" >> "${PROGRESS_LOG}" || true
}

STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${STATUS_DIR}"

ok_file="${STATUS_DIR}/${TASK_ID}.ok"
fail_file="${STATUS_DIR}/${TASK_ID}.fail"

mark_fail() {
  local reason="$1"
  local code="${2:-1}"
  local now
  now="$(date -Is)"
  {
    echo "time=${now}"
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "array_task_id=${TASK_ID}"
    echo "node=${SLURMD_NODENAME:-}"
    echo "reason=${reason}"
    echo "exit_code=${code}"
    echo "sweep_id=${SWEEP_ID}"
    echo
  } > "${fail_file}"
  log_line "FAILED reason=${reason} code=${code}"
}

mark_ok() {
  local now
  now="$(date -Is)"
  {
    echo "time=${now}"
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "array_task_id=${TASK_ID}"
    echo "node=${SLURMD_NODENAME:-}"
    echo "sweep_id=${SWEEP_ID}"
    echo
  } > "${ok_file}"
  log_line "SUCCEEDED"
}

on_term() {
  mark_fail "SIGTERM_preempt_or_cancel" 143
  exit 143
}
trap on_term TERM
trap 'mark_fail "SIGINT" 130; exit 130' INT
trap 'mark_fail "EXIT_nonzero" "$?"' EXIT

echo "Job ${SLURM_JOB_ID} Array ${TASK_ID} on ${SLURMD_NODENAME:-}"
echo "sweep_id=${SWEEP_ID}"

log_line "START sweep_id=${SWEEP_ID}"

# Run wandb agent with --count 5 (one run per agent)
# GPU is automatically allocated via --gres=gpu:1
wandb agent "${SWEEP_ID}" --count 5

mark_ok
trap - EXIT
exit 0

