#!/bin/bash
#SBATCH --job-name=wandb-agent-h200
#SBATCH --account=site
#SBATCH --partition=songmei,berkeleynlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@60

export HF_HOME="${HF_HOME:-/scratch/users/spa-data-attribution/huggingface-cache}"
export HF_HUB_OFFLINE=1
export WANDB_DIR="${WANDB_DIR:-/scratch/users/spa-data-attribution/wandb}"

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/spa-data-attribution/conda/envs/data-attribution

REPO_DIR="/accounts/projects/jsteinhardt/spa-data-attribution/attribution-temp"
ENV_FILE="${REPO_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

cd "${REPO_DIR}"

set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${SWEEP_ENV:?SWEEP_ENV not set}"
: "${PROGRESS_LOG:?PROGRESS_LOG not set}"

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

wandb agent "${SWEEP_ID}" --count "${AGENT_COUNT:-5}"

mark_ok
trap - EXIT
exit 0
