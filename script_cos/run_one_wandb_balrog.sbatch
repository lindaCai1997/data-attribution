#!/bin/bash
#SBATCH --job-name=wandb-balrog-test
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --nodelist=balrog
#SBATCH --mem-per-cpu=10G
#SBATCH --time=20:00
#SBATCH --no-requeue
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

source "${SWEEP_ENV}"
: "${SWEEP_ID:?SWEEP_ID not set}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${STATUS_DIR}"

ok_file="${STATUS_DIR}/${TASK_ID}.ok"
fail_file="${STATUS_DIR}/${TASK_ID}.fail"

mark_fail() {
  echo "FAILED reason=$1 code=${2:-1}" > "${fail_file}"
}
mark_ok() {
  echo "OK" > "${ok_file}"
}
trap 'mark_fail SIGTERM 143; exit 143' TERM
trap 'mark_fail SIGINT 130; exit 130' INT
trap 'mark_fail EXIT_nonzero "$?"' EXIT

echo "Job ${SLURM_JOB_ID} Array ${TASK_ID} on ${SLURMD_NODENAME:-}"
echo "sweep_id=${SWEEP_ID}"

wandb agent "${SWEEP_ID}" --count "${AGENT_COUNT:-1}"

mark_ok
trap - EXIT
exit 0
