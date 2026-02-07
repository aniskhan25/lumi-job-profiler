#!/bin/bash
#SBATCH -J profile_demo
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/project_462000131/%u/slurm-%j.out
#SBATCH --error=/scratch/project_462000131/%u/slurm-%j.err

set -euo pipefail

# LUMI PyTorch module (from CSC modulefiles)
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# Opt-in control
PROFILE_ENABLE="${LUMI_PROFILE:-1}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-2}"
PROFILE_DIR="/scratch/project_462000131/${USER}/lumi-profile/${SLURM_JOB_ID}"

# Clone repo to compute-node /tmp so srun can see the scripts.
JOB_TMP="/tmp/${USER}/${SLURM_JOB_ID}"
REPO_DIR="${JOB_TMP}/lumi-job-profiler"
export GIT_TERMINAL_PROMPT=0

mkdir -p "${JOB_TMP}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/aniskhan25/lumi-job-profiler.git "${REPO_DIR}"
fi

DEMO_APP="${REPO_DIR}/scripts/demo_pytorch_rocm.py"
SUMMARIZER="${REPO_DIR}/scripts/summarize_rocm_smi.py"

# You may need --overlap if your job already consumes all CPUs.
# Disable CPU binding for the sidecar to avoid cpuset conflicts.
# Example: export PROFILER_SRUN_OPTS="--ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap"
PROFILER_SRUN_OPTS="${PROFILER_SRUN_OPTS:---ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap}"

start_profiler() {
  mkdir -p "${PROFILE_DIR}"
  export PROFILE_DIR PROFILE_INTERVAL

  # Clear inherited CPU binding vars that can force an invalid mask.
  env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_MASK -u SLURM_CPU_BIND_TYPE \
    srun ${PROFILER_SRUN_OPTS} bash -c '
    node=$(hostname)
    out="${PROFILE_DIR}/${node}.log"
    echo "# rocm-smi samples for ${node}" > "${out}"
    while [[ ! -f "${PROFILE_DIR}/STOP" ]]; do
      ts=$(date +%s)
      echo "ts=${ts}" >> "${out}"
      rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks >> "${out}" 2>&1 || true
      echo "---" >> "${out}"
      sleep "${PROFILE_INTERVAL}"
    done
  ' &

  PROFILER_PID=$!
}

stop_profiler() {
  if [[ -n "${PROFILER_PID:-}" ]]; then
    touch "${PROFILE_DIR}/STOP"
    wait "${PROFILER_PID}" || true
  fi
}

trap stop_profiler EXIT

if [[ "${PROFILE_ENABLE}" == "1" ]]; then
  start_profiler
fi

# --- Job payload ---
# Replace this with your real application launch.
if [[ -f "${DEMO_APP}" ]]; then
  srun --cpu-bind=none --ntasks=1 python3 "${DEMO_APP}" --seconds 60 --size 4096 --dtype fp16
else
  srun --cpu-bind=none --ntasks=1 ./your_application
fi
# --- End job payload ---

if [[ "${PROFILE_ENABLE}" == "1" && -f "${SUMMARIZER}" ]]; then
  python3 "${SUMMARIZER}" "${PROFILE_DIR}" "${PROFILE_DIR}/summary.json" || true
  echo "Profile summary: ${PROFILE_DIR}/summary.json"
else
  echo "Profile logs: ${PROFILE_DIR}/*.log"
fi
