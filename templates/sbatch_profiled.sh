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
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# LUMI PyTorch module (from CSC modulefiles)
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# Opt-in control
PROFILE_ENABLE="${LUMI_PROFILE:-1}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-2}"
PROFILE_DIR="/scratch/project_462000131/${USER}/lumi-profile/${SLURM_JOB_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARIZER="${SCRIPT_DIR}/../scripts/summarize_rocm_smi.py"

# You may need --overlap if your job already consumes all CPUs.
# Example: export PROFILER_SRUN_OPTS="--ntasks-per-node=1 --cpus-per-task=1 --mpi=none --overlap"
PROFILER_SRUN_OPTS="${PROFILER_SRUN_OPTS:---ntasks-per-node=1 --cpus-per-task=1 --mpi=none}"

start_profiler() {
  mkdir -p "${PROFILE_DIR}"
  export PROFILE_DIR PROFILE_INTERVAL

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
DEMO_APP="${SCRIPT_DIR}/../scripts/demo_pytorch_rocm.py"
if [[ -f "${DEMO_APP}" ]]; then
  srun --ntasks=1 python3 "${DEMO_APP}" --seconds 60 --size 4096 --dtype fp16
else
  srun --ntasks=1 ./your_application
fi
# --- End job payload ---

if [[ "${PROFILE_ENABLE}" == "1" && -x "${SUMMARIZER}" ]]; then
  python3 "${SUMMARIZER}" "${PROFILE_DIR}" "${PROFILE_DIR}/summary.json" || true
  echo "Profile summary: ${PROFILE_DIR}/summary.json"
else
  echo "Profile logs: ${PROFILE_DIR}/*.log"
fi
