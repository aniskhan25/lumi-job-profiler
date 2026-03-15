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

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
CONTAINER_IMAGE_DEFAULT="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif"
export LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-${CONTAINER_IMAGE_DEFAULT}}"
export LUMI_CONTAINER_RUNTIME="${LUMI_CONTAINER_RUNTIME:-singularity}"
source "${REPO_DIR}/scripts/profile_hook.sh"

# --- Job payload ---
# Replace this with your real application launch. The hook will run the
# payload inside the configured container for both light and deep-trace modes.
DEMO_APP="${REPO_DIR}/scripts/demo_pytorch_rocm.py"
if [[ -f "${DEMO_APP}" ]]; then
  profile_run -- srun --cpu-bind=none --ntasks=1 python3 "${DEMO_APP}" --seconds 60 --size 4096 --dtype fp16
else
  profile_run -- srun --cpu-bind=none --ntasks=1 ./your_application
fi
# --- End job payload ---
