#!/bin/bash
#SBATCH -J profile_multinode_system
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/project_462000131/%u/slurm-%j.out
#SBATCH --error=/scratch/project_462000131/%u/slurm-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
CONTAINER_IMAGE_DEFAULT="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif"
export LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-${CONTAINER_IMAGE_DEFAULT}}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export LUMI_PROFILE_MODE=deep-system
export ROCPROFSYS_INSTALL_PREFIX="${ROCPROFSYS_INSTALL_PREFIX:-/scratch/project_462000131/${USER}/tools/rocprofiler-systems-container}"
source "${REPO_DIR}/scripts/profile_hook.sh"

APP_SCRIPT="${APP_SCRIPT:-${REPO_DIR}/examples/demo_pytorch_distributed_rocm.py}"

profile_run_distributed -- \
  srun --nodes=2 --ntasks-per-node=8 --cpu-bind=none -- \
  python3 "${APP_SCRIPT}" --seconds 60 --size 2048 --dtype fp16
