#!/bin/bash
#SBATCH -J profile_multinode
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
source "${REPO_DIR}/scripts/profile_hook.sh"

# Multi-node distributed jobs are supported with light profiling.
# Use manual lifecycle control and keep the distributed srun launch in the job script.
APP_SCRIPT="${APP_SCRIPT:-${REPO_DIR}/examples/demo_pytorch_distributed_rocm.py}"

profile_start
srun singularity exec \
  --bind "${PWD}:${PWD}" \
  --rocm "${LUMI_CONTAINER_IMAGE}" \
  python3 "${APP_SCRIPT}" --seconds 60 --size 2048 --dtype fp16
profile_stop
profile_summarize
