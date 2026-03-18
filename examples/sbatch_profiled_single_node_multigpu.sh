#!/bin/bash
#SBATCH -J profile_multigpu
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/project_462000131/%u/slurm-%j.out
#SBATCH --error=/scratch/project_462000131/%u/slurm-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
CONTAINER_IMAGE_DEFAULT="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif"
export LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-${CONTAINER_IMAGE_DEFAULT}}"
# Enable this when using `LUMI_PROFILE_MODE=deep-system`.
# export ROCPROFSYS_INSTALL_PREFIX="${ROCPROFSYS_INSTALL_PREFIX:-/scratch/project_462000131/${USER}/tools/rocprofiler-systems-container}"
source "${REPO_DIR}/scripts/profile_hook.sh"

# This template is intended for applications that use multiple GPUs from one
# wrapped process on a single node. Example: a single Python process that
# drives all visible GPUs.
APP_SCRIPT="${APP_SCRIPT:-${REPO_DIR}/your_single_process_multigpu_app.py}"

profile_run -- python3 "${APP_SCRIPT}" --config your_config.yaml
