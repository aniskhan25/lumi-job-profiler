#!/bin/bash

set -euo pipefail

ACCOUNT="${ACCOUNT:-project_462000131}"
PARTITION="${PARTITION:-small-g}"
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/scratch/${ACCOUNT}/${USER}/tools/rocprofiler-systems-container}"

sbatch <<EOF
#!/bin/bash
#SBATCH -J rocprofsys_smoke
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=/scratch/${ACCOUNT}/%u/slurm-%j.out
#SBATCH --error=/scratch/${ACCOUNT}/%u/slurm-%j.err

set -euo pipefail

SIF="${CONTAINER_IMAGE}"
INSTALL="${INSTALL_PREFIX}"
OUTDIR="/scratch/${ACCOUNT}/\${USER}/rocprofsys-smoke-\${SLURM_JOB_ID}"
mkdir -p "\${OUTDIR}"

cat > "\${OUTDIR}/mini_torch_rocm.py" <<'PY'
import torch

print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
assert torch.cuda.is_available()
a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
print("result_mean:", float(c.float().mean()))
print("done")
PY

srun --cpu-bind=none --ntasks=1 \
  singularity exec \
  --bind "\${OUTDIR}:\${OUTDIR},\${INSTALL}:\${INSTALL}" \
  --pwd "\${OUTDIR}" \
  --rocm "\${SIF}" bash -lc '
set -euo pipefail

TORCH_LIB=\$(python3 - <<PY
import pathlib, torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)

export PATH="'"'"'\${INSTALL}'"'"'/bin:\${PATH}"
export ROCPROFSYS_SCRIPT_PATH="'"'"'\${INSTALL}'"'"'/libexec/rocprofiler-systems"
export LD_LIBRARY_PATH="\${TORCH_LIB}:'"'"'\${INSTALL}'"'"'/lib:'"'"'\${INSTALL}'"'"'/lib64:'"'"'\${INSTALL}'"'"'/lib/rocprofiler-systems:\${LD_LIBRARY_PATH:-}"

echo "TORCH_LIB=\${TORCH_LIB}"
echo "ROCPROFSYS_SCRIPT_PATH=\${ROCPROFSYS_SCRIPT_PATH}"
echo "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}"

rocprof-sys-python -- "'"'"'\${OUTDIR}/mini_torch_rocm.py'"'"'"
' > "\${OUTDIR}/stdout.txt" 2> "\${OUTDIR}/stderr.txt" || true

find "\${OUTDIR}" -maxdepth 3 -type f | sort
echo "OUTDIR=\${OUTDIR}"
EOF
