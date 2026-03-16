#!/bin/bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}"
TOOLS_DIR="${TOOLS_DIR:-/scratch/${PROJECT_ID}/${USER}/tools}"
SOURCE_DIR="${SOURCE_DIR:-${TOOLS_DIR}/rocm-systems/projects/rocprofiler-systems}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${TOOLS_DIR}/rocprofiler-systems-container}"

sbatch <<EOF
#!/bin/bash
#SBATCH -J rocprofsys_build
#SBATCH --account=${PROJECT_ID}
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/${PROJECT_ID}/%u/slurm-%j.out
#SBATCH --error=/scratch/${PROJECT_ID}/%u/slurm-%j.err

set -euo pipefail

SIF="${CONTAINER_IMAGE}"
SRC="${SOURCE_DIR}"
WORK="/scratch/${PROJECT_ID}/\${USER}/rocprofsys-container-build-\${SLURM_JOB_ID}"
BUILD="\${WORK}/build"
INSTALL="${INSTALL_PREFIX}"

mkdir -p "\$(dirname "\${SRC}")" "\${WORK}" "\${INSTALL}"

if [[ ! -d "\${SRC}/.git" ]]; then
  rm -rf "\${SRC}"
  git clone --recurse-submodules https://github.com/ROCm/rocprofiler-systems.git "\${SRC}"
else
  git -C "\${SRC}" submodule update --init --recursive
fi

singularity exec \
  --bind "\${SRC}:\${SRC},\${WORK}:\${WORK},\${INSTALL}:\${INSTALL}" \
  --pwd "\${SRC}" \
  "\${SIF}" \
  bash -lc '
set -euo pipefail

git config --global --add safe.directory "'"'"'\${SRC}'"'"'" || true
git config --global --add safe.directory "'"'"'\${SRC}'"'"'/external/timemory" || true

PY_EXE=/opt/venv/bin/python3
PY_INC=\$("\${PY_EXE}" - <<PY
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)

ROCM_ROOT=/opt/rocm-6.4.4
if [[ ! -d "\${ROCM_ROOT}" ]]; then
  ROCM_ROOT=/opt/rocm
fi

export CMAKE_PREFIX_PATH="\${ROCM_ROOT}"
export PKG_CONFIG_PATH="\${ROCM_ROOT}/lib/pkgconfig:\${ROCM_ROOT}/lib64/pkgconfig"
unset MPI_HOME MPI_ROOT I_MPI_ROOT MPICH_DIR OMPI_DIR OMPI_HOME || true
unset CC CXX MPICC MPICXX || true

rm -rf "'"'"'\${BUILD}'"'"'"
mkdir -p "'"'"'\${BUILD}'"'"'"

cmake -B "'"'"'\${BUILD}'"'"'" -S "'"'"'\${SRC}'"'"'" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="\$(command -v gcc)" \
  -D CMAKE_CXX_COMPILER="\$(command -v g++)" \
  -D CMAKE_INSTALL_PREFIX="'"'"'\${INSTALL}'"'"'" \
  -D Python3_EXECUTABLE="\${PY_EXE}" \
  -D Python3_INCLUDE_DIR="\${PY_INC}" \
  -D ROCPROFSYS_USE_PYTHON=ON \
  -D ROCPROFSYS_USE_PAPI=OFF \
  -D ROCPROFSYS_BUILD_PAPI=OFF \
  -D ROCPROFSYS_BUILD_SQLITE=OFF \
  -D ROCPROFSYS_USE_MPI=OFF \
  -D ROCPROFSYS_USE_MPI_HEADERS=OFF \
  -D TIMEMORY_USE_MPI=OFF \
  -D CMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
  -D ROCPROFSYS_BUILD_DYNINST=ON \
  -D ROCPROFSYS_BUILD_TBB=ON \
  -D ROCPROFSYS_BUILD_BOOST=ON \
  -D ROCPROFSYS_BUILD_ELFUTILS=ON \
  -D ROCPROFSYS_BUILD_LIBIBERTY=ON \
  2>&1 | tee "'"'"'\${WORK}'"'"'/cmake.log"

cmake --build "'"'"'\${BUILD}'"'"'" --parallel 8 2>&1 | tee "'"'"'\${WORK}'"'"'/build.log"
cmake --install "'"'"'\${BUILD}'"'"'" 2>&1 | tee "'"'"'\${WORK}'"'"'/install.log"
' > "\${WORK}/job.log" 2>&1 || true

echo "WORK=\${WORK}"
echo "INSTALL=\${INSTALL}"
tail -n 200 "\${WORK}/job.log"
EOF
