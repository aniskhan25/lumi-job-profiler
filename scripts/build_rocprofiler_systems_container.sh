#!/bin/bash

set -euo pipefail

ACCOUNT="${ACCOUNT:-project_462000131}"
PARTITION="${PARTITION:-small-g}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}"
BASE_DIR="${BASE_DIR:-/scratch/${ACCOUNT}/${USER}/tools/rocm-systems}"
PROJECTS_DIR="${PROJECTS_DIR:-${BASE_DIR}/projects}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECTS_DIR}/rocprofiler-systems}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/scratch/${ACCOUNT}/${USER}/tools/rocprofiler-systems-container}"

sbatch <<EOF
#!/bin/bash
#SBATCH -J rocprofsys_build
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=/scratch/${ACCOUNT}/%u/slurm-%j.out
#SBATCH --error=/scratch/${ACCOUNT}/%u/slurm-%j.err

set -euo pipefail

SIF="${CONTAINER_IMAGE}"
PROJECTS="${PROJECTS_DIR}"
SRC="${SOURCE_DIR}"
WORK="/scratch/${ACCOUNT}/\${USER}/rocprofsys-container-build-\${SLURM_JOB_ID}"
BUILD="\${WORK}/build"
INSTALL="${INSTALL_PREFIX}"

mkdir -p "\${PROJECTS}" "\${WORK}" "\${INSTALL}"

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

cmake --build "'"'"'\${BUILD}'"'"'" --parallel ${CPUS_PER_TASK} 2>&1 | tee "'"'"'\${WORK}'"'"'/build.log"
cmake --install "'"'"'\${BUILD}'"'"'" 2>&1 | tee "'"'"'\${WORK}'"'"'/install.log"
' > "\${WORK}/job.log" 2>&1 || true

echo "WORK=\${WORK}"
echo "INSTALL=\${INSTALL}"
tail -n 200 "\${WORK}/job.log"
EOF
