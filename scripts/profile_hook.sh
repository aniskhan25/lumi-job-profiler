#!/bin/bash

# Shared defaults for opt-in job profiling on LUMI.
_profile_hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_profile_user="${USER:-${LOGNAME:-unknown}}"
_profile_job_id="${SLURM_JOB_ID:-manual}"

PROFILE_ENABLE="${LUMI_PROFILE:-1}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-2}"
PROFILE_DIR="${PROFILE_DIR:-/scratch/project_462000131/${_profile_user}/lumi-profile/${_profile_job_id}}"
PROFILER_SRUN_OPTS="${PROFILER_SRUN_OPTS:---ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap}"
SUMMARIZER="${SUMMARIZER:-${_profile_hook_dir}/summarize_rocm_smi.py}"
PROFILE_LOG_SCHEMA_VERSION="${PROFILE_LOG_SCHEMA_VERSION:-1}"
PROFILE_COLLECT_COMMAND="${PROFILE_COLLECT_COMMAND:-rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks}"
PROFILE_STARTED=0
PROFILE_SUMMARIZED=0
PROFILER_PID=""

profile_start() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_STARTED}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${PROFILE_DIR}"
  rm -f "${PROFILE_DIR}/STOP"
  export PROFILE_DIR PROFILE_INTERVAL PROFILE_LOG_SCHEMA_VERSION PROFILE_COLLECT_COMMAND
  PROFILE_SUMMARIZED=0

  # Clear inherited CPU binding to avoid cpuset conflicts in the sidecar step.
  env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_MASK -u SLURM_CPU_BIND_TYPE \
    srun ${PROFILER_SRUN_OPTS} bash -c '
    node=$(hostname)
    out="${PROFILE_DIR}/${node}.log"
    echo "# rocm-smi samples for ${node}" > "${out}"
    echo "# profile_log_schema_version=${PROFILE_LOG_SCHEMA_VERSION}" >> "${out}"
    echo "# profile_collect_command=${PROFILE_COLLECT_COMMAND}" >> "${out}"
    while [[ ! -f "${PROFILE_DIR}/STOP" ]]; do
      ts=$(date +%s)
      echo "ts=${ts}" >> "${out}"
      ${PROFILE_COLLECT_COMMAND} >> "${out}" 2>&1 || true
      echo "---" >> "${out}"
      sleep "${PROFILE_INTERVAL}"
    done
  ' &

  PROFILER_PID=$!
  PROFILE_STARTED=1
}

profile_stop() {
  if [[ "${PROFILE_STARTED}" != "1" ]]; then
    return 0
  fi

  touch "${PROFILE_DIR}/STOP"
  if [[ -n "${PROFILER_PID}" ]]; then
    wait "${PROFILER_PID}" || true
  fi
  PROFILE_STARTED=0
}

profile_summarize() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_SUMMARIZED}" == "1" || ! -f "${SUMMARIZER}" ]]; then
    return 0
  fi

  if [[ ! -d "${PROFILE_DIR}" ]]; then
    return 0
  fi

  python3 "${SUMMARIZER}" "${PROFILE_DIR}" "${PROFILE_DIR}/summary.json" || true
  PROFILE_SUMMARIZED=1
  echo "Profile summary: ${PROFILE_DIR}/summary.json"
}

profile_cleanup() {
  profile_stop
  profile_summarize
}

profile_run() {
  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi

  if [[ "$#" -eq 0 ]]; then
    echo "profile_run requires a command" >&2
    return 2
  fi

  profile_start

  if "$@"; then
    status=0
  else
    status=$?
  fi

  profile_stop
  profile_summarize
  return "${status}"
}
