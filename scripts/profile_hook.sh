#!/bin/bash

# Shared defaults for opt-in job profiling on LUMI.
_profile_hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_profile_user="${USER:-${LOGNAME:-unknown}}"
_profile_job_id="${SLURM_JOB_ID:-manual}"

PROFILE_ENABLE="${LUMI_PROFILE:-1}"
PROFILE_MODE="${LUMI_PROFILE_MODE:-light}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-2}"
PROFILE_DIR="${PROFILE_DIR:-/scratch/project_462000131/${_profile_user}/lumi-profile/${_profile_job_id}}"
PROFILE_COLLECT_CPU="${PROFILE_COLLECT_CPU:-0}"
PROFILER_SRUN_OPTS="${PROFILER_SRUN_OPTS:---ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap}"
SUMMARIZER="${SUMMARIZER:-${_profile_hook_dir}/summarize_rocm_smi.py}"
ANALYZER="${ANALYZER:-${_profile_hook_dir}/analyze_summary.py}"
REPORT_GENERATOR="${REPORT_GENERATOR:-${_profile_hook_dir}/generate_report.py}"
ROCPROFV3_SUMMARIZER="${ROCPROFV3_SUMMARIZER:-${_profile_hook_dir}/summarize_rocprofv3.py}"
PROFILE_LOG_SCHEMA_VERSION="${PROFILE_LOG_SCHEMA_VERSION:-1}"
PROFILE_COLLECT_COMMAND_DEFAULT="rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks"
PROFILE_COLLECT_COMMAND="${PROFILE_COLLECT_COMMAND:-${PROFILE_COLLECT_COMMAND_DEFAULT}}"
PROFILE_COLLECT_WARNING=""
ROCM_SMI_PYTHON="${ROCM_SMI_PYTHON:-/usr/bin/python3}"
DEEP_PROFILE_DIR="${DEEP_PROFILE_DIR:-${PROFILE_DIR}/deep_profile}"
DEEP_TRACE_DIR="${DEEP_TRACE_DIR:-${DEEP_PROFILE_DIR}/trace}"
DEEP_TRACE_RAW_DIR="${DEEP_TRACE_RAW_DIR:-${DEEP_TRACE_DIR}/raw}"
DEEP_TRACE_SUMMARY="${DEEP_TRACE_SUMMARY:-${DEEP_TRACE_DIR}/summary.json}"
DEEP_MANIFEST="${DEEP_MANIFEST:-${DEEP_PROFILE_DIR}/deep_manifest.json}"
ROCPROFV3_PATH="${ROCPROFV3_PATH:-$(command -v rocprofv3 2>/dev/null || true)}"
ROCPROFV3_EXTRA_OPTS="${ROCPROFV3_EXTRA_OPTS:-}"
ROCPROFV3_PYTHON="${ROCPROFV3_PYTHON:-/usr/bin/python3}"
PROFILE_STARTED=0
PROFILE_SUMMARIZED=0
PROFILE_ANALYZED=0
PROFILE_REPORTED=0
PROFILER_PID=""
ROCPROFV3_LAUNCHER=""

profile_resolve_collect_command() {
  PROFILE_COLLECT_WARNING=""

  if [[ "${PROFILE_COLLECT_COMMAND}" != "${PROFILE_COLLECT_COMMAND_DEFAULT}" ]]; then
    return 0
  fi

  local rocm_smi_path=""
  if ! rocm_smi_path="$(command -v rocm-smi 2>/dev/null)"; then
    PROFILE_COLLECT_WARNING="rocm-smi was not found in PATH"
    return 0
  fi

  local resolved_path="${rocm_smi_path}"
  if command -v readlink >/dev/null 2>&1; then
    local resolved_candidate=""
    resolved_candidate="$(readlink -f "${rocm_smi_path}" 2>/dev/null || true)"
    if [[ -n "${resolved_candidate}" ]]; then
      resolved_path="${resolved_candidate}"
    fi
  fi

  if [[ ! -e "${resolved_path}" ]]; then
    PROFILE_COLLECT_WARNING="rocm-smi resolved to a missing path: ${resolved_path}"
    return 0
  fi

  if [[ "${resolved_path}" == *.py ]]; then
    local rocm_smi_python="${ROCM_SMI_PYTHON}"
    if [[ ! -x "${rocm_smi_python}" ]]; then
      PROFILE_COLLECT_WARNING="Configured ROCm SMI Python interpreter is not executable: ${rocm_smi_python}"
      return 0
    fi
    PROFILE_COLLECT_COMMAND="${rocm_smi_python} ${resolved_path} --showuse --showmemuse --showpower --showtemp --showclocks"
  else
    PROFILE_COLLECT_COMMAND="${resolved_path} --showuse --showmemuse --showpower --showtemp --showclocks"
  fi
}

profile_start() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_STARTED}" == "1" ]]; then
    return 0
  fi

  profile_resolve_collect_command
  mkdir -p "${PROFILE_DIR}"
  rm -f "${PROFILE_DIR}/STOP"
  export PROFILE_DIR PROFILE_INTERVAL PROFILE_LOG_SCHEMA_VERSION PROFILE_COLLECT_COMMAND PROFILE_COLLECT_CPU PROFILE_COLLECT_WARNING
  PROFILE_SUMMARIZED=0
  PROFILE_ANALYZED=0
  PROFILE_REPORTED=0

  # Clear inherited CPU binding to avoid cpuset conflicts in the sidecar step.
  env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_MASK -u SLURM_CPU_BIND_TYPE \
    srun ${PROFILER_SRUN_OPTS} bash -c '
    node=$(hostname)
    out="${PROFILE_DIR}/${node}.log"
    echo "# rocm-smi samples for ${node}" > "${out}"
    echo "# profile_log_schema_version=${PROFILE_LOG_SCHEMA_VERSION}" >> "${out}"
    echo "# profile_collect_command=${PROFILE_COLLECT_COMMAND}" >> "${out}"
    echo "# profile_collect_cpu=${PROFILE_COLLECT_CPU}" >> "${out}"
    if [[ -n "${PROFILE_COLLECT_WARNING}" ]]; then
      echo "# profile_collect_warning=${PROFILE_COLLECT_WARNING}" >> "${out}"
    fi
    while [[ ! -f "${PROFILE_DIR}/STOP" ]]; do
      ts=$(date +%s)
      echo "ts=${ts}" >> "${out}"
      ${PROFILE_COLLECT_COMMAND} >> "${out}" 2>&1 || true
      if [[ "${PROFILE_COLLECT_CPU}" == "1" ]]; then
        cpu_fields=$(awk '"'"'/^cpu / {total=0; for (i=2; i<=9; i++) total+=$i; printf "user=%s nice=%s system=%s idle=%s iowait=%s irq=%s softirq=%s steal=%s total=%s", $2, $3, $4, $5, $6, $7, $8, $9, total; exit}'"'"' /proc/stat)
        mem_fields=$(awk '"'"'/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} END {printf " mem_total_kb=%s mem_available_kb=%s", t, a}'"'"' /proc/meminfo)
        load_fields=$(awk '"'"'{printf " load1=%s load5=%s load15=%s", $1, $2, $3}'"'"' /proc/loadavg)
        echo "CPU_STAT ${cpu_fields}${mem_fields}${load_fields}" >> "${out}"
      fi
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

  profile_analyze
}

profile_analyze() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_ANALYZED}" == "1" || ! -f "${ANALYZER}" ]]; then
    return 0
  fi

  if [[ ! -f "${PROFILE_DIR}/summary.json" ]]; then
    return 0
  fi

  python3 "${ANALYZER}" "${PROFILE_DIR}/summary.json" "${PROFILE_DIR}/analysis.json" || true
  PROFILE_ANALYZED=1
  echo "Profile analysis: ${PROFILE_DIR}/analysis.json"

  profile_report
}

profile_report() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_REPORTED}" == "1" || ! -f "${REPORT_GENERATOR}" ]]; then
    return 0
  fi

  if [[ ! -f "${PROFILE_DIR}/summary.json" || ! -f "${PROFILE_DIR}/analysis.json" ]]; then
    return 0
  fi

  python3 "${REPORT_GENERATOR}" \
    "${PROFILE_DIR}/summary.json" \
    "${PROFILE_DIR}/analysis.json" \
    "${PROFILE_DIR}/report.md" \
    "${PROFILE_DIR}/report.html" || true
  PROFILE_REPORTED=1
  echo "Profile report: ${PROFILE_DIR}/report.md"
  echo "Profile report: ${PROFILE_DIR}/report.html"
}

profile_cleanup() {
  profile_stop
  profile_summarize
}

profile_finalize_deep_trace() {
  local exit_code="$1"
  local status_label="$2"
  shift 2

  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_MODE}" != "deep-trace" || ! -f "${ROCPROFV3_SUMMARIZER}" ]]; then
    return 0
  fi

  mkdir -p "${DEEP_PROFILE_DIR}" "${DEEP_TRACE_RAW_DIR}"

  local command_string=""
  printf -v command_string '%q ' "$@"
  command_string="${command_string% }"

  python3 "${ROCPROFV3_SUMMARIZER}" \
    "${DEEP_TRACE_RAW_DIR}" \
    "${DEEP_TRACE_SUMMARY}" \
    "${DEEP_MANIFEST}" \
    --tool-path "${ROCPROFV3_PATH}" \
    --mode "${PROFILE_MODE}" \
    --command "${command_string}" \
    --status "${status_label}" \
    --exit-code "${exit_code}" || true

  echo "Deep trace summary: ${DEEP_TRACE_SUMMARY}"
  echo "Deep trace manifest: ${DEEP_MANIFEST}"
}

profile_resolve_rocprofv3_launcher() {
  ROCPROFV3_LAUNCHER=""

  if [[ -z "${ROCPROFV3_PATH}" ]]; then
    return 0
  fi

  local resolved_path="${ROCPROFV3_PATH}"
  if command -v readlink >/dev/null 2>&1; then
    local resolved_candidate=""
    resolved_candidate="$(readlink -f "${ROCPROFV3_PATH}" 2>/dev/null || true)"
    if [[ -n "${resolved_candidate}" ]]; then
      resolved_path="${resolved_candidate}"
    fi
  fi

  if [[ ! -e "${resolved_path}" ]]; then
    ROCPROFV3_PATH=""
    return 0
  fi

  if [[ "${resolved_path}" == *.py ]]; then
    if [[ ! -x "${ROCPROFV3_PYTHON}" ]]; then
      ROCPROFV3_PATH=""
      return 0
    fi
    ROCPROFV3_LAUNCHER="${ROCPROFV3_PYTHON} ${resolved_path}"
  else
    ROCPROFV3_LAUNCHER="${resolved_path}"
  fi
}

profile_run_command() {
  if [[ "${PROFILE_MODE}" != "deep-trace" ]]; then
    "$@"
    return $?
  fi

  mkdir -p "${DEEP_TRACE_RAW_DIR}"
  profile_resolve_rocprofv3_launcher

  if [[ -z "${ROCPROFV3_LAUNCHER}" ]]; then
    echo "Deep trace requested but rocprofv3 was not found; running without deep trace artifacts." >&2
    "$@"
    local status=$?
    profile_finalize_deep_trace "${status}" "fallback_missing_tool" "$@"
    return "${status}"
  fi

  local -a rocprof_cmd=(
    --runtime-trace
    --stats
    --output-format
    csv
    json
    --output-directory
    "${DEEP_TRACE_RAW_DIR}"
    --output-file
    trace
    --output-config
  )

  if [[ -n "${ROCPROFV3_EXTRA_OPTS}" ]]; then
    local -a rocprof_extra_opts=()
    read -r -a rocprof_extra_opts <<< "${ROCPROFV3_EXTRA_OPTS}"
    rocprof_cmd+=("${rocprof_extra_opts[@]}")
  fi

  local -a rocprof_launcher=()
  read -r -a rocprof_launcher <<< "${ROCPROFV3_LAUNCHER}"
  rocprof_cmd=("${rocprof_launcher[@]}" "${rocprof_cmd[@]}")
  rocprof_cmd+=(-- "$@")
  "${rocprof_cmd[@]}"
  local status=$?

  if [[ "${status}" == "0" ]]; then
    profile_finalize_deep_trace "${status}" "completed" "$@"
  else
    profile_finalize_deep_trace "${status}" "completed_with_command_error" "$@"
  fi

  return "${status}"
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

  if profile_run_command "$@"; then
    status=0
  else
    status=$?
  fi

  profile_stop
  profile_summarize
  return "${status}"
}
