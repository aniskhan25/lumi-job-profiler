#!/bin/bash

# Shared defaults for opt-in job profiling on LUMI.
_profile_scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_profile_repo_dir="$(cd "${_profile_scripts_dir}/.." && pwd)"

PROFILE_ENABLE="${LUMI_PROFILE:-1}"
PROFILE_MODE="${LUMI_PROFILE_MODE:-light}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-2}"
PROFILE_DIR="${PROFILE_DIR:-/scratch/project_462000131/${USER}/lumi-profile/${SLURM_JOB_ID:-manual}}"
PROFILE_COLLECT_CPU="${PROFILE_COLLECT_CPU:-0}"
PROFILER_SRUN_OPTS="${PROFILER_SRUN_OPTS:---ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap}"
SUMMARIZER="${SUMMARIZER:-${_profile_repo_dir}/src/summarize_rocm_smi.py}"
ANALYZER="${ANALYZER:-${_profile_repo_dir}/src/analyze_summary.py}"
REPORT_GENERATOR="${REPORT_GENERATOR:-${_profile_repo_dir}/src/generate_report.py}"
ROCPROFV3_SUMMARIZER="${ROCPROFV3_SUMMARIZER:-${_profile_repo_dir}/src/summarize_rocprofv3.py}"
ROCPROFSYS_SUMMARIZER="${ROCPROFSYS_SUMMARIZER:-${_profile_repo_dir}/src/summarize_rocprofsys.py}"
PROFILE_LOG_SCHEMA_VERSION="${PROFILE_LOG_SCHEMA_VERSION:-1}"
PROFILE_COLLECT_COMMAND="${PROFILE_COLLECT_COMMAND:-rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks}"
DEEP_PROFILE_DIR="${DEEP_PROFILE_DIR:-${PROFILE_DIR}/deep_profile}"
DEEP_TRACE_DIR="${DEEP_TRACE_DIR:-${DEEP_PROFILE_DIR}/trace}"
DEEP_TRACE_RAW_DIR="${DEEP_TRACE_RAW_DIR:-${DEEP_TRACE_DIR}/raw}"
DEEP_TRACE_SUMMARY="${DEEP_TRACE_SUMMARY:-${DEEP_TRACE_DIR}/summary.json}"
DEEP_SYSTEM_DIR="${DEEP_SYSTEM_DIR:-${DEEP_PROFILE_DIR}/system}"
DEEP_SYSTEM_RAW_DIR="${DEEP_SYSTEM_RAW_DIR:-${DEEP_SYSTEM_DIR}/raw}"
DEEP_SYSTEM_SUMMARY="${DEEP_SYSTEM_SUMMARY:-${DEEP_SYSTEM_DIR}/summary.json}"
DEEP_MANIFEST="${DEEP_MANIFEST:-${DEEP_PROFILE_DIR}/deep_manifest.json}"
ROCPROFV3_PATH="${ROCPROFV3_PATH:-$(command -v rocprofv3 2>/dev/null || true)}"
ROCPROFV3_EXTRA_OPTS="${ROCPROFV3_EXTRA_OPTS:-}"
LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-}"
LUMI_CONTAINER_USE_ROCM="${LUMI_CONTAINER_USE_ROCM:-1}"
LUMI_CONTAINER_BIND_EXTRA="${LUMI_CONTAINER_BIND_EXTRA:-}"
LUMI_CONTAINER_WORKDIR="${LUMI_CONTAINER_WORKDIR:-${DEEP_PROFILE_DIR}/container_workdir}"
LUMI_CONTAINER_ROCPROFV3="${LUMI_CONTAINER_ROCPROFV3:-rocprofv3}"
ROCPROFSYS_INSTALL_PREFIX="${ROCPROFSYS_INSTALL_PREFIX:-}"
ROCPROFSYS_EXTRA_OPTS="${ROCPROFSYS_EXTRA_OPTS:-}"
LUMI_CONTAINER_ROCPROFSYS_RUN="${LUMI_CONTAINER_ROCPROFSYS_RUN:-rocprof-sys-run}"
LUMI_CONTAINER_ROCPROFSYS_PYTHON="${LUMI_CONTAINER_ROCPROFSYS_PYTHON:-rocprof-sys-python}"
PROFILE_STARTED=0
PROFILE_SUMMARIZED=0
PROFILE_ANALYZED=0
PROFILE_REPORTED=0
PROFILER_PID=""
PROFILE_DEEP_TOOL_CMD=()
PROFILE_ROCPROFV3_TRACE_ARGS=()
PROFILE_CONTAINER_CMD=()
PROFILE_DEEP_TRACE_TOOL_PATH=""

. "${_profile_repo_dir}/scripts/lib/profile_container.sh"
. "${_profile_repo_dir}/scripts/lib/profile_distributed.sh"
. "${_profile_repo_dir}/scripts/lib/profile_deep_trace.sh"
. "${_profile_repo_dir}/scripts/lib/profile_deep_system.sh"

profile_start() {
  if [[ "${PROFILE_ENABLE}" != "1" || "${PROFILE_STARTED}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${PROFILE_DIR}"
  rm -f "${PROFILE_DIR}/STOP"
  export PROFILE_DIR PROFILE_INTERVAL PROFILE_LOG_SCHEMA_VERSION PROFILE_COLLECT_COMMAND PROFILE_COLLECT_CPU
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

profile_finalize_deep_profile() {
  local exit_code="$1"
  local status_label="$2"
  shift 2

  if [[ "${PROFILE_ENABLE}" != "1" ]]; then
    return 0
  fi

  local summarizer=""
  local raw_dir=""
  local summary_output=""
  case "${PROFILE_MODE}" in
    deep-trace)
      summarizer="${ROCPROFV3_SUMMARIZER}"
      raw_dir="${DEEP_TRACE_RAW_DIR}"
      summary_output="${DEEP_TRACE_SUMMARY}"
      ;;
    deep-system)
      summarizer="${ROCPROFSYS_SUMMARIZER}"
      raw_dir="${DEEP_SYSTEM_RAW_DIR}"
      summary_output="${DEEP_SYSTEM_SUMMARY}"
      ;;
    *)
      return 0
      ;;
  esac

  if [[ ! -f "${summarizer}" ]]; then
    return 0
  fi

  mkdir -p "${DEEP_PROFILE_DIR}" "${raw_dir}"

  local command_string=""
  printf -v command_string '%q ' "$@"
  command_string="${command_string% }"

  python3 "${summarizer}" \
    "${raw_dir}" \
    "${summary_output}" \
    "${DEEP_MANIFEST}" \
    --tool-path "${PROFILE_DEEP_TRACE_TOOL_PATH:-${ROCPROFV3_PATH}}" \
    --mode "${PROFILE_MODE}" \
    --command "${command_string}" \
    --status "${status_label}" \
    --exit-code "${exit_code}" || true

  echo "Deep profile summary: ${summary_output}"
  echo "Deep trace manifest: ${DEEP_MANIFEST}"
}

profile_run_command() {
  local deep_profile_enabled=0
  local status=0
  PROFILE_DEEP_TRACE_TOOL_PATH=""

  if [[ "${PROFILE_MODE}" == "deep-trace" || "${PROFILE_MODE}" == "deep-system" ]]; then
    deep_profile_enabled=1
    mkdir -p "${DEEP_TRACE_RAW_DIR}"
    mkdir -p "${DEEP_SYSTEM_RAW_DIR}"
  fi

  if profile_container_enabled; then
    local container_build_status=0
    profile_build_container_command "$@" || container_build_status=$?
    if [[ "${container_build_status}" != "0" ]]; then
      return "${container_build_status}"
    fi

    if [[ "${deep_profile_enabled}" == "1" ]]; then
      if [[ "${PROFILE_MODE}" == "deep-trace" ]]; then
        PROFILE_DEEP_TRACE_TOOL_PATH="singularity exec ${LUMI_CONTAINER_IMAGE} ${LUMI_CONTAINER_ROCPROFV3}"
        local -a container_rocprof_cmd=("${PROFILE_CONTAINER_CMD[@]}")
        profile_build_rocprofv3_trace_args
        container_rocprof_cmd+=("${PROFILE_ROCPROFV3_TRACE_ARGS[@]}")
        container_rocprof_cmd+=(-- "$@")
        "${container_rocprof_cmd[@]}"
      else
        profile_build_rocprofsys_container_command "$@" || return $?
        "${PROFILE_DEEP_TOOL_CMD[@]}"
      fi
    else
      local -a container_cmd=("${PROFILE_CONTAINER_CMD[@]}" "$@")
      "${container_cmd[@]}"
    fi
  else
    if [[ "${deep_profile_enabled}" == "1" ]]; then
      echo "Deep profiling is supported only for container launches. Set LUMI_CONTAINER_IMAGE to a supported PyTorch container; running without deep profile artifacts." >&2
      "$@"
      status=$?
      profile_finalize_deep_profile "${status}" "fallback_unsupported_host_deep_profile" "$@"
      return "${status}"
    fi

    "$@"
  fi

  status=$?

  if [[ "${deep_profile_enabled}" == "1" ]]; then
    if [[ "${status}" == "0" ]]; then
      profile_finalize_deep_profile "${status}" "completed" "$@"
    else
      profile_finalize_deep_profile "${status}" "completed_with_command_error" "$@"
    fi
  fi

  return "${status}"
}

profile_run_distributed_command() {
  local -a original_command=("$@")
  local -a srun_command=(srun)
  local -a payload=()
  local -a container_cmd=()
  local -a distributed_cmd=()
  local deep_profile_enabled=0
  local status=0

  PROFILE_DEEP_TRACE_TOOL_PATH=""

  profile_split_distributed_command "${original_command[@]}" || return $?
  payload=("${PROFILE_DISTRIBUTED_PAYLOAD[@]}")
  srun_command+=("${PROFILE_DISTRIBUTED_SRUN_ARGS[@]}")

  if [[ "${PROFILE_MODE}" == "deep-trace" || "${PROFILE_MODE}" == "deep-system" ]]; then
    deep_profile_enabled=1
  fi

  if profile_container_enabled; then
    local container_build_status=0
    profile_build_container_command "${payload[@]}" || container_build_status=$?
    if [[ "${container_build_status}" != "0" ]]; then
      return "${container_build_status}"
    fi
    container_cmd=("${PROFILE_CONTAINER_CMD[@]}")
  fi

  if [[ "${deep_profile_enabled}" == "1" ]]; then
    if ! profile_container_enabled; then
      echo "Distributed deep profiling is supported only for container launches. Set LUMI_CONTAINER_IMAGE to a supported PyTorch container; running without deep profile artifacts." >&2
      "${srun_command[@]}" -- "${payload[@]}"
      status=$?
      profile_finalize_deep_profile "${status}" "fallback_unsupported_host_deep_profile" "${original_command[@]}"
      return "${status}"
    fi

    if [[ "${PROFILE_MODE}" == "deep-trace" ]]; then
      local -a rocprof_extra_opts=()
      local extra_opt=""
      if [[ -n "${ROCPROFV3_EXTRA_OPTS}" ]]; then
        read -r -a rocprof_extra_opts <<< "${ROCPROFV3_EXTRA_OPTS}"
      fi

      local script=""
      script+="set -euo pipefail"$'\n'
      script+='host=$(hostname)'$'\n'
      script+='rank=${SLURM_PROCID:-0}'$'\n'
      script+="rank_dir=$(printf '%q' \"${DEEP_TRACE_RAW_DIR}\")/\${host}/rank-\${rank}"$'\n'
      script+='mkdir -p "${rank_dir}"'$'\n'
      script+='rocprof_cmd=('
      script+="$(printf '%q ' "${LUMI_CONTAINER_ROCPROFV3}" --runtime-trace --stats --output-format csv json --output-directory)"
      script+='"${rank_dir}" '
      script+="$(printf '%q ' --output-file trace)"
      for extra_opt in "${rocprof_extra_opts[@]}"; do
        script+="$(printf '%q ' "${extra_opt}")"
      done
      script+="-- "
      script+="$(profile_quote_args_for_shell "${payload[@]}")"
      script+=')'$'\n'
      script+='"${rocprof_cmd[@]}"'$'\n'

      PROFILE_DEEP_TRACE_TOOL_PATH="singularity exec ${LUMI_CONTAINER_IMAGE} ${LUMI_CONTAINER_ROCPROFV3}"
      distributed_cmd=("${srun_command[@]}" "${container_cmd[@]}" bash -lc "${script}")
    else
      profile_build_rocprofsys_container_command --distributed "${payload[@]}" || return $?
      distributed_cmd=("${srun_command[@]}" "${PROFILE_DEEP_TOOL_CMD[@]}")
    fi
  else
    if profile_container_enabled; then
      distributed_cmd=("${srun_command[@]}" "${container_cmd[@]}" "${payload[@]}")
    else
      distributed_cmd=("${srun_command[@]}" -- "${payload[@]}")
    fi
  fi

  "${distributed_cmd[@]}"
  status=$?

  if [[ "${deep_profile_enabled}" == "1" ]]; then
    if [[ "${status}" == "0" ]]; then
      profile_finalize_deep_profile "${status}" "completed" "${original_command[@]}"
    else
      profile_finalize_deep_profile "${status}" "completed_with_command_error" "${original_command[@]}"
    fi
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

profile_run_distributed() {
  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi

  if [[ "$#" -eq 0 ]]; then
    echo "profile_run_distributed requires an srun command" >&2
    return 2
  fi

  profile_start

  if profile_run_distributed_command "$@"; then
    status=0
  else
    status=$?
  fi

  profile_stop
  profile_summarize
  return "${status}"
}
