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
ROCPROFSYS_SUMMARIZER="${ROCPROFSYS_SUMMARIZER:-${_profile_hook_dir}/summarize_rocprofsys.py}"
PROFILE_LOG_SCHEMA_VERSION="${PROFILE_LOG_SCHEMA_VERSION:-1}"
PROFILE_COLLECT_COMMAND_DEFAULT="rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks"
PROFILE_COLLECT_COMMAND="${PROFILE_COLLECT_COMMAND:-${PROFILE_COLLECT_COMMAND_DEFAULT}}"
PROFILE_COLLECT_WARNING=""
ROCM_SMI_PYTHON="${ROCM_SMI_PYTHON:-/usr/bin/python3}"
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
ROCPROFV3_PYTHON="${ROCPROFV3_PYTHON:-/usr/bin/python3}"
LUMI_CONTAINER_RUNTIME="${LUMI_CONTAINER_RUNTIME:-singularity}"
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
ROCPROFV3_LAUNCHER=""
PROFILE_ROCPROFV3_CMD=()
PROFILE_CONTAINER_CMD=()
PROFILE_CONTAINER_BIND_SPECS=()
PROFILE_SRUN_ARGS=()
PROFILE_SRUN_PAYLOAD_ARGS=()
PROFILE_DEEP_TRACE_TOOL_PATH=""

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

profile_is_python_script() {
  local path="$1"
  local first_line=""

  if [[ ! -r "${path}" ]]; then
    return 1
  fi

  IFS= read -r first_line < "${path}" || true
  [[ "${path}" == *.py || "${first_line}" == '#!'*python* ]]
}

profile_container_enabled() {
  [[ -n "${LUMI_CONTAINER_IMAGE}" ]]
}

profile_container_runtime_available() {
  command -v "${LUMI_CONTAINER_RUNTIME}" >/dev/null 2>&1
}

profile_add_container_bind_spec() {
  local spec="$1"
  local existing=""

  [[ -n "${spec}" ]] || return 0

  for existing in "${PROFILE_CONTAINER_BIND_SPECS[@]}"; do
    [[ "${existing}" == "${spec}" ]] && return 0
  done

  PROFILE_CONTAINER_BIND_SPECS+=("${spec}")
}

profile_add_container_bind_path() {
  local path="$1"
  [[ -n "${path}" ]] || return 0
  profile_add_container_bind_spec "${path}:${path}"
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

  if profile_is_python_script "${resolved_path}"; then
    if [[ ! -x "${ROCPROFV3_PYTHON}" ]]; then
      ROCPROFV3_PATH=""
      return 0
    fi
    ROCPROFV3_LAUNCHER="${ROCPROFV3_PYTHON} ${resolved_path}"
  else
    ROCPROFV3_LAUNCHER="${resolved_path}"
  fi
}

profile_build_rocprofv3_command() {
  PROFILE_ROCPROFV3_CMD=()

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
  )

  if [[ -n "${ROCPROFV3_EXTRA_OPTS}" ]]; then
    local -a rocprof_extra_opts=()
    read -r -a rocprof_extra_opts <<< "${ROCPROFV3_EXTRA_OPTS}"
    rocprof_cmd+=("${rocprof_extra_opts[@]}")
  fi

  local -a rocprof_launcher=()
  read -r -a rocprof_launcher <<< "${ROCPROFV3_LAUNCHER}"
  PROFILE_ROCPROFV3_CMD=("${rocprof_launcher[@]}" "${rocprof_cmd[@]}")
}

profile_collect_container_bind_specs() {
  PROFILE_CONTAINER_BIND_SPECS=()

  profile_add_container_bind_path "${PROFILE_DIR}"
  profile_add_container_bind_path "${DEEP_PROFILE_DIR}"
  profile_add_container_bind_path "${DEEP_TRACE_DIR}"
  profile_add_container_bind_path "${DEEP_TRACE_RAW_DIR}"
  profile_add_container_bind_path "${DEEP_SYSTEM_DIR}"
  profile_add_container_bind_path "${DEEP_SYSTEM_RAW_DIR}"
  profile_add_container_bind_path "${LUMI_CONTAINER_WORKDIR}"
  profile_add_container_bind_path "${ROCPROFSYS_INSTALL_PREFIX}"
  profile_add_container_bind_path "${PWD}"
  profile_add_container_bind_path "${SLURM_SUBMIT_DIR:-}"

  if [[ "$#" -ge 1 ]]; then
    if [[ "$1" == /* ]]; then
      profile_add_container_bind_path "$(dirname "$1")"
    fi

    if [[ "$#" -ge 2 && "$2" == /* ]]; then
      case "$(basename "$1")" in
        python|python[0-9]*|bash|sh|env)
          profile_add_container_bind_path "$(dirname "$2")"
          ;;
      esac
    fi
  fi

  if [[ -n "${LUMI_CONTAINER_BIND_EXTRA}" ]]; then
    local old_ifs="${IFS}"
    local entry=""
    IFS=','
    for entry in ${LUMI_CONTAINER_BIND_EXTRA}; do
      profile_add_container_bind_spec "${entry}"
    done
    IFS="${old_ifs}"
  fi
}

profile_build_container_command() {
  PROFILE_CONTAINER_CMD=()

  if ! profile_container_enabled; then
    return 1
  fi

  if [[ ! -e "${LUMI_CONTAINER_IMAGE}" ]]; then
    echo "Configured container image does not exist: ${LUMI_CONTAINER_IMAGE}" >&2
    return 2
  fi

  if ! profile_container_runtime_available; then
    echo "Configured container runtime was not found in PATH: ${LUMI_CONTAINER_RUNTIME}" >&2
    return 2
  fi

  mkdir -p "${PROFILE_DIR}" "${DEEP_PROFILE_DIR}" "${DEEP_TRACE_DIR}" "${DEEP_TRACE_RAW_DIR}" "${LUMI_CONTAINER_WORKDIR}"
  profile_collect_container_bind_specs "$@"

  local bind_arg=""
  local spec=""
  for spec in "${PROFILE_CONTAINER_BIND_SPECS[@]}"; do
    if [[ -n "${bind_arg}" ]]; then
      bind_arg+=",${spec}"
    else
      bind_arg="${spec}"
    fi
  done

  PROFILE_CONTAINER_CMD=("${LUMI_CONTAINER_RUNTIME}" exec)
  if [[ -n "${bind_arg}" ]]; then
    PROFILE_CONTAINER_CMD+=(--bind "${bind_arg}")
  fi
  if [[ "${LUMI_CONTAINER_USE_ROCM}" == "1" ]]; then
    PROFILE_CONTAINER_CMD+=(--rocm)
  fi
  PROFILE_CONTAINER_CMD+=(--pwd "${LUMI_CONTAINER_WORKDIR}" "${LUMI_CONTAINER_IMAGE}")
}

profile_is_python_launcher() {
  local exe="$1"
  local base
  base="$(basename "${exe}")"
  [[ "${base}" == python || "${base}" == python[0-9]* ]]
}

profile_payload_is_python_script() {
  if [[ "$#" -lt 2 ]]; then
    return 1
  fi

  profile_is_python_launcher "$1" && [[ "$2" != "-" && "$2" != -* ]]
}

profile_quote_args_for_shell() {
  local quoted=""
  printf -v quoted '%q ' "$@"
  printf '%s' "${quoted% }"
}

profile_build_rocprofsys_container_command() {
  local -a payload=("$@")
  local prefix="${ROCPROFSYS_INSTALL_PREFIX}"

  if [[ -z "${prefix}" ]]; then
    echo "Deep system profiling requires ROCPROFSYS_INSTALL_PREFIX to point to the rocprofiler-systems install." >&2
    return 2
  fi

  if [[ ! -d "${prefix}" ]]; then
    echo "Configured rocprofiler-systems install prefix does not exist: ${prefix}" >&2
    return 2
  fi

  mkdir -p "${DEEP_SYSTEM_RAW_DIR}"

  local payload_cmd=""
  local python_probe=""
  local runner=""
  local output_opts=""
  if profile_payload_is_python_script "${payload[@]}"; then
    runner="${LUMI_CONTAINER_ROCPROFSYS_PYTHON}"
    payload_cmd="-- $(profile_quote_args_for_shell "${payload[@]:1}")"
    python_probe="$(profile_quote_args_for_shell "${payload[0]}")"
  else
    runner="${LUMI_CONTAINER_ROCPROFSYS_RUN}"
    payload_cmd="-- $(profile_quote_args_for_shell "${payload[@]}")"
    python_probe="python3"
    output_opts="--output $(printf '%q' "${DEEP_SYSTEM_RAW_DIR}/rocprofsys") "
  fi

  local extra_opts=""
  if [[ -n "${ROCPROFSYS_EXTRA_OPTS}" ]]; then
    extra_opts="${ROCPROFSYS_EXTRA_OPTS} "
  fi

  local script=""
  script+="set -euo pipefail"$'\n'
  script+="source $(printf '%q' "${prefix}/share/rocprofiler-systems/setup-env.sh")"$'\n'
  script+="export PATH=$(printf '%q' "${prefix}/bin"):\${PATH}"$'\n'
  script+="export ROCPROFSYS_SCRIPT_PATH=$(printf '%q' "${prefix}/libexec/rocprofiler-systems")"$'\n'
  script+="TORCH_LIB=\$(${python_probe} -c \"import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')\")"$'\n'
  script+="export LD_LIBRARY_PATH=\${TORCH_LIB}:$(printf '%q' "${prefix}/lib"):$(printf '%q' "${prefix}/lib64"):$(printf '%q' "${prefix}/lib/rocprofiler-systems"):\${LD_LIBRARY_PATH:-}"$'\n'
  script+="cd $(printf '%q' "${DEEP_SYSTEM_RAW_DIR}")"$'\n'
  script+="$(printf '%q' "${runner}") ${extra_opts}${output_opts}${payload_cmd}"$'\n'

  PROFILE_DEEP_TRACE_TOOL_PATH="${LUMI_CONTAINER_RUNTIME} exec ${LUMI_CONTAINER_IMAGE} ${runner}"
  PROFILE_ROCPROFV3_CMD=("${PROFILE_CONTAINER_CMD[@]}" bash -lc "${script}")
}

profile_srun_option_takes_value() {
  case "$1" in
    -A|--account|\
    -c|--cpus-per-task|--cpus-per-gpu|\
    -C|--constraint|\
    -D|--chdir|\
    -e|--error|\
    -G|--gpus|--gpus-per-node|--gpus-per-task|\
    -g|--gres|\
    -i|--input|\
    -J|--job-name|\
    -m|--distribution|\
    --mem|--mem-per-cpu|--mem-per-gpu|\
    --mpi|\
    -N|--nodes|\
    -n|--ntasks|--ntasks-per-node|\
    -o|--output|\
    -p|--partition|\
    -q|--qos|\
    --reservation|\
    -t|--time|\
    --threads-per-core|\
    -w|--nodelist|\
    -x|--exclude|\
    --cpu-bind|--gpu-bind|--hint|--export)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

profile_split_srun_command() {
  PROFILE_SRUN_ARGS=()
  PROFILE_SRUN_PAYLOAD_ARGS=()

  if [[ "$#" -eq 0 || "$1" != "srun" ]]; then
    return 1
  fi

  shift

  local expecting_value=0
  local token=""

  while [[ "$#" -gt 0 ]]; do
    token="$1"
    shift

    if [[ "${expecting_value}" == "1" ]]; then
      PROFILE_SRUN_ARGS+=("${token}")
      expecting_value=0
      continue
    fi

    if [[ "${token}" == "--" ]]; then
      PROFILE_SRUN_ARGS+=("${token}")
      PROFILE_SRUN_PAYLOAD_ARGS=("$@")
      return 0
    fi

    if [[ "${token}" != "-"* || "${token}" == "-" ]]; then
      PROFILE_SRUN_PAYLOAD_ARGS=("${token}" "$@")
      return 0
    fi

    PROFILE_SRUN_ARGS+=("${token}")

    if [[ "${token}" == --*=* ]]; then
      continue
    fi

    if profile_srun_option_takes_value "${token}"; then
      expecting_value=1
    fi
  done

  return 0
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

  if profile_split_srun_command "$@" && [[ "${#PROFILE_SRUN_PAYLOAD_ARGS[@]}" -gt 0 ]]; then
    if profile_container_enabled; then
      local container_build_status=0
      profile_build_container_command "${PROFILE_SRUN_PAYLOAD_ARGS[@]}" || container_build_status=$?
      if [[ "${container_build_status}" != "0" ]]; then
        return "${container_build_status}"
      fi

      if [[ "${deep_profile_enabled}" == "1" ]]; then
        local -a traced_srun_cmd=(
          srun
          "${PROFILE_SRUN_ARGS[@]}"
        )
        if [[ "${PROFILE_MODE}" == "deep-trace" ]]; then
          PROFILE_DEEP_TRACE_TOOL_PATH="${LUMI_CONTAINER_RUNTIME} exec ${LUMI_CONTAINER_IMAGE} ${LUMI_CONTAINER_ROCPROFV3}"
          traced_srun_cmd+=(
            "${PROFILE_CONTAINER_CMD[@]}"
            "${LUMI_CONTAINER_ROCPROFV3}"
            --runtime-trace
            --stats
            --output-format
            csv
            json
            --output-directory
            "${DEEP_TRACE_RAW_DIR}"
            --output-file
            trace
          )
          if [[ -n "${ROCPROFV3_EXTRA_OPTS}" ]]; then
            local -a rocprof_extra_opts=()
            read -r -a rocprof_extra_opts <<< "${ROCPROFV3_EXTRA_OPTS}"
            traced_srun_cmd+=("${rocprof_extra_opts[@]}")
          fi
          traced_srun_cmd+=(-- "${PROFILE_SRUN_PAYLOAD_ARGS[@]}")
        else
          profile_build_rocprofsys_container_command "${PROFILE_SRUN_PAYLOAD_ARGS[@]}" || return $?
          traced_srun_cmd+=("${PROFILE_ROCPROFV3_CMD[@]}")
        fi
        "${traced_srun_cmd[@]}"
      else
        local -a container_srun_cmd=(
          srun
          "${PROFILE_SRUN_ARGS[@]}"
          "${PROFILE_CONTAINER_CMD[@]}"
          "${PROFILE_SRUN_PAYLOAD_ARGS[@]}"
        )
        "${container_srun_cmd[@]}"
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
  else
    if profile_container_enabled; then
      local container_build_status=0
      profile_build_container_command "$@" || container_build_status=$?
      if [[ "${container_build_status}" != "0" ]]; then
        return "${container_build_status}"
      fi

      if [[ "${deep_profile_enabled}" == "1" ]]; then
        if [[ "${PROFILE_MODE}" == "deep-trace" ]]; then
          PROFILE_DEEP_TRACE_TOOL_PATH="${LUMI_CONTAINER_RUNTIME} exec ${LUMI_CONTAINER_IMAGE} ${LUMI_CONTAINER_ROCPROFV3}"
          local -a container_rocprof_cmd=(
            "${PROFILE_CONTAINER_CMD[@]}"
            "${LUMI_CONTAINER_ROCPROFV3}"
            --runtime-trace
            --stats
            --output-format
            csv
            json
            --output-directory
            "${DEEP_TRACE_RAW_DIR}"
            --output-file
            trace
          )
          if [[ -n "${ROCPROFV3_EXTRA_OPTS}" ]]; then
            local -a rocprof_extra_opts=()
            read -r -a rocprof_extra_opts <<< "${ROCPROFV3_EXTRA_OPTS}"
            container_rocprof_cmd+=("${rocprof_extra_opts[@]}")
          fi
          container_rocprof_cmd+=(-- "$@")
          "${container_rocprof_cmd[@]}"
        else
          profile_build_rocprofsys_container_command "$@" || return $?
          "${PROFILE_ROCPROFV3_CMD[@]}"
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
