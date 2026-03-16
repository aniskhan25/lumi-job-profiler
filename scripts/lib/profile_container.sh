#!/bin/bash

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
