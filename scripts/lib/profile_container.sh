#!/bin/bash

profile_container_enabled() {
  [[ -n "${LUMI_CONTAINER_IMAGE}" ]]
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

  if ! command -v singularity >/dev/null 2>&1; then
    echo "singularity was not found in PATH" >&2
    return 2
  fi

  mkdir -p "${PROFILE_DIR}" "${DEEP_PROFILE_DIR}" "${DEEP_TRACE_DIR}" "${DEEP_TRACE_RAW_DIR}" "${LUMI_CONTAINER_WORKDIR}"

  local -a bind_specs=(
    "${PROFILE_DIR}:${PROFILE_DIR}"
    "${DEEP_PROFILE_DIR}:${DEEP_PROFILE_DIR}"
    "${DEEP_TRACE_DIR}:${DEEP_TRACE_DIR}"
    "${DEEP_TRACE_RAW_DIR}:${DEEP_TRACE_RAW_DIR}"
    "${DEEP_SYSTEM_DIR}:${DEEP_SYSTEM_DIR}"
    "${DEEP_SYSTEM_RAW_DIR}:${DEEP_SYSTEM_RAW_DIR}"
    "${LUMI_CONTAINER_WORKDIR}:${LUMI_CONTAINER_WORKDIR}"
    "${PWD}:${PWD}"
  )

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    bind_specs+=("${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}")
  fi

  if [[ -n "${ROCPROFSYS_INSTALL_PREFIX}" ]]; then
    bind_specs+=("${ROCPROFSYS_INSTALL_PREFIX}:${ROCPROFSYS_INSTALL_PREFIX}")
  fi

  if [[ -n "${LUMI_CONTAINER_BIND_EXTRA}" ]]; then
    local old_ifs="${IFS}"
    local entry=""
    IFS=','
    for entry in ${LUMI_CONTAINER_BIND_EXTRA}; do
      [[ -n "${entry}" ]] && bind_specs+=("${entry}")
    done
    IFS="${old_ifs}"
  fi

  local bind_arg=""
  bind_arg="$(IFS=,; printf '%s' "${bind_specs[*]}")"

  PROFILE_CONTAINER_CMD=(singularity exec --bind "${bind_arg}")
  if [[ "${LUMI_CONTAINER_USE_ROCM}" == "1" ]]; then
    PROFILE_CONTAINER_CMD+=(--rocm)
  fi
  PROFILE_CONTAINER_CMD+=(--pwd "${LUMI_CONTAINER_WORKDIR}" "${LUMI_CONTAINER_IMAGE}")
}

profile_payload_is_python_script() {
  if [[ "$#" -lt 2 ]]; then
    return 1
  fi

  case "$(basename "$1")" in
    python|python[0-9]*)
      [[ "$2" != "-" && "$2" != -* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

profile_quote_args_for_shell() {
  local quoted=""
  printf -v quoted '%q ' "$@"
  printf '%s' "${quoted% }"
}
