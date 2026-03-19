#!/bin/bash

profile_build_rocprofsys_container_command() {
  local distributed=0
  if [[ "$#" -gt 0 && "$1" == "--distributed" ]]; then
    distributed=1
    shift
  fi

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
  if [[ "${distributed}" == "1" ]]; then
    script+='host=$(hostname)'$'\n'
    script+='rank=${SLURM_PROCID:-0}'$'\n'
    script+="rank_dir=$(printf '%q' "${DEEP_SYSTEM_RAW_DIR}")/\${host}/rank-\${rank}"$'\n'
    script+='mkdir -p "${rank_dir}"'$'\n'
    script+='cd "${rank_dir}"'$'\n'
    if [[ "${runner}" == "${LUMI_CONTAINER_ROCPROFSYS_RUN}" ]]; then
      output_opts='--output "${rank_dir}/rocprofsys" '
    fi
  else
    script+="cd $(printf '%q' "${DEEP_SYSTEM_RAW_DIR}")"$'\n'
  fi
  script+="$(printf '%q' "${runner}") ${extra_opts}${output_opts}${payload_cmd}"$'\n'

  PROFILE_DEEP_TRACE_TOOL_PATH="singularity exec ${LUMI_CONTAINER_IMAGE} ${runner}"
  PROFILE_DEEP_TOOL_CMD=("${PROFILE_CONTAINER_CMD[@]}" bash -lc "${script}")
}
