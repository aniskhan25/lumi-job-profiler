#!/bin/bash

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
