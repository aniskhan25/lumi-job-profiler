#!/bin/bash

PROFILE_DISTRIBUTED_SRUN_ARGS=()
PROFILE_DISTRIBUTED_PAYLOAD=()

profile_split_distributed_command() {
  PROFILE_DISTRIBUTED_SRUN_ARGS=()
  PROFILE_DISTRIBUTED_PAYLOAD=()

  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi

  if [[ "$#" -eq 0 || "$1" != "srun" ]]; then
    echo "profile_run_distributed requires a command of the form: srun [srun opts] -- <payload>" >&2
    return 2
  fi
  shift

  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    PROFILE_DISTRIBUTED_SRUN_ARGS+=("$1")
    shift
  done

  if [[ "$#" -eq 0 ]]; then
    echo "profile_run_distributed requires an explicit '--' separator before the payload." >&2
    return 2
  fi

  PROFILE_DISTRIBUTED_PAYLOAD=("$@")
}
