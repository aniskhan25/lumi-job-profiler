#!/bin/bash

profile_build_rocprofv3_trace_args() {
  PROFILE_ROCPROFV3_TRACE_ARGS=(
    "${LUMI_CONTAINER_ROCPROFV3}"
    --hip-trace
    --kernel-trace
    --memory-copy-trace
    --scratch-memory-trace
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
    PROFILE_ROCPROFV3_TRACE_ARGS+=("${rocprof_extra_opts[@]}")
  fi
}
