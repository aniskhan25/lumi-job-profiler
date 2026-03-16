# LUMI Job Profiler (Demo)

This repo contains a **user opt-in** profiling demo for LUMI GPU jobs (AMD/ROCm). It samples GPU metrics with `rocm-smi` during a job and produces a compact JSON summary at the end.

## What’s Included

- `bin/profile_hook.sh`: shell helper for minimal opt-in profiling in existing jobs
- `examples/sbatch_profiled.sh`: example Slurm job using the profiling helper
- `scripts/summarize_rocm_smi.py`: best-effort parser that generates `summary.json`
- `scripts/summarize_rocprofv3.py`: best-effort parser that generates deep-trace summaries and manifests
- `scripts/summarize_rocprofsys.py`: best-effort parser that generates deep-system summaries and manifests
- `scripts/analyze_summary.py`: rule-based analyzer that generates `analysis.json`
- `scripts/generate_report.py`: report generator that emits `report.md` and `report.html`
- `examples/demo_pytorch_rocm.py`: a PyTorch ROCm demo workload to generate GPU activity
- `docs/implementation_plan.md`: system-level plan for a full feedback loop

## Requirements

- LUMI login/compute environment
- Slurm `sbatch`/`srun`
- ROCm installed on compute nodes (`rocm-smi` available)
- `singularity` available on compute nodes for containerized PyTorch runs
- A supported PyTorch container image for deep tracing. The example template uses:
  - `/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif`

## Quick Start (Existing Job)

1. Clone the repo on shared scratch:

```bash
cd /scratch/<project_id>/$USER
git clone https://github.com/aniskhan25/lumi-job-profiler.git
cd lumi-job-profiler
```

2. In your existing `sbatch` script, add the hook after your module loads:

```bash
source /scratch/<project_id>/$USER/lumi-job-profiler/bin/profile_hook.sh
```

For containerized PyTorch jobs, also export the container image before `profile_run`:

```bash
export LUMI_CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
```

3. Wrap your current launch command:

```bash
profile_run -- python3 myprog.py <options>
```

4. Submit the job as usual:

```bash
sbatch your_job.sh
```

5. After the job completes, find logs and summary here:

```
/scratch/<project_id>/$USER/lumi-profile/<job_id>/
  <node>.log
  summary.json
  analysis.json
  report.md
  report.html
  deep_profile/
    deep_manifest.json
    trace/
      summary.json
      raw/
```

## Example Template

If you want a complete working example, use `examples/sbatch_profiled.sh`.

## Manual Lifecycle Control

If your job has multiple phases and you only want to profile part of it, use the lifecycle functions directly:

```bash
source /scratch/<project_id>/$USER/lumi-job-profiler/bin/profile_hook.sh
trap profile_cleanup EXIT

export LUMI_CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
singularity exec --bind "${PWD}:${PWD}" --rocm "${LUMI_CONTAINER_IMAGE}" python3 prepare_data.py

profile_start
srun singularity exec --bind "${PWD}:${PWD}" --rocm "${LUMI_CONTAINER_IMAGE}" python3 train.py --epochs 10
profile_stop

python3 collect_results.py
profile_summarize
```

Available functions:

- `profile_start`: launch the profiling sidecar
- `profile_stop`: stop sampling and wait for the sidecar
- `profile_summarize`: write `summary.json`
- `profile_cleanup`: run `profile_stop` and `profile_summarize` safely
- `profile_run -- <command>`: convenience wrapper for single-command jobs

`profile_run` now supports direct commands only. Keep Slurm allocation details in the job script itself and pass the application launch directly to the hook.

Deep-trace note:

- `LUMI_PROFILE_MODE=deep-trace` currently applies to the wrapped `profile_run -- <command>` path
- deep-trace is supported for container launches configured with `LUMI_CONTAINER_IMAGE`
- when the wrapped command starts with `srun`, the hook injects `singularity exec ... rocprofv3` inside the `srun` step so tracing attaches to the Slurm task instead of the `srun` launcher
- `LUMI_PROFILE_MODE=deep-system` runs `rocprofiler-systems` inside the container and writes perfetto-style system traces
- host-side deep profiling with the retiring `pytorch/2.7` module stack is treated as unsupported
- manual lifecycle control still manages the lightweight `rocm-smi` sidecar

## Controls

The helper is enabled by default. You can override behavior with:

- `LUMI_PROFILE=0` to disable
- `PROFILE_INTERVAL=2` to change sampling interval (seconds)
- `PROFILE_COLLECT_CPU=1` to collect optional host CPU, memory, and load metrics
- `LUMI_PROFILE_MODE=deep-trace` to keep the lightweight profile and also run `rocprofv3` for `profile_run`
- `LUMI_PROFILE_MODE=deep-system` to keep the lightweight profile and also run `rocprofiler-systems` for `profile_run`
- `LUMI_CONTAINER_IMAGE=/path/to/container.sif` to run the profiled payload inside a container
- `LUMI_CONTAINER_BIND_EXTRA="/path/a:/path/a,/path/b:/path/b"` to add extra bind mounts for container runs
- `ROCPROFV3_EXTRA_OPTS="..."` to append extra `rocprofv3` options in deep-trace mode
- `ROCPROFSYS_INSTALL_PREFIX=/scratch/<project_id>/$USER/tools/rocprofiler-systems-container` to enable `deep-system`
- `ROCPROFSYS_EXTRA_OPTS="..."` to append extra `rocprof-sys` options in deep-system mode
- `PROFILER_SRUN_OPTS="--ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap"` to adjust the sidecar launch
- `PROFILE_DIR=/scratch/<project_id>/$USER/lumi-profile/$SLURM_JOB_ID` to override the output directory

Example:

```bash
LUMI_PROFILE=1 PROFILE_INTERVAL=1 sbatch your_job.sh
```

Deep-trace example:

```bash
LUMI_PROFILE_MODE=deep-trace \
LUMI_CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif \
sbatch your_job.sh
```

Deep-system example:

```bash
LUMI_PROFILE_MODE=deep-system \
LUMI_CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif \
ROCPROFSYS_INSTALL_PREFIX=/scratch/<project_id>/$USER/tools/rocprofiler-systems-container \
sbatch your_job.sh
```

## Demo Workload

The example template runs a PyTorch ROCm workload when available:

```
examples/demo_pytorch_rocm.py --seconds 60 --size 4096 --dtype fp16
```

If the demo script is missing, the template falls back to `./your_application`.

## What Gets Profiled

The profiler samples `rocm-smi` during the profiled section of the job with:

```bash
rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks
```

This captures:

- GPU utilization
- GPU memory allocation (VRAM%)
- GPU memory read/write activity
- GPU package power
- GPU temperatures
- GPU clock frequencies (`fclk`, `mclk`, `sclk`, `socclk`)

If `PROFILE_COLLECT_CPU=1` is set, the profiler also captures lightweight host metrics:

- CPU utilization
- CPU iowait
- memory used percentage
- load averages

`summary.json` also records available Slurm job-structure metadata such as:

- `ntasks`
- `cpus_per_task`
- `gpus_requested`
- `gpus_per_node`

The profiler writes raw samples to per-node log files:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/<node>.log
```

Each raw log includes metadata headers such as:

```text
# profile_log_schema_version=1
# profile_collect_command=rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks
```

At the end of profiling, these raw samples are summarized into:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/summary.json
```

The profiler also writes a rule-based analysis artifact:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/analysis.json
```

The profiler also generates user-facing reports:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/report.md
/scratch/<project_id>/$USER/lumi-profile/<job_id>/report.html
```

When `LUMI_PROFILE_MODE=deep-trace` is set and `rocprofv3` is available, the profiler also writes:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/deep_manifest.json
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/trace/summary.json
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/trace/raw/
```

When `LUMI_PROFILE_MODE=deep-system` is set and `ROCPROFSYS_INSTALL_PREFIX` points to a working install, the profiler writes:

```text
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/deep_manifest.json
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/system/summary.json
/scratch/<project_id>/$USER/lumi-profile/<job_id>/deep_profile/system/raw/
```

For `deep-system`, the main artifact is `perfetto-trace-*.proto` under `deep_profile/system/raw/rocprofsys-*/.../`. Open that file in [Perfetto UI](https://ui.perfetto.dev) for timeline analysis.

Deep profiling keeps the existing lightweight artifacts and adds either `rocprofv3` trace artifacts (`deep-trace`) or `rocprofiler-systems` artifacts (`deep-system`) for the wrapped command. The supported path is container-first: the hook runs the profiled payload inside `LUMI_CONTAINER_IMAGE` and launches the selected tool inside that container. If the container or the requested deep tool is unavailable, the job still runs and `deep_manifest.json` records the fallback.

`summary.json` contains:

- `collection`: summary schema version, generation time, raw log schema versions, collection command, inferred sampling interval
- `job`: job metadata derived from the Slurm environment when available
- `job_metrics`: job-level derived metrics such as average GPU utilization, peak VRAM utilization, and active/effective GPU estimates
- `nodes`: per-node, per-GPU aggregates such as average, p95, and max values for metrics present in the raw `rocm-smi` output
- `warnings`: parse warnings or missing-data notices

## Output Format

`summary.json` includes these top-level sections:

- `collection`
- `job`
- `job_metrics`
- `nodes`
- `warnings`

Under `nodes`, the parser emits per-node, per-GPU aggregates (avg, p95, max) for common metrics when present in `rocm-smi` output:

- GPU utilization
- VRAM utilization
- memory read/write activity
- Power
- Temperature
- Core/memory clocks

When CPU collection is enabled, node summaries may also include:

- CPU utilization
- CPU idle percentage
- CPU iowait percentage
- memory used percentage
- `load1`, `load5`, `load15`

The parser is best‑effort and tolerant of missing fields.

`analysis.json` includes:

- `efficiency`: efficiency class and the utilization score used to classify the job
- `root_causes`: rule-based findings such as overscaling, parallelism mismatch, CPU bottlenecks, or likely stalls
- `recommendations`: deduplicated advisory actions derived from the findings
- `job`: copied job metadata from `summary.json`

`report.md` and `report.html` include:

- a job summary
- efficiency classification and key metrics
- a per-GPU overview table
- lightweight textual utilization bars
- findings, recommendations, and documentation links where available
- deep-trace artifact status and top `rocprofv3` entries when deep mode is used

## Appendix: rocprofiler-systems Setup on LUMI

`rocprofiler-systems` is not preinstalled in the tested LUMI environment used by this repo. The supported `deep-system` path assumes you first build a working install inside the same multitorch container used for profiling.

Helper scripts:

- Build the tool:
  - `bin/build_rocprofiler_systems_container.sh`
- Smoke-test the install against a minimal PyTorch ROCm workload:
  - `bin/smoke_test_rocprofiler_systems_container.sh`

Run the build from a login node:

```bash
REPO_DIR=/scratch/<project_id>/$USER/lumi-job-profiler
"${REPO_DIR}/bin/build_rocprofiler_systems_container.sh"
```

The build script:

- clones `https://github.com/ROCm/rocprofiler-systems.git` with submodules if needed
- builds inside the multitorch ROCm 6.4 container
- disables MPI-related autodetection that broke the source build on LUMI
- installs to `/scratch/<project_id>/$USER/tools/rocprofiler-systems-container` by default

Supported overrides:

- `PROJECT_ID`
- `CONTAINER_IMAGE`
- `TOOLS_DIR`
- `INSTALL_PREFIX`
- `SOURCE_DIR`

Example:

```bash
REPO_DIR=/scratch/<project_id>/$USER/lumi-job-profiler
PROJECT_ID=<project_id> \
INSTALL_PREFIX=/scratch/<project_id>/$USER/tools/rocprofiler-systems-container \
"${REPO_DIR}/bin/build_rocprofiler_systems_container.sh"
```

Smoke-test the install:

```bash
REPO_DIR=/scratch/<project_id>/$USER/lumi-job-profiler
INSTALL_PREFIX=/scratch/<project_id>/$USER/tools/rocprofiler-systems-container \
"${REPO_DIR}/bin/smoke_test_rocprofiler_systems_container.sh"
```

The smoke test verifies:

- `rocprof-sys-python` starts in the container
- the required runtime environment is set:
  - `ROCPROFSYS_SCRIPT_PATH`
  - `LD_LIBRARY_PATH` including PyTorch `torch/lib` and the rocprofiler-systems libraries
- a minimal PyTorch ROCm script completes under the profiler
- Perfetto-compatible outputs are produced under the smoke-test output directory

Once the build succeeds, enable `deep-system` in normal profiled jobs with:

```bash
REPO_DIR=/scratch/<project_id>/$USER/lumi-job-profiler
LUMI_PROFILE_MODE=deep-system \
LUMI_CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif \
ROCPROFSYS_INSTALL_PREFIX=/scratch/<project_id>/$USER/tools/rocprofiler-systems-container \
sbatch "${REPO_DIR}/examples/sbatch_profiled.sh"
```

The hook handles the extra runtime environment automatically when `ROCPROFSYS_INSTALL_PREFIX` is set.

## Development

Run the parser tests with:

```bash
python3 -m unittest discover -s tests
```

## Limitations (Demo Scope)

- No cluster‑wide hooks; per‑job opt‑in only.
- Sampling overhead exists; keep intervals reasonable.
- `rocm-smi` output varies across GPUs and driver versions.

## Next Steps

- Add stricter parsing for the exact `rocm-smi` format on LUMI
- Store summaries in a structured DB
- Implement full post‑job reporting pipeline

## License

TBD
