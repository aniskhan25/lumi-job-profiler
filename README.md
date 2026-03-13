# LUMI Job Profiler (Demo)

This repo contains a **user opt-in** profiling demo for LUMI GPU jobs (AMD/ROCm). It samples GPU metrics with `rocm-smi` during a job and produces a compact JSON summary at the end.

## What’s Included

- `scripts/profile_hook.sh`: shell helper for minimal opt-in profiling in existing jobs
- `templates/sbatch_profiled.sh`: example Slurm job using the profiling helper
- `scripts/summarize_rocm_smi.py`: best-effort parser that generates `summary.json`
- `scripts/demo_pytorch_rocm.py`: a PyTorch ROCm demo workload to generate GPU activity
- `implementation_plan.md`: system-level plan for a full feedback loop

## Requirements

- LUMI login/compute environment
- Slurm `sbatch`/`srun`
- ROCm installed on compute nodes (`rocm-smi` available)
- PyTorch module on LUMI (template uses `pytorch/2.7`)

## Quick Start (Existing Job)

1. Clone the repo on shared scratch:

```bash
cd /scratch/project_462000131/anisrahm
git clone https://github.com/aniskhan25/lumi-job-profiler.git
cd lumi-job-profiler
```

2. In your existing `sbatch` script, add the hook after your module loads:

```bash
source /scratch/project_462000131/anisrahm/lumi-job-profiler/scripts/profile_hook.sh
```

3. Wrap your current launch command:

```bash
profile_run -- srun python3 myprog.py <options>
```

4. Submit the job as usual:

```bash
sbatch your_job.sh
```

5. After the job completes, find logs and summary here:

```
/scratch/project_462000131/<username>/lumi-profile/<jobid>/
  <node>.log
  summary.json
```

## Example Template

If you want a complete working example, use [templates/sbatch_profiled.sh](/Users/anisrahm/Documents/lumi-job-profiler/templates/sbatch_profiled.sh).

## Manual Lifecycle Control

If your job has multiple phases and you only want to profile part of it, use the lifecycle functions directly:

```bash
source /scratch/project_462000131/anisrahm/lumi-job-profiler/scripts/profile_hook.sh
trap profile_cleanup EXIT

module load pytorch/2.7
python3 prepare_data.py

profile_start
srun python3 train.py --epochs 10
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

## Controls

The helper is enabled by default. You can override behavior with:

- `LUMI_PROFILE=0` to disable
- `PROFILE_INTERVAL=2` to change sampling interval (seconds)
- `PROFILER_SRUN_OPTS="--ntasks-per-node=1 --cpus-per-task=1 --mpi=none --cpu-bind=none --overlap"` to adjust the sidecar launch
- `PROFILE_DIR=/scratch/project_462000131/$USER/lumi-profile/$SLURM_JOB_ID` to override the output directory

Example:

```bash
LUMI_PROFILE=1 PROFILE_INTERVAL=1 sbatch your_job.sh
```

## Demo Workload

The example template runs a PyTorch ROCm workload when available:

```
scripts/demo_pytorch_rocm.py --seconds 60 --size 4096 --dtype fp16
```

If the demo script is missing, the template falls back to `./your_application`.

## Output Format

`summary.json` contains per-node, per-GPU aggregates (avg, p95, max) for common metrics when present in `rocm-smi` output:

- GPU utilization
- VRAM utilization
- Power
- Temperature
- Core/memory clocks

The parser is best‑effort and tolerant of missing fields.

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
