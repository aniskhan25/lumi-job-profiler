# LUMI Job Profiler (Demo)

This repo contains a **user opt-in** profiling demo for LUMI GPU jobs (AMD/ROCm). It samples GPU metrics with `rocm-smi` during a job and produces a compact JSON summary at the end.

## What’s Included

- `templates/sbatch_profiled.sh`: opt-in Slurm job template with profiling sidecar
- `scripts/summarize_rocm_smi.py`: best-effort parser that generates `summary.json`
- `scripts/demo_pytorch_rocm.py`: a PyTorch ROCm demo workload to generate GPU activity
- `implementation_plan.md`: system-level plan for a full feedback loop

## Requirements

- LUMI login/compute environment
- Slurm `sbatch`/`srun`
- ROCm installed on compute nodes (`rocm-smi` available)
- PyTorch module on LUMI (template uses `pytorch/2.7`)

## Quick Start (Opt‑In Profiling)

1. Edit job settings in `templates/sbatch_profiled.sh` if needed.
2. Submit the job:

```bash
sbatch templates/sbatch_profiled.sh
```

3. After the job completes, find logs and summary here:

```
$SCRATCH/lumi-profile/$SLURM_JOB_ID/
  <node>.log
  summary.json
```

## Opt‑In Controls

The profiling sidecar is enabled by default. You can override behavior with:

- `LUMI_PROFILE=0` to disable
- `PROFILE_INTERVAL=2` to change sampling interval (seconds)
- `PROFILER_SRUN_OPTS="--ntasks-per-node=1 --cpus-per-task=1 --mpi=none --overlap"` to adjust the sidecar launch

Example:

```bash
LUMI_PROFILE=1 PROFILE_INTERVAL=1 sbatch templates/sbatch_profiled.sh
```

## Demo Workload

The template runs a PyTorch ROCm workload when available:

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
