# Post-Job GPU Feedback & Education Loop
## Detailed Implementation Plan (Software Development Focus)

This document specifies implementation-level details for building the post-job feedback and education loop on LUMI (AMD GPUs, ROCm stack). It is written for a mixed HPC systems + software engineering team and focuses on data flow, components, interfaces, and logic (not timelines or costs).

---

## 1. System Architecture Overview

### 1.1 High-Level Components

```
SLURM Scheduler
  | job start / end hooks
  v
Job Prologue
  |
  v
Metrics Collectors ---> Metrics Ingest API / Daemon
  |
  v
Job Epilogue --------> Analysis Engine
  |
  v
SLURM DB ------------> Report Generator
                          |
                          v
                    User Delivery (email / portal)
```

---

## 2. Data Collection Layer

### 2.1 Job Identification & Metadata

Source
- SLURM environment variables
- `sacct`, `scontrol`
- Job launch context (`srun` or `sbatch`)

Captured fields
- `job_id`
- `user_id`
- `project_id`
- `job_name`
- `partition`
- `nodes`
- `gpus_requested`
- `gpus_per_node`
- `walltime_requested`
- `walltime_used`
- `exit_code`

Implementation
- Parsed once in job epilogue
- Serialized as JSON
- Stored in central job record table

### 2.2 GPU Metrics Collection

Collection mechanisms
- ROCm SMI (`rocm-smi`) sampling
- Slurm GPU accounting (where available)
- Lightweight sidecar process per node

Metrics
- GPU utilization (sampled, per GPU)
- GPU memory usage (avg / peak)
- GPU active vs idle time
- GPU count with non-zero utilization

Optional ROCm-specific metrics (if available)
- `gfx_clock` (core clock), `mem_clock`
- `power` (average / peak)
- `temperature`
- `pcie_bandwidth`

Sampling strategy
- 1 to 5 second sampling interval
- Rolling aggregation on node
- Summary statistics only forwarded

ROCm sampling caveats
- `rocm-smi` counters can vary by GPU model and driver version
- Prefer per-GCD or per-GPU consistency checks before aggregating
- Document missing fields explicitly instead of zero-filling

Output format
```json
{
  "job_id": 123456,
  "node": "nid00123",
  "gpu_id": 0,
  "avg_util": 22.1,
  "p95_util": 41.3,
  "peak_mem_gb": 14.2
}
```

### 2.3 CPU & Process Metrics

Metrics
- CPU utilization per task
- CPU wait / idle time
- MPI rank count
- Threads per rank

Purpose
- Infer CPU bottlenecks
- Detect GPU oversubscription or undersubscription

---

## 3. Metrics Ingest & Storage

### 3.1 Ingest API / Daemon

Responsibilities
- Accept metric payloads from compute nodes
- Validate schema
- Associate metrics with job records
- Perform basic aggregation

Design
- Stateless service
- Writes to time-series or relational store
- Backpressure-safe (drop raw samples if overloaded)

### 3.2 Data Storage Model

Tables / collections
- `jobs`
- `gpu_metrics_summary`
- `cpu_metrics_summary`
- `job_efficiency_results`

Key design decision
- Store aggregated summaries only, not full traces
- Retain raw samples only for short diagnostic window (optional)

---

## 4. Analysis Engine

### 4.1 Efficiency Metric Computation

For each job:

Derived metrics
- Average GPU utilization (job-wide)
- Peak GPU memory usage
- Effective GPUs used

```
effective_gpus = sum(util_gpu_i > threshold) / total_gpus
```

- GPU-hours wasted estimate

### 4.2 Job Classification Logic

Efficiency classes
- Efficient: >= 70%
- Acceptable: 40 to 69%
- Inefficient: 15 to 39%
- Wasted: < 15%

Stored as
```json
{
  "job_id": 123456,
  "efficiency_class": "INEFFICIENT",
  "avg_gpu_util": 21.3
}
```

### 4.3 Root-Cause Heuristic Engine

Rule-based inference
- Low GPU util + low memory -> overscaling
- Low GPU util + high CPU -> CPU bottleneck
- Util drop after start -> I/O or sync
- High GPU count + few active GPUs -> parallelism mismatch

Implementation
- Deterministic rule set
- Multiple causes allowed
- Each cause tagged with confidence score

---

## 5. Recommendation Engine

### 5.1 Resource Right-Sizing

Logic
- Suggest GPU count based on:
- Effective GPUs used
- Peak memory footprint
- Scaling efficiency estimate

Example
```json
{
  "recommended_gpus": 2,
  "reason": "Only 2 GPUs showed sustained utilization >30%"
}
```

### 5.2 Job Script Modification Generator

Inputs
- Original `sbatch` options
- Recommendation results

Outputs
- Diff-style suggestions:

```
--gpus=8        -> --gpus=2
--ntasks=16     -> --ntasks=2
```

No automatic modification; advisory only.

---

## 6. Report Generation

### 6.1 Report Template Engine

Sections
- Job summary
- Observed resource usage
- Efficiency classification
- Root-cause explanation
- Recommendations
- Estimated savings

Format
- Markdown -> HTML
- Plain-text fallback

### 6.2 Visualization Generation

Graphs
- GPU utilization over time
- Memory usage vs capacity
- Per-GPU utilization heatmap

Constraints
- Max 3 visuals per report
- Pre-rendered images (PNG/SVG)
- No interactive UI required

---

## 7. Educational Mapping Layer

### 7.1 Issue-to-Resource Mapping

Lookup table
```json
{
  "overscaling": "docs/gpu-scaling.html",
  "cpu_bottleneck": "docs/data-loaders.html",
  "mpi_mismatch": "docs/mpi-gpu-mapping.html",
  "rocm_hip_tuning": "docs/rocm-hip-tuning.html"
}
```

Injection
- Append links to report
- Add short explanatory text block

### 7.2 Micro-Lesson Generator

Pattern
- 2 to 3 sentences
- Focused on detected issue
- Uses neutral, non-judgmental language
- Include ROCm/HIP-specific guidance when applicable

---

## 8. User Delivery Mechanisms

### 8.1 Email Delivery

Trigger
- Job completion
- Only for GPU jobs

Controls
- Opt-out available
- Batched for multi-job users

### 8.2 Web Portal Integration

Features
- Job efficiency dashboard
- Historical trends
- Downloadable reports

Authentication
- Existing LUMI identity system

---

## 9. User & Project Efficiency History

### 9.1 Rolling Aggregation

Stored metrics
- Percentage of jobs per efficiency class
- Median GPU utilization
- Trend indicators

### 9.2 Escalation Hooks

Logic
- Repeated inefficiency triggers advisory flag
- Flag exposed to support staff and schedulers

---

## 10. Policy Integration (Read-Only)

Exports
- Efficiency scores
- Waste estimates

Usage
- Inform quota multipliers
- Inform priority decisions
- No automatic enforcement at this stage

---

## 11. Observability & Maintenance

### 11.1 System Health Metrics

- Missing job reports
- Collector failures
- Analysis latency

### 11.2 Safe Failure Modes

- If metrics missing: generate partial report
- Never block job completion
- Never auto-cancel jobs

---

## 12. Security & Privacy

- Metrics visible only to job owner + project PI
- Aggregated stats anonymized
- No command-line content captured

---

## End State

The implemented system produces consistent, actionable post-job feedback, reinforces efficient GPU usage behavior, and creates a data foundation for fair scheduling and policy decisions without introducing punitive or intrusive controls.
