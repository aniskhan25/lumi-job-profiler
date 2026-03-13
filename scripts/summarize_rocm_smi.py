#!/usr/bin/env python3
"""Summarize rocm-smi log files into a compact JSON report."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean

SUMMARY_SCHEMA_VERSION = 1
DEFAULT_ACTIVE_GPU_THRESHOLD_PCT = 10.0
HEADER_PREFIX = "# "
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
GPU_LINE_RE = re.compile(r"^\s*\d+\b")
KV_LINE_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*(?P<label>.+?)\s*:\s*(?P<value>.+)$")


def parse_number(token):
    match = NUM_RE.search(token)
    return float(match.group(0)) if match else None


def parse_value_with_units(text):
    """Parse values like '(400Mhz)' or '400 MHz' into MHz (float)."""
    match = re.search(r"\((\d+(?:\.\d+)?)\s*([A-Za-z]+)\)", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)?", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit in ("mhz", "m"):
        return value
    if unit in ("ghz", "g"):
        return value * 1000.0
    return value


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def normalize_header(token):
    return token.strip().lower().replace("(", "").replace(")", "").replace("%", "%")


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_header_metadata(lines):
    metadata = {}
    for line in lines:
        if not line.startswith(HEADER_PREFIX):
            continue
        payload = line[len(HEADER_PREFIX):]
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def extract_tables(lines):
    """Yield (header_tokens, data_lines) blocks."""
    i = 0
    while i < len(lines):
        line = lines[i]
        if "GPU" in line and "%" in line and "ts=" not in line:
            header = line.strip()
            header_tokens = header.split()
            data_lines = []
            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "---" or lines[j].startswith("ts="):
                    break
                if GPU_LINE_RE.match(lines[j]):
                    data_lines.append(lines[j].rstrip())
                j += 1
            if data_lines:
                yield header_tokens, data_lines
            i = j
            continue
        i += 1


def temperature_key(label):
    label_lower = label.lower()
    if "sensor edge" in label_lower:
        return "temp_edge_c"
    if "sensor junction" in label_lower:
        return "temp_junction_c"
    if "sensor memory" in label_lower:
        return "temp_memory_c"
    if "sensor hbm 0" in label_lower:
        return "temp_hbm0_c"
    if "sensor hbm 1" in label_lower:
        return "temp_hbm1_c"
    if "sensor hbm 2" in label_lower:
        return "temp_hbm2_c"
    if "sensor hbm 3" in label_lower:
        return "temp_hbm3_c"
    if "temperature" in label_lower:
        return "temp_c"
    return None


def parse_kv_lines(lines):
    """Parse key/value style rocm-smi output lines."""
    metrics = defaultdict(lambda: defaultdict(list))

    for line in lines:
        match = KV_LINE_RE.match(line.strip())
        if not match:
            continue

        gpu_id = match.group("gpu")
        label = match.group("label").strip()
        value = match.group("value").strip()

        if label == "GPU use (%)":
            val = parse_number(value)
            if val is not None:
                metrics[gpu_id]["gpu_util_pct"].append(val)
            continue

        if label == "GPU Memory Allocated (VRAM%)":
            val = parse_number(value)
            if val is not None:
                metrics[gpu_id]["vram_util_pct"].append(val)
            continue

        if label == "GPU Memory Read/Write Activity (%)":
            val = parse_number(value)
            if val is not None:
                metrics[gpu_id]["mem_rw_activity_pct"].append(val)
            continue

        if label == "Average Graphics Package Power (W)":
            val = parse_number(value)
            if val is not None:
                metrics[gpu_id]["power_w"].append(val)
            continue

        if label.startswith("fclk clock level"):
            val = parse_value_with_units(value)
            if val is not None:
                metrics[gpu_id]["fclk_mhz"].append(val)
            continue

        if label.startswith("mclk clock level"):
            val = parse_value_with_units(value)
            if val is not None:
                metrics[gpu_id]["mclk_mhz"].append(val)
            continue

        if label.startswith("sclk clock level"):
            val = parse_value_with_units(value)
            if val is not None:
                metrics[gpu_id]["sclk_mhz"].append(val)
            continue

        if label.startswith("socclk clock level"):
            val = parse_value_with_units(value)
            if val is not None:
                metrics[gpu_id]["socclk_mhz"].append(val)
            continue

        temp_key = temperature_key(label)
        if temp_key:
            val = parse_number(value)
            if val is not None:
                metrics[gpu_id][temp_key].append(val)
            continue

    return metrics


def parse_sample(lines):
    """Parse a single rocm-smi sample block into per-gpu metrics."""
    metrics = defaultdict(lambda: defaultdict(list))

    for header_tokens, data_lines in extract_tables(lines):
        header = [normalize_header(t) for t in header_tokens]

        def find_col(*names):
            for name in names:
                if name in header:
                    return header.index(name)
            return None

        col_gpu = find_col("gpu")
        col_gpu_util = find_col("gpu%", "gpu%", "gpuuse%", "gpuuse")
        col_vram = find_col("vram%", "mem%", "memuse%")
        col_temp = find_col("temp", "temperature")
        col_power = find_col("avgpwr", "power", "pwr", "avgpower")
        col_sclk = find_col("sclk")
        col_mclk = find_col("mclk")

        for line in data_lines:
            tokens = line.split()
            if col_gpu is None or col_gpu >= len(tokens):
                continue
            gpu_id = tokens[col_gpu]

            def grab(col):
                if col is None or col >= len(tokens):
                    return None
                return parse_number(tokens[col])

            val = grab(col_gpu_util)
            if val is not None:
                metrics[gpu_id]["gpu_util_pct"].append(val)

            val = grab(col_vram)
            if val is not None:
                metrics[gpu_id]["vram_util_pct"].append(val)

            val = grab(col_temp)
            if val is not None:
                metrics[gpu_id]["temp_c"].append(val)

            val = grab(col_power)
            if val is not None:
                metrics[gpu_id]["power_w"].append(val)

            val = grab(col_sclk)
            if val is not None:
                metrics[gpu_id]["sclk_mhz"].append(val)

            val = grab(col_mclk)
            if val is not None:
                metrics[gpu_id]["mclk_mhz"].append(val)

    kv_metrics = parse_kv_lines(lines)
    for gpu_id, gpu_metrics in kv_metrics.items():
        for key, values in gpu_metrics.items():
            metrics[gpu_id][key].extend(values)

    return metrics


def summarize_metric(values):
    if not values:
        return None
    return {
        "avg": mean(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def infer_interval_seconds(timestamps):
    if len(timestamps) < 2:
        return None
    deltas = []
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return None
    return mean(deltas)


def build_job_metadata(log_dir):
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "project_id": os.environ.get("SLURM_JOB_ACCOUNT"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "nodes_requested": os.environ.get("SLURM_JOB_NUM_NODES"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "tasks_per_node": os.environ.get("SLURM_TASKS_PER_NODE"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "log_dir": os.path.abspath(log_dir),
    }


def build_collection_metadata(log_dir, node_headers, node_stats):
    schema_versions = sorted(
        {
            header["profile_log_schema_version"]
            for header in node_headers.values()
            if header.get("profile_log_schema_version")
        }
    )
    commands = sorted(
        {
            header["profile_collect_command"]
            for header in node_headers.values()
            if header.get("profile_collect_command")
        }
    )
    intervals = [
        stats["interval_seconds"]
        for stats in node_stats.values()
        if stats.get("interval_seconds") is not None
    ]
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "log_dir": os.path.abspath(log_dir),
        "raw_log_schema_versions": schema_versions,
        "collect_commands": commands,
        "sampling_interval_seconds": mean(intervals) if intervals else None,
    }


def compute_job_metrics(summary, active_gpu_threshold_pct=DEFAULT_ACTIVE_GPU_THRESHOLD_PCT):
    gpu_utils = []
    vram_utils = []
    active_gpu_estimates = []
    total_gpu_slots = 0
    nodes_with_metrics = 0

    for node_stats in summary["nodes"].values():
        if not node_stats["gpus"]:
            continue
        nodes_with_metrics += 1
        total_gpu_slots += len(node_stats["gpus"])
        node_active = 0

        for gpu_stats in node_stats["gpus"].values():
            util_stats = gpu_stats.get("gpu_util_pct")
            if util_stats and util_stats.get("avg") is not None:
                gpu_utils.append(util_stats["avg"])
                if util_stats["avg"] >= active_gpu_threshold_pct:
                    node_active += 1

            vram_stats = gpu_stats.get("vram_util_pct")
            if vram_stats and vram_stats.get("max") is not None:
                vram_utils.append(vram_stats["max"])

        active_gpu_estimates.append(node_active)

    avg_gpu_util = mean(gpu_utils) if gpu_utils else None
    peak_vram_util = max(vram_utils) if vram_utils else None
    total_active_gpus = sum(active_gpu_estimates)

    return {
        "active_gpu_threshold_pct": active_gpu_threshold_pct,
        "nodes_with_gpu_metrics": nodes_with_metrics,
        "total_gpu_slots_observed": total_gpu_slots,
        "total_active_gpus_estimate": total_active_gpus,
        "effective_gpus_estimate": total_active_gpus,
        "avg_gpu_util_pct": avg_gpu_util,
        "peak_vram_util_pct": peak_vram_util,
    }


def summarize_logs(log_dir):
    summary = {
        "log_dir": os.path.abspath(log_dir),
        "nodes": {},
        "warnings": [],
    }
    node_headers = {}
    node_stats_map = {}

    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".log"):
            continue
        path = os.path.join(log_dir, name)
        node = os.path.splitext(name)[0]
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f]

        header_metadata = parse_header_metadata(lines)
        timestamps = []
        sample_blocks = []
        current = []
        seen_timestamp = False
        for line in lines:
            if line.startswith("ts="):
                if current and seen_timestamp:
                    sample_blocks.append(current)
                try:
                    timestamps.append(int(line.split("=", 1)[1]))
                except ValueError:
                    pass
                current = [line]
                seen_timestamp = True
            elif line.strip() == "---":
                if current and seen_timestamp:
                    sample_blocks.append(current)
                    current = []
            else:
                if seen_timestamp:
                    current.append(line)
        if current and seen_timestamp:
            sample_blocks.append(current)

        node_stats = {
            "log_file": path,
            "samples": len(sample_blocks),
            "start_ts": min(timestamps) if timestamps else None,
            "end_ts": max(timestamps) if timestamps else None,
            "duration_seconds": (
                max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
            ),
            "interval_seconds": infer_interval_seconds(timestamps),
            "metadata": header_metadata,
            "gpus": {},
        }

        combined = defaultdict(lambda: defaultdict(list))
        for block in sample_blocks:
            metrics = parse_sample(block)
            for gpu_id, gpu_metrics in metrics.items():
                for key, values in gpu_metrics.items():
                    combined[gpu_id][key].extend(values)

        if not combined:
            summary["warnings"].append(f"No parseable metrics in {name}")

        for gpu_id, gpu_metrics in combined.items():
            node_stats["gpus"][gpu_id] = {
                key: summarize_metric(values)
                for key, values in gpu_metrics.items()
            }

        summary["nodes"][node] = node_stats
        node_headers[node] = header_metadata
        node_stats_map[node] = node_stats

    summary["collection"] = build_collection_metadata(log_dir, node_headers, node_stats_map)
    summary["job"] = build_job_metadata(log_dir)
    summary["job_metrics"] = compute_job_metrics(summary)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Directory containing *.log files")
    parser.add_argument("output", nargs="?", default=None, help="Output JSON file")
    args = parser.parse_args()

    summary = summarize_logs(args.log_dir)
    output = args.output

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
