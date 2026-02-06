#!/usr/bin/env python3
"""Summarize rocm-smi log files into a compact JSON report.

This is a best-effort parser intended for demo use. It extracts common
columns (GPU%, VRAM%, power, temp, clocks) when present, and always emits
sample counts and timestamps.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
GPU_LINE_RE = re.compile(r"^\s*\d+\b")


def parse_number(token):
    match = NUM_RE.search(token)
    return float(match.group(0)) if match else None


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

    return metrics


def summarize_metric(values):
    if not values:
        return None
    return {
        "avg": mean(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def summarize_logs(log_dir):
    summary = {
        "log_dir": os.path.abspath(log_dir),
        "nodes": {},
        "warnings": [],
    }

    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".log"):
            continue
        path = os.path.join(log_dir, name)
        node = os.path.splitext(name)[0]
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f]

        timestamps = []
        sample_blocks = []
        current = []
        for line in lines:
            if line.startswith("ts="):
                if current:
                    sample_blocks.append(current)
                    current = []
                try:
                    timestamps.append(int(line.split("=", 1)[1]))
                except ValueError:
                    pass
                current.append(line)
            elif line.strip() == "---":
                if current:
                    sample_blocks.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            sample_blocks.append(current)

        node_stats = {
            "log_file": path,
            "samples": len(sample_blocks),
            "start_ts": min(timestamps) if timestamps else None,
            "end_ts": max(timestamps) if timestamps else None,
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
