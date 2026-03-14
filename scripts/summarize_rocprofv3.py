#!/usr/bin/env python3
"""Summarize rocprofv3 trace artifacts and emit a deep-trace manifest."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path


TRACE_SCHEMA_VERSION = 1
DEEP_MANIFEST_SCHEMA_VERSION = 1
TOP_N = 5
TRACE_ROW_SUFFIX = "_trace"
STATS_SUFFIX = "_stats"


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_number(value):
    if value in (None, ""):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return None


def first_value(row, candidates):
    for key in candidates:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def classify_artifact(path):
    stem = path.stem
    if stem.endswith(STATS_SUFFIX):
        category = "stats"
        domain = stem[: -len(STATS_SUFFIX)]
    elif stem.endswith(TRACE_ROW_SUFFIX):
        category = "trace"
        domain = stem[: -len(TRACE_ROW_SUFFIX)]
    else:
        category = "other"
        domain = stem
    if domain.startswith("trace_"):
        domain = domain[len("trace_") :]
    return category, domain


def normalize_top_row(row):
    return {
        "name": first_value(row, ["Name", "KernelName", "Function", "Operation", "Kind"]) or "unknown",
        "calls": as_number(first_value(row, ["Calls", "Count", "Instances"])),
        "total_duration_ns": as_number(
            first_value(
                row,
                [
                    "TotalDurationNs",
                    "TotalDurationNs_Mean",
                    "TotalTimeNs",
                    "DurationNs",
                    "TotalDuration",
                ],
            )
        ),
        "average_duration_ns": as_number(first_value(row, ["AverageNs", "AverageDurationNs", "AvgNs"])),
        "percentage": as_number(first_value(row, ["Percentage", "Pct", "Percent"])),
    }


def summarize_stats_file(path):
    rows = read_csv_rows(path)

    def sort_key(row):
        total = as_number(
            first_value(
                row,
                [
                    "TotalDurationNs",
                    "TotalDurationNs_Mean",
                    "TotalTimeNs",
                    "DurationNs",
                    "TotalDuration",
                ],
            )
        )
        calls = as_number(first_value(row, ["Calls", "Count", "Instances"]))
        return (total if total is not None else -1, calls if calls is not None else -1)

    top_rows = sorted(rows, key=sort_key, reverse=True)[:TOP_N]
    return {
        "path": str(path),
        "row_count": len(rows),
        "top": [normalize_top_row(row) for row in top_rows],
    }


def summarize_trace_file(path):
    rows = read_csv_rows(path)
    return {
        "path": str(path),
        "row_count": len(rows),
    }


def build_trace_summary(raw_dir, tool_path, mode, command, status, exit_code):
    raw_path = Path(raw_dir)
    artifacts = []
    trace_stats = {}
    stats_summaries = {}
    warnings = []

    if not raw_path.exists():
        warnings.append(f"Trace directory does not exist: {raw_path}")
    else:
        for path in sorted(raw_path.iterdir()):
            if path.is_dir():
                continue
            artifacts.append(str(path))
            if path.suffix.lower() != ".csv":
                continue
            category, domain = classify_artifact(path)
            if category == "stats":
                stats_summaries[domain] = summarize_stats_file(path)
            elif category == "trace":
                trace_stats[domain] = summarize_trace_file(path)

    preview = {
        "stats_files_discovered": len(stats_summaries),
        "trace_files_discovered": len(trace_stats),
        "hip_api_trace_rows": trace_stats.get("hip_api", {}).get("row_count"),
        "kernel_dispatch_trace_rows": trace_stats.get("kernel_dispatch", {}).get("row_count"),
        "memory_copy_trace_rows": trace_stats.get("memory_copy", {}).get("row_count"),
        "top_hip_apis": stats_summaries.get("hip_api", {}).get("top", []),
        "top_kernel_dispatches": stats_summaries.get("kernel_dispatch", {}).get("top", []),
        "top_memory_copies": stats_summaries.get("memory_copy", {}).get("top", []),
    }

    if not artifacts:
        warnings.append("No rocprofv3 artifacts were discovered.")

    return {
        "deep_trace_schema_version": TRACE_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "mode": mode,
        "status": status,
        "tool": {
            "name": "rocprofv3",
            "path": tool_path,
        },
        "command": command,
        "exit_code": exit_code,
        "trace_dir": str(raw_path),
        "artifacts": artifacts,
        "trace_stats": trace_stats,
        "stats": stats_summaries,
        "preview": preview,
        "warnings": warnings,
    }


def build_deep_manifest(trace_summary, summary_output, manifest_output):
    summary_path = Path(summary_output)
    manifest_path = Path(manifest_output)
    return {
        "deep_manifest_schema_version": DEEP_MANIFEST_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "mode": trace_summary.get("mode"),
        "status": trace_summary.get("status"),
        "tool": trace_summary.get("tool", {}),
        "command": trace_summary.get("command"),
        "exit_code": trace_summary.get("exit_code"),
        "artifacts": {
            "trace_summary": str(summary_path),
            "trace_raw_dir": trace_summary.get("trace_dir"),
            "deep_manifest": str(manifest_path),
        },
        "trace_summary_preview": trace_summary.get("preview", {}),
        "warnings": trace_summary.get("warnings", []),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", help="Directory containing rocprofv3 outputs")
    parser.add_argument("summary_output", help="Output JSON file for the deep trace summary")
    parser.add_argument("manifest_output", help="Output JSON file for the deep manifest")
    parser.add_argument("--tool-path", default="", help="Resolved rocprofv3 executable path")
    parser.add_argument("--mode", default="deep-trace", help="Deep profiling mode name")
    parser.add_argument("--command", default="", help="Command executed under rocprofv3")
    parser.add_argument("--status", default="completed", help="Trace collection status")
    parser.add_argument("--exit-code", type=int, default=0, help="Wrapped command exit code")
    args = parser.parse_args()

    trace_summary = build_trace_summary(
        raw_dir=args.raw_dir,
        tool_path=args.tool_path,
        mode=args.mode,
        command=args.command,
        status=args.status,
        exit_code=args.exit_code,
    )
    manifest = build_deep_manifest(trace_summary, args.summary_output, args.manifest_output)

    write_json(args.summary_output, trace_summary)
    write_json(args.manifest_output, manifest)


if __name__ == "__main__":
    main()
