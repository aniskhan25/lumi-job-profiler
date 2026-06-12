#!/usr/bin/env python3
"""Summarize rocprofv3 trace artifacts and emit a deep-trace manifest."""

import argparse
import csv
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deep_profile_common import build_deep_manifest as build_manifest
from deep_profile_common import iso_timestamp, write_json


TRACE_SCHEMA_VERSION = 1
TOP_N = 5
TRACE_ROW_SUFFIX = "_trace"
STATS_SUFFIX = "_stats"
RUNTIME_BUFFER_KEYS = (
    "kernel_dispatch",
    "hip_api",
    "hsa_api",
    "memory_copy",
    "marker_api",
    "rccl_api",
    "scratch_memory",
)
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
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def read_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


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
    domain = canonical_domain(domain)
    return category, domain


def canonical_domain(domain):
    aliases = {
        "kernel": "kernel_dispatch",
    }
    return aliases.get(domain, domain)


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


def summarize_trace_results_json(path):
    payload = read_json(path)
    entries = payload.get("rocprofiler-sdk-tool", [])
    if not isinstance(entries, list) or not entries:
        return {
            "path": str(path),
            "buffer_record_counts": {},
            "summary_entry_count": 0,
        }

    entry = entries[0] if isinstance(entries[0], dict) else {}
    buffer_records = entry.get("buffer_records", {})
    counts = {}
    if isinstance(buffer_records, dict):
        for key in RUNTIME_BUFFER_KEYS:
            value = buffer_records.get(key, [])
            counts[key] = len(value) if isinstance(value, list) else None

    summary_entries = entry.get("summary", [])
    summary_entry_count = len(summary_entries) if isinstance(summary_entries, list) else None
    return {
        "path": str(path),
        "buffer_record_counts": counts,
        "summary_entry_count": summary_entry_count,
    }


def collect_distributed_layout(paths, raw_path):
    node_dirs = set()
    rank_dirs = set()
    for path in paths:
        try:
            relative = path.relative_to(raw_path)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) >= 3 and parts[1].startswith("rank-"):
            node_dirs.add(parts[0])
            rank_dirs.add("/".join(parts[:2]))
    return sorted(node_dirs), sorted(rank_dirs)


def build_trace_summary(raw_dir, tool_path, mode, command, status, exit_code):
    raw_path = Path(raw_dir)
    artifacts = []
    trace_stats = {}
    stats_summaries = {}
    warnings = []
    trace_results_summary = None
    trace_results_summaries = []
    node_dirs = []
    rank_dirs = []

    if not raw_path.exists():
        warnings.append(f"Trace directory does not exist: {raw_path}")
    else:
        csv_trace_candidates = {}
        csv_stats_candidates = {}
        discovered_paths = []
        for path in sorted(raw_path.rglob("*")):
            if path.is_dir():
                continue
            discovered_paths.append(path)
            artifacts.append(str(path))
            if path.name == "trace_results.json":
                trace_results_summaries.append(summarize_trace_results_json(path))
                continue
            if path.suffix.lower() != ".csv":
                continue
            category, domain = classify_artifact(path)
            if category == "stats":
                csv_stats_candidates.setdefault(domain, []).append(summarize_stats_file(path))
            elif category == "trace":
                csv_trace_candidates.setdefault(domain, []).append(summarize_trace_file(path))

        for domain, file_summaries in csv_trace_candidates.items():
            trace_stats[domain] = {
                "file_count": len(file_summaries),
                "row_count": sum(item["row_count"] for item in file_summaries),
                "sample_paths": [item["path"] for item in file_summaries[:3]],
            }

        for domain, file_summaries in csv_stats_candidates.items():
            top_candidates = []
            for item in file_summaries:
                top_candidates.extend(item["top"])
            top_candidates = sorted(
                top_candidates,
                key=lambda row: (
                    row.get("total_duration_ns") or -1,
                    row.get("calls") or -1,
                ),
                reverse=True,
            )[:TOP_N]
            stats_summaries[domain] = {
                "file_count": len(file_summaries),
                "row_count": sum(item["row_count"] for item in file_summaries),
                "sample_paths": [item["path"] for item in file_summaries[:3]],
                "top": top_candidates,
            }

        node_dirs, rank_dirs = collect_distributed_layout(discovered_paths, raw_path)

    runtime_record_counts = {}
    if trace_results_summaries:
        runtime_record_counts = {key: 0 for key in RUNTIME_BUFFER_KEYS}
        summary_entry_count = 0
        for item in trace_results_summaries:
            counts = item.get("buffer_record_counts", {})
            for key in RUNTIME_BUFFER_KEYS:
                value = counts.get(key)
                if value is None:
                    runtime_record_counts[key] = None
                elif runtime_record_counts.get(key) is not None:
                    runtime_record_counts[key] += value
            if item.get("summary_entry_count") is not None:
                summary_entry_count += item["summary_entry_count"]
        trace_results_summary = {
            "file_count": len(trace_results_summaries),
            "sample_paths": [item["path"] for item in trace_results_summaries[:3]],
            "buffer_record_counts": runtime_record_counts,
            "summary_entry_count": summary_entry_count,
        }

    effective_hip_api_rows = trace_stats.get("hip_api", {}).get("row_count")
    effective_kernel_dispatch_rows = trace_stats.get("kernel_dispatch", {}).get("row_count")
    effective_memory_copy_rows = trace_stats.get("memory_copy", {}).get("row_count")
    if effective_hip_api_rows is None:
        effective_hip_api_rows = runtime_record_counts.get("hip_api")
    if effective_kernel_dispatch_rows is None:
        effective_kernel_dispatch_rows = runtime_record_counts.get("kernel_dispatch")
    if effective_memory_copy_rows is None:
        effective_memory_copy_rows = runtime_record_counts.get("memory_copy")

    preview = {
        "stats_files_discovered": len(stats_summaries),
        "trace_files_discovered": len(trace_stats),
        "hip_api_trace_rows": effective_hip_api_rows,
        "kernel_dispatch_trace_rows": effective_kernel_dispatch_rows,
        "memory_copy_trace_rows": effective_memory_copy_rows,
        "top_hip_apis": stats_summaries.get("hip_api", {}).get("top", []),
        "top_kernel_dispatches": stats_summaries.get("kernel_dispatch", {}).get("top", []),
        "top_memory_copies": stats_summaries.get("memory_copy", {}).get("top", []),
        "runtime_record_counts": runtime_record_counts,
        "distributed_node_count": len(node_dirs),
        "distributed_rank_count": len(rank_dirs),
        "distributed_node_sample": node_dirs[:3],
        "distributed_rank_sample": rank_dirs[:3],
    }

    if not artifacts:
        warnings.append("No rocprofv3 artifacts were discovered.")
    elif trace_results_summary and runtime_record_counts and not any(
        (value or 0) > 0 for value in runtime_record_counts.values() if value is not None
    ):
        warnings.append(
            "rocprofv3 produced metadata artifacts but captured no runtime events. "
            "This usually indicates a runtime attachment or tool initialization failure on the target environment."
        )
        if status == "completed":
            status = "completed_without_runtime_events"

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
        "trace_results": trace_results_summary,
        "preview": preview,
        "warnings": warnings,
    }


def build_deep_manifest(trace_summary, summary_output, manifest_output):
    return build_manifest(trace_summary, summary_output, manifest_output, trace_summary.get("trace_dir", ""))


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
