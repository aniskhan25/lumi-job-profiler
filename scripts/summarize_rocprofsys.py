#!/usr/bin/env python3
"""Summarize rocprofiler-systems artifacts and emit a deep-profile manifest."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


SYSTEM_SCHEMA_VERSION = 1
DEEP_MANIFEST_SCHEMA_VERSION = 1


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_system_summary(raw_dir, tool_path, mode, command, status, exit_code):
    raw_path = Path(raw_dir)
    artifacts = []
    warnings = []
    perfetto_files = []
    metadata_files = []
    functions_files = []

    if not raw_path.exists():
        warnings.append(f"System profile directory does not exist: {raw_path}")
    else:
        for path in sorted(raw_path.rglob("*")):
            if path.is_dir():
                continue
            artifacts.append(str(path))
            if path.name.startswith("perfetto-trace-") and path.suffix == ".proto":
                perfetto_files.append(str(path))
            elif path.name.startswith("metadata-") and path.suffix == ".json":
                metadata_files.append(str(path))
            elif path.name.startswith("functions-") and path.suffix == ".json":
                functions_files.append(str(path))

    preview = {
        "perfetto_trace_files": len(perfetto_files),
        "metadata_files": len(metadata_files),
        "functions_files": len(functions_files),
        "perfetto_trace_sample": perfetto_files[:3],
        "metadata_sample": metadata_files[:3],
        "functions_sample": functions_files[:3],
    }

    if not artifacts:
        warnings.append("No rocprofiler-systems artifacts were discovered.")
    elif not perfetto_files:
        warnings.append("rocprofiler-systems completed but did not produce a perfetto trace.")
        if status == "completed":
            status = "completed_without_perfetto_trace"

    return {
        "deep_system_schema_version": SYSTEM_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "mode": mode,
        "status": status,
        "tool": {
            "name": "rocprofiler-systems",
            "path": tool_path,
        },
        "command": command,
        "exit_code": exit_code,
        "raw_dir": str(raw_path),
        "artifacts": artifacts,
        "preview": preview,
        "warnings": warnings,
    }


def build_deep_manifest(system_summary, summary_output, manifest_output):
    summary_path = Path(summary_output)
    manifest_path = Path(manifest_output)
    return {
        "deep_manifest_schema_version": DEEP_MANIFEST_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "mode": system_summary.get("mode"),
        "status": system_summary.get("status"),
        "tool": system_summary.get("tool", {}),
        "command": system_summary.get("command"),
        "exit_code": system_summary.get("exit_code"),
        "artifacts": {
            "trace_summary": str(summary_path),
            "trace_raw_dir": system_summary.get("raw_dir"),
            "deep_manifest": str(manifest_path),
        },
        "trace_summary_preview": system_summary.get("preview", {}),
        "warnings": system_summary.get("warnings", []),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", help="Directory containing rocprofiler-systems outputs")
    parser.add_argument("summary_output", help="Output JSON file for the deep system summary")
    parser.add_argument("manifest_output", help="Output JSON file for the deep manifest")
    parser.add_argument("--tool-path", default="", help="Resolved rocprofiler-systems launcher path")
    parser.add_argument("--mode", default="deep-system", help="Deep profiling mode name")
    parser.add_argument("--command", default="", help="Command executed under rocprofiler-systems")
    parser.add_argument("--status", default="completed", help="Collection status")
    parser.add_argument("--exit-code", type=int, default=0, help="Wrapped command exit code")
    args = parser.parse_args()

    system_summary = build_system_summary(
        raw_dir=args.raw_dir,
        tool_path=args.tool_path,
        mode=args.mode,
        command=args.command,
        status=args.status,
        exit_code=args.exit_code,
    )
    manifest = build_deep_manifest(system_summary, args.summary_output, args.manifest_output)
    write_json(args.summary_output, system_summary)
    write_json(args.manifest_output, manifest)


if __name__ == "__main__":
    main()
