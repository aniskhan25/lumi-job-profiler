#!/usr/bin/env python3
"""Summarize rocprofiler-systems artifacts and emit a deep-profile manifest."""

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deep_profile_common import build_deep_manifest as build_manifest
from deep_profile_common import iso_timestamp, write_json


SYSTEM_SCHEMA_VERSION = 1


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
    return build_manifest(system_summary, summary_output, manifest_output, system_summary.get("raw_dir", ""))


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
