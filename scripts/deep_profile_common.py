#!/usr/bin/env python3
"""Shared helpers for deep-profile summary scripts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path


DEEP_MANIFEST_SCHEMA_VERSION = 1


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_deep_manifest(summary_payload, summary_output, manifest_output, raw_dir):
    summary_path = Path(summary_output)
    manifest_path = Path(manifest_output)
    return {
        "deep_manifest_schema_version": DEEP_MANIFEST_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "mode": summary_payload.get("mode"),
        "status": summary_payload.get("status"),
        "tool": summary_payload.get("tool", {}),
        "command": summary_payload.get("command"),
        "exit_code": summary_payload.get("exit_code"),
        "artifacts": {
            "trace_summary": str(summary_path),
            "trace_raw_dir": str(raw_dir),
            "deep_manifest": str(manifest_path),
        },
        "trace_summary_preview": summary_payload.get("preview", {}),
        "warnings": summary_payload.get("warnings", []),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
