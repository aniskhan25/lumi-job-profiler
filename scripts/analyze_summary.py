#!/usr/bin/env python3
"""Analyze summary.json and emit a rule-based analysis artifact."""

import argparse
from datetime import datetime, timezone
import json
from statistics import mean


ANALYSIS_SCHEMA_VERSION = 1
EFFICIENCY_THRESHOLDS = [
    ("EFFICIENT", 70.0),
    ("ACCEPTABLE", 40.0),
    ("INEFFICIENT", 15.0),
    ("WASTED", 0.0),
]


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def classify_efficiency(avg_gpu_util_pct):
    if avg_gpu_util_pct is None:
        return {
            "class": "UNKNOWN",
            "score_pct": None,
            "reason": "No GPU utilization metrics were available.",
        }

    for label, threshold in EFFICIENCY_THRESHOLDS:
        if avg_gpu_util_pct >= threshold:
            return {
                "class": label,
                "score_pct": avg_gpu_util_pct,
                "reason": f"Average GPU utilization was {avg_gpu_util_pct:.1f}%.",
            }

    return {
        "class": "UNKNOWN",
        "score_pct": avg_gpu_util_pct,
        "reason": "Utilization could not be classified.",
    }


def collect_gpu_util_details(summary):
    details = []
    for node_name, node_stats in summary.get("nodes", {}).items():
        for gpu_id, gpu_stats in node_stats.get("gpus", {}).items():
            util_stats = gpu_stats.get("gpu_util_pct")
            vram_stats = gpu_stats.get("vram_util_pct")
            mem_rw_stats = gpu_stats.get("mem_rw_activity_pct")
            if not util_stats:
                continue
            details.append(
                {
                    "node": node_name,
                    "gpu_id": gpu_id,
                    "util_avg": util_stats.get("avg"),
                    "util_p95": util_stats.get("p95"),
                    "vram_max": vram_stats.get("max") if vram_stats else None,
                    "mem_rw_avg": mem_rw_stats.get("avg") if mem_rw_stats else None,
                }
            )
    return details


def infer_root_causes(summary):
    job_metrics = summary.get("job_metrics", {})
    gpu_details = collect_gpu_util_details(summary)
    findings = []

    avg_gpu_util = job_metrics.get("avg_gpu_util_pct")
    peak_vram = job_metrics.get("peak_vram_util_pct")
    total_gpu_slots = job_metrics.get("total_gpu_slots_observed") or 0
    active_gpus = job_metrics.get("total_active_gpus_estimate") or 0

    if avg_gpu_util is None:
        return findings

    if avg_gpu_util < 30.0 and peak_vram is not None and peak_vram < 20.0 and total_gpu_slots > 1:
        recommended_gpus = max(active_gpus, 1)
        findings.append(
            {
                "cause": "overscaling",
                "confidence": 0.85,
                "evidence": (
                    f"Average GPU utilization is {avg_gpu_util:.1f}% and peak VRAM use is "
                    f"{peak_vram:.1f}% across {total_gpu_slots} observed GPUs."
                ),
                "recommendation": {
                    "type": "right_size_gpus",
                    "recommended_gpus": recommended_gpus,
                    "reason": "Low utilization and low VRAM usage suggest the job is over-provisioned.",
                },
            }
        )

    if avg_gpu_util < 40.0 and total_gpu_slots > 0 and active_gpus < total_gpu_slots:
        findings.append(
            {
                "cause": "parallelism_mismatch",
                "confidence": 0.8,
                "evidence": (
                    f"Only {active_gpus} of {total_gpu_slots} observed GPUs exceeded the active threshold "
                    f"while average utilization was {avg_gpu_util:.1f}%."
                ),
                "recommendation": {
                    "type": "align_ranks_and_gpus",
                    "recommended_gpus": max(active_gpus, 1),
                    "reason": "Match the requested GPU count to the number of GPUs doing sustained work.",
                },
            }
        )

    util_gaps = [
        d["util_p95"] - d["util_avg"]
        for d in gpu_details
        if d.get("util_p95") is not None and d.get("util_avg") is not None
    ]
    if util_gaps:
        avg_gap = mean(util_gaps)
        if avg_gpu_util < 50.0 and avg_gap >= 25.0:
            findings.append(
                {
                    "cause": "sync_or_io_stalls",
                    "confidence": 0.65,
                    "evidence": (
                        f"Average GPU utilization is {avg_gpu_util:.1f}% but the average p95-to-mean gap "
                        f"is {avg_gap:.1f} percentage points, suggesting bursty execution."
                    ),
                    "recommendation": {
                        "type": "investigate_stalls",
                        "reason": "Inspect data loading, synchronization points, or other host-side stalls.",
                    },
                }
            )

    if not findings and avg_gpu_util >= 70.0:
        findings.append(
            {
                "cause": "well_utilized",
                "confidence": 0.9,
                "evidence": f"Average GPU utilization is {avg_gpu_util:.1f}%, which is already high.",
                "recommendation": {
                    "type": "none",
                    "reason": "No immediate GPU efficiency issue was detected from the lightweight profile.",
                },
            }
        )

    return findings


def build_recommendations(root_causes):
    recommendations = []
    seen = set()
    for finding in root_causes:
        recommendation = finding.get("recommendation")
        if not recommendation:
            continue
        key = json.dumps(recommendation, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        recommendations.append(recommendation)
    return recommendations


def analyze_summary(summary):
    efficiency = classify_efficiency(summary.get("job_metrics", {}).get("avg_gpu_util_pct"))
    root_causes = infer_root_causes(summary)
    recommendations = build_recommendations(root_causes)

    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "input_summary_schema_version": summary.get("collection", {}).get("summary_schema_version"),
        "job": summary.get("job", {}),
        "efficiency": efficiency,
        "root_causes": root_causes,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", help="Path to summary.json")
    parser.add_argument("output", nargs="?", default=None, help="Output JSON file")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    analysis = analyze_summary(summary)
    payload = json.dumps(analysis, indent=2, sort_keys=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
