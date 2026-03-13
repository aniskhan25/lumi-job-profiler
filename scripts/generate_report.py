#!/usr/bin/env python3
"""Generate Markdown and HTML reports from summary and analysis artifacts."""

import argparse
from datetime import datetime, timezone
import html
import json
import os


REPORT_SCHEMA_VERSION = 1
ISSUE_RESOURCES = {
    "overscaling": "docs/gpu-scaling.html",
    "parallelism_mismatch": "docs/mpi-gpu-mapping.html",
    "sync_or_io_stalls": "docs/data-loaders.html",
    "cpu_bottleneck": "docs/data-loaders.html",
    "well_utilized": "docs/rocm-hip-tuning.html",
}


def iso_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt_pct(value):
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def fmt_num(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"


def bar(value, scale=100.0, width=20):
    if value is None:
        return "." * width
    filled = max(0, min(width, round((value / scale) * width)))
    return "#" * filled + "." * (width - filled)


def node_gpu_rows(summary):
    rows = []
    for node_name, node_stats in sorted(summary.get("nodes", {}).items()):
        for gpu_id, gpu_stats in sorted(node_stats.get("gpus", {}).items()):
            util = gpu_stats.get("gpu_util_pct", {})
            vram = gpu_stats.get("vram_util_pct", {})
            power = gpu_stats.get("power_w", {})
            rows.append(
                {
                    "node": node_name,
                    "gpu_id": gpu_id,
                    "util_avg": util.get("avg"),
                    "util_p95": util.get("p95"),
                    "vram_max": vram.get("max"),
                    "power_avg": power.get("avg"),
                }
            )
    return rows


def recommendation_lines(analysis):
    recommendations = analysis.get("recommendations", [])
    if not recommendations:
        return ["No recommendations were generated."]
    lines = []
    for item in recommendations:
        text = item.get("reason", "No recommendation text available.")
        if item.get("recommended_gpus") is not None:
            text = f"{text} Recommended GPUs: {item['recommended_gpus']}."
        lines.append(text)
    return lines


def root_cause_lines(analysis):
    lines = []
    for finding in analysis.get("root_causes", []):
        cause = finding.get("cause", "unknown")
        evidence = finding.get("evidence", "No evidence provided.")
        confidence = finding.get("confidence")
        resource = ISSUE_RESOURCES.get(cause)
        confidence_text = f"confidence {confidence:.2f}" if confidence is not None else "confidence n/a"
        line = f"{cause}: {evidence} ({confidence_text})"
        if resource:
            line = f"{line}. Resource: {resource}"
        lines.append(line)
    if not lines:
        lines.append("No root causes were inferred.")
    return lines


def build_markdown(summary, analysis):
    job = summary.get("job", {})
    job_metrics = summary.get("job_metrics", {})
    efficiency = analysis.get("efficiency", {})
    rows = node_gpu_rows(summary)

    lines = [
        "# LUMI Job Profiling Report",
        "",
        "## Job Summary",
        "",
        f"- Job ID: {job.get('job_id') or 'n/a'}",
        f"- User: {job.get('user') or 'n/a'}",
        f"- Project: {job.get('project_id') or 'n/a'}",
        f"- Partition: {job.get('partition') or 'n/a'}",
        f"- Tasks: {job.get('ntasks') or 'n/a'}",
        f"- CPUs per task: {job.get('cpus_per_task') or 'n/a'}",
        f"- GPUs requested: {job.get('gpus_requested') or 'n/a'}",
        f"- GPUs per node: {job.get('gpus_per_node') or 'n/a'}",
        f"- Generated: {analysis.get('generated_at') or 'n/a'}",
        "",
        "## Efficiency",
        "",
        f"- Class: {efficiency.get('class', 'UNKNOWN')}",
        f"- Score: {fmt_pct(efficiency.get('score_pct'))}",
        f"- Reason: {efficiency.get('reason', 'n/a')}",
        "",
        "## Key Metrics",
        "",
        f"- Average GPU utilization: {fmt_pct(job_metrics.get('avg_gpu_util_pct'))}",
        f"- Peak VRAM utilization: {fmt_pct(job_metrics.get('peak_vram_util_pct'))}",
        f"- Total observed GPU slots: {fmt_num(job_metrics.get('total_gpu_slots_observed'))}",
        f"- Active GPU estimate: {fmt_num(job_metrics.get('total_active_gpus_estimate'))}",
        f"- Average CPU utilization: {fmt_pct(job_metrics.get('avg_cpu_util_pct'))}",
        f"- Average CPU iowait: {fmt_pct(job_metrics.get('avg_cpu_iowait_pct'))}",
        f"- Peak memory used: {fmt_pct(job_metrics.get('peak_memory_used_pct'))}",
        f"- Average load1: {fmt_num(job_metrics.get('avg_load1'))}",
        "",
        "## GPU Overview",
        "",
        "| Node | GPU | Avg Util | P95 Util | Peak VRAM | Avg Power | Util Bar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    if rows:
        for row in rows:
            lines.append(
                f"| {row['node']} | {row['gpu_id']} | {fmt_pct(row['util_avg'])} | "
                f"{fmt_pct(row['util_p95'])} | {fmt_pct(row['vram_max'])} | "
                f"{fmt_num(row['power_avg'])} W | `{bar(row['util_avg'])}` |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | `....................` |")

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )
    for line in root_cause_lines(analysis):
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for line in recommendation_lines(analysis):
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary schema version: {summary.get('collection', {}).get('summary_schema_version', 'n/a')}",
            f"- Analysis schema version: {analysis.get('analysis_schema_version', 'n/a')}",
            f"- Report schema version: {REPORT_SCHEMA_VERSION}",
        ]
    )

    return "\n".join(lines) + "\n"


def render_html(summary, analysis):
    job = summary.get("job", {})
    job_metrics = summary.get("job_metrics", {})
    efficiency = analysis.get("efficiency", {})
    rows = node_gpu_rows(summary)

    findings_html = "".join(
        f"<li>{html.escape(line)}</li>" for line in root_cause_lines(analysis)
    )
    recommendations_html = "".join(
        f"<li>{html.escape(line)}</li>" for line in recommendation_lines(analysis)
    )

    if rows:
        table_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(row['node'])}</td>"
                f"<td>{html.escape(str(row['gpu_id']))}</td>"
                f"<td>{html.escape(fmt_pct(row['util_avg']))}</td>"
                f"<td>{html.escape(fmt_pct(row['util_p95']))}</td>"
                f"<td>{html.escape(fmt_pct(row['vram_max']))}</td>"
                f"<td>{html.escape(fmt_num(row['power_avg']))} W</td>"
                f"<td><code>{html.escape(bar(row['util_avg']))}</code></td>"
                "</tr>"
            )
            for row in rows
        )
    else:
        table_rows = (
            "<tr><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td>"
            "<td>n/a</td><td>n/a</td><td><code>....................</code></td></tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LUMI Job Profiling Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 960px; line-height: 1.5; color: #1a1a1a; }}
    h1, h2 {{ color: #123a63; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d7dde5; padding: 0.5rem; text-align: left; }}
    th {{ background: #eef4fb; }}
    .panel {{ background: #f7f9fc; border: 1px solid #d7dde5; padding: 1rem; margin-bottom: 1.5rem; }}
    code {{ background: #eef4fb; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>LUMI Job Profiling Report</h1>
    <div class="panel">
    <strong>Job ID:</strong> {html.escape(str(job.get('job_id') or 'n/a'))}<br>
    <strong>User:</strong> {html.escape(str(job.get('user') or 'n/a'))}<br>
    <strong>Project:</strong> {html.escape(str(job.get('project_id') or 'n/a'))}<br>
    <strong>Partition:</strong> {html.escape(str(job.get('partition') or 'n/a'))}<br>
    <strong>Tasks:</strong> {html.escape(str(job.get('ntasks') or 'n/a'))}<br>
    <strong>CPUs per task:</strong> {html.escape(str(job.get('cpus_per_task') or 'n/a'))}<br>
    <strong>GPUs requested:</strong> {html.escape(str(job.get('gpus_requested') or 'n/a'))}<br>
    <strong>GPUs per node:</strong> {html.escape(str(job.get('gpus_per_node') or 'n/a'))}<br>
    <strong>Generated:</strong> {html.escape(str(analysis.get('generated_at') or 'n/a'))}
  </div>
  <h2>Efficiency</h2>
  <ul>
    <li><strong>Class:</strong> {html.escape(str(efficiency.get('class', 'UNKNOWN')))}</li>
    <li><strong>Score:</strong> {html.escape(fmt_pct(efficiency.get('score_pct')))}</li>
    <li><strong>Reason:</strong> {html.escape(str(efficiency.get('reason', 'n/a')))}</li>
  </ul>
  <h2>Key Metrics</h2>
  <ul>
    <li><strong>Average GPU utilization:</strong> {html.escape(fmt_pct(job_metrics.get('avg_gpu_util_pct')))}</li>
    <li><strong>Peak VRAM utilization:</strong> {html.escape(fmt_pct(job_metrics.get('peak_vram_util_pct')))}</li>
    <li><strong>Total observed GPU slots:</strong> {html.escape(fmt_num(job_metrics.get('total_gpu_slots_observed')))}</li>
    <li><strong>Active GPU estimate:</strong> {html.escape(fmt_num(job_metrics.get('total_active_gpus_estimate')))}</li>
    <li><strong>Average CPU utilization:</strong> {html.escape(fmt_pct(job_metrics.get('avg_cpu_util_pct')))}</li>
    <li><strong>Average CPU iowait:</strong> {html.escape(fmt_pct(job_metrics.get('avg_cpu_iowait_pct')))}</li>
    <li><strong>Peak memory used:</strong> {html.escape(fmt_pct(job_metrics.get('peak_memory_used_pct')))}</li>
    <li><strong>Average load1:</strong> {html.escape(fmt_num(job_metrics.get('avg_load1')))}</li>
  </ul>
  <h2>GPU Overview</h2>
  <table>
    <thead>
      <tr><th>Node</th><th>GPU</th><th>Avg Util</th><th>P95 Util</th><th>Peak VRAM</th><th>Avg Power</th><th>Util Bar</th></tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  <h2>Findings</h2>
  <ul>{findings_html}</ul>
  <h2>Recommendations</h2>
  <ul>{recommendations_html}</ul>
</body>
</html>
"""


def generate_report(summary, analysis):
    markdown = build_markdown(summary, analysis)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "markdown": markdown,
        "html": render_html(summary, analysis),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", help="Path to summary.json")
    parser.add_argument("analysis_json", help="Path to analysis.json")
    parser.add_argument("markdown_output", nargs="?", default=None, help="Path to report.md")
    parser.add_argument("html_output", nargs="?", default=None, help="Path to report.html")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(args.analysis_json, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    report = generate_report(summary, analysis)

    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as f:
            f.write(report["markdown"])
    else:
        print(report["markdown"])

    if args.html_output:
        with open(args.html_output, "w", encoding="utf-8") as f:
            f.write(report["html"])


if __name__ == "__main__":
    main()
