import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER_PATH = REPO_ROOT / "scripts" / "summarize_rocm_smi.py"
ANALYZER_PATH = REPO_ROOT / "scripts" / "analyze_summary.py"
REPORT_PATH = REPO_ROOT / "scripts" / "generate_report.py"
DEEP_TRACE_PATH = REPO_ROOT / "scripts" / "summarize_rocprofv3.py"
DEEP_SYSTEM_PATH = REPO_ROOT / "scripts" / "summarize_rocprofsys.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005028.log"
CPU_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005029_cpu.log"
DEEP_TRACE_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "rocprofv3_trace" / "raw"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateReportTests(unittest.TestCase):
    def setUp(self):
        self.summarizer = load_module(SUMMARIZER_PATH, "summarize_rocm_smi")
        self.analyzer = load_module(ANALYZER_PATH, "analyze_summary")
        self.reporter = load_module(REPORT_PATH, "generate_report")
        self.deep_trace = load_module(DEEP_TRACE_PATH, "summarize_rocprofv3")
        self.deep_system = load_module(DEEP_SYSTEM_PATH, "summarize_rocprofsys")
        self.temp_dir = tempfile.mkdtemp(prefix="lumi-profiler-report-")
        shutil.copy(FIXTURE_PATH, pathlib.Path(self.temp_dir) / "nid005028.log")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_report_contains_core_sections(self):
        summary = self.summarizer.summarize_logs(self.temp_dir)
        analysis = self.analyzer.analyze_summary(summary)
        report = self.reporter.generate_report(summary, analysis)

        self.assertEqual(report["report_schema_version"], 1)
        self.assertIn("# LUMI Job Profiling Report", report["markdown"])
        self.assertIn("## Efficiency", report["markdown"])
        self.assertIn("## GPU Overview", report["markdown"])
        self.assertIn("well_utilized", report["markdown"])
        self.assertIn("`###################.`", report["markdown"])

        self.assertIn("<h1>LUMI Job Profiling Report</h1>", report["html"])
        self.assertIn("<h2>Recommendations</h2>", report["html"])
        self.assertIn("No immediate GPU efficiency issue was detected", report["html"])
        self.assertIn("<code>###################.</code>", report["html"])

    def test_report_includes_optional_cpu_metrics(self):
        cpu_only_dir = tempfile.mkdtemp(prefix="lumi-profiler-report-cpu-")
        try:
            shutil.copy(CPU_FIXTURE_PATH, pathlib.Path(cpu_only_dir) / "nid005029.log")
            summary = self.summarizer.summarize_logs(cpu_only_dir)
        finally:
            shutil.rmtree(cpu_only_dir)

        analysis = self.analyzer.analyze_summary(summary)
        report = self.reporter.generate_report(summary, analysis)

        self.assertIn("Average CPU utilization", report["markdown"])
        self.assertIn("Peak memory used", report["markdown"])
        self.assertIn("Average CPU utilization", report["html"])
        self.assertIn("cpu_bottleneck", report["markdown"])

    def test_report_includes_deep_trace_section(self):
        summary = self.summarizer.summarize_logs(self.temp_dir)
        analysis = self.analyzer.analyze_summary(summary)
        deep_manifest = self.deep_trace.build_deep_manifest(
            self.deep_trace.build_trace_summary(
                raw_dir=DEEP_TRACE_FIXTURE_DIR,
                tool_path="/usr/bin/rocprofv3",
                mode="deep-trace",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            ),
            pathlib.Path(self.temp_dir) / "deep_profile" / "trace" / "summary.json",
            pathlib.Path(self.temp_dir) / "deep_profile" / "deep_manifest.json",
        )

        report = self.reporter.generate_report(summary, analysis, deep_manifest)

        self.assertIn("## Deep Trace", report["markdown"])
        self.assertIn("hipLaunchKernel", report["markdown"])
        self.assertIn("void gemm_kernel", report["markdown"])
        self.assertIn("<h2>Deep Trace</h2>", report["html"])
        self.assertIn("hipLaunchKernel", report["html"])

    def test_report_surfaces_empty_runtime_trace_warning(self):
        summary = self.summarizer.summarize_logs(self.temp_dir)
        analysis = self.analyzer.analyze_summary(summary)
        deep_trace_dir = pathlib.Path(self.temp_dir) / "deep_profile" / "trace" / "raw"
        deep_trace_dir.mkdir(parents=True)
        (deep_trace_dir / "trace_agent_info.csv").write_text("Agent_Id,Name\n0,gfx90a\n", encoding="utf-8")
        (deep_trace_dir / "trace_results.json").write_text(
            json.dumps(
                {
                    "rocprofiler-sdk-tool": [
                        {
                            "buffer_records": {
                                "kernel_dispatch": [],
                                "hip_api": [],
                                "hsa_api": [],
                                "memory_copy": [],
                                "marker_api": [],
                                "rccl_api": [],
                                "scratch_memory": [],
                            },
                            "summary": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        deep_manifest = self.deep_trace.build_deep_manifest(
            self.deep_trace.build_trace_summary(
                raw_dir=deep_trace_dir,
                tool_path="/usr/bin/rocprofv3",
                mode="deep-trace",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            ),
            pathlib.Path(self.temp_dir) / "deep_profile" / "trace" / "summary.json",
            pathlib.Path(self.temp_dir) / "deep_profile" / "deep_manifest.json",
        )

        report = self.reporter.generate_report(summary, analysis, deep_manifest)

        self.assertIn("completed_without_runtime_events", report["markdown"])
        self.assertIn("captured no runtime events", report["markdown"])
        self.assertIn("captured no runtime events", report["html"])

    def test_report_includes_deep_system_section(self):
        summary = self.summarizer.summarize_logs(self.temp_dir)
        analysis = self.analyzer.analyze_summary(summary)
        system_dir = pathlib.Path(self.temp_dir) / "deep_profile" / "system" / "raw" / "rocprofsys-python-output" / "2026-03-16_00.04"
        system_dir.mkdir(parents=True)
        (system_dir / "perfetto-trace-32409.proto").write_text("perfetto", encoding="utf-8")
        (system_dir / "metadata-32409.json").write_text("{}", encoding="utf-8")
        (system_dir / "functions-32409.json").write_text("{}", encoding="utf-8")
        deep_manifest = self.deep_system.build_deep_manifest(
            self.deep_system.build_system_summary(
                raw_dir=pathlib.Path(self.temp_dir) / "deep_profile" / "system" / "raw",
                tool_path="/tmp/rocprof-sys-python",
                mode="deep-system",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            ),
            pathlib.Path(self.temp_dir) / "deep_profile" / "system" / "summary.json",
            pathlib.Path(self.temp_dir) / "deep_profile" / "deep_manifest.json",
        )

        report = self.reporter.generate_report(summary, analysis, deep_manifest)

        self.assertIn("## Deep System", report["markdown"])
        self.assertIn("Perfetto trace files: 1", report["markdown"])
        self.assertIn("https://ui.perfetto.dev", report["markdown"])
        self.assertIn("<h2>Deep System</h2>", report["html"])


if __name__ == "__main__":
    unittest.main()
