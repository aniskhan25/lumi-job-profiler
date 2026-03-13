import importlib.util
import pathlib
import shutil
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER_PATH = REPO_ROOT / "scripts" / "summarize_rocm_smi.py"
ANALYZER_PATH = REPO_ROOT / "scripts" / "analyze_summary.py"
REPORT_PATH = REPO_ROOT / "scripts" / "generate_report.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005028.log"
CPU_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005029_cpu.log"


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


if __name__ == "__main__":
    unittest.main()
