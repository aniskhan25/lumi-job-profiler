import importlib.util
import pathlib
import shutil
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = REPO_ROOT / "src" / "analyze_summary.py"
SUMMARIZER_PATH = REPO_ROOT / "src" / "summarize_rocm_smi.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005028.log"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = load_module(ANALYZER_PATH, "analyze_summary")
        self.summarizer = load_module(SUMMARIZER_PATH, "summarize_rocm_smi")
        self.temp_dir = tempfile.mkdtemp(prefix="lumi-profiler-analysis-")
        shutil.copy(FIXTURE_PATH, pathlib.Path(self.temp_dir) / "nid005028.log")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_real_fixture_is_classified_as_efficient(self):
        summary = self.summarizer.summarize_logs(self.temp_dir)
        analysis = self.analyzer.analyze_summary(summary)

        self.assertEqual(analysis["analysis_schema_version"], 1)
        self.assertEqual(analysis["input_summary_schema_version"], 1)
        self.assertEqual(analysis["efficiency"]["class"], "EFFICIENT")
        self.assertTrue(analysis["root_causes"])
        self.assertEqual(analysis["root_causes"][0]["cause"], "well_utilized")
        self.assertEqual(analysis["recommendations"][0]["type"], "none")

    def test_low_utilization_summary_triggers_overscaling_and_parallelism_rules(self):
        summary = {
            "collection": {"summary_schema_version": 1},
            "job": {"job_id": "12345", "ntasks": "1"},
            "job_metrics": {
                "avg_gpu_util_pct": 18.0,
                "peak_vram_util_pct": 6.0,
                "total_gpu_slots_observed": 4,
                "total_active_gpus_estimate": 1,
            },
            "nodes": {
                "nid001": {
                    "gpus": {
                        "0": {
                            "gpu_util_pct": {"avg": 52.0, "p95": 87.0, "max": 91.0},
                            "vram_util_pct": {"avg": 6.0, "p95": 6.0, "max": 6.0},
                        },
                        "1": {
                            "gpu_util_pct": {"avg": 8.0, "p95": 40.0, "max": 48.0},
                            "vram_util_pct": {"avg": 4.0, "p95": 4.0, "max": 4.0},
                        },
                        "2": {
                            "gpu_util_pct": {"avg": 7.0, "p95": 38.0, "max": 44.0},
                            "vram_util_pct": {"avg": 3.0, "p95": 3.0, "max": 3.0},
                        },
                        "3": {
                            "gpu_util_pct": {"avg": 5.0, "p95": 36.0, "max": 42.0},
                            "vram_util_pct": {"avg": 3.0, "p95": 3.0, "max": 3.0},
                        },
                    }
                }
            },
        }

        analysis = self.analyzer.analyze_summary(summary)

        self.assertEqual(analysis["efficiency"]["class"], "INEFFICIENT")
        causes = {item["cause"] for item in analysis["root_causes"]}
        self.assertIn("overscaling", causes)
        self.assertIn("parallelism_mismatch", causes)
        self.assertIn("sync_or_io_stalls", causes)

        recommendation_types = {item["type"] for item in analysis["recommendations"]}
        self.assertIn("right_size_gpus", recommendation_types)
        self.assertIn("align_ranks_and_gpus", recommendation_types)
        self.assertIn("investigate_stalls", recommendation_types)

    def test_cpu_heavy_summary_triggers_cpu_bottleneck(self):
        summary = {
            "collection": {"summary_schema_version": 1},
            "job": {"job_id": "12346", "ntasks": "4", "cpus_per_task": "8"},
            "job_metrics": {
                "avg_gpu_util_pct": 24.0,
                "peak_vram_util_pct": 12.0,
                "total_gpu_slots_observed": 4,
                "total_active_gpus_estimate": 4,
                "avg_cpu_util_pct": 82.0,
                "avg_cpu_iowait_pct": 6.0,
            },
            "nodes": {
                "nid002": {
                    "gpus": {
                        "0": {"gpu_util_pct": {"avg": 24.0, "p95": 44.0, "max": 52.0}},
                        "1": {"gpu_util_pct": {"avg": 26.0, "p95": 46.0, "max": 55.0}},
                        "2": {"gpu_util_pct": {"avg": 22.0, "p95": 41.0, "max": 48.0}},
                        "3": {"gpu_util_pct": {"avg": 24.0, "p95": 43.0, "max": 50.0}},
                    }
                }
            },
        }

        analysis = self.analyzer.analyze_summary(summary)

        causes = {item["cause"] for item in analysis["root_causes"]}
        self.assertIn("cpu_bottleneck", causes)
        recommendation_types = {item["type"] for item in analysis["recommendations"]}
        self.assertIn("inspect_cpu_pipeline", recommendation_types)


if __name__ == "__main__":
    unittest.main()
