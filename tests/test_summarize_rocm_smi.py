import importlib.util
import math
import os
import pathlib
import shutil
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "summarize_rocm_smi.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005028.log"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_rocm_smi", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SummarizeRocmSmiTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.mkdtemp(prefix="lumi-profiler-test-")
        shutil.copy(FIXTURE_PATH, pathlib.Path(self.temp_dir) / "nid005028.log")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_summary_schema_and_job_metrics(self):
        summary = self.module.summarize_logs(self.temp_dir)

        self.assertEqual(summary["log_dir"], self.temp_dir)
        self.assertEqual(summary["collection"]["summary_schema_version"], 1)
        self.assertEqual(summary["collection"]["raw_log_schema_versions"], ["1"])
        self.assertEqual(
            summary["collection"]["collect_commands"],
            ["rocm-smi --showuse --showmemuse --showpower --showtemp --showclocks"],
        )
        self.assertTrue(summary["collection"]["generated_at"].endswith("+00:00"))
        self.assertEqual(summary["warnings"], [])

        node = summary["nodes"]["nid005028"]
        self.assertEqual(node["samples"], 2)
        self.assertEqual(node["start_ts"], 1770500741)
        self.assertEqual(node["end_ts"], 1770500743)
        self.assertEqual(node["duration_seconds"], 2)
        self.assertEqual(node["interval_seconds"], 2)
        self.assertEqual(node["metadata"]["profile_log_schema_version"], "1")

        gpu0 = node["gpus"]["0"]
        self.assertAlmostEqual(gpu0["gpu_util_pct"]["avg"], 95.0)
        self.assertAlmostEqual(gpu0["gpu_util_pct"]["p95"], 97.7)
        self.assertEqual(gpu0["gpu_util_pct"]["max"], 98.0)
        self.assertAlmostEqual(gpu0["power_w"]["avg"], 306.0)
        self.assertEqual(gpu0["vram_util_pct"]["max"], 2.0)
        self.assertEqual(gpu0["sclk_mhz"]["max"], 1700.0)

        job_metrics = summary["job_metrics"]
        self.assertEqual(job_metrics["active_gpu_threshold_pct"], 10.0)
        self.assertEqual(job_metrics["nodes_with_gpu_metrics"], 1)
        self.assertEqual(job_metrics["total_gpu_slots_observed"], 1)
        self.assertEqual(job_metrics["total_active_gpus_estimate"], 1)
        self.assertEqual(job_metrics["effective_gpus_estimate"], 1)
        self.assertAlmostEqual(job_metrics["avg_gpu_util_pct"], 95.0)
        self.assertEqual(job_metrics["peak_vram_util_pct"], 2.0)

    def test_collection_sampling_interval_is_averaged(self):
        summary = self.module.summarize_logs(self.temp_dir)
        self.assertTrue(math.isclose(summary["collection"]["sampling_interval_seconds"], 2.0))


if __name__ == "__main__":
    unittest.main()
