import importlib.util
import math
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "summarize_rocm_smi.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005028.log"
CPU_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nid005029_cpu.log"


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

    def test_optional_cpu_metrics_are_parsed_and_aggregated(self):
        shutil.copy(CPU_FIXTURE_PATH, pathlib.Path(self.temp_dir) / "nid005029.log")

        summary = self.module.summarize_logs(self.temp_dir)

        self.assertTrue(summary["collection"]["collect_cpu_metrics"])
        node = summary["nodes"]["nid005029"]
        self.assertEqual(node["metadata"]["profile_collect_cpu"], "1")
        self.assertIn("cpu", node)
        self.assertAlmostEqual(node["cpu"]["cpu_util_pct"]["avg"], 75.0)
        self.assertAlmostEqual(node["cpu"]["cpu_iowait_pct"]["avg"], 5.0)
        self.assertAlmostEqual(node["cpu"]["memory_used_pct"]["max"], 65.0)
        self.assertAlmostEqual(node["cpu"]["load1"]["avg"], 4.5)

        self.assertAlmostEqual(summary["job_metrics"]["avg_cpu_util_pct"], 75.0)
        self.assertAlmostEqual(summary["job_metrics"]["avg_cpu_iowait_pct"], 5.0)
        self.assertAlmostEqual(summary["job_metrics"]["peak_memory_used_pct"], 65.0)
        self.assertAlmostEqual(summary["job_metrics"]["avg_load1"], 4.5)

    def test_job_metadata_prefers_physical_gpu_request_over_gpu_id_list(self):
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_GPUS_PER_NODE": "1",
                "SLURM_GPUS": "2",
                "SLURM_JOB_GPUS": "0,1",
            },
            clear=False,
        ):
            metadata = self.module.build_job_metadata(self.temp_dir)

        self.assertEqual(metadata["gpus_requested"], "1")
        self.assertEqual(metadata["gpus_per_node"], "1")
        self.assertEqual(metadata["job_gpu_ids"], "0,1")


if __name__ == "__main__":
    unittest.main()
