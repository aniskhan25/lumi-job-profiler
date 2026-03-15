import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "summarize_rocprofv3.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "rocprofv3_trace" / "raw"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SummarizeRocprofv3Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(SCRIPT_PATH, "summarize_rocprofv3")

    def test_trace_summary_extracts_top_entries(self):
        summary = self.module.build_trace_summary(
            raw_dir=FIXTURE_DIR,
            tool_path="/usr/bin/rocprofv3",
            mode="deep-trace",
            command="srun python3 demo.py",
            status="completed",
            exit_code=0,
        )

        self.assertEqual(summary["deep_trace_schema_version"], 1)
        self.assertEqual(summary["preview"]["hip_api_trace_rows"], 3)
        self.assertEqual(summary["preview"]["kernel_dispatch_trace_rows"], 2)
        self.assertEqual(summary["preview"]["memory_copy_trace_rows"], 1)
        self.assertEqual(summary["preview"]["top_hip_apis"][0]["name"], "hipLaunchKernel")
        self.assertEqual(summary["preview"]["top_kernel_dispatches"][0]["name"], "void gemm_kernel")
        self.assertEqual(summary["tool"]["path"], "/usr/bin/rocprofv3")
        self.assertEqual(summary["command"], "srun python3 demo.py")

    def test_manifest_records_artifact_locations(self):
        summary = self.module.build_trace_summary(
            raw_dir=FIXTURE_DIR,
            tool_path="/usr/bin/rocprofv3",
            mode="deep-trace",
            command="srun python3 demo.py",
            status="completed",
            exit_code=0,
        )

        with tempfile.TemporaryDirectory(prefix="lumi-profiler-deep-") as tmpdir:
            summary_path = pathlib.Path(tmpdir) / "summary.json"
            manifest_path = pathlib.Path(tmpdir) / "deep_manifest.json"
            manifest = self.module.build_deep_manifest(summary, summary_path, manifest_path)

        self.assertEqual(manifest["deep_manifest_schema_version"], 1)
        self.assertEqual(manifest["artifacts"]["trace_summary"], str(summary_path))
        self.assertEqual(manifest["artifacts"]["deep_manifest"], str(manifest_path))
        self.assertEqual(manifest["trace_summary_preview"]["top_hip_apis"][0]["name"], "hipLaunchKernel")

    def test_json_only_trace_with_empty_runtime_buffers_is_reported_explicitly(self):
        payload = {
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

        with tempfile.TemporaryDirectory(prefix="lumi-profiler-rocprof-json-") as tmpdir:
            raw_dir = pathlib.Path(tmpdir)
            (raw_dir / "trace_agent_info.csv").write_text("Agent_Id,Name\n0,gfx90a\n", encoding="utf-8")
            (raw_dir / "trace_results.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = self.module.build_trace_summary(
                raw_dir=raw_dir,
                tool_path="/usr/bin/rocprofv3",
                mode="deep-trace",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            )

        self.assertEqual(summary["status"], "completed_without_runtime_events")
        self.assertEqual(summary["preview"]["kernel_dispatch_trace_rows"], 0)
        self.assertEqual(summary["preview"]["hip_api_trace_rows"], 0)
        self.assertEqual(summary["preview"]["memory_copy_trace_rows"], 0)
        self.assertIn("runtime_record_counts", summary["preview"])
        self.assertIn("captured no runtime events", summary["warnings"][0])

    def test_kernel_trace_alias_maps_to_kernel_dispatch_preview(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-rocprof-kernel-") as tmpdir:
            raw_dir = pathlib.Path(tmpdir)
            (raw_dir / "trace_kernel_trace.csv").write_text(
                "StartNs,EndNs,KernelName\n0,5000,void gemm_kernel\n",
                encoding="utf-8",
            )
            (raw_dir / "trace_hip_api_trace.csv").write_text(
                "StartNs,EndNs,Function\n0,1000,hipLaunchKernel\n",
                encoding="utf-8",
            )

            summary = self.module.build_trace_summary(
                raw_dir=raw_dir,
                tool_path="/usr/bin/rocprofv3",
                mode="deep-trace",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            )

        self.assertEqual(summary["preview"]["kernel_dispatch_trace_rows"], 1)
        self.assertEqual(summary["trace_stats"]["kernel_dispatch"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
