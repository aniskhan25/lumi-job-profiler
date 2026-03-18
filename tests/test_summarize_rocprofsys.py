import importlib.util
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "src" / "summarize_rocprofsys.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SummarizeRocprofSysTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(SCRIPT_PATH, "summarize_rocprofsys")

    def test_system_summary_discovers_perfetto_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-rocpd-") as tmpdir:
            raw_dir = pathlib.Path(tmpdir) / "rocprofsys-python-output" / "2026-03-16_00.04"
            raw_dir.mkdir(parents=True)
            (raw_dir / "perfetto-trace-32409.proto").write_text("perfetto", encoding="utf-8")
            (raw_dir / "metadata-32409.json").write_text("{}", encoding="utf-8")
            (raw_dir / "functions-32409.json").write_text("{}", encoding="utf-8")

            summary = self.module.build_system_summary(
                raw_dir=pathlib.Path(tmpdir),
                tool_path="/tmp/rocprof-sys-python",
                mode="deep-system",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["preview"]["perfetto_trace_files"], 1)
        self.assertEqual(summary["preview"]["metadata_files"], 1)
        self.assertEqual(summary["preview"]["functions_files"], 1)
        self.assertTrue(summary["preview"]["perfetto_trace_sample"][0].endswith(".proto"))

    def test_system_summary_warns_when_no_perfetto_trace_exists(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-rocpd-") as tmpdir:
            raw_dir = pathlib.Path(tmpdir)
            (raw_dir / "stdout.txt").write_text("ok\n", encoding="utf-8")

            summary = self.module.build_system_summary(
                raw_dir=raw_dir,
                tool_path="/tmp/rocprof-sys-python",
                mode="deep-system",
                command="srun python3 demo.py",
                status="completed",
                exit_code=0,
            )

        self.assertEqual(summary["status"], "completed_without_perfetto_trace")
        self.assertIn("did not produce a perfetto trace", summary["warnings"][0])

    def test_system_summary_aggregates_nested_rank_outputs(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-rocpd-dist-") as tmpdir:
            raw_dir = pathlib.Path(tmpdir)
            rank0 = raw_dir / "nid000001" / "rank-0" / "rocprofsys-python-output" / "2026-03-16_00.04"
            rank1 = raw_dir / "nid000002" / "rank-1" / "rocprofsys-python-output" / "2026-03-16_00.04"
            rank0.mkdir(parents=True)
            rank1.mkdir(parents=True)
            (rank0 / "perfetto-trace-1.proto").write_text("perfetto", encoding="utf-8")
            (rank1 / "perfetto-trace-2.proto").write_text("perfetto", encoding="utf-8")

            summary = self.module.build_system_summary(
                raw_dir=raw_dir,
                tool_path="/tmp/rocprof-sys-python",
                mode="deep-system",
                command="srun --ntasks=2 -- python3 demo.py",
                status="completed",
                exit_code=0,
            )

        self.assertEqual(summary["preview"]["perfetto_trace_files"], 2)
        self.assertEqual(summary["preview"]["distributed_node_count"], 2)
        self.assertEqual(summary["preview"]["distributed_rank_count"], 2)


if __name__ == "__main__":
    unittest.main()
