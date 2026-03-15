import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "profile_hook.sh"


class ProfileHookTests(unittest.TestCase):
    def run_bash(self, script, env=None):
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        result = subprocess.run(
            ["bash", "-lc", script],
            capture_output=True,
            text=True,
            env=merged_env,
        )
        return result

    def test_split_srun_command_keeps_payload_after_option_values(self):
        command = textwrap.dedent(
            f"""
            source "{SCRIPT_PATH}"
            profile_split_srun_command srun --cpu-bind none -n 1 python3 demo.py --seconds 60
            printf 'srun=%s\\n' "${{PROFILE_SRUN_ARGS[*]}}"
            printf 'payload=%s\\n' "${{PROFILE_SRUN_PAYLOAD_ARGS[*]}}"
            """
        )

        result = self.run_bash(command)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("srun=--cpu-bind none -n 1", result.stdout)
        self.assertIn("payload=python3 demo.py --seconds 60", result.stdout)

    def test_profile_run_command_injects_rocprofv3_inside_srun(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-hook-") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()

            srun_log = tmp_path / "srun.args"
            rocprof_log = tmp_path / "rocprof.args"
            app_log = tmp_path / "app.args"

            fake_srun = tmp_path / "srun"
            fake_srun.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{srun_log}"
                    while [ "$#" -gt 0 ]; do
                      case "$1" in
                        --)
                          shift
                          break
                          ;;
                        -*)
                          shift
                          ;;
                        *)
                          break
                          ;;
                      esac
                    done
                    exec "$@"
                    """
                )
            )
            fake_srun.chmod(0o755)

            fake_rocprof = tmp_path / "rocprofv3"
            fake_rocprof.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{rocprof_log}"
                    while [ "$#" -gt 0 ]; do
                      if [ "$1" = "--" ]; then
                        shift
                        break
                      fi
                      shift
                    done
                    exec "$@"
                    """
                )
            )
            fake_rocprof.chmod(0o755)

            fake_app = tmp_path / "app.sh"
            fake_app.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{app_log}"
                    exit 0
                    """
                )
            )
            fake_app.chmod(0o755)

            command = textwrap.dedent(
                f"""
                source "{SCRIPT_PATH}"
                profile_finalize_deep_trace() {{ :; }}
                PROFILE_MODE=deep-trace
                DEEP_TRACE_RAW_DIR="{raw_dir}"
                ROCPROFV3_PATH="{fake_rocprof}"
                PATH="{tmp_path}:$PATH"
                profile_run_command srun --cpu-bind=none --ntasks=1 "{fake_app}" alpha beta
                """
            )

            result = self.run_bash(command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            srun_args = srun_log.read_text().splitlines()
            rocprof_args = rocprof_log.read_text().splitlines()
            app_args = app_log.read_text().splitlines()

            self.assertEqual(srun_args[0:2], ["--cpu-bind=none", "--ntasks=1"])
            self.assertEqual(pathlib.Path(srun_args[2]).resolve(), fake_rocprof.resolve())
            self.assertIn("--runtime-trace", srun_args)
            self.assertIn("--output-directory", srun_args)
            self.assertIn(str(raw_dir), srun_args)
            self.assertEqual(srun_args[-4:], ["--", str(fake_app), "alpha", "beta"])

            self.assertIn("--runtime-trace", rocprof_args)
            self.assertIn("--stats", rocprof_args)
            self.assertEqual(rocprof_args[-4:], ["--", str(fake_app), "alpha", "beta"])
            self.assertEqual(app_args, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
