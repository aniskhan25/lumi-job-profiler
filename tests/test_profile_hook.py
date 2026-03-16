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

    def test_profile_run_command_wraps_container_directly(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-hook-") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            singularity_log = tmp_path / "singularity.args"
            app_log = tmp_path / "app.args"
            container_image = tmp_path / "container.sif"
            workdir = tmp_path / "workdir"
            container_image.write_text("", encoding="utf-8")
            workdir.mkdir()

            fake_singularity = tmp_path / "singularity"
            fake_singularity.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{singularity_log}"
                    while [ "$#" -gt 0 ]; do
                      case "$1" in
                        exec)
                          shift
                          ;;
                        --bind|--pwd)
                          shift
                          shift
                          ;;
                        --rocm)
                          shift
                          ;;
                        *.sif)
                          shift
                          break
                          ;;
                        *)
                          shift
                          ;;
                      esac
                    done
                    exec "$@"
                    """
                )
            )
            fake_singularity.chmod(0o755)

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
                PATH="{tmp_path}:$PATH"
                LUMI_CONTAINER_IMAGE="{container_image}"
                LUMI_CONTAINER_WORKDIR="{workdir}"
                profile_run_command "{fake_app}" alpha beta
                """
            )

            result = self.run_bash(command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            singularity_args = singularity_log.read_text().splitlines()
            app_args = app_log.read_text().splitlines()

            self.assertEqual(singularity_args[0], "exec")
            self.assertIn("--bind", singularity_args)
            self.assertIn("--pwd", singularity_args)
            self.assertIn(str(container_image), singularity_args)
            self.assertEqual(singularity_args[-3:], [str(fake_app), "alpha", "beta"])
            self.assertEqual(app_args, ["alpha", "beta"])

    def test_profile_run_command_injects_container_rocprofv3_directly(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-hook-") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            workdir = tmp_path / "workdir"
            workdir.mkdir()
            container_image = tmp_path / "container.sif"
            container_image.write_text("", encoding="utf-8")

            singularity_log = tmp_path / "singularity.args"
            rocprof_log = tmp_path / "rocprof.args"
            app_log = tmp_path / "app.args"

            fake_singularity = tmp_path / "singularity"
            fake_singularity.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{singularity_log}"
                    while [ "$#" -gt 0 ]; do
                      case "$1" in
                        exec)
                          shift
                          ;;
                        --bind|--pwd)
                          shift
                          shift
                          ;;
                        --rocm)
                          shift
                          ;;
                        *.sif)
                          shift
                          break
                          ;;
                        *)
                          shift
                          ;;
                      esac
                    done
                    exec "$@"
                    """
                )
            )
            fake_singularity.chmod(0o755)

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
                profile_finalize_deep_profile() {{ :; }}
                PROFILE_MODE=deep-trace
                DEEP_TRACE_RAW_DIR="{raw_dir}"
                PATH="{tmp_path}:$PATH"
                LUMI_CONTAINER_IMAGE="{container_image}"
                LUMI_CONTAINER_WORKDIR="{workdir}"
                profile_run_command "{fake_app}" alpha beta
                """
            )

            result = self.run_bash(command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            singularity_args = singularity_log.read_text().splitlines()
            rocprof_args = rocprof_log.read_text().splitlines()
            app_args = app_log.read_text().splitlines()

            self.assertEqual(singularity_args[0], "exec")
            self.assertIn("--bind", singularity_args)
            self.assertIn(str(container_image), singularity_args)
            self.assertIn("rocprofv3", singularity_args)
            self.assertIn("--runtime-trace", rocprof_args)
            self.assertIn("--stats", rocprof_args)
            self.assertIn(str(raw_dir), rocprof_args)
            self.assertEqual(rocprof_args[-4:], ["--", str(fake_app), "alpha", "beta"])
            self.assertEqual(app_args, ["alpha", "beta"])

    def test_profile_run_command_marks_host_deep_trace_as_unsupported(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-hook-") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            app_log = tmp_path / "app.args"

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
                profile_finalize_deep_profile() {{ printf 'finalize=%s\\n' "$2"; }}
                PROFILE_MODE=deep-trace
                profile_run_command "{fake_app}" alpha beta
                """
            )

            result = self.run_bash(command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("fallback_unsupported_host_deep_profile", result.stdout)
            self.assertIn("supported only for container launches", result.stderr)

    def test_profile_run_command_injects_rocprofsys_python_inside_container(self):
        with tempfile.TemporaryDirectory(prefix="lumi-profiler-hook-") as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            raw_dir = tmp_path / "system_raw"
            raw_dir.mkdir()
            workdir = tmp_path / "workdir"
            workdir.mkdir()
            container_image = tmp_path / "container.sif"
            container_image.write_text("", encoding="utf-8")
            install_prefix = tmp_path / "rocprof-sys-install"
            (install_prefix / "bin").mkdir(parents=True)
            (install_prefix / "libexec" / "rocprofiler-systems").mkdir(parents=True)
            (install_prefix / "share" / "rocprofiler-systems").mkdir(parents=True)
            (install_prefix / "lib").mkdir(parents=True)
            (install_prefix / "lib64").mkdir(parents=True)
            (install_prefix / "lib" / "rocprofiler-systems").mkdir(parents=True)
            (install_prefix / "share" / "rocprofiler-systems" / "setup-env.sh").write_text("", encoding="utf-8")

            singularity_log = tmp_path / "singularity.args"
            rocprofsys_log = tmp_path / "rocprofsys.args"
            app_log = tmp_path / "app.args"

            fake_singularity = tmp_path / "singularity"
            fake_singularity.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{singularity_log}"
                    while [ "$#" -gt 0 ]; do
                      case "$1" in
                        exec)
                          shift
                          ;;
                        --bind|--pwd)
                          shift
                          shift
                          ;;
                        --rocm)
                          shift
                          ;;
                        *.sif)
                          shift
                          break
                          ;;
                        *)
                          shift
                          ;;
                      esac
                    done
                    exec "$@"
                    """
                )
            )
            fake_singularity.chmod(0o755)

            fake_python = tmp_path / "python3"
            fake_python.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    if [ "$1" = "-" ] || [ "$1" = "-c" ]; then
                      if [ "$1" = "-" ]; then
                        cat >/dev/null
                      fi
                      printf '%s\n' "/tmp/fake-torch-lib"
                      exit 0
                    fi
                    exec /bin/sh "$@"
                    """
                )
            )
            fake_python.chmod(0o755)

            fake_rocprofsys = tmp_path / "rocprof-sys-python"
            fake_rocprofsys.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "{rocprofsys_log}"
                    while [ "$#" -gt 0 ]; do
                      case "$1" in
                        --output)
                          shift
                          shift
                          ;;
                        -*)
                          shift
                          ;;
                        *)
                          break
                          ;;
                      esac
                    done
                    app="$1"
                    shift
                    exec "$app" "$@"
                    """
                )
            )
            fake_rocprofsys.chmod(0o755)

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
                PROFILE_DIR="{tmp_path}/profile"
                DEEP_PROFILE_DIR="{tmp_path}/profile/deep_profile"
                source "{SCRIPT_PATH}"
                profile_finalize_deep_profile() {{ :; }}
                PROFILE_MODE=deep-system
                DEEP_SYSTEM_RAW_DIR="{raw_dir}"
                PATH="{tmp_path}:$PATH"
                LUMI_CONTAINER_IMAGE="{container_image}"
                LUMI_CONTAINER_WORKDIR="{workdir}"
                ROCPROFSYS_INSTALL_PREFIX="{install_prefix}"
                profile_run_command "{fake_python}" "{fake_app}" alpha beta
                """
            )

            result = self.run_bash(command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            singularity_args = singularity_log.read_text().splitlines()
            rocprofsys_args = rocprofsys_log.read_text().splitlines()
            app_args = app_log.read_text().splitlines()

            self.assertEqual(singularity_args[0], "exec")
            self.assertIn("bash", singularity_args)
            self.assertIn("rocprof-sys-python", " ".join(singularity_args))
            self.assertNotIn("--output", rocprofsys_args)
            self.assertEqual(rocprofsys_args[0], "--")
            self.assertIn(str(fake_app), rocprofsys_args)
            self.assertEqual(app_args, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
