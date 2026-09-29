import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training_ops.runtime.h200 import watch_rollouts_tqdm as renderer


class WatchRolloutsRendererTest(unittest.TestCase):
    def test_parses_current_telemetry_format(self):
        line = "PIPELINE_TELEMETRY accepted_buffer=12 accepted_target=24 accepted_unit=rollouts"
        self.assertEqual(renderer._parse_progress(line), (12, 24))

    def test_parses_legacy_format(self):
        self.assertEqual(renderer._parse_progress("Train batch 7/24"), (7, 24))

    def test_ignores_unrelated_lines(self):
        self.assertIsNone(renderer._parse_progress("worker is healthy"))

    def test_buffer_reset_restarts_progress_bar(self):
        first_bar = mock.Mock(total=24, n=12)
        second_bar = mock.Mock(total=24, n=2)
        with mock.patch.object(renderer, "tqdm", side_effect=[first_bar, second_bar]) as make_bar:
            with mock.patch.object(
                sys,
                "stdin",
                io.StringIO(
                    "PIPELINE_TELEMETRY accepted_buffer=12 accepted_target=24\n"
                    "PIPELINE_TELEMETRY accepted_buffer=2 accepted_target=24\n"
                ),
            ), mock.patch.object(sys, "argv", ["watch_rollouts_tqdm.py", "3"]):
                renderer.main()

        self.assertEqual(make_bar.call_count, 2)
        first_bar.close.assert_called_once_with()

    def test_success_increments_step_and_resets_buffer(self):
        first_bar = mock.Mock(total=24, n=12)
        second_bar = mock.Mock(total=24, n=1)
        with mock.patch.object(renderer, "tqdm", side_effect=[first_bar, second_bar]) as make_bar:
            with mock.patch.object(
                sys,
                "stdin",
                io.StringIO(
                    "PIPELINE_TELEMETRY accepted_buffer=12 accepted_target=24\n"
                    "SUCCESS Step 4\n"
                    "PIPELINE_TELEMETRY accepted_buffer=1 accepted_target=24\n"
                ),
            ), mock.patch.object(sys, "argv", ["watch_rollouts_tqdm.py", "3"]), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                renderer.main()

        self.assertIn("SUCCESS Step 4", output.getvalue())
        self.assertEqual(make_bar.call_args_list[1].kwargs["desc"], "Step 5 rollouts")
        first_bar.close.assert_called_once_with()


class WatchRolloutsScriptTest(unittest.TestCase):
    def run_script(
        self,
        initial_state,
        final_state=None,
        renderer_exit=0,
        final_inspect_failure=False,
        clean_marker=False,
        follow_exit=0,
        progress_lines=True,
        marker_final_state=None,
    ):
        script_path = Path(__file__).parents[1] / "runtime/h200/watch_rollouts.sh"
        def normalize_state(value):
            if len(value.split("|")) == 5:
                return f"container-a|2026-09-29T07:00:00.000000000Z|{value}"
            return value

        initial_state = normalize_state(initial_state)
        final_state = normalize_state(final_state or initial_state)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sudo").write_text(
                "#!/bin/sh\nexec \"$@\"\n", encoding="utf-8"
            )
            (root / "docker").write_text(
                "#!/usr/bin/env bash\n"
                "set -u\n"
                "counter=${INSPECT_COUNTER:?}\n"
                "case \"${1:-}\" in\n"
                "  inspect)\n"
                "    count=0\n"
                "    if [ -f \"$counter\" ]; then count=$(cat \"$counter\"); fi\n"
                "    count=$((count + 1))\n"
                "    printf '%s\\n' \"$count\" >\"$counter\"\n"
                "    if [ \"$count\" -gt 1 ] && [ \"${FINAL_INSPECT_FAILURE:-0}\" = 1 ]; then\n"
                "      printf '%s\\n' 'docker daemon unavailable' >&2\n"
                "      exit 42\n"
                "    fi\n"
                "    if [ \"$count\" = 1 ]; then printf '%s\\n' \"$INITIAL_STATE\"; elif [ \"$count\" -gt 2 ] && [ -n \"${MARKER_FINAL_STATE:-}\" ]; then printf '%s\\n' \"$MARKER_FINAL_STATE\"; else printf '%s\\n' \"$FINAL_STATE\"; fi\n"
                "    ;;\n"
                "  exec)\n"
                "    case \"$*\" in\n"
                "      *'-c import tqdm'*) exit 0 ;;\n"
                "    esac\n"
                "    cat >/dev/null\n"
                "    exit \"${RENDERER_EXIT:-0}\"\n"
                "    ;;\n"
                "  logs)\n"
                "    case \"$*\" in\n"
                "      *'--tail 20000'*)\n"
                "        if [ \"${CLEAN_MARKER:-0}\" = 1 ]; then printf '%s\\n' 'Orchestrator finished.'; fi\n"
                "        exit 0\n"
                "        ;;\n"
                "      *'--tail 200'*)\n"
                "        if [ \"${PROGRESS_LINES:-1}\" = 1 ]; then printf '%s\\n' 'PIPELINE_TELEMETRY accepted_buffer=1 accepted_target=2'; fi\n"
                "        ;;\n"
                "      *'--follow'*)\n"
                "        if [ \"${FOLLOW_EXIT:-0}\" != 0 ]; then exit \"$FOLLOW_EXIT\"; fi\n"
                "        if [ \"${PROGRESS_LINES:-1}\" = 1 ]; then printf '%s\\n' 'PIPELINE_TELEMETRY accepted_buffer=2 accepted_target=2'; fi\n"
                "        ;;\n"
                "    esac\n"
                "    ;;\n"
                "  *) exit 99 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            renderer_path = root / "renderer.py"
            renderer_path.write_text("# test renderer\n", encoding="utf-8")
            for path in (root / "sudo", root / "docker"):
                path.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{root}:/usr/bin:/bin",
                    "INITIAL_STATE": initial_state,
                    "FINAL_STATE": final_state,
                    "MARKER_FINAL_STATE": marker_final_state or "",
                    "RENDERER_EXIT": str(renderer_exit),
                    "INSPECT_COUNTER": str(root / "inspect-count"),
                    "FINAL_INSPECT_FAILURE": "1" if final_inspect_failure else "0",
                    "CLEAN_MARKER": "1" if clean_marker else "0",
                    "FOLLOW_EXIT": str(follow_exit),
                    "PROGRESS_LINES": "1" if progress_lines else "0",
                }
            )
            return subprocess.run(
                ["bash", str(script_path), "orchestrator", str(renderer_path)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_normal_container_exit_does_not_fail_on_renderer_sigkill(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=exited exit_code=0 oom_killed=false", result.stdout)
        self.assertIn("completed normally", result.stdout)

    def test_normal_renderer_exit_requires_clean_marker(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=0,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("stopped before clean completion", result.stderr)

    def test_successful_pipeline_accepts_clean_marker(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=0,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed normally", result.stdout)

    def test_clean_renderer_exit_accepts_current_marker_without_progress_lines(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=0,
            clean_marker=True,
            progress_lines=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed normally", result.stdout)

    def test_clean_renderer_sigkill_accepts_current_marker_without_progress_lines(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
            progress_lines=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed normally", result.stdout)

    def test_same_id_restart_after_marker_is_not_clean_completion(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
            marker_final_state=(
                "container-a|2026-09-29T07:00:01.000000000Z|"
                "false|exited|0|false|"
            ),
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("identity changed after completion marker", result.stderr)
        self.assertNotIn("completed normally", result.stdout)

    def test_renderer_sigkill_requires_clean_marker(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("stopped abnormally", result.stderr)

    def test_non_sigkill_renderer_failure_is_not_suppressed(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=2,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("stopped abnormally", result.stderr)

    def test_successful_pipeline_still_fails_for_final_oom(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|137|true|memory limit",
            renderer_exit=0,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("oom_killed=true", result.stdout)

    def test_successful_pipeline_still_fails_for_final_inspect_error(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=0,
            final_inspect_failure=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("state unavailable after producer exit 0 and renderer exit 0", result.stderr)

    def test_old_marker_from_reused_container_is_not_clean_completion(self):
        result = self.run_script(
            "true|running|0|false|",
            "container-b|2026-09-29T07:00:01.000000000Z|false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("identity changed during watch", result.stderr)
        self.assertNotIn("completed normally", result.stdout)

    def test_old_marker_from_restarted_container_is_not_clean_completion(self):
        result = self.run_script(
            "true|running|0|false|",
            "container-a|2026-09-29T07:00:01.000000000Z|false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("identity changed during watch", result.stderr)

    def test_producer_failure_is_not_masked_by_renderer_sigkill(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
            clean_marker=True,
            follow_exit=42,
        )

        self.assertEqual(result.returncode, 42)
        self.assertIn("producer failed with exit code 42", result.stderr)
        self.assertIn("renderer exit code 137", result.stderr)

    def test_producer_sigkill_is_not_masked_by_renderer_success(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=0,
            clean_marker=True,
            follow_exit=137,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("producer failed with exit code 137", result.stderr)
        self.assertIn("renderer exit code 0", result.stderr)

    def test_initial_created_state_is_not_normal_completion(self):
        result = self.run_script("false|created|0|false|")

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("completed normally", result.stdout)

    def test_initial_state_error_is_not_normal_completion(self):
        result = self.run_script("false|exited|0|false|startup failure")

        self.assertEqual(result.returncode, 1)
        self.assertIn("error=startup failure", result.stdout)
        self.assertNotIn("completed normally", result.stdout)

    def test_renderer_sigkill_is_not_masked_while_container_is_running(self):
        result = self.run_script(
            "true|running|0|false|",
            "true|running|0|false|",
            renderer_exit=137,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("status=running exit_code=0 oom_killed=false", result.stdout)
        self.assertIn("stopped abnormally", result.stderr)

    def test_oom_kill_is_not_masked(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|137|true|memory limit",
            renderer_exit=137,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("status=exited exit_code=137 oom_killed=true", result.stdout)
        self.assertIn("error=memory limit", result.stdout)
        self.assertIn("stopped abnormally", result.stderr)

    def test_state_inspect_failure_is_not_masked(self):
        result = self.run_script(
            "true|running|0|false|",
            "false|exited|0|false|",
            renderer_exit=137,
            final_inspect_failure=True,
        )

        self.assertEqual(result.returncode, 137)
        self.assertIn("state unavailable after producer exit 0 and renderer exit 137", result.stderr)
        self.assertIn("docker daemon unavailable", result.stderr)

    def test_initial_normal_stop_reports_exact_state(self):
        result = self.run_script("false|exited|0|false|")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=exited exit_code=0 oom_killed=false", result.stdout)
        self.assertIn("completed normally", result.stdout)

    def test_seeds_tail_only_and_bounds_completed_step_scan(self):
        script = (
            Path(__file__).parents[1] / "runtime/h200/watch_rollouts.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'docker logs --tail 20000 "$container"',
            script,
        )
        self.assertIn(
            'docker logs --tail 200 "$container"',
            script,
        )
        self.assertNotIn(
            'docker logs --since "$since" --tail 200 "$container"',
            script,
        )
        self.assertIn(
            "grep -aE 'PIPELINE_TELEMETRY[[:space:]]|Train batch|SUCCESS[[:space:]]+Step'",
            script,
        )
        self.assertIn(
            "grep --line-buffered -aE 'PIPELINE_TELEMETRY[[:space:]]|Train batch|SUCCESS[[:space:]]+Step'",
            script,
        )
        self.assertIn(
            "{{.Id}}|{{.State.StartedAt}}|{{.State.Running}}|{{.State.Status}}|{{.State.ExitCode}}|{{.State.OOMKilled}}|{{.State.Error}}",
            script,
        )
        self.assertIn("Orchestrator finished", script)
        self.assertIn(
            'docker logs --since "$initial_container_started_at" --tail 20000 "$container_id"',
            script,
        )
        self.assertIn('pipeline_statuses=("${PIPESTATUS[@]}")', script)


if __name__ == "__main__":
    unittest.main()
