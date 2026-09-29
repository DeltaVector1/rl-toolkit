import contextlib
import io
import sys
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


if __name__ == "__main__":
    unittest.main()
