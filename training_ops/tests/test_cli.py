import contextlib
import io
import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

import training_ops


class CliTest(unittest.TestCase):
    def test_stop_rejects_extra_arguments_without_stopping(self):
        for argument in ("--help", "unexpected"):
            with self.subTest(argument=argument), mock.patch.object(training_ops, "stop") as stop:
                with mock.patch.object(sys, "argv", ["training-ops", "stop", argument]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exited:
                        training_ops.cli()
                self.assertEqual(exited.exception.code, 2)
                stop.assert_not_called()

    def test_bare_stop_stops_once(self):
        with mock.patch.object(training_ops, "stop") as stop, mock.patch.object(
            sys, "argv", ["training-ops", "stop"]
        ):
            training_ops.cli()
        stop.assert_called_once_with()

    def test_retention_defaults_to_dry_run(self):
        with mock.patch.object(
            training_ops,
            "_broadcast_retention_cleanup",
            return_value={"dry_run": True},
        ) as cleanup, mock.patch.object(
            sys,
            "argv",
            ["training-ops", "retention", "/tmp/run_demo", "--evidence", "/tmp/evidence.json"],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        cleanup.assert_called_once_with(
            "/tmp/run_demo",
            "/tmp/evidence.json",
            keep_latest=2,
            partial_age_seconds=1800.0,
            max_entries=8,
            max_bytes=50 * 1024**3,
            dry_run=True,
        )

    def test_retention_apply_parses_bounded_options(self):
        with mock.patch.object(
            training_ops,
            "_broadcast_retention_cleanup",
            return_value={"dry_run": False},
        ) as cleanup, mock.patch.object(
            sys,
            "argv",
            [
                "training-ops", "retention", "/tmp/run_demo", "--evidence", "/tmp/evidence.json",
                "--apply", "--keep-latest", "3", "--partial-age-seconds", "60",
                "--max-entries", "2", "--max-bytes", "1024",
            ],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        cleanup.assert_called_once_with(
            "/tmp/run_demo",
            "/tmp/evidence.json",
            keep_latest=3,
            partial_age_seconds=60.0,
            max_entries=2,
            max_bytes=1024,
            dry_run=False,
        )

    def test_retention_rejects_missing_evidence_without_calling_cleanup(self):
        with mock.patch.object(training_ops, "_broadcast_retention_cleanup") as cleanup, mock.patch.object(
            sys, "argv", ["training-ops", "retention", "/tmp/run_demo"]
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        cleanup.assert_not_called()

    def test_watch_rollouts_builds_exact_h200_foreground_command(self):
        options = ["-p", "20099", "-i", "/tmp/h200-key"]
        destination = "root@38.255.28.19"
        with mock.patch.object(
            training_ops,
            "_ssh_destination",
            return_value=(options, [], destination),
        ) as ssh_destination, mock.patch.object(
            training_ops.subprocess,
            "run",
            return_value=mock.Mock(returncode=0),
        ) as run_process:
            self.assertEqual(training_ops._watch_rollouts_action(["--host", "h200"]), 0)

        ssh_destination.assert_called_once_with(training_ops.orchestrator_host)
        remote_script = shlex.join(
            [
                "bash",
                f"{training_ops.runtime_root}/h200/watch_rollouts.sh",
                training_ops.orchestrator_container,
                f"{training_ops.runtime_root}/h200/watch_rollouts_tqdm.py",
            ]
        )
        run_process.assert_called_once_with(
            [{"ConnectTimeout=25": "ConnectTimeout=5", "ServerAliveInterval=15": "ServerAliveInterval=2",
              "ServerAliveCountMax=4": "ServerAliveCountMax=2"}.get(value, value)
             for value in training_ops.ssh_command] + options + ["-tt", destination, remote_script],
            check=False,
            timeout=610,
        )

    def test_watch_rollouts_rejects_arguments(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            training_ops._watch_rollouts_action(["10m"])

    def test_watch_rollouts_propagates_remote_failure(self):
        with mock.patch.object(
            training_ops.subprocess,
            "run",
            return_value=mock.Mock(returncode=17),
        ):
            with self.assertRaisesRegex(RuntimeError, "exit code 17"):
                training_ops._watch_rollouts_action(["--host", "h200"])

    def test_watch_rollouts_propagates_keyboard_interrupt(self):
        with mock.patch.object(
            training_ops.subprocess,
            "run",
            side_effect=KeyboardInterrupt,
        ) as run_process:
            with self.assertRaises(KeyboardInterrupt):
                training_ops._watch_rollouts_action(["--host", "h200"])
        run_process.assert_called_once()

    def test_watch_rollouts_fallback_uses_deployed_repository_root(self):
        script_path = (
            training_ops.repo_root / "training_ops/runtime/h200/watch_rollouts.sh"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn(
            'repository=$(cd -- "$runtime_dir/../../.." && pwd)',
            script,
        )
        runtime_dir = Path(training_ops.remote_repo) / "training_ops/runtime/h200"
        self.assertEqual((runtime_dir / "../../..").resolve(), Path(training_ops.remote_repo))


if __name__ == "__main__":
    unittest.main()
