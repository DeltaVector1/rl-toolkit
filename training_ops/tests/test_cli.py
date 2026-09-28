import contextlib
import io
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
