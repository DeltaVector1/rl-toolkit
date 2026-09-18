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


if __name__ == "__main__":
    unittest.main()
