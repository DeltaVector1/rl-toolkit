import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class MicrobatchAddressTest(unittest.TestCase):
    def test_profile_is_selected_from_active_run_output(self):
        self.assertEqual(
            training_ops._profile_for_run("x85")["trainer"],
            "configs/paused/glm52/trainer.toml",
        )
        self.assertEqual(
            training_ops._profile_for_run("muse21")["trainer"],
            "configs/current/trainer.toml",
        )

    def test_unknown_run_has_no_profile_fallback(self):
        with self.assertRaisesRegex(training_ops.Refused, "no configured trainer profile"):
            training_ops._profile_for_run("unknown-run")

    def test_ssh_alias_resolution_is_local_and_returns_internal_ip(self):
        result = training_ops.subprocess.CompletedProcess(
            ["ssh"], 0, "host n76\nhostname 10.92.20.76\n", ""
        )
        with mock.patch.object(training_ops.subprocess, "run", return_value=result) as run:
            self.assertEqual(training_ops._ssh_address("n76"), "10.92.20.76")

        command = run.call_args.args[0]
        self.assertIn("-G", command)
        self.assertEqual(command[-1], "n76")
        self.assertIs(training_ops.subprocess.DEVNULL, run.call_args.kwargs["stdin"])

    def test_ssh_alias_resolution_rejects_non_ip_hostname(self):
        result = training_ops.subprocess.CompletedProcess(
            ["ssh"], 0, "hostname trainer.example\n", ""
        )
        with mock.patch.object(training_ops.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "not an IP address"):
                training_ops._ssh_address("n76")


if __name__ == "__main__":
    unittest.main()
