import sys
import subprocess
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class MicrobatchAddressTest(unittest.TestCase):
    def test_profile_is_selected_from_active_run_output(self):
        self.assertEqual(
            training_ops._profile_for_run("glm-1")["trainer"],
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

    def test_ssh_timeout_terminates_the_full_process_group(self):
        process = mock.Mock(pid=4312, returncode=124)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["ssh"], 1),
            ("", ""),
        ]
        with mock.patch.object(training_ops.subprocess, "Popen", return_value=process) as popen:
            with mock.patch.object(training_ops.os, "killpg") as killpg:
                self.assertEqual(training_ops._ssh("n76", "true", timeout=1), (124, "TIMEOUT"))

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(4312, training_ops.signal.SIGTERM)

    def test_inference_script_uses_the_run_profile_ports(self):
        glm_script = training_ops._inference_script(
            "h200a", training_ops.glm_model, training_ops.glm_inference_container, "glm52"
        )
        muse_script = training_ops._inference_script(
            "h200a", training_ops.muse_model, training_ops.muse_inference_container, "muse"
        )
        self.assertIn("for port in 8000", glm_script)
        self.assertNotIn("8001", glm_script)
        self.assertIn("for port in 8001 8002 8003 8004 8005 8006 8007 8008", muse_script)


if __name__ == "__main__":
    unittest.main()
