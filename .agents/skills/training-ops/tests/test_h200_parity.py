import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import training_ops


class H200ParityTest(unittest.TestCase):
    def _operation_ssh(
        self,
        commands,
        *,
        fail_start=False,
        fail_capture=False,
        fail_cleanup=False,
        main_status="absent",
    ):
        secret = "strong-bootstrap-token-for-test-only-0123456789"
        secret_inputs = []

        def call(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            if "sha256sum --" in command:
                return 0, "config-sha\n"
            if "docker ps -a" in command and "glm52-optims-h200-" in command:
                return 0, "absent\n"
            if "docker ps -a" in command and "glm52-optims-h200" in command:
                return 0, f"{main_status}\n"
            if "start_inference.sh" in command and fail_start:
                return 1, f"start rejected {secret}"
            if "capture-h200" in command and fail_capture:
                return 1, f"capture rejected {secret}"
            if command.startswith("rm -f --") and fail_cleanup:
                return 1, "cleanup failed"
            return 0, ""

        def secret_call(host, command, provided, label, timeout):
            secret_inputs.append(provided)
            code, output = call(host, command, timeout)
            clean = training_ops._redact_sensitive(output, (provided,))
            if code:
                raise RuntimeError(f"{label} failed on {host}: {clean}")
            return clean

        return secret, call, secret_call, secret_inputs

    def _run_operation(self, ssh_call, secret_call, secret, *, keep_engine=False):
        metadata = json.dumps(
            {"model_revision": "a" * 40, "adapter_sha256": "b" * 64}
        )
        with patch.object(training_ops.secrets, "token_urlsafe", return_value=secret), patch.object(
            training_ops.secrets, "token_hex", return_value="1" * 16
        ), patch.object(
            training_ops, "_diagnostic_capture_preflight", return_value={"files": 20, "estimated_bytes": 1}
        ), patch.object(
            training_ops, "_remote_ok", return_value=metadata
        ), patch.object(
            training_ops, "_sha256_file", return_value="config-sha"
        ), patch.object(
            training_ops, "_copy_parity_diagnostic_input"
        ), patch.object(
            training_ops, "_h200_parity_container_preflight"
        ), patch.object(
            training_ops, "_configured_h200_engine_startup_grace", return_value=7200
        ), patch.object(training_ops, "_ssh", side_effect=ssh_call), patch.object(
            training_ops, "_sensitive_ssh_input", side_effect=secret_call
        ):
            return training_ops.diagnose_glm52_policy_parity(
                "qualification-1", keep_engine=keep_engine
            )

    def test_diagnostic_capture_preflight_fits_existing_h200_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "h200.toml"
            config.write_text(
                "[env_vars]\n"
                'PRIME_PARITY_DIAGNOSTICS = "1"\n'
                'PRIME_PARITY_DIAGNOSTIC_TOKENS = "256"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_FILES = "64"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_BYTES = "100000000"\n'
            )
            result = training_ops._diagnostic_capture_preflight(config, "h200")
        self.assertEqual(result, {"files": 20, "estimated_bytes": 96_485_376})

    def test_sensitive_ssh_input_keeps_token_out_of_command_and_redacts_error(self):
        secret = "strong-bootstrap-token-for-test-only-0123456789"
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=f"remote repeated {secret}"
        )
        with patch.object(training_ops.subprocess, "run", return_value=completed) as run:
            with self.assertRaises(RuntimeError) as caught:
                training_ops._sensitive_ssh_input(
                    "h200a", "run isolated command", secret, "diagnostic", 30
                )
        self.assertNotIn(secret, " ".join(run.call_args.args[0]))
        self.assertEqual(run.call_args.kwargs["input"], secret + "\n")
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_diagnostic_capture_preflight_refuses_existing_cap_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "h200.toml"
            config.write_text(
                "[env_vars]\n"
                'PRIME_PARITY_DIAGNOSTICS = "1"\n'
                'PRIME_PARITY_DIAGNOSTIC_TOKENS = "256"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_FILES = "64"\n'
                'PRIME_PARITY_DIAGNOSTIC_MAX_BYTES = "96485375"\n'
            )
            with self.assertRaises(training_ops.Refused):
                training_ops._diagnostic_capture_preflight(config, "h200")

    def test_post_launch_inspection_accepts_exact_command_and_config(self):
        command = [
            "@",
            training_ops.glm_parity_inference_config,
            "--vllm-extra",
            training_ops.glm_parity_vllm_extra,
        ]
        inspect = "\n".join(
            [json.dumps(True), json.dumps(training_ops.inference_image), json.dumps(command)]
        )
        config = json.dumps(
            {
                "max_model_len": 32769,
                "tp": 8,
                "dp": 1,
                "lora_rank": 32,
                "router_trace": True,
            }
        )
        with patch.object(training_ops, "_ssh", side_effect=[(0, inspect), (0, config)]):
            result = training_ops._h200_parity_container_preflight("h200a")
        self.assertEqual(result["config"]["max_model_len"], 32769)

    def test_post_launch_inspection_refuses_wrong_config_command(self):
        command = [
            "@",
            f"{training_ops.remote_repo}/configs/h200/inference.toml",
            "--vllm-extra",
            training_ops.glm_parity_vllm_extra,
        ]
        inspect = "\n".join(
            [json.dumps(True), json.dumps(training_ops.inference_image), json.dumps(command)]
        )
        with patch.object(training_ops, "_ssh", return_value=(0, inspect)):
            with self.assertRaises(training_ops.Refused):
                training_ops._h200_parity_container_preflight("h200a")

    def test_diagnostic_operation_redacts_token_and_cleans_up_after_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_capture=True
        )
        with self.assertRaises(RuntimeError) as caught:
            self._run_operation(ssh_call, secret_call, secret, keep_engine=True)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))
        self.assertTrue(
            any(
                'name=glm52-optims-h200' in command
                and 'sudo docker rm -f "$name"' in command
                for command in commands
            )
        )

    def test_diagnostic_operation_uses_memory_token_and_exact_cleanup(self):
        commands = []
        secret, ssh_call, secret_call, secret_inputs = self._operation_ssh(
            commands, main_status="exited"
        )
        result = self._run_operation(ssh_call, secret_call, secret)
        self.assertNotIn(secret, json.dumps(result, sort_keys=True))
        self.assertEqual(secret_inputs, [secret, secret])
        self.assertFalse(any(secret in command for command in commands))
        start = next(
            command
            for command in commands
            if "start_inference.sh" in command and "prime_deployment=" in command
        )
        self.assertIn("prime_require_absent=1", start)
        self.assertTrue(any("capture-h200" in command for command in commands))
        self.assertFalse(any("training.env" in command for command in commands))
        self.assertFalse(
            any(
                "trainer-traces" in command and "rm " in command
                for command in commands
            )
        )
        removals = [command for command in commands if "docker rm" in command]
        self.assertIn(
            "sudo docker rm glm52-optims-h200 >/dev/null",
            removals,
        )
        final_cleanup = next(command for command in removals if 'rm -f "$name"' in command)
        self.assertIn('name=glm52-optims-h200', final_cleanup)
        self.assertIn('name=^/${name}$', final_cleanup)
        self.assertFalse(any("*" in command for command in removals))

    def test_diagnostic_operation_cleans_exact_container_after_start_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(commands, fail_start=True)
        with self.assertRaises(RuntimeError) as caught:
            self._run_operation(ssh_call, secret_call, secret, keep_engine=True)
        self.assertNotIn(secret, str(caught.exception))
        self.assertTrue(
            any(
                'name=glm52-optims-h200' in command
                and 'sudo docker rm -f "$name"' in command
                for command in commands
            )
        )

    def test_diagnostic_operation_reports_cleanup_failure(self):
        commands = []
        secret, ssh_call, secret_call, _ = self._operation_ssh(
            commands, fail_cleanup=True
        )
        with self.assertRaisesRegex(RuntimeError, "artifact cleanup failed"):
            self._run_operation(ssh_call, secret_call, secret)


if __name__ == "__main__":
    unittest.main()
