import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import training_ops


class H200ParityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
