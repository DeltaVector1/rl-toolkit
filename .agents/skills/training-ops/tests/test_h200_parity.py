import json
import unittest
from unittest.mock import patch

import training_ops


class H200ParityTest(unittest.TestCase):
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
