import ast
import json
import pathlib
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import training_ops


ROOT = pathlib.Path(training_ops.__file__).resolve().parents[5]
SCRIPT = ROOT / "scripts/ops/benchmark_fused_qkv_a_lora_sm89.py"


class LoraFusionSmokeTest(unittest.TestCase):
    def test_registered_command_is_bounded_and_inventory_driven(self):
        host, image, script, module, remote_timeout, ssh_timeout = (
            training_ops._lora_fusion_smoke_spec()
        )
        self.assertEqual(host, training_ops.trainer_head)
        self.assertEqual(image, training_ops.containers["trainer_image"])
        self.assertEqual(script, "scripts/ops/benchmark_fused_qkv_a_lora_sm89.py")
        self.assertEqual(
            module,
            "source/src/prime_rl/trainer/models/layers/lora/multi_linear.py",
        )
        self.assertEqual((remote_timeout, ssh_timeout), (180, 240))

    def test_remote_command_is_gpu_zero_ephemeral_and_model_free(self):
        command = training_ops._lora_fusion_smoke_command(
            "/data/glm52-rl/deployment-glm52-optims",
            "prime-rl:glm52-v080-qual",
            "scripts/ops/benchmark_fused_qkv_a_lora_sm89.py",
            "source/src/prime_rl/trainer/models/layers/lora/multi_linear.py",
            "glm52-lora-fusion-smoke-0123456789abcdef",
            0,
            180,
        )
        for text in (
            "--rm",
            "--gpus device=\"$gpu\"",
            "--network none",
            "--read-only",
            "--iterations 20",
            "timeout --foreground --signal=TERM --kill-after=20s 180s",
            "docker rm -f",
            "nvidia-smi",
        ):
            self.assertIn(text, command)
        for text in ("src=/models/", "rollout", "PRIME_BATCH_CACHE", "ZiphuAI"):
            self.assertNotIn(text, command)

    def test_benchmark_declares_glm_shapes_and_twenty_repeats(self):
        source = SCRIPT.read_text()
        tree = ast.parse(source)
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and node.targets[0].id
            in {"RANK", "HIDDEN", "Q_OUTPUT", "KV_OUTPUT", "DEFAULT_ITERATIONS"}
        }
        self.assertEqual(
            tuple(constants[name] for name in ("RANK", "HIDDEN", "Q_OUTPUT", "KV_OUTPUT")),
            (32, 6144, 2048, 576),
        )
        self.assertGreaterEqual(constants["DEFAULT_ITERATIONS"], 20)
        self.assertIn("_FusedPairLoRA", source)

    def test_command_runs_only_after_live_service_preflight(self):
        result = {"status": "passed", "iterations": 20}
        with patch.object(
            training_ops, "_ssh_control_masters", return_value=nullcontext()
        ), patch.object(
            training_ops, "_assert_live_services_are_down"
        ) as preflight, patch.object(
            training_ops, "deploy", return_value=True
        ) as deploy, patch.object(
            training_ops, "_ssh", return_value=(0, json.dumps(result))
        ) as ssh:
            observed = training_ops.qualify_glm52_lora_fusion()
        self.assertEqual(observed, result)
        preflight.assert_called_once_with(
            [training_ops.trainer_head], include_inference=False
        )
        self.assertEqual(deploy.call_count, 2)
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args.kwargs["timeout"], 240)


if __name__ == "__main__":
    unittest.main()
