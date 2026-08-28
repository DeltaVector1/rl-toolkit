import ast
import inspect
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
            "assert_container_name_free",
            "docker ps -a",
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
        self.assertIn("_SingleLoRA.apply", source)
        self.assertIn("_FusedPairLoRA", source)
        self.assertIn("_error_metrics", source)
        self.assertIn("max_abs_error", source)
        self.assertIn("max_relative_error", source)
        self.assertIn("reference_scale", source)
        self.assertIn("reference_norm", source)
        self.assertIn("candidate_norm", source)
        self.assertIn("norm_relative_error", source)
        self.assertIn("reference_gradient_norms", source)
        self.assertIn('"forward_backward"', source)
        self.assertIn("_perturbation_checks", source)
        self.assertIn("_zero_candidate_checks", source)
        self.assertIn("forward output", source)
        self.assertIn("kv_a output", source)
        self.assertIn("input gradient", source)
        self.assertIn("parameter gradient", source)
        for name in ("q_a", "q_b", "kv_a", "kv_b"):
            self.assertIn(f'"{name} gradient"', source)
        multi_linear = (
            ROOT / "source/src/prime_rl/trainer/models/layers/lora/multi_linear.py"
        ).read_text()
        self.assertIn("def _lora_mm", multi_linear)

    def test_preflight_inventory_includes_configured_and_protected_trainers(self):
        names = training_ops._lora_fusion_smoke_trainer_containers()
        self.assertIn(training_ops.containers["trainer"], names)
        self.assertIn(
            training_ops.containers["glm52_qualification_trainer"], names
        )
        self.assertIn(training_ops.protected_trainer, names)

    def test_preflight_checks_every_trainer_host(self):
        seen = []

        def fake_ssh(host, command, timeout=120, connect_timeout=None):
            seen.append((host, command, timeout))
            return 0, ""

        with patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            training_ops._lora_fusion_smoke_preflight()
        self.assertEqual(
            {host for host, _command, _timeout in seen}, set(training_ops.trainer_hosts)
        )
        self.assertTrue(all(timeout == 60 for _host, _command, timeout in seen))
        self.assertTrue(
            all("docker ps --format" in command for _host, command, _timeout in seen)
        )
        self.assertTrue(
            all("nvidia-smi" in command for _host, command, _timeout in seen)
        )

    def test_preflight_running_check_allows_stopped_container_names(self):
        source = inspect.getsource(training_ops._lora_fusion_smoke_preflight)
        self.assertIn("docker ps --format", source)
        self.assertNotIn("docker ps -a", source)

    def test_preflight_refuses_a_configured_container(self):
        def fake_ssh(host, command, timeout=120, connect_timeout=None):
            if host == training_ops.trainer_head:
                return 20, "configured trainer container is present"
            return 0, ""

        with patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            with self.assertRaises(training_ops.Refused):
                training_ops._lora_fusion_smoke_preflight()

    def test_preflight_refuses_a_compute_process(self):
        def fake_ssh(host, command, timeout=120, connect_timeout=None):
            if host == training_ops.trainer_head:
                return 21, "GPU 0 has compute processes"
            return 0, ""

        with patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            with self.assertRaises(training_ops.Refused):
                training_ops._lora_fusion_smoke_preflight()

    def test_command_runs_only_after_live_service_preflight(self):
        result = {"status": "passed", "iterations": 20}
        with patch.object(
            training_ops, "_ssh_control_masters", return_value=nullcontext()
        ), patch.object(
            training_ops, "_lora_fusion_smoke_preflight"
        ) as smoke_preflight, patch.object(
            training_ops, "_assert_live_services_are_down"
        ) as preflight, patch.object(
            training_ops, "deploy", return_value=True
        ) as deploy, patch.object(
            training_ops, "_ssh", return_value=(0, json.dumps(result))
        ) as ssh:
            observed = training_ops.qualify_glm52_lora_fusion()
        self.assertEqual(observed, result)
        smoke_preflight.assert_called_once_with()
        preflight.assert_called_once_with(
            [training_ops.trainer_head], include_inference=False
        )
        self.assertEqual(deploy.call_count, 2)
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args.kwargs["timeout"], 240)


if __name__ == "__main__":
    unittest.main()
