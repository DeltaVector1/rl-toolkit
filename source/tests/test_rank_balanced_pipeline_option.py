from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/qualification/glm52-native-interleaved-rank-balanced.toml"
EXPECTED_BOUNDARIES = [
    0, 3, 7, 11, 15, 19, 23, 27, 31,
    36, 42, 48, 54, 60, 66, 72, 78,
]
EXPECTED_TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}


class RankBalancedPipelineOptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("rb") as file:
            cls.config = tomllib.load(file)

    def test_option_is_explicit_and_disabled_elsewhere(self):
        model = self.config["model"]
        self.assertEqual(model["pp_schedule"], "interleaved_1f1b")
        self.assertEqual(model["pp_boundary_plan"], "explicit")
        self.assertEqual(model["pp"], 8)
        self.assertNotIn("pp_virtual_layer_boundaries", tomllib.loads(
            (ROOT / "configs/qualification/glm52-native-b2048.toml").read_text()
        )["model"])

    def test_layers_are_owned_once_by_pp8_pairs(self):
        bounds = self.config["model"]["pp_virtual_layer_boundaries"]
        self.assertEqual(bounds, EXPECTED_BOUNDARIES)
        ranges = list(zip(bounds, bounds[1:]))
        self.assertEqual(len(ranges), 16)
        owned = [layer for start, end in ranges for layer in range(start, end)]
        self.assertEqual(len(owned), 78)
        self.assertEqual(sorted(owned), list(range(78)))

        counts = [end - start for start, end in ranges]
        self.assertEqual(
            [counts[rank] + counts[rank + 8] for rank in range(8)],
            [8, 10, 10, 10, 10, 10, 10, 10],
        )
        self.assertEqual(counts[0] + counts[8], 8)

    def test_cutoff_maps_to_layer_48(self):
        model = self.config["model"]
        physical = model["pp_layer_boundaries"]
        cutoff = model["ac"]["keep_routed_expert_activations_from_pp_stage"]
        self.assertEqual(physical[cutoff], 48)
        self.assertEqual(model["pp_virtual_layer_boundaries"][11], 48)

    def test_lora_targets_are_complete(self):
        self.assertEqual(
            set(self.config["model"]["lora"]["target_modules"]),
            EXPECTED_TARGETS,
        )

    def test_option_contains_no_host_assignment(self):
        text = CONFIG_PATH.read_text()
        self.assertNotRegex(text, r"(?im)^\s*(host|hostname|ssh)\s*=")
        self.assertNotRegex(text, r"\bn(?:79|82|84|85)\b")

    def test_operations_and_existing_allowlists_register_option(self):
        operations = tomllib.loads((ROOT / "configs/operations.toml").read_text())
        profile = operations["profiles"]["glm52-native-interleaved-rank-balanced"]
        self.assertEqual(profile["trainer"], str(CONFIG_PATH.relative_to(ROOT)))
        command = operations["qualification"]["glm52_rank_balanced_cuda"]
        self.assertEqual(command["host_group"], "trainer_head")
        self.assertEqual(command["gpus"], 8)
        self.assertEqual(command["network"], "host")
        self.assertEqual(command["outer_timeout_seconds"], 300)
        self.assertEqual(command["process_group_timeout_seconds"], 120)
        self.assertNotIn("n82", json.dumps(command))

        sources = [
            ROOT / "scripts/china/cluster.sh",
            ROOT / "source/src/prime_rl/trainer/rl/train.py",
            ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py",
        ]
        for source in sources:
            self.assertIn("glm52-native-interleaved-rank-balanced", source.read_text())

    def test_cuda_helper_checks_identity_before_process_group_init(self):
        source = (ROOT / "source/tests/test_final_pp16_cuda.py").read_text()
        launch = source.index("distributed_launch()")
        initialize = source.index("dist.init_process_group(", launch)
        self.assertLess(launch, initialize)
        self.assertIn("world_size != PHYSICAL", source[launch:initialize])
        self.assertIn("local_world_size != PHYSICAL", source[launch:initialize])
        self.assertIn("torch.cuda.set_device(device)", source[launch:initialize])
        self.assertIn("timeout=timedelta(seconds=timeout_seconds)", source)
        self.assertIn("device_id=device", source)

    def test_safe_cuda_command_uses_inventory_and_exact_cleanup(self):
        module_path = ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
        spec = importlib.util.spec_from_file_location(
            "training_ops_rank_balanced_test", module_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        training_ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = training_ops
        spec.loader.exec_module(training_ops)

        report = {
            "virtual_boundaries": EXPECTED_BOUNDARIES,
            "real_forward_actions": 1520,
            "real_backward_actions": 1520,
            "projected_peak_bytes_max": 1,
        }
        result = {"status": "pass", "ranks": 8, "repeats": 3, "reports": [report] * 3}
        calls = []

        def fake_ssh(host, command, timeout=120, **_kwargs):
            calls.append((host, command, timeout))
            if len(calls) == 1:
                return 0, "GPUS 8\nPIDS\nRUNNING\n"
            if len(calls) == 2:
                return 0, "STALE none\n"
            if len(calls) == 3:
                return 0, json.dumps(result)
            return 0, "CONTAINERS\nPIDS\n"

        deployed = []
        training_ops._assert_live_services_are_down = (
            lambda nodes, include_inference: self.assertEqual(
                (nodes, include_inference), ([training_ops.trainer_head], False)
            )
        )
        training_ops._ssh = fake_ssh
        training_ops.deploy = (
            lambda path, **kwargs: deployed.append((path, kwargs)) or True
        )
        training_ops.secrets.token_hex = lambda _size: "0123456789abcdef"

        observed = training_ops.qualify_glm52_rank_balanced_cuda()
        self.assertEqual(observed["status"], "pass")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[0] == training_ops.trainer_head for call in calls))
        self.assertEqual(
            [path for path, _options in deployed],
            [
                "source/tests/test_final_pp16_cuda.py",
                "source/src/prime_rl/trainer/distributed/pp_runner.py",
            ],
        )
        self.assertTrue(
            all(options["nodes"] == [training_ops.trainer_head] for _, options in deployed)
        )
        launch = calls[2][1]
        self.assertIn("--network host", launch)
        self.assertIn("--nproc-per-node=8", launch)
        self.assertIn("--kill-after=20s 300s", launch)
        self.assertIn("PRIME_PP_CUDA_PROCESS_GROUP_TIMEOUT_SECONDS=120", launch)
        self.assertIn("NCCL_SOCKET_IFNAME=vlan130", launch)
        self.assertIn("trap cleanup EXIT", launch)
        self.assertIn("docker inspect --type container", launch)
        self.assertIn("docker rm -f", launch)
        self.assertNotIn("n79", launch)
        self.assertNotIn("rollout", launch.lower())
        self.assertNotIn("cache", launch.lower())
        self.assertNotIn("/models", launch)
        self.assertEqual(calls[2][2], 360)


if __name__ == "__main__":
    unittest.main()
