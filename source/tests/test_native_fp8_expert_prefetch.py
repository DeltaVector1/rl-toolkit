import ast
import unittest
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "source/src/prime_rl/trainer/model.py"


class NativeFP8ExpertPrefetchTest(unittest.TestCase):
    def test_native_fp8_expert_prefetch_is_explicit(self):
        tree = ast.parse(MODEL.read_text())
        setup = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_fsdp"
        )
        source = ast.unparse(setup)

        self.assertEqual(
            source.count("isinstance(next_mlp.experts, FSDPModule)"),
            1,
        )
        self.assertEqual(
            source.count("isinstance(prev_mlp.experts, FSDPModule)"),
            1,
        )
        self.assertIn("prefetch_modules.append(next_mlp.experts)", source)
        self.assertIn("prefetch_modules.append(prev_mlp.experts)", source)
        self.assertEqual(source.count("config.native_fp8_expert_prefetch"), 2)

        trainer_config = (
            ROOT
            / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
        ).read_text()
        self.assertIn("native_fp8_expert_prefetch: bool = False", trainer_config)

    def test_b2048_profile_uses_64_full_context_microbatches(self):
        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        relative = operations["profiles"]["glm52-native-b2048"]["trainer"]
        with (ROOT / relative).open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["max_steps"], 2)
        self.assertEqual(
            config["data"], {"fake": {"batch_size": 64, "generate_samples": True}}
        )
        self.assertEqual(config["model"]["seq_len"], 32768)
        self.assertEqual(config["model"]["pp"], 8)
        self.assertEqual(config["model"]["cp"], 8)
        self.assertEqual(config["model"]["ep"], 8)
        self.assertIs(config["model"]["reshard_after_forward"], True)
        self.assertFalse(config["model"]["native_fp8_expert_prefetch"])

    def test_stability_profile_matches_production_shape(self):
        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        relative = operations["profiles"]["glm52-native-stability20"]["trainer"]
        self.assertTrue(relative.startswith("configs/qualification/"))
        with (ROOT / relative).open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["max_steps"], 20)
        self.assertFalse(config["enable_router_replay"])
        self.assertEqual(
            config["data"], {"fake": {"batch_size": 32, "generate_samples": True}}
        )
        self.assertEqual(config["model"]["seq_len"], 32768)
        self.assertEqual(config["model"]["pp"], 8)
        self.assertEqual(config["model"]["cp"], 8)
        self.assertEqual(config["model"]["ep"], 8)
        self.assertIs(config["model"]["reshard_after_forward"], True)
        self.assertFalse(config["model"]["native_fp8_expert_prefetch"])
        self.assertTrue(config["model"]["freeze_moe_router"])
        self.assertEqual(config["optim"]["lr"], 0.0)
        self.assertEqual(config["model"]["lora"]["rank"], 32)
        self.assertIn("experts", config["model"]["lora"]["target_modules"])
        self.assertIn("shared_expert", config["model"]["lora"]["target_modules"])

        data_source = (ROOT / "source/src/prime_rl/trainer/rl/data.py").read_text()
        self.assertIn("self.batch_counter * 1000 + micro_batch_idx", data_source)

    def test_production_keeps_rejected_features_disabled(self):
        with (ROOT / "configs/paused/trainer.glm52-production.toml").open("rb") as stream:
            production = tomllib.load(stream)
        self.assertIs(production["model"]["reshard_after_forward"], True)
        self.assertFalse(production["model"]["native_fp8_expert_prefetch"])


if __name__ == "__main__":
    unittest.main()
