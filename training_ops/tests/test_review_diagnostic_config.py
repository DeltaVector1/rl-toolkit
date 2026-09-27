import copy
import tomllib
import unittest
from pathlib import Path

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


root = Path(__file__).resolve().parents[2]
diagnostic = root / "configs/trinity-decensor-review-diagnostic"
production = root / "configs/trinity-decensor-experimental-fixed"


def load_config(directory: Path, filename: str) -> dict:
    with (directory / filename).open("rb") as stream:
        return tomllib.load(stream)


def without_diagnostic_identity(kind: str, config: dict) -> dict:
    result = copy.deepcopy(config)
    if kind == "trainer":
        result.pop("output_dir")
        result.pop("max_steps")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        for tag in ("50-step", "batch32", "review-diagnostic"):
            if tag in result["swanlab"]["tags"]:
                result["swanlab"]["tags"].remove(tag)
        result["file_monitor"].pop("filename")
    elif kind == "orchestrator":
        result.pop("output_dir")
        result.pop("max_steps")
        result.pop("batch_size")
        result.pop("max_inflight_rollouts")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        for tag in ("50-step", "batch32", "review-diagnostic"):
            if tag in result["swanlab"]["tags"]:
                result["swanlab"]["tags"].remove(tag)
        result["model"]["lora"].pop("name")
        result["weight_broadcast"].pop("source_url")
        result["weight_broadcast"].pop("source_dir")
        result["train"]["env"][0]["env"]["taskset"]["id"] = "decensor-experimental-v1"
    return result


class ReviewDiagnosticConfigTest(unittest.TestCase):
    def test_profile_is_parseable_and_resolves_as_one_unique_run(self):
        selected = training_ops._config_dir(
            "configs/trinity-decensor-review-diagnostic",
            require_orchestrator=True,
            require_inference=True,
        )
        self.assertEqual(
            training_ops._selected_run_name(selected),
            "trinity-decensor-experimental-fixed-diag-d29a5958",
        )
        self.assertEqual(
            training_ops._config_for_run(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                config_dir="configs/trinity-decensor-review-diagnostic",
            ),
            selected,
        )
        for filename in ("trainer.toml", "orchestrator.toml", "inference.toml"):
            self.assertIsInstance(load_config(diagnostic, filename), dict)

    def test_profile_keeps_training_contract_and_uses_batch32(self):
        diagnostic_trainer = load_config(diagnostic, "trainer.toml")
        diagnostic_orchestrator = load_config(diagnostic, "orchestrator.toml")
        diagnostic_inference = load_config(diagnostic, "inference.toml")
        production_trainer = load_config(production, "trainer.toml")
        production_orchestrator = load_config(production, "orchestrator.toml")
        production_inference = load_config(production, "inference.toml")

        self.assertEqual(
            without_diagnostic_identity("trainer", diagnostic_trainer),
            without_diagnostic_identity("trainer", production_trainer),
        )
        self.assertEqual(
            without_diagnostic_identity("orchestrator", diagnostic_orchestrator),
            without_diagnostic_identity("orchestrator", production_orchestrator),
        )
        self.assertEqual(
            {key: value for key, value in diagnostic_inference.items() if key != "env_vars"},
            {key: value for key, value in production_inference.items() if key != "env_vars"},
        )

        self.assertEqual(diagnostic_trainer["max_steps"], 1)
        self.assertEqual(diagnostic_orchestrator["max_steps"], 1)
        self.assertEqual(diagnostic_orchestrator["batch_size"], 32)
        self.assertEqual(diagnostic_orchestrator["group_size"], 16)
        self.assertEqual(diagnostic_orchestrator["max_inflight_rollouts"], 32)
        self.assertEqual(diagnostic_orchestrator["pool_size"], 4)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["pool"]["num_workers"], 16)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["taskset"]["task"]["judge"]["max_concurrent_requests"], 48)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["taskset"]["task"]["user"]["max_concurrent_requests"], 48)
        self.assertEqual(diagnostic_orchestrator["optim"]["lr"], 8e-6)
        self.assertEqual(diagnostic_trainer["optim"]["lr"], 8e-6)
        self.assertEqual(diagnostic_trainer["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(diagnostic_trainer["model"]["seq_len"], 32768)
        self.assertEqual(diagnostic_orchestrator["seq_len"], 32768)
        self.assertEqual(diagnostic_orchestrator["train"]["sampling"]["max_completion_tokens"], 16384)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["agent"]["max_output_tokens"], 16384)
        self.assertEqual(diagnostic_trainer["model"]["quantization"]["type"], "fp8")
        self.assertFalse(diagnostic_trainer["model"]["quantization"]["dequantize_checkpoint"])

    def test_all_diagnostic_identities_are_unique_and_production_is_untouched(self):
        diagnostic_trainer = load_config(diagnostic, "trainer.toml")
        diagnostic_orchestrator = load_config(diagnostic, "orchestrator.toml")
        diagnostic_inference = load_config(diagnostic, "inference.toml")
        production_trainer = load_config(production, "trainer.toml")
        production_orchestrator = load_config(production, "orchestrator.toml")
        production_inference = load_config(production, "inference.toml")

        self.assertNotEqual(diagnostic_trainer["output_dir"], production_trainer["output_dir"])
        self.assertNotEqual(diagnostic_orchestrator["output_dir"], production_orchestrator["output_dir"])
        self.assertNotEqual(diagnostic_trainer["swanlab"]["name"], production_trainer["swanlab"]["name"])
        self.assertNotEqual(diagnostic_orchestrator["swanlab"]["group"], production_orchestrator["swanlab"]["group"])
        self.assertNotEqual(diagnostic_orchestrator["model"]["lora"]["name"], production_orchestrator["model"]["lora"]["name"])
        self.assertNotEqual(diagnostic_orchestrator["weight_broadcast"]["source_url"], production_orchestrator["weight_broadcast"]["source_url"])
        self.assertNotEqual(diagnostic_inference["env_vars"]["PRIME_LORA_CACHE_DIR"], production_inference["env_vars"]["PRIME_LORA_CACHE_DIR"])
        self.assertEqual(diagnostic_orchestrator["ckpt"]["resume_step"], -1)
        self.assertTrue(diagnostic_orchestrator["ckpt"]["skip_progress"])
        self.assertTrue(diagnostic_orchestrator["output_dir"].endswith("/run_trinity-decensor-experimental-fixed-diag-d29a5958"))
        self.assertEqual(
            diagnostic_orchestrator["weight_broadcast"]["source_dir"],
            diagnostic_trainer["output_dir"],
        )
        self.assertIn("batch32", diagnostic_orchestrator["swanlab"]["tags"])
        self.assertIn("review-diagnostic", diagnostic_orchestrator["swanlab"]["tags"])

        for path in diagnostic.glob("*.toml"):
            self.assertNotIn("n79", path.read_text(encoding="utf-8").lower())
        self.assertEqual(production_trainer["output_dir"], "/data/glm52-rl/runs/trinity-decensor-experimental-fixed")
        self.assertEqual(production_orchestrator["batch_size"], 1024)
        self.assertEqual(production_orchestrator["max_inflight_rollouts"], 256)
        self.assertEqual(production_inference["env_vars"]["PRIME_LORA_CACHE_DIR"], "/data/glm52-rl/lora-cache")


if __name__ == "__main__":
    unittest.main()
