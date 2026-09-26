import copy
import tomllib
import unittest
from pathlib import Path

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "configs/trinity-decensor-experimental-parity"
FIXED = ROOT / "configs/trinity-decensor-experimental-fixed"


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _without_identity(kind: str, config: dict) -> dict:
    result = copy.deepcopy(config)
    if kind == "trainer":
        result.pop("output_dir")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        result["file_monitor"].pop("filename")
    elif kind == "orchestrator":
        result.pop("output_dir")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        result["file_monitor"].pop("filename")
        result["model"]["lora"].pop("name")
        result["weight_broadcast"].pop("source_url")
        result["weight_broadcast"].pop("source_dir")
    return result


class FixedConfigTest(unittest.TestCase):
    def test_fixed_profile_matches_parity_except_run_identity(self):
        for filename in ("trainer.toml", "orchestrator.toml", "inference.toml"):
            source = _load(SOURCE / filename)
            fixed = _load(FIXED / filename)
            self.assertEqual(
                _without_identity(filename[:-5], source),
                _without_identity(filename[:-5], fixed),
                filename,
            )

        trainer = _load(FIXED / "trainer.toml")
        orchestrator = _load(FIXED / "orchestrator.toml")
        inference = _load(FIXED / "inference.toml")
        run = "trinity-decensor-experimental-fixed"

        self.assertEqual(training_ops._selected_run_name(
            training_ops._config_dir(f"configs/{run}")
        ), run)
        selected = training_ops._config_for_run(run)
        self.assertEqual(selected["trainer"], f"configs/{run}/trainer.toml")
        self.assertEqual(selected["orchestrator"], f"configs/{run}/orchestrator.toml")
        self.assertEqual(selected["inference"], f"configs/{run}/inference.toml")

        self.assertEqual(trainer["output_dir"], f"/data/glm52-rl/runs/{run}")
        self.assertEqual(
            orchestrator["output_dir"],
            f"/data/glm52-rl/runs/{run}/run_{run}",
        )
        self.assertEqual(
            orchestrator["weight_broadcast"]["source_url"],
            f"http://127.0.0.1:18100/{run}",
        )
        self.assertEqual(
            orchestrator["weight_broadcast"]["source_dir"],
            trainer["output_dir"],
        )
        self.assertEqual(orchestrator["model"]["lora"]["name"], f"{run}-r32-node-local")

        self.assertEqual(trainer["max_steps"], 50)
        self.assertEqual(trainer["model"]["seq_len"], 32768)
        self.assertEqual(trainer["model"]["lora"]["rank"], 32)
        self.assertEqual(trainer["model"]["lora"]["alpha"], 64)
        self.assertEqual(trainer["optim"]["lr"], 8e-6)
        self.assertEqual(trainer["optim"]["max_norm"], 1.0)
        self.assertEqual(
            trainer["parity_guard"],
            {
                "max_unmasked_mismatch_kl": 0.1,
                "max_masked_mismatch_kl": 0.5,
                "max_masked_fraction": 0.06,
            },
        )
        self.assertTrue(trainer["enable_router_replay"])
        self.assertEqual(orchestrator["max_steps"], 50)
        self.assertEqual(orchestrator["seq_len"], 32768)
        self.assertEqual(orchestrator["batch_size"], 1024)
        self.assertEqual(orchestrator["group_size"], 16)
        self.assertEqual(orchestrator["optim"]["lr"], 8e-6)
        self.assertEqual(orchestrator["train"]["env"][0]["name"], "decensor-experimental")
        self.assertEqual(
            orchestrator["train"]["env"][0]["env"]["agent"]["timeout"]["rollout"],
            3600,
        )
        self.assertEqual(inference["model"]["dtype"], "bfloat16")
        self.assertTrue(inference["enable_return_routed_experts"])


if __name__ == "__main__":
    unittest.main()
