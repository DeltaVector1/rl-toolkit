import copy
import hashlib
import json
import tomllib
import unittest
from pathlib import Path


root = Path(__file__).resolve().parents[2]
config_dir = root / "configs/trinity-decensor-experimental-parity"
expected_baselines = {
    "trainer.toml": "39c81f83e63310b051d1ff85f5bd7b733c09ce53d858aba184c41f2a4fea433a",
    "orchestrator.toml": "f840d4021af76106a4ea5d2bf8815bc35d2d66a6791948bace4acd6def52d388",
}


def load_config(filename: str) -> dict:
    with (config_dir / filename).open("rb") as stream:
        return tomllib.load(stream)


def semantic_fingerprint(config: dict) -> str:
    baseline = copy.deepcopy(config)
    baseline.pop("max_steps")
    encoded = json.dumps(
        baseline,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class TrinityExperimentalParityStep244ConfigTest(unittest.TestCase):
    def test_both_configs_parse_and_share_step_244(self):
        trainer = load_config("trainer.toml")
        orchestrator = load_config("orchestrator.toml")

        self.assertEqual(trainer["max_steps"], 244)
        self.assertEqual(orchestrator["max_steps"], 244)
        self.assertEqual(trainer["max_steps"], orchestrator["max_steps"])

    def test_only_max_steps_differs_from_the_step_50_baseline(self):
        for filename, expected in expected_baselines.items():
            with self.subTest(filename=filename):
                self.assertEqual(semantic_fingerprint(load_config(filename)), expected)

    def test_run_identity_and_fresh_rollout_contract_are_unchanged(self):
        trainer = load_config("trainer.toml")
        orchestrator = load_config("orchestrator.toml")

        self.assertEqual(
            trainer["output_dir"],
            "/data/glm52-rl/runs/trinity-decensor-experimental-parity",
        )
        self.assertEqual(
            orchestrator["output_dir"],
            "/data/glm52-rl/runs/trinity-decensor-experimental-parity/"
            "run_trinity-decensor-experimental-parity",
        )
        self.assertEqual(
            orchestrator["model"]["lora"]["name"],
            "trinity-decensor-experimental-parity-r32-node-local",
        )
        self.assertEqual(
            orchestrator["weight_broadcast"],
            {
                "type": "filesystem",
                "source_url": "http://127.0.0.1:18100/"
                "trinity-decensor-experimental-parity",
                "source_dir": "/data/glm52-rl/runs/"
                "trinity-decensor-experimental-parity",
            },
        )
        self.assertEqual(trainer["rollout_transport"], {"type": "filesystem"})
        self.assertEqual(orchestrator["rollout_transport"], {"type": "filesystem"})
        self.assertEqual(
            orchestrator["ckpt"],
            {
                "interval": 5,
                "resume_step": -1,
                "skip_progress": True,
                "keep_last": 2,
                "keep_interval": 100,
            },
        )
        self.assertNotIn("initial_adapter_path", trainer["model"]["lora"])
        self.assertNotIn("initial_adapter_path", orchestrator["model"]["lora"])
        source = "\n".join(
            (config_dir / filename).read_text(encoding="utf-8")
            for filename in ("trainer.toml", "orchestrator.toml")
        )
        for marker in ("resume_adapter", "cached_rollout", "rollout_batch", "train_rollouts.bin"):
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
