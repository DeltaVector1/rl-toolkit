import tomllib
import unittest
from copy import deepcopy
from pathlib import Path


class Packed1f1bZeroLoraConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.config = cls.read_config("trinity-h200-synthetic-packed-1f1b-zero-lora-parity")
        cls.baseline = cls.read_config("trinity-h200-synthetic-packed-1f1b-parity")
        cls.production = cls.read_config("trinity-decensor-experimental-fixed")

    @classmethod
    def read_config(cls, directory):
        return tomllib.loads((cls.root / "configs" / directory / "trainer.toml").read_text())

    def test_only_output_and_lora_change(self):
        config = deepcopy(self.config)
        self.assertNotIn("name", config)
        self.assertEqual(
            config.pop("output_dir"),
            "/data/glm52-rl/qualifications/"
            "trinity-h200-synthetic-packed-1f1b-zero-lora-parity/packed-1f1b-zero-lora-run",
        )
        config["output_dir"] = self.baseline["output_dir"]
        self.assertEqual(config["model"].pop("lora"), self.production["model"]["lora"])
        self.assertEqual(config, self.baseline)

    def test_exact_production_targets_and_scale(self):
        lora = self.config["model"]["lora"]
        self.assertEqual(lora["rank"], 32)
        self.assertEqual(lora["alpha"], 64)
        self.assertEqual(lora["alpha"] / lora["rank"], 2)
        self.assertEqual(lora["dropout"], 0)
        self.assertEqual(lora["target_modules"], [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj", "shared_expert", "experts",
        ])

    def test_fresh_trainer_only_without_update_or_export(self):
        self.assertIsNone(self.config.get("ckpt"))
        self.assertIsNone(self.config["model"]["lora"].get("initial_adapter_path"))
        self.assertFalse(self.config["data"]["fake"].get("export_final_adapter", False))
        self.assertEqual(self.config["max_steps"], 1)
        self.assertEqual(self.config["optim"]["lr"], 0)
        self.assertEqual(self.config["data"]["fake"]["batch_size"], 8)
        self.assertEqual(self.config["model"]["pp_schedule"], "1f1b")

    def test_config_validates_with_trainer_schema_when_available(self):
        try:
            from prime_rl.configs.trainer import TrainerConfig
        except ModuleNotFoundError as error:
            if error.name not in {"prime_rl", "prime_rl.configs", "prime_rl.configs.trainer"}:
                raise
            self.skipTest("TrainerConfig schema is not installed in this local environment")

        parsed = TrainerConfig.model_validate(self.config)
        baseline = TrainerConfig.model_validate(self.baseline)
        self.assertIsNone(parsed.ckpt)
        self.assertIsNone(parsed.model.lora.initial_adapter_path)
        self.assertEqual(parsed.model.lora.modules_to_save, [])
        self.assertFalse(parsed.model.lora.freeze_a)
        self.assertFalse(parsed.data.fake.export_final_adapter)
        self.assertEqual(parsed.model.lora.alpha / parsed.model.lora.rank, 2)
        self.assertEqual(parsed.output_dir.name, "packed-1f1b-zero-lora-run")
        self.assertEqual(
            parsed.model_dump(exclude={"output_dir": True, "model": {"lora"}}),
            baseline.model_dump(exclude={"output_dir": True, "model": {"lora"}}),
        )


if __name__ == "__main__":
    unittest.main()
