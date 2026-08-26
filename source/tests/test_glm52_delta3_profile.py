"""Consistency checks for the strict three-step adapter qualification."""

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_delta3_profile_is_trainer_only_and_allowlisted() -> None:
    with (ROOT / "configs/operations.toml").open("rb") as stream:
        operations = tomllib.load(stream)
    profile = operations["profiles"]["glm52-native-delta3"]
    assert profile == {
        "kind": "glm52",
        "trainer": "configs/qualification/glm52-native-delta3.toml",
    }

    with (ROOT / profile["trainer"]).open("rb") as stream:
        config = tomllib.load(stream)
    assert config["max_steps"] == 3
    assert "ckpt" not in config
    assert "rollout_cache" not in config["data"]
    assert config["data"]["fake"]["adapter_export_steps"] == [1, 2, 3]
    assert config["data"]["fake"]["export_final_adapter"] is True
    assert (
        config["data"]["fake"]["batch_size"]
        * config["data"]["fake"]["samples_per_micro_batch"]
        == 2048
    )
    assert config["model"]["seq_len"] == 32768
    assert (
        config["model"]["pp"]
        == config["model"]["cp"]
        == config["model"]["ep"]
        == 8
    )
    assert config["model"]["quantization"]["type"] == "fp8"
    assert config["model"]["lora"]["rank"] == 32
    assert config["optim"]["lr"] == 8e-6

    cluster = (ROOT / "scripts/china/cluster.sh").read_text()
    trainer = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    training_ops = (
        ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    ).read_text()
    assert (
        "glm52-native-b2048|glm52-native-delta3|glm52-native-stability20"
        in cluster
    )
    assert '"glm52-native-delta3": (3, True)' in trainer
    assert '"glm52-native-delta3": 3' in training_ops
    assert '"glm52-native-delta3",' in training_ops
