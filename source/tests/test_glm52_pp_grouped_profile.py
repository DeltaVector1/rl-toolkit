"""Local checks for the isolated grouped-P2P b2048 qualification profile."""

import importlib.util
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAME = "glm52-native-pp-grouped"
PROFILE_PATH = "configs/qualification/glm52-native-pp-grouped.toml"
STABILITY_PROFILE_NAME = "glm52-native-pp-grouped-stability20"
STABILITY_PROFILE_PATH = (
    "configs/qualification/glm52-native-pp-grouped-stability20.toml"
)
TARGETS = {
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


def _load(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def _training_ops():
    path = ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    spec = importlib.util.spec_from_file_location("grouped_profile_training_ops", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grouped_profile_matches_plain_b2048_shape_without_cache_or_extra_changes():
    operations = _load("configs/operations.toml")
    profile = operations["profiles"][PROFILE_NAME]
    assert profile == {
        "kind": "glm52",
        "trainer_container": "glm52_qualification_trainer",
        "trainer": PROFILE_PATH,
        "launch_env": {
            "PRIME_PHASE_PROFILE_RECORD_PATH": (
                "/data/glm52-rl/qualifications/glm52-native-pp-grouped/"
                "phase-profile.json"
            )
        },
    }

    config = _load(PROFILE_PATH)
    assert config["output_dir"] == (
        "/data/glm52-rl/qualifications/glm52-native-pp-grouped"
    )
    assert config["max_steps"] == 2
    assert config["data"]["fake"] == {
        "batch_size": 64,
        "generate_samples": True,
        "samples_per_micro_batch": 32,
        "group_size": 16,
    }
    fake = config["data"]["fake"]
    assert fake["batch_size"] * fake["samples_per_micro_batch"] == 2048
    assert "rollout_cache" not in config["data"]

    model = config["model"]
    assert (model["pp"], model["cp"], model["ep"]) == (8, 8, 8)
    assert model["dp_replicate"] == 1
    assert model["global_ep"] is False
    assert model["ep_comm_backend"] == "torch"
    assert model["pp_schedule"] == "1f1b_async"
    assert model["pp_limit_receive_buffers"] is True
    assert model["pp_transfer_sparse_indices"] is True
    assert model["pp_boundary_plan"] == "explicit"
    assert model["pp_layer_boundaries"] == [0, 8, 18, 28, 38, 48, 58, 68, 78]
    assert model["freeze_moe_router"] is True
    assert model.get("debug", {}).get("force_balanced_routing", False) is False
    assert model["lora"]["rank"] == 32
    assert model["lora"]["alpha"] == 64
    assert set(model["lora"]["target_modules"]) == TARGETS
    assert len(model["lora"]["target_modules"]) == 10
    assert "lora_fusion" not in config and "stage_input_transport" not in model
    assert config["env_vars"] == {
        "PRIME_LORA_MM_PROFILE": "sm89-r32",
        "PRIME_PHASE_PROFILE": "1",
        "PRIME_PHASE_PROFILE_STEPS": "2",
        "PRIME_PHASE_PROFILE_SAMPLE_EVERY": "8",
    }
    assert config["optim"]["lr"] == 8e-6


def test_grouped_profile_uses_only_existing_read_only_trainer_mounts():
    training_ops = _training_ops()
    mounts = training_ops._qualification_mounts(PROFILE_NAME)
    assert "source/src/prime_rl/trainer/distributed/async_schedule.py" in mounts
    assert "source/src/prime_rl/trainer/distributed/pp_runner.py" in mounts
    assert "source/src/prime_rl/trainer/phase_profiler.py" in mounts
    assert "source/src/prime_rl/trainer/rl/train.py" in mounts
    assert "source/tests/test_pp_async_cuda.py" not in mounts
    assert training_ops._qualification_launch_env(
        _load("configs/operations.toml")["profiles"][PROFILE_NAME]
    ) == _load("configs/operations.toml")["profiles"][PROFILE_NAME]["launch_env"]

    cluster = (ROOT / "scripts/china/cluster.sh").read_text()
    train = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    assert PROFILE_NAME in cluster
    assert PROFILE_NAME in train
    assert "PRIME_PHASE_PROFILE_RECORD_PATH" in cluster


def test_grouped_schedule_is_not_the_production_default():
    production = _load("configs/paused/trainer.glm52-production.toml")
    assert production["model"]["pp_schedule"] == "1f1b"
    assert _load("configs/qualification/glm52-native-b2048.toml")["model"][
        "pp_schedule"
    ] == "1f1b"


def test_grouped_stability_profile_keeps_the_same_shape_for_twenty_steps():
    operations = _load("configs/operations.toml")
    assert operations["profiles"][STABILITY_PROFILE_NAME] == {
        "kind": "glm52",
        "trainer_container": "glm52_qualification_trainer",
        "trainer": STABILITY_PROFILE_PATH,
    }

    config = _load(STABILITY_PROFILE_PATH)
    assert config["output_dir"] == (
        "/data/glm52-rl/qualifications/glm52-native-pp-grouped-stability20"
    )
    assert config["max_steps"] == 20
    assert config["data"]["fake"] == {
        "batch_size": 64,
        "generate_samples": True,
        "samples_per_micro_batch": 32,
        "group_size": 16,
    }
    assert config["data"]["fake"]["batch_size"] * config["data"]["fake"][
        "samples_per_micro_batch"
    ] == 2048
    model = config["model"]
    assert (model["pp"], model["cp"], model["ep"]) == (8, 8, 8)
    assert model["pp_schedule"] == "1f1b_async"
    assert model["pp_layer_boundaries"] == [0, 8, 18, 28, 38, 48, 58, 68, 78]
    assert model["lora"]["rank"] == 32
    assert set(model["lora"]["target_modules"]) == TARGETS
    assert len(model["lora"]["target_modules"]) == 10
    assert config["env_vars"] == {"PRIME_LORA_MM_PROFILE": "sm89-r32"}
    assert "rollout_cache" not in config["data"]
