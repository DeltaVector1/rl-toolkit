"""Consistency checks for the strict three-step adapter qualification."""

import importlib.util
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_delta3_profile_is_trainer_only_and_allowlisted() -> None:
    with (ROOT / "configs/operations.toml").open("rb") as stream:
        operations = tomllib.load(stream)
    profile = operations["profiles"]["glm52-native-delta3"]
    assert profile == {
        "kind": "glm52",
        "trainer_container": "glm52_qualification_trainer",
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
        "glm52-native-b2048|glm52-native-b2048-memory|glm52-native-b2048-nsys|glm52-native-interleaved-rank-balanced|glm52-native-delta3|glm52-native-stability20"
        in cluster
    )
    assert '"glm52-native-delta3": (3, True)' in trainer
    assert '"glm52-native-b2048-memory": (2, False)' in trainer
    assert '"glm52-native-delta3": 3' in training_ops
    assert '"glm52-native-delta3",' in training_ops


def test_b2048_memory_profile_has_strict_rank_zero_step_one_capture() -> None:
    with (ROOT / "configs/operations.toml").open("rb") as stream:
        operations = tomllib.load(stream)
    profile = operations["profiles"]["glm52-native-b2048-memory"]
    capture = {
        "PRIME_CUDA_MEMORY_CAPTURE": "1",
        "PRIME_CUDA_MEMORY_CAPTURE_RANK": "0",
        "PRIME_CUDA_MEMORY_CAPTURE_STEP": "1",
        "PRIME_CUDA_MEMORY_CAPTURE_PATH": (
            "/data/glm52-rl/qualifications/glm52-native-b2048-memory/"
            "cuda-memory-capture.json"
        ),
        "PRIME_BF16_COMBINE_ROUNDS": "8",
    }
    assert profile == {
        "kind": "glm52",
        "trainer_container": "glm52_qualification_trainer",
        "trainer": "configs/qualification/glm52-native-b2048-memory.toml",
        "launch_env": capture,
    }
    with (ROOT / profile["trainer"]).open("rb") as stream:
        config = tomllib.load(stream)
    assert config["output_dir"] == capture["PRIME_CUDA_MEMORY_CAPTURE_PATH"].removesuffix(
        "/cuda-memory-capture.json"
    )
    assert config["data"]["fake"]["batch_size"] * config["data"]["fake"]["samples_per_micro_batch"] == 2048

    module_path = ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    spec = importlib.util.spec_from_file_location("training_ops_memory_allowlist", module_path)
    assert spec is not None and spec.loader is not None
    training_ops = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training_ops
    spec.loader.exec_module(training_ops)
    assert training_ops._qualification_launch_env(profile) == capture
    assert capture["PRIME_BF16_COMBINE_ROUNDS"] == "8"
    for name, other in operations["profiles"].items():
        if name != "glm52-native-b2048-memory":
            assert "PRIME_BF16_COMBINE_ROUNDS" not in other.get("launch_env", {})
    gate = "source/tests/test_final_pp16_cuda.py"
    assert gate not in training_ops.mounted_files()
    assert gate not in training_ops._qualification_source_paths("glm52-native-b2048")
    assert gate in training_ops._qualification_source_paths(
        "glm52-native-b2048-memory"
    )
    assert not training_ops._mounted_on(gate, "n82")
    assert training_ops._mounted_on(
        gate, "n82", profile_name="glm52-native-b2048-memory"
    )
    assert not training_ops._mounted_on(
        gate, "h200a", profile_name="glm52-native-b2048-memory"
    )
    for key, bad in (
        ("PRIME_CUDA_MEMORY_CAPTURE_RANK", "1"),
        ("PRIME_CUDA_MEMORY_CAPTURE_STEP", "2"),
        ("PRIME_CUDA_MEMORY_CAPTURE_PATH", "/tmp/capture.json"),
    ):
        with_bad = {**capture, key: bad}
        try:
            training_ops._qualification_launch_env({"launch_env": with_bad})
        except training_ops.Refused:
            pass
        else:
            raise AssertionError(f"invalid memory capture value accepted: {key}")
