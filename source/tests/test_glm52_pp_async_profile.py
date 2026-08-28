"""Checks for the isolated full-shape asynchronous PP qualification."""

import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "configs/qualification/glm52-native-pp-async.toml"
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


def test_async_profile_is_isolated_and_full_shape() -> None:
    operations = _load("configs/operations.toml")
    assert operations["profiles"]["glm52-native-pp-async"] == {
        "kind": "glm52",
        "trainer": PROFILE,
    }

    config = _load(PROFILE)
    assert config["output_dir"] == "/data/glm52-rl/qualifications/glm52-native-pp-async"
    assert config["max_steps"] == 2
    assert config["data"]["fake"] == {
        "batch_size": 64,
        "generate_samples": True,
        "samples_per_micro_batch": 32,
        "group_size": 16,
    }
    assert config["data"]["fake"]["batch_size"] * config["data"]["fake"]["samples_per_micro_batch"] == 2048

    model = config["model"]
    assert (model["pp"], model["cp"], model["ep"]) == (8, 8, 8)
    assert model["dp_replicate"] == 1
    assert model["global_ep"] is False
    assert model["ep_comm_backend"] == "torch"
    assert model["pp_schedule"] == "1f1b_async"
    assert model["pp_limit_receive_buffers"] is True
    assert model["pp_transfer_sparse_indices"] is True
    assert model["pp_layer_boundaries"] == [0, 8, 18, 28, 38, 48, 58, 68, 78]
    assert model["freeze_moe_router"] is True
    assert model["ac"]["keep_routed_expert_activations_from_pp_stage"] == 5
    assert model["quantization"] == {
        "type": "fp8",
        "dequantize_checkpoint": False,
        "diagnose_linear_backward": False,
        "fused_qkv_a_fp8": True,
        "expert_activation_transport": "bf16_forward_fp8_local_recompute_backward",
        "w8a16_modules": [],
    }
    assert model["optimization_dtype"] == model["reduce_dtype"] == "bfloat16"
    assert model["lora"]["rank"] == 32
    assert set(model["lora"]["target_modules"]) == TARGETS
    assert len(model["lora"]["target_modules"]) == 10
    assert config["optim"]["lr"] == 8e-6

    env_vars = config["env_vars"]
    assert env_vars == {"PRIME_LORA_MM_PROFILE": "sm89-r32"}
    forbidden = {
        "PRIME_NSYS_PROFILE",
        "PRIME_CUDA_PROFILE",
        "PRIME_PHASE_PROFILE",
        "PRIME_EXPERT_PHASE_PROFILE",
        "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD",
        "PRIME_INT4_COMBINE_ROUNDS",
    }
    assert not forbidden & set(env_vars)
    assert model["fsdp_cpu_offload"] is False
    assert model["cold_cpu_offload"] is False
    assert model["optim_cpu_offload"] is False
    assert model["native_fp8_expert_prefetch"] is False
    assert model["ac_offloading"] == "None"


def test_async_profile_is_allowlisted_and_validated_before_setup() -> None:
    cluster = (ROOT / "scripts/china/cluster.sh").read_text()
    train = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    config = (
        ROOT / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
    ).read_text()
    training_ops = (
        ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    ).read_text()

    assert "glm52-native-pp-async" in cluster
    assert "glm52-native-pp-async" in train
    assert '"glm52-native-pp-async": 2' in training_ops
    assert '"glm52-native-pp-async",' in training_ops
    assert "validate_async_1f1b_topology" in config
    assert "1f1b_async is only available in the GLM async qualification profile" in train
    assert train.index("config.model.validate_async_1f1b_topology()") < train.index(
        "setup_torch_distributed("
    )


def test_async_topology_runtime_uses_model_checkpoint_config() -> None:
    """Exercise the schedule argument with the real TrainerConfig shape."""
    train_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
    tree = ast.parse(train_path.read_text())
    expression = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "async_topology":
                expression = keyword.value
                break
        if expression is not None:
            break
    assert expression is not None

    config = SimpleNamespace(
        model=SimpleNamespace(
            pp_schedule="1f1b_async",
            pp_limit_receive_buffers=True,
            ac=SimpleNamespace(keep_routed_expert_activations_from_pp_stage=5),
        )
    )
    parallel_dims = SimpleNamespace(pp=8, cp=8, ep=8)
    topology = eval(
        compile(ast.Expression(expression), str(train_path), "eval"),
        {"config": config, "parallel_dims": parallel_dims},
    )
    assert topology["cutoff"] == 5
    assert topology["pp"] == topology["cp"] == topology["ep"] == 8
