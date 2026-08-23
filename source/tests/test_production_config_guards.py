import tomllib
import ast
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def test_fake_data_qualifications_do_not_enable_router_replay() -> None:
    for path in (ROOT / "configs/qualification").glob("*.toml"):
        config = _load(str(path.relative_to(ROOT)))
        if "fake" in config.get("data", {}):
            assert config.get("enable_router_replay", False) is False, path


def test_current_glm52_profile_is_batch_1024_without_rollout_cache() -> None:
    operations = _load("configs/operations.toml")
    assert set(operations["profiles"]) == {
        "current",
        "glm52-native-small",
        "glm52-policy-parity",
        "glm52-native-b2048",
        "glm52-native-stability20",
    }
    current = operations["profiles"]["current"]
    assert current == {
        "kind": "glm52",
        "trainer": "configs/paused/trainer.glm52-production.toml",
        "orchestrator": "configs/paused/orchestrator.glm52-production.toml",
        "environment": "configs/paused/env.glm52-production.toml",
    }
    trainer = _load(current["trainer"])
    orchestrator = _load(current["orchestrator"])
    assert orchestrator["batch_size"] == 1024
    assert "rollout_cache" not in trainer.get("data", {})

    for name, profile in operations["profiles"].items():
        profile_trainer = _load(profile["trainer"])
        assert "rollout_cache" not in profile_trainer.get("data", {}), name
        if "orchestrator" in profile:
            assert _load(profile["orchestrator"])["batch_size"] == 1024, name

    large_batch = operations["profiles"]["glm52-native-b2048"]
    assert "orchestrator" not in large_batch
    assert _load(large_batch["trainer"])["data"]["fake"]["batch_size"] == 64
    assert operations["requirements"]["glm52"] == {
        "batch_size": 1024,
        "context_length": 32768,
        "trainer_seconds_min": 600,
        "trainer_seconds_max": 900,
        "rollout_seconds_min": 480,
        "rollout_seconds_max": 720,
    }


def test_glm52_launchers_cannot_use_live_muse_container_names() -> None:
    cluster = (ROOT / "scripts/china/cluster.sh").read_text()
    watchdog = (ROOT / "scripts/h200/trainer_failure_watchdog.sh").read_text()
    edge = (ROOT / "scripts/h200/start_orchestrator.sh").read_text()

    assert "deployment-glm52-optims" in cluster.splitlines()[4]
    assert "glm52-optims-trainer" in cluster.splitlines()[5]
    assert "glm52-decensor-trainer" not in watchdog
    assert "glm52-decensor-orchestrator" not in watchdog
    assert 'prime_trainer_container="$trainer_container"' in edge
    assert 'prime_orchestrator_container="$orchestrator_container"' in edge


def test_current_configuration_matches_training_expectations() -> None:
    trainer = _load("configs/china/trainer.muse13.toml")
    orchestrator = _load("configs/h200/orchestrator.muse13.toml")

    assert orchestrator["batch_size"] == 1024
    assert orchestrator["group_size"] == 16
    assert orchestrator["seq_len"] == 65536
    assert orchestrator["max_inflight_rollouts"] == 128
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 16384
    assert orchestrator["optim"]["lr"] == 8e-6
    assert trainer["model"]["seq_len"] == 65536
    assert trainer["rollout_transport"]["type"] == "filesystem"
    assert trainer["model"]["optimization_dtype"] == "bfloat16"
    assert trainer["model"]["lora"]["rank"] == 128
    assert trainer["model"]["ac"] == {"mode": "full", "freq": 1}
    assert trainer["optim"] == {
        "type": "muon",
        "lr": 8e-6,
        "weight_decay": 0.0,
        "max_norm": 1.0,
    }


def test_current_swanlab_tags_fit_service_limit() -> None:
    for relative in (
        "configs/china/trainer.muse13.toml",
        "configs/h200/orchestrator.muse13.toml",
    ):
        config = _load(relative)
        assert len(config["swanlab"]["tags"]) <= 30
        assert all(len(tag) <= 20 for tag in config["swanlab"]["tags"])


def test_muse_saves_only_adapters_not_optimizer_or_merged_weights() -> None:
    """Muse persistence is the filesystem LoRA adapter, not trainer state."""
    for relative in (
        "configs/china/trainer.muse13.toml",
        "configs/china/trainer.muse-qualification.toml",
        "configs/china/trainer.muse-direct-parity.toml",
    ):
        trainer = _load(relative)
        assert trainer["ckpt"]["weights_only"] is True
        assert trainer["ckpt"]["skip_gather_master_weights"] is True


def test_muse_production_fails_closed_on_parity() -> None:
    trainer = _load("configs/china/trainer.muse13.toml")
    assert trainer["parity_guard"] == {
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
    }


def test_muse13_uses_direct_sticky_replicas_and_bounded_rollouts() -> None:
    orchestrator = _load("configs/h200/orchestrator.muse13.toml")
    urls = orchestrator["model"]["client"]["base_url"]

    assert urls == [f"http://127.0.0.1:{port}/v1" for port in range(8001, 8009)]
    assert orchestrator["model"]["client"]["admin_base_url"] == urls
    assert orchestrator["train"]["env"][0]["env"]["agent"]["timeout"]["rollout"] == 1200
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 16384


def test_production_allocator_does_not_strand_large_reserved_blocks() -> None:
    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    trainer_case = launcher.split("case ${1:-} in", 1)[1].split("  orchestrator)", 1)[0]
    allocator = next(
        line for line in trainer_case.splitlines() if "PYTORCH_CUDA_ALLOC_CONF" in line
    )
    assert "expandable_segments:True" in allocator
    assert "max_split_size_mb" not in allocator


def test_pipeline_loader_uses_mesh_data_parallel_rank() -> None:
    train = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    data = (ROOT / "source/src/prime_rl/trainer/rl/data.py").read_text()

    assert train.count('parallel_dims.get_mesh("dp").get_local_rank()') >= 2
    assert "world.rank // non_dp_world_size" not in data
    assert "verify_pipeline_batch_identity" in train


def test_glm52_production_profile_matches_requirements() -> None:
    trainer = _load("configs/paused/trainer.glm52-production.toml")
    orchestrator = _load("configs/paused/orchestrator.glm52-production.toml")
    environment = _load("configs/paused/env.glm52-production.toml")
    inference = _load("configs/h200/inference.toml")
    operations = _load("configs/operations.toml")
    run = "glm52-1"
    run_dir = f"/data/glm52-rl/runs/{run}"
    run_name = f"run_{run}"

    assert operations["profiles"]["current"] == {
        "kind": "glm52",
        "trainer": "configs/paused/trainer.glm52-production.toml",
        "orchestrator": "configs/paused/orchestrator.glm52-production.toml",
        "environment": "configs/paused/env.glm52-production.toml",
    }
    assert orchestrator["batch_size"] == 1024
    assert orchestrator["group_size"] == 16
    assert orchestrator["seq_len"] == 32768
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 16384
    assert orchestrator["train"]["source"][0]["env"]["assistant"]["max_output_tokens"] == 16384
    assert environment["env"]["assistant"]["max_output_tokens"] == 16384
    assert environment["env"]["taskset"]["dataset_name"] == orchestrator["train"]["source"][0]["env"]["taskset"]["dataset_name"]
    assert trainer["model"]["seq_len"] == 32768
    assert trainer["model"]["pp"] == 8
    assert trainer["model"]["cp"] == 8
    assert trainer["model"]["ep"] == 8
    assert trainer["model"]["cp_style"] == "ring"
    assert trainer["model"]["quantization"]["type"] == "fp8"
    assert trainer["model"]["quantization"]["dequantize_checkpoint"] is False
    assert trainer["optim"] == {
        "type": "muon",
        "lr": 8e-6,
        "weight_decay": 0.0,
        "max_norm": 1.0,
    }
    assert orchestrator["optim"]["lr"] == 8e-6
    assert trainer["enable_router_replay"] is True
    assert trainer["model"]["freeze_moe_router"] is True
    assert inference["enable_return_routed_experts"] is True
    assert "rollout_cache" not in trainer.get("data", {})
    assert trainer["max_steps"] >= operations["limits"]["final_steps"]
    assert trainer["output_dir"] == run_dir
    assert orchestrator["output_dir"] == f"{run_dir}/{run_name}"
    assert environment["output_dir"] == f"{run_dir}/{run_name}/env"
    assert orchestrator["weight_broadcast"]["source_dir"] == run_dir
    assert orchestrator["weight_broadcast"]["source_url"] == (
        f"http://127.0.0.1:18100/{run}"
    )

    targets = {
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
    assert set(trainer["model"]["lora"]["target_modules"]) == targets
    assert set(inference["lora_target_modules"]) == targets
    assert trainer["model"]["lora"]["rank"] == 32
    assert trainer["model"]["lora"]["alpha"] == 64
    assert trainer["model"]["lora"]["dropout"] == 0.0
    assert inference["max_lora_rank"] == 32
    assert inference["model"]["max_model_len"] == 32768
    assert operations["models"]["glm52"] == inference["model"]["name"]
    assert operations["containers"]["inference_image"] == "prime-rl:glm52-v080-full"
    assert operations["containers"]["glm52_model_volume"] == "glm52-fp8-model"
    assert inference["parallel"] == {"tp": 8, "dp": 1}
    assert operations["inference"]["h200a"]["glm52"] == {
        "gpu_count": 8,
        "tensor_parallel_size": 8,
        "ports": [8000],
    }


def test_glm52_production_rejects_bad_parity_before_update() -> None:
    trainer = _load("configs/paused/trainer.glm52-production.toml")
    assert trainer["parity_guard"] == {
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
    }

    source_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
    source = source_path.read_text()
    guard_position = source.index("violations = parity_guard_failures")
    update_position = source.index("optimizer.step()", guard_position)
    broadcast_position = source.index("broadcast_weights", update_position)
    assert guard_position < update_position < broadcast_position

    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "parity_guard_failures"
    )
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    check = namespace["parity_guard_failures"]
    guard = type(
        "Guard",
        (),
        {
            "max_unmasked_mismatch_kl": 0.1,
            "max_masked_mismatch_kl": 0.5,
            "max_masked_fraction": 0.06,
        },
    )()

    assert check(
        {
            "unmasked_mismatch_kl/mean": 0.1,
            "masked_mismatch_kl/mean": 0.5,
            "is_masked/mean": 0.06,
        },
        guard,
    ) == []
    assert check({}, guard)
    assert check(
        {
            "unmasked_mismatch_kl/mean": math.nan,
            "masked_mismatch_kl/mean": 0.1,
            "is_masked/mean": 0.01,
        },
        guard,
    )
    assert check(
        {
            "unmasked_mismatch_kl/mean": 0.100001,
            "masked_mismatch_kl/mean": 0.500001,
            "is_masked/mean": 0.060001,
        },
        guard,
    )


def test_glm52_sparse_indexer_is_always_frozen() -> None:
    model_source = (ROOT / "source/src/prime_rl/trainer/model.py").read_text()
    setup_start = model_source.index("def setup_model")
    setup_source = model_source[setup_start:]
    assert "freeze_sparse_indexer(model)" in setup_source


def test_glm52_production_acceptance_reads_native_fp8() -> None:
    skill_source = ROOT / ".agents/skills/training-ops/src"
    sys.path.insert(0, str(skill_source))
    try:
        import training_ops

        report = training_ops._acceptance_from_text(
            "glm52-1",
            (ROOT / "configs/paused/trainer.glm52-production.toml").read_text(),
            (ROOT / "configs/paused/orchestrator.glm52-production.toml").read_text(),
            "",
            "",
            "",
            "inside the noise",
        )
    finally:
        sys.path.remove(str(skill_source))

    assert report.native_fp8 is True
    assert report.precision_ok is True
    assert not any("native FP8" in alarm for alarm in report.alarms)
