import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def test_current_configuration_matches_training_expectations() -> None:
    trainer = _load("configs/china/trainer.muse16.toml")
    orchestrator = _load("configs/h200/orchestrator.muse16.toml")

    assert orchestrator["batch_size"] == 1024
    assert orchestrator["group_size"] == 16
    assert orchestrator["seq_len"] == 65536
    assert orchestrator["max_inflight_rollouts"] == 256
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
        "configs/china/trainer.muse16.toml",
        "configs/h200/orchestrator.muse16.toml",
    ):
        config = _load(relative)
        assert len(config["swanlab"]["tags"]) <= 30
        assert all(len(tag) <= 20 for tag in config["swanlab"]["tags"])


def test_muse_saves_only_adapters_not_optimizer_or_merged_weights() -> None:
    """Muse persistence is the filesystem LoRA adapter, not trainer state."""
    for relative in (
        "configs/china/trainer.muse13.toml",
        "configs/china/trainer.muse16.toml",
        "configs/china/trainer.muse-qualification.toml",
        "configs/china/trainer.muse-direct-parity.toml",
    ):
        trainer = _load(relative)
        assert trainer["ckpt"]["weights_only"] is True
        assert trainer["ckpt"]["skip_gather_master_weights"] is True


def test_muse_production_fails_closed_on_parity() -> None:
    trainer = _load("configs/china/trainer.muse16.toml")
    assert trainer["parity_guard"] == {
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
    }


def test_muse_adapter_resume_reaches_the_trainer_container() -> None:
    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    trainer = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    assert '-e PRIME_RL_RESUME_ADAPTER="${PRIME_RL_RESUME_ADAPTER:-}"' in launcher
    assert '-e PRIME_RL_RESUME_ADAPTER_STEP="${PRIME_RL_RESUME_ADAPTER_STEP:-}"' in launcher
    assert "dist.all_gather_object(loaded_by_rank, loaded)" in trainer
    assert "globally_loaded != adapter_keys" in trainer


def test_muse16_uses_direct_sticky_replicas_and_bounded_rollouts() -> None:
    orchestrator = _load("configs/h200/orchestrator.muse16.toml")
    urls = orchestrator["model"]["client"]["base_url"]

    assert urls == [f"http://127.0.0.1:{port}/v1" for port in range(8001, 8009)]
    assert orchestrator["model"]["client"]["admin_base_url"] == urls
    assert orchestrator["train"]["env"][0]["env"]["agent"]["timeout"]["rollout"] == 1200
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 16384
    assert orchestrator["ckpt"]["resume_step"] == 60


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
