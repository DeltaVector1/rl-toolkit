import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def test_current_configuration_matches_training_expectations() -> None:
    trainer = _load("configs/current/trainer.toml")
    orchestrator = _load("configs/current/orchestrator.toml")

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
    assert "initial_adapter_path" not in trainer["model"]["lora"]
    assert trainer["model"]["ac"] == {"mode": "full", "freq": 1}
    assert trainer["optim"] == {
        "type": "muon",
        "lr": 8e-6,
        "weight_decay": 0.0,
        "max_norm": 1.0,
    }


def test_muse_saves_only_adapters_not_optimizer_or_merged_weights() -> None:
    """Muse persistence is the filesystem LoRA adapter, not trainer state."""
    for relative in (
        "configs/current/trainer.toml",
        "configs/qualification/muse-trainer.toml",
        "configs/qualification/muse-parity.toml",
    ):
        trainer = _load(relative)
        assert trainer["ckpt"]["weights_only"] is True
        assert trainer["ckpt"]["skip_gather_master_weights"] is True


def test_muse_production_fails_closed_on_parity() -> None:
    trainer = _load("configs/current/trainer.toml")
    assert trainer["parity_guard"] == {
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
    }
