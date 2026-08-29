"""Focused guards for the dedicated GLM MTP production profile."""

import importlib.util
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mtp20_profile_is_strict_and_uses_the_production_run_name():
    operations = tomllib.loads((ROOT / "configs/operations.toml").read_text())
    profile = operations["profiles"]["glm52-mtp20"]
    assert profile["trainer"] == "configs/paused/trainer.glm52-mtp20.toml"
    assert profile["orchestrator"] == "configs/paused/orchestrator.glm52-mtp20.toml"
    assert profile["environment"] == "configs/paused/env.glm52-mtp20.toml"
    identity = profile["identity"]
    assert identity["mtp_method"] == "mtp"
    assert identity["mtp_tokens"] == 1
    assert identity["rollout_cache"] is False
    assert identity["inference_container"] == "glm52-optims-mtp20-h200"

    trainer = tomllib.loads((ROOT / profile["trainer"]).read_text())
    orchestrator = tomllib.loads((ROOT / profile["orchestrator"]).read_text())
    assert orchestrator["output_dir"].endswith("/runs/glm52-mtp20/run_glm52-mtp20")
    assert orchestrator["batch_size"] == 2048
    assert orchestrator["group_size"] == 16
    assert orchestrator["seq_len"] == 32768
    assert trainer["model"]["lora"]["rank"] == 32
    assert len(trainer["model"]["lora"]["target_modules"]) == 10


def test_bad_mtp_identity_is_refused():
    operations = tomllib.loads((ROOT / "configs/operations.toml").read_text())
    profile = operations["profiles"]["glm52-mtp20"]
    trainer = tomllib.loads((ROOT / profile["trainer"]).read_text())
    orchestrator = tomllib.loads((ROOT / profile["orchestrator"]).read_text())
    environment = tomllib.loads((ROOT / profile["environment"]).read_text())
    skill_source = ROOT / ".agents/skills/training-ops/src"
    sys.path.insert(0, str(skill_source))
    try:
        import training_ops

        broken = dict(profile)
        broken["identity"] = dict(profile["identity"])
        broken["identity"]["mtp_method"] = "disabled"
        failures = training_ops._production20_identity_failures(
            "glm52-mtp20", broken, trainer, orchestrator, environment
        )
    finally:
        sys.path.remove(str(skill_source))
    assert "MTP must be explicitly enabled for the MTP production profile" in failures


def test_production_launcher_has_the_qualified_mount_and_identity_contract():
    launcher_path = ROOT / "scripts/h200/start_mtp_production.sh"
    launcher = launcher_path.read_text()
    assert "inference.glm52-mtp20.toml" in launcher
    assert "--restart=unless-stopped" in launcher
    assert "sha256:2f803571329212bfb4e27982efec7559f20379bcde73ac6ccee6bb8be3c46ade" in launcher
    assert "-v \"$model_volume\":/models/glm52-fp8:ro" in launcher
    for name in (
        "PRIME_LORA_BASE_MODEL_REPO_ID",
        "PRIME_LORA_BASE_MODEL_REVISION",
        "PRIME_LORA_BASE_MODEL_VARIANT",
        "PRIME_LORA_BASE_MODEL_CONTENT_SHA256",
        "PRIME_LORA_BASE_MODEL_MANIFEST_SHA256",
        "PRIME_LORA_BASE_MODEL_TRAINER_PATH",
        "PRIME_LORA_BASE_MODEL_H200_ALIAS",
    ):
        assert name in launcher
    assert "PRIME_GLM_ADAPTER_VALIDATE=1" in launcher
    assert "PRIME_LORA_CACHE_DIR" in launcher
    assert "PRIME_ADAPTER_PORTS" in launcher
    assert "PRIME_GLM_ADAPTER_CODEC" in launcher
    assert "mtp-production-container:" in launcher
    overlay = _load(
        ROOT / "scripts/h200/flashmla_overlay_preflight.py", "mtp_production_overlay"
    )
    assert len(overlay.RUNTIME_MOUNTS) == 12
    for name, target in overlay.RUNTIME_MOUNTS.items():
        assert (
            f"source=$overlay_root/{overlay.FILES[name]},target={target},readonly"
            in launcher
        )
