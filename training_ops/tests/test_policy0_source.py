import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import training_ops
from training_ops.runtime.policy0_source import (
    CANONICAL_TARGETS,
    Policy0SourceError,
    build_apply_script,
    build_plan,
    parse_result,
)


def fixed_paths():
    source = "/data/glm52-rl/runs/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed/broadcasts/step_0"
    destination = "/data/glm52-rl/runs/trinity-decensor-experimental-fixed/policy0-qualification/step_0"
    return source, destination


def legacy_config():
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "trinity",
        "r": 32,
        "lora_alpha": 64.0,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "experts"],
        "modules_to_save": None,
    }


def local_plan(tmp_path):
    root = tmp_path / "runs"
    source = root / "preserved" / "broadcasts" / "step_0"
    destination = root / "policy0-qualification" / "step_0"
    source.mkdir(parents=True)
    (source / "STABLE").write_text("")
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    config_bytes = json.dumps(legacy_config(), indent=2).encode()
    (source / "adapter_config.json").write_bytes(config_bytes)
    weight_bytes = (source / "adapter_model.safetensors").read_bytes()
    weight_sha = hashlib.sha256(weight_bytes).hexdigest()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    (source / "adapter_model.safetensors.sha256").write_text(weight_sha + "\n")
    (source / "adapter_manifest.json").write_text(json.dumps({
        "version": 1,
        "kind": "lora-adapter",
        "mode": "full",
        "target_step": 0,
        "base_step": None,
        "full_size": len(weight_bytes),
        "full_sha256": weight_sha,
        "target_size": len(weight_bytes),
        "target_sha256": weight_sha,
        "config_size": len(config_bytes),
        "config_sha256": config_sha,
        "encoded_size": len(weight_bytes),
        "chunk_size": 0,
        "chunk_count": 0,
        "files": [
            {"name": "adapter_config.json", "size": len(config_bytes), "sha256": config_sha},
            {"name": "adapter_model.safetensors", "size": len(weight_bytes), "sha256": weight_sha},
        ],
    }, separators=(",", ":")) + "\n")
    plan = {
        "source_dir": str(source),
        "source_adapter": str(source / "adapter_model.safetensors"),
        "source_config": str(source / "adapter_config.json"),
        "destination_dir": str(destination),
        "configured_source_dir": str(root),
        "destination_url": "http://source/policy0-qualification/step_0",
    }
    return plan, source, destination


def add_transport_chunk(source):
    chunk = b"chunked"
    chunk_dir = source / "transport-chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk-0").write_bytes(chunk)
    manifest_path = source / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    weight = source / "adapter_model.safetensors"
    chunk_sha = hashlib.sha256(chunk).hexdigest()
    manifest.update({
        "transport": "chunked-full-zstd",
        "encoded_size": len(chunk),
        "chunk_size": len(chunk),
        "chunk_count": 1,
        "chunks": [{
            "file": "chunk-0", "offset": 0, "raw_size": weight.stat().st_size,
            "wire_size": len(chunk), "wire_sha256": chunk_sha,
            "raw_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(), "codec": "zstd",
        }],
    })
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")


def run_script(script):
    return subprocess.run(["bash", "-ceu", script], text=True, capture_output=True)


def test_plan_has_no_live_call_and_derives_destination_url():
    source, destination = fixed_paths()
    with patch("training_ops._ssh") as ssh:
        plan = training_ops._prepare_policy0_source_action([
            "configs/trinity-decensor-experimental-fixed", "plan",
            "--source-adapter-dir", source,
            "--destination-adapter-dir", destination,
        ])
    ssh.assert_not_called()
    assert plan["trainer_head"] == "n82"
    assert plan["destination_url"].endswith("/policy0-qualification/step_0")
    assert "--reflink=auto" in plan["apply_script"]
    assert "SOURCE_RESULT" in plan["apply_script"]


def test_apply_calls_only_configured_trainer_head_and_returns_identities():
    source, destination = fixed_paths()
    output = "SOURCE_RESULT created " + "a" * 64 + " 10 " + "b" * 64 + " 20\n"
    with patch("training_ops._ssh", return_value=(0, output)) as ssh:
        result = training_ops._prepare_policy0_source_action([
            "configs/trinity-decensor-experimental-fixed", "apply",
            "--source-adapter-dir", source,
            "--destination-adapter-dir", destination,
        ])
    assert ssh.call_args.args[0] == "n82"
    assert result["adapter_identity"]["size"] == 10
    assert "docker start" not in ssh.call_args.args[1]


def test_plan_refuses_destination_outside_or_non_qualification_path():
    source, _ = fixed_paths()
    with pytest.raises(RuntimeError, match="qualification-only"):
        training_ops._prepare_policy0_source_action([
            "configs/trinity-decensor-experimental-fixed", "plan",
            "--source-adapter-dir", source,
            "--destination-adapter-dir", "/data/glm52-rl/runs/trinity-decensor-experimental-fixed/broadcasts/copy/step_0",
        ])
    with pytest.raises(RuntimeError, match="below weight_broadcast"):
        training_ops._prepare_policy0_source_action([
            "configs/trinity-decensor-experimental-fixed", "plan",
            "--source-adapter-dir", source,
            "--destination-adapter-dir", "/data/other/qualification/step_0",
        ])


def test_script_accepts_legacy_source_preserves_it_and_publishes_canonical(tmp_path):
    plan, source, destination = local_plan(tmp_path)
    result = run_script(build_apply_script(plan))
    assert result.returncode == 0, result.stderr
    fields = result.stdout.strip().split()
    assert fields[0:2] == ["SOURCE_RESULT", "created"]
    assert (source / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert (source / "adapter_config.json").read_text() == json.dumps(legacy_config(), indent=2)
    assert (destination / "STABLE").is_file()
    corrected = json.loads((destination / "adapter_config.json").read_text())
    assert corrected["lora_alpha"] == 64
    assert type(corrected["lora_alpha"]) is int
    assert corrected["target_modules"] == list(CANONICAL_TARGETS)
    assert not list(destination.parent.glob(".policy0-source.*"))
    expected_sha = hashlib.sha256((destination / "adapter_config.json").read_bytes()).hexdigest()
    assert fields[4] == expected_sha


def test_script_copies_and_validates_referenced_transport_chunks(tmp_path):
    plan, source, destination = local_plan(tmp_path)
    add_transport_chunk(source)
    result = run_script(build_apply_script(plan))
    assert result.returncode == 0, result.stderr
    assert (destination / "transport-chunks" / "chunk-0").read_bytes() == b"chunked"
    manifest = json.loads((destination / "adapter_manifest.json").read_text())
    assert manifest["target_url"] == plan["destination_url"]
    assert manifest["config_sha256"] == hashlib.sha256(
        (destination / "adapter_config.json").read_bytes()
    ).hexdigest()
    assert manifest["chunks"][0]["wire_sha256"] == hashlib.sha256(b"chunked").hexdigest()


def test_script_is_idempotent_and_refuses_conflicting_destination(tmp_path):
    plan, _, destination = local_plan(tmp_path)
    first = run_script(build_apply_script(plan))
    assert first.returncode == 0
    second = run_script(build_apply_script(plan))
    assert second.returncode == 0
    assert second.stdout.split()[1] == "idempotent"
    (destination / "adapter_model.safetensors").write_bytes(b"conflict")
    conflict = run_script(build_apply_script(plan))
    assert conflict.returncode != 0
    assert "SOURCE_RESULT" not in conflict.stdout


def test_script_refuses_unexpected_legacy_config(tmp_path):
    plan, source, _ = local_plan(tmp_path)
    value = legacy_config()
    value["unexpected"] = True
    (source / "adapter_config.json").write_text(json.dumps(value))
    result = run_script(build_apply_script(plan))
    assert result.returncode != 0
    assert "unexpected fields" in result.stderr


@pytest.mark.parametrize(("field", "value", "message"), [
    ("r", 16, "rank is not 32"),
    ("lora_dropout", 0.1, "dropout is not zero"),
])
def test_script_refuses_rank_or_dropout_mismatch(tmp_path, field, value, message):
    plan, source, _ = local_plan(tmp_path)
    config = legacy_config()
    config[field] = value
    (source / "adapter_config.json").write_text(json.dumps(config))
    result = run_script(build_apply_script(plan))
    assert result.returncode != 0
    assert message in result.stderr


def test_script_has_source_change_and_no_replace_race_guards(tmp_path):
    plan, _, _ = local_plan(tmp_path)
    script = build_apply_script(plan)
    assert "source changed while creating corrected adapter" in script
    assert "renameat2" in script
    assert "RENAME_NOREPLACE" in script
    assert "destination appeared during atomic publish" in script
    assert "mv --" not in script


def test_parse_result_requires_full_identities():
    plan = {"source_url": "http://source", "destination_url": "http://destination"}
    adapter = "a" * 64
    config = "b" * 64
    result = parse_result(f"SOURCE_RESULT created {adapter} 10 {config} 20\n", plan)
    assert result["config_identity"]["size"] == 20
    with pytest.raises(Policy0SourceError):
        parse_result("SOURCE_RESULT created bad 10 bad 20\n", plan)
