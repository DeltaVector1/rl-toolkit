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
    (source / "adapter_config.json").write_text(json.dumps(legacy_config(), indent=2))
    plan = {
        "source_dir": str(source),
        "source_adapter": str(source / "adapter_model.safetensors"),
        "source_config": str(source / "adapter_config.json"),
        "destination_dir": str(destination),
        "configured_source_dir": str(root),
    }
    return plan, source, destination


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


def test_parse_result_requires_full_identities():
    plan = {"source_url": "http://source", "destination_url": "http://destination"}
    adapter = "a" * 64
    config = "b" * 64
    result = parse_result(f"SOURCE_RESULT created {adapter} 10 {config} 20\n", plan)
    assert result["config_identity"]["size"] == 20
    with pytest.raises(Policy0SourceError):
        parse_result("SOURCE_RESULT created bad 10 bad 20\n", plan)
