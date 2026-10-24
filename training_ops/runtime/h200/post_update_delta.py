"""Bounded-memory validation of the exported step-1 adapter delta."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping

# These are loaded only when the validator reaches tensor comparison.  The
# controller and its unit tests do not need the trainer image dependencies.
safe_open = None
torch = None


def _tensor_backend():
    global safe_open, torch
    if safe_open is None:
        try:
            from safetensors import safe_open as _safe_open
        except ImportError as error:  # pragma: no cover - trainer image only.
            raise Step1DeltaError("safetensors is required") from error
        safe_open = _safe_open
    if torch is None:
        try:
            import torch as _torch
        except ImportError as error:  # pragma: no cover - trainer image only.
            raise Step1DeltaError("torch is required") from error
        torch = _torch
    return safe_open, torch


STEP1 = 1
FORWARD_BACKWARD_LIMIT_SECONDS = 600.0
DEFAULT_CHUNK_ELEMENTS = 1 << 20
_DIGEST_FIELDS = (
    "reference_sha256",
    "trainer_config_sha256",
    "code_sha256",
    "loss_sha256",
)
_EXPECTED_IDENTITY_FIELDS = {
    "run_id",
    "reference_sha256",
    "trainer_config_sha256",
    "trainer_config_size",
    "code_sha256",
    "code_size",
    "loss_sha256",
    "loss_size",
    "loss_mode",
}
_EXPECTED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
)
_EXPECTED_TARGET_MODULE_SET = frozenset(_EXPECTED_TARGET_MODULES)
_EXPECTED_CONFIG_KEYS = frozenset(
    {
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "modules_to_save",
        "target_modules",
    }
)
_BASE_CONFIG_FIELDS = (
    "peft_type",
    "task_type",
    "base_model_name_or_path",
    "bias",
    "modules_to_save",
)
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class Step1DeltaError(ValueError):
    """The exported step-1 adapter or its evidence is unsafe."""


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise Step1DeltaError(f"{label} is missing or not regular: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise Step1DeltaError(f"{label} cannot be opened: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Step1DeltaError(f"{label} is not regular: {path}")
    finally:
        os.close(fd)


def _identity(path: Path, label: str) -> dict[str, int | str]:
    _regular(path, label)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise Step1DeltaError(f"{label} cannot be read: {path}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Step1DeltaError(f"{label} is not regular: {path}")
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return {"sha256": digest.hexdigest(), "size": info.st_size}
    except OSError as error:
        raise Step1DeltaError(f"{label} cannot be hashed: {path}") from error
    finally:
        os.close(fd)


def _regular_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise Step1DeltaError(f"{label} is missing or not a regular directory: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise Step1DeltaError(f"{label} cannot be opened: {path}") from error
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise Step1DeltaError(f"{label} is not a regular directory: {path}")
    finally:
        os.close(fd)


def _stable_identity(path: Path) -> dict[str, int | str]:
    """Verify the atomic export marker is the exact empty 0644 regular file."""
    _regular(path, "step-1 STABLE marker")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise Step1DeltaError("step-1 STABLE marker cannot be opened") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Step1DeltaError("step-1 STABLE marker is not regular")
        if stat.S_IMODE(info.st_mode) != 0o644:
            raise Step1DeltaError("step-1 STABLE marker mode is not 644")
        if info.st_size != 0:
            raise Step1DeltaError("step-1 STABLE marker is not empty")
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != _EMPTY_SHA256:
            raise Step1DeltaError("step-1 STABLE marker hash is not empty-file SHA256")
        return {"sha256": _EMPTY_SHA256, "size": 0}
    except OSError as error:
        raise Step1DeltaError("step-1 STABLE marker cannot be read") from error
    finally:
        os.close(fd)


def _validate_qualification_layout(policy_path: Path, step_dir: Path) -> Path:
    if (
        step_dir.name != "step_1"
        or step_dir.parent.name != "broadcasts"
        or step_dir.parent.parent.name != "run_trainer_only"
    ):
        raise Step1DeltaError("step-1 adapter path is not the verified export slot")
    output_root = step_dir.parent.parent.parent
    expected_dirs = (
        (output_root, "qualification output root"),
        (output_root / "run_trainer_only", "run_trainer_only directory"),
        (output_root / "run_trainer_only" / "broadcasts", "broadcasts directory"),
        (output_root / "run_trainer_only" / "broadcasts" / "step_1", "step-1 directory"),
    )
    for path, label in expected_dirs:
        _regular_directory(path, label)
    try:
        relative_policy = policy_path.relative_to(output_root)
    except ValueError as error:
        raise Step1DeltaError("policy-0 adapter is outside the qualification output root") from error
    if not relative_policy.parts or any(part in {".", ".."} for part in relative_policy.parts):
        raise Step1DeltaError("policy-0 adapter path is not canonical")
    parent = output_root
    for part in relative_policy.parts[:-1]:
        parent = parent / part
        _regular_directory(parent, "policy-0 adapter parent")
    _regular(policy_path, "policy-0 adapter")
    policy_config = policy_path.parent / "adapter_config.json"
    _regular(policy_config, "policy-0 adapter config")
    return policy_config


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise Step1DeltaError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise Step1DeltaError(f"{label} is not an object")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_adapter_config(step_config_path: Path, policy_config_path: Path) -> None:
    step_config = _read_json_object(step_config_path, "step-1 adapter config")
    policy_config = _read_json_object(policy_config_path, "policy-0 adapter config")
    for label, config in (("policy-0", policy_config), ("step-1", step_config)):
        if set(config) != _EXPECTED_CONFIG_KEYS:
            raise Step1DeltaError(f"{label} adapter config key set differs")
        if config.get("peft_type") != "LORA":
            raise Step1DeltaError(f"{label} adapter config peft_type differs")
        if config.get("task_type") != "CAUSAL_LM":
            raise Step1DeltaError(f"{label} adapter config task_type differs")
        if config.get("base_model_name_or_path") != "/data/trinity-thinking-fp8":
            raise Step1DeltaError(f"{label} adapter config base model differs")
        if config.get("bias") != "none":
            raise Step1DeltaError(f"{label} adapter config bias differs")
        if config.get("modules_to_save") is not None:
            raise Step1DeltaError(f"{label} adapter config modules_to_save differs")
        if type(config.get("r")) is not int or config["r"] != 32:
            raise Step1DeltaError(f"{label} adapter config rank is not 32")
        alpha = config.get("lora_alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) != 64:
            raise Step1DeltaError(f"{label} adapter config alpha is not 64")
        dropout = config.get("lora_dropout")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(float(dropout)) or float(dropout) != 0.0:
            raise Step1DeltaError(f"{label} adapter config dropout is not zero")
        targets = config.get("target_modules")
        if (
            not isinstance(targets, list)
            or any(not isinstance(target, str) for target in targets)
            or len(targets) != len(set(targets))
            or set(targets) != _EXPECTED_TARGET_MODULE_SET
        ):
            raise Step1DeltaError(f"{label} adapter config target_modules differ")
    for field in _BASE_CONFIG_FIELDS:
        if field not in policy_config or field not in step_config:
            raise Step1DeltaError(f"adapter config is missing base identity field: {field}")
        if policy_config[field] != step_config[field]:
            raise Step1DeltaError(f"adapter config base identity differs: {field}")
    for field in ("peft_type", "task_type", "base_model_name_or_path", "bias"):
        if not isinstance(policy_config[field], str) or not policy_config[field]:
            raise Step1DeltaError(f"adapter config base identity is invalid: {field}")


def _validate_expected_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(expected, Mapping) or set(expected) != _EXPECTED_IDENTITY_FIELDS:
        raise Step1DeltaError("expected run/reference/trainer-config/code identity is incomplete")
    result = dict(expected)
    if not isinstance(result["run_id"], str) or not result["run_id"]:
        raise Step1DeltaError("expected run identity is invalid")
    for field in _DIGEST_FIELDS:
        if not _valid_digest(result[field]):
            raise Step1DeltaError(f"expected {field} is invalid")
    for field in ("trainer_config_size", "code_size", "loss_size"):
        if type(result[field]) is not int or result[field] <= 0:
            raise Step1DeltaError(f"expected {field} is invalid")
    if not isinstance(result["loss_mode"], str) or result["loss_mode"] != "644":
        raise Step1DeltaError("expected loss mode is invalid")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step1DeltaError(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise Step1DeltaError(f"{label} is not finite")
    return value


def _validate_evidence(
    policy_identity: Mapping[str, Any],
    step_identity: Mapping[str, Any],
    expected: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    if not isinstance(evidence, Mapping):
        raise Step1DeltaError("trainer success evidence is missing")
    if evidence.get("run_id") != expected["run_id"]:
        raise Step1DeltaError("trainer run identity differs")
    if type(evidence.get("completed_optimizer_steps")) is not int or evidence["completed_optimizer_steps"] != 1:
        raise Step1DeltaError("exactly one completed optimizer step is required")
    if _finite_number(evidence.get("learning_rate"), "learning rate") != 8e-6:
        raise Step1DeltaError("learning rate must be 8e-6")
    _finite_number(evidence.get("loss"), "loss")
    grad_norm = _finite_number(evidence.get("grad_norm"), "grad norm")
    if grad_norm <= 0:
        raise Step1DeltaError("grad norm must be positive")
    forward_backward_seconds = _finite_number(
        evidence.get("forward_backward_seconds"), "forward/backward time"
    )
    if forward_backward_seconds < 0 or forward_backward_seconds > FORWARD_BACKWARD_LIMIT_SECONDS:
        raise Step1DeltaError("forward/backward time exceeds 600 seconds")
    export_seconds = _finite_number(evidence.get("export_seconds"), "export seconds")
    if export_seconds <= 0:
        raise Step1DeltaError("export seconds must be positive")
    if evidence.get("export_completed") is not True:
        raise Step1DeltaError("step-1 export is not complete")
    if evidence.get("initial_adapter_sha256") != policy_identity["sha256"] or evidence.get("initial_adapter_size") != policy_identity["size"]:
        raise Step1DeltaError("initial adapter identity differs")
    if evidence.get("final_adapter_sha256") != step_identity["sha256"] or evidence.get("final_adapter_size") != step_identity["size"]:
        raise Step1DeltaError("final adapter identity differs")
    for field in ("reference_sha256", "code_sha256"):
        if evidence.get(field) != expected[field]:
            raise Step1DeltaError(f"trainer evidence {field} differs")
    if evidence.get("trainer_config_sha256") != expected["trainer_config_sha256"]:
        raise Step1DeltaError("trainer config identity differs")
    if evidence.get("trainer_config_size") != expected["trainer_config_size"]:
        raise Step1DeltaError("trainer config size differs")
    if evidence.get("code_size") != expected["code_size"]:
        raise Step1DeltaError("trainer evidence code size differs")
    loss_identity = evidence.get("loss_identity")
    expected_loss = {
        "sha256": expected["loss_sha256"],
        "size": expected["loss_size"],
        "mode": expected["loss_mode"],
    }
    if loss_identity != expected_loss:
        raise Step1DeltaError("trainer loss identity differs")


def _validate_step1_files(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if step_dir.is_symlink() or not step_dir.is_dir():
        raise Step1DeltaError("step-1 adapter directory is missing or not regular")
    if step_dir.name != "step_1" or step_dir.parent.name != "broadcasts" or step_dir.parent.parent.name != "run_trainer_only":
        raise Step1DeltaError("step-1 adapter path is not the verified export slot")
    model = step_dir / "adapter_model.safetensors"
    config = step_dir / "adapter_config.json"
    sidecar = step_dir / "adapter_model.safetensors.sha256"
    manifest_path = step_dir / "adapter_manifest.json"
    stable = step_dir / "STABLE"
    paths = {
        "adapter_model.safetensors": model,
        "adapter_config.json": config,
        "adapter_model.safetensors.sha256": sidecar,
        "adapter_manifest.json": manifest_path,
        "STABLE": stable,
    }
    if {path.name for path in step_dir.iterdir()} != set(paths):
        raise Step1DeltaError("step-1 adapter has unexpected files")
    for name, path in paths.items():
        _regular(path, f"step-1 {name}")
    model_identity = _identity(model, "step-1 adapter")
    config_identity = _identity(config, "step-1 config")
    stable_identity = _stable_identity(stable)
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Step1DeltaError("step-1 manifest or sidecar is invalid") from error
    if sidecar_value != model_identity["sha256"]:
        raise Step1DeltaError("step-1 adapter sidecar differs")
    expected_files = {
        "adapter_config.json": {**config_identity},
        "adapter_model.safetensors": {**model_identity},
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("kind") != "lora-adapter"
        or manifest.get("mode") != "full"
        or manifest.get("target_step") != STEP1
        or manifest.get("base_step") is not None
        or manifest.get("target_sha256") != model_identity["sha256"]
        or manifest.get("target_size") != model_identity["size"]
        or manifest.get("full_sha256") != model_identity["sha256"]
        or manifest.get("full_size") != model_identity["size"]
        or manifest.get("config_sha256") != config_identity["sha256"]
        or manifest.get("config_size") != config_identity["size"]
    ):
        raise Step1DeltaError("step-1 manifest identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise Step1DeltaError("step-1 manifest file list is invalid")
    records: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise Step1DeltaError("step-1 manifest file record is invalid")
        if item["name"] in records:
            raise Step1DeltaError("step-1 manifest has duplicate file records")
        records[item["name"]] = item
    if set(records) != set(expected_files) or any(
        records[name].get("sha256") != identity["sha256"] or records[name].get("size") != identity["size"]
        for name, identity in expected_files.items()
    ):
        raise Step1DeltaError("step-1 manifest file identities differ")
    artifact_identities = {
        name: _identity(path, f"step-1 {name}")
        for name, path in paths.items()
    }
    artifact_identities["STABLE"] = stable_identity
    return model_identity, config_identity, artifact_identities


def _tensor_blocks(shape: tuple[int, ...], chunk_elements: int):
    if not shape:
        yield ()
        return

    def recurse(axis: int, prefix: list[slice], remaining: int):
        if axis == len(shape):
            yield tuple(prefix)
            return
        tail = math.prod(shape[axis + 1:])
        width = max(1, min(shape[axis], remaining // tail if tail else remaining))
        for start in range(0, shape[axis], width):
            end = min(shape[axis], start + width)
            yield from recurse(axis + 1, prefix + [slice(start, end)], max(1, remaining // (end - start)))

    yield from recurse(0, [], chunk_elements)


def _compare_tensors(policy_path: Path, step_path: Path, chunk_elements: int) -> dict[str, Any]:
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise Step1DeltaError("chunk size must be a positive integer")
    open_safetensors, torch_module = _tensor_backend()
    try:
        policy_handle = open_safetensors(str(policy_path), framework="pt", device="cpu")
        step_handle = open_safetensors(str(step_path), framework="pt", device="cpu")
    except Exception as error:
        raise Step1DeltaError("cannot open safetensors adapter") from error
    changed_elements = 0
    changed_tensors = 0
    max_abs = 0.0
    sum_squares = 0.0
    try:
        with policy_handle as policy, step_handle as step:
            policy_keys = set(policy.keys())
            step_keys = set(step.keys())
            if policy_keys != step_keys:
                raise Step1DeltaError("safetensors tensor key sets differ")
            for key in sorted(policy_keys):
                policy_slice = policy.get_slice(key)
                step_slice = step.get_slice(key)
                policy_shape = tuple(policy_slice.get_shape())
                step_shape = tuple(step_slice.get_shape())
                if policy_shape != step_shape or policy_slice.get_dtype() != step_slice.get_dtype():
                    raise Step1DeltaError(f"safetensors tensor metadata differs: {key}")
                tensor_changed = False
                for indexes in _tensor_blocks(policy_shape, chunk_elements):
                    # Only scalar tensors use get_tensor.  Every non-scalar
                    # read is a bounded slice, which also supports BF16.
                    policy_values = policy.get_tensor(key) if not indexes else policy_slice[indexes]
                    step_values = step.get_tensor(key) if not indexes else step_slice[indexes]
                    if policy_values.device.type != "cpu" or step_values.device.type != "cpu":
                        raise Step1DeltaError("safetensors tensor was not read on CPU")
                    if not bool(torch_module.isfinite(policy_values).all()) or not bool(torch_module.isfinite(step_values).all()):
                        raise Step1DeltaError(f"safetensors tensor {key} contains non-finite values")
                    policy_float = policy_values.to(dtype=torch_module.float64)
                    step_float = step_values.to(dtype=torch_module.float64)
                    difference = step_float - policy_float
                    if not bool(torch_module.isfinite(difference).all()):
                        raise Step1DeltaError(f"safetensors tensor {key} has a non-finite delta")
                    nonzero = policy_values != step_values
                    count = int(torch_module.count_nonzero(nonzero).item())
                    if count:
                        tensor_changed = True
                        changed_elements += count
                        max_abs = max(max_abs, float(torch_module.abs(difference[nonzero]).max().item()))
                    try:
                        sum_squares += float(torch_module.sum(difference * difference, dtype=torch_module.float64).item())
                    except (RuntimeError, OverflowError) as error:
                        raise Step1DeltaError("safetensors delta norm is not finite") from error
                    if not math.isfinite(sum_squares):
                        raise Step1DeltaError("safetensors delta norm is not finite")
                if tensor_changed:
                    changed_tensors += 1
    except Step1DeltaError:
        raise
    except Exception as error:
        raise Step1DeltaError("safetensors tensor read failed") from error
    if changed_elements == 0:
        raise Step1DeltaError("step-1 adapter has no nonzero delta")
    return {
        "changed_element_count": changed_elements,
        "changed_tensor_count": changed_tensors,
        "max_abs_delta": max_abs,
        "l2_delta": math.sqrt(sum_squares),
    }


def _atomic_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise Step1DeltaError(f"refusing to clobber existing receipt: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_step1_adapter_delta(
    policy0_adapter_path: Path,
    step1_dir: Path,
    expected_identity: Mapping[str, Any],
    trainer_evidence: Mapping[str, Any],
    receipt_path: Path,
    *,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    failure_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one exported step-1 adapter without loading either file whole."""
    failure_path = failure_receipt_path or receipt_path.with_name(receipt_path.stem + ".failure.json")
    try:
        expected = _validate_expected_identity(expected_identity)
        policy_path = Path(policy0_adapter_path)
        step_path = Path(step1_dir)
        policy_config_path = _validate_qualification_layout(policy_path, step_path)
        policy_identity = _identity(policy_path, "policy-0 adapter")
        policy_config_identity = _identity(policy_config_path, "policy-0 adapter config")
        model_identity, config_identity, artifact_identities = _validate_step1_files(step_path)
        if policy_config_identity != config_identity:
            raise Step1DeltaError("policy-0 and step-1 adapter config bytes differ")
        _validate_adapter_config(step_path / "adapter_config.json", policy_config_path)
        before = {
            "policy0_adapter": policy_identity,
            "policy0_config": policy_config_identity,
            "step1_adapter": model_identity,
            "step1_config": config_identity,
            "step1_manifest": artifact_identities["adapter_manifest.json"],
            "step1_sidecar": artifact_identities["adapter_model.safetensors.sha256"],
            "step1_stable": artifact_identities["STABLE"],
            "artifacts": artifact_identities,
        }
        metrics = _compare_tensors(
            policy_path,
            step_path / "adapter_model.safetensors",
            chunk_elements,
        )
        _, _, after_artifact_identities = _validate_step1_files(step_path)
        after = {
            "policy0_adapter": _identity(policy_path, "policy-0 adapter"),
            "policy0_config": _identity(policy_config_path, "policy-0 adapter config"),
            "step1_adapter": after_artifact_identities["adapter_model.safetensors"],
            "step1_config": after_artifact_identities["adapter_config.json"],
            "step1_manifest": after_artifact_identities["adapter_manifest.json"],
            "step1_sidecar": after_artifact_identities["adapter_model.safetensors.sha256"],
            "step1_stable": after_artifact_identities["STABLE"],
            "artifacts": after_artifact_identities,
        }
        if before != after:
            raise Step1DeltaError("adapter source changed during validation")
        _validate_evidence(policy_identity, model_identity, expected, trainer_evidence)
        receipt = {
            "schema_version": 1,
            "kind": "h200_step1_adapter_delta",
            "run_id": expected["run_id"],
            "policy0_adapter": policy_identity,
            "policy0_config": policy_config_identity,
            "step1_adapter": model_identity,
            "step1_config": config_identity,
            "step1_manifest_sha256": artifact_identities["adapter_manifest.json"]["sha256"],
            "trainer_config": {
                "sha256": expected["trainer_config_sha256"],
                "size": expected["trainer_config_size"],
            },
            "identities_before": before,
            "identities_after": after,
            "expected_identity": dict(expected),
            "trainer_evidence": dict(trainer_evidence),
            **metrics,
        }
        _atomic_no_clobber(Path(receipt_path), receipt)
        return receipt
    except Step1DeltaError as error:
        try:
            _atomic_no_clobber(
                Path(failure_path),
                {
                    "schema_version": 1,
                    "kind": "h200_step1_adapter_delta_failure",
                    "error": str(error),
                },
            )
        except Step1DeltaError:
            pass
        raise
    except Exception as error:
        wrapped = Step1DeltaError(f"step-1 adapter delta validation failed: {error}")
        try:
            _atomic_no_clobber(
                Path(failure_path),
                {
                    "schema_version": 1,
                    "kind": "h200_step1_adapter_delta_failure",
                    "error": str(wrapped),
                },
            )
        except Step1DeltaError:
            pass
        raise wrapped from error
