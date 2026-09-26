"""Fail-closed checks for the separate H200 post-update smoke qualification.

This module does not start a trainer or inference service.  It validates the
local evidence that an operator must collect after one real trainer update:
the reference, exported adapter, transfer, load, and post-update receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .prepare_trinity_policy0_qualification import (
    QualificationError,
    validate_reference as validate_policy0_reference,
)


POST_UPDATE_STEP = 1
POST_UPDATE_LOAD_SLOT = "trinity-h200-post-update-qualification-slot1"
POST_UPDATE_REQUEST_MODE = "serialized_generate_completion_logprobs"
FORWARD_BACKWARD_GATE_SECONDS = 600.0

_EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}
_IDENTITY_FIELDS = ("target_sha256", "target_size", "config_sha256", "config_size")
_RECEIPT_FIELDS = (
    "source_url",
    "target_sha256",
    "target_size",
    "config_sha256",
    "config_size",
    "lora_name",
    "server_instance",
)


class PostUpdateQualificationError(ValueError):
    """The post-update smoke qualification evidence is unsafe or incomplete."""


def post_update_adapter_path(output_dir: str | Path) -> Path:
    """Return the only exported step path accepted by this qualification."""
    root = Path(output_dir)
    return root / "run_trainer_only" / "broadcasts" / f"step_{POST_UPDATE_STEP}"


def post_update_load_target(output_dir: str | Path) -> dict[str, Any]:
    """Return the verified step and slot used to load the real update."""
    return {
        "step": POST_UPDATE_STEP,
        "slot": POST_UPDATE_LOAD_SLOT,
        "adapter_path": str(post_update_adapter_path(output_dir)),
    }


def validate_forward_backward_seconds(value: Any) -> float:
    """Keep the existing 600-second F+B gate separate from adapter timings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostUpdateQualificationError("forward/backward time is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise PostUpdateQualificationError("forward/backward time is not finite")
    if value > FORWARD_BACKWARD_GATE_SECONDS:
        raise PostUpdateQualificationError(
            f"forward/backward time exceeds the {FORWARD_BACKWARD_GATE_SECONDS:g}s gate"
        )
    return value


def validate_post_update_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one non-rescore reference with real, aligned advantages."""
    if not isinstance(value, Mapping):
        raise PostUpdateQualificationError("post-update reference must be an object")
    request_mode = value.get("request_mode")
    if request_mode == "serialized_generate_decode256_prefill_rescore":
        raise PostUpdateQualificationError(
            "post-update qualification rejects rescore-mode references"
        )
    if request_mode != POST_UPDATE_REQUEST_MODE:
        raise PostUpdateQualificationError(
            "post-update reference must use completion qualification mode"
        )
    try:
        validated = validate_policy0_reference(value)
    except QualificationError as error:
        raise PostUpdateQualificationError(str(error)) from error

    input_ids = validated["input_ids"]
    loss_mask = validated["loss_mask"]
    advantages = validated.get("advantages")
    if not isinstance(advantages, list):
        raise PostUpdateQualificationError(
            "post-update reference advantages are missing"
        )
    if len(advantages) != len(input_ids):
        raise PostUpdateQualificationError(
            "post-update reference advantages do not align with input tokens"
        )
    numeric: list[float] = []
    for index, advantage in enumerate(advantages):
        if isinstance(advantage, bool) or not isinstance(advantage, (int, float)):
            raise PostUpdateQualificationError(
                f"post-update reference advantage {index} is not numeric"
            )
        value_float = float(advantage)
        if not math.isfinite(value_float):
            raise PostUpdateQualificationError(
                f"post-update reference advantage {index} is not finite"
            )
        if not loss_mask[index] and value_float != 0.0:
            raise PostUpdateQualificationError(
                "post-update reference advantages are misaligned with the loss mask"
            )
        numeric.append(value_float)
    if not any(value_float != 0.0 for value_float, keep in zip(numeric, loss_mask) if keep):
        raise PostUpdateQualificationError(
            "post-update reference advantages are all zero"
        )
    return dict(validated)


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PostUpdateQualificationError(f"{label} is missing or not regular: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PostUpdateQualificationError(f"cannot read adapter artifact: {path}") from error
    return digest.hexdigest()


def _all_zero(path: Path) -> bool:
    with path.open("rb") as stream:
        saw_bytes = False
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            saw_bytes = saw_bytes or bool(block)
            if any(block):
                return False
    return not saw_bytes


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        for left_block, right_block in zip(
            iter(lambda: left_stream.read(1024 * 1024), b""),
            iter(lambda: right_stream.read(1024 * 1024), b""),
        ):
            if left_block != right_block:
                return False
    return True


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _adapter_identity(adapter_dir: Path) -> dict[str, Any]:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise PostUpdateQualificationError("post-update adapter directory is missing")
    if adapter_dir.name != f"step_{POST_UPDATE_STEP}" or adapter_dir.parent.name != "broadcasts" or adapter_dir.parent.parent.name != "run_trainer_only":
        raise PostUpdateQualificationError("post-update adapter path is not the verified step-1 slot")
    adapter = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    sidecar = adapter_dir / "adapter_model.safetensors.sha256"
    manifest_path = adapter_dir / "adapter_manifest.json"
    stable = adapter_dir / "STABLE"
    for path, label in (
        (adapter, "post-update adapter weights"),
        (config, "post-update adapter config"),
        (sidecar, "post-update adapter checksum"),
        (manifest_path, "post-update adapter manifest"),
        (stable, "post-update adapter STABLE marker"),
    ):
        _regular(path, label)
    target_sha = _sha256(adapter)
    config_sha = _sha256(config)
    target_size = adapter.stat().st_size
    config_size = config.stat().st_size
    if target_size <= 0 or _all_zero(adapter):
        raise PostUpdateQualificationError("post-update adapter weights are empty or all zero")
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise PostUpdateQualificationError("post-update adapter checksum is invalid") from error
    if sidecar_value != target_sha:
        raise PostUpdateQualificationError("post-update adapter checksum sidecar differs")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter_config = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PostUpdateQualificationError("post-update adapter metadata is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("kind") != "lora-adapter"
        or manifest.get("mode") != "full"
        or manifest.get("target_step") != POST_UPDATE_STEP
        or manifest.get("base_step") is not None
        or manifest.get("target_sha256") != target_sha
        or manifest.get("target_size") != target_size
        or manifest.get("full_sha256") != target_sha
        or manifest.get("full_size") != target_size
        or manifest.get("config_sha256") != config_sha
        or manifest.get("config_size") != config_size
    ):
        raise PostUpdateQualificationError("post-update adapter manifest identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise PostUpdateQualificationError("post-update adapter manifest file list is invalid")
    records = {item.get("name"): item for item in files if isinstance(item, dict)}
    if set(records) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise PostUpdateQualificationError("post-update adapter manifest file list is invalid")
    for name, size, digest in (
        ("adapter_config.json", config_size, config_sha),
        ("adapter_model.safetensors", target_size, target_sha),
    ):
        record = records[name]
        if record.get("size") != size or record.get("sha256") != digest:
            raise PostUpdateQualificationError("post-update adapter manifest file identity differs")
    if (
        type(adapter_config.get("r")) is not int
        or adapter_config.get("r") != 32
        or type(adapter_config.get("lora_alpha")) is not int
        or adapter_config.get("lora_alpha") != 64
        or isinstance(adapter_config.get("lora_dropout"), bool)
        or not isinstance(adapter_config.get("lora_dropout"), (int, float))
        or not math.isfinite(float(adapter_config.get("lora_dropout")))
        or float(adapter_config.get("lora_dropout")) != 0.0
        or not isinstance(adapter_config.get("target_modules"), list)
        or any(not isinstance(target, str) for target in adapter_config.get("target_modules", []))
        or set(adapter_config["target_modules"]) != _EXPECTED_TARGETS
        or len(adapter_config["target_modules"]) != len(_EXPECTED_TARGETS)
    ):
        raise PostUpdateQualificationError("post-update adapter LoRA configuration differs")
    expected_entries = {
        "STABLE",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.safetensors.sha256",
        "adapter_manifest.json",
    }
    if {path.name for path in adapter_dir.iterdir()} != expected_entries:
        raise PostUpdateQualificationError("post-update adapter has unexpected files")
    return {
        "adapter_path": adapter,
        "config_path": config,
        "manifest_path": manifest_path,
        "stable_path": stable,
        "target_sha256": target_sha,
        "target_size": target_size,
        "config_sha256": config_sha,
        "config_size": config_size,
        "manifest_sha256": _sha256(manifest_path),
    }


def _check_identity(value: Mapping[str, Any], identity: Mapping[str, Any], label: str) -> None:
    for field in _IDENTITY_FIELDS:
        if field.endswith("_sha256") and not _valid_digest(value.get(field)):
            raise PostUpdateQualificationError(f"{label} {field} is invalid")
        if field.endswith("_size") and (type(value.get(field)) is not int or value[field] < 0):
            raise PostUpdateQualificationError(f"{label} {field} is invalid")
        if value.get(field) != identity[field]:
            raise PostUpdateQualificationError(f"{label} {field} differs from exported adapter")


def _check_receipt(value: Mapping[str, Any], identity: Mapping[str, Any], label: str, source_url: str | None = None) -> str:
    if not isinstance(value, Mapping):
        raise PostUpdateQualificationError(f"{label} evidence is missing")
    for field in _RECEIPT_FIELDS:
        if field not in value:
            raise PostUpdateQualificationError(f"{label} receipt is missing {field}")
    if not isinstance(value["source_url"], str) or not value["source_url"]:
        raise PostUpdateQualificationError(f"{label} receipt source URL is invalid")
    if source_url is not None and value["source_url"] != source_url:
        raise PostUpdateQualificationError(f"{label} receipt source URL differs")
    if value["lora_name"] != POST_UPDATE_LOAD_SLOT or not isinstance(value["server_instance"], str) or not value["server_instance"]:
        raise PostUpdateQualificationError(f"{label} receipt slot or server identity is invalid")
    _check_identity(value, identity, label)
    return value["source_url"]


def validate_post_update_evidence(
    adapter_dir: Path,
    policy0_adapter_path: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a fresh exported step, transfer, load, and post-update receipt."""
    if not isinstance(evidence, Mapping):
        raise PostUpdateQualificationError("post-update evidence is missing")
    for field in ("save", "transfer", "load", "post_update"):
        if not isinstance(evidence.get(field), Mapping):
            raise PostUpdateQualificationError(f"post-update {field} evidence is missing")
    policy0_adapter_path = Path(policy0_adapter_path)
    _regular(policy0_adapter_path, "policy-0 adapter")
    identity = _adapter_identity(Path(adapter_dir))
    policy0_sha = _sha256(policy0_adapter_path)
    if identity["target_sha256"] == policy0_sha or _same_bytes(identity["adapter_path"], policy0_adapter_path):
        raise PostUpdateQualificationError(
            "post-update adapter bytes or SHA differs from policy-0"
        )

    save = evidence["save"]
    if save.get("path") != str(identity["adapter_path"]) or save.get("step") != POST_UPDATE_STEP:
        raise PostUpdateQualificationError("post-update save evidence does not name exported step 1")
    _check_identity(save, identity, "post-update save evidence")
    if save.get("manifest_sha256") != identity["manifest_sha256"]:
        raise PostUpdateQualificationError("post-update save evidence manifest differs")

    transfer = evidence["transfer"]
    if transfer.get("path") != str(Path(adapter_dir)):
        raise PostUpdateQualificationError("post-update transfer evidence path differs")
    _check_identity(transfer, identity, "post-update transfer evidence")

    source_url = _check_receipt(evidence["load"], identity, "post-update load")
    post_update = evidence["post_update"]
    if post_update.get("step") != POST_UPDATE_STEP:
        raise PostUpdateQualificationError("post-update receipt does not name exported step 1")
    _check_receipt(post_update, identity, "post-update", source_url)
    return {
        "step": POST_UPDATE_STEP,
        "slot": POST_UPDATE_LOAD_SLOT,
        "adapter_path": str(identity["adapter_path"]),
        "source_url": source_url,
        "target_sha256": identity["target_sha256"],
        "target_size": identity["target_size"],
        "config_sha256": identity["config_sha256"],
        "config_size": identity["config_size"],
    }
