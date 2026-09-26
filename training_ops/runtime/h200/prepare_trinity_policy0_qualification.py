"""Validate the exact step-0 Trinity policy reference contract.

This module is local and side-effect free.  ``training_ops`` uses its
validators before it stages the three files through its existing fan-out path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class QualificationError(ValueError):
    """The policy-0 qualification artifact is not safe to stage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise QualificationError(f"cannot read qualification artifact {path}") from error
    return digest.hexdigest()


def validate_lora_config(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Require adapter settings to equal the trainer's configured settings."""
    if not isinstance(value, Mapping):
        raise QualificationError("adapter_config.json must be an object")
    for key in ("r", "lora_alpha", "lora_dropout"):
        expected_value = expected.get(key)
        actual_value = value.get(key)
        if key == "lora_dropout":
            try:
                actual_float = float(actual_value)
                expected_float = float(expected_value)
            except (TypeError, ValueError) as error:
                raise QualificationError(f"adapter config differs in {key}") from error
            if isinstance(actual_value, bool) or actual_float != expected_float:
                raise QualificationError(f"adapter config differs in {key}")
        elif type(actual_value) is not int or actual_value != expected_value:
            raise QualificationError(f"adapter config differs in {key}")
    expected_targets = expected.get("target_modules")
    actual_targets = value.get("target_modules")
    if (
        not isinstance(expected_targets, list)
        or not isinstance(actual_targets, list)
        or any(not isinstance(target, str) for target in expected_targets)
        or any(not isinstance(target, str) for target in actual_targets)
        or set(actual_targets) != set(expected_targets)
        or len(actual_targets) != len(expected_targets)
    ):
        raise QualificationError("adapter config target_modules differ")
    return dict(value)


def validate_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require one fresh, full-token, policy-0 trainer reference."""
    if not isinstance(value, Mapping):
        raise QualificationError("reference must be an object")
    if value.get("policy") != "policy-0":
        raise QualificationError("reference is not a policy-0 reference")
    if value.get("request_mode") != "serialized_generate_completion_prompt_logprobs":
        raise QualificationError("reference does not use completion qualification mode")
    if value.get("source_kind") in {"rollout_batch", "cached_rollout_batch"}:
        raise QualificationError("cached rollout reference is not allowed")
    input_ids = value.get("input_ids")
    logprobs = value.get("inference_logprobs")
    loss_mask = value.get("loss_mask")
    lengths = value.get("sequence_lengths")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
        or not isinstance(logprobs, list)
        or len(logprobs) != len(input_ids)
        or not isinstance(loss_mask, list)
        or len(loss_mask) != len(input_ids)
        or any(not isinstance(flag, bool) for flag in loss_mask)
        or not isinstance(lengths, list)
        or any(type(length) is not int or length <= 0 for length in lengths)
        or sum(lengths) != len(input_ids)
    ):
        raise QualificationError("reference token, logprob, mask, or length fields do not align")
    if not any(loss_mask) or all(loss_mask):
        raise QualificationError("reference must contain prompt and generated loss-mask regions")
    first_generated = loss_mask.index(True)
    if any(loss_mask[:first_generated]) or not all(loss_mask[first_generated:]):
        raise QualificationError("reference loss mask is not prompt-then-completion")
    route = value.get("routed_experts")
    if not isinstance(route, Mapping) or route.get("shape", [None])[0] != len(input_ids) - 1:
        raise QualificationError("reference routes must omit the final token row")
    for field in ("policy_identity_before", "policy_identity_after", "cache_salt"):
        if field not in value or value[field] is None:
            raise QualificationError(f"reference is missing {field}")
    if value["policy_identity_before"] != value["policy_identity_after"]:
        raise QualificationError("reference policy identity changed during generation")
    generated_ids = value.get("generated_token_ids")
    if (
        not isinstance(generated_ids, list)
        or value.get("generated_token_count") != len(generated_ids)
        or len(generated_ids) != sum(loss_mask)
    ):
        raise QualificationError("reference generated token metadata is not aligned")
    return dict(value)


def validate_local_artifacts(
    adapter_path: Path,
    config_path: Path,
    reference_path: Path,
    trainer_lora: Mapping[str, Any],
) -> dict[str, dict[str, int | str]]:
    """Validate the source files and return exact identities for each file."""
    try:
        config = json.loads(config_path.read_text())
        reference = json.loads(reference_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError("qualification source JSON is unreadable") from error
    validate_lora_config(config, trainer_lora)
    validate_reference(reference)
    files = {
        "adapter_model.safetensors": adapter_path,
        "adapter_config.json": config_path,
        "reference.json": reference_path,
    }
    result: dict[str, dict[str, int | str]] = {}
    for name, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise QualificationError(f"qualification source file is missing: {path}")
        result[name] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return result
