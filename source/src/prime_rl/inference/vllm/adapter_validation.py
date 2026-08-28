"""Fail-closed validation for the GLM-5.2 rank-32 adapter export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_MODEL_NAME = "adapter_model.safetensors"
ADAPTER_CHECKSUM_NAME = "adapter_model.safetensors.sha256"
ADAPTER_READY_NAME = "READY"

GLM_MODEL_ALIAS_ENV = "PRIME_LORA_BASE_MODEL_ALIAS"
GLM_MODEL_REVISION_ENV = "PRIME_LORA_BASE_MODEL_REVISION"
GLM_MODEL_SHA256_ENV = "PRIME_LORA_BASE_MODEL_SHA256"
GLM_VALIDATION_ENV = "PRIME_GLM_ADAPTER_VALIDATE"

GLM_LAYER_COUNT = 78
GLM_DENSE_LAYER_COUNT = 3
GLM_ROUTED_EXPERT_COUNT = 256
GLM_LORA_RANK = 32
GLM_LORA_ALPHA = 64

_MAX_HEADER_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TARGET_MODULES = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "experts",
    }
)


class AdapterValidationError(ValueError):
    """Raised when an adapter cannot be proven compatible with GLM-5.2."""


def validation_enabled() -> bool:
    value = os.environ.get(GLM_VALIDATION_ENV, "0")
    if value not in {"0", "1"}:
        raise AdapterValidationError(f"{GLM_VALIDATION_ENV} must be 0 or 1")
    return value == "1" or os.environ.get("PRIME_MOE_LORA_FORMAT") == "2d"


def expected_model_identity() -> dict[str, str]:
    values = {
        "alias": os.environ.get(GLM_MODEL_ALIAS_ENV, ""),
        "revision": os.environ.get(GLM_MODEL_REVISION_ENV, ""),
        "sha256": os.environ.get(GLM_MODEL_SHA256_ENV, ""),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise AdapterValidationError(
            "GLM adapter validation requires canonical model identity: "
            + ", ".join(missing)
        )
    if not _REVISION.fullmatch(values["revision"]):
        raise AdapterValidationError("canonical GLM model revision is invalid")
    if not _SHA256.fullmatch(values["sha256"]):
        raise AdapterValidationError("canonical GLM model SHA-256 is invalid")
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise AdapterValidationError("adapter safetensors header is truncated")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= _MAX_HEADER_BYTES:
                raise AdapterValidationError("adapter safetensors header size is invalid")
            encoded = source.read(header_bytes)
            if len(encoded) != header_bytes:
                raise AdapterValidationError("adapter safetensors header is truncated")
    except OSError as error:
        raise AdapterValidationError(f"cannot read adapter safetensors: {path}") from error
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError("adapter safetensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise AdapterValidationError("adapter safetensors header is not an object")
    return header, header_bytes


def _expected_shapes() -> dict[str, tuple[int, int]]:
    rank = GLM_LORA_RANK
    values: dict[str, tuple[int, int]] = {}

    def add(prefix: str, target: str, input_width: int, output_width: int) -> None:
        values[f"{prefix}.{target}.lora_A.weight"] = (rank, input_width)
        values[f"{prefix}.{target}.lora_B.weight"] = (output_width, rank)

    for layer in range(GLM_LAYER_COUNT):
        prefix = f"model.layers.{layer}.self_attn"
        add(prefix, "q_a_proj", 6144, 2048)
        add(prefix, "q_b_proj", 2048, 24576)
        add(prefix, "kv_a_proj_with_mqa", 6144, 576)
        add(prefix, "kv_b_proj", 512, 32768)
        add(prefix, "o_proj", 16384, 6144)

    for layer in range(GLM_DENSE_LAYER_COUNT):
        prefix = f"model.layers.{layer}.mlp"
        add(prefix, "gate_proj", 6144, 3072)
        add(prefix, "up_proj", 6144, 3072)
        add(prefix, "down_proj", 3072, 6144)

    for layer in range(GLM_DENSE_LAYER_COUNT, GLM_LAYER_COUNT):
        shared_prefix = f"model.layers.{layer}.mlp.shared_experts"
        for target in ("gate_proj", "up_proj"):
            add(shared_prefix, target, 6144, 2048)
        add(shared_prefix, "down_proj", 2048, 6144)
        for expert in range(GLM_ROUTED_EXPERT_COUNT):
            expert_prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            for target in ("gate_proj", "up_proj"):
                add(expert_prefix, target, 6144, 2048)
            add(expert_prefix, "down_proj", 2048, 6144)
    return values


def expected_tensor_shapes() -> dict[str, tuple[int, int]]:
    return _expected_shapes()


def _validate_safetensors(
    path: Path, expected: dict[str, tuple[int, int]]
) -> None:
    header, header_bytes = _read_header(path)
    actual_names = {name for name in header if name != "__metadata__"}
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} ({missing[:2]})")
        if extra:
            details.append(f"extra={len(extra)} ({extra[:2]})")
        raise AdapterValidationError("adapter tensor keys differ: " + "; ".join(details))

    intervals: list[tuple[int, int, str]] = []
    for name in sorted(expected_names):
        spec = header.get(name)
        if not isinstance(spec, dict):
            raise AdapterValidationError(f"adapter tensor metadata is invalid: {name}")
        if spec.get("dtype") != "BF16":
            raise AdapterValidationError(f"adapter tensor dtype is not BF16: {name}")
        shape = spec.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            or tuple(shape) != expected[name]
        ):
            raise AdapterValidationError(
                f"adapter tensor shape differs: {name}={shape}, expected={expected[name]}"
            )
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
            or offsets[1] < offsets[0]
            or offsets[1] - offsets[0] != 2 * math.prod(shape)
        ):
            raise AdapterValidationError(f"adapter tensor offsets are invalid: {name}")
        intervals.append((offsets[0], offsets[1], name))

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise AdapterValidationError(f"adapter tensor data is not contiguous: {name}")
        cursor = end
    try:
        file_size = path.stat().st_size
    except OSError as error:
        raise AdapterValidationError("cannot stat adapter safetensors") from error
    if file_size != 8 + header_bytes + cursor:
        raise AdapterValidationError("adapter safetensors file size differs from its header")


def validate_glm52_adapter(
    adapter_dir: str | Path,
    *,
    expected_alias: str,
    expected_revision: str,
    expected_sha256: str,
    is_3d_lora_weight: bool = False,
) -> dict[str, object]:
    """Validate one complete, global-shape GLM-5.2 2D adapter export."""
    if is_3d_lora_weight is not False:
        raise AdapterValidationError("GLM adapter must use 2D per-expert weights")
    if not expected_alias or not _REVISION.fullmatch(expected_revision or ""):
        raise AdapterValidationError("expected GLM model identity is invalid")
    if not _SHA256.fullmatch(expected_sha256 or ""):
        raise AdapterValidationError("expected GLM model SHA-256 is invalid")

    root = Path(adapter_dir)
    if root.is_symlink() or not root.is_dir():
        raise AdapterValidationError("adapter path is not a directory")
    config_path = root / ADAPTER_CONFIG_NAME
    model_path = root / ADAPTER_MODEL_NAME
    checksum_path = root / ADAPTER_CHECKSUM_NAME
    for path in (config_path, model_path, checksum_path):
        if path.is_symlink() or not path.is_file():
            raise AdapterValidationError(f"adapter file is missing or unsafe: {path.name}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError("adapter configuration is not valid JSON") from error
    if not isinstance(config, dict):
        raise AdapterValidationError("adapter configuration is not an object")

    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise AdapterValidationError("adapter type is not LoRA causal language modeling")
    if type(config.get("r")) is not int or config["r"] != GLM_LORA_RANK:
        raise AdapterValidationError("adapter rank is not 32")
    if type(config.get("lora_alpha")) not in {int, float} or config["lora_alpha"] != GLM_LORA_ALPHA:
        raise AdapterValidationError("adapter alpha is not 64")
    if type(config.get("lora_dropout")) not in {int, float} or config["lora_dropout"] != 0.0:
        raise AdapterValidationError("adapter dropout is not zero")
    if config.get("bias") != "none" or config.get("modules_to_save") is not None:
        raise AdapterValidationError("adapter has unsupported trainable base modules")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or len(targets) != len(_TARGET_MODULES)
        or any(not isinstance(target, str) for target in targets)
        or set(targets) != _TARGET_MODULES
    ):
        raise AdapterValidationError("adapter target modules differ from GLM-5.2")
    if config.get("base_model_name_or_path") != expected_alias:
        raise AdapterValidationError("adapter base model alias differs")
    if config.get("base_model_revision") != expected_revision:
        raise AdapterValidationError("adapter base model revision differs")
    if config.get("base_model_sha256") != expected_sha256:
        raise AdapterValidationError("adapter base model SHA-256 differs")

    try:
        checksum = checksum_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise AdapterValidationError("adapter checksum file cannot be read") from error
    if checksum != f"{_sha256_file(model_path)}\n":
        raise AdapterValidationError("adapter checksum does not match its bytes")
    _validate_safetensors(model_path, _expected_shapes())

    return {
        "tensor_count": len(_expected_shapes()),
        "rank": GLM_LORA_RANK,
        "alpha": GLM_LORA_ALPHA,
        "is_3d_lora_weight": False,
        "base_model_revision": expected_revision,
        "base_model_sha256": expected_sha256,
        "adapter_sha256": checksum.rstrip("\n"),
    }
