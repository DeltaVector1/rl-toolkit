"""Exact adjacent-step LoRA adapter deltas.

The delta is an XOR of the serialized safetensors bytes.  It is therefore a
byte-for-byte operation: in particular, BF16 payloads are never decoded or
rounded.  The XOR stream is split into independently compressed zlib frames so
that a corrupt or interrupted transfer cannot produce a usable adapter.

This module is opt-in.  The caller must choose the delta transfer explicitly;
the normal adapter download remains the fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Mapping


ADAPTER_DELTA_VERSION = 1
ADAPTER_DELTA_CODEC = "xor-zlib-frames"
ADAPTER_DELTA_MANIFEST = "adapter_delta.json"
ADAPTER_DELTA_FILE = "adapter_model.safetensors.delta"
# Explicit *_NAME spellings make the on-disk protocol easy to use from
# launchers without duplicating string literals.
ADAPTER_DELTA_MANIFEST_NAME = ADAPTER_DELTA_MANIFEST
ADAPTER_DELTA_FILE_NAME = ADAPTER_DELTA_FILE
ADAPTER_DELTA_ENABLE_ENV = "PRIME_ADAPTER_DELTA"
ADAPTER_DELTA_MAX_RATIO_ENV = "PRIME_ADAPTER_DELTA_MAX_RATIO"
ADAPTER_DELTA_MAX_RATIO = 0.90
ADAPTER_DELTA_FRAME_BYTES = 8 * 1024 * 1024
_FRAME_HEADER = struct.Struct("!II")
_IO_CHUNK = 16 * 1024 * 1024
_MAX_SAFETENSORS_HEADER = 256 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")

FULL_TRANSFER = "full"
DELTA_TRANSFER = "delta"


class AdapterDeltaError(OSError):
    """Raised when an adapter delta cannot be trusted or applied."""


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_IO_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def adapter_cache_identity(source_url: str, target_sha256: str) -> str:
    """Hash the URL and target digest used for a completed cache directory."""
    if not _valid_sha(target_sha256):
        raise ValueError("adapter target SHA-256 is invalid")
    return _sha256_bytes(f"{source_url}\x00{target_sha256}".encode())


def _read_safetensors_header(path: Path) -> Mapping[str, object]:
    """Read and validate the JSON header without loading tensor payloads."""
    try:
        with path.open("rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise AdapterDeltaError(f"safetensors header is truncated: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > _MAX_SAFETENSORS_HEADER:
                raise AdapterDeltaError(f"safetensors header length is invalid: {path.name}")
            header = source.read(header_length)
            if len(header) != header_length:
                raise AdapterDeltaError(f"safetensors header is truncated: {path.name}")
    except OSError as error:
        raise AdapterDeltaError(f"cannot read safetensors header: {path}") from error
    try:
        value = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterDeltaError(f"safetensors header is not JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise AdapterDeltaError(f"safetensors header is not an object: {path.name}")
    # The first tensor header item may be __metadata__, but all tensor entries
    # must have the fields that make the serialized layout unambiguous.  This
    # catches a target with changed shapes or offsets before it reaches vLLM.
    for name, spec in value.items():
        if name == "__metadata__":
            if not isinstance(spec, dict) or not all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in spec.items()
            ):
                raise AdapterDeltaError(f"safetensors user metadata is invalid: {path.name}")
            continue
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise AdapterDeltaError(f"safetensors tensor metadata is invalid: {path.name}")
        if not isinstance(spec.get("dtype"), str) or not isinstance(spec.get("shape"), list):
            raise AdapterDeltaError(f"safetensors tensor metadata is incomplete: {path.name}")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) and item >= 0 for item in offsets)
            or offsets[1] < offsets[0]
        ):
            raise AdapterDeltaError(f"safetensors tensor offsets are invalid: {path.name}")
    return value


def safetensors_metadata_sha256(path: Path) -> str:
    """Hash canonical tensor metadata, including user metadata and offsets."""
    header = _read_safetensors_header(path)
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(canonical)


def adapter_identity(adapter_path: Path, config_path: Path) -> dict[str, object]:
    """Return the content identity used by the delta manifest and READY file."""
    size = adapter_path.stat().st_size
    return {
        "target_sha256": sha256_file(adapter_path),
        "adapter_config_sha256": sha256_file(config_path),
        "tensor_metadata_sha256": safetensors_metadata_sha256(adapter_path),
        "uncompressed_size": size,
    }


def _max_ratio(value: float | None = None) -> float:
    if value is not None:
        ratio = float(value)
    else:
        raw = os.environ.get(ADAPTER_DELTA_MAX_RATIO_ENV)
        ratio = float(raw) if raw is not None else ADAPTER_DELTA_MAX_RATIO
    if not 0.0 < ratio < 1.0:
        raise ValueError("adapter delta max ratio must be between zero and one")
    return ratio


def wire_size_break_even(full_wire_size: int, manifest_size: int = 0) -> int:
    """Return the largest delta size that is strictly smaller on the wire."""
    if full_wire_size <= 0 or manifest_size < 0:
        raise ValueError("wire sizes must be non-negative and full size must be positive")
    return max(0, full_wire_size - manifest_size - 1)


def _xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise AdapterDeltaError("adapter delta inputs have different lengths")
    return bytes(a ^ b for a, b in zip(left, right))


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _delta_paths(target_dir: Path) -> tuple[Path, Path]:
    return target_dir / ADAPTER_DELTA_FILE, target_dir / ADAPTER_DELTA_MANIFEST


def build_adapter_delta(
    base_dir: Path,
    target_dir: Path,
    *,
    base_url: str = "",
    target_url: str = "",
    max_ratio: float | None = None,
    frame_bytes: int = ADAPTER_DELTA_FRAME_BYTES,
) -> Path | None:
    """Build an adjacent adapter delta, or return ``None`` for full transfer.

    The target directory is never advertised until both the delta and manifest
    are complete.  A poor ratio removes any stale delta advertisement so the
    receiver explicitly selects the normal full adapter transfer.
    """
    delta_path, manifest_path = _delta_paths(target_dir)
    if frame_bytes <= 0 or frame_bytes > ADAPTER_DELTA_FRAME_BYTES:
        raise ValueError("adapter delta frame size is invalid")
    try:
        base_model = base_dir / "adapter_model.safetensors"
        target_model = target_dir / "adapter_model.safetensors"
        base_config = base_dir / "adapter_config.json"
        target_config = target_dir / "adapter_config.json"
        if any(not path.is_file() for path in (base_model, target_model, base_config, target_config)):
            _remove_if_exists(delta_path)
            _remove_if_exists(manifest_path)
            return None
        if base_model.stat().st_size != target_model.stat().st_size:
            _remove_if_exists(delta_path)
            _remove_if_exists(manifest_path)
            return None
        base_identity = adapter_identity(base_model, base_config)
        target_identity = adapter_identity(target_model, target_config)
    except (OSError, AdapterDeltaError, ValueError):
        _remove_if_exists(delta_path)
        _remove_if_exists(manifest_path)
        return None

    temporary = target_dir / f".{ADAPTER_DELTA_FILE}.{os.getpid()}.partial"
    _remove_if_exists(temporary)
    try:
        with (
            base_model.open("rb") as base,
            target_model.open("rb") as target,
            temporary.open("xb") as output,
        ):
            while target_chunk := target.read(frame_bytes):
                base_chunk = base.read(len(target_chunk))
                if len(base_chunk) != len(target_chunk):
                    raise AdapterDeltaError("base adapter ended before target adapter")
                encoded = zlib.compress(_xor(base_chunk, target_chunk), level=1)
                output.write(_FRAME_HEADER.pack(len(target_chunk), len(encoded)))
                output.write(encoded)
            if base.read(1):
                raise AdapterDeltaError("base adapter is longer than target adapter")
            output.flush()
            os.fsync(output.fileno())
        delta_size = temporary.stat().st_size
        ratio = delta_size / target_identity["uncompressed_size"]
        if ratio >= _max_ratio(max_ratio):
            _remove_if_exists(temporary)
            _remove_if_exists(delta_path)
            _remove_if_exists(manifest_path)
            return None
        delta_sha256 = sha256_file(temporary)
        manifest: dict[str, object] = {
            "version": ADAPTER_DELTA_VERSION,
            "codec": ADAPTER_DELTA_CODEC,
            "codec_version": ADAPTER_DELTA_VERSION,
            "frame_bytes": frame_bytes,
            "base_url": base_url,
            "target_url": target_url,
            "base_sha256": base_identity["target_sha256"],
            "target_sha256": target_identity["target_sha256"],
            "base_adapter_config_sha256": base_identity["adapter_config_sha256"],
            "adapter_config_sha256": target_identity["adapter_config_sha256"],
            "base_tensor_metadata_sha256": base_identity["tensor_metadata_sha256"],
            "tensor_metadata_sha256": target_identity["tensor_metadata_sha256"],
            "base_size": base_identity["uncompressed_size"],
            "uncompressed_size": target_identity["uncompressed_size"],
            "delta_size": delta_size,
            "delta_sha256": delta_sha256,
        }
        manifest_temporary = target_dir / f".{ADAPTER_DELTA_MANIFEST}.{os.getpid()}.partial"
        _remove_if_exists(manifest_temporary)
        try:
            with manifest_temporary.open("x", encoding="utf-8") as output:
                json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, delta_path)
            os.replace(manifest_temporary, manifest_path)
        finally:
            _remove_if_exists(manifest_temporary)
    except BaseException:
        _remove_if_exists(temporary)
        raise
    return manifest_path


def load_adapter_delta_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterDeltaError(f"cannot read adapter delta manifest: {path}") from error
    if not isinstance(value, dict):
        raise AdapterDeltaError("adapter delta manifest is not an object")
    required = (
        "version",
        "codec",
        "codec_version",
        "base_sha256",
        "target_sha256",
        "adapter_config_sha256",
        "tensor_metadata_sha256",
        "uncompressed_size",
        "delta_size",
        "delta_sha256",
    )
    if any(key not in value for key in required):
        raise AdapterDeltaError("adapter delta manifest is missing an identity field")
    if value["version"] != ADAPTER_DELTA_VERSION or value["codec"] != ADAPTER_DELTA_CODEC:
        raise AdapterDeltaError("adapter delta codec is unsupported")
    if value["codec_version"] != ADAPTER_DELTA_VERSION:
        raise AdapterDeltaError("adapter delta codec version is unsupported")
    for key in ("base_sha256", "target_sha256", "adapter_config_sha256", "tensor_metadata_sha256", "delta_sha256"):
        if not _valid_sha(value[key]):
            raise AdapterDeltaError(f"adapter delta manifest has invalid {key}")
    for key in ("uncompressed_size", "delta_size"):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise AdapterDeltaError(f"adapter delta manifest has invalid {key}")
    return value


def select_adapter_transfer(
    manifest: Mapping[str, object] | None,
    *,
    base_available: bool,
    base_sha256: str | None,
    target_sha256: str,
    target_size: int,
    enabled: bool,
    max_ratio: float | None = None,
    full_wire_size: int | None = None,
    manifest_wire_size: int = 0,
) -> str:
    """Choose ``delta`` only when every identity and size gate passes."""
    if not enabled or manifest is None or not base_available:
        return FULL_TRANSFER
    if full_wire_size is not None and (full_wire_size <= 0 or manifest_wire_size < 0):
        return FULL_TRANSFER
    try:
        if (
            manifest.get("base_sha256") != base_sha256
            or manifest.get("target_sha256") != target_sha256
            or manifest.get("uncompressed_size") != target_size
            or not isinstance(manifest.get("delta_size"), int)
            or manifest["delta_size"] / target_size >= _max_ratio(max_ratio)
            or (
                full_wire_size is not None
                and manifest["delta_size"] + manifest_wire_size >= full_wire_size
            )
        ):
            return FULL_TRANSFER
    except (TypeError, ValueError):
        return FULL_TRANSFER
    return DELTA_TRANSFER


def _read_exact(source: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = source.read(remaining)
        if not block:
            raise AdapterDeltaError("adapter delta frame ended early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def apply_adapter_delta(
    base_path: Path,
    delta_path: Path,
    target_path: Path,
    manifest: Mapping[str, object],
) -> None:
    """Apply and verify a delta, replacing only the supplied temp target path."""
    validated = load_adapter_delta_manifest_from_mapping(manifest)
    if sha256_file(base_path) != validated["base_sha256"]:
        raise AdapterDeltaError("adapter delta base SHA-256 does not match")
    if base_path.stat().st_size != validated.get("base_size", validated["uncompressed_size"]):
        raise AdapterDeltaError("adapter delta base size does not match")
    if delta_path.stat().st_size != validated["delta_size"]:
        raise AdapterDeltaError("adapter delta file size does not match manifest")
    if sha256_file(delta_path) != validated["delta_sha256"]:
        raise AdapterDeltaError("adapter delta file SHA-256 does not match manifest")

    temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.assembling")
    _remove_if_exists(temporary)
    digest = hashlib.sha256()
    written = 0
    try:
        with base_path.open("rb") as base, delta_path.open("rb") as delta, temporary.open("xb") as target:
            while True:
                header = delta.read(_FRAME_HEADER.size)
                if not header:
                    break
                if len(header) != _FRAME_HEADER.size:
                    raise AdapterDeltaError("adapter delta frame header ended early")
                raw_size, encoded_size = _FRAME_HEADER.unpack(header)
                if raw_size <= 0 or raw_size > ADAPTER_DELTA_FRAME_BYTES or encoded_size <= 0:
                    raise AdapterDeltaError("adapter delta frame header is invalid")
                encoded = _read_exact(delta, encoded_size)
                try:
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(encoded)
                    raw += decoder.flush()
                except zlib.error as error:
                    raise AdapterDeltaError("adapter delta frame is corrupt") from error
                if (
                    not decoder.eof
                    or decoder.unused_data
                    or decoder.unconsumed_tail
                    or len(raw) != raw_size
                ):
                    raise AdapterDeltaError("adapter delta frame has wrong raw size")
                base_chunk = base.read(raw_size)
                if len(base_chunk) != raw_size:
                    raise AdapterDeltaError("adapter delta is longer than base adapter")
                target_chunk = _xor(base_chunk, raw)
                target.write(target_chunk)
                digest.update(target_chunk)
                written += raw_size
            if delta.read(1):
                raise AdapterDeltaError("adapter delta has trailing data")
            if base.read(1):
                raise AdapterDeltaError("adapter delta is shorter than base adapter")
            target.flush()
            os.fsync(target.fileno())
        if written != validated["uncompressed_size"]:
            raise AdapterDeltaError("reconstructed adapter size does not match manifest")
        if digest.hexdigest() != validated["target_sha256"]:
            raise AdapterDeltaError("reconstructed adapter SHA-256 does not match manifest")
        os.replace(temporary, target_path)
    except BaseException:
        _remove_if_exists(temporary)
        raise


def load_adapter_delta_manifest_from_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Validate an already downloaded manifest without a temporary JSON file."""
    if not isinstance(value, Mapping):
        raise AdapterDeltaError("adapter delta manifest is not an object")
    # Reuse the same checks as the file loader.  Keep this independent of JSON
    # serialization so callers can validate a response before writing files.
    required = (
        "version",
        "codec",
        "codec_version",
        "base_sha256",
        "target_sha256",
        "adapter_config_sha256",
        "tensor_metadata_sha256",
        "uncompressed_size",
        "delta_size",
        "delta_sha256",
    )
    if any(key not in value for key in required):
        raise AdapterDeltaError("adapter delta manifest is missing an identity field")
    if value["version"] != ADAPTER_DELTA_VERSION or value["codec"] != ADAPTER_DELTA_CODEC:
        raise AdapterDeltaError("adapter delta codec is unsupported")
    if value["codec_version"] != ADAPTER_DELTA_VERSION:
        raise AdapterDeltaError("adapter delta codec version is unsupported")
    for key in ("base_sha256", "target_sha256", "adapter_config_sha256", "tensor_metadata_sha256", "delta_sha256"):
        if not _valid_sha(value[key]):
            raise AdapterDeltaError(f"adapter delta manifest has invalid {key}")
    for key in ("uncompressed_size", "delta_size"):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise AdapterDeltaError(f"adapter delta manifest has invalid {key}")
    return dict(value)


# Friendly aliases for scripts and older callers.
build_xor_adapter_delta = build_adapter_delta
apply_xor_adapter_delta = apply_adapter_delta
adapter_delta_transfer = select_adapter_transfer
build_adapter_delta_checkpoint = build_adapter_delta
apply_adapter_delta_file = apply_adapter_delta
