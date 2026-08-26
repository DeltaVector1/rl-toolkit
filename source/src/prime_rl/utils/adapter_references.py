"""Lossless content-addressed transfer for an initial LoRA adapter."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO, Mapping


ADAPTER_REFERENCE_VERSION = 1
ADAPTER_REFERENCE_CODEC = "tensor-references-v1"
ADAPTER_REFERENCE_MANIFEST = "adapter_model.safetensors.references.json"
ADAPTER_REFERENCE_HEADER = "adapter_model.safetensors.references.header"
ADAPTER_REFERENCE_PAYLOAD = "adapter_model.safetensors.references.payload"
ADAPTER_REFERENCE_MAX_HEADER_BYTES = 256 * 1024 * 1024
ADAPTER_REFERENCE_MAX_SEGMENTS = 1_000_000
ADAPTER_REFERENCE_MAX_MANIFEST_BYTES = 512 * 1024 * 1024
ADAPTER_REFERENCE_MAX_OUTPUT_BYTES = 1 << 40
_READ_BYTES = 16 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class AdapterReferenceError(OSError):
    """Raised when a reference package cannot be trusted."""


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(_READ_BYTES), b""):
                digest.update(block)
    except OSError as error:
        raise AdapterReferenceError(f"cannot read {path}") from error
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_header(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise AdapterReferenceError("safetensors header length is truncated")
            length = struct.unpack("<Q", prefix)[0]
            if length <= 0 or length > ADAPTER_REFERENCE_MAX_HEADER_BYTES:
                raise AdapterReferenceError("safetensors header length is invalid")
            body = source.read(length)
            if len(body) != length:
                raise AdapterReferenceError("safetensors header is truncated")
    except OSError as error:
        if isinstance(error, AdapterReferenceError):
            raise
        raise AdapterReferenceError(f"cannot read safetensors header: {path}") from error
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterReferenceError("safetensors header is not valid JSON") from error
    if not isinstance(value, dict):
        raise AdapterReferenceError("safetensors header is not an object")
    payload_size = path.stat().st_size - 8 - length
    if payload_size < 0:
        raise AdapterReferenceError("safetensors payload is missing")
    previous_end = 0
    tensor_ranges: list[tuple[int, int]] = []
    for name, spec in value.items():
        if name == "__metadata__":
            if not isinstance(spec, dict) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in spec.items()
            ):
                raise AdapterReferenceError("safetensors metadata is invalid")
            continue
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise AdapterReferenceError("safetensors tensor metadata is invalid")
        if not isinstance(spec.get("dtype"), str) or not isinstance(spec.get("shape"), list):
            raise AdapterReferenceError("safetensors tensor metadata is incomplete")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int and item >= 0 for item in offsets)
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise AdapterReferenceError("safetensors tensor offsets are invalid")
        tensor_ranges.append((offsets[0], offsets[1]))
    for start, end in sorted(tensor_ranges):
        if start < previous_end:
            raise AdapterReferenceError("safetensors tensor ranges overlap")
        previous_end = end
    return prefix + body, value


def _read_exact(source: BinaryIO, size: int) -> bytes:
    if size < 0:
        raise AdapterReferenceError("reference read size is negative")
    pieces: list[bytes] = []
    remaining = size
    while remaining:
        piece = source.read(min(_READ_BYTES, remaining))
        if not piece:
            raise AdapterReferenceError("reference payload ended early")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _package_paths(target_dir: Path) -> tuple[Path, Path, Path]:
    return (
        target_dir / ADAPTER_REFERENCE_MANIFEST,
        target_dir / ADAPTER_REFERENCE_HEADER,
        target_dir / ADAPTER_REFERENCE_PAYLOAD,
    )


def adapter_reference_identity(model_path: Path, config_path: Path) -> dict[str, object]:
    header, parsed = _read_header(model_path)
    return {
        "target_sha256": _sha256_file(model_path),
        "adapter_config_sha256": _sha256_file(config_path),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "tensor_metadata_sha256": _canonical_json_sha256(parsed),
        "target_size": model_path.stat().st_size,
        "header_size": len(header),
        "payload_size": model_path.stat().st_size - len(header),
    }


def build_adapter_references(
    target_dir: Path,
    *,
    target_url: str = "",
    max_payload_ratio: float = 0.95,
) -> Path | None:
    """Build one deduplicated payload package beside an adapter export.

    Each tensor and each non-tensor gap maps to one exact byte range in a
    content-addressed payload file.  The header is copied byte-for-byte.
    """
    if not 0.0 < max_payload_ratio < 1.0:
        raise ValueError("reference payload ratio must be between zero and one")
    manifest_path, header_path, payload_path = _package_paths(target_dir)
    model_path = target_dir / "adapter_model.safetensors"
    config_path = target_dir / "adapter_config.json"
    if not model_path.is_file() or not config_path.is_file():
        for path in (manifest_path, header_path, payload_path):
            _remove(path)
        return None
    header, parsed = _read_header(model_path)
    identity = adapter_reference_identity(model_path, config_path)
    payload_size = int(identity["payload_size"])
    tensors = sorted(
        (
            int(spec["data_offsets"][0]),
            int(spec["data_offsets"][1]),
        )
        for name, spec in parsed.items()
        if name != "__metadata__"
    )
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in tensors:
        if start > cursor:
            ranges.append((cursor, start))
        if end > start:
            ranges.append((start, end))
        cursor = max(cursor, end)
    if cursor < payload_size:
        ranges.append((cursor, payload_size))
    if len(ranges) > ADAPTER_REFERENCE_MAX_SEGMENTS:
        raise AdapterReferenceError("reference segment count is too large")

    temporary_payload = payload_path.with_name(f".{payload_path.name}.{os.getpid()}.partial")
    temporary_header = header_path.with_name(f".{header_path.name}.{os.getpid()}.partial")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.partial")
    for path in (temporary_payload, temporary_header, temporary_manifest):
        _remove(path)
    references: dict[tuple[int, str], tuple[int, int]] = {}
    segments: list[dict[str, object]] = []
    unique_bytes = 0
    try:
        with model_path.open("rb") as source, temporary_payload.open("xb") as payload:
            source.seek(len(header))
            for start, end in ranges:
                source.seek(len(header) + start)
                raw = _read_exact(source, end - start)
                digest = hashlib.sha256(raw).hexdigest()
                key = (len(raw), digest)
                reference = references.get(key)
                if reference is None:
                    reference = (unique_bytes, len(raw))
                    references[key] = reference
                    payload.write(raw)
                    unique_bytes += len(raw)
                segments.append(
                    {
                        "file_start": start,
                        "size": len(raw),
                        "payload_start": reference[0],
                        "sha256": digest,
                    }
                )
            payload.flush()
            os.fsync(payload.fileno())
        if unique_bytes / max(1, payload_size) >= max_payload_ratio:
            for path in (temporary_payload, temporary_header, temporary_manifest):
                _remove(path)
            for path in (manifest_path, header_path, payload_path):
                _remove(path)
            return None
        with temporary_header.open("xb") as output:
            output.write(header)
            output.flush()
            os.fsync(output.fileno())
        manifest: dict[str, object] = {
            "version": ADAPTER_REFERENCE_VERSION,
            "codec": ADAPTER_REFERENCE_CODEC,
            "target_url": target_url,
            "target_sha256": identity["target_sha256"],
            "adapter_config_sha256": identity["adapter_config_sha256"],
            "header_sha256": identity["header_sha256"],
            "tensor_metadata_sha256": identity["tensor_metadata_sha256"],
            "target_size": identity["target_size"],
            "header_size": identity["header_size"],
            "payload_size": identity["payload_size"],
            "unique_payload_size": unique_bytes,
            "payload_sha256": _sha256_file(temporary_payload),
            "segments": segments,
        }
        with temporary_manifest.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_payload, payload_path)
        os.replace(temporary_header, header_path)
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        for path in (temporary_payload, temporary_header, temporary_manifest):
            _remove(path)
        raise
    return manifest_path


def load_adapter_reference_manifest(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > ADAPTER_REFERENCE_MAX_MANIFEST_BYTES:
            raise AdapterReferenceError("reference manifest is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if isinstance(error, AdapterReferenceError):
            raise
        raise AdapterReferenceError(f"cannot read reference manifest: {path}") from error
    return validate_adapter_reference_manifest(value)


def validate_adapter_reference_manifest(value: Mapping[str, object]) -> dict[str, object]:
    required = (
        "version",
        "codec",
        "target_sha256",
        "adapter_config_sha256",
        "header_sha256",
        "tensor_metadata_sha256",
        "target_size",
        "header_size",
        "payload_size",
        "unique_payload_size",
        "payload_sha256",
        "segments",
    )
    if not isinstance(value, Mapping) or any(key not in value for key in required):
        raise AdapterReferenceError("reference manifest is missing an identity field")
    if value["version"] != ADAPTER_REFERENCE_VERSION or value["codec"] != ADAPTER_REFERENCE_CODEC:
        raise AdapterReferenceError("reference codec is unsupported")
    for key in (
        "target_sha256",
        "adapter_config_sha256",
        "header_sha256",
        "tensor_metadata_sha256",
        "payload_sha256",
    ):
        if not _valid_sha(value[key]):
            raise AdapterReferenceError(f"reference manifest has invalid {key}")
    for key in ("target_size", "header_size", "payload_size", "unique_payload_size"):
        if type(value[key]) is not int or value[key] < 0:
            raise AdapterReferenceError(f"reference manifest has invalid {key}")
    if (
        value["header_size"] < 8
        or value["header_size"] > ADAPTER_REFERENCE_MAX_HEADER_BYTES
        or value["target_size"] > ADAPTER_REFERENCE_MAX_OUTPUT_BYTES
        or value["target_size"] != value["header_size"] + value["payload_size"]
        or value["unique_payload_size"] > value["payload_size"]
    ):
        raise AdapterReferenceError("reference manifest sizes are inconsistent")
    segments = value["segments"]
    if not isinstance(segments, list) or not segments or len(segments) > ADAPTER_REFERENCE_MAX_SEGMENTS:
        raise AdapterReferenceError("reference manifest segments are invalid")
    expected_file_start = 0
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise AdapterReferenceError("reference segment is invalid")
        if any(key not in segment for key in ("file_start", "size", "payload_start", "sha256")):
            raise AdapterReferenceError("reference segment is incomplete")
        if (
            type(segment["file_start"]) is not int
            or type(segment["size"]) is not int
            or type(segment["payload_start"]) is not int
            or segment["file_start"] != expected_file_start
            or segment["size"] <= 0
            or segment["payload_start"] < 0
            or segment["payload_start"] + segment["size"] > value["unique_payload_size"]
            or not _valid_sha(segment["sha256"])
        ):
            raise AdapterReferenceError("reference segment bounds are invalid")
        expected_file_start += segment["size"]
    if expected_file_start != value["payload_size"]:
        raise AdapterReferenceError("reference segments do not cover the payload")
    return dict(value)


def apply_adapter_references(
    header_path: Path,
    payload_path: Path,
    target_path: Path,
    manifest: Mapping[str, object],
) -> None:
    """Reconstruct and verify a target, publishing it only after all checks."""
    checked = validate_adapter_reference_manifest(manifest)
    if header_path.stat().st_size != checked["header_size"] or _sha256_file(header_path) != checked["header_sha256"]:
        raise AdapterReferenceError("reference header identity differs")
    if payload_path.stat().st_size != checked["unique_payload_size"] or _sha256_file(payload_path) != checked["payload_sha256"]:
        raise AdapterReferenceError("reference payload identity differs")
    temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.assembling")
    _remove(temporary)
    digest = hashlib.sha256()
    try:
        with header_path.open("rb") as header, payload_path.open("rb") as payload, temporary.open("xb") as target:
            header_bytes = _read_exact(header, checked["header_size"])
            target.write(header_bytes)
            digest.update(header_bytes)
            for segment in checked["segments"]:
                payload.seek(segment["payload_start"])
                raw = _read_exact(payload, segment["size"])
                if hashlib.sha256(raw).hexdigest() != segment["sha256"]:
                    raise AdapterReferenceError("reference segment identity differs")
                target.write(raw)
                digest.update(raw)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != checked["target_size"] or digest.hexdigest() != checked["target_sha256"]:
            raise AdapterReferenceError("reconstructed adapter identity differs")
        os.replace(temporary, target_path)
    except BaseException:
        _remove(temporary)
        raise

