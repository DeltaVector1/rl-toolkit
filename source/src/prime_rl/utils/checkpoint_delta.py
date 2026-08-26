"""Lossless XOR deltas for immutable sharded safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np


DELTA_MANIFEST = "xor-delta-v1.json"
IO_CHUNK = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(IO_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise OSError("checkpoint delta inputs have different lengths")
    return np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8),
        np.frombuffer(right, dtype=np.uint8),
    ).tobytes()


def build_xor_delta_checkpoint(
    base_dir: Path,
    target_dir: Path,
    *,
    base_url: str,
    target_url: str,
) -> Path | None:
    """Build compressed byte-exact deltas when two checkpoints have the same shards."""
    base_index_path = base_dir / "model.safetensors.index.json"
    target_index_path = target_dir / "model.safetensors.index.json"
    if not base_index_path.is_file() or not target_index_path.is_file():
        return None

    with base_index_path.open("r", encoding="utf-8") as source:
        base_index = json.load(source)
    with target_index_path.open("r", encoding="utf-8") as source:
        target_index = json.load(source)
    base_shards = sorted(set(base_index.get("weight_map", {}).values()))
    target_shards = sorted(set(target_index.get("weight_map", {}).values()))
    if not target_shards or base_shards != target_shards:
        return None

    manifest_path = target_dir / DELTA_MANIFEST
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as source:
                existing = json.load(source)
            if existing.get("base_url") == base_url and existing.get("target_url") == target_url:
                return manifest_path
        except (OSError, ValueError):
            pass

    files = []
    for filename in target_shards:
        base_path = base_dir / filename
        target_path = target_dir / filename
        if not base_path.is_file() or not target_path.is_file():
            return None
        if base_path.stat().st_size != target_path.stat().st_size:
            return None

        delta_name = f"{filename}.xor.zlib"
        delta_path = target_dir / delta_name
        temporary = target_dir / f"{delta_name}.partial"
        compressor = zlib.compressobj(level=1)
        target_digest = hashlib.sha256()
        with (
            base_path.open("rb") as base,
            target_path.open("rb") as target,
            temporary.open("wb") as output,
        ):
            while target_chunk := target.read(IO_CHUNK):
                base_chunk = base.read(len(target_chunk))
                if len(base_chunk) != len(target_chunk):
                    raise OSError(f"base checkpoint ended early: {filename}")
                target_digest.update(target_chunk)
                output.write(compressor.compress(_xor_bytes(base_chunk, target_chunk)))
            if base.read(1):
                raise OSError(f"base checkpoint is longer than target: {filename}")
            output.write(compressor.flush())
        temporary.replace(delta_path)
        files.append(
            {
                "filename": filename,
                "delta": delta_name,
                "size": target_path.stat().st_size,
                "sha256": target_digest.hexdigest(),
            }
        )

    manifest = {
        "version": 1,
        "base_url": base_url,
        "target_url": target_url,
        "index_sha256": sha256_file(target_index_path),
        "files": files,
    }
    temporary_manifest = target_dir / f"{DELTA_MANIFEST}.partial"
    temporary_manifest.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


def apply_xor_delta_file(
    base_path: Path,
    delta_path: Path,
    target_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Reconstruct one target file without materializing an uncompressed delta."""
    temporary = target_path.with_name(f"{target_path.name}.assembling")
    decompressor = zlib.decompressobj()
    digest = hashlib.sha256()
    written = 0

    with base_path.open("rb") as base, delta_path.open("rb") as delta, temporary.open("wb") as target:
        def consume(delta_chunk: bytes) -> None:
            nonlocal written
            base_chunk = base.read(len(delta_chunk))
            if len(base_chunk) != len(delta_chunk):
                raise OSError(f"delta is longer than base file: {base_path.name}")
            target_chunk = _xor_bytes(base_chunk, delta_chunk)
            target.write(target_chunk)
            digest.update(target_chunk)
            written += len(target_chunk)

        while compressed := delta.read(1024 * 1024):
            pending = compressed
            while pending:
                expanded = decompressor.decompress(pending, IO_CHUNK)
                pending = decompressor.unconsumed_tail
                if expanded:
                    consume(expanded)
                elif not pending:
                    break
            while True:
                expanded = decompressor.decompress(b"", IO_CHUNK)
                if not expanded:
                    break
                consume(expanded)

        tail = decompressor.flush()
        if tail:
            consume(tail)
        if base.read(1):
            raise OSError(f"delta is shorter than base file: {base_path.name}")

    if not decompressor.eof:
        raise OSError(f"incomplete compressed delta: {delta_path.name}")
    if written != expected_size:
        raise OSError(f"reconstructed size mismatch for {target_path.name}: {written} != {expected_size}")
    if digest.hexdigest() != expected_sha256:
        raise OSError(f"reconstructed checksum mismatch for {target_path.name}")
    temporary.replace(target_path)


# Adapter exports use the adjacent-step protocol in ``adapter_delta``.  Keep
# these imports here so deployment code that already imports checkpoint-delta
# helpers can opt in without creating a second module path.
from prime_rl.utils.adapter_delta import (  # noqa: E402  (compatibility exports)
    ADAPTER_DELTA_CODEC,
    ADAPTER_DELTA_FILE,
    ADAPTER_DELTA_FILE_NAME,
    ADAPTER_DELTA_MANIFEST,
    ADAPTER_DELTA_MANIFEST_NAME,
    ADAPTER_DELTA_VERSION,
    apply_adapter_delta,
    apply_adapter_delta_file,
    apply_xor_adapter_delta,
    build_adapter_delta,
    build_adapter_delta_checkpoint,
    build_xor_adapter_delta,
    select_adapter_transfer,
    wire_size_break_even,
)
