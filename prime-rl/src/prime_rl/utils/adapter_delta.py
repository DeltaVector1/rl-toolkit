"""Exact, chunked transport artifacts for consecutive LoRA adapters."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


adapter_manifest_name = "adapter_manifest.json"
adapter_config_name = "adapter_config.json"
adapter_weights_name = "adapter_model.safetensors"
adapter_transport_name = "chunked-shuffled-xor"
default_chunk_size = 16 * 1024 * 1024
default_workers = 8
max_workers_limit = 8
io_chunk_size = 16 * 1024 * 1024


def _load_zstd():
    library_name = ctypes.util.find_library("zstd") or "libzstd.so.1"
    try:
        library = ctypes.CDLL(library_name)
    except OSError as exc:
        raise RuntimeError("libzstd is required for adapter transport") from exc
    size = ctypes.c_size_t
    library.ZSTD_compressBound.argtypes = [size]
    library.ZSTD_compressBound.restype = size
    library.ZSTD_compress.argtypes = [ctypes.c_void_p, size, ctypes.c_void_p, size, ctypes.c_int]
    library.ZSTD_compress.restype = size
    library.ZSTD_isError.argtypes = [size]
    library.ZSTD_isError.restype = ctypes.c_uint
    library.ZSTD_getErrorName.argtypes = [size]
    library.ZSTD_getErrorName.restype = ctypes.c_char_p
    library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, size]
    library.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    library.ZSTD_decompress.argtypes = [ctypes.c_void_p, size, ctypes.c_void_p, size]
    library.ZSTD_decompress.restype = size
    return library


_zstd = _load_zstd()


def _codec_name() -> str:
    return "zstd"


def _compress(value: bytes) -> bytes:
    source = ctypes.create_string_buffer(value, max(1, len(value)))
    destination = ctypes.create_string_buffer(_zstd.ZSTD_compressBound(len(value)))
    result = _zstd.ZSTD_compress(destination, len(destination), source, len(value), 1)
    if _zstd.ZSTD_isError(result):
        error = _zstd.ZSTD_getErrorName(result).decode("utf-8", "replace")
        raise OSError(f"zstd compression failed: {error}")
    return bytes(destination[:result])


def _decompress(value: bytes, codec: str) -> bytes:
    if codec.startswith(adapter_transport_name):
        if "zstd" not in codec:
            raise ValueError("adapter transport must use zstd")
        codec = "zstd"
    if codec == "zstd":
        source = ctypes.create_string_buffer(value, max(1, len(value)))
        content_size = _zstd.ZSTD_getFrameContentSize(source, len(value))
        if content_size >= (1 << 64) - 2:
            raise OSError("zstd payload does not contain its decompressed size")
        destination = ctypes.create_string_buffer(max(1, content_size))
        result = _zstd.ZSTD_decompress(destination, content_size, source, len(value))
        if _zstd.ZSTD_isError(result):
            error = _zstd.ZSTD_getErrorName(result).decode("utf-8", "replace")
            raise OSError(f"zstd decompression failed: {error}")
        if result != content_size:
            raise OSError(f"zstd size mismatch: {result} != {content_size}")
        return bytes(destination[:result])
    raise ValueError(f"unsupported adapter transport codec: {codec}")


def _xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("adapter chunks must have equal sizes")
    return np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8),
        np.frombuffer(right, dtype=np.uint8),
    ).tobytes()


def shuffle_bf16_bytes(value: bytes) -> bytes:
    """Group the two bytes of each BF16 value by byte position."""
    paired_size = len(value) - (len(value) % 2)
    if paired_size == 0:
        return value
    pairs = np.frombuffer(value[:paired_size], dtype=np.uint8).reshape(-1, 2)
    shuffled = np.concatenate((pairs[:, 0], pairs[:, 1]))
    if paired_size != len(value):
        return shuffled.tobytes() + value[paired_size:]
    return shuffled.tobytes()


def unshuffle_bf16_bytes(value: bytes) -> bytes:
    paired_size = len(value) - (len(value) % 2)
    if paired_size == 0:
        return value
    pair_count = paired_size // 2
    shuffled = np.frombuffer(value[:paired_size], dtype=np.uint8)
    pairs = np.empty((pair_count, 2), dtype=np.uint8)
    pairs[:, 0] = shuffled[:pair_count]
    pairs[:, 1] = shuffled[pair_count:]
    restored = pairs.reshape(-1).tobytes()
    if paired_size != len(value):
        return restored + value[paired_size:]
    return restored


def encode_delta_chunk(base: bytes, target: bytes) -> bytes:
    """XOR equal-position bytes, then apply the two-byte BF16 shuffle."""
    return _compress(shuffle_bf16_bytes(_xor(base, target)))


def _encode_delta_chunk(base: bytes, target: bytes) -> tuple[bytes, bytes]:
    raw = shuffle_bf16_bytes(_xor(base, target))
    return raw, _compress(raw)


def decode_adapter_chunk(payload: bytes, *, codec: str, max_output_size: int) -> bytes:
    if max_output_size <= 0:
        raise ValueError("adapter chunk output limit must be positive")
    decoded = unshuffle_bf16_bytes(_decompress(payload, codec))
    if len(decoded) != max_output_size:
        raise ValueError("adapter chunk does not match its declared size")
    return decoded


def decode_delta_chunk(base: bytes, encoded: bytes, codec: str) -> bytes:
    return _xor(base, decode_adapter_chunk(encoded, codec=codec, max_output_size=len(base)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(io_chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_ranges(size: int, chunk_size: int) -> list[tuple[int, int]]:
    if size < 0:
        raise ValueError("adapter file size cannot be negative")
    if chunk_size <= 0 or chunk_size % 2:
        raise ValueError("adapter chunk size must be a positive even number")
    return [(offset, min(offset + chunk_size, size)) for offset in range(0, size, chunk_size)]


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _remove_transport_chunks(directory: Path) -> None:
    for prefix in (f"{adapter_config_name}.transport-", f"{adapter_weights_name}.transport-"):
        for path in directory.glob(f"{prefix}*"):
            if path.is_file():
                path.unlink()


def _write_encoded_chunks(
    target_path: Path,
    base_path: Path | None,
    destination: Path,
    *,
    mode: str,
    chunk_size: int,
    max_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    target_size = target_path.stat().st_size
    ranges = _chunk_ranges(target_size, chunk_size)
    if base_path is not None and base_path.stat().st_size != target_size:
        raise ValueError("adjacent adapter files must have the same size")

    codec = _codec_name() if mode == "delta" else "identity"
    chunks: list[dict[str, Any]] = []
    pending: dict[int, Future[Any]] = {}
    written_size = 0
    target = target_path.open("rb")
    base = base_path.open("rb") if base_path is not None else None

    def save_chunk(index: int, raw: bytes, payload: bytes, offset: int, raw_size: int) -> None:
        name = f"{target_path.name}.transport-{mode}-{index:06d}"
        path = destination / name
        _atomic_write(path, payload)
        chunks.append(
            {
                "file": name,
                "offset": offset,
                "raw_size": raw_size,
                "wire_size": len(payload),
                "wire_sha256": hashlib.sha256(payload).hexdigest(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "codec": codec,
            }
        )

    def drain(index: int, start: int, end: int) -> None:
        nonlocal written_size
        result = pending.pop(index).result()
        if mode == "delta":
            raw, payload = result
        else:
            raw = payload = result
        save_chunk(index, raw, payload, start, end - start)
        written_size += len(payload)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for index, (start, end) in enumerate(ranges):
                target_chunk = target.read(end - start)
                if len(target_chunk) != end - start:
                    raise OSError(f"adapter file ended early: {target_path.name}")
                base_chunk = base.read(end - start) if base is not None else None
                if base is not None and len(base_chunk) != end - start:
                    raise OSError(f"base adapter file ended early: {base_path.name}")
                if mode == "delta":
                    assert base_chunk is not None
                    pending[index] = pool.submit(_encode_delta_chunk, base_chunk, target_chunk)
                else:
                    pending[index] = pool.submit(bytes, target_chunk)
                if len(pending) >= max_workers:
                    drain(index - len(pending) + 1, *ranges[index - len(pending) + 1])
            for index, (start, end) in enumerate(ranges):
                if index in pending:
                    drain(index, start, end)
    finally:
        target.close()
        if base is not None:
            base.close()

    chunks.sort(key=lambda item: int(item["offset"]))
    return chunks, written_size


def _file_record(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = sha256_file(path)
    return {"name": path.name, "size": size, "sha256": digest}


def publish_adapter_manifest(
    target_dir: Path,
    base_dir: Path | None = None,
    *,
    target_url: str | None = None,
    base_url: str | None = None,
    target_step: int | None = None,
    base_step: int | None = None,
    chunk_size: int = default_chunk_size,
    max_workers: int = default_workers,
) -> dict[str, Any]:
    """Publish full or exact delta chunks for one adapter directory.

    The normal adapter files remain in place. The manifest and transport
    chunks are published only after all chunks have been written.
    """
    if max_workers <= 0 or max_workers > max_workers_limit:
        raise ValueError(f"adapter worker count must be between 1 and {max_workers_limit}")
    build_started = time.perf_counter()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_config = target_dir / adapter_config_name
    target_weights = target_dir / adapter_weights_name
    if not target_config.is_file() or not target_weights.is_file():
        raise FileNotFoundError("adapter directory must contain config and weights")

    _remove_transport_chunks(target_dir)
    usable_base = (
        base_dir is not None
        and (base_dir / adapter_weights_name).is_file()
        and (base_dir / adapter_weights_name).stat().st_size == target_weights.stat().st_size
    )
    base_dir = base_dir.resolve() if usable_base and base_dir is not None else None
    if base_dir is None:
        base_url = None
        base_step = None
    selected_mode = "delta" if base_dir is not None else "full"

    if selected_mode == "delta":
        weight_chunks, weight_encoded_size = _write_encoded_chunks(
            target_weights,
            base_dir / adapter_weights_name,
            target_dir,
            mode=selected_mode,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )
    else:
        weight_chunks = []
        weight_encoded_size = target_weights.stat().st_size
    if selected_mode == "delta" and weight_encoded_size >= target_weights.stat().st_size:
        _remove_transport_chunks(target_dir)
        selected_mode = "full"
        weight_chunks = []
        weight_encoded_size = target_weights.stat().st_size
        base_dir = None
        base_url = None
        base_step = None

    files = [
        _file_record(target_config),
        _file_record(target_weights),
    ]
    weights_record = files[1]
    manifest: dict[str, Any] = {
        "version": 1,
        "kind": "lora-adapter",
        "mode": selected_mode,
        "target_step": target_step,
        "base_step": base_step,
        "full_size": weights_record["size"],
        "full_sha256": weights_record["sha256"],
        "target_size": weights_record["size"],
        "target_sha256": weights_record["sha256"],
        "config_size": files[0]["size"],
        "config_sha256": files[0]["sha256"],
        "encoded_size": weight_encoded_size,
        "chunk_size": chunk_size,
        "chunk_count": len(weight_chunks),
        "build_seconds": round(time.perf_counter() - build_started, 3),
        "files": files,
    }
    if target_url is not None:
        manifest["target_url"] = target_url
    if base_url is not None:
        manifest["base_url"] = base_url
    if selected_mode == "delta":
        manifest["transport"] = f"{adapter_transport_name}-{_codec_name()}"
        manifest["chunks"] = [
            {
                "file": item["file"],
                "offset": item["offset"],
                "raw_size": item["raw_size"],
                "wire_size": item["wire_size"],
                "wire_sha256": item["wire_sha256"],
                "raw_sha256": item["raw_sha256"],
                "codec": item["codec"],
            }
            for item in weight_chunks
        ]
    if base_dir is not None:
        manifest["base_size"] = (base_dir / adapter_weights_name).stat().st_size
        manifest["base_sha256"] = sha256_file(base_dir / adapter_weights_name)
        base_config = base_dir / adapter_config_name
        if base_config.is_file():
            manifest["base_config_sha256"] = sha256_file(base_config)
    _atomic_write(
        target_dir / f"{adapter_weights_name}.sha256",
        f"{weights_record['sha256']}\n".encode("ascii"),
    )
    _atomic_json(target_dir / adapter_manifest_name, manifest)
    return manifest
