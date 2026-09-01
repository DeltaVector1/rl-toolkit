"""Exact, chunked transport artifacts for consecutive LoRA adapters."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import struct
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


adapter_manifest_name = "adapter_manifest.json"
adapter_config_name = "adapter_config.json"
adapter_weights_name = "adapter_model.safetensors"
adapter_transport_name = "chunked-shuffled-xor"
predictive_transport_name = "chunked-bf16-linear-zigzag-sparse-zstd"
default_chunk_size = 16 * 1024 * 1024
default_workers = 8
max_workers_limit = 8
io_chunk_size = 16 * 1024 * 1024

_predictive_magic = b"PRD2"
_predictive_version = 2
_predictive_flags = 0
_predictive_header = struct.Struct("<4sBBHQQQQQQQ")
_predictive_header_size = _predictive_header.size
_predictive_max_varint_bytes = 3
_predictive_max_raw_size = 256 * 1024 * 1024


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


def _decompressed_size(value: bytes) -> int:
    source = ctypes.create_string_buffer(value, max(1, len(value)))
    content_size = _zstd.ZSTD_getFrameContentSize(source, len(value))
    if content_size >= (1 << 64) - 2:
        raise OSError("zstd payload does not contain its decompressed size")
    return int(content_size)


def _decompress_bounded(value: bytes, expected_size: int, *, maximum: int) -> bytes:
    if expected_size < 0 or expected_size > maximum:
        raise ValueError("compressed payload size is outside the permitted range")
    content_size = _decompressed_size(value)
    if content_size != expected_size:
        raise OSError(f"zstd size mismatch: {content_size} != {expected_size}")
    source = ctypes.create_string_buffer(value, max(1, len(value)))
    destination = ctypes.create_string_buffer(max(1, expected_size))
    result = _zstd.ZSTD_decompress(destination, expected_size, source, len(value))
    if _zstd.ZSTD_isError(result):
        error = _zstd.ZSTD_getErrorName(result).decode("utf-8", "replace")
        raise OSError(f"zstd decompression failed: {error}")
    if result != expected_size:
        raise OSError(f"zstd size mismatch: {result} != {expected_size}")
    return bytes(destination[:result])


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


def _signed_residual(value: int, modulus: int) -> int:
    value %= modulus
    midpoint = modulus // 2
    return value if value < midpoint else value - modulus


def _zigzag(value: int) -> int:
    return value * 2 if value >= 0 else -value * 2 - 1


def _unzigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _encode_leb128(value: int) -> bytes:
    if value < 0 or value > 0xFFFF:
        raise ValueError("predictive residual is outside the uint16 range")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _decode_leb128_values(value: bytes, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("predictive nonzero count cannot be negative")
    output = np.empty(count, dtype=np.uint16)
    position = 0
    for index in range(count):
        decoded = 0
        length = 0
        while True:
            if position >= len(value) or length >= _predictive_max_varint_bytes:
                raise ValueError("predictive values contain a truncated or oversized varint")
            byte = value[position]
            position += 1
            decoded |= (byte & 0x7F) << (7 * length)
            length += 1
            if not byte & 0x80:
                break
        if length > 1 and decoded < (1 << (7 * (length - 1))):
            raise ValueError("predictive values contain a non-canonical varint")
        if decoded > 0xFFFF:
            raise ValueError("predictive value exceeds uint16")
        output[index] = decoded
    if position != len(value):
        raise ValueError("predictive values contain trailing bytes")
    return output


def _round_float32_to_bf16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding) >> np.uint32(16)
    nan = ((bits >> np.uint32(23)) & np.uint32(0xFF)) == np.uint32(0xFF)
    nan &= (bits & np.uint32(0x7FFFFF)) != np.uint32(0)
    if np.any(nan):
        rounded[nan] = (bits[nan] >> np.uint32(16)) | np.uint32(1)
    return rounded.astype("<u2", copy=False)


def _bf16_predictor(base2: bytes, base1: bytes) -> np.ndarray:
    if len(base1) != len(base2) or len(base1) % 2:
        raise ValueError("BF16 predictive data must have an even, equal size")
    first = np.frombuffer(base1, dtype="<u2").astype(np.uint32)
    second = np.frombuffer(base2, dtype="<u2").astype(np.uint32)
    first_float = (first << np.uint32(16)).view(np.float32)
    second_float = (second << np.uint32(16)).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = np.float32(2.0) * first_float - second_float
    return _round_float32_to_bf16(predicted)


def _predictive_layout(raw_size: int, global_offset: int, data_offset: int) -> tuple[int, int, int]:
    if raw_size < 0 or global_offset < 0 or data_offset < 0:
        raise ValueError("predictive offsets and sizes must be non-negative")
    end = global_offset + raw_size
    prefix_end = min(end, data_offset)
    prefix_size = max(0, prefix_end - global_offset)
    data_start = max(global_offset, data_offset)
    data_size = max(0, end - data_start)
    if data_size and ((data_start - data_offset) % 2 or data_size % 2):
        raise ValueError("predictive BF16 chunks must cover complete aligned values")
    return prefix_size, data_start, data_size // 2


def _predictive_units(
    base2: bytes,
    base1: bytes,
    target: bytes,
    *,
    global_offset: int,
    data_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(base2) == len(base1) == len(target)):
        raise ValueError("predictive chunks must have equal sizes")
    prefix_size, data_start, data_words = _predictive_layout(
        len(target), global_offset, data_offset
    )
    residual = np.empty(prefix_size + data_words, dtype=np.uint16)
    widths = np.empty_like(residual)
    if prefix_size:
        header_base = np.frombuffer(base1[:prefix_size], dtype=np.uint8).astype(np.uint16)
        header_target = np.frombuffer(target[:prefix_size], dtype=np.uint8).astype(np.uint16)
        residual[:prefix_size] = (header_target - header_base) & np.uint16(0xFF)
        widths[:prefix_size] = np.uint16(8)
    if data_words:
        local_data_start = data_start - global_offset
        local_data_end = local_data_start + data_words * 2
        base1_words = np.frombuffer(base1[local_data_start:local_data_end], dtype="<u2")
        target_words = np.frombuffer(target[local_data_start:local_data_end], dtype="<u2").astype(np.uint32)
        predictor = _bf16_predictor(
            base2[local_data_start:local_data_end],
            base1[local_data_start:local_data_end],
        ).astype(np.uint32)
        residual[prefix_size:] = (target_words - predictor) & np.uint32(0xFFFF)
        widths[prefix_size:] = np.uint16(16)
    return residual, widths


def _predictive_canonical_values(residual: np.ndarray, widths: np.ndarray) -> tuple[bytes, bytes, int]:
    nonzero = residual != 0
    mask = np.packbits(nonzero, bitorder="little").tobytes()
    values = bytearray()
    for number, width in zip(residual[nonzero], widths[nonzero], strict=True):
        modulus = 1 << int(width)
        values.extend(_encode_leb128(_zigzag(_signed_residual(int(number), modulus))))
    return mask, bytes(values), int(nonzero.sum())


def encode_predictive_chunk(
    base2: bytes,
    base1: bytes,
    target: bytes,
    global_offset: int,
    data_offset: int,
) -> bytes:
    """Encode one exact BF16 safetensors chunk against two adjacent bases."""
    residual, widths = _predictive_units(
        base2,
        base1,
        target,
        global_offset=global_offset,
        data_offset=data_offset,
    )
    mask, values, nonzero_count = _predictive_canonical_values(residual, widths)
    mask_payload = _compress(mask)
    values_payload = _compress(values)
    header = _predictive_header.pack(
        _predictive_magic,
        _predictive_version,
        _predictive_flags,
        _predictive_header_size,
        len(target),
        global_offset,
        data_offset,
        len(residual),
        nonzero_count,
        len(mask_payload),
        len(values_payload),
    )
    return header + mask_payload + values_payload


def _encode_predictive_chunk(
    base2: bytes,
    base1: bytes,
    target: bytes,
    global_offset: int,
    data_offset: int,
) -> tuple[bytes, bytes]:
    residual, widths = _predictive_units(
        base2,
        base1,
        target,
        global_offset=global_offset,
        data_offset=data_offset,
    )
    mask, values, nonzero_count = _predictive_canonical_values(residual, widths)
    mask_payload = _compress(mask)
    values_payload = _compress(values)
    header = _predictive_header.pack(
        _predictive_magic,
        _predictive_version,
        _predictive_flags,
        _predictive_header_size,
        len(target),
        global_offset,
        data_offset,
        len(residual),
        nonzero_count,
        len(mask_payload),
        len(values_payload),
    )
    return values, header + mask_payload + values_payload


def _predictive_payload_parts(payload: bytes) -> tuple[tuple[int, ...], bytes, bytes]:
    if len(payload) < _predictive_header_size:
        raise ValueError("predictive payload is shorter than its fixed header")
    header = _predictive_header.unpack(payload[:_predictive_header_size])
    magic, version, flags, header_size = header[:4]
    if magic != _predictive_magic or version != _predictive_version:
        raise ValueError("predictive payload identity is invalid")
    if flags != _predictive_flags or header_size != _predictive_header_size:
        raise ValueError("predictive payload header is invalid")
    raw_size, unit_count, nonzero_count = header[4], header[7], header[8]
    if raw_size > _predictive_max_raw_size or unit_count > raw_size or nonzero_count > unit_count:
        raise ValueError("predictive payload counts are outside the permitted range")
    mask_size, values_size = header[-2:]
    if mask_size <= 0 or values_size <= 0:
        raise ValueError("predictive payload sections cannot be empty")
    expected_size = _predictive_header_size + mask_size + values_size
    if expected_size != len(payload):
        raise ValueError("predictive payload has an invalid size")
    mask_start = _predictive_header_size
    return header, payload[mask_start : mask_start + mask_size], payload[mask_start + mask_size :]


def _predictive_mask_indices(mask: bytes, count: int, nonzero_count: int) -> np.ndarray:
    if count < 0 or nonzero_count < 0 or nonzero_count > count:
        raise ValueError("predictive counts are invalid")
    expected_size = (count + 7) // 8
    if len(mask) != expected_size:
        raise ValueError("predictive mask size is invalid")
    if count % 8 and mask[-1] & (0xFF << (count % 8)):
        raise ValueError("predictive mask has nonzero padding bits")
    indices = np.flatnonzero(np.unpackbits(np.frombuffer(mask, dtype=np.uint8), bitorder="little")[:count])
    if len(indices) != nonzero_count:
        raise ValueError("predictive mask count is invalid")
    return indices


def decode_predictive_chunk(
    base2: bytes,
    base1: bytes,
    payload: bytes,
    global_offset: int,
    data_offset: int,
    *,
    max_output_size: int | None = None,
) -> bytes:
    """Decode one predictive chunk without trusting unbounded payload sizes."""
    header, mask_payload, values_payload = _predictive_payload_parts(payload)
    _, _, _, _, raw_size, payload_offset, payload_data_offset, unit_count, nonzero_count, _, _ = header
    if payload_offset != global_offset or payload_data_offset != data_offset:
        raise ValueError("predictive payload offsets do not match the requested chunk")
    if len(base1) != len(base2) or len(base1) != raw_size:
        raise ValueError("predictive bases do not match the declared chunk size")
    if max_output_size is not None and (max_output_size < 0 or raw_size > max_output_size):
        raise ValueError("predictive output exceeds its caller limit")
    prefix_size, data_start, data_words = _predictive_layout(raw_size, global_offset, data_offset)
    expected_units = prefix_size + data_words
    if unit_count != expected_units:
        raise ValueError("predictive unit count does not match the chunk layout")
    mask_size = (unit_count + 7) // 8
    mask = _decompress_bounded(mask_payload, mask_size, maximum=mask_size)
    maximum_values_size = nonzero_count * _predictive_max_varint_bytes
    values_size = _decompressed_size(values_payload)
    if values_size < nonzero_count or values_size > maximum_values_size:
        raise ValueError("predictive values size is outside the permitted range")
    values = _decompress_bounded(values_payload, values_size, maximum=maximum_values_size)
    indices = _predictive_mask_indices(mask, unit_count, nonzero_count)
    encoded_values = _decode_leb128_values(values, nonzero_count)
    residual = np.zeros(unit_count, dtype=np.uint32)
    residual[indices] = encoded_values
    signed = (residual >> np.uint32(1)) ^ -(residual & np.uint32(1))
    output = bytearray(raw_size)
    if prefix_size:
        header_residual = signed[:prefix_size] & np.uint32(0xFF)
        header_base = np.frombuffer(base1[:prefix_size], dtype=np.uint8).astype(np.uint32)
        output[:prefix_size] = ((header_base + header_residual) & np.uint32(0xFF)).astype(np.uint8).tobytes()
    if data_words:
        local_data_start = data_start - global_offset
        local_data_end = local_data_start + data_words * 2
        predictor = _bf16_predictor(
            base2[local_data_start:local_data_end],
            base1[local_data_start:local_data_end],
        ).astype(np.uint32)
        words = (predictor + signed[prefix_size:]) & np.uint32(0xFFFF)
        output[local_data_start:local_data_end] = words.astype("<u2").tobytes()
    return bytes(output)


def predictive_raw_sha256(payload: bytes) -> str:
    """Return the digest of the canonical sparse zigzag value stream."""
    header, _, values_payload = _predictive_payload_parts(payload)
    nonzero_count = header[8]
    values_size = _decompressed_size(values_payload)
    maximum_values_size = nonzero_count * _predictive_max_varint_bytes
    if values_size < nonzero_count or values_size > maximum_values_size:
        raise ValueError("predictive values size is outside the permitted range")
    values = _decompress_bounded(values_payload, values_size, maximum=maximum_values_size)
    return hashlib.sha256(values).hexdigest()


def predictive_correction_bytes(payload: bytes) -> bytes:
    """Return the canonical sparse zigzag bytes from one predictive chunk."""
    header, _, values_payload = _predictive_payload_parts(payload)
    nonzero_count = header[8]
    values_size = _decompressed_size(values_payload)
    maximum_values_size = nonzero_count * _predictive_max_varint_bytes
    if values_size < nonzero_count or values_size > maximum_values_size:
        raise ValueError("predictive values size is outside the permitted range")
    return _decompress_bounded(values_payload, values_size, maximum=maximum_values_size)


def decode_linear_zigzag_sparse_chunk(
    base2: bytes,
    base1: bytes,
    payload: bytes,
    *,
    offset: int,
    data_offset: int,
    raw_size: int,
    codec: str,
) -> tuple[bytes, bytes]:
    """Decode the v2 transport API used by the inference worker."""
    if codec != _codec_name():
        raise ValueError("predictive chunk codec must be zstd")
    target = decode_predictive_chunk(
        base2,
        base1,
        payload,
        offset,
        data_offset,
        max_output_size=raw_size,
    )
    return target, predictive_correction_bytes(payload)


def safetensors_data_offset(path: Path) -> int:
    """Return the byte offset at which a safetensors file stores tensor data."""
    size = path.stat().st_size
    if size < 8:
        raise ValueError("safetensors file is shorter than its header length")
    with path.open("rb") as source:
        header_length_bytes = source.read(8)
        if len(header_length_bytes) != 8:
            raise OSError("safetensors header length is incomplete")
        header_length = int.from_bytes(header_length_bytes, "little")
        if header_length > size - 8:
            raise ValueError("safetensors header exceeds the file")
        header = source.read(header_length)
    try:
        parsed = json.loads(header)
    except (TypeError, ValueError) as exc:
        raise ValueError("safetensors header is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("safetensors header is not an object")
    return 8 + header_length


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
    base2_path: Path | None = None,
    data_offset: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    target_size = target_path.stat().st_size
    ranges = _chunk_ranges(target_size, chunk_size)
    if base_path is not None and base_path.stat().st_size != target_size:
        raise ValueError("adjacent adapter files must have the same size")
    if base2_path is not None and base2_path.stat().st_size != target_size:
        raise ValueError("second adjacent adapter files must have the same size")

    codec = _codec_name() if mode in {"delta", "predictive"} else "identity"
    chunks: list[dict[str, Any]] = []
    pending: dict[int, Future[Any]] = {}
    written_size = 0
    target = target_path.open("rb")
    base = base_path.open("rb") if base_path is not None else None
    base2 = base2_path.open("rb") if base2_path is not None else None

    def save_chunk(index: int, raw: bytes, payload: bytes, offset: int, raw_size: int) -> None:
        name = f"{target_path.name}.transport-{mode}-{index:06d}"
        path = destination / name
        _atomic_write(path, payload)
        chunk = {
            "file": name,
            "offset": offset,
            "raw_size": raw_size,
            "wire_size": len(payload),
            "wire_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "codec": codec,
        }
        if mode == "predictive":
            chunk["data_offset"] = data_offset
        chunks.append(chunk)

    def drain(index: int, start: int, end: int) -> None:
        nonlocal written_size
        result = pending.pop(index).result()
        if mode in {"delta", "predictive"}:
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
                base2_chunk = base2.read(end - start) if base2 is not None else None
                if base2 is not None and len(base2_chunk) != end - start:
                    raise OSError(f"second base adapter file ended early: {base2_path.name}")
                if mode == "delta":
                    assert base_chunk is not None
                    pending[index] = pool.submit(_encode_delta_chunk, base_chunk, target_chunk)
                elif mode == "predictive":
                    if base_chunk is None or base2_chunk is None or data_offset is None:
                        raise ValueError("predictive chunks require two bases and a data offset")
                    pending[index] = pool.submit(
                        _encode_predictive_chunk,
                        base2_chunk,
                        base_chunk,
                        target_chunk,
                        start,
                        data_offset,
                    )
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
        if base2 is not None:
            base2.close()

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
    base2_dir: Path | None = None,
    target_url: str | None = None,
    base_url: str | None = None,
    base2_url: str | None = None,
    target_step: int | None = None,
    base_step: int | None = None,
    base2_step: int | None = None,
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
    data_offset: int | None = None
    predictive_requested = base2_dir is not None or base2_url is not None or base2_step is not None
    if predictive_requested:
        if base_dir is None or base2_dir is None:
            raise ValueError("predictive transport requires both adjacent base directories")
        if not all(isinstance(step, int) and step >= 0 for step in (target_step, base_step, base2_step)):
            raise ValueError("predictive transport requires target, base, and second-base steps")
        if target_step != base_step + 1 or base_step != base2_step + 1:
            raise ValueError("predictive transport requires three adjacent policy steps")
        base_dir = base_dir.resolve()
        base2_dir = base2_dir.resolve()
        for directory, description in ((base_dir, "base"), (base2_dir, "second base")):
            weights_path = directory / adapter_weights_name
            config_path = directory / adapter_config_name
            if not weights_path.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"predictive {description} adapter is incomplete")
            if weights_path.stat().st_size != target_weights.stat().st_size:
                raise ValueError("predictive adapter files must have equal sizes")
        target_config_sha256 = sha256_file(target_config)
        if sha256_file(base_dir / adapter_config_name) != target_config_sha256:
            raise ValueError("predictive base adapter configuration identity differs")
        if sha256_file(base2_dir / adapter_config_name) != target_config_sha256:
            raise ValueError("predictive second-base adapter configuration identity differs")
        data_offset = safetensors_data_offset(target_weights)
        if data_offset % 2 or target_weights.stat().st_size % 2:
            raise ValueError("predictive safetensors data and file size must be even-aligned")
        if safetensors_data_offset(base_dir / adapter_weights_name) != data_offset:
            raise ValueError("predictive base safetensors data offset differs")
        if safetensors_data_offset(base2_dir / adapter_weights_name) != data_offset:
            raise ValueError("predictive second-base safetensors data offset differs")
        selected_mode = "predictive"
    else:
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

    if selected_mode in {"delta", "predictive"}:
        weight_chunks, weight_encoded_size = _write_encoded_chunks(
            target_weights,
            base_dir / adapter_weights_name,
            target_dir,
            mode=selected_mode,
            chunk_size=chunk_size,
            max_workers=max_workers,
            base2_path=base2_dir / adapter_weights_name if selected_mode == "predictive" and base2_dir is not None else None,
            data_offset=data_offset,
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
    if selected_mode == "predictive" and weight_encoded_size >= target_weights.stat().st_size:
        _remove_transport_chunks(target_dir)
        raise ValueError("predictive transport is not smaller than the full adapter")

    files = [
        _file_record(target_config),
        _file_record(target_weights),
    ]
    weights_record = files[1]
    manifest: dict[str, Any] = {
        "version": 2 if selected_mode == "predictive" else 1,
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
    if base2_url is not None:
        manifest["base2_url"] = base2_url
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
    elif selected_mode == "predictive":
        manifest.update(
            {
                "transport": predictive_transport_name,
                "base2_step": base2_step,
                "data_offset": data_offset,
                "chunk_alignment": 2,
                "chunk_coverage": {"offset": 0, "size": weights_record["size"]},
                "predictor": "bf16-float32-second-order-rne",
                "value_encoding": "sparse-zigzag-leb128",
                "mask_encoding": "zstd-bit-mask-little-endian",
                "chunks": [
                    {
                        "file": item["file"],
                        "offset": item["offset"],
                        "raw_size": item["raw_size"],
                        "wire_size": item["wire_size"],
                        "wire_sha256": item["wire_sha256"],
                        "raw_sha256": item["raw_sha256"],
                        "codec": item["codec"],
                        "data_offset": item["data_offset"],
                    }
                    for item in weight_chunks
                ],
            }
        )
    if base_dir is not None:
        manifest["base_size"] = (base_dir / adapter_weights_name).stat().st_size
        manifest["base_sha256"] = sha256_file(base_dir / adapter_weights_name)
        base_config = base_dir / adapter_config_name
        if base_config.is_file():
            manifest["base_config_sha256"] = sha256_file(base_config)
    if selected_mode == "predictive" and base2_dir is not None:
        base2_weights = base2_dir / adapter_weights_name
        manifest["base2_size"] = base2_weights.stat().st_size
        manifest["base2_sha256"] = sha256_file(base2_weights)
        base2_config = base2_dir / adapter_config_name
        manifest["base2_config_size"] = base2_config.stat().st_size
        manifest["base2_config_sha256"] = sha256_file(base2_config)
        manifest["base_config_size"] = (base_dir / adapter_config_name).stat().st_size
    _atomic_write(
        target_dir / f"{adapter_weights_name}.sha256",
        f"{weights_record['sha256']}\n".encode("ascii"),
    )
    _atomic_json(target_dir / adapter_manifest_name, manifest)
    return manifest
