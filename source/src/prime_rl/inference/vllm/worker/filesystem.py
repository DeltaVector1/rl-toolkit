import argparse
from contextlib import contextmanager
import ctypes
import ctypes.util
import fcntl
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

try:
    from prime_rl.utils.adapter_delta import (
        ADAPTER_DELTA_ENABLE_ENV,
        ADAPTER_DELTA_FILE,
        ADAPTER_DELTA_MANIFEST,
        AdapterDeltaError,
        adapter_cache_identity,
        apply_adapter_delta,
        load_adapter_delta_manifest,
        safetensors_metadata_sha256,
        select_adapter_transfer,
    )
except ModuleNotFoundError:  # unit tests load this worker file in isolation
    import importlib.util

    _adapter_delta_path = Path(__file__).parents[3] / "utils" / "adapter_delta.py"
    _adapter_delta_spec = importlib.util.spec_from_file_location(
        "prime_rl_utils_adapter_delta", _adapter_delta_path
    )
    if _adapter_delta_spec is None or _adapter_delta_spec.loader is None:
        raise
    _adapter_delta_module = importlib.util.module_from_spec(_adapter_delta_spec)
    _adapter_delta_spec.loader.exec_module(_adapter_delta_module)
    ADAPTER_DELTA_ENABLE_ENV = _adapter_delta_module.ADAPTER_DELTA_ENABLE_ENV
    ADAPTER_DELTA_FILE = _adapter_delta_module.ADAPTER_DELTA_FILE
    ADAPTER_DELTA_MANIFEST = _adapter_delta_module.ADAPTER_DELTA_MANIFEST
    AdapterDeltaError = _adapter_delta_module.AdapterDeltaError
    adapter_cache_identity = _adapter_delta_module.adapter_cache_identity
    apply_adapter_delta = _adapter_delta_module.apply_adapter_delta
    load_adapter_delta_manifest = _adapter_delta_module.load_adapter_delta_manifest
    safetensors_metadata_sha256 = _adapter_delta_module.safetensors_metadata_sha256
    select_adapter_transfer = _adapter_delta_module.select_adapter_transfer

_PARALLEL_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
_PARALLEL_DOWNLOAD_STREAMS = 8
_GLM_RANGE_GZIP_ENV = "PRIME_GLM_ADAPTER_GZIP"
_GLM_RANGE_CODEC_ENV = "PRIME_GLM_ADAPTER_CODEC"
_GLM_RANGE_CHUNK_BYTES = 256 * 1024 * 1024
_GLM_RANGE_ATTEMPTS = 3
_ADAPTER_CONFIG_NAME = "adapter_config.json"
_ADAPTER_CHECKSUM_NAME = "adapter_model.safetensors.sha256"
_ADAPTER_MODEL_NAME = "adapter_model.safetensors"
_ADAPTER_READY_NAME = "READY"
_ADAPTER_IDENTITY_VERSION = 1
_ADAPTER_CACHE_MAX_BYTES_ENV = "PRIME_LORA_CACHE_MAX_BYTES"
_ADAPTER_CACHE_MAX_STALE_ENV = "PRIME_LORA_CACHE_MAX_STALE_ENTRIES"
# Two 30.77 GB adapter generations fit by default: the active generation and
# the adjacent base needed for the next delta.  A single older generation is
# retained for a safe retry, and is removed first when either cap is reached.
_ADAPTER_CACHE_MAX_BYTES = 2 * 30_770_000_000
_ADAPTER_CACHE_MAX_STALE_ENTRIES = 1
_ADAPTER_CACHE_PARTIAL_MAX_AGE_ENV = "PRIME_LORA_CACHE_PARTIAL_MAX_AGE_SECONDS"
_ADAPTER_CACHE_PARTIAL_MAX_AGE_SECONDS = 60 * 60
_ADAPTER_FULL_WIRE_SIZE_ENV = "PRIME_ADAPTER_FULL_WIRE_SIZE"
_ZSTD_FRAME_BYTES = 8 * 1024 * 1024
_zstd = None
_LOGGER = logging.getLogger(__name__)


class AdapterDiskSpaceError(OSError):
    """Raised when staging cannot safely assemble another adapter."""


def _zstd_library():
    global _zstd
    if _zstd is None:
        library = ctypes.util.find_library("zstd") or "libzstd.so.1"
        try:
            loaded = ctypes.CDLL(library)
        except OSError as error:
            raise OSError("zstd adapter transfer requires libzstd") from error
        size_t = ctypes.c_size_t
        loaded.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            size_t,
            ctypes.c_void_p,
            size_t,
        ]
        loaded.ZSTD_decompress.restype = size_t
        loaded.ZSTD_isError.argtypes = [size_t]
        loaded.ZSTD_isError.restype = ctypes.c_uint
        loaded.ZSTD_getErrorName.argtypes = [size_t]
        loaded.ZSTD_getErrorName.restype = ctypes.c_char_p
        _zstd = loaded
    return _zstd


def _zstd_decompress(encoded: bytes, raw_size: int) -> bytes:
    library = _zstd_library()
    output = ctypes.create_string_buffer(raw_size)
    result = int(library.ZSTD_decompress(output, raw_size, encoded, len(encoded)))
    if library.ZSTD_isError(result):
        message = library.ZSTD_getErrorName(result).decode("utf-8", errors="replace")
        raise OSError(f"zstd adapter decompression failed: {message}")
    if result != raw_size:
        raise OSError(f"zstd frame has wrong raw length: {result} != {raw_size}")
    return output.raw[:result]


def _read_exact(source, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            raise EOFError("zstd adapter frame ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _copy_zstd_range(source, output, raw_size: int) -> None:
    remaining = raw_size
    while remaining:
        encoded_size, frame_size = struct.unpack("!II", _read_exact(source, 8))
        if (
            encoded_size == 0
            or frame_size == 0
            or frame_size > _ZSTD_FRAME_BYTES
            or frame_size > remaining
        ):
            raise OSError("zstd adapter frame header is invalid")
        encoded = _read_exact(source, encoded_size)
        output.write(_zstd_decompress(encoded, frame_size))
        remaining -= frame_size


def adapter_cache_path(source_url: str, target_sha256: str | None = None) -> Path:
    """Return a cache path bound to the URL and, when known, target bytes.

    The optional digest keeps callers that only need a lock path compatible.
    Completed adapter caches always use the digest-bound form.
    """
    cache_name = (
        hashlib.sha256(source_url.encode()).hexdigest()
        if target_sha256 is None
        else adapter_cache_identity(source_url, target_sha256)
    )
    return Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora")) / cache_name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_identity(config: Path, checksum: Path, model: Path | None = None) -> dict[str, object]:
    """Read the export identity without trusting a cache marker."""
    expected_model = checksum.read_text(encoding="ascii").strip()
    if len(expected_model) != 64 or any(
        character not in "0123456789abcdef" for character in expected_model
    ):
        raise ValueError("adapter SHA-256 file is invalid")
    config_sha256 = _sha256_file(config)
    identity = {
        "version": _ADAPTER_IDENTITY_VERSION,
        "adapter_config_sha256": config_sha256,
        "adapter_model_sha256": expected_model,
        "target_sha256": expected_model,
    }
    if model is not None:
        identity["adapter_model_size"] = model.stat().st_size
        try:
            identity["tensor_metadata_sha256"] = safetensors_metadata_sha256(model)
        except (OSError, ValueError, AdapterDeltaError):
            # Existing development fixtures use arbitrary model bytes.  Delta
            # mode requires real safetensors and will reject these later; the
            # normal full transfer keeps its established identity behavior.
            identity["tensor_metadata_sha256"] = None
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def _cached_adapter_matches(destination: Path, expected: dict[str, object]) -> bool:
    """Accept a cached adapter only after checking all bound content."""
    ready = destination / _ADAPTER_READY_NAME
    if not ready.is_file() or ready.is_symlink():
        return False
    try:
        cached = json.loads(ready.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError):
        return False
    config = destination / _ADAPTER_CONFIG_NAME
    checksum = destination / _ADAPTER_CHECKSUM_NAME
    model = destination / _ADAPTER_MODEL_NAME
    if any(
        not path.is_file() or path.is_symlink()
        for path in (config, checksum, model)
    ):
        return False
    try:
        actual = _adapter_identity(config, checksum, model)
        return (
            all(cached.get(key) == value for key, value in expected.items())
            and cached == actual
            and _sha256_file(model) == expected["adapter_model_sha256"]
        )
    except (OSError, UnicodeError, ValueError):
        return False


def _adapter_cache_limits() -> tuple[int, int]:
    try:
        max_bytes = int(
            os.environ.get(
                _ADAPTER_CACHE_MAX_BYTES_ENV, str(_ADAPTER_CACHE_MAX_BYTES)
            )
        )
        max_stale = int(
            os.environ.get(
                _ADAPTER_CACHE_MAX_STALE_ENV,
                str(_ADAPTER_CACHE_MAX_STALE_ENTRIES),
            )
        )
    except ValueError as error:
        raise OSError("adapter cache limits must be integers") from error
    if max_bytes <= 0 or max_stale < 0:
        raise OSError("adapter cache limits are invalid")
    return max_bytes, max_stale


def _adapter_cache_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        if child.is_file():
            total += child.stat().st_size
    return total


def _adapter_cache_lock(cache_dir: Path):
    """Lock all staging and pruning for one cache, regardless of source URL."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (cache_dir / ".adapter-cache.lock").open("w")


@contextmanager
def _adapter_stage_lock(cache_dir: Path, source_lock_name: str):
    """Serialize cache mutations globally, then preserve per-source locking."""
    with _adapter_cache_lock(cache_dir) as cache_lock:
        fcntl.flock(cache_lock, fcntl.LOCK_EX)
        with (cache_dir / f"{source_lock_name}.lock").open("w") as source_lock:
            fcntl.flock(source_lock, fcntl.LOCK_EX)
            yield


def _partial_max_age() -> float:
    try:
        age = float(
            os.environ.get(
                _ADAPTER_CACHE_PARTIAL_MAX_AGE_ENV,
                str(_ADAPTER_CACHE_PARTIAL_MAX_AGE_SECONDS),
            )
        )
    except ValueError as error:
        raise OSError("adapter partial age limit must be numeric") from error
    if age < 0:
        raise OSError("adapter partial age limit must be non-negative")
    return age


def _prune_adapter_cache_locked(cache_dir: Path, keep: set[Path]) -> None:
    """Bound stale adapter storage; caller holds the cache-global lock."""
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        return
    max_bytes, max_stale = _adapter_cache_limits()
    partial_age = _partial_max_age()
    now = time.time()
    # A process can die after creating its unique assembly directory.  Remove
    # only old partials; recent ones belong to another active transfer and are
    # counted below instead of being touched.
    partials = []
    for entry in cache_dir.iterdir():
        if (
            entry.is_dir()
            and not entry.is_symlink()
            and ".partial-" in entry.name
        ):
            age = max(0.0, now - entry.stat().st_mtime)
            if age >= partial_age:
                shutil.rmtree(entry)
            else:
                partials.append(entry)
    entries = []
    for entry in cache_dir.iterdir():
        if entry.is_symlink() or not entry.is_dir() or entry in keep:
            continue
        if not (entry / _ADAPTER_READY_NAME).is_file():
            continue
        entries.append((entry.stat().st_mtime, entry))
    entries.sort()
    completed = [
        entry
        for entry in cache_dir.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and (entry / _ADAPTER_READY_NAME).is_file()
    ]
    total = sum(_adapter_cache_size(entry) for entry in (*completed, *partials))
    stale_count = len(entries)
    for _mtime, entry in entries:
        if stale_count <= max_stale and total <= max_bytes:
            break
        size = _adapter_cache_size(entry)
        shutil.rmtree(entry)
        stale_count -= 1
        total -= size


def _prune_adapter_cache(cache_dir: Path, keep: set[Path]) -> None:
    """Lock and bound stale adapter storage."""
    with _adapter_cache_lock(cache_dir) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prune_adapter_cache_locked(cache_dir, keep)


def _safe_prune_adapter_cache(cache_dir: Path, keep: set[Path]) -> None:
    """Pruning is cleanup; never turn a verified publication into a failure."""
    try:
        _prune_adapter_cache(cache_dir, keep)
    except Exception as error:
        _LOGGER.warning("adapter cache cleanup failed after verified staging: %s", error)


def _safe_prune_adapter_cache_locked(cache_dir: Path, keep: set[Path]) -> None:
    """Best-effort cleanup for callers already holding the cache lock."""
    try:
        _prune_adapter_cache_locked(cache_dir, keep)
    except Exception as error:
        _LOGGER.warning("adapter cache cleanup failed after verified staging: %s", error)


def _cache_adapter_bytes(cache_dir: Path) -> int:
    """Count completed and in-progress adapter bytes for disk preflight."""
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        return 0
    total = 0
    for entry in cache_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            total += _adapter_cache_size(entry)
    return total


def _check_adapter_disk_space(cache_dir: Path, target_size: int, base_size: int = 0) -> None:
    """Require room for active, base, and an assembling target before transfer."""
    if target_size <= 0 or base_size < 0:
        raise OSError("adapter disk preflight sizes are invalid")
    active_size = 0
    if cache_dir.is_dir() and not cache_dir.is_symlink():
        for entry in cache_dir.iterdir():
            model = entry / _ADAPTER_MODEL_NAME
            if entry.is_dir() and not entry.is_symlink() and model.is_file():
                active_size = max(active_size, model.stat().st_size)
    required = active_size + base_size + target_size
    available = shutil.disk_usage(cache_dir).free
    if available < required:
        raise AdapterDiskSpaceError(
            f"insufficient adapter disk space: need {required} bytes, have {available}"
        )


def _content_length(source_urls: list[str]) -> int:
    error = None
    for source_url in source_urls:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(source_url, method="HEAD"), timeout=60
            ) as response:
                total = int(response.headers["Content-Length"])
                if total <= 0:
                    raise OSError("adapter content length is invalid")
                return total
        except (KeyError, ValueError, OSError) as failure:
            error = failure
    raise OSError(f"cannot read adapter content length: {error}")


def _full_wire_size_estimate(source_url: str, target_size: int) -> int | None:
    """Get a current complete-transfer size without downloading the model."""
    configured = os.environ.get(_ADAPTER_FULL_WIRE_SIZE_ENV)
    if configured is not None:
        try:
            value = int(configured)
        except ValueError:
            return None
        return value if value > 0 else None
    codec = os.environ.get(_GLM_RANGE_CODEC_ENV)
    if codec is None:
        gzip_required = os.environ.get(_GLM_RANGE_GZIP_ENV, "0")
        if gzip_required not in {"0", "1"}:
            return None
        codec = "gzip" if gzip_required == "1" else "identity"
    if codec == "identity":
        # The sidecar is an exact byte count for the normal uncompressed path.
        return target_size
    if codec not in {"gzip", "zstd"}:
        return None
    try:
        request = urllib.request.Request(
            source_url,
            method="HEAD",
            headers={"Accept-Encoding": codec},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            explicit = response.headers.get("X-Prime-Wire-Length")
            if explicit is not None:
                value = int(explicit)
                return value if value > 0 else None
            if response.headers.get("Content-Encoding", "").lower() != codec:
                return None
            value = int(response.headers["Content-Length"])
            return value if value > 0 else None
    except (KeyError, ValueError, OSError):
        return None


def _download_range(
    source_urls: list[str],
    source_index: int,
    destination: Path,
    start: int,
    end: int,
    max_attempts: int = 10,
) -> None:
    codec = os.environ.get(_GLM_RANGE_CODEC_ENV)
    if codec is None:
        gzip_required = os.environ.get(_GLM_RANGE_GZIP_ENV, "0")
        if gzip_required not in {"0", "1"}:
            raise OSError(f"{_GLM_RANGE_GZIP_ENV} must be 0 or 1")
        codec = "gzip" if gzip_required == "1" else "identity"
    if codec not in {"identity", "gzip", "zstd"}:
        raise OSError(f"{_GLM_RANGE_CODEC_ENV} must be identity, gzip, or zstd")
    for attempt in range(max_attempts):
        try:
            source_url = source_urls[(source_index + attempt) % len(source_urls)]
            size = end - start + 1
            if destination.is_symlink():
                raise OSError(f"adapter range path is a symlink: {destination.name}")
            present = destination.stat().st_size if destination.exists() else 0
            if present == size:
                return
            if present > size:
                destination.unlink()
                present = 0
            requested_start = start + present
            headers = {"Range": f"bytes={requested_start}-{end}"}
            if codec != "identity":
                headers["Accept-Encoding"] = codec
            request = urllib.request.Request(source_url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise OSError("adapter server did not honor byte range")
                content_range = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", ""),
                )
                if (
                    content_range is None
                    or int(content_range.group(1)) != requested_start
                    or int(content_range.group(2)) != end
                    or int(content_range.group(3)) <= end
                ):
                    raise OSError("adapter server returned the wrong byte range")
                if codec == "gzip":
                    if response.headers.get("Content-Encoding", "").lower() != "gzip":
                        raise OSError("adapter server did not return required gzip")
                    if response.headers.get("X-Prime-Raw-Length") != str(
                        end - requested_start + 1
                    ):
                        raise OSError("adapter gzip range length differs")
                    with (
                        gzip.GzipFile(fileobj=response, mode="rb") as source,
                        destination.open("ab" if present else "wb") as output,
                    ):
                        shutil.copyfileobj(
                            source, output, length=16 * 1024 * 1024
                        )
                elif codec == "zstd":
                    if response.headers.get("Content-Encoding", "").lower() != "zstd":
                        raise OSError("adapter server did not return required zstd")
                    if response.headers.get("X-Prime-Codec") != "zstd-frames-v1":
                        raise OSError("adapter server did not return framed zstd")
                    if response.headers.get("X-Prime-Raw-Length") != str(
                        end - requested_start + 1
                    ):
                        raise OSError("adapter zstd range length differs")
                    with destination.open("ab" if present else "wb") as output:
                        _copy_zstd_range(
                            response, output, end - requested_start + 1
                        )
                else:
                    with destination.open("ab" if present else "wb") as output:
                        shutil.copyfileobj(
                            response, output, length=16 * 1024 * 1024
                        )
            if destination.stat().st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except (EOFError, OSError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(2**attempt, 10))


def _publish_download(temporary: Path, destination: Path, create_only: bool) -> None:
    if create_only:
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    else:
        temporary.replace(destination)


def _download_glm_chunks(
    source_urls: list[str], parts: Path, ranges: list[tuple[int, int, int]]
) -> list[int]:
    pending = Queue()
    for index, start, end in ranges:
        pending.put((index, start, end, -1))
    cancelled = threading.Event()
    finished = threading.Event()
    attempts = [0] * len(ranges)
    errors = []
    lock = threading.Lock()

    def worker(source_index: int, source_url: str) -> None:
        while not finished.is_set():
            try:
                index, start, end, previous_source = pending.get(timeout=0.1)
            except Empty:
                continue
            if cancelled.is_set():
                pending.task_done()
                continue
            if previous_source == source_index and len(source_urls) > 1:
                pending.put((index, start, end, previous_source))
                pending.task_done()
                time.sleep(0.01)
                continue
            with lock:
                attempts[index] += 1
            try:
                _download_range(
                    [source_url],
                    0,
                    parts / f"part-{index:06d}-{start:012x}-{end:012x}",
                    start,
                    end,
                    max_attempts=1,
                )
            except (EOFError, OSError) as error:
                if attempts[index] < _GLM_RANGE_ATTEMPTS:
                    pending.put((index, start, end, source_index))
                else:
                    with lock:
                        errors.append(
                            OSError(
                                f"adapter chunk {index} failed after "
                                f"{attempts[index]} attempts: {error}"
                            )
                        )
                    cancelled.set()
            except BaseException as error:
                with lock:
                    errors.append(error)
                cancelled.set()
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=len(source_urls)) as executor:
        futures = [
            executor.submit(worker, source_index, source_url)
            for source_index, source_url in enumerate(source_urls)
        ]
        try:
            pending.join()
        except BaseException:
            cancelled.set()
            raise
        finally:
            finished.set()
        for future in as_completed(futures):
            future.result()
    if errors:
        raise errors[0]
    return attempts


def download(
    source_url: str,
    destination: Path,
    create_only: bool = False,
    _attempt_report: list[int] | None = None,
) -> None:
    ports = [value.strip() for value in os.environ.get("PRIME_ADAPTER_PORTS", "").split(",") if value.strip()]
    source_urls = [source_url]
    if ports:
        parsed = urlsplit(source_url)
        if parsed.hostname not in ("127.0.0.1", "localhost"):
            raise OSError("adapter tunnel ports require a loopback source URL")
        source_urls = [
            urlunsplit((parsed.scheme, f"{parsed.hostname}:{int(port)}", parsed.path, parsed.query, parsed.fragment))
            for port in ports
        ]
    total = _content_length(source_urls)
    if create_only and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"adapter destination already exists: {destination}")
    if not create_only and destination.exists() and destination.stat().st_size == total:
        return

    streams = len(source_urls) if ports else max(
        1,
        int(os.environ.get("PRIME_ADAPTER_DOWNLOAD_STREAMS", str(_PARALLEL_DOWNLOAD_STREAMS))),
    )
    glm_gzip = os.environ.get(_GLM_RANGE_GZIP_ENV, "0") == "1"
    if total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        range_urls = source_urls if glm_gzip else [source_url]
        attempts = len(range_urls) if glm_gzip else 10
        if not create_only:
            _download_range(
                range_urls, 0, destination, 0, total - 1, max_attempts=attempts
            )
            return
        assembling = destination.with_name(f"{destination.name}.assembling")
        assembling.unlink(missing_ok=True)
        try:
            _download_range(
                range_urls, 0, assembling, 0, total - 1, max_attempts=attempts
            )
            _publish_download(assembling, destination, True)
        except BaseException as error:
            if glm_gzip:
                try:
                    assembling.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(f"cannot clean {assembling}: {cleanup_error}")
            raise
        return

    if not create_only:
        destination.unlink(missing_ok=True)
    parts = destination.with_name(f"{destination.name}.parts")
    assembling = destination.with_name(f"{destination.name}.assembling")
    if assembling.is_symlink():
        raise OSError(f"adapter assembly path is unsafe: {assembling}")
    parts.mkdir(exist_ok=True)
    if parts.is_symlink() or not parts.is_dir():
        raise OSError(f"adapter parts path is unsafe: {parts}")
    chunk = (
        _GLM_RANGE_CHUNK_BYTES
        if glm_gzip
        else (total + streams - 1) // streams
    )
    ranges = [
        (index, start, min(start + chunk, total) - 1)
        for index, start in enumerate(range(0, total, chunk))
    ]
    try:
        if glm_gzip:
            attempts = _download_glm_chunks(source_urls, parts, ranges)
            if _attempt_report is not None:
                _attempt_report.extend(attempts)
        else:
            with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
                futures = [
                    executor.submit(
                        _download_range,
                        source_urls,
                        index,
                        parts / f"part-{index:03d}",
                        start,
                        end,
                    )
                    for index, start, end in ranges
                ]
                for future in futures:
                    future.result()

        with assembling.open("wb") as output:
            for index, start, end in ranges:
                name = (
                    f"part-{index:06d}-{start:012x}-{end:012x}"
                    if glm_gzip
                    else f"part-{index:03d}"
                )
                with (parts / name).open("rb") as part:
                    shutil.copyfileobj(part, output, length=16 * 1024 * 1024)
        if assembling.stat().st_size != total:
            raise OSError(f"assembled adapter file has the wrong size: {destination.name}")
        _publish_download(assembling, destination, create_only)
    except BaseException as error:
        if glm_gzip:
            try:
                assembling.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"cannot clean {assembling}: {cleanup_error}")
            try:
                if parts.exists():
                    shutil.rmtree(parts)
            except OSError as cleanup_error:
                error.add_note(f"cannot clean {parts}: {cleanup_error}")
        raise
    shutil.rmtree(parts)


def _download_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("expected_size", type=int)
    args = parser.parse_args()
    if (
        len(args.expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in args.expected_sha256)
        or args.expected_size <= 0
    ):
        raise ValueError("adapter identity is invalid")
    started = time.monotonic()
    chunk_attempts = []
    download(
        args.source_url,
        args.destination,
        create_only=True,
        _attempt_report=chunk_attempts,
    )
    actual_size = args.destination.stat().st_size
    actual_sha256 = _sha256_file(args.destination)
    if actual_size != args.expected_size or actual_sha256 != args.expected_sha256:
        args.destination.unlink()
        raise ValueError("downloaded adapter identity differs")
    print(
        json.dumps(
            {
                "bytes": actual_size,
                "seconds": time.monotonic() - started,
                "sha256": actual_sha256,
                "chunk_attempts": chunk_attempts,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _download_optional(source_url: str, destination: Path) -> bool:
    """Download an optional delta artifact and report absence as fallback."""
    try:
        download(source_url, destination, create_only=True)
    except (EOFError, OSError, ValueError):
        destination.unlink(missing_ok=True)
        return False
    return True


def _delta_enabled() -> bool:
    value = os.environ.get(ADAPTER_DELTA_ENABLE_ENV, "0")
    if value not in {"0", "1"}:
        raise OSError(f"{ADAPTER_DELTA_ENABLE_ENV} must be 0 or 1")
    return value == "1"


def stage_lora_adapter_files(source_url: str) -> Path:
    """Stage one adapter, using an exact adjacent delta only when enabled.

    The complete adapter is downloaded into a unique directory.  The active
    cache directory is changed only after the complete model, checksum,
    safetensors metadata, and READY marker pass verification.
    """
    cache_root = adapter_cache_path(source_url)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with _adapter_stage_lock(cache_root.parent, cache_root.name):
        temporary = Path(
            tempfile.mkdtemp(prefix=f"{cache_root.name}.partial-", dir=cache_root.parent)
        )
        try:
            for filename in (_ADAPTER_CONFIG_NAME, _ADAPTER_CHECKSUM_NAME):
                download(
                    f"{source_url.rstrip('/')}/{filename}",
                    temporary / filename,
                    create_only=True,
                )
            basic_identity = _adapter_identity(
                temporary / _ADAPTER_CONFIG_NAME,
                temporary / _ADAPTER_CHECKSUM_NAME,
            )
            target_sha256 = basic_identity["adapter_model_sha256"]
            destination = adapter_cache_path(source_url, target_sha256)
            if destination.is_symlink() or (
                destination.exists() and not destination.is_dir()
            ):
                raise OSError(f"adapter cache path is unsafe: {destination}")
            if destination.is_dir() and _cached_adapter_matches(destination, basic_identity):
                keep = {destination}
                if _delta_enabled():
                    hit_manifest = temporary / ADAPTER_DELTA_MANIFEST
                    if _download_optional(
                        f"{source_url.rstrip('/')}/{ADAPTER_DELTA_MANIFEST}",
                        hit_manifest,
                    ):
                        try:
                            hit_identity = load_adapter_delta_manifest(hit_manifest)
                            hit_base = adapter_cache_path(
                                source_url, hit_identity["base_sha256"]
                            )
                            hit_base_expected = {
                                "version": _ADAPTER_IDENTITY_VERSION,
                                "adapter_model_sha256": hit_identity["base_sha256"],
                                "adapter_model_size": hit_identity["base_size"],
                                "adapter_config_sha256": hit_identity[
                                    "base_adapter_config_sha256"
                                ],
                                "tensor_metadata_sha256": hit_identity[
                                    "base_tensor_metadata_sha256"
                                ],
                            }
                            if hit_base.is_dir() and _cached_adapter_matches(
                                hit_base, hit_base_expected
                            ):
                                keep.add(hit_base)
                        except (
                            AdapterDeltaError,
                            OSError,
                            ValueError,
                            TypeError,
                            KeyError,
                        ):
                            pass
                _safe_prune_adapter_cache_locked(destination.parent, keep)
                return destination

            transfer = "full"
            base_destination = None
            delta_manifest = None
            if _delta_enabled():
                manifest_path = temporary / ADAPTER_DELTA_MANIFEST
                if _download_optional(
                    f"{source_url.rstrip('/')}/{ADAPTER_DELTA_MANIFEST}",
                    manifest_path,
                ):
                    try:
                        delta_manifest = load_adapter_delta_manifest(manifest_path)
                        # The model size is not known until the manifest is
                        # checked.  The manifest itself supplies the target
                        # size and must bind the target checksum and config.
                        target_size = int(delta_manifest["uncompressed_size"])
                        manifest_target = {
                            "target_sha256": target_sha256,
                            "adapter_config_sha256": basic_identity["adapter_config_sha256"],
                            "uncompressed_size": target_size,
                        }
                        if any(
                            delta_manifest.get(key) != value
                            for key, value in manifest_target.items()
                        ):
                            raise AdapterDeltaError("adapter delta target identity differs")
                        _check_adapter_disk_space(
                            destination.parent,
                            target_size,
                            delta_manifest["base_size"],
                        )
                        base_sha256 = delta_manifest["base_sha256"]
                        base_destination = adapter_cache_path(source_url, base_sha256)
                        base_expected = {
                            "version": _ADAPTER_IDENTITY_VERSION,
                            "adapter_model_sha256": base_sha256,
                            "adapter_model_size": delta_manifest["base_size"],
                            "adapter_config_sha256": delta_manifest.get(
                                "base_adapter_config_sha256"
                            ),
                            "tensor_metadata_sha256": delta_manifest.get(
                                "base_tensor_metadata_sha256"
                            ),
                        }
                        base_available = (
                            base_destination.is_dir()
                            and base_expected["adapter_config_sha256"] is not None
                            and _cached_adapter_matches(base_destination, base_expected)
                        )
                        choice = select_adapter_transfer(
                            delta_manifest,
                            base_available=base_available,
                            base_sha256=base_sha256 if base_available else None,
                            target_sha256=target_sha256,
                            target_size=target_size,
                            enabled=True,
                            full_wire_size=_full_wire_size_estimate(
                                f"{source_url.rstrip('/')}/{_ADAPTER_MODEL_NAME}",
                                target_size,
                            ),
                            manifest_wire_size=manifest_path.stat().st_size,
                        )
                        if choice == "delta":
                            delta_path = temporary / ADAPTER_DELTA_FILE
                            if not _download_optional(
                                f"{source_url.rstrip('/')}/{ADAPTER_DELTA_FILE}",
                                delta_path,
                            ):
                                raise AdapterDeltaError("adapter delta artifact is unavailable")
                            apply_adapter_delta(
                                base_destination / _ADAPTER_MODEL_NAME,
                                delta_path,
                                temporary / _ADAPTER_MODEL_NAME,
                                delta_manifest,
                            )
                            if (
                                safetensors_metadata_sha256(
                                    temporary / _ADAPTER_MODEL_NAME
                                )
                                != delta_manifest["tensor_metadata_sha256"]
                            ):
                                raise AdapterDeltaError(
                                    "reconstructed safetensors metadata differs"
                                )
                            transfer = "delta"
                        else:
                            raise AdapterDeltaError(
                                "adapter delta base is absent, mismatched, or too large"
                            )
                    except AdapterDiskSpaceError:
                        raise
                    except (AdapterDeltaError, OSError, ValueError, TypeError, KeyError):
                        # Delta is an optimization.  A failed or ineligible
                        # delta must select the complete current transfer.
                        transfer = "full"
                        temporary.joinpath(_ADAPTER_MODEL_NAME).unlink(missing_ok=True)

            if transfer == "full":
                download(
                    f"{source_url.rstrip('/')}/{_ADAPTER_MODEL_NAME}",
                    temporary / _ADAPTER_MODEL_NAME,
                    create_only=True,
                )
                actual_model = _sha256_file(temporary / _ADAPTER_MODEL_NAME)
                if actual_model != target_sha256:
                    raise ValueError(
                        "downloaded adapter SHA-256 does not match the trainer export"
                    )

            marker = _adapter_identity(
                temporary / _ADAPTER_CONFIG_NAME,
                temporary / _ADAPTER_CHECKSUM_NAME,
                temporary / _ADAPTER_MODEL_NAME,
            )
            if marker["adapter_model_sha256"] != target_sha256:
                raise ValueError("staged adapter identity changed during transfer")
            # Delta mode must bind metadata.  Full mode also records it when
            # this is a real safetensors file, which makes later base checks
            # strict without changing the default full transfer.
            (temporary / _ADAPTER_READY_NAME).write_text(
                json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            keep = {destination}
            # Keep the adjacent base even when applying the delta failed and
            # we fell back to the full transfer: it can still be the active
            # adapter, and is the only safe base for the next retry.
            if base_destination is not None and base_destination.is_dir():
                keep.add(base_destination)
            _safe_prune_adapter_cache_locked(destination.parent, keep)
            temporary = None
            return destination
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)


def stage_full_checkpoint_files(source_url: str) -> Path:
    """Stage a complete checkpoint for the legacy Muse endpoint.

    This helper only restores the endpoint import contract.  Adapter delta
    staging above is independent from full-weight transfer.
    """
    destination = Path(
        os.environ.get("PRIME_FULL_CHECKPOINT_CACHE_DIR", "/tmp/prime-rl-checkpoints")
    ) / hashlib.sha256(source_url.encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.partial-", dir=destination.parent))
    try:
        # The full checkpoint endpoint serves its index and files.  Download
        # the index first, then the shard names declared by that index.
        index_name = "model.safetensors.index.json"
        download(f"{source_url.rstrip('/')}/{index_name}", temporary / index_name, create_only=True)
        index = json.loads((temporary / index_name).read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise OSError("checkpoint index has no shards")
        for filename in shards:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise OSError("checkpoint index contains an unsafe shard name")
            download(f"{source_url.rstrip('/')}/{filename}", temporary / filename, create_only=True)
        temporary.rename(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(_download_cli())


from torch.nn import Module  # noqa: E402
from vllm.model_executor.model_loader import (  # noqa: E402
    DefaultModelLoader,
    get_model_loader,
)

from prime_rl.inference.vllm.worker.weight_transfer import (  # noqa: E402
    load_weights_checkpoint_layerwise,
)


# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize the broadcaster."""
        ...

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def parity_router_diagnostic(self, payload: dict) -> dict:
        from prime_rl.parity_router_trace import run_h200_router_diagnostic

        return run_h200_router_diagnostic(self.model_runner, payload)

    def stage_lora_adapter(self, source_url: str) -> str:
        """Stage an adapter and return its digest-bound cache directory."""
        return str(stage_lora_adapter_files(source_url))

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        # Get vLLM model runner and model
        # When enforce_eager=True, model isn't wrapped by torch.compile so no .runnable attr
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        # Get vLLM model loader
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )


from prime_rl.parity_router_trace import install_vllm_parity_trace  # noqa: E402

install_vllm_parity_trace()
