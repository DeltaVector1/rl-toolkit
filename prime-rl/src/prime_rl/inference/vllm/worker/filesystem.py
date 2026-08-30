import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import time
import tomllib
import urllib.request
from urllib.error import HTTPError
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.utils.checkpoint_delta import (
    DELTA_MANIFEST,
    apply_xor_delta_file,
    sha256_file,
)

try:
    from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
except ImportError:
    # The official Muse vLLM image contains the upstream layerwise reload
    # lifecycle but not Prime-RL's small compatibility wrapper.
    def load_weights_checkpoint_layerwise(model, weights_iterator, model_config, vllm_config) -> None:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(model)
        model.load_weights(weights_iterator)
        finalize_layerwise_reload(model, model_config)


def adapter_cache_path(source_url: str) -> Path:
    return Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora")) / hashlib.sha256(
        source_url.encode()
    ).hexdigest()


_PARALLEL_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
_PARALLEL_DOWNLOAD_STREAMS = 8
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_ADAPTER_MANIFEST = "adapter_manifest.json"
_ADAPTER_DIGEST = "adapter_model.safetensors.sha256"
_ADAPTER_STATUS = ".adapter-stage-status.json"
_ADAPTER_DELTA_TRANSPORT = "chunked-shuffled-xor-zstd"
_ADAPTER_MAX_CHUNKS = 65536
_ADAPTER_LANE_TIMEOUT_S = float(os.environ.get("PRIME_LORA_LANE_TIMEOUT_S", "45"))
_ADAPTER_LANE_RETRIES = int(os.environ.get("PRIME_LORA_LANE_RETRIES", "6"))
_ADAPTER_STATUS_INTERVAL_S = float(os.environ.get("PRIME_LORA_STATUS_INTERVAL_S", "1"))
_JSON_LOCK = threading.Lock()


def _delta_stage_timeout_s() -> float:
    configured = os.environ.get("PRIME_LORA_DELTA_STAGE_TIMEOUT_S")
    if configured is not None:
        try:
            value = float(configured)
        except ValueError as exc:
            raise OSError("PRIME_LORA_DELTA_STAGE_TIMEOUT_S must be a number") from exc
    else:
        config_path = Path(
            os.environ.get(
                "PRIME_RL_OPERATIONS_CONFIG",
                "/data/glm52-rl/deployment/configs/operations.toml",
            )
        )
        try:
            with config_path.open("rb") as stream:
                config = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise OSError(f"operations config is missing: {config_path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise OSError(f"cannot read operations config: {config_path}") from exc
        else:
            limits = config.get("limits")
            if not isinstance(limits, dict) or "adapter_delta_stage_timeout_seconds" not in limits:
                raise OSError("operations config has no adapter delta stage timeout")
            value = limits["adapter_delta_stage_timeout_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise OSError("adapter delta stage timeout must be positive")
    return float(value)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("adapter delta staging exceeded its time limit")


def _acquire_stage_lock(lock, deadline: float | None) -> None:
    if deadline is None:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return
    while True:
        _check_deadline(deadline)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _request_timeout(deadline: float | None) -> float:
    _check_deadline(deadline)
    if deadline is None:
        return _ADAPTER_LANE_TIMEOUT_S
    return min(_ADAPTER_LANE_TIMEOUT_S, deadline - time.monotonic())


def _wait_for_futures(
    executor: ThreadPoolExecutor,
    futures,
    deadline: float | None,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        for future in futures:
            if deadline is None:
                future.result()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("adapter delta staging exceeded its time limit")
            try:
                future.result(timeout=remaining)
            except FutureTimeoutError as exc:
                raise TimeoutError("adapter delta staging exceeded its time limit") from exc
    except BaseException:
        if cancel_event is not None:
            cancel_event.set()
        for future in futures:
            future.cancel()
        # Do not let the caller clean a path while a worker can still write it.
        # Socket reads and retry sleeps check the cancellation event and have
        # bounded timeouts, so this join completes after cancellation.
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _content_length(source_url: str, deadline: float | None = None) -> int:
    _check_deadline(deadline)
    with urllib.request.urlopen(
        urllib.request.Request(source_url, method="HEAD"), timeout=_request_timeout(deadline)
    ) as response:
        _check_deadline(deadline)
        try:
            return int(response.headers["Content-Length"])
        except (TypeError, ValueError) as exc:
            raise OSError(f"adapter source has no valid Content-Length: {source_url}") from exc


def _read_source_text(source_url: str, deadline: float | None = None) -> str:
    return _read_source_bytes(source_url, deadline=deadline).decode("utf-8")


def _read_source_bytes(source_url: str, deadline: float | None = None) -> bytes:
    _check_deadline(deadline)
    with urllib.request.urlopen(source_url, timeout=_request_timeout(deadline)) as response:
        chunks = []
        while True:
            _check_deadline(deadline)
            chunk = response.read(16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _check_deadline(deadline)
        return b"".join(chunks)


def _source_manifest(source_url: str, deadline: float | None = None) -> dict[str, object]:
    """Read the immutable adapter manifest published beside a STABLE marker.

    Older broadcasts have only the model checksum sidecar. Keep that fallback,
    but always return a local manifest with a size and digest for every file.
    """
    source_root = source_url.rstrip("/")
    _check_deadline(deadline)
    manifest_url = f"{source_root}/{_ADAPTER_MANIFEST}"
    try:
        value = json.loads(_read_source_text(manifest_url, deadline=deadline))
    except HTTPError as exc:
        if exc.code != 404:
            raise
        value = None
    if value is not None:
        if not isinstance(value, dict) or value.get("version") != 1:
            raise OSError("invalid adapter source manifest")
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise OSError("adapter source manifest has no files")
        files: dict[str, dict[str, object]] = {}
        for item in raw_files:
            _check_deadline(deadline)
            if not isinstance(item, dict):
                raise OSError("invalid adapter source manifest file")
            name = item.get("name")
            size = item.get("size")
            digest = item.get("sha256")
            if (
                not isinstance(name, str)
                or name not in _ADAPTER_FILES
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise OSError("invalid adapter source manifest entry")
            files[name] = {"size": size, "sha256": digest}
        if set(files) != set(_ADAPTER_FILES):
            raise OSError("adapter source manifest is missing an adapter file")
        return {"version": 1, "source_url": source_url, "files": files}

    model_url = f"{source_root}/{_ADAPTER_FILES[1]}"
    config_url = f"{source_root}/{_ADAPTER_FILES[0]}"
    digest_text = _read_source_text(
        f"{source_root}/{_ADAPTER_DIGEST}", deadline=deadline
    ).strip().split()
    if not digest_text or re.fullmatch(r"[0-9a-f]{64}", digest_text[0]) is None:
        raise OSError("adapter source has no valid model checksum")
    config_bytes = _read_source_bytes(config_url, deadline=deadline)
    return {
        "version": 1,
        "source_url": source_url,
        "files": {
            _ADAPTER_FILES[0]: {
                "size": _content_length(config_url, deadline=deadline),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            _ADAPTER_FILES[1]: {
                "size": _content_length(model_url, deadline=deadline),
                "sha256": digest_text[0],
            },
        },
    }


def _adapter_source_urls(source_url: str) -> list[str]:
    ports = [
        value.strip()
        for value in os.environ.get("PRIME_LORA_SOURCE_PORTS", "").split(",")
        if value.strip()
    ]
    if not ports:
        return [source_url]
    parsed = urlsplit(source_url)
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        raise OSError("adapter tunnel pool requires a loopback source URL")
    source_urls: list[str] = []
    for port in ports:
        try:
            value = int(port)
        except ValueError as exc:
            raise OSError(f"invalid adapter source port: {port!r}") from exc
        if not 1 <= value <= 65535:
            raise OSError(f"invalid adapter source port: {port!r}")
        source_urls.append(
            urlunsplit(
                (
                    parsed.scheme,
                    f"{parsed.hostname}:{value}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        )
    return source_urls


def _read_chunked_xor_manifest(
    source_url: str, deadline: float | None = None
) -> dict[str, object] | None:
    """Read the exact version-one adapter transport manifest."""
    source_root = source_url.rstrip("/")
    _check_deadline(deadline)
    try:
        value = json.loads(
            _read_source_text(f"{source_root}/{_ADAPTER_MANIFEST}", deadline=deadline)
        )
    except HTTPError as exc:
        if exc.code == 404:
            raise OSError("adapter manifest is missing; transfer mode is unknown") from exc
        raise
    if not isinstance(value, dict) or value.get("version") != 1:
        raise OSError("invalid adapter transport manifest")
    mode = value.get("mode")
    if mode == "full":
        return {"mode": "full"}
    if mode != "delta" or value.get("kind") != "lora-adapter":
        raise OSError("adapter manifest selects an unsupported mode")
    if value.get("transport") != _ADAPTER_DELTA_TRANSPORT:
        raise OSError("adapter manifest selects an unsupported transport")

    target_step = value.get("target_step")
    base_step = value.get("base_step")
    if type(target_step) is not int or target_step <= 0 or base_step != target_step - 1:
        raise OSError("adapter manifest steps are not consecutive")
    parsed_source = urlsplit(source_url)
    if parsed_source.scheme not in ("http", "https") or not parsed_source.hostname:
        raise OSError("adapter source URL must use HTTP or HTTPS")
    match = re.fullmatch(r"step_(\d+)", Path(parsed_source.path.rstrip("/")).name)
    if match is None or int(match.group(1)) != target_step:
        raise OSError("adapter source URL does not identify target step")
    base_path = str(Path(parsed_source.path.rstrip("/")).parent / f"step_{base_step}")
    base_url = urlunsplit(
        (parsed_source.scheme, parsed_source.netloc, base_path, parsed_source.query, parsed_source.fragment)
    )

    base_sha256 = value.get("base_sha256")
    target_sha256 = value.get("target_sha256")
    full_sha256 = value.get("full_sha256")
    target_size = value.get("target_size")
    full_size = value.get("full_size")
    config_sha256 = value.get("config_sha256")
    config_size = value.get("config_size")
    base_config_sha256 = value.get("base_config_sha256")

    def valid_digest(item: object) -> bool:
        return isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) is not None

    if (
        not valid_digest(base_sha256)
        or not valid_digest(target_sha256)
        or target_sha256 != full_sha256
        or type(target_size) is not int
        or target_size <= 0
        or target_size != full_size
        or not valid_digest(config_sha256)
        or type(config_size) is not int
        or config_size < 0
        or not valid_digest(base_config_sha256)
    ):
        raise OSError("adapter transport manifest is missing target identity")

    raw_chunks = value.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks or len(raw_chunks) > _ADAPTER_MAX_CHUNKS:
        raise OSError("adapter transport manifest has an invalid chunk list")
    chunks: list[dict[str, object]] = []
    seen_files: set[str] = set()
    for item in raw_chunks:
        _check_deadline(deadline)
        if not isinstance(item, dict):
            raise OSError("adapter transport chunk is not an object")
        name = item.get("file")
        offset = item.get("offset")
        raw_size = item.get("raw_size")
        wire_size = item.get("wire_size")
        wire_sha256 = item.get("wire_sha256")
        raw_sha256 = item.get("raw_sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in seen_files
            or type(offset) is not int
            or offset < 0
            or type(raw_size) is not int
            or raw_size <= 0
            or type(wire_size) is not int
            or wire_size <= 0
            or offset + raw_size > target_size
            or not valid_digest(wire_sha256)
            or not valid_digest(raw_sha256)
            or item.get("codec") != "zstd"
        ):
            raise OSError("adapter transport chunk identity is invalid")
        seen_files.add(name)
        chunks.append(
            {
                "file": name,
                "offset": offset,
                "raw_size": raw_size,
                "wire_size": wire_size,
                "wire_sha256": wire_sha256,
                "raw_sha256": raw_sha256,
            }
        )
    chunks.sort(key=lambda item: int(item["offset"]))
    expected_offset = 0
    for item in chunks:
        if item["offset"] != expected_offset:
            raise OSError("adapter transport chunks do not cover the target exactly")
        expected_offset += int(item["raw_size"])
    if expected_offset != target_size:
        raise OSError("adapter transport chunks do not cover the target")
    _check_deadline(deadline)
    chunk_count = value.get("chunk_count")
    if chunk_count is not None and chunk_count != len(chunks):
        raise OSError("adapter transport chunk count differs")
    return {
        "version": 1,
        "mode": "delta",
        "transport": _ADAPTER_DELTA_TRANSPORT,
        "base_url": base_url,
        "base_sha256": base_sha256,
        "target_sha256": target_sha256,
        "target_size": target_size,
        "config_sha256": config_sha256,
        "config_size": config_size,
        "base_config_sha256": base_config_sha256,
        "chunks": chunks,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    with temporary.open("rb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_stage_status(cache_root: Path, value: dict[str, object]) -> None:
    # Calls can come from the range workers. The global staging flock prevents
    # cross-process stages; this lock keeps status writes ordered within one.
    with _JSON_LOCK:
        _write_json(cache_root / _ADAPTER_STATUS, value)


def adapter_stage_status() -> dict[str, object]:
    cache_root = Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora"))
    path = cache_root / _ADAPTER_STATUS
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (FileNotFoundError, OSError, ValueError):
        return {"status": "idle", "bytes_done": 0, "bytes_total": 0}
    return value if isinstance(value, dict) else {"status": "idle", "bytes_done": 0, "bytes_total": 0}


def _adapter_reader_lock(path: Path) -> Path:
    return path.parent / f".{path.name}.readers.lock"


@contextmanager
def adapter_reader_lock(path: Path):
    """Hold a shared lease while vLLM reads an adapter from the cache."""
    lock_path = _adapter_reader_lock(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _cleanup_retired_adapters(cache_root: Path, active_name: str) -> None:
    prefix = f"{active_name}.retired-"
    for candidate in cache_root.iterdir():
        if not candidate.name.startswith(prefix) or not candidate.is_dir():
            continue
        lock_path = _adapter_reader_lock(cache_root / active_name)
        try:
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                shutil.rmtree(candidate)
                fcntl.flock(lock, fcntl.LOCK_UN)
        except (BlockingIOError, FileNotFoundError):
            # A vLLM load still has the old directory open. Leave it for the
            # next stage; deleting it here would create stale open files.
            continue


def _cleanup_previous_adapter_caches(cache_root: Path, active: Path) -> None:
    """Remove old adapters while retaining the active predecessor."""
    if not cache_root.exists():
        return

    ready_candidates: list[Path] = []
    for candidate in cache_root.iterdir():
        if (
            candidate == active
            or not candidate.is_dir()
            or re.fullmatch(r"[0-9a-f]{64}", candidate.name) is None
            or not (candidate / "READY").is_file()
        ):
            continue
        ready_candidates.append(candidate)

    keep = {active}
    try:
        transport = json.loads((active / "transport.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        transport = None
    base_url = transport.get("base_url") if isinstance(transport, dict) else None
    if isinstance(base_url, str):
        predecessor = adapter_cache_path(base_url)
        if predecessor in ready_candidates:
            keep.add(predecessor)

    for candidate in ready_candidates:
        if candidate in keep:
            continue
        try:
            with _adapter_reader_lock(candidate).open("a+") as reader_lock:
                fcntl.flock(reader_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if candidate.exists() and (candidate / "READY").is_file():
                    shutil.rmtree(candidate)
                fcntl.flock(reader_lock, fcntl.LOCK_UN)
        except (BlockingIOError, FileNotFoundError):
            continue


def _download_range(
    source_urls: list[str],
    source_index: int,
    destination: Path,
    start: int,
    end: int,
    progress=None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Download one inclusive byte range and resume its local part file."""
    size = end - start + 1
    for attempt in range(_ADAPTER_LANE_RETRIES):
        if cancel_event is not None and cancel_event.is_set():
            return
        _check_deadline(deadline)
        try:
            source_url = source_urls[(source_index + attempt) % len(source_urls)]
            present = destination.stat().st_size if destination.exists() else 0
            if present == size:
                return
            if present > size:
                destination.unlink()
                present = 0
            request = urllib.request.Request(
                source_url,
                headers={"Range": f"bytes={start + present}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=_request_timeout(deadline)) as response:
                if response.status != 206:
                    raise OSError("adapter server did not honor byte range")
                with destination.open("ab" if present else "wb") as output:
                    while present < size:
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        _check_deadline(deadline)
                        chunk = response.read(min(16 * 1024 * 1024, size - present))
                        if not chunk:
                            raise OSError(f"adapter range ended early: {destination.name}")
                        _check_deadline(deadline)
                        output.write(chunk)
                        present += len(chunk)
                        _check_deadline(deadline)
                        if progress is not None:
                            progress(source_index, len(chunk))
            if destination.stat().st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except TimeoutError:
            raise
        except OSError:
            if attempt + 1 == _ADAPTER_LANE_RETRIES:
                raise
            sleep_for = min(2**attempt, 10)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
                sleep_for = min(sleep_for, remaining)
            if cancel_event is None:
                time.sleep(sleep_for)
            else:
                cancel_event.wait(sleep_for)


def _sha256(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            _check_deadline(deadline)
            chunk = source.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    _check_deadline(deadline)
    return digest.hexdigest()


def _decode_transport_chunk(payload: bytes, *, codec: str, raw_size: int) -> bytes:
    """Use the shared bounded decoder for one transport chunk."""
    try:
        from prime_rl.utils.adapter_delta import decode_adapter_chunk
    except ImportError as exc:
        raise OSError("chunked adapter transport decoder is unavailable") from exc
    try:
        decoded = decode_adapter_chunk(
            payload,
            codec=codec,
            max_output_size=raw_size,
        )
    except Exception as exc:
        raise OSError("adapter transport chunk decompression failed") from exc
    if not isinstance(decoded, bytes) or len(decoded) != raw_size:
        raise OSError("adapter transport chunk has the wrong decoded size")
    return decoded


def _verified_adapter_base(
    base_url: str,
    expected_sha256: str,
    *,
    expected_config_sha256: str | None = None,
    deadline: float | None = None,
) -> tuple[Path, Path]:
    """Return a cached base only after checking its published identity."""
    _check_deadline(deadline)
    base_dir = adapter_cache_path(base_url)
    model = base_dir / _ADAPTER_FILES[1]
    config = base_dir / _ADAPTER_FILES[0]
    if not (base_dir / "READY").is_file() or not model.is_file() or not config.is_file():
        raise OSError("required base adapter is not cached")
    try:
        local_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OSError("cached base adapter has no valid manifest") from exc
    if not isinstance(local_manifest, dict) or local_manifest.get("source_url") != base_url:
        raise OSError("cached base adapter URL differs from requested base")
    files = local_manifest.get("files")
    if not isinstance(files, dict):
        raise OSError("cached base adapter manifest has no files")
    model_spec = files.get(_ADAPTER_FILES[1])
    config_spec = files.get(_ADAPTER_FILES[0])
    if not isinstance(model_spec, dict) or model_spec.get("sha256") != expected_sha256:
        raise OSError("cached base adapter manifest checksum differs")
    if expected_config_sha256 is not None and (
        not isinstance(config_spec, dict) or config_spec.get("sha256") != expected_config_sha256
    ):
        raise OSError("cached base adapter configuration checksum differs")
    checksum = base_dir / _ADAPTER_DIGEST
    if not checksum.is_file() or _read_checksum(checksum) != expected_sha256:
        raise OSError("cached base adapter sidecar checksum differs")
    if expected_config_sha256 is not None and _sha256(config, deadline) != expected_config_sha256:
        raise OSError("cached base adapter configuration bytes differ")
    _check_deadline(deadline)
    return model, config


def _read_checksum(path: Path) -> str:
    value = path.read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OSError(f"invalid adapter checksum: {path.name}")
    return value


def _preallocate(path: Path, size: int) -> None:
    with path.open("wb") as output:
        if not hasattr(os, "posix_fallocate"):
            raise OSError("adapter receiver requires posix_fallocate")
        os.posix_fallocate(output.fileno(), 0, size)


def _pwrite_all(file_descriptor: int, value: bytes, offset: int) -> None:
    written = 0
    while written < len(value):
        count = os.pwrite(file_descriptor, value[written:], offset + written)
        if count <= 0:
            raise OSError("adapter transport output write failed")
        written += count


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    try:
        from prime_rl.utils.adapter_delta import _xor
    except ImportError as exc:
        raise OSError("chunked adapter XOR helper is unavailable") from exc
    try:
        return _xor(left, right)
    except Exception as exc:
        raise OSError("adapter transport XOR failed") from exc


def _transport_raw_sha256(delta: bytes) -> str:
    try:
        from prime_rl.utils.adapter_delta import shuffle_bf16_bytes
    except ImportError as exc:
        raise OSError("chunked adapter shuffle helper is unavailable") from exc
    return hashlib.sha256(shuffle_bf16_bytes(delta)).hexdigest()


def _cleanup_delta_stage_artifacts(destination: Path) -> None:
    """Remove private delta staging paths after all transfer workers stop."""
    for suffix in (".partial", ".parts", ".assembling"):
        candidate = destination.with_name(f"{destination.name}{suffix}")
        try:
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
            else:
                candidate.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


def _stage_chunked_xor_adapter(
    source_url: str,
    transport: dict[str, object],
    source_manifest: dict[str, object],
    deadline: float | None = None,
) -> Path:
    destination = adapter_cache_path(source_url)
    started = time.monotonic()
    _check_deadline(deadline)
    base_url = transport["base_url"]
    base_sha256 = transport["base_sha256"]
    target_sha256 = transport["target_sha256"]
    target_size = transport["target_size"]
    config_sha256 = transport["config_sha256"]
    assert isinstance(base_url, str)
    assert isinstance(base_sha256, str)
    assert isinstance(target_sha256, str)
    assert isinstance(target_size, int)
    assert isinstance(config_sha256, str)
    base_model, _ = _verified_adapter_base(
        base_url,
        base_sha256,
        expected_config_sha256=(
            transport["base_config_sha256"]
            if isinstance(transport.get("base_config_sha256"), str)
            else None
        ),
        deadline=deadline,
    )
    chunks = transport["chunks"]
    assert isinstance(chunks, list)
    chunks = sorted(chunks, key=lambda item: int(item["offset"]))
    temporary = destination.with_name(f"{destination.name}.partial")
    transport_identity = json.dumps(transport, sort_keys=True, separators=(",", ":"))
    identity_path = temporary / "transport-manifest.json"
    if identity_path.is_file():
        try:
            if identity_path.read_text(encoding="utf-8") != transport_identity:
                shutil.rmtree(temporary)
        except OSError:
            shutil.rmtree(temporary)
    temporary.mkdir(exist_ok=True)
    identity_path.write_text(transport_identity, encoding="utf-8")
    chunk_root = temporary / "transport-chunks"
    chunk_root.mkdir(exist_ok=True)
    total_wire_bytes = sum(int(item["wire_size"]) for item in chunks)
    config_size = transport.get("config_size")
    if isinstance(config_size, int):
        total_wire_bytes += config_size
    progress_lock = threading.Lock()
    progress_bytes = sum(
        int(item["wire_size"])
        for item in chunks
        if (
            isinstance(item, dict)
            and isinstance(item.get("file"), str)
            and (
                chunk_root / Path(str(item["file"])).name
            ).is_file()
            and (chunk_root / Path(str(item["file"])).name).stat().st_size == int(item["wire_size"])
        )
    )
    last_status = 0.0

    def report(lane: int, amount: int) -> None:
        nonlocal progress_bytes, last_status
        _check_deadline(deadline)
        with progress_lock:
            progress_bytes += amount
            now = time.monotonic()
            if now - last_status < _ADAPTER_STATUS_INTERVAL_S and progress_bytes < total_wire_bytes:
                return
            last_status = now
            _write_stage_status(
                destination.parent,
                {
                    "status": "staging",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "transport": _ADAPTER_DELTA_TRANSPORT,
                    "bytes_done": progress_bytes,
                    "bytes_total": total_wire_bytes,
                    "lane": lane,
                    "updated_at": time.time(),
                    "elapsed_seconds": round(now - started, 3),
                },
            )

    _write_stage_status(
        destination.parent,
        {
            "status": "staging",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_DELTA_TRANSPORT,
            "bytes_done": progress_bytes,
            "bytes_total": total_wire_bytes,
            "updated_at": time.time(),
            "elapsed_seconds": 0.0,
        },
    )

    lane_urls = _adapter_source_urls(source_url)
    output_path = temporary / _ADAPTER_FILES[1]
    if not output_path.is_file() or output_path.stat().st_size != target_size:
        _preallocate(output_path, target_size)

    cancel_event = threading.Event()

    def fetch(index: int, item: dict[str, object]) -> None:
        if cancel_event.is_set():
            return
        name = item["file"]
        assert isinstance(name, str)
        relative = Path(name)
        chunk_path = chunk_root / relative
        chunk_path.parent.mkdir(parents=True, exist_ok=True)

        def lane_report(_lane: int, amount: int) -> None:
            report(index % len(lane_urls), amount)

        lane_url = lane_urls[index % len(lane_urls)]
        chunk_url = f"{lane_url.rstrip('/')}/{name}"
        download(
            chunk_url,
            chunk_path,
            expected_size=int(item["wire_size"]),
            expected_sha256=str(item["wire_sha256"]),
            progress=lane_report,
            source_urls=[chunk_url],
            deadline=deadline,
            cancel_event=cancel_event,
        )

        if cancel_event.is_set():
            return
        _check_deadline(deadline)
        encoded = chunk_path.read_bytes()
        if len(encoded) != int(item["wire_size"]):
            raise OSError("adapter transport chunk size changed")
        if _sha256(chunk_path, deadline) != item["wire_sha256"]:
            raise OSError("adapter transport chunk checksum mismatch")
        _check_deadline(deadline)
        raw = _decode_transport_chunk(
            encoded,
            codec=str(transport["transport"]),
            raw_size=int(item["raw_size"]),
        )
        _check_deadline(deadline)
        raw_sha256 = item.get("raw_sha256")
        if _transport_raw_sha256(raw) != raw_sha256:
            raise OSError("adapter transport decoded chunk checksum mismatch")
        offset = int(item["offset"])
        if cancel_event.is_set():
            return
        base_fd = os.open(base_model, os.O_RDONLY)
        output_fd = os.open(output_path, os.O_RDWR)
        try:
            base_chunk = os.pread(base_fd, len(raw), offset)
            if len(base_chunk) != len(raw):
                raise OSError("cached base adapter ended before transport chunk")
            _pwrite_all(output_fd, _xor_bytes(base_chunk, raw), offset)
        finally:
            os.close(output_fd)
            os.close(base_fd)
        _check_deadline(deadline)
        chunk_path.unlink()

    executor = ThreadPoolExecutor(max_workers=min(len(chunks), _PARALLEL_DOWNLOAD_STREAMS))
    futures = [executor.submit(fetch, index, item) for index, item in enumerate(chunks)]
    _wait_for_futures(executor, futures, deadline, cancel_event)

    if _sha256(output_path, deadline) != target_sha256:
        raise OSError("reconstructed adapter checksum mismatch")
    if output_path.stat().st_size != target_size:
        raise OSError("reconstructed adapter size mismatch")

    config_path = temporary / _ADAPTER_FILES[0]
    download(
        f"{source_url.rstrip('/')}/{_ADAPTER_FILES[0]}",
        config_path,
        expected_size=(config_size if isinstance(config_size, int) else None),
        expected_sha256=config_sha256,
        progress=report,
        deadline=deadline,
    )
    if _sha256(config_path, deadline) != config_sha256:
        raise OSError("adapter configuration checksum mismatch")
    shutil.rmtree(chunk_root)
    (temporary / _ADAPTER_DIGEST).write_text(f"{target_sha256}\n", encoding="ascii")
    _write_json(temporary / "manifest.json", source_manifest)
    _write_json(
        temporary / "transport.json",
        {
            "mode": _ADAPTER_DELTA_TRANSPORT,
            "source_url": source_url,
            "base_url": base_url,
            "base_sha256": base_sha256,
            "target_sha256": target_sha256,
        },
    )
    ready = temporary / "READY"
    _check_deadline(deadline)
    ready.write_text("ready\n", encoding="utf-8")
    with ready.open("rb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with _adapter_reader_lock(destination).open("a+") as reader_lock:
        _acquire_stage_lock(reader_lock, deadline)
        _check_deadline(deadline)
        if destination.exists():
            retired = destination.with_name(
                f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}"
            )
            os.replace(destination, retired)
        os.replace(temporary, destination)
        fcntl.flock(reader_lock, fcntl.LOCK_UN)
    _cleanup_retired_adapters(destination.parent, destination.name)
    _cleanup_previous_adapter_caches(destination.parent, destination)
    _write_stage_status(
        destination.parent,
        {
            "status": "ready",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_DELTA_TRANSPORT,
            "bytes_done": total_wire_bytes,
            "bytes_total": total_wire_bytes,
            "updated_at": time.time(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    return destination


def download(
    source_url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    progress=None,
    source_urls: list[str] | None = None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    _check_deadline(deadline)
    explicit_source_urls = source_urls is not None
    source_urls = _adapter_source_urls(source_url) if source_urls is None else source_urls
    if not source_urls:
        raise OSError("adapter download has no source lane")
    total = (
        expected_size
        if expected_size is not None
        else _content_length(source_urls[0], deadline=deadline)
    )
    _check_deadline(deadline)
    if destination.exists() and destination.stat().st_size == total:
        if expected_sha256 is None or _sha256(destination, deadline) == expected_sha256:
            return
        destination.unlink()

    # Full-checkpoint tunnels are not adapter sources. Reusing their 128-port
    # list here created 128 adapter lanes on every replica and collapsed the
    # link. Adapter fan-out is opt-in and has its own setting.
    adapter_ports = explicit_source_urls or len(source_urls) > 1 or source_urls[0] != source_url

    streams = len(source_urls) if adapter_ports else max(
        1, int(os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS", _PARALLEL_DOWNLOAD_STREAMS))
    )
    if total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        # Preserve the old single-file resume path for small metadata files and
        # for an explicitly disabled parallel transfer.
        for attempt in range(_ADAPTER_LANE_RETRIES):
            if cancel_event is not None and cancel_event.is_set():
                return
            _check_deadline(deadline)
            try:
                offset = destination.stat().st_size if destination.exists() else 0
                if offset > total:
                    destination.unlink()
                    offset = 0
                request = urllib.request.Request(
                    source_urls[attempt % len(source_urls)],
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                )
                with urllib.request.urlopen(request, timeout=_request_timeout(deadline)) as response:
                    if offset and response.status != 206:
                        destination.unlink(missing_ok=True)
                        raise OSError("adapter server did not honor byte range")
                    with destination.open("ab" if offset else "wb") as output:
                        while offset < total:
                            if cancel_event is not None and cancel_event.is_set():
                                return
                            _check_deadline(deadline)
                            chunk = response.read(min(16 * 1024 * 1024, total - offset))
                            if not chunk:
                                raise OSError(f"adapter file ended early: {destination.name}")
                            output.write(chunk)
                            offset += len(chunk)
                            _check_deadline(deadline)
                            if progress is not None:
                                progress(0, len(chunk))
                if destination.stat().st_size != total:
                    raise OSError(f"incomplete adapter file: {destination.name}")
                if expected_sha256 is not None and _sha256(destination, deadline) != expected_sha256:
                    # A complete but bad file must not be resumed from EOF on
                    # the next bounded attempt.
                    destination.unlink(missing_ok=True)
                    raise OSError(f"adapter checksum mismatch: {destination.name}")
                return
            except TimeoutError:
                raise
            except OSError:
                if attempt + 1 == _ADAPTER_LANE_RETRIES:
                    raise
                sleep_for = min(2**attempt, 10)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter delta staging exceeded its time limit")
                    sleep_for = min(sleep_for, remaining)
                if cancel_event is None:
                    time.sleep(sleep_for)
                else:
                    cancel_event.wait(sleep_for)
        return

    # Long-distance adapter downloads otherwise use one TCP congestion window
    # across a 217 ms, lossy link. Independent range files are resumable and
    # let one reset retry without discarding the other completed ranges.
    parts_dir = destination.with_name(f"{destination.name}.parts")
    parts_dir.mkdir(exist_ok=True)
    chunk = (total + streams - 1) // streams
    ranges = [
        (index, start, min(start + chunk, total) - 1)
        for index, start in enumerate(range(0, total, chunk))
    ]

    identity = {
        "source_url": source_url,
        "size": total,
        "sha256": expected_sha256,
        "streams": len(ranges),
    }
    identity_path = parts_dir / "manifest.json"
    try:
        with identity_path.open("r", encoding="utf-8") as source:
            existing_identity = json.load(source)
    except (FileNotFoundError, OSError, ValueError):
        existing_identity = None
    if existing_identity != identity:
        # Parts are private to this transfer. Remove only mismatched private
        # parts; never remove a published adapter directory here.
        shutil.rmtree(parts_dir)
        parts_dir.mkdir()
        _write_json(identity_path, identity)

    executor = ThreadPoolExecutor(max_workers=len(ranges))
    cancel_event = cancel_event or threading.Event()
    futures = [
        executor.submit(
            _download_range,
            source_urls,
            index,
            parts_dir / f"part-{index:03d}",
            start,
            end,
            progress,
            deadline,
            cancel_event,
        )
        for index, start, end in ranges
    ]
    _wait_for_futures(executor, futures, deadline, cancel_event)

    assembling = destination.with_name(f"{destination.name}.assembling")
    with assembling.open("wb") as output:
        for index, _, _ in ranges:
            with (parts_dir / f"part-{index:03d}").open("rb") as part:
                while True:
                    _check_deadline(deadline)
                    chunk_data = part.read(16 * 1024 * 1024)
                    if not chunk_data:
                        break
                    output.write(chunk_data)
                    _check_deadline(deadline)
    if assembling.stat().st_size != total:
        raise OSError(f"assembled adapter file has the wrong size: {destination.name}")
    if expected_sha256 is not None and _sha256(assembling, deadline) != expected_sha256:
        # Part sizes alone cannot detect a bad byte from a reset connection.
        # Drop the private parts after a whole-file mismatch so the next
        # stage attempt does not repeatedly assemble the same bad payload.
        shutil.rmtree(parts_dir)
        assembling.unlink(missing_ok=True)
        raise OSError(f"adapter checksum mismatch: {destination.name}")
    assembling.replace(destination)
    shutil.rmtree(parts_dir)


def stage_lora_adapter_files(source_url: str, require_delta: bool = False) -> Path:
    """Download one remote adapter into the host-shared cache.

    The API server can call this before it pauses generation. Worker-side
    loading calls the same helper as a fail-safe, but returns immediately when
    the prefetch already wrote ``READY``.
    """
    destination = adapter_cache_path(source_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    delta_deadline = started + _delta_stage_timeout_s() if require_delta else None
    # The cache is shared by all replicas on this host. One global writer is
    # intentional: per-URL writers caused old policies to queue behind one
    # another and made it possible to publish a stale policy after a newer one.
    with (destination.parent / "adapter-staging.lock").open("w") as host_lock:
        _acquire_stage_lock(host_lock, delta_deadline)
        transport: dict[str, object] | None = None
        try:
            transport = _read_chunked_xor_manifest(source_url, deadline=delta_deadline)
            if (
                delta_deadline is None
                and transport is not None
                and transport.get("mode") == "delta"
            ):
                delta_deadline = time.monotonic() + _delta_stage_timeout_s()
            manifest = _source_manifest(
                source_url,
                deadline=delta_deadline
                if require_delta
                or (transport is not None and transport.get("mode") == "delta")
                else None,
            )
            files = manifest["files"]
            assert isinstance(files, dict)

            if require_delta and (transport is None or transport.get("mode") != "delta"):
                raise OSError("delta adapter transport is required; full adapter fallback is disabled")

            if transport is not None and transport.get("mode") == "delta":
                target_spec = files.get(_ADAPTER_FILES[1])
                config_spec = files.get(_ADAPTER_FILES[0])
                if (
                    not isinstance(target_spec, dict)
                    or target_spec.get("sha256") != transport["target_sha256"]
                    or target_spec.get("size") != transport["target_size"]
                    or not isinstance(config_spec, dict)
                    or config_spec.get("sha256") != transport["config_sha256"]
                    or config_spec.get("size") != transport["config_size"]
                ):
                    raise OSError("adapter transport target differs from source manifest")
                if (destination / "READY").is_file():
                    try:
                        with (destination / "transport.json").open("r", encoding="utf-8") as source:
                            proof = json.load(source)
                    except (FileNotFoundError, OSError, ValueError):
                        proof = None
                    if (
                        isinstance(proof, dict)
                        and proof.get("mode") == _ADAPTER_DELTA_TRANSPORT
                        and proof.get("source_url") == source_url
                        and proof.get("target_sha256") == transport["target_sha256"]
                        and (destination / _ADAPTER_FILES[1]).is_file()
                        and _sha256(destination / _ADAPTER_FILES[1], delta_deadline)
                        == transport["target_sha256"]
                        and (destination / _ADAPTER_FILES[0]).is_file()
                        and _sha256(destination / _ADAPTER_FILES[0], delta_deadline)
                        == transport["config_sha256"]
                    ):
                        _write_stage_status(
                            destination.parent,
                            {
                                "status": "ready",
                                "source_url": source_url,
                                "path": destination.as_posix(),
                                "transport": _ADAPTER_DELTA_TRANSPORT,
                                "bytes_done": 0,
                                "bytes_total": 0,
                                "updated_at": time.time(),
                            },
                        )
                        _cleanup_retired_adapters(destination.parent, destination.name)
                        _cleanup_previous_adapter_caches(destination.parent, destination)
                        return destination
                _check_deadline(delta_deadline)
                return _stage_chunked_xor_adapter(
                    source_url, transport, manifest, deadline=delta_deadline
                )
            transport = None

            if (destination / "READY").is_file():
                try:
                    with (destination / "manifest.json").open("r", encoding="utf-8") as source:
                        local_manifest = json.load(source)
                except (FileNotFoundError, OSError, ValueError):
                    local_manifest = None
                if local_manifest == manifest:
                    _write_stage_status(
                        destination.parent,
                        {
                            "status": "ready",
                            "source_url": source_url,
                            "path": destination.as_posix(),
                            "bytes_done": sum(int(item["size"]) for item in files.values()),
                            "bytes_total": sum(int(item["size"]) for item in files.values()),
                            "updated_at": time.time(),
                        },
                    )
                    _cleanup_retired_adapters(destination.parent, destination.name)
                    _cleanup_previous_adapter_caches(destination.parent, destination)
                    return destination

            total_bytes = sum(int(item["size"]) for item in files.values())
            progress_lock = threading.Lock()
            progress_bytes = 0
            last_status = 0.0

            def report(lane: int, amount: int) -> None:
                nonlocal progress_bytes, last_status
                with progress_lock:
                    progress_bytes += amount
                    now = time.monotonic()
                    if now - last_status < _ADAPTER_STATUS_INTERVAL_S and progress_bytes < total_bytes:
                        return
                    last_status = now
                    _write_stage_status(
                        destination.parent,
                        {
                            "status": "staging",
                            "source_url": source_url,
                            "path": destination.as_posix(),
                            "bytes_done": progress_bytes,
                            "bytes_total": total_bytes,
                            "lane": lane,
                            "updated_at": time.time(),
                            "elapsed_seconds": round(now - started, 3),
                        },
                    )

            _write_stage_status(
                destination.parent,
                {
                    "status": "staging",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "bytes_done": 0,
                    "bytes_total": total_bytes,
                    "updated_at": time.time(),
                    "elapsed_seconds": 0.0,
                },
            )
            temporary = destination.with_name(f"{destination.name}.partial")
            temporary.mkdir(exist_ok=True)
            _write_json(temporary / "manifest.json", manifest)
            for filename in _ADAPTER_FILES:
                item = files[filename]
                assert isinstance(item, dict)
                expected_sha256 = item.get("sha256")
                download(
                    f"{source_url.rstrip('/')}/{filename}",
                    temporary / filename,
                    expected_size=int(item["size"]),
                    expected_sha256=expected_sha256 if isinstance(expected_sha256, str) else None,
                    progress=report,
                )
                if expected_sha256 is None:
                    item["sha256"] = _sha256(temporary / filename)
                progress_bytes = sum(
                    int(files[name]["size"])
                    for name in _ADAPTER_FILES
                    if (temporary / name).is_file() and (temporary / name).stat().st_size == int(files[name]["size"])
                )
            _write_json(temporary / "manifest.json", manifest)
            model_digest = files[_ADAPTER_FILES[1]].get("sha256")
            if not isinstance(model_digest, str):
                raise OSError("adapter source manifest has no model checksum")
            (temporary / _ADAPTER_DIGEST).write_text(f"{model_digest}\n", encoding="ascii")
            ready = temporary / "READY"
            ready.write_text("ready\n", encoding="utf-8")
            with ready.open("rb") as stream:
                os.fsync(stream.fileno())

            # Keep the exclusive lease through both renames. This removes the
            # short missing-path window between retiring the old directory and
            # publishing the new one.
            with _adapter_reader_lock(destination).open("a+") as reader_lock:
                fcntl.flock(reader_lock, fcntl.LOCK_EX)
                if destination.exists():
                    retired = destination.with_name(
                        f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}"
                    )
                    os.replace(destination, retired)
                os.replace(temporary, destination)
                fcntl.flock(reader_lock, fcntl.LOCK_UN)
            _cleanup_retired_adapters(destination.parent, destination.name)
            _cleanup_previous_adapter_caches(destination.parent, destination)
            _write_stage_status(
                destination.parent,
                {
                    "status": "ready",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "bytes_done": total_bytes,
                    "bytes_total": total_bytes,
                    "updated_at": time.time(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            return destination
        except BaseException as exc:
            if require_delta or (isinstance(transport, dict) and transport.get("mode") == "delta"):
                _cleanup_delta_stage_artifacts(destination)
            _write_stage_status(
                destination.parent,
                {
                    "status": "failed",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            raise


def full_checkpoint_cache_path(source_url: str) -> Path:
    return Path(os.environ.get("PRIME_FULL_CHECKPOINT_CACHE_DIR", "/tmp/prime-rl-checkpoints")) / hashlib.sha256(
        source_url.encode()
    ).hexdigest()


def prune_full_checkpoint_cache(cache_root: Path, keep: Path) -> list[Path]:
    """Remove checkpoint directories that are older than the policy being staged."""
    removed: list[Path] = []
    if not cache_root.exists():
        return removed
    keep_names = {keep.name, f"{keep.name}.partial"}
    for candidate in cache_root.iterdir():
        if not candidate.is_dir() or candidate.name in keep_names:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


def stage_full_checkpoint_files(source_url: str) -> Path:
    """Download and atomically publish one remote sharded safetensors checkpoint."""
    destination = full_checkpoint_cache_path(source_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f"{destination.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "READY").exists():
            return destination
        temporary = destination.with_name(f"{destination.name}.partial")
        temporary.mkdir(exist_ok=True)
        index_name = "model.safetensors.index.json"
        used_delta = False
        try:
            with urllib.request.urlopen(
                f"{source_url.rstrip('/')}/{DELTA_MANIFEST}", timeout=60
            ) as response:
                delta_manifest = json.load(response)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            delta_manifest = None

        if isinstance(delta_manifest, dict) and delta_manifest.get("version") == 1:
            base_url = delta_manifest.get("base_url")
            if (
                delta_manifest.get("target_url") == source_url
                and isinstance(base_url, str)
            ):
                base_dir = full_checkpoint_cache_path(base_url)
                if (base_dir / "READY").exists():
                    files = delta_manifest.get("files")
                    if not isinstance(files, list) or not files:
                        raise OSError("delta manifest has no checkpoint files")
                    for spec in files:
                        filename = spec.get("filename")
                        delta_name = spec.get("delta")
                        if (
                            not isinstance(filename, str)
                            or Path(filename).name != filename
                            or not filename.endswith(".safetensors")
                            or not isinstance(delta_name, str)
                            or Path(delta_name).name != delta_name
                        ):
                            raise OSError("unsafe checkpoint delta filename")
                        delta_path = temporary / delta_name
                        download(f"{source_url.rstrip('/')}/{delta_name}", delta_path)
                        apply_xor_delta_file(
                            base_dir / filename,
                            delta_path,
                            temporary / filename,
                            expected_size=int(spec["size"]),
                            expected_sha256=str(spec["sha256"]),
                        )
                        delta_path.unlink()
                    index_path = temporary / index_name
                    download(f"{source_url.rstrip('/')}/{index_name}", index_path)
                    if sha256_file(index_path) != delta_manifest.get("index_sha256"):
                        raise OSError("checkpoint index checksum mismatch")
                    used_delta = True

        if not used_delta:
            index_path = temporary / index_name
            download(f"{source_url.rstrip('/')}/{index_name}", index_path)
            with index_path.open("r", encoding="utf-8") as source:
                index = json.load(source)
            shard_names = sorted(set(index["weight_map"].values()))
            if not shard_names:
                raise OSError("full checkpoint index contains no weight shards")
            for filename in shard_names:
                if Path(filename).name != filename or not filename.endswith(".safetensors"):
                    raise OSError(f"unsafe checkpoint shard name: {filename!r}")
                download(f"{source_url.rstrip('/')}/{filename}", temporary / filename)
        (temporary / "READY").touch()
        shutil.rmtree(destination, ignore_errors=True)
        temporary.rename(destination)
        # Keep the old policy until the replacement is complete. Generation
        # still uses it while staging, and a lossless delta needs its files.
        prune_full_checkpoint_cache(destination.parent, destination)
    return destination


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

    def stage_lora_adapter(self, source_url: str, require_delta: bool = False) -> None:
        stage_lora_adapter_files(source_url, require_delta=require_delta)

    def stage_full_checkpoint(self, source_url: str) -> str:
        return stage_full_checkpoint_files(source_url).as_posix()

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        if weight_path.startswith(("http://", "https://")):
            weight_path = stage_full_checkpoint_files(weight_path).as_posix()
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
