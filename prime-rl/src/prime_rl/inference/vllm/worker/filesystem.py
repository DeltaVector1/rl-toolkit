import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.request
from urllib.error import HTTPError
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
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
_ADAPTER_LANE_TIMEOUT_S = float(os.environ.get("PRIME_LORA_LANE_TIMEOUT_S", "45"))
_ADAPTER_LANE_RETRIES = int(os.environ.get("PRIME_LORA_LANE_RETRIES", "6"))
_ADAPTER_STATUS_INTERVAL_S = float(os.environ.get("PRIME_LORA_STATUS_INTERVAL_S", "1"))
_JSON_LOCK = threading.Lock()


def _content_length(source_url: str) -> int:
    with urllib.request.urlopen(
        urllib.request.Request(source_url, method="HEAD"), timeout=_ADAPTER_LANE_TIMEOUT_S
    ) as response:
        try:
            return int(response.headers["Content-Length"])
        except (TypeError, ValueError) as exc:
            raise OSError(f"adapter source has no valid Content-Length: {source_url}") from exc


def _read_source_text(source_url: str) -> str:
    return _read_source_bytes(source_url).decode("utf-8")


def _read_source_bytes(source_url: str) -> bytes:
    with urllib.request.urlopen(source_url, timeout=_ADAPTER_LANE_TIMEOUT_S) as response:
        return response.read()


def _source_manifest(source_url: str) -> dict[str, object]:
    """Read the immutable adapter manifest published beside a STABLE marker.

    Older broadcasts have only the model checksum sidecar. Keep that fallback,
    but always return a local manifest with a size and digest for every file.
    """
    source_root = source_url.rstrip("/")
    manifest_url = f"{source_root}/{_ADAPTER_MANIFEST}"
    try:
        value = json.loads(_read_source_text(manifest_url))
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
    digest_text = _read_source_text(f"{source_root}/{_ADAPTER_DIGEST}").strip().split()
    if not digest_text or re.fullmatch(r"[0-9a-f]{64}", digest_text[0]) is None:
        raise OSError("adapter source has no valid model checksum")
    config_bytes = _read_source_bytes(config_url)
    return {
        "version": 1,
        "source_url": source_url,
        "files": {
            _ADAPTER_FILES[0]: {
                "size": _content_length(config_url),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            _ADAPTER_FILES[1]: {"size": _content_length(model_url), "sha256": digest_text[0]},
        },
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


def _download_range(
    source_urls: list[str],
    source_index: int,
    destination: Path,
    start: int,
    end: int,
    progress=None,
) -> None:
    """Download one inclusive byte range and resume its local part file."""
    size = end - start + 1
    for attempt in range(_ADAPTER_LANE_RETRIES):
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
            with urllib.request.urlopen(request, timeout=_ADAPTER_LANE_TIMEOUT_S) as response:
                if response.status != 206:
                    raise OSError("adapter server did not honor byte range")
                with destination.open("ab" if present else "wb") as output:
                    while present < size:
                        chunk = response.read(min(16 * 1024 * 1024, size - present))
                        if not chunk:
                            raise OSError(f"adapter range ended early: {destination.name}")
                        output.write(chunk)
                        present += len(chunk)
                        if progress is not None:
                            progress(source_index, len(chunk))
            if destination.stat().st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except (OSError, TimeoutError):
            if attempt + 1 == _ADAPTER_LANE_RETRIES:
                raise
            time.sleep(min(2**attempt, 10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    source_url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    progress=None,
) -> None:
    total = _content_length(source_url)
    if expected_size is not None and total != expected_size:
        raise OSError(
            f"adapter source size changed for {source_url}: expected {expected_size}, got {total}"
        )
    if destination.exists() and destination.stat().st_size == total:
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return
        destination.unlink()

    # Full-checkpoint tunnels are not adapter sources. Reusing their 128-port
    # list here created 128 adapter lanes on every replica and collapsed the
    # link. Adapter fan-out is opt-in and has its own setting.
    adapter_ports = [
        value.strip()
        for value in os.environ.get("PRIME_LORA_SOURCE_PORTS", "").split(",")
        if value.strip()
    ]
    source_urls = [source_url]
    if adapter_ports:
        parsed = urlsplit(source_url)
        if parsed.hostname not in ("127.0.0.1", "localhost"):
            raise OSError("adapter tunnel pool requires a loopback source URL")
        source_urls = [
            urlunsplit(
                (
                    parsed.scheme,
                    f"{parsed.hostname}:{int(port)}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
            for port in checkpoint_ports
        ]

    streams = len(source_urls) if adapter_ports else max(
        1, int(os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS", _PARALLEL_DOWNLOAD_STREAMS))
    )
    if total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        # Preserve the old single-file resume path for small metadata files and
        # for an explicitly disabled parallel transfer.
        for attempt in range(_ADAPTER_LANE_RETRIES):
            try:
                offset = destination.stat().st_size if destination.exists() else 0
                if offset > total:
                    destination.unlink()
                    offset = 0
                request = urllib.request.Request(
                    source_url,
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                )
                with urllib.request.urlopen(request, timeout=_ADAPTER_LANE_TIMEOUT_S) as response:
                    if offset and response.status != 206:
                        destination.unlink(missing_ok=True)
                        raise OSError("adapter server did not honor byte range")
                    with destination.open("ab" if offset else "wb") as output:
                        while offset < total:
                            chunk = response.read(min(16 * 1024 * 1024, total - offset))
                            if not chunk:
                                raise OSError(f"adapter file ended early: {destination.name}")
                            output.write(chunk)
                            offset += len(chunk)
                            if progress is not None:
                                progress(0, len(chunk))
                if destination.stat().st_size != total:
                    raise OSError(f"incomplete adapter file: {destination.name}")
                if expected_sha256 is not None and _sha256(destination) != expected_sha256:
                    # A complete but bad file must not be resumed from EOF on
                    # the next bounded attempt.
                    destination.unlink(missing_ok=True)
                    raise OSError(f"adapter checksum mismatch: {destination.name}")
                return
            except (OSError, TimeoutError):
                if attempt + 1 == _ADAPTER_LANE_RETRIES:
                    raise
                time.sleep(min(2**attempt, 10))
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

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(
                _download_range,
                source_urls,
                index,
                parts_dir / f"part-{index:03d}",
                start,
                end,
                progress,
            )
            for index, start, end in ranges
        ]
        for future in futures:
            future.result()

    assembling = destination.with_name(f"{destination.name}.assembling")
    with assembling.open("wb") as output:
        for index, _, _ in ranges:
            with (parts_dir / f"part-{index:03d}").open("rb") as part:
                shutil.copyfileobj(part, output, length=16 * 1024 * 1024)
    if assembling.stat().st_size != total:
        raise OSError(f"assembled adapter file has the wrong size: {destination.name}")
    if expected_sha256 is not None and _sha256(assembling) != expected_sha256:
        # Part sizes alone cannot detect a bad byte from a reset connection.
        # Drop the private parts after a whole-file mismatch so the next
        # stage attempt does not repeatedly assemble the same bad payload.
        shutil.rmtree(parts_dir)
        assembling.unlink(missing_ok=True)
        raise OSError(f"adapter checksum mismatch: {destination.name}")
    assembling.replace(destination)
    shutil.rmtree(parts_dir)


def stage_lora_adapter_files(source_url: str) -> Path:
    """Download one remote adapter into the host-shared cache.

    The API server can call this before it pauses generation. Worker-side
    loading calls the same helper as a fail-safe, but returns immediately when
    the prefetch already wrote ``READY``.
    """
    destination = adapter_cache_path(source_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The cache is shared by all replicas on this host. One global writer is
    # intentional: per-URL writers caused old policies to queue behind one
    # another and made it possible to publish a stale policy after a newer one.
    with (destination.parent / "adapter-staging.lock").open("w") as host_lock:
        fcntl.flock(host_lock, fcntl.LOCK_EX)
        started = time.monotonic()
        try:
            manifest = _source_manifest(source_url)
            files = manifest["files"]
            assert isinstance(files, dict)

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

    def stage_lora_adapter(self, source_url: str) -> None:
        stage_lora_adapter_files(source_url)

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
