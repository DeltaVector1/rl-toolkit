import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.request
from urllib.error import HTTPError
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


def _content_length(source_url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(source_url, method="HEAD"), timeout=60) as response:
        return int(response.headers["Content-Length"])


def _download_range(
    source_urls: list[str],
    source_index: int,
    destination: Path,
    start: int,
    end: int,
) -> None:
    """Download one inclusive byte range and resume its local part file."""
    for attempt in range(10):
        try:
            source_url = source_urls[(source_index + attempt) % len(source_urls)]
            size = end - start + 1
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
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise OSError("adapter server did not honor byte range")
                with destination.open("ab" if present else "wb") as output:
                    shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
            if destination.stat().st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 10))


def download(source_url: str, destination: Path) -> None:
    total = _content_length(source_url)
    if destination.exists() and destination.stat().st_size == total:
        return

    checkpoint_ports = [
        value.strip()
        for value in os.environ.get("PRIME_FULL_CHECKPOINT_PORTS", "").split(",")
        if value.strip()
    ]
    source_urls = [source_url]
    if checkpoint_ports:
        parsed = urlsplit(source_url)
        if parsed.hostname not in ("127.0.0.1", "localhost"):
            raise OSError("checkpoint tunnel pool requires a loopback source URL")
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

    streams = len(source_urls) if checkpoint_ports else max(
        1, int(os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS", _PARALLEL_DOWNLOAD_STREAMS))
    )
    if total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        # Preserve the old single-file resume path for small metadata files and
        # for an explicitly disabled parallel transfer.
        for attempt in range(10):
            try:
                offset = destination.stat().st_size if destination.exists() else 0
                if offset > total:
                    destination.unlink()
                    offset = 0
                request = urllib.request.Request(
                    source_url,
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status != 206:
                        destination.unlink(missing_ok=True)
                        raise OSError("adapter server did not honor byte range")
                    with destination.open("ab" if offset else "wb") as output:
                        shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
                if destination.stat().st_size != total:
                    raise OSError(f"incomplete adapter file: {destination.name}")
                return
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(min(2**attempt, 10))
        return

    # Long-distance adapter downloads otherwise use one TCP congestion window
    # across a 217 ms, lossy link. Independent range files are resumable and
    # let one reset retry without discarding the other completed ranges.
    destination.unlink(missing_ok=True)
    parts_dir = destination.with_name(f"{destination.name}.parts")
    parts_dir.mkdir(exist_ok=True)
    chunk = (total + streams - 1) // streams
    ranges = [
        (index, start, min(start + chunk, total) - 1)
        for index, start in enumerate(range(0, total, chunk))
    ]
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(
                _download_range,
                source_urls,
                index,
                parts_dir / f"part-{index:03d}",
                start,
                end,
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
    with (destination.parent / f"{destination.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "READY").exists():
            return destination
        temporary = destination.with_name(f"{destination.name}.partial")
        temporary.mkdir(exist_ok=True)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            download(f"{source_url.rstrip('/')}/{filename}", temporary / filename)
        (temporary / "READY").touch()
        shutil.rmtree(destination, ignore_errors=True)
        temporary.rename(destination)
    return destination


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
