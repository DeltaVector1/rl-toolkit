import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

_PARALLEL_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
_PARALLEL_DOWNLOAD_STREAMS = 8
_GLM_RANGE_GZIP_ENV = "PRIME_GLM_ADAPTER_GZIP"


def adapter_cache_path(source_url: str) -> Path:
    return Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora")) / hashlib.sha256(
        source_url.encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    gzip_required = os.environ.get(_GLM_RANGE_GZIP_ENV, "0")
    if gzip_required not in {"0", "1"}:
        raise OSError(f"{_GLM_RANGE_GZIP_ENV} must be 0 or 1")
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
            requested_start = start + present
            headers = {"Range": f"bytes={requested_start}-{end}"}
            if gzip_required == "1":
                headers["Accept-Encoding"] = "gzip"
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
                if gzip_required == "1":
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
                else:
                    with destination.open("ab" if present else "wb") as output:
                        shutil.copyfileobj(
                            response, output, length=16 * 1024 * 1024
                        )
            if destination.stat().st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except (EOFError, OSError):
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 10))


def _publish_download(temporary: Path, destination: Path, create_only: bool) -> None:
    if create_only:
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    else:
        temporary.replace(destination)


def download(source_url: str, destination: Path, create_only: bool = False) -> None:
    total = _content_length(source_url)
    if create_only and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"adapter destination already exists: {destination}")
    if not create_only and destination.exists() and destination.stat().st_size == total:
        return

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

    streams = len(source_urls) if ports else max(
        1,
        int(os.environ.get("PRIME_ADAPTER_DOWNLOAD_STREAMS", str(_PARALLEL_DOWNLOAD_STREAMS))),
    )
    if total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        if not create_only:
            _download_range([source_url], 0, destination, 0, total - 1)
            return
        assembling = destination.with_name(f"{destination.name}.assembling")
        assembling.unlink(missing_ok=True)
        _download_range([source_url], 0, assembling, 0, total - 1)
        _publish_download(assembling, destination, True)
        return

    if not create_only:
        destination.unlink(missing_ok=True)
    parts = destination.with_name(f"{destination.name}.parts")
    parts.mkdir(exist_ok=True)
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
                parts / f"part-{index:03d}",
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
            with (parts / f"part-{index:03d}").open("rb") as part:
                shutil.copyfileobj(part, output, length=16 * 1024 * 1024)
    if assembling.stat().st_size != total:
        raise OSError(f"assembled adapter file has the wrong size: {destination.name}")
    _publish_download(assembling, destination, create_only)
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
    download(args.source_url, args.destination, create_only=True)
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
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


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

    def stage_lora_adapter(self, source_url: str) -> None:
        destination = adapter_cache_path(source_url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (destination.parent / f"{destination.name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (destination / "READY").exists():
                return
            temporary = destination.with_name(f"{destination.name}.partial")
            temporary.mkdir(exist_ok=True)
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors.sha256",
                "adapter_model.safetensors",
            ):
                download(f"{source_url.rstrip('/')}/{filename}", temporary / filename)
            expected_digest = (temporary / "adapter_model.safetensors.sha256").read_text().strip()
            if len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest):
                raise ValueError("adapter SHA-256 file is invalid")
            actual_digest = _sha256_file(temporary / "adapter_model.safetensors")
            if actual_digest != expected_digest:
                raise ValueError("downloaded adapter SHA-256 does not match the trainer export")
            (temporary / "READY").touch()
            shutil.rmtree(destination, ignore_errors=True)
            temporary.rename(destination)

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
