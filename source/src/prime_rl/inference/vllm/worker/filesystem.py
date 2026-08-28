import argparse
import ctypes
import ctypes.util
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

_PARALLEL_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
_PARALLEL_DOWNLOAD_STREAMS = 8
_GLM_RANGE_GZIP_ENV = "PRIME_GLM_ADAPTER_GZIP"
_GLM_RANGE_CODEC_ENV = "PRIME_GLM_ADAPTER_CODEC"
_GLM_RANGE_CHUNK_BYTES = 256 * 1024 * 1024
_GLM_RANGE_ATTEMPTS = 3
_ZSTD_FRAME_BYTES = 8 * 1024 * 1024
_zstd = None


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
                self._validate_staged_glm_adapter(destination)
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
            self._validate_staged_glm_adapter(temporary)
            (temporary / "READY").touch()
            shutil.rmtree(destination, ignore_errors=True)
            temporary.rename(destination)

    @staticmethod
    def _validate_staged_glm_adapter(adapter_path: Path) -> dict[str, object] | None:
        try:
            from prime_rl.inference.vllm.adapter_validation import (
                expected_model_identity,
                validation_enabled,
                validate_glm52_adapter,
            )

            if validation_enabled():
                identity = expected_model_identity()
                return validate_glm52_adapter(
                    adapter_path,
                    expected_repo_id=identity["repo_id"],
                    expected_revision=identity["revision"],
                    expected_variant=identity["variant"],
                    expected_content_sha256=identity["content_sha256"],
                    expected_manifest_sha256=identity["manifest_sha256"],
                )
        except ImportError:
            raise RuntimeError("GLM adapter validation module is unavailable")
        return None

    def validate_lora_adapter(self, adapter_path: str) -> dict[str, object] | None:
        """Validate a local adapter on every vLLM worker when enabled."""
        return self._validate_staged_glm_adapter(Path(adapter_path))

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
