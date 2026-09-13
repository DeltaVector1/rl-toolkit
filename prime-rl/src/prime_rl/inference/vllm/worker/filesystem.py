import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
import stat
import threading
import time
import urllib.request
from urllib.error import HTTPError, URLError
from contextlib import contextmanager
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.utils.checkpoint_delta import (
    DELTA_MANIFEST,
    apply_xor_delta_file,
)

def adapter_cache_path(source_url: str) -> Path:
    return Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora")) / hashlib.sha256(
        source_url.encode()
    ).hexdigest()


_PARALLEL_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
_DOWNLOAD_STREAMS_ENV = "PRIME_LORA_DOWNLOAD_STREAMS"
_MAX_DOWNLOAD_STREAMS = 128
full_download_streams = 48
default_delta_stage_timeout_s = 600.0
full_adapter_stage_timeout_s = 1800.0
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_ADAPTER_MANIFEST = "adapter_manifest.json"
_ADAPTER_DIGEST = "adapter_model.safetensors.sha256"
_RELAY_MANIFEST = "relay_manifest.json"
_RELAYING_MARKER = "RELAYING"
_WIRE_READY_SUFFIX = ".wire-ready"
_COMPLETED_SUFFIX = ".done"
_ADAPTER_STATUS = ".adapter-stage-status.json"
_ADAPTER_DELTA_TRANSPORT = "chunked-shuffled-xor-zstd"
_ADAPTER_FULL_TRANSPORT = "chunked-full-zstd"
_ADAPTER_LINEAR_TRANSPORT = "chunked-bf16-linear-zigzag-sparse-zstd"
_ADAPTER_KNOWN_TRANSPORTS = {_ADAPTER_DELTA_TRANSPORT, _ADAPTER_LINEAR_TRANSPORT}
_ADAPTER_KNOWN_CHUNK_CODECS = {"zstd"}
_ADAPTER_MAX_CHUNKS = 65536
_FULL_CHECKPOINT_CACHE_MANIFEST = "checkpoint-cache.json"
_ADAPTER_LANE_TIMEOUT_S = float(os.environ.get("PRIME_LORA_LANE_TIMEOUT_S", "45"))
_ADAPTER_LANE_RETRIES = int(os.environ.get("PRIME_LORA_LANE_RETRIES", "6"))
_ADAPTER_VISIBILITY_RETRIES = 300
_PREDICTIVE_RECONSTRUCT_WORKERS = 32
_ADAPTER_CHUNK_TIMEOUT_S = float(os.environ.get("PRIME_LORA_CHUNK_TIMEOUT_S", "60"))
_ADAPTER_CLEANUP_TIMEOUT_S = 30.0
_ADAPTER_STATUS_INTERVAL_S = float(os.environ.get("PRIME_LORA_STATUS_INTERVAL_S", "1"))
_ADAPTER_METADATA_MAX_BYTES = 2 * 1024 * 1024
_ADAPTER_MARKER_MAX_BYTES = 4 * 1024
_ADAPTER_CLEANUP_ENTRY_LIMIT = 256
_ADAPTER_DIRECTORY_ENTRY_LIMIT = 8192
_ADAPTER_CLEANUP_CANDIDATE_LIMIT = 64
persistent_read_timeout_seconds = 5.0
persistent_connection_idle_seconds = 30.0
lane_rotation_delay_seconds = 0.1
_JSON_LOCK = threading.Lock()
_adapter_stage_publish_lock = threading.Lock()


class _AdapterMetadataError(OSError):
    """A metadata response is invalid and must not move to another lane."""


def adapter_stage_limit_seconds(policy_step: int) -> float:
    """Return the bounded operational staging limit for one adapter step."""
    if type(policy_step) is not int or policy_step <= 0:
        raise OSError("adapter policy step must be a positive integer")
    if policy_step == 1:
        return 600.0
    if policy_step == 2:
        return 600.0
    return 300.0


def _adapter_step_from_source_url(source_url: str) -> int:
    match = re.fullmatch(r"step_(\d+)", Path(urlsplit(source_url).path.rstrip("/")).name)
    if match is None:
        raise OSError("adapter source URL does not identify one policy step")
    step = int(match.group(1))
    if step <= 0:
        raise OSError("adapter source URL does not identify a positive policy step")
    return step


def _reject_symlink_path(path: Path, label: str = "adapter path") -> None:
    """Reject a path or any existing parent that is a symbolic link."""
    current = path
    while True:
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            current = current.parent
            if current == path or current == current.parent:
                return
            continue
        except NotADirectoryError as exc:
            raise OSError(f"{label} has a non-directory parent: {path}") from exc
        if stat.S_ISLNK(mode):
            raise OSError(f"{label} is a symbolic link: {path}")
        if current == current.parent:
            return
        current = current.parent


def _ensure_directory(path: Path, label: str = "adapter directory") -> None:
    directory_fd = _open_directory_nofollow(path, create=True, label=label)
    os.close(directory_fd)


def _open_directory_nofollow(
    path: Path,
    *,
    create: bool,
    label: str,
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise OSError("adapter filesystem requires O_NOFOLLOW and O_DIRECTORY")
    absolute = path if path.is_absolute() else Path.cwd() / path
    flags = os.O_RDONLY | directory_flag | nofollow
    try:
        current_fd = os.open(absolute.anchor, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise OSError(f"{label} is not a safe directory: {path}") from exc
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=current_fd)
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except NotADirectoryError as exc:
                try:
                    component_stat = os.stat(
                        component,
                        dir_fd=current_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    raise exc
                if stat.S_ISLNK(component_stat.st_mode):
                    raise OSError(f"{label} is a symbolic link: {path}") from exc
                raise
            os.close(current_fd)
            current_fd = next_fd
    except BaseException:
        os.close(current_fd)
        raise
    return current_fd


def _open_nofollow(
    path: Path | str,
    flags: int,
    mode: int = 0o600,
    *,
    dir_fd: int | None = None,
    label: str = "adapter path",
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise OSError("adapter filesystem requires O_NOFOLLOW")
    owns_parent_fd = False
    if dir_fd is None and isinstance(path, Path) and path.parent != Path("."):
        dir_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label=f"{label} parent",
        )
        path = path.name
        owns_parent_fd = True
    elif dir_fd is None:
        _reject_symlink_path(Path(path))
    try:
        file_descriptor = os.open(path, flags | nofollow, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in {getattr(os, "ELOOP", 40), 40}:
            raise OSError(f"{label} is a symbolic link: {path}") from exc
        raise
    finally:
        if owns_parent_fd:
            os.close(dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise OSError(f"{label} is not a regular file: {path}")
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _write_text_atomic(
    path: Path,
    value: str,
    *,
    encoding: str,
    parent_fd: int | None = None,
) -> None:
    owns_parent_fd = parent_fd is None
    if owns_parent_fd:
        parent_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label="adapter file parent",
        )
    assert parent_fd is not None
    temporary_name = f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        file_descriptor = _open_nofollow(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            dir_fd=parent_fd,
        )
        try:
            output = os.fdopen(file_descriptor, "wb")
        except BaseException:
            os.close(file_descriptor)
            raise
        try:
            output.write(value.encode(encoding))
            output.flush()
            os.fsync(output.fileno())
        finally:
            output.close()
        try:
            existing = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise OSError(f"adapter output is a symbolic link: {path}")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        if owns_parent_fd:
            os.close(parent_fd)


def _write_text_entry_atomic(
    parent_fd: int,
    name: str,
    value: str,
    *,
    encoding: str,
) -> None:
    temporary_name = f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    file_descriptor = _open_nofollow(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        dir_fd=parent_fd,
        label="adapter temporary file",
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(value.encode(encoding))
            output.flush()
            os.fsync(output.fileno())
        try:
            existing = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise OSError(f"adapter output is a symbolic link: {name}")
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


def _read_bytes_nofollow(
    path: Path | str,
    *,
    dir_fd: int | None = None,
    max_bytes: int | None = None,
) -> bytes:
    file_descriptor = _open_nofollow(path, os.O_RDONLY, dir_fd=dir_fd)
    try:
        if max_bytes is not None:
            if type(max_bytes) is not int or max_bytes < 0:
                raise ValueError("maximum read size must be non-negative")
            if os.fstat(file_descriptor).st_size > max_bytes:
                raise OSError(f"file exceeds the {max_bytes}-byte read limit: {path}")
        with os.fdopen(file_descriptor, "rb") as source:
            file_descriptor = -1
            if max_bytes is None:
                return source.read()
            chunks: list[bytes] = []
            remaining = max_bytes
            while True:
                chunk = source.read(min(1024 * 1024, remaining + 1))
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
                remaining -= len(chunk)
                if remaining < 0:
                    raise OSError(f"file exceeds the {max_bytes}-byte read limit: {path}")
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _read_text_nofollow(
    path: Path | str,
    *,
    encoding: str,
    dir_fd: int | None = None,
    max_bytes: int | None = None,
) -> str:
    return _read_bytes_nofollow(path, dir_fd=dir_fd, max_bytes=max_bytes).decode(encoding)


def _sha256_entry(parent_fd: int, name: str, label: str) -> str:
    digest = hashlib.sha256()
    with os.fdopen(
        _open_nofollow(name, os.O_RDONLY, dir_fd=parent_fd, label=label),
        "rb",
    ) as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_entry_nofollow(
    source_directory: Path,
    source_name: str,
    destination_directory: Path,
    destination_name: str,
) -> None:
    """Copy one regular entry through held no-follow directory descriptors."""
    source_fd = _open_directory_nofollow(
        source_directory,
        create=False,
        label="cached adapter directory",
    )
    destination_fd = _open_directory_nofollow(
        destination_directory,
        create=False,
        label="adapter partial directory",
    )
    input_fd = output_fd = -1
    try:
        input_fd = _open_nofollow(source_name, os.O_RDONLY, dir_fd=source_fd, label="cached adapter file")
        output_fd = _open_nofollow(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            dir_fd=destination_fd,
            label="adapter file",
        )
        with os.fdopen(input_fd, "rb") as source, os.fdopen(output_fd, "wb") as destination:
            input_fd = output_fd = -1
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    finally:
        if input_fd >= 0:
            os.close(input_fd)
        if output_fd >= 0:
            os.close(output_fd)
        os.close(destination_fd)
        os.close(source_fd)


def _open_directory_entry_nofollow(parent_fd: int, name: str, label: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise OSError("adapter filesystem requires O_NOFOLLOW and O_DIRECTORY")
    try:
        return os.open(
            name,
            os.O_RDONLY | directory_flag | nofollow,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in {getattr(os, "ELOOP", 40), 40}:
            raise OSError(f"{label} is a symbolic link: {name}") from exc
        raise


def _stat_regular_entry(parent_fd: int, name: str, label: str) -> os.stat_result | None:
    try:
        entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(entry_stat.st_mode):
        raise OSError(f"{label} is a symbolic link: {name}")
    if not stat.S_ISREG(entry_stat.st_mode):
        raise OSError(f"{label} is not a regular file: {name}")
    return entry_stat


def _ensure_directory_entry(parent_fd: int, name: str, label: str) -> int:
    try:
        return _open_directory_entry_nofollow(parent_fd, name, label)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return _open_directory_entry_nofollow(parent_fd, name, label)


def _directory_entry_has_marker(parent_fd: int, name: str, marker: str, label: str) -> bool:
    entry_fd = _open_directory_entry_nofollow(parent_fd, name, label)
    try:
        try:
            marker_fd = _open_nofollow(marker, os.O_RDONLY, dir_fd=entry_fd)
        except FileNotFoundError:
            return False
        else:
            os.close(marker_fd)
            return True
    finally:
        os.close(entry_fd)


def _remove_directory_entry(
    parent_fd: int,
    name: str,
    label: str,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    directory_fd = _open_directory_entry_nofollow(parent_fd, name, label)
    try:
        _remove_opened_directory_contents(
            directory_fd,
            label,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        held_stat = os.fstat(directory_fd)
        if (current_stat.st_dev, current_stat.st_ino) != (
            held_stat.st_dev,
            held_stat.st_ino,
        ):
            raise OSError(f"{label} was replaced during cleanup: {name}")
        os.rmdir(name, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise OSError(f"{label} was replaced during cleanup: {name}") from exc
    finally:
        os.close(directory_fd)


def _remove_opened_directory_contents(
    directory_fd: int,
    label: str,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    child_count = 0
    with os.scandir(directory_fd) as entries:
        child_names = []
        for entry in entries:
            _check_deadline(deadline, cancel_event)
            child_count += 1
            if child_count > _ADAPTER_DIRECTORY_ENTRY_LIMIT:
                raise OSError(f"{label} contains too many entries")
            child_names.append(entry.name)
    for child_name in child_names:
        _check_deadline(deadline, cancel_event)
        child_stat = os.stat(
            child_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(child_stat.st_mode):
            raise OSError(f"{label} contains a symbolic link: {child_name}")
        if stat.S_ISDIR(child_stat.st_mode):
            child_fd = _open_directory_entry_nofollow(
                directory_fd,
                child_name,
                label,
            )
            try:
                _remove_opened_directory_entry(
                    directory_fd,
                    child_name,
                    child_fd,
                    label,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
            finally:
                os.close(child_fd)
        else:
            os.unlink(child_name, dir_fd=directory_fd)


def _remove_opened_directory_entry(
    parent_fd: int,
    name: str,
    directory_fd: int,
    label: str,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Remove a directory through its already-held descriptor."""
    _remove_opened_directory_contents(
        directory_fd,
        label,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    try:
        current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise OSError(f"{label} was replaced during cleanup: {name}") from exc
    held_stat = os.fstat(directory_fd)
    if (current_stat.st_dev, current_stat.st_ino) != (
        held_stat.st_dev,
        held_stat.st_ino,
    ):
        raise OSError(f"{label} was replaced during cleanup: {name}")
    os.rmdir(name, dir_fd=parent_fd)


def _replace_paths(source: Path, destination: Path, label: str) -> None:
    source_parent = Path(os.path.abspath(source.parent))
    destination_parent = Path(os.path.abspath(destination.parent))
    if source_parent != destination_parent:
        raise OSError(f"{label} requires source and destination in one directory")
    parent_fd = _open_directory_nofollow(
        source_parent,
        create=False,
        label=f"{label} parent",
    )
    try:
        os.replace(
            source.name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)


def _unlink_path(path: Path, label: str) -> None:
    parent_fd = _open_directory_nofollow(
        path.parent,
        create=False,
        label=f"{label} parent",
    )
    try:
        try:
            os.unlink(path.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
    finally:
        os.close(parent_fd)


def _reset_staging_directory_identity(
    path: Path,
    identity_name: str,
    identity: str,
    label: str,
) -> None:
    """Remove a mismatched private directory through held descriptors."""
    try:
        parent_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label=f"{label} parent",
        )
    except FileNotFoundError:
        return
    try:
        try:
            entry_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(entry_stat.st_mode):
            raise OSError(f"{label} is a symbolic link: {path}")
        if not stat.S_ISDIR(entry_stat.st_mode):
            os.unlink(path.name, dir_fd=parent_fd)
            return
        directory_fd = _open_directory_entry_nofollow(parent_fd, path.name, label)
        try:
            try:
                existing = _read_text_nofollow(
                    identity_name,
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            except (FileNotFoundError, OSError):
                existing = None
            if existing == identity:
                return
            _remove_opened_directory_entry(
                parent_fd,
                path.name,
                directory_fd,
                label,
            )
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def _path_stat(path: Path, label: str) -> os.stat_result | None:
    try:
        parent_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label=f"{label} parent",
        )
    except FileNotFoundError:
        return None
    try:
        try:
            entry_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(entry_stat.st_mode):
            raise OSError(f"{label} is a symbolic link: {path}")
        return entry_stat
    finally:
        os.close(parent_fd)


def _path_regular_size(path: Path, label: str) -> int | None:
    entry_stat = _path_stat(path, label)
    if entry_stat is None:
        return None
    if not stat.S_ISREG(entry_stat.st_mode):
        raise OSError(f"{label} is not a regular file: {path}")
    return entry_stat.st_size


def _configured_download_streams() -> int:
    raw_value = os.environ.get(_DOWNLOAD_STREAMS_ENV)
    if raw_value is None or not raw_value.strip():
        raise OSError(f"{_DOWNLOAD_STREAMS_ENV} must be configured")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise OSError(f"{_DOWNLOAD_STREAMS_ENV} must be an integer") from exc
    if value < 1 or value > _MAX_DOWNLOAD_STREAMS:
        raise OSError(
            f"{_DOWNLOAD_STREAMS_ENV} must be between 1 and {_MAX_DOWNLOAD_STREAMS}"
        )
    return value


def _delta_stage_timeout_s() -> float:
    configured = os.environ.get("PRIME_LORA_DELTA_STAGE_TIMEOUT_S")
    if configured is None:
        value = default_delta_stage_timeout_s
    else:
        try:
            value = float(configured)
        except ValueError as exc:
            raise OSError("PRIME_LORA_DELTA_STAGE_TIMEOUT_S must be a number") from exc
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _is_finite_number(value)
        or value <= 0
    ):
        raise OSError("adapter delta stage timeout must be positive")
    return float(value)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TimeoutError("adapter staging was cancelled")


def _check_deadline(
    deadline: float | None,
    cancel_event: threading.Event | None = None,
) -> None:
    _check_cancelled(cancel_event)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("adapter delta staging exceeded its time limit")


def _acquire_stage_lock(
    lock,
    deadline: float | None,
    cancel_event: threading.Event | None = None,
) -> None:
    if deadline is None:
        while True:
            _check_cancelled(cancel_event)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if cancel_event is None:
                    time.sleep(0.1)
                else:
                    cancel_event.wait(0.1)
    while True:
        _check_deadline(deadline, cancel_event)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            wait_for = min(0.1, max(0.0, deadline - time.monotonic()))
            if cancel_event is None:
                time.sleep(wait_for)
            else:
                cancel_event.wait(wait_for)


def _request_timeout(
    deadline: float | None,
    cancel_event: threading.Event | None = None,
    maximum: float | None = None,
) -> float:
    _check_deadline(deadline, cancel_event)
    timeout = _ADAPTER_LANE_TIMEOUT_S
    if maximum is not None:
        if not _is_finite_number(maximum) or maximum <= 0:
            raise OSError("adapter request timeout must be positive")
        timeout = min(timeout, float(maximum))
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("adapter delta staging exceeded its time limit")
    return min(timeout, remaining)


def _configured_chunk_timeout() -> float:
    """Return the total time limit for one resumable chunk request."""
    if not _is_finite_number(_ADAPTER_CHUNK_TIMEOUT_S) or _ADAPTER_CHUNK_TIMEOUT_S <= 0:
        raise OSError("PRIME_LORA_CHUNK_TIMEOUT_S must be positive")
    if not _is_finite_number(_ADAPTER_LANE_TIMEOUT_S) or _ADAPTER_LANE_TIMEOUT_S <= 0:
        raise OSError("PRIME_LORA_LANE_TIMEOUT_S must be positive")
    return _ADAPTER_CHUNK_TIMEOUT_S


def _chunk_request_timeout(
    deadline: float | None,
    attempt: int,
    configured_timeout: float,
    cancel_event: threading.Event | None = None,
) -> float:
    """Bound one chunk attempt without extending the whole stage."""
    timeout = configured_timeout
    if not _is_finite_number(timeout) or timeout <= 0:
        raise OSError("adapter request timeout must be positive")
    if deadline is None:
        return timeout
    _check_deadline(deadline, cancel_event)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("adapter delta staging exceeded its time limit")
    return min(timeout, remaining)


def _wait_for_futures(
    executor: ThreadPoolExecutor,
    futures,
    deadline: float | None,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        for future in futures:
            _check_cancelled(cancel_event)
            if deadline is None:
                future.result()
                _check_cancelled(cancel_event)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("adapter delta staging exceeded its time limit")
            try:
                future.result(timeout=remaining)
            except FutureTimeoutError as exc:
                raise TimeoutError("adapter delta staging exceeded its time limit") from exc
            _check_cancelled(cancel_event)
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


def _run_bounded_futures(
    items,
    worker,
    width: int,
    deadline: float | None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run at most ``width`` active or queued tasks and refill on completion."""
    if width < 1:
        raise OSError("adapter download stream width must be positive")
    entries = iter(enumerate(items))
    pending = set()
    executor = ThreadPoolExecutor(max_workers=min(width, len(items))) if items else None
    if executor is None:
        return
    try:
        while len(pending) < width:
            try:
                index, item = next(entries)
            except StopIteration:
                break
            pending.add(executor.submit(worker, index, item))
        while pending:
            _check_deadline(deadline, cancel_event)
            timeout = None
            if deadline is not None:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
            done, _ = wait(
                pending,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("adapter delta staging exceeded its time limit")
            for future in done:
                pending.remove(future)
                future.result()
                _check_cancelled(cancel_event)
            while len(pending) < width:
                try:
                    index, item = next(entries)
                except StopIteration:
                    break
                pending.add(executor.submit(worker, index, item))
        _check_cancelled(cancel_event)
    except BaseException:
        if cancel_event is not None:
            cancel_event.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _content_length(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    last_error: OSError | None = None
    for candidate in _metadata_source_urls(source_url):
        _check_deadline(deadline, cancel_event)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(candidate, method="HEAD"),
                timeout=_request_timeout(
                    deadline,
                    cancel_event,
                    maximum=_configured_chunk_timeout(),
                ),
            ) as response:
                _check_deadline(deadline, cancel_event)
                try:
                    return int(response.headers["Content-Length"])
                except (TypeError, ValueError) as exc:
                    raise _AdapterMetadataError(
                        f"adapter source has no valid Content-Length: {source_url}"
                    ) from exc
        except HTTPError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            last_error = exc
        except OSError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def _read_source_text(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    *,
    max_bytes: int | None = None,
) -> str:
    return _read_source_bytes(
        source_url,
        deadline=deadline,
        cancel_event=cancel_event,
        max_bytes=max_bytes,
    ).decode("utf-8")


def _read_source_bytes(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    *,
    max_bytes: int | None = None,
) -> bytes:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("maximum read size must be non-negative")
    last_error: OSError | None = None
    for candidate in _metadata_source_urls(source_url):
        _check_deadline(deadline, cancel_event)
        try:
            with urllib.request.urlopen(
                candidate,
                timeout=_request_timeout(
                    deadline,
                    cancel_event,
                    maximum=_configured_chunk_timeout(),
                ),
            ) as response:
                headers = getattr(response, "headers", {})
                content_length = (
                    headers.get("Content-Length") if hasattr(headers, "get") else None
                )
                if content_length is not None:
                    try:
                        content_length_value = int(content_length)
                    except (TypeError, ValueError) as exc:
                        raise _AdapterMetadataError(
                            f"source has an invalid Content-Length: {source_url}"
                        ) from exc
                    if content_length_value < 0:
                        raise _AdapterMetadataError(
                            f"source has a negative Content-Length: {source_url}"
                        )
                    if max_bytes is not None and content_length_value > max_bytes:
                        raise _AdapterMetadataError(
                            f"source exceeds the {max_bytes}-byte read limit: {source_url}"
                        )
                chunks = []
                total = 0
                while True:
                    _check_deadline(deadline, cancel_event)
                    read_size = 16 * 1024 * 1024
                    if max_bytes is not None:
                        read_size = min(read_size, max_bytes - total + 1)
                    chunk = response.read(read_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise _AdapterMetadataError(
                            f"source exceeds the {max_bytes}-byte read limit: {source_url}"
                        )
                _check_deadline(deadline, cancel_event)
                return b"".join(chunks)
        except HTTPError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            last_error = exc
        except OSError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def _source_manifest(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    *,
    published_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read the immutable adapter manifest published beside a STABLE marker.

    Older broadcasts have only the model checksum sidecar. Keep that fallback,
    but always return a local manifest with a size and digest for every file.
    """
    source_root = source_url.rstrip("/")
    _check_deadline(deadline, cancel_event)
    value = published_manifest
    if value is None:
        manifest_url = f"{source_root}/{_ADAPTER_MANIFEST}"
        try:
            value = json.loads(
                _read_source_text(
                    manifest_url,
                    deadline=deadline,
                    cancel_event=cancel_event,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except HTTPError as exc:
            if exc.code != 404:
                raise
            value = None
    if value is not None:
        if not isinstance(value, dict) or value.get("version") not in {1, 2}:
            raise OSError("invalid adapter source manifest")
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise OSError("adapter source manifest has no files")
        files: dict[str, dict[str, object]] = {}
        for item in raw_files:
            _check_deadline(deadline, cancel_event)
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
        f"{source_root}/{_ADAPTER_DIGEST}",
        deadline=deadline,
        cancel_event=cancel_event,
        max_bytes=_ADAPTER_MARKER_MAX_BYTES,
    ).strip().split()
    if not digest_text or re.fullmatch(r"[0-9a-f]{64}", digest_text[0]) is None:
        raise OSError("adapter source has no valid model checksum")
    config_bytes = _read_source_bytes(
        config_url,
        deadline=deadline,
        cancel_event=cancel_event,
        max_bytes=_ADAPTER_METADATA_MAX_BYTES,
    )
    return {
        "version": 1,
        "source_url": source_url,
        "files": {
            _ADAPTER_FILES[0]: {
                "size": _content_length(
                    config_url,
                    deadline=deadline,
                    cancel_event=cancel_event,
                ),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            _ADAPTER_FILES[1]: {
                "size": _content_length(
                    model_url,
                    deadline=deadline,
                    cancel_event=cancel_event,
                ),
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


def _metadata_source_urls(source_url: str) -> list[str]:
    """Return every configured lane, starting with the requested lane."""
    source_urls = _adapter_source_urls(source_url)
    if source_url in source_urls:
        start = source_urls.index(source_url)
        source_urls = source_urls[start:] + source_urls[:start]
    return source_urls


def _is_retryable_metadata_error(exception: BaseException) -> bool:
    """Identify connection failures and temporary source HTTP responses."""
    message = str(exception).lower()
    if re.search(
        r"\b(?:checksum|sha256|digest|identity|protocol|semantic|disk)\b|"
        r"content-length|invalid .*manifest|unsupported .*manifest|"
        r"wrong .*step|policy step|no space|space left|permission denied|"
        r"read-only filesystem",
        message,
    ):
        return False
    if isinstance(exception, HTTPError):
        return exception.code in {404, 503}
    if isinstance(exception, _AdapterMetadataError):
        return False
    if isinstance(exception, (TimeoutError, ConnectionError, socket.timeout)):
        return True
    if isinstance(exception, URLError):
        reason = exception.reason
        if isinstance(reason, BaseException):
            return _is_retryable_metadata_error(reason)
        return False
    return False


def _responsive_adapter_source_urls(
    source_url: str,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> tuple[list[str], int]:
    """Remove source lanes that cannot reach the adapter server."""
    source_urls = _adapter_source_urls(source_url)
    if len(source_urls) == 1:
        return source_urls, 1

    def probe(candidate: str) -> bool:
        config_url = f"{candidate.rstrip('/')}/{_ADAPTER_FILES[0]}"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(config_url, method="HEAD"),
                timeout=_request_timeout(
                    deadline,
                    cancel_event,
                    maximum=_configured_chunk_timeout(),
                ),
            ) as response:
                status = int(response.status)
                if 200 <= status < 400:
                    return True
                if status in {404, 503}:
                    return False
                raise OSError(f"adapter source probe returned HTTP status {status}")
        except HTTPError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            return False
        except OSError as exc:
            if not _is_retryable_metadata_error(exc):
                raise
            return False

    executor = ThreadPoolExecutor(max_workers=len(source_urls))
    try:
        results = list(executor.map(probe, source_urls))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    _check_deadline(deadline, cancel_event)
    responsive = [url for url, ready in zip(source_urls, results, strict=True) if ready]
    if not responsive:
        raise OSError(f"none of {len(source_urls)} adapter source lanes responded")
    return responsive, len(source_urls)


def _read_v1_transport_chunks(
    value: dict[str, object],
    target_size: int,
    deadline: float | None,
    cancel_event: threading.Event | None,
    label: str,
) -> list[dict[str, object]]:
    raw_chunks = value.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks or len(raw_chunks) > _ADAPTER_MAX_CHUNKS:
        raise OSError(f"{label} has an invalid chunk list")
    chunks: list[dict[str, object]] = []
    seen_files: set[str] = set()

    def valid_digest(item: object) -> bool:
        return isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) is not None

    for item in raw_chunks:
        _check_deadline(deadline, cancel_event)
        if not isinstance(item, dict):
            raise OSError(f"{label} chunk is not an object")
        name = item.get("file")
        offset = item.get("offset")
        raw_size = item.get("raw_size")
        wire_size = item.get("wire_size")
        wire_sha256 = item.get("wire_sha256")
        raw_sha256 = item.get("raw_sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or name in {*_ADAPTER_FILES, _ADAPTER_MANIFEST}
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
            raise OSError(f"{label} chunk identity is invalid")
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
            raise OSError(f"{label} chunks do not cover the target exactly")
        expected_offset += int(item["raw_size"])
    if expected_offset != target_size:
        raise OSError(f"{label} chunks do not cover the target")
    chunk_count = value.get("chunk_count")
    if chunk_count is not None and chunk_count != len(chunks):
        raise OSError(f"{label} chunk count differs")
    return chunks


def _read_chunked_v1_manifest(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, object] | None:
    """Read the exact version-one adapter transport manifest."""
    source_root = source_url.rstrip("/")
    _check_deadline(deadline, cancel_event)
    try:
        value = json.loads(
            _read_source_text(
                f"{source_root}/{_ADAPTER_MANIFEST}",
                deadline=deadline,
                cancel_event=cancel_event,
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except HTTPError as exc:
        if exc.code in {404, 503}:
            raise OSError("adapter manifest is missing; transfer mode is unknown") from exc
        raise
    if not isinstance(value, dict) or value.get("version") != 1:
        raise OSError("invalid adapter transport manifest")
    mode = value.get("mode")
    if mode == "full":
        if value.get("transport") != _ADAPTER_FULL_TRANSPORT:
            return {"mode": "full", "source_manifest": value}
        if value.get("kind") != "lora-adapter" or value.get("target_step") != 0:
            raise OSError("full zstd adapter manifest is not policy zero")
        parsed_source = urlsplit(source_url)
        if parsed_source.scheme not in ("http", "https") or not parsed_source.hostname:
            raise OSError("adapter source URL must use HTTP or HTTPS")
        match = re.fullmatch(r"step_(\d+)", Path(parsed_source.path.rstrip("/")).name)
        if match is None or int(match.group(1)) != 0:
            raise OSError("adapter source URL does not identify policy zero")
        target_size = value.get("target_size")
        full_size = value.get("full_size")
        target_sha256 = value.get("target_sha256")
        full_sha256 = value.get("full_sha256")
        config_size = value.get("config_size")
        config_sha256 = value.get("config_sha256")

        def valid_digest(item: object) -> bool:
            return isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) is not None

        if (
            type(target_size) is not int
            or target_size <= 0
            or full_size != target_size
            or not valid_digest(target_sha256)
            or full_sha256 != target_sha256
            or type(config_size) is not int
            or config_size < 0
            or not valid_digest(config_sha256)
        ):
            raise OSError("full zstd adapter manifest is missing target identity")
        files = value.get("files")
        file_map = {
            item.get("name"): item
            for item in files
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        } if isinstance(files, list) else {}
        if (
            len(file_map) != 2
            or set(file_map) != set(_ADAPTER_FILES)
            or file_map[_ADAPTER_FILES[1]].get("size") != target_size
            or file_map[_ADAPTER_FILES[1]].get("sha256") != target_sha256
            or file_map[_ADAPTER_FILES[0]].get("size") != config_size
            or file_map[_ADAPTER_FILES[0]].get("sha256") != config_sha256
        ):
            raise OSError("full zstd adapter manifest files differ from target identity")
        chunks = _read_v1_transport_chunks(
            value, target_size, deadline, cancel_event, "full zstd adapter manifest"
        )
        return {
            "version": 1,
            "mode": "full",
            "transport": _ADAPTER_FULL_TRANSPORT,
            "source_manifest": value,
            "target_sha256": target_sha256,
            "target_size": target_size,
            "config_sha256": config_sha256,
            "config_size": config_size,
            "chunks": chunks,
        }
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

    chunks = _read_v1_transport_chunks(
        value, target_size, deadline, cancel_event, "adapter transport manifest"
    )
    _check_deadline(deadline, cancel_event)
    return {
        "version": 1,
        "mode": "delta",
        "transport": _ADAPTER_DELTA_TRANSPORT,
        "source_manifest": value,
        "base_url": base_url,
        "base_sha256": base_sha256,
        "target_sha256": target_sha256,
        "target_size": target_size,
        "config_sha256": config_sha256,
        "config_size": config_size,
        "base_config_sha256": base_config_sha256,
        "chunks": chunks,
    }


def _read_chunked_linear_manifest(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    """Read and validate the version-two two-predecessor transport manifest."""
    source_root = source_url.rstrip("/")
    try:
        value = json.loads(
            _read_source_text(
                f"{source_root}/{_ADAPTER_MANIFEST}",
                deadline=deadline,
                cancel_event=cancel_event,
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except HTTPError as exc:
        if exc.code in {404, 503}:
            raise OSError("adapter manifest is missing; transfer mode is unknown") from exc
        raise
    if not isinstance(value, dict) or value.get("version") != 2:
        raise OSError("invalid version-two adapter transport manifest")
    if value.get("kind") != "lora-adapter" or value.get("mode") != "delta":
        raise OSError("version-two adapter manifest is not a delta")
    if value.get("transport") != _ADAPTER_LINEAR_TRANSPORT:
        raise OSError("adapter manifest selects an unsupported version-two transport")

    target_step = value.get("target_step")
    base_step = value.get("base_step")
    base2_step = value.get("base2_step")
    if (
        type(target_step) is not int
        or target_step < 2
        or type(base_step) is not int
        or type(base2_step) is not int
        or base_step != target_step - 1
        or base2_step != target_step - 2
    ):
        raise OSError("version-two adapter manifest steps are not consecutive")
    parsed_source = urlsplit(source_url)
    if parsed_source.scheme not in ("http", "https") or not parsed_source.hostname:
        raise OSError("adapter source URL must use HTTP or HTTPS")
    match = re.fullmatch(r"step_(\d+)", Path(parsed_source.path.rstrip("/")).name)
    if match is None or int(match.group(1)) != target_step:
        raise OSError("adapter source URL does not identify target step")

    def step_url(step: int) -> str:
        path = str(Path(parsed_source.path.rstrip("/")).parent / f"step_{step}")
        return urlunsplit(
            (parsed_source.scheme, parsed_source.netloc, path, parsed_source.query, parsed_source.fragment)
        )

    base_url = step_url(base_step)
    base2_url = step_url(base2_step)
    if value.get("target_url", source_url) != source_url:
        raise OSError("version-two target URL differs from the requested source")
    if value.get("base_url") != base_url or value.get("base2_url") != base2_url:
        raise OSError("version-two base URLs do not match the target step")

    def digest(name: str) -> str:
        result = value.get(name)
        if not isinstance(result, str) or re.fullmatch(r"[0-9a-f]{64}", result) is None:
            raise OSError(f"version-two adapter manifest has no valid {name}")
        return result

    def positive_size(name: str) -> int:
        result = value.get(name)
        if type(result) is not int or result <= 0:
            raise OSError(f"version-two adapter manifest has no valid {name}")
        return result

    def config_size(name: str) -> int:
        result = value.get(name)
        if type(result) is not int or result < 0:
            raise OSError(f"version-two adapter manifest has no valid {name}")
        return result

    target_size = positive_size("target_size")
    config_size_value = config_size("config_size")
    base_size = positive_size("base_size")
    base2_size = positive_size("base2_size")
    base_config_size = config_size("base_config_size")
    base2_config_size = config_size("base2_config_size")
    if base_size != target_size or base2_size != target_size:
        raise OSError("version-two adapter weights are not equal-sized files")
    data_offset = value.get("data_offset")
    if type(data_offset) is not int or data_offset < 0 or data_offset > target_size:
        raise OSError("version-two adapter data offset is invalid")
    target_sha256 = digest("target_sha256")
    if value.get("full_sha256", target_sha256) != target_sha256:
        raise OSError("version-two target checksum differs from full checksum")
    if value.get("full_size", target_size) != target_size:
        raise OSError("version-two target size differs from full size")
    config_sha256 = digest("config_sha256")
    base_sha256 = digest("base_sha256")
    base_config_sha256 = digest("base_config_sha256")
    base2_sha256 = digest("base2_sha256")
    base2_config_sha256 = digest("base2_config_sha256")
    if (
        value.get("chunk_alignment") != 2
        or value.get("chunk_coverage") != {"offset": 0, "size": target_size}
        or value.get("predictor") != "bf16-float32-second-order-rne"
        or value.get("value_encoding") != "sparse-zigzag-leb128"
        or value.get("mask_encoding") != "zstd-bit-mask-little-endian"
    ):
        raise OSError("version-two adapter encoding identity is invalid")

    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise OSError("version-two adapter manifest has no files")
    files: dict[str, dict[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise OSError("invalid version-two adapter manifest file")
        name = item.get("name")
        size = item.get("size")
        file_digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or name not in _ADAPTER_FILES
            or name in files
            or type(size) is not int
            or size < 0
            or not isinstance(file_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", file_digest) is None
        ):
            raise OSError("invalid version-two adapter manifest file entry")
        files[name] = {"size": size, "sha256": file_digest}
    if set(files) != set(_ADAPTER_FILES):
        raise OSError("version-two adapter manifest is missing an adapter file")
    if (
        files[_ADAPTER_FILES[1]] != {"size": target_size, "sha256": target_sha256}
        or files[_ADAPTER_FILES[0]] != {"size": config_size_value, "sha256": config_sha256}
    ):
        raise OSError("version-two target files differ from manifest identity")

    raw_chunks = value.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks or len(raw_chunks) > _ADAPTER_MAX_CHUNKS:
        raise OSError("version-two adapter manifest has an invalid chunk list")
    chunks: list[dict[str, object]] = []
    seen_files: set[str] = set()
    for item in raw_chunks:
        _check_deadline(deadline, cancel_event)
        if not isinstance(item, dict):
            raise OSError("version-two adapter chunk is not an object")
        name = item.get("file")
        offset = item.get("offset")
        chunk_data_offset = item.get("data_offset")
        raw_size = item.get("raw_size")
        wire_size = item.get("wire_size")
        wire_sha256 = item.get("wire_sha256")
        raw_sha256 = item.get("raw_sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or name in {*_ADAPTER_FILES, _ADAPTER_MANIFEST}
            or name in seen_files
            or type(offset) is not int
            or offset < 0
            or type(chunk_data_offset) is not int
            or chunk_data_offset < 0
            or chunk_data_offset != data_offset
            or type(raw_size) is not int
            or raw_size <= 0
            or offset + raw_size > target_size
            or type(wire_size) is not int
            or wire_size <= 0
            or not isinstance(wire_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", wire_sha256) is None
            or not isinstance(raw_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None
            or item.get("codec") not in _ADAPTER_KNOWN_CHUNK_CODECS
        ):
            raise OSError("version-two adapter chunk identity is invalid")
        data_start = max(offset, data_offset)
        data_size = offset + raw_size - data_start
        if data_size and ((data_start - data_offset) % 2 or data_size % 2):
            raise OSError("version-two adapter chunk identity is invalid")
        seen_files.add(name)
        chunks.append(
            {
                "file": name,
                "offset": offset,
                "data_offset": chunk_data_offset,
                "raw_size": raw_size,
                "wire_size": wire_size,
                "wire_sha256": wire_sha256,
                "raw_sha256": raw_sha256,
                "codec": item["codec"],
            }
        )
    chunks.sort(key=lambda item: int(item["offset"]))
    expected_offset = 0
    for item in chunks:
        if int(item["offset"]) != expected_offset:
            raise OSError("version-two adapter chunks do not cover the target exactly")
        expected_offset += int(item["raw_size"])
    if expected_offset != target_size:
        raise OSError("version-two adapter chunks do not cover the target")
    chunk_count = value.get("chunk_count")
    if chunk_count is not None and chunk_count != len(chunks):
        raise OSError("version-two adapter chunk count differs")
    return {
        "version": 2,
        "mode": "delta",
        "transport": _ADAPTER_LINEAR_TRANSPORT,
        "source_manifest": value,
        "base_url": base_url,
        "base_sha256": base_sha256,
        "base_size": base_size,
        "base_config_size": base_config_size,
        "base_config_sha256": base_config_sha256,
        "base2_url": base2_url,
        "base2_sha256": base2_sha256,
        "base2_size": base2_size,
        "base2_config_size": base2_config_size,
        "base2_config_sha256": base2_config_sha256,
        "data_offset": data_offset,
        "target_sha256": target_sha256,
        "target_size": target_size,
        "config_sha256": config_sha256,
        "config_size": config_size_value,
        "chunks": chunks,
    }


def _read_chunked_xor_manifest(
    source_url: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    """Read a known exact adapter transport manifest."""
    source_root = source_url.rstrip("/")
    _check_deadline(deadline, cancel_event)
    try:
        raw = json.loads(
            _read_source_text(
                f"{source_root}/{_ADAPTER_MANIFEST}",
                deadline=deadline,
                cancel_event=cancel_event,
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except HTTPError as exc:
        if exc.code in {404, 503}:
            raise OSError("adapter manifest is missing; transfer mode is unknown") from exc
        raise OSError(f"adapter manifest request failed with HTTP status {exc.code}") from exc
    if not isinstance(raw, dict):
        raise OSError("invalid adapter transport manifest")
    if raw.get("version") == 2:
        return _read_chunked_linear_manifest(source_url, deadline, cancel_event)
    if raw.get("version") == 1:
        return _read_chunked_v1_manifest(source_url, deadline, cancel_event)
    raise OSError("adapter manifest version is unsupported")


def _read_delta_manifest_with_visibility_wait(
    source_url: str,
    *,
    deadline: float | None,
    cancel_event: threading.Event | None,
) -> dict[str, object]:
    """Wait for the origin to publish a v2 manifest under one deadline."""
    for attempt in range(_ADAPTER_VISIBILITY_RETRIES):
        try:
            return _read_chunked_xor_manifest(
                source_url,
                deadline=deadline,
                cancel_event=cancel_event,
            )
        except OSError as exc:
            if "adapter manifest is missing" not in str(exc):
                raise
            _check_deadline(deadline, cancel_event)
            if attempt + 1 == _ADAPTER_VISIBILITY_RETRIES:
                raise
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("adapter delta staging exceeded its time limit") from exc
            wait_for = 1.0 if remaining is None else min(1.0, remaining)
            if cancel_event is None:
                time.sleep(wait_for)
            else:
                cancel_event.wait(wait_for)
    raise OSError("adapter manifest visibility wait ended without a result")


def _write_json(
    path: Path,
    value: dict[str, object],
    *,
    parent_fd: int | None = None,
) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, separators=(",", ":")),
        encoding="utf-8",
        parent_fd=parent_fd,
    )


def _write_marker(path: Path, value: str) -> None:
    """Publish a state marker with one atomic rename."""
    _write_text_atomic(path, value, encoding="ascii")


def _proof_path(chunk_path: Path, suffix: str) -> Path:
    return chunk_path.with_name(f"{chunk_path.name}{suffix}")


def _proof_identity_matches(
    proof_path: Path,
    item: dict[str, object],
    *,
    completed: bool,
) -> tuple[bool, dict[str, object] | None]:
    """Check a chunk proof without following a symbolic link."""
    _reject_symlink_path(proof_path, "adapter proof path")
    if not proof_path.is_file():
        return False, None
    try:
        proof = json.loads(
            _read_text_nofollow(
                proof_path,
                encoding="utf-8",
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except (OSError, ValueError):
        return False, None
    if not isinstance(proof, dict):
        return False, None
    identity_matches = all(
        proof.get(name) == item.get(name)
        for name in (
            "file",
            "offset",
            "data_offset",
            "raw_size",
            "wire_size",
            "wire_sha256",
            "raw_sha256",
        )
    )
    if not identity_matches:
        return False, proof
    if not completed:
        return proof.get("type") == "wire-ready", proof
    return (
        proof.get("correction_sha256") == item.get("raw_sha256")
        and isinstance(proof.get("output_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", proof["output_sha256"]) is not None,
        proof,
    )


def _discard_proof(path: Path) -> None:
    """Remove only a regular proof."""
    try:
        parent_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label="adapter proof parent",
        )
    except FileNotFoundError:
        return
    try:
        if _stat_regular_entry(parent_fd, path.name, "adapter proof path") is not None:
            os.unlink(path.name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _publish_relay_manifest(
    directory: Path,
    source_url: str,
    source_manifest: dict[str, object],
    transport: dict[str, object],
    validated_files: dict[str, object],
    *,
    allow_partial: bool = False,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    directory_fd: int | None = None,
) -> None:
    """Publish the already validated files that the local relay may serve."""
    owns_directory_fd = directory_fd is None
    if owns_directory_fd:
        directory_fd = _open_directory_nofollow(
            directory,
            create=False,
            label="adapter relay directory",
        )
    assert directory_fd is not None
    mode = transport.get("mode")
    try:
        if mode not in {"full", "delta"}:
            raise OSError("adapter relay transport mode is invalid")
        transport_name = transport.get("transport")
        if mode == "delta" and transport_name not in _ADAPTER_KNOWN_TRANSPORTS:
            raise OSError("adapter relay transport is unsupported")
        if mode == "full" and transport_name not in {None, _ADAPTER_FULL_TRANSPORT}:
            raise OSError("adapter relay full transport is unsupported")

        adapter_manifest_path = directory / _ADAPTER_MANIFEST
        relay_source_manifest = source_manifest
        if mode == "full" and transport_name == _ADAPTER_FULL_TRANSPORT:
            relay_source_manifest = dict(source_manifest)
            for key in ("transport", "chunks", "chunk_count", "encoded_size"):
                relay_source_manifest.pop(key, None)
        _write_json(adapter_manifest_path, relay_source_manifest, parent_fd=directory_fd)
        _check_deadline(deadline, cancel_event)
        manifest_stat = _stat_regular_entry(
            directory_fd,
            _ADAPTER_MANIFEST,
            "adapter relay manifest",
        )
        if manifest_stat is None:
            raise OSError("adapter relay manifest was not written")
        relay_files: dict[str, dict[str, object]] = {
            _ADAPTER_MANIFEST: {
                "path": _ADAPTER_MANIFEST,
                "size": manifest_stat.st_size,
                "sha256": _sha256(
                    adapter_manifest_path,
                    deadline,
                    cancel_event,
                    parent_fd=directory_fd,
                ),
            }
        }
        for filename in _ADAPTER_FILES:
            _check_deadline(deadline, cancel_event)
            spec = validated_files.get(filename)
            if not isinstance(spec, dict):
                raise OSError(f"adapter relay validation is missing: {filename}")
            size = spec.get("size")
            digest = spec.get("sha256")
            entry_stat = _stat_regular_entry(
                directory_fd,
                filename,
                "adapter relay file",
            )
            if (
                type(size) is not int
                or size < 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or (not allow_partial and (entry_stat is None or entry_stat.st_size != size))
            ):
                raise OSError(f"adapter relay file checksum differs: {filename}")
            relay_files[filename] = {
                "path": filename,
                "size": size,
                "sha256": digest,
            }

        if mode == "delta":
            chunks = transport.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise OSError("adapter relay delta has no chunks")
            chunk_fd = _open_directory_entry_nofollow(
                directory_fd,
                "transport-chunks",
                "adapter relay chunk directory",
            )
            try:
                for item in chunks:
                    _check_deadline(deadline, cancel_event)
                    if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                        raise OSError("adapter relay delta chunk is invalid")
                    filename = item["file"]
                    if (
                        not filename
                        or filename in {".", ".."}
                        or Path(filename).name != filename
                        or filename in relay_files
                    ):
                        raise OSError("adapter relay delta chunk is missing")
                    wire_size = item.get("wire_size")
                    wire_sha256 = item.get("wire_sha256")
                    entry_stat = _stat_regular_entry(
                        chunk_fd,
                        filename,
                        "adapter relay chunk",
                    )
                    if (
                        type(wire_size) is not int
                        or wire_size <= 0
                        or not isinstance(wire_sha256, str)
                        or re.fullmatch(r"[0-9a-f]{64}", wire_sha256) is None
                        or (not allow_partial and (entry_stat is None or entry_stat.st_size != wire_size))
                    ):
                        raise OSError(f"adapter relay delta chunk checksum differs: {filename}")
                    relay_files[filename] = {
                        "path": f"transport-chunks/{filename}",
                        "size": wire_size,
                        "sha256": wire_sha256,
                    }
            finally:
                os.close(chunk_fd)

        _write_json(
            directory / _RELAY_MANIFEST,
            {
                "version": 1,
                "source_url": source_url,
                "transport": transport_name if mode == "delta" else "full",
                "files": relay_files,
            },
            parent_fd=directory_fd,
        )
    finally:
        if owns_directory_fd:
            os.close(directory_fd)


def _hold_relay_directory(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        directory = Path(args[0] if args else kwargs["directory"])
        try:
            directory_fd = _open_directory_nofollow(
                directory,
                create=False,
                label="adapter cache directory",
            )
        except (FileNotFoundError, OSError):
            return False
        try:
            return function(*args, **kwargs, _directory_fd=directory_fd)
        finally:
            os.close(directory_fd)

    return wrapped


@_hold_relay_directory
def _relay_manifest_matches(
    directory: Path,
    source_url: str,
    transport: dict[str, object],
    *,
    _directory_fd: int,
) -> bool:
    """Check the READY cache's relay index without hashing large files."""
    directory_fd = _directory_fd
    try:
        value = json.loads(
            _read_text_nofollow(
                _RELAY_MANIFEST,
                encoding="utf-8",
                dir_fd=directory_fd,
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except (FileNotFoundError, OSError, ValueError):
        return False
    try:
        ready = _read_bytes_nofollow(
            "READY",
            dir_fd=directory_fd,
            max_bytes=_ADAPTER_MARKER_MAX_BYTES,
        )
    except (FileNotFoundError, OSError):
        return False
    if ready != b"ready\n":
        return False
    if not isinstance(value, dict) or value.get("version") != 1:
        return False
    mode = transport.get("mode")
    expected_transport = transport.get("transport") if mode == "delta" else "full"
    if mode == "delta" and expected_transport not in _ADAPTER_KNOWN_TRANSPORTS:
        return False
    if value.get("source_url") != source_url or value.get("transport") != expected_transport:
        return False
    files = value.get("files")
    if not isinstance(files, dict):
        return False
    expected: dict[str, tuple[str, int, str | None]] = {}
    try:
        manifest_stat = _stat_regular_entry(
            directory_fd,
            _ADAPTER_MANIFEST,
            "adapter relay manifest",
        )
        if manifest_stat is None:
            return False
        manifest_size = manifest_stat.st_size
        manifest_digest = _sha256_entry(
            directory_fd,
            _ADAPTER_MANIFEST,
            "adapter relay manifest",
        )
    except (FileNotFoundError, OSError):
        return False
    expected[_ADAPTER_MANIFEST] = (
        _ADAPTER_MANIFEST,
        manifest_size,
        manifest_digest,
    )
    config_size = transport.get("config_size")
    config_sha256 = transport.get("config_sha256")
    target_size = transport.get("target_size")
    target_sha256 = transport.get("target_sha256")
    if mode == "delta":
        adapter_specs = {
            _ADAPTER_FILES[0]: (config_size, config_sha256),
            _ADAPTER_FILES[1]: (target_size, target_sha256),
        }
    elif mode == "full":
        source_manifest = transport.get("source_manifest")
        source_files = source_manifest.get("files") if isinstance(source_manifest, dict) else None
        adapter_specs = {}
        if isinstance(source_files, list):
            for item in source_files:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    adapter_specs[item["name"]] = (item.get("size"), item.get("sha256"))
    else:
        return False
    for filename, (size, digest) in adapter_specs.items():
        if filename not in _ADAPTER_FILES or type(size) is not int or size < 0 or not isinstance(digest, str):
            return False
        expected[filename] = (filename, size, digest)
    if set(expected) != set(_ADAPTER_FILES) | {_ADAPTER_MANIFEST}:
        return False

    chunks = transport.get("chunks") if mode == "delta" else None
    if mode == "delta":
        if not isinstance(chunks, list) or not chunks:
            return False
        for item in chunks:
            if not isinstance(item, dict):
                return False
            filename = item.get("file")
            size = item.get("wire_size")
            digest = item.get("wire_sha256")
            if (
                not isinstance(filename, str)
                or not filename
                or filename in {".", ".."}
                or Path(filename).name != filename
                or filename in expected
                or type(size) is not int
                or size <= 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                return False
            expected[filename] = (f"transport-chunks/{filename}", size, digest)
    if set(files) != set(expected):
        return False
    for filename, (relative, size, digest) in expected.items():
        spec = files.get(filename)
        if not isinstance(spec, dict) or spec.get("path") != relative or spec.get("size") != size:
            return False
        if not isinstance(spec.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is None:
            return False
        if digest is not None and spec["sha256"] != digest:
            return False
        if relative.startswith("transport-chunks/"):
            chunk_name = relative.removeprefix("transport-chunks/")
            try:
                chunk_fd = _open_directory_entry_nofollow(
                    directory_fd,
                    "transport-chunks",
                    "adapter relay chunk directory",
                )
                try:
                    entry_stat = _stat_regular_entry(
                        chunk_fd,
                        chunk_name,
                        "adapter relay chunk",
                    )
                finally:
                    os.close(chunk_fd)
            except (FileNotFoundError, OSError):
                return False
        else:
            try:
                entry_stat = _stat_regular_entry(
                    directory_fd,
                    relative,
                    "adapter relay file",
                )
            except (FileNotFoundError, OSError):
                return False
        if entry_stat is None or entry_stat.st_size != size:
            return False
    return True


def _write_stage_status(
    cache_root: Path,
    value: dict[str, object],
    *,
    cache_fd: int | None = None,
) -> None:
    # Calls can come from the range workers. The global staging flock prevents
    # cross-process stages; this lock keeps status writes ordered within one.
    stage_id = os.environ.get("PRIME_LORA_STAGE_ID")
    if stage_id:
        value = {**value, "stage_id": stage_id, "pid": os.getpid()}
    if value.get("status") == "ready":
        value = {**value, "ready_at_monotonic": time.monotonic()}
        try:
            manifest_path = Path(str(value["path"])) / "manifest.json"
            manifest = json.loads(
                _read_text_nofollow(
                    manifest_path,
                    encoding="utf-8",
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
            target = manifest["files"]["adapter_model.safetensors"]
            if isinstance(target, dict) and isinstance(target.get("sha256"), str):
                value = {**value, "target_sha256": target["sha256"]}
        except (KeyError, OSError, TypeError, ValueError):
            pass
        verified_files = value.get("verified_files")
        stage_path = value.get("path")
        if isinstance(verified_files, dict) and isinstance(stage_path, str):
            try:
                stage_fd = _open_directory_nofollow(
                    Path(stage_path),
                    create=False,
                    label="adapter ready directory",
                )
                try:
                    file_stats: dict[str, dict[str, int]] = {}
                    for name in _ADAPTER_FILES:
                        entry = _stat_regular_entry(stage_fd, name, "adapter ready file")
                        if entry is None:
                            raise OSError(f"adapter ready file is missing: {name}")
                        file_stats[name] = {
                            "dev": int(entry.st_dev),
                            "ino": int(entry.st_ino),
                            "size": int(entry.st_size),
                            "mtime_ns": int(entry.st_mtime_ns),
                        }
                    value = {**value, "file_stats": file_stats}
                finally:
                    os.close(stage_fd)
            except (OSError, TypeError, ValueError):
                value.pop("file_stats", None)
    with _JSON_LOCK:
        _write_json(cache_root / _ADAPTER_STATUS, value, parent_fd=cache_fd)


def adapter_stage_status() -> dict[str, object]:
    cache_root = Path(os.environ.get("PRIME_LORA_CACHE_DIR", "/tmp/prime-rl-lora"))
    try:
        cache_fd = _open_directory_nofollow(
            cache_root,
            create=False,
            label="adapter cache directory",
        )
        try:
            value = json.loads(
                _read_text_nofollow(
                    _ADAPTER_STATUS,
                    encoding="utf-8",
                    dir_fd=cache_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        finally:
            os.close(cache_fd)
    except FileNotFoundError:
        return {"status": "idle", "bytes_done": 0, "bytes_total": 0}
    except (OSError, ValueError) as exc:
        return {
            "status": "failed",
            "bytes_done": 0,
            "bytes_total": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "status": "failed",
            "bytes_done": 0,
            "bytes_total": 0,
            "error": "ValueError: adapter status is not an object",
        }
    return value


def _stage_timing_fields(
    started: float,
    *,
    transfer_started: float | None = None,
    transfer_finished: float | None = None,
    reconstruction_started: float | None = None,
    reconstruction_finished: float | None = None,
    reconstruction_worker_seconds: float = 0.0,
    verification_started: float | None = None,
    verification_finished: float | None = None,
    cache_hit: bool = False,
) -> dict[str, object]:
    def elapsed(begin: float | None, end: float | None) -> float:
        if begin is None or end is None:
            return 0.0
        return max(0.0, end - begin)

    now = time.monotonic()
    reconstruction_wall_seconds = elapsed(reconstruction_started, reconstruction_finished)
    return {
        "transfer_wall_seconds": round(elapsed(transfer_started, transfer_finished), 3),
        "reconstruction_wall_seconds": round(reconstruction_wall_seconds, 3),
        "reconstruct_seconds": round(reconstruction_wall_seconds, 3),
        "reconstruction_worker_seconds": round(max(0.0, reconstruction_worker_seconds), 3),
        "final_verification_wall_seconds": round(
            elapsed(verification_started, verification_finished), 3
        ),
        "total_stage_wall_seconds": round(max(0.0, now - started), 3),
        "cache_hit": cache_hit,
    }


def _adapter_reader_lock(path: Path) -> Path:
    return path.parent / f".{path.name}.readers.lock"


@contextmanager
def adapter_reader_lock(path: Path, *, deadline: float | None = None):
    """Hold a shared lease while vLLM reads an adapter from the cache."""
    deadline = deadline or (time.monotonic() + _ADAPTER_CLEANUP_TIMEOUT_S)
    lock_path = _adapter_reader_lock(path)
    parent_fd = _open_directory_nofollow(
        lock_path.parent,
        create=False,
        label="adapter lock directory",
    )
    try:
        with os.fdopen(
            _open_nofollow(
                lock_path.name,
                os.O_RDWR | os.O_CREAT,
                dir_fd=parent_fd,
                label="adapter reader lock",
            ),
            "a+",
        ) as lock:
            while True:
                _check_deadline(deadline)
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter reader lock acquisition exceeded its time limit")
                    time.sleep(min(0.1, remaining))
            try:
                yield path
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        os.close(parent_fd)


def _cleanup_retired_adapters(
    cache_root: Path,
    active_name: str,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    prefix = f"{active_name}.retired-"
    try:
        cache_fd = _open_directory_nofollow(
            cache_root,
            create=False,
            label="adapter cache directory",
        )
    except FileNotFoundError:
        return
    try:
        entry_count = 0
        with os.scandir(cache_fd) as entries:
            names = []
            for entry in entries:
                _check_deadline(deadline, cancel_event)
                entry_count += 1
                if entry_count > _ADAPTER_CLEANUP_ENTRY_LIMIT:
                    raise OSError("adapter cache has too many cleanup entries")
                names.append(entry.name)
        for name in names:
            _check_deadline(deadline, cancel_event)
            if not name.startswith(prefix):
                continue
            try:
                entry_stat = os.stat(name, dir_fd=cache_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(entry_stat.st_mode):
                raise OSError(f"retired adapter path is a symbolic link: {name}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue
            lock_name = f".{active_name}.readers.lock"
            try:
                lock_fd = _open_nofollow(
                    lock_name,
                    os.O_RDWR | os.O_CREAT,
                    dir_fd=cache_fd,
                )
                with os.fdopen(lock_fd, "a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _remove_directory_entry(
                        cache_fd,
                        name,
                        "retired adapter path",
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
                    fcntl.flock(lock, fcntl.LOCK_UN)
            except (BlockingIOError, FileNotFoundError):
                continue
    finally:
        os.close(cache_fd)


def reconcile_retired_adapter(
    destination: Path,
    source_url: str | None = None,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Restore a complete retired cache if a stage died between renames."""
    deadline = deadline or (time.monotonic() + default_delta_stage_timeout_s)
    with adapter_reader_lock(destination, deadline=deadline):
        parent_fd = _open_directory_nofollow(destination.parent, create=False, label="adapter cache")
        try:
            def valid(directory_fd: int) -> bool:
                try:
                    def metadata(name: str, encoding: str) -> str:
                        entry = _stat_regular_entry(directory_fd, name, "adapter metadata")
                        maximum = (
                            _ADAPTER_MARKER_MAX_BYTES
                            if name in {"READY", _ADAPTER_DIGEST}
                            else _ADAPTER_METADATA_MAX_BYTES
                        )
                        if entry is None or entry.st_size > maximum:
                            raise ValueError("adapter metadata is too large")
                        return _read_text_nofollow(
                            name,
                            encoding=encoding,
                            dir_fd=directory_fd,
                            max_bytes=maximum,
                        )

                    if metadata("READY", "ascii") != "ready\n":
                        return False
                    manifest = json.loads(metadata("manifest.json", "utf-8"))
                    if not isinstance(manifest, dict) or manifest.get("version") != 1:
                        return False
                    if source_url is not None and manifest.get("source_url") != source_url:
                        return False
                    files = manifest.get("files")
                    if not isinstance(files, dict) or set(files) != set(_ADAPTER_FILES):
                        return False
                    for name in _ADAPTER_FILES:
                        record = files.get(name)
                        entry = _stat_regular_entry(directory_fd, name, "adapter file")
                        if (
                            not isinstance(record, dict)
                            or entry is None
                            or type(record.get("size")) is not int
                            or entry.st_size != record["size"]
                            or not isinstance(record.get("sha256"), str)
                            or _sha256(
                                Path(name),
                                deadline,
                                cancel_event,
                                parent_fd=directory_fd,
                            )
                            != record["sha256"]
                        ):
                            return False
                    if metadata(_ADAPTER_DIGEST, "ascii").strip() != files[_ADAPTER_FILES[1]]["sha256"]:
                        return False
                    transport = metadata("transport.json", "utf-8")
                    transport_value = json.loads(transport)
                    return (
                        isinstance(transport_value, dict)
                        and (source_url is None or transport_value.get("source_url") == source_url)
                        and transport_value.get("target_sha256") == files[_ADAPTER_FILES[1]]["sha256"]
                    )
                except (OSError, UnicodeError, ValueError, TypeError, KeyError):
                    return False

            try:
                current_fd = _open_directory_entry_nofollow(parent_fd, destination.name, "adapter destination")
            except FileNotFoundError:
                current_fd = None
            if current_fd is not None:
                os.close(current_fd)
                _cleanup_retired_adapters(
                    destination.parent,
                    destination.name,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
                return
            restored = False
            prefix = f"{destination.name}.retired-"
            names = []
            entry_count = 0
            with os.scandir(parent_fd) as entries:
                for entry in entries:
                    _check_deadline(deadline, cancel_event)
                    entry_count += 1
                    if entry_count > _ADAPTER_CLEANUP_ENTRY_LIMIT:
                        raise OSError("adapter cache has too many recovery entries")
                    if entry.name.startswith(prefix):
                        names.append(entry.name)
                        if len(names) > _ADAPTER_CLEANUP_CANDIDATE_LIMIT:
                            raise OSError("too many retired adapter candidates")
            for name in sorted(names, reverse=True):
                try:
                    candidate_fd = _open_directory_entry_nofollow(parent_fd, name, "retired adapter")
                except (FileNotFoundError, OSError):
                    continue
                try:
                    _check_deadline(deadline, cancel_event)
                    candidate_stat = os.fstat(candidate_fd)
                    current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    if (
                        stat.S_ISDIR(candidate_stat.st_mode)
                        and (candidate_stat.st_dev, candidate_stat.st_ino)
                        == (current_stat.st_dev, current_stat.st_ino)
                        and valid(candidate_fd)
                    ):
                        os.replace(
                            name,
                            destination.name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                        )
                        restored = True
                        break
                finally:
                    os.close(candidate_fd)
            if restored:
                _cleanup_retired_adapters(
                    destination.parent,
                    destination.name,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
        finally:
            os.close(parent_fd)


def _cleanup_previous_adapter_caches(
    cache_root: Path,
    active: Path,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Remove old adapters while retaining the active predecessor."""
    try:
        cache_fd = _open_directory_nofollow(
            cache_root,
            create=False,
            label="adapter cache directory",
        )
    except FileNotFoundError:
        return

    try:
        active_name = active.name
        ready_candidates: list[str] = []
        candidate_steps: dict[str, int] = {}

        def cached_step(directory_fd: int) -> int | None:
            for metadata_name in ("transport.json", "manifest.json"):
                try:
                    metadata = json.loads(
                        _read_text_nofollow(
                            metadata_name,
                            encoding="utf-8",
                            dir_fd=directory_fd,
                            max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                        )
                    )
                except (FileNotFoundError, OSError, ValueError):
                    continue
                source_url = metadata.get("source_url") if isinstance(metadata, dict) else None
                if not isinstance(source_url, str):
                    continue
                match = re.fullmatch(
                    r"step_(\d+)",
                    Path(urlsplit(source_url).path.rstrip("/")).name,
                )
                if match is not None:
                    return int(match.group(1))
            return None

        entry_count = 0
        with os.scandir(cache_fd) as entries:
            names = []
            for entry in entries:
                _check_deadline(deadline, cancel_event)
                entry_count += 1
                if entry_count > _ADAPTER_CLEANUP_ENTRY_LIMIT:
                    raise OSError("adapter cache has too many cleanup entries")
                names.append(entry.name)
        for name in names:
            _check_deadline(deadline, cancel_event)
            if name == active_name or re.fullmatch(r"[0-9a-f]{64}", name) is None:
                continue
            if len(ready_candidates) >= _ADAPTER_CLEANUP_CANDIDATE_LIMIT:
                raise OSError("too many cached adapter candidates")
            try:
                entry_stat = os.stat(name, dir_fd=cache_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(entry_stat.st_mode):
                raise OSError(f"cached adapter path is a symbolic link: {name}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue
            candidate_fd = _open_directory_entry_nofollow(
                cache_fd,
                name,
                "cached adapter path",
            )
            try:
                try:
                    ready_fd = _open_nofollow("READY", os.O_RDONLY, dir_fd=candidate_fd)
                except FileNotFoundError:
                    continue
                else:
                    os.close(ready_fd)
                ready_candidates.append(name)
                step = cached_step(candidate_fd)
                if step is not None:
                    candidate_steps[name] = step
            finally:
                os.close(candidate_fd)

        keep = {active_name}
        active_step = None
        try:
            active_fd = _open_directory_entry_nofollow(
                cache_fd,
                active_name,
                "active adapter path",
            )
            try:
                transport = json.loads(
                    _read_text_nofollow(
                        "transport.json",
                        encoding="utf-8",
                        dir_fd=active_fd,
                        max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                    )
                )
                active_step = cached_step(active_fd)
            finally:
                os.close(active_fd)
        except (FileNotFoundError, OSError, ValueError):
            transport = None
        for key in ("base_url", "base2_url"):
            predecessor_url = transport.get(key) if isinstance(transport, dict) else None
            if isinstance(predecessor_url, str):
                predecessor_name = adapter_cache_path(predecessor_url).name
                if predecessor_name in ready_candidates:
                    keep.add(predecessor_name)

        if active_step is not None:
            keep.update(
                name
                for name, step in candidate_steps.items()
                if step > active_step
            )

        for name in ready_candidates:
            if name in keep:
                continue
            try:
                lock_name = f".{name}.readers.lock"
                lock_fd = _open_nofollow(
                    lock_name,
                    os.O_RDWR | os.O_CREAT,
                    dir_fd=cache_fd,
                )
                with os.fdopen(lock_fd, "a+") as reader_lock:
                    fcntl.flock(reader_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _remove_directory_entry(
                        cache_fd,
                        name,
                        "cached adapter path",
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
                    fcntl.flock(reader_lock, fcntl.LOCK_UN)
            except (BlockingIOError, FileNotFoundError):
                continue
    finally:
        os.close(cache_fd)


def _cleanup_adapter_caches(
    cache_root: Path,
    active: Path,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    _cleanup_retired_adapters(
        cache_root,
        active.name,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    _cleanup_previous_adapter_caches(
        cache_root,
        active,
        deadline=deadline,
        cancel_event=cancel_event,
    )


def _full_cache_matches_manifest(
    candidate: Path,
    files: dict[str, object],
    *,
    verify_files: bool = True,
) -> bool:
    """Check a READY full cache using its already-published file proof."""
    _reject_symlink_path(candidate, "cached adapter path")
    try:
        directory_fd = _open_directory_nofollow(
            candidate,
            create=False,
            label="cached adapter path",
        )
    except FileNotFoundError:
        return False
    try:
        try:
            if (
                _read_bytes_nofollow(
                    "READY",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_MARKER_MAX_BYTES,
                )
                != b"ready\n"
            ):
                return False
            _reject_symlink_path(candidate / "manifest.json", "cached adapter manifest")
            manifest = json.loads(
                _read_text_nofollow(
                    "manifest.json",
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
            sidecar = _read_text_nofollow(
                _ADAPTER_DIGEST,
                encoding="ascii",
                dir_fd=directory_fd,
                max_bytes=_ADAPTER_MARKER_MAX_BYTES,
            ).strip()
        except (FileNotFoundError, ValueError, UnicodeError):
            return False
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or not isinstance(manifest.get("source_url"), str)
            or not manifest["source_url"].startswith(("http://", "https://"))
            or manifest.get("files") != files
        ):
            return False
        model_spec = files.get(_ADAPTER_FILES[1])
        if not isinstance(model_spec, dict) or sidecar != model_spec.get("sha256"):
            return False
        for filename, spec in files.items():
            if not isinstance(spec, dict) or type(spec.get("size")) is not int:
                return False
            entry = _stat_regular_entry(directory_fd, filename, "cached adapter file")
            if entry is None or entry.st_size != spec["size"]:
                return False
            if verify_files and _sha256_entry(
                directory_fd,
                filename,
                "cached adapter file",
            ) != spec.get("sha256"):
                return False
        return True
    finally:
        os.close(directory_fd)


def _reuse_full_adapter_cache(
    source_url: str,
    destination: Path,
    manifest: dict[str, object],
    transport: dict[str, object],
    *,
    cancel_event: threading.Event | None = None,
    committed_event: threading.Event | None = None,
    deadline: float | None = None,
) -> bool:
    """Publish a new URL entry from one verified local full cache."""
    deadline = deadline or (time.monotonic() + _ADAPTER_CLEANUP_TIMEOUT_S)
    if transport.get("mode") != "full":
        return False
    source_manifest = transport.get("source_manifest")
    files = manifest.get("files")
    if not isinstance(source_manifest, dict) or not isinstance(files, dict):
        raise OSError("full adapter cache reuse has no source manifest")
    if source_manifest.get("target_step") != 0:
        return False
    cache_root = destination.parent
    try:
        cache_fd = _open_directory_nofollow(
            cache_root,
            create=False,
            label="adapter cache directory",
        )
    except FileNotFoundError:
        return False
    try:
        entry_count = 0
        candidate_count = 0
        with os.scandir(cache_fd) as entries:
            names = []
            for entry in entries:
                _check_deadline(deadline, cancel_event)
                entry_count += 1
                if entry_count > _ADAPTER_CLEANUP_ENTRY_LIMIT:
                    raise OSError("adapter cache has too many reuse entries")
                names.append(entry.name)
        for name in sorted(names):
            if name == destination.name or re.fullmatch(r"[0-9a-f]{64}", name) is None:
                continue
            candidate_count += 1
            if candidate_count > _ADAPTER_CLEANUP_CANDIDATE_LIMIT:
                raise OSError("too many full adapter cache candidates")
            _check_deadline(deadline, cancel_event)
            candidate = cache_root / name
            _reject_symlink_path(candidate, "cached adapter path")
            candidate_stat = _path_stat(candidate, "cached adapter path")
            if candidate_stat is None:
                continue
            if not stat.S_ISDIR(candidate_stat.st_mode):
                raise OSError(f"cached adapter path is not a directory: {candidate}")
            with adapter_reader_lock(candidate, deadline=deadline):
                if not _full_cache_matches_manifest(candidate, files, verify_files=False):
                    continue
                if not _full_cache_matches_manifest(candidate, files):
                    return False
                _check_cancelled(cancel_event)
                temporary = destination.with_name(f"{destination.name}.partial")
                _reject_symlink_path(temporary, "adapter partial directory")
                _preflight_adapter_space(
                    cache_root,
                    [
                        (temporary / _ADAPTER_FILES[0], int(files[_ADAPTER_FILES[0]]["size"])),
                        (temporary / _ADAPTER_FILES[1], int(files[_ADAPTER_FILES[1]]["size"])),
                    ],
                )
                if _path_stat(temporary, "adapter partial directory") is not None:
                    parent_fd = _open_directory_nofollow(
                        cache_root,
                        create=False,
                        label="adapter partial directory parent",
                    )
                    try:
                        _remove_directory_entry(
                            parent_fd,
                            temporary.name,
                            "adapter partial directory",
                            deadline=deadline,
                            cancel_event=cancel_event,
                        )
                    finally:
                        os.close(parent_fd)
                _ensure_directory(temporary, "adapter partial directory")
                try:
                    _write_json(temporary / "manifest.json", manifest)
                    for filename in _ADAPTER_FILES:
                        _check_cancelled(cancel_event)
                        _copy_entry_nofollow(candidate, filename, temporary, filename)
                    _publish_relay_manifest(
                        temporary,
                        source_url,
                        source_manifest,
                        transport,
                        files,
                        deadline=deadline,
                        cancel_event=cancel_event,
                    )
                    model_sha256 = files[_ADAPTER_FILES[1]]["sha256"]
                    _write_text_atomic(
                        temporary / _ADAPTER_DIGEST,
                        f"{model_sha256}\n",
                        encoding="ascii",
                    )
                    with _adapter_stage_publish_lock:
                        _check_cancelled(cancel_event)
                        _write_text_atomic(temporary / "READY", "ready\n", encoding="ascii")
                        with os.fdopen(
                            _open_nofollow(
                                _adapter_reader_lock(destination),
                                os.O_RDWR | os.O_CREAT,
                            ),
                            "a+",
                        ) as reader_lock:
                            _acquire_stage_lock(reader_lock, deadline, cancel_event)
                            if _path_stat(destination, "adapter destination") is not None:
                                retired = destination.with_name(
                                    f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}"
                                )
                                _replace_paths(destination, retired, "retired adapter")
                            _replace_paths(temporary, destination, "adapter cache reuse")
                            if committed_event is not None:
                                committed_event.set()
                            fcntl.flock(reader_lock, fcntl.LOCK_UN)
                    print(f"adapter cache reuse: local full cache imported for {source_url}")
                    return True
                except BaseException:
                    try:
                        if _path_stat(temporary, "adapter partial directory") is not None:
                            parent_fd = _open_directory_nofollow(
                                cache_root,
                                create=False,
                                label="adapter partial directory parent",
                            )
                            try:
                                _remove_directory_entry(
                                    parent_fd,
                                    temporary.name,
                                    "adapter partial directory",
                                    deadline=deadline,
                                    cancel_event=cancel_event,
                                )
                            finally:
                                os.close(parent_fd)
                    except (FileNotFoundError, OSError):
                        pass
                    raise
    finally:
        os.close(cache_fd)
    print(f"adapter cache reuse: no verified full cache for {source_url}; using network staging")
    return False


def _download_range(
    source_urls: list[str],
    source_index: int,
    destination: Path,
    parent_fd: int,
    start: int,
    end: int,
    progress=None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    typed_transport: bool = False,
    retry_http_statuses: set[int] | None = None,
    per_request_timeout_seconds: float | None = None,
) -> None:
    """Download one inclusive byte range and resume its local part file."""
    destination_name = destination.name
    size = end - start + 1
    attempt_limit = (
        max(_ADAPTER_LANE_RETRIES, len(source_urls))
        if deadline is not None
        else _ADAPTER_LANE_RETRIES
    )
    for attempt in range(attempt_limit):
        _check_deadline(deadline, cancel_event)
        attempt_deadline = deadline
        if per_request_timeout_seconds is not None:
            active_timeout = _chunk_request_timeout(
                deadline,
                attempt,
                per_request_timeout_seconds,
                cancel_event,
            )
            attempt_deadline = time.monotonic() + active_timeout
            if deadline is not None:
                attempt_deadline = min(attempt_deadline, deadline)
        try:
            source_url = source_urls[(source_index + attempt) % len(source_urls)]
            entry_stat = _stat_regular_entry(parent_fd, destination_name, "adapter range file")
            present = entry_stat.st_size if entry_stat is not None else 0
            if present == size:
                return
            if present > size:
                os.unlink(destination_name, dir_fd=parent_fd)
                present = 0
            request = urllib.request.Request(
                source_url,
                headers={"Range": f"bytes={start + present}-{end}"},
            )
            request_timeout = (
                _request_timeout(deadline, cancel_event)
                if per_request_timeout_seconds is None
                else _request_timeout(
                    attempt_deadline,
                    cancel_event,
                    maximum=_ADAPTER_LANE_TIMEOUT_S,
                )
            )
            response_open = (
                _adapter_urlopen(request, request_timeout)
                if typed_transport
                else urllib.request.urlopen(
                    request,
                    timeout=request_timeout,
                )
            )
            with response_open as response:
                if response.status != 206:
                    if typed_transport:
                        raise _PersistentProtocolError(
                            "adapter server did not honor byte range"
                        )
                    raise OSError("adapter server did not honor byte range")
                with os.fdopen(
                    _open_nofollow(
                        destination_name,
                        os.O_WRONLY | os.O_CREAT | (os.O_APPEND if present else os.O_TRUNC),
                        dir_fd=parent_fd,
                        label="adapter range file",
                    ),
                    "ab" if present else "wb",
                ) as output:
                    if typed_transport:
                        chunks = _adapter_response_chunks(
                            response,
                            size - present,
                            attempt_deadline,
                            cancel_event,
                        )
                    else:
                        chunks = iter(())
                    if typed_transport:
                        for chunk in chunks:
                            output.write(chunk)
                            present += len(chunk)
                            _check_deadline(attempt_deadline, cancel_event)
                            if progress is not None:
                                progress(source_index, len(chunk))
                    else:
                        while present < size:
                            _check_deadline(attempt_deadline, cancel_event)
                            chunk = response.read(min(16 * 1024 * 1024, size - present))
                            if not chunk:
                                raise OSError(f"adapter range ended early: {destination.name}")
                            _check_deadline(attempt_deadline, cancel_event)
                            output.write(chunk)
                            present += len(chunk)
                            _check_deadline(attempt_deadline, cancel_event)
                            if progress is not None:
                                progress(source_index, len(chunk))
            entry_stat = _stat_regular_entry(parent_fd, destination_name, "adapter range file")
            if entry_stat is None or entry_stat.st_size != size:
                raise OSError(f"incomplete adapter range: {destination.name}")
            return
        except _PersistentTransportError:
            if not typed_transport:
                raise
            if not _adapter_retry_wait(
                attempt,
                attempt_limit,
                len(source_urls),
                deadline,
                cancel_event,
            ):
                raise
        except HTTPError as exc:
            if typed_transport and (
                retry_http_statuses is None or exc.code not in retry_http_statuses
            ):
                raise
            if not _adapter_retry_wait(
                attempt,
                attempt_limit,
                len(source_urls),
                deadline,
                cancel_event,
                exc.code if typed_transport else None,
            ):
                raise
        except OSError:
            if typed_transport:
                raise
            # socket.timeout is retryable; only the absolute stage deadline
            # remains terminal.
            _check_deadline(deadline, cancel_event)
            if attempt + 1 == attempt_limit:
                raise
            sleep_for = (
                lane_rotation_delay_seconds
                if len(source_urls) > 1
                else min(2**attempt, 10)
            )
            _check_cancelled(cancel_event)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("adapter delta staging exceeded its time limit")
                sleep_for = min(sleep_for, remaining)
            if cancel_event is None:
                time.sleep(sleep_for)
            else:
                cancel_event.wait(sleep_for)


class _PersistentTransportError(OSError):
    """A connection or response body failed before the requested bytes arrived."""


class _PersistentProtocolError(OSError):
    """A response violated the adapter transport contract."""


class _PersistentLocalError(OSError):
    """A local adapter destination failed and must not be retried."""


def _adapter_urlopen(request, timeout: float):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except HTTPError:
        raise
    except (OSError, http.client.HTTPException, EOFError) as exc:
        raise _PersistentTransportError("adapter network request failed") from exc


def _adapter_response_chunks(
    response,
    size: int,
    deadline: float | None,
    cancel_event: threading.Event | None,
):
    read_available = getattr(response, "read1", None) or response.read
    remaining = size
    while remaining:
        _check_deadline(deadline, cancel_event)
        try:
            chunk = read_available(min(16 * 1024 * 1024, remaining))
        except (OSError, http.client.HTTPException, EOFError) as exc:
            raise _PersistentTransportError("adapter network read failed") from exc
        _check_deadline(deadline, cancel_event)
        if not chunk:
            raise _PersistentTransportError("adapter response ended early")
        if len(chunk) > remaining:
            raise _PersistentProtocolError("adapter response body is too long")
        yield chunk
        remaining -= len(chunk)


def _adapter_retry_wait(
    attempt: int,
    attempt_limit: int,
    source_count: int,
    deadline: float | None,
    cancel_event: threading.Event | None,
    status_code: int | None = None,
) -> bool:
    _check_deadline(deadline, cancel_event)
    if attempt + 1 == attempt_limit:
        return False
    sleep_for = (
        1.0
        if status_code in {404, 503}
        else (lane_rotation_delay_seconds if source_count > 1 else min(2**attempt, 10))
    )
    _check_cancelled(cancel_event)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("adapter download exceeded its time limit")
        sleep_for = min(sleep_for, remaining)
    if cancel_event is None:
        time.sleep(sleep_for)
    else:
        cancel_event.wait(sleep_for)
    return True


class _PersistentAdapterConnection:
    """One private HTTP connection that is used by one adapter lane at a time."""

    def __init__(self) -> None:
        self._connection = None
        self._origin = None
        self._last_completed_at = None

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._origin = None
        self._last_completed_at = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _open(self, source_url: str, timeout: float):
        parsed = urlsplit(source_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise _PersistentProtocolError("adapter source URL is not HTTP or HTTPS")
        try:
            port = parsed.port
        except ValueError as exc:
            raise _PersistentProtocolError("adapter source URL has an invalid port") from exc
        origin = (parsed.scheme, parsed.hostname, port)
        if (
            self._connection is not None
            and self._last_completed_at is not None
            and time.monotonic() - self._last_completed_at
            >= persistent_connection_idle_seconds
        ):
            self.close()
        if self._connection is not None and self._origin != origin:
            self.close()
        if self._connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            self._connection = connection_type(
                parsed.hostname,
                port,
                timeout=timeout,
            )
            self._origin = origin
        else:
            self._connection.timeout = timeout
            if self._connection.sock is not None:
                self._connection.sock.settimeout(timeout)
        return self._connection, parsed

    def read_range(
        self,
        source_url: str,
        *,
        start: int,
        end: int,
        total: int,
        expected_sha256: str | None,
        timeout: float,
        stage_deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        consume,
    ) -> None:
        if timeout <= 0 or persistent_read_timeout_seconds <= 0:
            raise OSError("adapter persistent read timeout must be positive")
        attempt_deadline = time.monotonic() + timeout
        if stage_deadline is not None:
            attempt_deadline = min(attempt_deadline, stage_deadline)

        def remaining_timeout() -> float:
            _check_cancelled(cancel_event)
            remaining = attempt_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("adapter persistent request exceeded its time limit")
            return min(persistent_read_timeout_seconds, remaining)

        try:
            connection, parsed = self._open(source_url, remaining_timeout())
        except _PersistentProtocolError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException, EOFError) as exc:
            self.close()
            raise _PersistentTransportError("adapter persistent request failed") from exc
        request_path = parsed.path or "/"
        if parsed.query:
            request_path += f"?{parsed.query}"
        try:
            connection.request(
                "GET",
                request_path,
                headers={
                    "Connection": "keep-alive",
                    "Range": f"bytes={start}-{end}",
                },
            )
            if connection.sock is not None:
                connection.sock.settimeout(remaining_timeout())
            response = connection.getresponse()
            remaining_timeout()
        except (OSError, TimeoutError, http.client.HTTPException, EOFError) as exc:
            self.close()
            raise _PersistentTransportError("adapter persistent request failed") from exc

        def close_response() -> None:
            try:
                response.close()
            except OSError:
                pass

        if response.status in {404, 503}:
            status = response.status
            reason = response.reason or "adapter source unavailable"
            close_response()
            self.close()
            raise HTTPError(source_url, status, reason, response.headers, None)
        if response.status != 206:
            close_response()
            self.close()
            raise _PersistentProtocolError(
                f"adapter server returned unexpected HTTP status {response.status}"
            )

        expected_length = end - start + 1
        content_length = response.getheader("Content-Length")
        if content_length is None or not content_length.isdigit() or int(content_length) != expected_length:
            close_response()
            self.close()
            raise _PersistentProtocolError("adapter response Content-Length differs")
        content_range = response.getheader("Content-Range")
        expected_range = f"bytes {start}-{end}/{total}"
        if content_range != expected_range:
            close_response()
            self.close()
            raise _PersistentProtocolError("adapter response Content-Range differs")
        etag = response.getheader("ETag")
        if etag is not None:
            normalized_etag = etag.strip()
            if (
                len(normalized_etag) < 2
                or normalized_etag[0] != '"'
                or normalized_etag[-1] != '"'
                or expected_sha256 is None
                or normalized_etag[1:-1] != expected_sha256
            ):
                close_response()
                self.close()
                raise _PersistentProtocolError("adapter response ETag differs")

        if connection.sock is not None:
            try:
                connection.sock.settimeout(remaining_timeout())
            except (OSError, TimeoutError) as exc:
                close_response()
                self.close()
                raise _PersistentTransportError(
                    "adapter response read timeout setup failed"
                ) from exc
        remaining = expected_length
        try:
            while remaining:
                if connection.sock is not None:
                    try:
                        connection.sock.settimeout(remaining_timeout())
                    except (OSError, TimeoutError) as exc:
                        self.close()
                        raise _PersistentTransportError(
                            "adapter response read timeout setup failed"
                        ) from exc
                try:
                    chunk = response.read(min(16 * 1024 * 1024, remaining))
                except (OSError, TimeoutError, http.client.HTTPException, EOFError) as exc:
                    self.close()
                    raise _PersistentTransportError("adapter response ended during read") from exc
                remaining_timeout()
                if not chunk:
                    self.close()
                    raise _PersistentTransportError("adapter response body ended early")
                if len(chunk) > remaining:
                    self.close()
                    raise _PersistentProtocolError("adapter response body is too long")
                try:
                    consume(chunk)
                except OSError as exc:
                    self.close()
                    if isinstance(exc, (TimeoutError, socket.timeout)):
                        raise
                    raise _PersistentLocalError(
                        f"adapter local destination write failed: {exc}"
                    ) from exc
                except BaseException:
                    self.close()
                    raise
                remaining -= len(chunk)
                remaining_timeout()
            self._last_completed_at = time.monotonic()
        finally:
            close_response()


class _PersistentAdapterLanePool:
    """Keep one locked connection per source lane and bound retries to that lane."""

    def __init__(
        self,
        source_urls_or_count: list[str] | int,
        width: int | None = None,
    ) -> None:
        if isinstance(source_urls_or_count, int):
            self._source_urls = []
            primary_urls = [None] * max(1, source_urls_or_count)
        else:
            self._source_urls = list(source_urls_or_count)
            lane_count = len(self._source_urls) if width is None else width
            if lane_count < 1 or lane_count > len(self._source_urls):
                raise OSError("persistent adapter lane width is invalid")
            primary_urls = self._source_urls[:lane_count]
        self._primary_urls = primary_urls
        self._lanes = [
            (threading.Lock(), _PersistentAdapterConnection())
            for _ in primary_urls
        ]

    @property
    def slot_count(self) -> int:
        return len(self._lanes)

    def urls_for_lane(self, lane: int, name: str) -> list[str]:
        if not self._source_urls:
            raise OSError("persistent adapter lane URLs are unavailable")
        primary = self._primary_urls[lane % self.slot_count]
        assert isinstance(primary, str)
        return [
            f"{primary.rstrip('/')}/{name}",
            *(
                f"{source.rstrip('/')}/{name}"
                for source in self._source_urls
                if source != primary
            ),
        ]

    def close(self) -> None:
        for _lock, connection in self._lanes:
            with _lock:
                connection.close()

    def download(
        self,
        source_urls: list[str],
        lane: int,
        destination: Path,
        parent_fd: int,
        *,
        expected_size: int,
        expected_sha256: str,
        progress=None,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        per_request_timeout_seconds: float | None = None,
    ) -> None:
        if not source_urls:
            raise OSError("adapter download has no source lane")
        lock, connection = self._lanes[lane % len(self._lanes)]
        attempt_limit = (
            max(_ADAPTER_LANE_RETRIES, len(source_urls))
            if deadline is not None
            else _ADAPTER_LANE_RETRIES
        )
        with lock:
            for attempt in range(attempt_limit):
                _check_deadline(deadline, cancel_event)
                entry_stat = _stat_regular_entry(
                    parent_fd,
                    destination.name,
                    "adapter transport chunk",
                )
                present = entry_stat.st_size if entry_stat is not None else 0
                if present > expected_size:
                    os.unlink(destination.name, dir_fd=parent_fd)
                    present = 0
                elif present == expected_size:
                    if _sha256(
                        destination,
                        deadline,
                        cancel_event,
                        parent_fd=parent_fd,
                    ) == expected_sha256:
                        return
                    os.unlink(destination.name, dir_fd=parent_fd)
                    present = 0

                source_url = source_urls[attempt % len(source_urls)]
                timeout = (
                    _request_timeout(deadline, cancel_event)
                    if per_request_timeout_seconds is None
                    else _chunk_request_timeout(
                        deadline,
                        attempt,
                        per_request_timeout_seconds,
                        cancel_event,
                    )
                )
                try:
                    with os.fdopen(
                        _open_nofollow(
                            destination.name,
                            os.O_WRONLY
                            | os.O_CREAT
                            | (os.O_APPEND if present else os.O_TRUNC),
                            dir_fd=parent_fd,
                            label="adapter transport chunk",
                        ),
                        "ab" if present else "wb",
                    ) as output:
                        offset = present

                        def consume(chunk: bytes) -> None:
                            nonlocal offset
                            _check_deadline(deadline, cancel_event)
                            output.write(chunk)
                            offset += len(chunk)
                            if progress is not None:
                                progress(lane, len(chunk))

                        connection.read_range(
                            source_url,
                            start=present,
                            end=expected_size - 1,
                            total=expected_size,
                            expected_sha256=expected_sha256,
                            timeout=timeout,
                            stage_deadline=deadline,
                            cancel_event=cancel_event,
                            consume=consume,
                        )
                except HTTPError as exc:
                    connection.close()
                    if exc.code not in {404, 503} or attempt + 1 == attempt_limit:
                        raise
                except _PersistentTransportError:
                    connection.close()
                    if attempt + 1 == attempt_limit:
                        raise
                except (TimeoutError, socket.timeout):
                    connection.close()
                    if attempt + 1 == attempt_limit:
                        raise
                except _PersistentProtocolError:
                    connection.close()
                    raise
                except _PersistentLocalError:
                    connection.close()
                    raise
                else:
                    entry_stat = _stat_regular_entry(
                        parent_fd,
                        destination.name,
                        "adapter transport chunk",
                    )
                    if entry_stat is None or entry_stat.st_size != expected_size:
                        connection.close()
                        if attempt + 1 == attempt_limit:
                            raise _PersistentTransportError(
                                "incomplete adapter transport chunk"
                            )
                    elif _sha256(
                        destination,
                        deadline,
                        cancel_event,
                        parent_fd=parent_fd,
                    ) != expected_sha256:
                        connection.close()
                        raise OSError("adapter transport chunk checksum mismatch")
                    else:
                        return

                _check_deadline(deadline, cancel_event)
                sleep_for = (
                    lane_rotation_delay_seconds
                    if len(source_urls) > 1
                    else min(2**attempt, 10)
                )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter delta staging exceeded its time limit")
                    sleep_for = min(sleep_for, remaining)
                _check_cancelled(cancel_event)
                if cancel_event is None:
                    time.sleep(sleep_for)
                else:
                    cancel_event.wait(sleep_for)
        raise OSError("adapter persistent download retries exhausted")


def _download_persistent_file(
    source_urls: list[str],
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    progress=None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    per_request_timeout_seconds: float | None = None,
) -> None:
    """Download one small adapter file with typed persistent retries."""
    if not source_urls:
        raise OSError("adapter download has no source lane")
    pool = _PersistentAdapterLanePool(source_urls, width=1)
    parent_fd = _open_directory_nofollow(
        destination.parent,
        create=False,
        label="adapter download parent",
    )
    try:
        pool.download(
            source_urls,
            0,
            destination,
            parent_fd,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            progress=progress,
            deadline=deadline,
            cancel_event=cancel_event,
            per_request_timeout_seconds=per_request_timeout_seconds,
        )
    finally:
        pool.close()
        os.close(parent_fd)


def _sha256(
    path: Path,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    *,
    parent_fd: int | None = None,
) -> str:
    digest = hashlib.sha256()
    with os.fdopen(
        _open_nofollow(
            path.name if parent_fd is not None else path,
            os.O_RDONLY,
            dir_fd=parent_fd,
            label="adapter checksum file",
        ),
        "rb",
    ) as source:
        while True:
            _check_deadline(deadline, cancel_event)
            chunk = source.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    _check_deadline(deadline, cancel_event)
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


def _decode_linear_transport_chunk(
    base2: bytes,
    base1: bytes,
    payload: bytes,
    *,
    offset: int,
    data_offset: int,
    raw_size: int,
    codec: str,
) -> tuple[bytes, str]:
    """Decode one version-two chunk and return its canonical correction digest."""
    if codec not in _ADAPTER_KNOWN_CHUNK_CODECS:
        raise OSError("version-two adapter chunk codec is unsupported")
    try:
        from prime_rl.utils.adapter_delta import decode_linear_zigzag_sparse_chunk
    except ImportError as exc:
        raise OSError("version-two adapter transport decoder is unavailable") from exc
    try:
        target, correction = decode_linear_zigzag_sparse_chunk(
            base2,
            base1,
            payload,
            offset=offset,
            data_offset=data_offset,
            raw_size=raw_size,
            codec=codec,
        )
        correction_sha256 = hashlib.sha256(correction).hexdigest()
    except Exception as exc:
        raise OSError("version-two adapter transport reconstruction failed") from exc
    if not isinstance(target, bytes) or not isinstance(correction_sha256, str):
        raise OSError("version-two decoder did not return target and correction identity")
    if len(target) != raw_size or re.fullmatch(r"[0-9a-f]{64}", correction_sha256) is None:
        raise OSError("version-two adapter chunk has the wrong reconstructed size")
    return target, correction_sha256


def _verified_adapter_base(
    base_url: str,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
    expected_config_sha256: str | None = None,
    expected_config_size: int | None = None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, Path]:
    """Return a cached base only after checking its published identity."""
    _check_deadline(deadline, cancel_event)
    base_dir = adapter_cache_path(base_url)
    model = base_dir / _ADAPTER_FILES[1]
    config = base_dir / _ADAPTER_FILES[0]
    parent_fd = _open_directory_nofollow(
        base_dir.parent,
        create=False,
        label="cached adapter parent",
    )
    try:
        base_fd = _open_directory_entry_nofollow(
            parent_fd,
            base_dir.name,
            "cached adapter directory",
        )
    finally:
        os.close(parent_fd)
    try:
        if (
            _read_bytes_nofollow(
                "READY",
                dir_fd=base_fd,
                max_bytes=_ADAPTER_MARKER_MAX_BYTES,
            )
            != b"ready\n"
        ):
            raise OSError("required base adapter is not cached")
        model_stat = _stat_regular_entry(base_fd, _ADAPTER_FILES[1], "cached adapter model")
        config_stat = _stat_regular_entry(
            base_fd,
            _ADAPTER_FILES[0],
            "cached adapter configuration",
        )
        if model_stat is None or config_stat is None:
            raise OSError("required base adapter is not cached")
        try:
            local_manifest = json.loads(
                _read_text_nofollow(
                    "manifest.json",
                    encoding="utf-8",
                    dir_fd=base_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise OSError("cached base adapter has no valid manifest") from exc
        if (
            not isinstance(local_manifest, dict)
            or set(local_manifest) != {"version", "source_url", "files"}
            or local_manifest.get("version") != 1
            or local_manifest.get("source_url") != base_url
        ):
            raise OSError("cached base adapter URL differs from requested base")
        files = local_manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(_ADAPTER_FILES):
            raise OSError("cached base adapter manifest has no files")
        model_spec = files.get(_ADAPTER_FILES[1])
        config_spec = files.get(_ADAPTER_FILES[0])
        if (
            not isinstance(model_spec, dict)
            or set(model_spec) != {"size", "sha256"}
            or type(model_spec.get("size")) is not int
            or model_spec["size"] <= 0
            or not isinstance(model_spec.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", model_spec["sha256"]) is None
            or model_spec["sha256"] != expected_sha256
        ):
            raise OSError("cached base adapter manifest checksum differs")
        if (
            not isinstance(config_spec, dict)
            or set(config_spec) != {"size", "sha256"}
            or type(config_spec.get("size")) is not int
            or config_spec["size"] < 0
            or not isinstance(config_spec.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", config_spec["sha256"]) is None
        ):
            raise OSError("cached base adapter configuration checksum differs")
        if model_stat.st_size != model_spec["size"] or config_stat.st_size != config_spec["size"]:
            raise OSError("cached base adapter size differs")
        if expected_size is not None and (
            type(expected_size) is not int
            or expected_size <= 0
            or model_stat.st_size != expected_size
            or model_spec["size"] != expected_size
        ):
            raise OSError("cached base adapter size differs")
        if expected_config_sha256 is not None and (
            config_spec["sha256"] != expected_config_sha256
        ):
            raise OSError("cached base adapter configuration checksum differs")
        if expected_config_size is not None and (
            type(expected_config_size) is not int
            or expected_config_size < 0
            or config_stat.st_size != expected_config_size
            or config_spec["size"] != expected_config_size
        ):
            raise OSError("cached base adapter configuration size differs")
        checksum = _read_bytes_nofollow(
            _ADAPTER_DIGEST,
            dir_fd=base_fd,
            max_bytes=_ADAPTER_MARKER_MAX_BYTES,
        )
        if checksum != f"{expected_sha256}\n".encode("ascii"):
            raise OSError("cached base adapter sidecar checksum differs")
        if _sha256_entry(
            base_fd,
            _ADAPTER_FILES[0],
            "cached base adapter configuration",
        ) != config_spec["sha256"]:
            raise OSError("cached base adapter configuration bytes differ")
    finally:
        os.close(base_fd)
    _check_deadline(deadline, cancel_event)
    return model, config


def _preallocate(path: Path, size: int) -> None:
    if not hasattr(os, "posix_fallocate"):
        raise OSError("adapter receiver requires posix_fallocate")
    parent_fd = _open_directory_nofollow(
        path.parent,
        create=False,
        label="adapter preallocation parent",
    )
    try:
        file_descriptor = _open_nofollow(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            dir_fd=parent_fd,
            label="adapter preallocation file",
        )
        try:
            os.posix_fallocate(file_descriptor, 0, size)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(parent_fd)


def _allocated_bytes(path: Path, label: str) -> int:
    entry_stat = _path_stat(path, label)
    if entry_stat is None:
        return 0
    blocks = getattr(entry_stat, "st_blocks", None)
    if blocks is None:
        return entry_stat.st_size
    return max(0, int(blocks)) * 512


def _preflight_adapter_space(
    root: Path,
    allocations: list[tuple[Path, int]],
) -> None:
    """Check only bytes this stage must allocate; cached bases need no reserve."""
    required = 0
    for path, size in allocations:
        if type(size) is not int or size < 0:
            raise OSError(f"adapter allocation size is invalid: {path}")
        required += max(0, size - _allocated_bytes(path, "adapter staging file"))
    try:
        available = shutil.disk_usage(root).free
    except OSError as exc:
        raise OSError(f"adapter staging free-space check failed: {root}") from exc
    if available < required:
        raise OSError(
            f"insufficient adapter staging space: need {required} bytes, "
            f"have {available} bytes free"
        )


def _preflight_full_adapter_download_space(
    temporary: Path,
    files: dict[str, object],
) -> None:
    """Reserve the peak temporary and range-part space for a full download."""
    config = files.get(_ADAPTER_FILES[0])
    model = files.get(_ADAPTER_FILES[1])
    if not isinstance(config, dict) or not isinstance(model, dict):
        raise OSError("full adapter manifest has no file sizes")
    config_size = config.get("size")
    model_size = model.get("size")
    if (
        type(config_size) is not int
        or config_size < 0
        or type(model_size) is not int
        or model_size <= 0
    ):
        raise OSError("full adapter manifest has invalid file sizes")
    streams = _configured_download_streams()
    if model_size >= _PARALLEL_DOWNLOAD_THRESHOLD and streams > 1:
        chunk = (model_size + streams - 1) // streams
        part_allocations = []
        for index, start in enumerate(range(0, model_size, chunk)):
            part_size = min(start + chunk, model_size) - start
            part_allocations.append(
                (
                    temporary
                    / f"{_ADAPTER_FILES[1]}.parts"
                    / f"part-{index:03d}",
                    part_size,
                )
            )
        _preflight_adapter_space(
            temporary.parent,
            [
                (temporary / _ADAPTER_FILES[0], config_size),
                (temporary / f"{_ADAPTER_FILES[1]}.assembling", model_size),
                *part_allocations,
            ],
        )
    else:
        _preflight_adapter_space(
            temporary.parent,
            [
                (temporary / _ADAPTER_FILES[0], config_size),
                (temporary / _ADAPTER_FILES[1], model_size),
            ],
        )


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


def _transport_raw_sha256(delta: bytes, *, codec: str = _ADAPTER_DELTA_TRANSPORT) -> str:
    if codec == _ADAPTER_FULL_TRANSPORT:
        return hashlib.sha256(delta).hexdigest()
    try:
        from prime_rl.utils.adapter_delta import shuffle_bf16_bytes
    except ImportError as exc:
        raise OSError("chunked adapter shuffle helper is unavailable") from exc
    return hashlib.sha256(shuffle_bf16_bytes(delta)).hexdigest()


def _stage_chunked_linear_adapter(
    source_url: str,
    transport: dict[str, object],
    source_manifest: dict[str, object],
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    committed_event: threading.Event | None = None,
    status_metadata: dict[str, object] | None = None,
) -> Path:
    """Stage a version-two adapter from two exact cached predecessors."""
    cancel_event = cancel_event or threading.Event()
    destination = adapter_cache_path(source_url)
    started = time.monotonic()
    _check_deadline(deadline, cancel_event)
    temporary = destination.with_name(f"{destination.name}.partial")
    _reject_symlink_path(destination, "adapter destination")
    _reject_symlink_path(temporary, "adapter partial directory")
    base_url = transport["base_url"]
    base_sha256 = transport["base_sha256"]
    base_size = transport["base_size"]
    base_config_size = transport["base_config_size"]
    base_config_sha256 = transport["base_config_sha256"]
    base2_url = transport["base2_url"]
    base2_sha256 = transport["base2_sha256"]
    base2_size = transport["base2_size"]
    base2_config_size = transport["base2_config_size"]
    base2_config_sha256 = transport["base2_config_sha256"]
    target_sha256 = transport["target_sha256"]
    target_size = transport["target_size"]
    config_sha256 = transport["config_sha256"]
    config_size = transport["config_size"]
    if not all(isinstance(value, str) for value in (base_url, base_sha256, base2_url, base2_sha256, target_sha256, config_sha256, base_config_sha256, base2_config_sha256)):
        raise OSError("version-two adapter identity is invalid")
    if not all(
        isinstance(value, int)
        for value in (
            base_size,
            base_config_size,
            base2_size,
            base2_config_size,
            target_size,
            config_size,
        )
    ):
        raise OSError("version-two adapter sizes are invalid")
    base2_model, _ = _verified_adapter_base(
        base2_url,
        base2_sha256,
        expected_size=base2_size,
        expected_config_sha256=base2_config_sha256,
        expected_config_size=base2_config_size,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    base1_model, _ = _verified_adapter_base(
        base_url,
        base_sha256,
        expected_size=base_size,
        expected_config_sha256=base_config_sha256,
        expected_config_size=base_config_size,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    chunks = transport["chunks"]
    if not isinstance(chunks, list):
        raise OSError("version-two adapter chunks are invalid")
    chunks = sorted(chunks, key=lambda item: int(item["offset"]))
    transport_identity = json.dumps(transport, sort_keys=True, separators=(",", ":"))
    identity_path = temporary / "transport-manifest.json"
    _reset_staging_directory_identity(
        temporary,
        identity_path.name,
        transport_identity,
        "adapter partial directory",
    )
    _ensure_directory(temporary, "adapter partial directory")
    _write_text_atomic(identity_path, transport_identity, encoding="utf-8")
    chunk_root = temporary / "transport-chunks"
    _ensure_directory(chunk_root, "adapter transport chunk directory")
    total_wire_bytes = sum(int(item["wire_size"]) for item in chunks) + config_size
    progress_lock = threading.Lock()
    progress_bytes = 0
    for item in chunks:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            continue
        chunk_path = chunk_root / Path(str(item["file"])).name
        _reject_symlink_path(chunk_path, "adapter transport chunk")
        if _path_regular_size(chunk_path, "adapter transport chunk") != int(item["wire_size"]):
            continue
        done_path = _proof_path(chunk_path, _COMPLETED_SUFFIX)
        wire_ready_path = _proof_path(chunk_path, _WIRE_READY_SUFFIX)
        done_valid, _ = _proof_identity_matches(done_path, item, completed=True)
        wire_ready_valid, _ = _proof_identity_matches(wire_ready_path, item, completed=False)
        if done_valid or wire_ready_valid:
            progress_bytes += int(item["wire_size"])
    reconstruct_lock = threading.Lock()
    reconstruct_worker_seconds = 0.0
    reconstruction_started = None
    reconstruction_finished = None
    transfer_started = None
    transfer_finished = None
    verification_started = None
    verification_finished = None
    last_status = 0.0

    def write_stage_status(value: dict[str, object]) -> None:
        _write_stage_status(
            destination.parent,
            {
                **value,
                **_stage_timing_fields(
                    started,
                    transfer_started=transfer_started,
                    transfer_finished=transfer_finished,
                    reconstruction_started=reconstruction_started,
                    reconstruction_finished=reconstruction_finished,
                    reconstruction_worker_seconds=reconstruct_worker_seconds,
                    verification_started=verification_started,
                    verification_finished=verification_finished,
                ),
                **(status_metadata or {}),
            },
        )

    def report(lane: int, amount: int) -> None:
        nonlocal progress_bytes, last_status
        _check_deadline(deadline, cancel_event)
        with progress_lock:
            progress_bytes += amount
            now = time.monotonic()
            if now - last_status < _ADAPTER_STATUS_INTERVAL_S and progress_bytes < total_wire_bytes:
                return
            last_status = now
            write_stage_status(
                {
                    "status": "staging",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "transport": _ADAPTER_LINEAR_TRANSPORT,
                    "bytes_done": progress_bytes,
                    "bytes_total": total_wire_bytes,
                    "lane": lane,
                    "updated_at": time.time(),
                    "elapsed_seconds": round(now - started, 3),
                },
            )

    write_stage_status(
        {
            "status": "staging",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_LINEAR_TRANSPORT,
            "verified_files": source_manifest["files"],
            "bytes_done": progress_bytes,
            "bytes_total": total_wire_bytes,
            "updated_at": time.time(),
            "elapsed_seconds": 0.0,
        },
    )
    lane_urls, source_connection_count = _responsive_adapter_source_urls(
        source_url,
        deadline,
        cancel_event,
    )
    status_metadata = {
        **(status_metadata or {}),
        "source_connections": source_connection_count,
        "responsive_source_connections": len(lane_urls),
    }
    output_path = temporary / _ADAPTER_FILES[1]
    config_path = temporary / _ADAPTER_FILES[0]
    config_incoming_path = config_path.with_name(f".{config_path.name}.incoming")
    _reject_symlink_path(output_path, "adapter output")
    _reject_symlink_path(config_path, "adapter configuration file")
    _reject_symlink_path(config_incoming_path, "adapter configuration incoming file")
    chunk_allocations: list[tuple[Path, int]] = []
    for item in chunks:
        chunk_path = chunk_root / Path(str(item["file"])).name
        incoming_path = chunk_path.with_name(f".{chunk_path.name}.incoming")
        _reject_symlink_path(chunk_path, "adapter transport chunk")
        _reject_symlink_path(incoming_path, "adapter chunk incoming file")
        wire_size = int(item["wire_size"])
        reusable = False
        for proof_path, completed in (
            (_proof_path(chunk_path, _COMPLETED_SUFFIX), True),
            (_proof_path(chunk_path, _WIRE_READY_SUFFIX), False),
        ):
            proof_valid, _ = _proof_identity_matches(
                proof_path, item, completed=completed
            )
            if not proof_valid:
                continue
            try:
                reusable = (
                    _path_regular_size(chunk_path, "adapter transport chunk") == wire_size
                    and _sha256(chunk_path, deadline, cancel_event)
                    == str(item["wire_sha256"])
                )
            except (FileNotFoundError, OSError):
                reusable = False
            if reusable:
                break
        if not reusable:
            # The old final wire file remains until incoming is atomically
            # renamed over it, so reserve the incoming allocation as peak use.
            chunk_allocations.append((incoming_path, wire_size))
    _preflight_adapter_space(
        temporary,
        [(output_path, target_size), (config_incoming_path, config_size)]
        + chunk_allocations,
    )
    if _path_regular_size(output_path, "adapter output") != target_size:
        _preallocate(output_path, target_size)

    # Publish configuration atomically before the relay index. The peer must
    # never observe a relay index that points at an unfinished config file.
    with reconstruct_lock:
        transfer_started = time.monotonic()
    config_urls = [
        f"{lane_url.rstrip('/')}/{_ADAPTER_FILES[0]}"
        for lane_url in lane_urls
    ]
    _download_persistent_file(
        config_urls,
        config_incoming_path,
        expected_size=config_size,
        expected_sha256=config_sha256,
        progress=report,
        deadline=deadline,
        cancel_event=cancel_event,
        per_request_timeout_seconds=_configured_chunk_timeout(),
    )
    with reconstruct_lock:
        transfer_finished = time.monotonic()
    _reject_symlink_path(config_path, "adapter configuration file")
    _replace_paths(config_incoming_path, config_path, "adapter configuration")

    # The origin publishes the exact v2 index before transfer workers start.
    # The relay uses RELAYING to expose manifests, configuration, and verified
    # chunks. It never exposes this incomplete model.
    source_transport_manifest = transport.get("source_manifest")
    if not isinstance(source_transport_manifest, dict):
        raise OSError("version-two adapter source manifest is unavailable for relay")
    reader_lock_path = _adapter_reader_lock(destination)
    with os.fdopen(
        _open_nofollow(reader_lock_path, os.O_RDWR | os.O_CREAT),
        "a+",
    ):
        pass
    _publish_relay_manifest(
        temporary,
        source_url,
        source_transport_manifest,
        transport,
        source_manifest["files"],
        allow_partial=True,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    _write_marker(temporary / _RELAYING_MARKER, "relaying\n")

    def fetch_wire(index: int, item: dict[str, object]) -> None:
        nonlocal transfer_started, transfer_finished
        _check_cancelled(cancel_event)
        name = item["file"]
        if not isinstance(name, str):
            raise OSError("version-two adapter chunk has no file")
        chunk_path = chunk_root / name
        offset = int(item["offset"])
        data_offset = int(item["data_offset"])
        raw_size = int(item["raw_size"])
        wire_size = int(item["wire_size"])
        wire_sha256 = str(item["wire_sha256"])
        raw_sha256 = str(item["raw_sha256"])
        completed_path = _proof_path(chunk_path, _COMPLETED_SUFFIX)
        wire_ready_path = _proof_path(chunk_path, _WIRE_READY_SUFFIX)
        _reject_symlink_path(chunk_path, "adapter transport chunk")
        _reject_symlink_path(completed_path, "adapter completed proof")
        _reject_symlink_path(wire_ready_path, "adapter wire proof")
        completed_valid, _ = _proof_identity_matches(
            completed_path, item, completed=True
        )
        if completed_valid:
            try:
                completed_valid = (
                    _path_regular_size(chunk_path, "adapter transport chunk") == wire_size
                    and _sha256(chunk_path, deadline, cancel_event) == wire_sha256
                )
            except (FileNotFoundError, OSError):
                completed_valid = False
            if completed_valid:
                # The final file checksum validates all resumed regions together.
                _discard_proof(wire_ready_path)
                return
            _discard_proof(completed_path)
        if _path_stat(completed_path, "adapter completed proof") is not None:
            _discard_proof(completed_path)

        wire_ready_valid, _ = _proof_identity_matches(
            wire_ready_path, item, completed=False
        )
        if wire_ready_valid:
            try:
                if not (
                    _path_regular_size(chunk_path, "adapter transport chunk") == wire_size
                    and _sha256(chunk_path, deadline, cancel_event) == wire_sha256
                ):
                    wire_ready_valid = False
            except (FileNotFoundError, OSError):
                wire_ready_valid = False
            if not wire_ready_valid:
                _discard_proof(wire_ready_path)
        if not wire_ready_valid:
            if _path_stat(wire_ready_path, "adapter wire proof") is not None:
                _discard_proof(wire_ready_path)

            chunk_lane = index % len(lane_urls)

            def lane_report(_lane: int, amount: int) -> None:
                report(chunk_lane, amount)

            chunk_urls = [f"{lane_url.rstrip('/')}/{name}" for lane_url in lane_urls]
            chunk_urls = chunk_urls[chunk_lane:] + chunk_urls[:chunk_lane]
            incoming_path = chunk_path.with_name(f".{chunk_path.name}.incoming")
            transfer_started_at = time.monotonic()
            with reconstruct_lock:
                if transfer_started is None or transfer_started_at < transfer_started:
                    transfer_started = transfer_started_at
            _download_persistent_file(
                chunk_urls,
                incoming_path,
                expected_size=wire_size,
                expected_sha256=wire_sha256,
                progress=lane_report,
                deadline=deadline,
                cancel_event=cancel_event,
                per_request_timeout_seconds=_configured_chunk_timeout(),
            )
            transfer_finished_at = time.monotonic()
            with reconstruct_lock:
                if transfer_finished is None or transfer_finished_at > transfer_finished:
                    transfer_finished = transfer_finished_at
            _check_deadline(deadline, cancel_event)
            _reject_symlink_path(incoming_path, "adapter chunk incoming file")
            _replace_paths(incoming_path, chunk_path, "adapter transport chunk")
            if _path_regular_size(chunk_path, "adapter transport chunk") != wire_size:
                raise OSError("version-two adapter transport chunk checksum mismatch")

            # Publish verified wire data before the expensive reconstruction.
            # The atomic proof lets the peer fetch this chunk while this host
            # reconstructs its local model file.
            _write_json(
                wire_ready_path,
                {
                    "type": "wire-ready",
                    "file": name,
                    "offset": offset,
                    "data_offset": data_offset,
                    "raw_size": raw_size,
                    "wire_size": wire_size,
                    "wire_sha256": wire_sha256,
                    "raw_sha256": raw_sha256,
                },
            )

    def reconstruct(index: int, item: dict[str, object]) -> None:
        nonlocal reconstruct_worker_seconds, reconstruction_started, reconstruction_finished
        _check_cancelled(cancel_event)
        name = item["file"]
        if not isinstance(name, str):
            raise OSError("version-two adapter chunk has no file")
        chunk_path = chunk_root / name
        offset = int(item["offset"])
        data_offset = int(item["data_offset"])
        raw_size = int(item["raw_size"])
        wire_size = int(item["wire_size"])
        wire_sha256 = str(item["wire_sha256"])
        raw_sha256 = str(item["raw_sha256"])
        completed_path = _proof_path(chunk_path, _COMPLETED_SUFFIX)
        wire_ready_path = _proof_path(chunk_path, _WIRE_READY_SUFFIX)
        completed_valid, _ = _proof_identity_matches(
            completed_path, item, completed=True
        )
        if completed_valid:
            try:
                completed_valid = (
                    _path_regular_size(chunk_path, "adapter transport chunk") == wire_size
                    and _sha256(chunk_path, deadline, cancel_event) == wire_sha256
                )
            except (FileNotFoundError, OSError):
                completed_valid = False
        if completed_valid:
            _discard_proof(wire_ready_path)
            return
        if not _proof_identity_matches(wire_ready_path, item, completed=False)[0]:
            raise OSError("version-two adapter wire proof is missing")
        if (
            _path_regular_size(chunk_path, "adapter transport chunk") != wire_size
            or _sha256(chunk_path, deadline, cancel_event) != wire_sha256
        ):
            raise OSError("version-two adapter transport chunk checksum mismatch")
        encoded = _read_bytes_nofollow(chunk_path)
        reconstruction_started_at = time.monotonic()
        with reconstruct_lock:
            if reconstruction_started is None or reconstruction_started_at < reconstruction_started:
                reconstruction_started = reconstruction_started_at
        base_fd2 = base_fd1 = output_fd = -1
        try:
            base_fd2 = _open_nofollow(base2_model, os.O_RDONLY)
            base_fd1 = _open_nofollow(base1_model, os.O_RDONLY)
            output_fd = _open_nofollow(output_path, os.O_RDWR)
            base2_chunk = os.pread(base_fd2, raw_size, offset)
            base1_chunk = os.pread(base_fd1, raw_size, offset)
            if len(base2_chunk) != raw_size or len(base1_chunk) != raw_size:
                raise OSError("cached version-two base ended before transport chunk")
            reconstructed, correction_sha256 = _decode_linear_transport_chunk(
                base2_chunk,
                base1_chunk,
                encoded,
                offset=offset,
                data_offset=data_offset,
                raw_size=raw_size,
                codec=str(item["codec"]),
            )
            if correction_sha256 != raw_sha256:
                raise OSError("version-two adapter correction checksum mismatch")
            if len(reconstructed) != raw_size:
                raise OSError("version-two adapter chunk has the wrong size")
            _pwrite_all(output_fd, reconstructed, offset)
            output_sha256 = hashlib.sha256(reconstructed).hexdigest()
        finally:
            for file_descriptor in (output_fd, base_fd1, base_fd2):
                if file_descriptor >= 0:
                    os.close(file_descriptor)
        reconstruction_finished_at = time.monotonic()
        with reconstruct_lock:
            reconstruct_worker_seconds += reconstruction_finished_at - reconstruction_started_at
            if reconstruction_finished is None or reconstruction_finished_at > reconstruction_finished:
                reconstruction_finished = reconstruction_finished_at
        _check_deadline(deadline, cancel_event)
        _write_json(
            completed_path,
            {
                "file": name,
                "offset": offset,
                "data_offset": data_offset,
                "raw_size": raw_size,
                "wire_size": wire_size,
                "wire_sha256": wire_sha256,
                "raw_sha256": raw_sha256,
                "correction_sha256": correction_sha256,
                "output_sha256": output_sha256,
            },
        )
        # Keep the completed proof visible before removing the earlier proof.
        # This prevents a relay visibility gap between reconstruction and the
        # completed state.
        _discard_proof(wire_ready_path)

    download_streams = _configured_download_streams()
    reconstruct_executor = ThreadPoolExecutor(
        max_workers=min(len(chunks), _PREDICTIVE_RECONSTRUCT_WORKERS)
    )
    reconstruct_futures = []
    reconstruct_futures_lock = threading.Lock()

    def fetch_and_reconstruct(index: int, item: dict[str, object]) -> None:
        fetch_wire(index, item)
        future = reconstruct_executor.submit(reconstruct, index, item)
        with reconstruct_futures_lock:
            reconstruct_futures.append(future)

    download_executor = ThreadPoolExecutor(max_workers=min(len(chunks), download_streams))
    download_futures = [
        download_executor.submit(fetch_and_reconstruct, index, item)
        for index, item in enumerate(chunks)
    ]
    try:
        _wait_for_futures(download_executor, download_futures, deadline, cancel_event)
    except BaseException:
        reconstruct_executor.shutdown(wait=True, cancel_futures=True)
        raise
    with reconstruct_futures_lock:
        pending_reconstruction = list(reconstruct_futures)
    _wait_for_futures(
        reconstruct_executor,
        pending_reconstruction,
        deadline,
        cancel_event,
    )
    output_fd = _open_nofollow(output_path, os.O_RDONLY)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    verification_started = time.monotonic()
    try:
        if _sha256(output_path, deadline, cancel_event) != target_sha256:
            raise OSError("reconstructed version-two adapter checksum mismatch")
    finally:
        verification_finished = time.monotonic()
    if _path_regular_size(output_path, "adapter output") != target_size:
        raise OSError("reconstructed version-two adapter size mismatch")

    if _path_regular_size(config_path, "adapter configuration") is None or _sha256(config_path, deadline, cancel_event) != config_sha256:
        raise OSError("version-two adapter configuration checksum mismatch")
    _publish_relay_manifest(
        temporary,
        source_url,
        source_transport_manifest,
        transport,
        source_manifest["files"],
        deadline=deadline,
        cancel_event=cancel_event,
    )
    _write_text_atomic(temporary / _ADAPTER_DIGEST, f"{target_sha256}\n", encoding="ascii")
    _write_json(temporary / "manifest.json", source_manifest)
    _write_json(
        temporary / "transport.json",
        {
            "version": 2,
            "mode": "delta",
            "transport": _ADAPTER_LINEAR_TRANSPORT,
            "source_url": source_url,
            "base_url": base_url,
            "base_sha256": base_sha256,
            "base2_url": base2_url,
            "base2_sha256": base2_sha256,
            "target_sha256": target_sha256,
        },
    )
    with _adapter_stage_publish_lock:
        _check_deadline(deadline, cancel_event)
        _reject_symlink_path(temporary / _RELAYING_MARKER, "adapter relaying marker")
        _unlink_path(temporary / _RELAYING_MARKER, "adapter relaying marker")
        ready = temporary / "READY"
        _write_text_atomic(ready, "ready\n", encoding="utf-8")
        with os.fdopen(
            _open_nofollow(_adapter_reader_lock(destination), os.O_RDWR | os.O_CREAT),
            "a+",
        ) as reader_lock:
            _acquire_stage_lock(reader_lock, deadline, cancel_event)
            _check_deadline(deadline, cancel_event)
            if _path_stat(destination, "adapter destination") is not None:
                retired = destination.with_name(f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}")
                _replace_paths(destination, retired, "retired adapter")
            _replace_paths(temporary, destination, "adapter publish")
            if committed_event is not None:
                committed_event.set()
            fcntl.flock(reader_lock, fcntl.LOCK_UN)
    _cleanup_adapter_caches(
        destination.parent,
        destination,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    write_stage_status(
        {
            "status": "ready",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_LINEAR_TRANSPORT,
            "verified_files": source_manifest["files"],
            "bytes_done": total_wire_bytes,
            "bytes_total": total_wire_bytes,
            "updated_at": time.time(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    return destination


def _cleanup_delta_stage_artifacts(
    destination: Path,
    *,
    preserve_partial: bool = False,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Remove private delta staging paths after all transfer workers stop."""
    suffixes = (".parts", ".assembling")
    if not preserve_partial:
        suffixes = (".partial", *suffixes)
    candidate_names = [f"{destination.name}{suffix}" for suffix in suffixes]
    try:
        cache_fd = _open_directory_nofollow(
            destination.parent,
            create=False,
            label="adapter staging parent",
        )
    except FileNotFoundError:
        return True
    try:
        existing_entries: list[tuple[str, bool]] = []
        for name in candidate_names:
            try:
                entry_stat = os.stat(name, dir_fd=cache_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(entry_stat.st_mode):
                raise OSError(f"adapter staging path is a symbolic link: {name}")
            if not (stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISREG(entry_stat.st_mode)):
                raise OSError(f"adapter staging path has an unsafe file type: {name}")
            existing_entries.append((name, stat.S_ISDIR(entry_stat.st_mode)))
        if not existing_entries:
            return True
        lock_name = f".{destination.name}.readers.lock"
        try:
            reader_lock = os.fdopen(
                _open_nofollow(
                    lock_name,
                    os.O_RDWR | os.O_CREAT,
                    dir_fd=cache_fd,
                    label="adapter reader lock",
                ),
                "a+",
            )
        except FileNotFoundError:
            return True
        with reader_lock:
            try:
                _acquire_stage_lock(reader_lock, deadline, cancel_event)
            except TimeoutError:
                return False
            try:
                for name, is_directory in existing_entries:
                    if is_directory:
                        _remove_directory_entry(
                            cache_fd,
                            name,
                            "adapter staging path",
                            deadline=deadline,
                            cancel_event=cancel_event,
                        )
                    else:
                        os.unlink(name, dir_fd=cache_fd)
            finally:
                fcntl.flock(reader_lock, fcntl.LOCK_UN)
    finally:
        os.close(cache_fd)
    return True


def _preserve_adapter_stage_partial(exception: BaseException) -> bool:
    """Keep matching private stage data across retryable transport errors."""
    message = str(exception).lower()
    if any(
        marker in message
        for marker in (
            "manifest is missing",
            "transfer mode is unknown",
            "manifest is not visible",
        )
    ):
        return True
    if any(
        marker in message
        for marker in (
            "checksum",
            "digest",
            "identity",
            "protocol",
            "manifest",
            "symbolic link",
            "symlink",
            "unsafe",
            "path",
            "corrupt",
            "invalid",
            "malformed",
            "wrong size",
            "size mismatch",
            "did not honor byte range",
        )
    ):
        return False
    if isinstance(exception, TimeoutError):
        return True
    if isinstance(exception, HTTPError):
        return exception.code in {404, 429} or 500 <= exception.code <= 599
    if not isinstance(exception, OSError):
        return False
    return any(
        marker in message
        for marker in (
            "connection reset",
            "connection refused",
            "connection aborted",
            "timed out",
            "temporary failure",
            "adapter range ended early",
            "adapter file ended early",
            "incomplete adapter range",
            "incomplete adapter file",
            "remote end closed",
        )
    )


def _stage_chunked_xor_adapter(
    source_url: str,
    transport: dict[str, object],
    source_manifest: dict[str, object],
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    committed_event: threading.Event | None = None,
    status_metadata: dict[str, object] | None = None,
) -> Path:
    full_zstd = transport.get("transport") == _ADAPTER_FULL_TRANSPORT
    cancel_event = cancel_event or threading.Event()
    destination = adapter_cache_path(source_url)
    started = time.monotonic()
    _check_deadline(deadline, cancel_event)
    temporary = destination.with_name(f"{destination.name}.partial")
    _reject_symlink_path(destination, "adapter destination")
    _reject_symlink_path(temporary, "adapter partial directory")
    base_url = transport.get("base_url")
    base_sha256 = transport.get("base_sha256")
    target_sha256 = transport["target_sha256"]
    target_size = transport["target_size"]
    config_sha256 = transport["config_sha256"]
    if not full_zstd:
        assert isinstance(base_url, str)
        assert isinstance(base_sha256, str)
    assert isinstance(target_sha256, str)
    assert isinstance(target_size, int)
    assert isinstance(config_sha256, str)
    base_model = None
    if not full_zstd:
        base_model, _ = _verified_adapter_base(
            base_url,
            base_sha256,
            expected_config_sha256=(
                transport["base_config_sha256"]
                if isinstance(transport.get("base_config_sha256"), str)
                else None
            ),
            deadline=deadline,
            cancel_event=cancel_event,
        )
    chunks = transport["chunks"]
    assert isinstance(chunks, list)
    chunks = sorted(chunks, key=lambda item: int(item["offset"]))
    transport_identity = json.dumps(transport, sort_keys=True, separators=(",", ":"))
    identity_path = temporary / "transport-manifest.json"
    _reset_staging_directory_identity(
        temporary,
        identity_path.name,
        transport_identity,
        "adapter partial directory",
    )
    _ensure_directory(temporary, "adapter partial directory")
    _write_text_atomic(identity_path, transport_identity, encoding="utf-8")
    chunk_root = temporary / "transport-chunks"
    _ensure_directory(chunk_root, "adapter transport chunk directory")
    total_wire_bytes = sum(int(item["wire_size"]) for item in chunks)
    config_size = transport.get("config_size")
    if isinstance(config_size, int):
        total_wire_bytes += config_size
    progress_lock = threading.Lock()
    progress_bytes = 0
    timing_lock = threading.Lock()
    transfer_started = None
    transfer_finished = None
    reconstruction_started = None
    reconstruction_finished = None
    reconstruction_worker_seconds = 0.0
    verification_started = None
    verification_finished = None
    last_status = 0.0

    def write_stage_status(value: dict[str, object]) -> None:
        _write_stage_status(
            destination.parent,
            {
                **value,
                **_stage_timing_fields(
                    started,
                    transfer_started=transfer_started,
                    transfer_finished=transfer_finished,
                    reconstruction_started=reconstruction_started,
                    reconstruction_finished=reconstruction_finished,
                    reconstruction_worker_seconds=reconstruction_worker_seconds,
                    verification_started=verification_started,
                    verification_finished=verification_finished,
                ),
                **(status_metadata or {}),
            },
        )

    def report(lane: int, amount: int) -> None:
        nonlocal progress_bytes, last_status
        _check_deadline(deadline, cancel_event)
        with progress_lock:
            progress_bytes += amount
            now = time.monotonic()
            if now - last_status < _ADAPTER_STATUS_INTERVAL_S and progress_bytes < total_wire_bytes:
                return
            last_status = now
            write_stage_status(
                {
                    "status": "staging",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "transport": _ADAPTER_FULL_TRANSPORT if full_zstd else _ADAPTER_DELTA_TRANSPORT,
                    "bytes_done": progress_bytes,
                    "bytes_total": total_wire_bytes,
                    "lane": lane,
                    "updated_at": time.time(),
                    "elapsed_seconds": round(now - started, 3),
                },
            )

    write_stage_status(
        {
            "status": "staging",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_FULL_TRANSPORT if full_zstd else _ADAPTER_DELTA_TRANSPORT,
            "verified_files": source_manifest["files"],
            "bytes_done": progress_bytes,
            "bytes_total": total_wire_bytes,
            "updated_at": time.time(),
            "elapsed_seconds": 0.0,
        },
    )

    lane_urls, source_connection_count = _responsive_adapter_source_urls(
        source_url,
        deadline,
        cancel_event,
    )
    status_metadata = {
        **(status_metadata or {}),
        "source_connections": source_connection_count,
        "responsive_source_connections": len(lane_urls),
        "available_source_lanes": len(lane_urls),
    }
    download_streams = _configured_download_streams()
    persistent_width = min(download_streams, len(lane_urls))
    status_metadata["active_persistent_connections"] = (
        persistent_width if full_zstd else 0
    )
    output_path = temporary / _ADAPTER_FILES[1]
    config_path = temporary / _ADAPTER_FILES[0]
    persistent_pool = (
        _PersistentAdapterLanePool(lane_urls, width=persistent_width)
        if full_zstd
        else None
    )
    _reject_symlink_path(output_path, "adapter output")
    _reject_symlink_path(config_path, "adapter configuration file")
    allocations = [(output_path, target_size)] + [
        (chunk_root / Path(str(item["file"])).name, int(item["wire_size"]))
        for item in chunks
    ]
    if isinstance(config_size, int):
        allocations.append((config_path, config_size))
    _preflight_adapter_space(temporary, allocations)
    if _path_regular_size(output_path, "adapter output") != target_size:
        _preallocate(output_path, target_size)

    source_transport_manifest = transport.get("source_manifest")
    if not isinstance(source_transport_manifest, dict):
        raise OSError("adapter transport source manifest is unavailable for relay")
    if not full_zstd:
        config_transfer_started = time.monotonic()
        with timing_lock:
            if transfer_started is None or config_transfer_started < transfer_started:
                transfer_started = config_transfer_started
        config_urls = [
            f"{lane_url.rstrip('/')}/{_ADAPTER_FILES[0]}"
            for lane_url in lane_urls
        ]
        _download_persistent_file(
            config_urls,
            config_path,
            expected_size=config_size,
            expected_sha256=config_sha256,
            progress=report,
            deadline=deadline,
            cancel_event=cancel_event,
            per_request_timeout_seconds=_configured_chunk_timeout(),
        )
        config_transfer_finished = time.monotonic()
        with timing_lock:
            if transfer_finished is None or config_transfer_finished > transfer_finished:
                transfer_finished = config_transfer_finished
        if _sha256(config_path, deadline, cancel_event) != config_sha256:
            raise OSError("adapter configuration checksum mismatch")
        with os.fdopen(
            _open_nofollow(_adapter_reader_lock(destination), os.O_RDWR | os.O_CREAT),
            "a+",
        ):
            pass
        _publish_relay_manifest(
            temporary,
            source_url,
            source_transport_manifest,
            transport,
            source_manifest["files"],
            allow_partial=True,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        _write_marker(temporary / _RELAYING_MARKER, "relaying\n")

    def fetch(index: int, item: dict[str, object]) -> None:
        nonlocal transfer_started, transfer_finished
        nonlocal reconstruction_started, reconstruction_finished
        nonlocal reconstruction_worker_seconds
        _check_cancelled(cancel_event)
        name = item["file"]
        assert isinstance(name, str)
        relative = Path(name)
        chunk_path = chunk_root / relative
        _ensure_directory(chunk_path.parent, "adapter transport chunk directory")
        _reject_symlink_path(chunk_path, "adapter transport chunk")
        offset = int(item["offset"])
        raw_size = int(item["raw_size"])
        wire_size = int(item["wire_size"])
        wire_sha256 = str(item["wire_sha256"])
        raw_sha256 = str(item["raw_sha256"])
        completed_path = chunk_path.with_name(f"{chunk_path.name}.done")
        _reject_symlink_path(completed_path, "adapter completed proof")

        if persistent_pool is not None:
            chunk_lane = index % persistent_pool.slot_count
        else:
            chunk_lane = index % len(lane_urls)

        if _path_regular_size(completed_path, "adapter completed proof") is not None:
            try:
                proof = json.loads(
                    _read_text_nofollow(
                        completed_path,
                        encoding="utf-8",
                        max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                    )
                )
                proof_matches = (
                    isinstance(proof, dict)
                    and proof.get("file") == name
                    and proof.get("offset") == offset
                    and proof.get("raw_size") == raw_size
                    and proof.get("wire_size") == wire_size
                    and proof.get("wire_sha256") == wire_sha256
                    and proof.get("raw_sha256") == raw_sha256
                    and isinstance(proof.get("output_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", proof["output_sha256"]) is not None
                )
            except (OSError, ValueError):
                proof_matches = False
            if proof_matches:
                try:
                    proof_matches = (
                        _path_regular_size(chunk_path, "adapter transport chunk") == wire_size
                        and _sha256(chunk_path, deadline, cancel_event) == wire_sha256
                    )
                except (FileNotFoundError, OSError):
                    proof_matches = False
                if proof_matches:
                    # The final file checksum validates all resumed regions together.
                    report(chunk_lane, wire_size)
                    return
            _unlink_path(completed_path, "adapter completed proof")

        def lane_report(_lane: int, amount: int) -> None:
            report(chunk_lane, amount)

        if persistent_pool is not None:
            chunk_urls = persistent_pool.urls_for_lane(chunk_lane, name)
        else:
            chunk_urls = [f"{lane_url.rstrip('/')}/{name}" for lane_url in lane_urls]
            chunk_urls = chunk_urls[chunk_lane:] + chunk_urls[:chunk_lane]
        transfer_started_at = time.monotonic()
        with timing_lock:
            if transfer_started is None or transfer_started_at < transfer_started:
                transfer_started = transfer_started_at
        if persistent_pool is not None:
            persistent_pool.download(
                chunk_urls,
                chunk_lane,
                chunk_path,
                chunk_root_fd,
                expected_size=int(item["wire_size"]),
                expected_sha256=wire_sha256,
                progress=lane_report,
                deadline=deadline,
                cancel_event=cancel_event,
                per_request_timeout_seconds=_configured_chunk_timeout(),
            )
        else:
            _download_persistent_file(
                chunk_urls,
                chunk_path,
                expected_size=int(item["wire_size"]),
                expected_sha256=wire_sha256,
                progress=lane_report,
                deadline=deadline,
                cancel_event=cancel_event,
                per_request_timeout_seconds=_configured_chunk_timeout(),
            )
        transfer_finished_at = time.monotonic()
        with timing_lock:
            if transfer_finished is None or transfer_finished_at > transfer_finished:
                transfer_finished = transfer_finished_at

        _check_deadline(deadline, cancel_event)
        encoded = _read_bytes_nofollow(chunk_path)
        if len(encoded) != wire_size:
            raise OSError("adapter transport chunk size changed")
        _check_deadline(deadline, cancel_event)
        try:
            raw = _decode_transport_chunk(
                encoded,
                codec=str(transport["transport"]),
                raw_size=raw_size,
            )
        except OSError:
            if full_zstd:
                _unlink_path(chunk_path, "corrupt adapter transport chunk")
            raise
        _check_deadline(deadline, cancel_event)
        if _transport_raw_sha256(raw, codec=str(transport["transport"])) != raw_sha256:
            if full_zstd:
                _unlink_path(chunk_path, "corrupt adapter transport chunk")
            raise OSError("adapter transport decoded chunk checksum mismatch")
        _check_cancelled(cancel_event)
        reconstruction_started_at = time.monotonic()
        with timing_lock:
            if reconstruction_started is None or reconstruction_started_at < reconstruction_started:
                reconstruction_started = reconstruction_started_at
        output_fd = _open_nofollow(output_path, os.O_RDWR)
        try:
            if full_zstd:
                reconstructed = raw
            else:
                assert base_model is not None
                base_fd = _open_nofollow(base_model, os.O_RDONLY)
                try:
                    base_chunk = os.pread(base_fd, raw_size, offset)
                finally:
                    os.close(base_fd)
                if len(base_chunk) != len(raw):
                    raise OSError("cached base adapter ended before transport chunk")
                reconstructed = _xor_bytes(base_chunk, raw)
            if len(reconstructed) != raw_size:
                raise OSError("adapter transport reconstructed chunk has the wrong size")
            _pwrite_all(output_fd, reconstructed, offset)
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        reconstruction_finished_at = time.monotonic()
        with timing_lock:
            reconstruction_worker_seconds += reconstruction_finished_at - reconstruction_started_at
            if reconstruction_finished is None or reconstruction_finished_at > reconstruction_finished:
                reconstruction_finished = reconstruction_finished_at
        _check_deadline(deadline, cancel_event)
        _write_json(
            completed_path,
            {
                "file": name,
                "offset": offset,
                "raw_size": raw_size,
                "wire_size": wire_size,
                "wire_sha256": wire_sha256,
                "raw_sha256": raw_sha256,
                "output_sha256": hashlib.sha256(reconstructed).hexdigest(),
            },
        )

    chunk_root_fd = _open_directory_nofollow(
        chunk_root,
        create=False,
        label="adapter transport chunk directory",
    )
    try:
        if full_zstd:
            _run_bounded_futures(
                chunks,
                fetch,
                persistent_width,
                deadline,
                cancel_event,
            )
        else:
            executor = ThreadPoolExecutor(max_workers=min(len(chunks), download_streams))
            futures = [executor.submit(fetch, index, item) for index, item in enumerate(chunks)]
            _wait_for_futures(executor, futures, deadline, cancel_event)
    finally:
        if persistent_pool is not None:
            persistent_pool.close()
        os.close(chunk_root_fd)

    _reject_symlink_path(output_path, "adapter output")
    verification_started = time.monotonic()
    if _sha256(output_path, deadline, cancel_event) != target_sha256:
        raise OSError("reconstructed adapter checksum mismatch")
    if _path_regular_size(output_path, "adapter output") != target_size:
        raise OSError("reconstructed adapter size mismatch")
    verification_finished = time.monotonic()

    if full_zstd:
        if not isinstance(config_size, int) or config_size < 1:
            raise OSError("full adapter configuration size is invalid")
        config_transfer_started = time.monotonic()
        with timing_lock:
            if transfer_started is None or config_transfer_started < transfer_started:
                transfer_started = config_transfer_started
        config_urls = [
            f"{lane_url.rstrip('/')}/{_ADAPTER_FILES[0]}"
            for lane_url in lane_urls
        ]
        _download_persistent_file(
            config_urls,
            config_path,
            expected_size=config_size,
            expected_sha256=config_sha256,
            progress=report,
            deadline=deadline,
            cancel_event=cancel_event,
            per_request_timeout_seconds=_configured_chunk_timeout(),
        )
        config_transfer_finished = time.monotonic()
        with timing_lock:
            if transfer_finished is None or config_transfer_finished > transfer_finished:
                transfer_finished = config_transfer_finished
        if _sha256(config_path, deadline, cancel_event) != config_sha256:
            raise OSError("adapter configuration checksum mismatch")
    _publish_relay_manifest(
        temporary,
        source_url,
        source_transport_manifest,
        transport,
        source_manifest["files"],
        deadline=deadline,
        cancel_event=cancel_event,
    )
    _write_text_atomic(temporary / _ADAPTER_DIGEST, f"{target_sha256}\n", encoding="ascii")
    _write_json(temporary / "manifest.json", source_manifest)
    _write_json(
        temporary / "transport.json",
        {
            "mode": "full" if full_zstd else _ADAPTER_DELTA_TRANSPORT,
            "transport": _ADAPTER_FULL_TRANSPORT if full_zstd else _ADAPTER_DELTA_TRANSPORT,
            "source_url": source_url,
            "target_sha256": target_sha256,
            **({} if full_zstd else {"base_url": base_url, "base_sha256": base_sha256}),
        },
    )
    with _adapter_stage_publish_lock:
        ready = temporary / "READY"
        _check_deadline(deadline, cancel_event)
        if not full_zstd:
            _reject_symlink_path(
                temporary / _RELAYING_MARKER,
                "adapter relaying marker",
            )
            _unlink_path(temporary / _RELAYING_MARKER, "adapter relaying marker")
        _write_text_atomic(ready, "ready\n", encoding="utf-8")
        with os.fdopen(
            _open_nofollow(_adapter_reader_lock(destination), os.O_RDWR | os.O_CREAT),
            "a+",
        ) as reader_lock:
            _acquire_stage_lock(reader_lock, deadline, cancel_event)
            _check_deadline(deadline, cancel_event)
            if _path_stat(destination, "adapter destination") is not None:
                retired = destination.with_name(
                    f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}"
                )
                _replace_paths(destination, retired, "retired adapter")
            _replace_paths(temporary, destination, "adapter publish")
            if committed_event is not None:
                committed_event.set()
            fcntl.flock(reader_lock, fcntl.LOCK_UN)
    _cleanup_adapter_caches(
        destination.parent,
        destination,
        deadline=deadline,
        cancel_event=cancel_event,
    )
    write_stage_status(
        {
            "status": "ready",
            "source_url": source_url,
            "path": destination.as_posix(),
            "transport": _ADAPTER_FULL_TRANSPORT if full_zstd else _ADAPTER_DELTA_TRANSPORT,
            "verified_files": source_manifest["files"],
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
    parallel_ranges: bool = True,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    per_request_timeout_seconds: float | None = None,
    retry_http_statuses: set[int] | None = None,
) -> None:
    destination = Path(destination)
    _reject_symlink_path(destination, "adapter download destination")
    parent_fd = _open_directory_nofollow(
        destination.parent,
        create=False,
        label="adapter download parent",
    )
    try:
        return _download_with_parent(
            source_url,
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            progress=progress,
            source_urls=source_urls,
            parallel_ranges=parallel_ranges,
            deadline=deadline,
            cancel_event=cancel_event,
            per_request_timeout_seconds=per_request_timeout_seconds,
            retry_http_statuses=retry_http_statuses,
            parent_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)


def _download_with_parent(
    source_url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    progress=None,
    source_urls: list[str] | None = None,
    parallel_ranges: bool = True,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    per_request_timeout_seconds: float | None = None,
    retry_http_statuses: set[int] | None = None,
    stream_cap: int | None = None,
    typed_transport: bool = False,
    parent_fd: int,
) -> None:
    _check_deadline(deadline, cancel_event)
    if per_request_timeout_seconds is not None and (
        not _is_finite_number(per_request_timeout_seconds)
        or per_request_timeout_seconds <= 0
    ):
        raise OSError("adapter request timeout must be positive")
    explicit_source_urls = source_urls is not None
    source_urls = _adapter_source_urls(source_url) if source_urls is None else source_urls
    if not source_urls:
        raise OSError("adapter download has no source lane")
    total = (
        expected_size
        if expected_size is not None
        else _content_length(
            source_urls[0],
            deadline=deadline,
            cancel_event=cancel_event,
        )
    )
    _check_deadline(deadline, cancel_event)
    destination_stat = _stat_regular_entry(
        parent_fd,
        destination.name,
        "adapter download destination",
    )
    if destination_stat is not None and destination_stat.st_size == total:
        if expected_sha256 is None or _sha256(
            destination,
            deadline,
            cancel_event,
            parent_fd=parent_fd,
        ) == expected_sha256:
            return
        os.unlink(destination.name, dir_fd=parent_fd)

    # Full-checkpoint tunnels are not adapter sources. Reusing their 128-port
    # list here created 128 adapter lanes on every replica and collapsed the
    # link. Adapter fan-out is opt-in and has its own setting.
    adapter_ports = explicit_source_urls or len(source_urls) > 1 or source_urls[0] != source_url

    configured_streams = _configured_download_streams() if parallel_ranges else 1
    if stream_cap is not None and parallel_ranges:
        streams = min(len(source_urls), stream_cap)
    elif adapter_ports and parallel_ranges:
        streams = min(len(source_urls), configured_streams)
    elif not adapter_ports and parallel_ranges:
        streams = configured_streams
    else:
        streams = 1
    if total <= 0 or total < _PARALLEL_DOWNLOAD_THRESHOLD or streams == 1:
        # Preserve the old single-file resume path for small metadata files and
        # for an explicitly disabled parallel transfer.
        attempt_limit = (
            max(_ADAPTER_LANE_RETRIES, 64)
            if retry_http_statuses
            else _ADAPTER_LANE_RETRIES
        )
        if deadline is not None:
            attempt_limit = max(attempt_limit, len(source_urls))
        if retry_http_statuses and deadline is not None:
            attempt_limit = max(
                attempt_limit,
                math.ceil(max(0.0, deadline - time.monotonic()))
                + len(source_urls)
                + 1,
            )
        for attempt in range(attempt_limit):
            _check_deadline(deadline, cancel_event)
            attempt_deadline = deadline
            if per_request_timeout_seconds is not None:
                active_timeout = _chunk_request_timeout(
                    deadline,
                    attempt,
                    per_request_timeout_seconds,
                    cancel_event,
                )
                attempt_deadline = time.monotonic() + active_timeout
                if deadline is not None:
                    attempt_deadline = min(attempt_deadline, deadline)
            try:
                destination_stat = _stat_regular_entry(
                    parent_fd,
                    destination.name,
                    "adapter download destination",
                )
                offset = destination_stat.st_size if destination_stat is not None else 0
                if offset > total:
                    os.unlink(destination.name, dir_fd=parent_fd)
                    offset = 0
                elif destination_stat is not None and offset == total:
                    if expected_sha256 is None or _sha256(
                        destination,
                        deadline,
                        cancel_event,
                        parent_fd=parent_fd,
                    ) == expected_sha256:
                        return
                    os.unlink(destination.name, dir_fd=parent_fd)
                    offset = 0
                request = urllib.request.Request(
                    source_urls[attempt % len(source_urls)],
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                )
                request_timeout = (
                    _request_timeout(deadline, cancel_event)
                    if per_request_timeout_seconds is None
                    else _request_timeout(
                        attempt_deadline,
                        cancel_event,
                        maximum=_ADAPTER_LANE_TIMEOUT_S,
                    )
                )
                response_open = (
                    _adapter_urlopen(request, request_timeout)
                    if typed_transport
                    else urllib.request.urlopen(request, timeout=request_timeout)
                )
                with response_open as response:
                    if offset and response.status != 206:
                        if typed_transport:
                            raise _PersistentProtocolError(
                                "adapter server did not honor byte range"
                            )
                        os.unlink(destination.name, dir_fd=parent_fd)
                        raise OSError("adapter server did not honor byte range")
                    with os.fdopen(
                        _open_nofollow(
                            destination.name,
                            os.O_WRONLY
                            | os.O_CREAT
                            | (os.O_APPEND if offset else os.O_TRUNC),
                            dir_fd=parent_fd,
                            label="adapter download destination",
                        ),
                        "ab" if offset else "wb",
                    ) as output:
                        read_available = getattr(response, "read1", None)
                        if read_available is None:
                            read_available = response.read
                        if typed_transport:
                            chunks = _adapter_response_chunks(
                                response,
                                total - offset,
                                attempt_deadline,
                                cancel_event,
                            )
                            for chunk in chunks:
                                output.write(chunk)
                                offset += len(chunk)
                                if progress is not None:
                                    progress(0, len(chunk))
                        else:
                            while offset < total:
                                _check_deadline(attempt_deadline, cancel_event)
                                chunk = read_available(
                                    min(16 * 1024 * 1024, total - offset)
                                )
                                if not chunk:
                                    raise OSError(f"adapter file ended early: {destination.name}")
                                output.write(chunk)
                                offset += len(chunk)
                                if progress is not None:
                                    progress(0, len(chunk))
                                if offset < total:
                                    _check_deadline(attempt_deadline, cancel_event)
                destination_stat = _stat_regular_entry(
                    parent_fd,
                    destination.name,
                    "adapter download destination",
                )
                if destination_stat is None or destination_stat.st_size != total:
                    raise OSError(f"incomplete adapter file: {destination.name}")
                if expected_sha256 is not None and _sha256(
                    destination,
                    deadline,
                    cancel_event,
                    parent_fd=parent_fd,
                ) != expected_sha256:
                    # A complete but bad file must not be resumed from EOF on
                    # the next bounded attempt.
                    os.unlink(destination.name, dir_fd=parent_fd)
                    raise OSError(f"adapter checksum mismatch: {destination.name}")
                return
            except _PersistentTransportError:
                if not typed_transport:
                    raise
                _check_deadline(deadline, cancel_event)
                if attempt + 1 == attempt_limit:
                    raise
                sleep_for = (
                    lane_rotation_delay_seconds
                    if len(source_urls) > 1
                    else min(2**attempt, 10)
                )
                _check_cancelled(cancel_event)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter delta staging exceeded its time limit")
                    sleep_for = min(sleep_for, remaining)
                if cancel_event is None:
                    time.sleep(sleep_for)
                else:
                    cancel_event.wait(sleep_for)
            except HTTPError as exc:
                if retry_http_statuses is None or exc.code not in retry_http_statuses:
                    raise
                _check_deadline(deadline, cancel_event)
                if attempt + 1 == attempt_limit:
                    raise
                sleep_for = (
                    1.0
                    if exc.code in {404, 503}
                    else (
                        lane_rotation_delay_seconds
                        if len(source_urls) > 1
                        else min(2**attempt, 10)
                    )
                )
                _check_cancelled(cancel_event)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("adapter delta staging exceeded its time limit")
                    sleep_for = min(sleep_for, remaining)
                if cancel_event is None:
                    time.sleep(sleep_for)
                else:
                    cancel_event.wait(sleep_for)
            except OSError:
                if typed_transport:
                    raise
                # Retry network timeouts; keep the absolute stage limit.
                _check_deadline(deadline, cancel_event)
                if attempt + 1 == attempt_limit:
                    raise
                sleep_for = (
                    lane_rotation_delay_seconds
                    if len(source_urls) > 1
                    else min(2**attempt, 10)
                )
                _check_cancelled(cancel_event)
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
    parts_name = parts_dir.name
    _check_cancelled(cancel_event)
    parts_fd = _ensure_directory_entry(
        parent_fd,
        parts_name,
        "adapter parts directory",
    )
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
    try:
        try:
            existing_identity = json.loads(
                _read_text_nofollow(
                    "manifest.json",
                    encoding="utf-8",
                    dir_fd=parts_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except (FileNotFoundError, OSError, ValueError):
            existing_identity = None
        if existing_identity != identity:
            # Parts are private to this transfer. Remove only mismatched private
            # parts; never remove a published adapter directory here.
            os.close(parts_fd)
            parts_fd = -1
            _remove_directory_entry(
                parent_fd,
                parts_name,
                "adapter parts directory",
                deadline=deadline,
                cancel_event=cancel_event,
            )
            parts_fd = _ensure_directory_entry(
                parent_fd,
                parts_name,
                "adapter parts directory",
            )
            _write_text_entry_atomic(
                parts_fd,
                "manifest.json",
                json.dumps(identity, separators=(",", ":")),
                encoding="utf-8",
            )

        executor = ThreadPoolExecutor(max_workers=len(ranges))
        cancel_event = cancel_event or threading.Event()
        futures = [
            executor.submit(
                _download_range,
                source_urls,
                index,
                parts_dir / f"part-{index:03d}",
                parts_fd,
                start,
                end,
                progress,
                deadline,
                cancel_event,
                typed_transport=typed_transport,
                retry_http_statuses=retry_http_statuses,
                per_request_timeout_seconds=per_request_timeout_seconds,
            )
            for index, start, end in ranges
        ]
        _wait_for_futures(executor, futures, deadline, cancel_event)

        assembling = destination.with_name(f"{destination.name}.assembling")
        assembling_fd = _open_nofollow(
            assembling.name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            dir_fd=parent_fd,
            label="adapter assembling file",
        )
        try:
            with os.fdopen(assembling_fd, "wb") as output:
                assembling_fd = -1
                for index, _, _ in ranges:
                    part_name = f"part-{index:03d}"
                    with os.fdopen(
                        _open_nofollow(
                            part_name,
                            os.O_RDONLY,
                            dir_fd=parts_fd,
                            label="adapter range file",
                        ),
                        "rb",
                    ) as part:
                        while True:
                            _check_deadline(deadline, cancel_event)
                            chunk_data = part.read(16 * 1024 * 1024)
                            if not chunk_data:
                                break
                            output.write(chunk_data)
                            _check_deadline(deadline, cancel_event)
        finally:
            if assembling_fd >= 0:
                os.close(assembling_fd)
        assembling_stat = _stat_regular_entry(
            parent_fd,
            assembling.name,
            "adapter assembling file",
        )
        if assembling_stat is None or assembling_stat.st_size != total:
            raise OSError(f"assembled adapter file has the wrong size: {destination.name}")
        if expected_sha256 is not None and _sha256(
            assembling,
            deadline,
            cancel_event,
            parent_fd=parent_fd,
        ) != expected_sha256:
            # Part sizes alone cannot detect a bad byte from a reset connection.
            # Drop the private parts after a whole-file mismatch so the next
            # stage attempt does not repeatedly assemble the same bad payload.
            os.close(parts_fd)
            parts_fd = -1
            _remove_directory_entry(
                parent_fd,
                parts_name,
                "adapter parts directory",
                deadline=deadline,
                cancel_event=cancel_event,
            )
            os.unlink(assembling.name, dir_fd=parent_fd)
            raise OSError(f"adapter checksum mismatch: {destination.name}")
        _check_deadline(deadline, cancel_event)
        _stat_regular_entry(parent_fd, destination.name, "adapter download destination")
        os.replace(
            assembling.name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.close(parts_fd)
        parts_fd = -1
        _remove_directory_entry(
            parent_fd,
            parts_name,
            "adapter parts directory",
            deadline=deadline,
            cancel_event=cancel_event,
        )
    finally:
        if parts_fd >= 0:
            os.close(parts_fd)


def _download_checkpoint_entry(
    source_url: str,
    destination: Path,
    *,
    parent_fd: int,
    download_fn=None,
) -> None:
    """Download a checkpoint entry below its held cache directory."""
    if download_fn is not None:
        download_fn(source_url, destination)
        return
    _download_with_parent(
        source_url,
        destination,
        parent_fd=parent_fd,
    )


def stage_lora_adapter_files(
    source_url: str,
    require_delta: bool = False,
    *,
    stage_timeout_seconds: float | None = None,
    policy_step: int | None = None,
    cancel_event: threading.Event | None = None,
    committed_event: threading.Event | None = None,
) -> Path:
    """Download one remote adapter into the host-shared cache.

    The API server can call this before it pauses generation. Worker-side
    loading calls the same helper as a fail-safe, but returns immediately when
    the prefetch already wrote ``READY``.
    """
    _check_cancelled(cancel_event)
    started = time.monotonic()
    destination = adapter_cache_path(source_url)
    configured_delta_timeout = _delta_stage_timeout_s() if require_delta else None
    status_metadata: dict[str, object] = {}
    stage_limit: float | None = None
    if require_delta:
        source_step = _adapter_step_from_source_url(source_url)
        if policy_step is not None and policy_step != source_step:
            raise OSError("requested adapter policy step differs from its source URL")
        policy_step = source_step
        stage_limit = adapter_stage_limit_seconds(policy_step)
        configured_delta_timeout = min(configured_delta_timeout, stage_limit)
        status_metadata = {
            "policy_step": policy_step,
            "stage_timeout_seconds": stage_limit,
            "acceptance_target_seconds": 300.0,
        }
    if stage_timeout_seconds is not None:
        if (
            not require_delta
            or isinstance(stage_timeout_seconds, bool)
            or not isinstance(stage_timeout_seconds, (int, float))
            or not _is_finite_number(stage_timeout_seconds)
            or stage_timeout_seconds <= 0
            or configured_delta_timeout is None
            or stage_timeout_seconds > configured_delta_timeout
            or stage_timeout_seconds > stage_limit
        ):
            raise OSError("adapter delta stage timeout is invalid")
        configured_delta_timeout = float(stage_timeout_seconds)
    delta_deadline = (
        started + configured_delta_timeout
        if configured_delta_timeout is not None
        else None
    )
    operation_deadline = delta_deadline or (started + full_adapter_stage_timeout_s)
    status_metadata = {
        **status_metadata,
        "operation_timeout_seconds": round(operation_deadline - started, 3),
    }
    _ensure_directory(destination.parent, "adapter cache directory")
    reconcile_retired_adapter(
        destination,
        source_url,
        deadline=operation_deadline,
        cancel_event=cancel_event,
    )
    _reject_symlink_path(destination, "adapter destination")
    _reject_symlink_path(
        destination.with_name(f"{destination.name}.partial"),
        "adapter partial directory",
    )
    # The cache is shared by all replicas on this host. One global writer is
    # intentional: per-URL writers caused old policies to queue behind one
    # another and made it possible to publish a stale policy after a newer one.
    host_lock_path = destination.parent / "adapter-staging.lock"
    transfer_started = None
    transfer_finished = None
    verification_started = None
    verification_finished = None
    cache_hit = False

    def write_stage_status(value: dict[str, object]) -> None:
        _write_stage_status(
            destination.parent,
            {
                **value,
                **_stage_timing_fields(
                    started,
                    transfer_started=transfer_started,
                    transfer_finished=transfer_finished,
                    verification_started=verification_started,
                    verification_finished=verification_finished,
                    cache_hit=cache_hit,
                ),
                **status_metadata,
            },
        )

    with os.fdopen(
        _open_nofollow(host_lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC),
        "w",
    ) as host_lock:
        _acquire_stage_lock(host_lock, operation_deadline, cancel_event)
        transport: dict[str, object] | None = None
        try:
            if require_delta:
                transport = _read_delta_manifest_with_visibility_wait(
                    source_url,
                    deadline=operation_deadline,
                    cancel_event=cancel_event,
                )
            else:
                transport = _read_chunked_xor_manifest(
                    source_url,
                    deadline=operation_deadline,
                    cancel_event=cancel_event,
                )
            if require_delta:
                source_manifest = transport.get("source_manifest") if isinstance(transport, dict) else None
                if (
                    not isinstance(source_manifest, dict)
                    or source_manifest.get("target_step") != policy_step
                ):
                    raise OSError("validated adapter manifest does not identify the requested policy step")
            published_manifest = (
                transport.get("source_manifest")
                if isinstance(transport, dict)
                and isinstance(transport.get("source_manifest"), dict)
                else None
            )
            manifest = _source_manifest(
                source_url,
                deadline=operation_deadline,
                cancel_event=cancel_event,
                published_manifest=published_manifest,
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
                _reject_symlink_path(destination / "READY", "adapter ready marker")
                if _path_regular_size(destination / "READY", "adapter ready marker") is not None:
                    try:
                        _reject_symlink_path(destination / "transport.json", "adapter transport record")
                        proof = json.loads(
                            _read_text_nofollow(
                                destination / "transport.json",
                                encoding="utf-8",
                                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                            )
                        )
                    except (FileNotFoundError, OSError, ValueError):
                        proof = None
                    if (
                        isinstance(proof, dict)
                        and (
                            (
                                transport.get("transport") == _ADAPTER_DELTA_TRANSPORT
                                and proof.get("mode") == _ADAPTER_DELTA_TRANSPORT
                            )
                            or (
                                transport.get("transport") == _ADAPTER_LINEAR_TRANSPORT
                                and proof.get("mode") == "delta"
                                and proof.get("transport") == _ADAPTER_LINEAR_TRANSPORT
                                and proof.get("base2_url") == transport.get("base2_url")
                                and proof.get("base2_sha256") == transport.get("base2_sha256")
                            )
                        )
                        and proof.get("source_url") == source_url
                        and proof.get("target_sha256") == transport["target_sha256"]
                        and (
                            transport.get("transport") == _ADAPTER_DELTA_TRANSPORT
                            or (
                                proof.get("base_url") == transport.get("base_url")
                                and proof.get("base_sha256") == transport.get("base_sha256")
                            )
                        )
                        and _relay_manifest_matches(destination, source_url, transport)
                        and _path_regular_size(
                            destination / _ADAPTER_FILES[1],
                            "adapter model",
                        ) is not None
                        and _sha256(
                            destination / _ADAPTER_FILES[1],
                            operation_deadline,
                            cancel_event,
                        )
                        == transport["target_sha256"]
                        and _path_regular_size(
                            destination / _ADAPTER_FILES[0],
                            "adapter configuration",
                        ) is not None
                        and _sha256(
                            destination / _ADAPTER_FILES[0],
                            operation_deadline,
                            cancel_event,
                        )
                        == transport["config_sha256"]
                    ):
                        cache_hit = True
                        write_stage_status(
                            {
                                "status": "ready",
                                "source_url": source_url,
                                "path": destination.as_posix(),
                                "transport": transport["transport"],
                                "verified_files": files,
                                "bytes_done": 0,
                                "bytes_total": 0,
                                "updated_at": time.time(),
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                            },
                        )
                        _cleanup_adapter_caches(
                            destination.parent,
                            destination,
                            deadline=operation_deadline,
                            cancel_event=cancel_event,
                        )
                        if committed_event is not None:
                            with _adapter_stage_publish_lock:
                                committed_event.set()
                        return destination
                _check_deadline(operation_deadline, cancel_event)
                if transport.get("transport") == _ADAPTER_LINEAR_TRANSPORT:
                    return _stage_chunked_linear_adapter(
                        source_url,
                        transport,
                        manifest,
                        deadline=operation_deadline,
                        cancel_event=cancel_event,
                        committed_event=committed_event,
                        status_metadata=status_metadata,
                    )
                if transport.get("transport") == _ADAPTER_DELTA_TRANSPORT:
                    return _stage_chunked_xor_adapter(
                        source_url,
                        transport,
                        manifest,
                        deadline=operation_deadline,
                        cancel_event=cancel_event,
                        committed_event=committed_event,
                        status_metadata=status_metadata,
                    )
                raise OSError("adapter delta transport is unsupported")
            _reject_symlink_path(destination / "READY", "adapter ready marker")
            if _path_regular_size(destination / "READY", "adapter ready marker") is not None:
                try:
                    _reject_symlink_path(destination / "manifest.json", "adapter manifest")
                    local_manifest = json.loads(
                        _read_text_nofollow(
                            destination / "manifest.json",
                            encoding="utf-8",
                            max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                        )
                    )
                except (FileNotFoundError, OSError, ValueError):
                    local_manifest = None
                if (
                    local_manifest == manifest
                    and isinstance(transport, dict)
                    and _relay_manifest_matches(destination, source_url, transport)
                    and _full_cache_matches_manifest(destination, files)
                ):
                    cache_hit = True
                    write_stage_status(
                        {
                            "status": "ready",
                            "source_url": source_url,
                            "path": destination.as_posix(),
                            "bytes_done": 0,
                            "bytes_total": 0,
                            "updated_at": time.time(),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "verified_files": files,
                        },
                    )
                    _cleanup_adapter_caches(
                        destination.parent,
                        destination,
                        deadline=operation_deadline,
                        cancel_event=cancel_event,
                    )
                    if committed_event is not None:
                        with _adapter_stage_publish_lock:
                            committed_event.set()
                    return destination

            if (
                isinstance(transport, dict)
                and transport.get("mode") == "full"
                and transport.get("transport") == _ADAPTER_FULL_TRANSPORT
            ):
                return _stage_chunked_xor_adapter(
                    source_url,
                    transport,
                    manifest,
                    deadline=operation_deadline,
                    cancel_event=cancel_event,
                    committed_event=committed_event,
                    status_metadata=status_metadata,
                )

            if (
                isinstance(transport, dict)
                and transport.get("mode") == "full"
                and _reuse_full_adapter_cache(
                    source_url,
                    destination,
                    manifest,
                    transport,
                    cancel_event=cancel_event,
                    committed_event=committed_event,
                    deadline=operation_deadline,
                )
            ):
                    total_bytes = sum(int(item["size"]) for item in files.values())
                    _cleanup_adapter_caches(
                        destination.parent,
                        destination,
                        deadline=operation_deadline,
                        cancel_event=cancel_event,
                    )
                    cache_hit = True
                    write_stage_status(
                        {
                            "status": "ready",
                            "source_url": source_url,
                            "path": destination.as_posix(),
                            "transport": "full",
                            "bytes_done": 0,
                            "bytes_total": 0,
                            "reuse": "local-cache",
                            "verified_files": files,
                            "updated_at": time.time(),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                    )
                    return destination

            total_bytes = sum(int(item["size"]) for item in files.values())
            _preflight_full_adapter_download_space(
                destination.with_name(f"{destination.name}.partial"),
                files,
            )
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
                    write_stage_status(
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

            write_stage_status(
                {
                    "status": "staging",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "bytes_done": 0,
                    "bytes_total": total_bytes,
                    "verified_files": files,
                    "updated_at": time.time(),
                    "elapsed_seconds": 0.0,
                },
            )
            temporary = destination.with_name(f"{destination.name}.partial")
            _check_cancelled(cancel_event)
            _reject_symlink_path(temporary, "adapter partial directory")
            _ensure_directory(temporary, "adapter partial directory")
            _write_json(temporary / "manifest.json", manifest)
            temporary_fd = _open_directory_nofollow(
                temporary,
                create=False,
                label="adapter partial directory",
            )
            transfer_started = time.monotonic()
            try:
                for filename in _ADAPTER_FILES:
                    item = files[filename]
                    assert isinstance(item, dict)
                    expected_sha256 = item.get("sha256")
                    destination_path = temporary / filename
                    _reject_symlink_path(destination_path, "adapter download destination")
                    _download_with_parent(
                        f"{source_url.rstrip('/')}/{filename}",
                        destination_path,
                        expected_size=int(item["size"]),
                        expected_sha256=expected_sha256 if isinstance(expected_sha256, str) else None,
                        progress=report,
                        cancel_event=cancel_event,
                        deadline=operation_deadline,
                        per_request_timeout_seconds=_configured_chunk_timeout(),
                        stream_cap=full_download_streams,
                        parent_fd=temporary_fd,
                        retry_http_statuses={404, 503},
                        typed_transport=True,
                    )
                    if expected_sha256 is None:
                        item["sha256"] = _sha256(
                            destination_path,
                            cancel_event=cancel_event,
                        )
                    completed_bytes = 0
                    for name in _ADAPTER_FILES:
                        path = temporary / name
                        _reject_symlink_path(path, "adapter file")
                        if _path_regular_size(path, "adapter file") == int(files[name]["size"]):
                            completed_bytes += int(files[name]["size"])
                    progress_bytes = max(progress_bytes, completed_bytes)
            finally:
                os.close(temporary_fd)
            transfer_finished = time.monotonic()
            verification_started = transfer_finished
            _write_json(temporary / "manifest.json", manifest)
            source_transport_manifest = transport.get("source_manifest") if isinstance(transport, dict) else None
            if not isinstance(source_transport_manifest, dict):
                raise OSError("adapter source manifest is unavailable for relay")
            _publish_relay_manifest(
                temporary,
                source_url,
                source_transport_manifest,
                transport,
                files,
                deadline=operation_deadline,
                cancel_event=cancel_event,
            )
            model_digest = files[_ADAPTER_FILES[1]].get("sha256")
            if not isinstance(model_digest, str):
                raise OSError("adapter source manifest has no model checksum")
            _write_text_atomic(
                temporary / _ADAPTER_DIGEST,
                f"{model_digest}\n",
                encoding="ascii",
            )
            verification_finished = time.monotonic()
            with _adapter_stage_publish_lock:
                ready = temporary / "READY"
                _check_cancelled(cancel_event)
                _write_text_atomic(ready, "ready\n", encoding="utf-8")

                # Keep the exclusive lease through both renames. This removes the
                # short missing-path window between retiring the old directory and
                # publishing the new one.
                with os.fdopen(
                    _open_nofollow(_adapter_reader_lock(destination), os.O_RDWR | os.O_CREAT),
                    "a+",
                ) as reader_lock:
                    _acquire_stage_lock(reader_lock, operation_deadline, cancel_event)
                    _check_cancelled(cancel_event)
                    if _path_stat(destination, "adapter destination") is not None:
                        retired = destination.with_name(
                            f"{destination.name}.retired-{time.time_ns()}-{os.getpid()}"
                        )
                        _replace_paths(destination, retired, "retired adapter")
                    _replace_paths(temporary, destination, "adapter publish")
                    if committed_event is not None:
                        committed_event.set()
                    fcntl.flock(reader_lock, fcntl.LOCK_UN)
            _cleanup_adapter_caches(
                destination.parent,
                destination,
                deadline=operation_deadline,
                cancel_event=cancel_event,
            )
            write_stage_status(
                {
                    "status": "ready",
                    "source_url": source_url,
                    "path": destination.as_posix(),
                    "bytes_done": total_bytes,
                    "bytes_total": total_bytes,
                    "verified_files": files,
                    "updated_at": time.time(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            return destination
        except BaseException as exc:
            if require_delta or (
                isinstance(transport, dict) and transport.get("mode") in {"delta", "full"}
            ):
                cleanup_failure: BaseException | None = None
                try:
                    cleaned = _cleanup_delta_stage_artifacts(
                        destination,
                        preserve_partial=_preserve_adapter_stage_partial(exc),
                        deadline=time.monotonic() + _ADAPTER_CLEANUP_TIMEOUT_S,
                        cancel_event=threading.Event(),
                    )
                    if not cleaned:
                        cleanup_failure = OSError("adapter staging cleanup timed out")
                except BaseException as cleanup_exc:
                    cleanup_failure = cleanup_exc
                if cleanup_failure is not None:
                    write_stage_status(
                        {
                            "status": "failed",
                            "source_url": source_url,
                            "path": destination.as_posix(),
                            "error": (
                                f"{type(exc).__name__}: {exc}; cleanup failed: "
                                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                            ),
                            "updated_at": time.time(),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                    )
                    raise OSError(
                        f"adapter staging cleanup failed: {cleanup_failure}"
                    ) from exc
            write_stage_status(
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


def _full_checkpoint_cache_is_valid(
    cache_parent_fd: int,
    directory_name: str,
    source_url: str,
) -> bool:
    """Check the complete immutable contract before reusing a checkpoint."""
    directory_fd = _open_directory_entry_nofollow(
        cache_parent_fd,
        directory_name,
        "full checkpoint cache entry",
    )
    try:
        if (
            _read_bytes_nofollow(
                "READY",
                dir_fd=directory_fd,
                max_bytes=_ADAPTER_MARKER_MAX_BYTES,
            )
            != b"ready\n"
        ):
            return False
        try:
            cache_manifest = json.loads(
                _read_text_nofollow(
                    _FULL_CHECKPOINT_CACHE_MANIFEST,
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
            index = json.loads(
                _read_text_nofollow(
                    "model.safetensors.index.json",
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except FileNotFoundError:
            return False
        if not isinstance(cache_manifest, dict) or cache_manifest.get("version") != 1:
            return False
        if cache_manifest.get("source_url") != source_url:
            return False
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            return False
        raw_shard_names = list(index["weight_map"].values())
        if not raw_shard_names or any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in raw_shard_names
        ):
            return False
        shard_names = set(raw_shard_names)
        files = cache_manifest.get("files")
        if not isinstance(files, dict) or set(files) != shard_names:
            return False
        index_sha256 = cache_manifest.get("index_sha256")
        if (
            not isinstance(index_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", index_sha256) is None
            or _sha256_entry(
                directory_fd,
                "model.safetensors.index.json",
                "full checkpoint index",
            )
            != index_sha256
        ):
            return False
        for name in sorted(shard_names):
            spec = files.get(name)
            if not isinstance(spec, dict):
                return False
            expected_size = spec.get("size")
            expected_sha256 = spec.get("sha256")
            if (
                type(expected_size) is not int
                or expected_size <= 0
                or not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            ):
                return False
            shard_fd = _open_nofollow(
                name,
                os.O_RDONLY,
                dir_fd=directory_fd,
                label="full checkpoint shard",
            )
            try:
                if os.fstat(shard_fd).st_size != expected_size:
                    return False
            finally:
                os.close(shard_fd)
            if _sha256_entry(directory_fd, name, "full checkpoint shard") != expected_sha256:
                return False
        return True
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        return False
    finally:
        os.close(directory_fd)


def prune_full_checkpoint_cache(
    cache_root: Path,
    keep: Path,
    *,
    _cache_fd: int | None = None,
    deadline: float | None = None,
) -> list[Path]:
    """Remove checkpoint directories that are older than the policy being staged."""
    removed: list[Path] = []
    owns_cache_fd = _cache_fd is None
    if owns_cache_fd:
        try:
            cache_fd = _open_directory_nofollow(
                cache_root,
                create=False,
                label="full checkpoint cache directory",
            )
        except FileNotFoundError:
            return removed
    else:
        assert _cache_fd is not None
        cache_fd = _cache_fd
    keep_names = {keep.name, f"{keep.name}.partial"}
    try:
        entry_count = 0
        with os.scandir(cache_fd) as entries:
            names = []
            for entry in entries:
                _check_deadline(deadline)
                entry_count += 1
                if entry_count > _ADAPTER_CLEANUP_ENTRY_LIMIT:
                    raise OSError("full checkpoint cache has too many entries")
                names.append(entry.name)
        for name in names:
            _check_deadline(deadline)
            if name in keep_names:
                continue
            try:
                entry_stat = os.stat(name, dir_fd=cache_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(entry_stat.st_mode):
                raise OSError(f"full checkpoint cache entry is a symbolic link: {name}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue
            _remove_directory_entry(
                cache_fd,
                name,
                "full checkpoint cache entry",
                deadline=deadline,
            )
            removed.append(cache_root / name)
    finally:
        if owns_cache_fd:
            os.close(cache_fd)
    return removed


def stage_full_checkpoint_files(source_url: str, *, download_fn=None) -> Path:
    """Download and atomically publish one remote sharded safetensors checkpoint."""
    destination = full_checkpoint_cache_path(source_url)
    _ensure_directory(destination.parent, "full checkpoint cache directory")
    _reject_symlink_path(destination, "full checkpoint destination")
    temporary = destination.with_name(f"{destination.name}.partial")
    _reject_symlink_path(temporary, "full checkpoint partial directory")
    cache_fd = _open_directory_nofollow(
        destination.parent,
        create=False,
        label="full checkpoint cache directory",
    )
    temporary_fd = -1
    try:
        lock_name = f"{destination.name}.lock"
        with os.fdopen(
            _open_nofollow(
                lock_name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                dir_fd=cache_fd,
            ),
            "w",
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if _full_checkpoint_cache_is_valid(cache_fd, destination.name, source_url):
                    return destination
            except FileNotFoundError:
                pass
            temporary_fd = _ensure_directory_entry(
                cache_fd,
                temporary.name,
                "full checkpoint partial directory",
            )
            index_name = "model.safetensors.index.json"
            used_delta = False
            try:
                delta_manifest = json.loads(
                    _read_source_text(
                        f"{source_url.rstrip('/')}/{DELTA_MANIFEST}",
                        max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                    )
                )
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
                    _reject_symlink_path(base_dir, "full checkpoint base directory")
                    _reject_symlink_path(base_dir / "READY", "full checkpoint base ready marker")
                    try:
                        base_ready = _directory_entry_has_marker(
                            cache_fd,
                            base_dir.name,
                            "READY",
                            "full checkpoint base directory",
                        )
                    except FileNotFoundError:
                        base_ready = False
                    if base_ready:
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
                            _reject_symlink_path(delta_path, "full checkpoint delta file")
                            _download_checkpoint_entry(
                                f"{source_url.rstrip('/')}/{delta_name}",
                                delta_path,
                                parent_fd=temporary_fd,
                                download_fn=download_fn,
                            )
                            output_path = temporary / filename
                            _reject_symlink_path(output_path, "full checkpoint shard")
                            _reject_symlink_path(
                                base_dir / filename,
                                "full checkpoint base shard",
                            )
                            apply_xor_delta_file(
                                base_dir / filename,
                                delta_path,
                                output_path,
                                expected_size=int(spec["size"]),
                                expected_sha256=str(spec["sha256"]),
                                _root_fd=cache_fd,
                                _root_path=destination.parent,
                            )
                            _reject_symlink_path(delta_path, "full checkpoint delta file")
                            if _stat_regular_entry(
                                temporary_fd,
                                delta_name,
                                "full checkpoint delta file",
                            ) is not None:
                                os.unlink(delta_name, dir_fd=temporary_fd)
                        index_path = temporary / index_name
                        _download_checkpoint_entry(
                            f"{source_url.rstrip('/')}/{index_name}",
                            index_path,
                            parent_fd=temporary_fd,
                            download_fn=download_fn,
                        )
                        if _sha256(index_path, parent_fd=temporary_fd) != delta_manifest.get("index_sha256"):
                            raise OSError("checkpoint index checksum mismatch")
                        used_delta = True

            index_path = temporary / index_name
            if not used_delta:
                _download_checkpoint_entry(
                    f"{source_url.rstrip('/')}/{index_name}",
                    index_path,
                    parent_fd=temporary_fd,
                    download_fn=download_fn,
                )
            index = json.loads(
                _read_text_nofollow(
                    index_name,
                    encoding="utf-8",
                    dir_fd=temporary_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
                raise OSError("full checkpoint index has no weight map")
            raw_shard_names = list(index["weight_map"].values())
            if not raw_shard_names or any(
                not isinstance(filename, str)
                or not filename
                or filename in {".", ".."}
                or Path(filename).name != filename
                or not filename.endswith(".safetensors")
                for filename in raw_shard_names
            ):
                raise OSError("full checkpoint index contains unsafe weight shard names")
            shard_names = sorted(set(raw_shard_names))
            if used_delta:
                delta_files = delta_manifest.get("files")
                if not isinstance(delta_files, list) or {
                    spec.get("filename")
                    for spec in delta_files
                    if isinstance(spec, dict)
                } != set(shard_names):
                    raise OSError("checkpoint delta files do not match the checkpoint index")
            if not used_delta:
                for filename in shard_names:
                    _download_checkpoint_entry(
                        f"{source_url.rstrip('/')}/{filename}",
                        temporary / filename,
                        parent_fd=temporary_fd,
                        download_fn=download_fn,
                    )
            cache_manifest = {
                "version": 1,
                "source_url": source_url,
                "index_sha256": _sha256_entry(
                    temporary_fd,
                    index_name,
                    "full checkpoint index",
                ),
                "files": {},
            }
            file_records = cache_manifest["files"]
            assert isinstance(file_records, dict)
            for filename in shard_names:
                shard_fd = _open_nofollow(
                    filename,
                    os.O_RDONLY,
                    dir_fd=temporary_fd,
                    label="full checkpoint shard",
                )
                try:
                    shard_size = os.fstat(shard_fd).st_size
                finally:
                    os.close(shard_fd)
                if shard_size <= 0:
                    raise OSError(f"full checkpoint shard is empty: {filename}")
                file_records[filename] = {
                    "size": shard_size,
                    "sha256": _sha256_entry(
                        temporary_fd,
                        filename,
                        "full checkpoint shard",
                    ),
                }
            _write_text_entry_atomic(
                temporary_fd,
                _FULL_CHECKPOINT_CACHE_MANIFEST,
                json.dumps(cache_manifest, separators=(",", ":")),
                encoding="utf-8",
            )
            _write_text_entry_atomic(
                temporary_fd,
                "READY",
                "ready\n",
                encoding="ascii",
            )
            try:
                _open_directory_entry_nofollow(
                    cache_fd,
                    destination.name,
                    "full checkpoint destination",
                )
            except FileNotFoundError:
                pass
            else:
                _remove_directory_entry(
                    cache_fd,
                    destination.name,
                    "full checkpoint destination",
                )
            os.replace(
                temporary.name,
                destination.name,
                src_dir_fd=cache_fd,
                dst_dir_fd=cache_fd,
            )
            # Keep the old policy until the replacement is complete. Generation
            # still uses it while staging, and a lossless delta needs its files.
            prune_full_checkpoint_cache(
                destination.parent,
                destination,
                _cache_fd=cache_fd,
                deadline=time.monotonic() + _ADAPTER_CLEANUP_TIMEOUT_S,
            )
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(cache_fd)
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

    def stage_lora_adapter(
        self,
        source_url: str,
        require_delta: bool = False,
        policy_step: int | None = None,
    ) -> None:
        stage_lora_adapter_files(
            source_url,
            require_delta=require_delta,
            policy_step=policy_step,
        )

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
        from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise

        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )
