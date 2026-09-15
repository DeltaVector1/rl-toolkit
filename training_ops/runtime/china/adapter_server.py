import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
import re
import socket
import shutil
import stat
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from urllib.parse import unquote, urlsplit


adapter_config_name = "adapter_config.json"
adapter_weights_name = "adapter_model.safetensors"
adapter_manifest_name = "adapter_manifest.json"
adapter_digest_name = "adapter_model.safetensors.sha256"
stable_name = "STABLE"
cache_name = ".adapter-predictive-cache"
lock_name = ".adapter-predictive-upgrade.lock"
digest_pattern = re.compile(r"[0-9a-f]{64}\Z")
step_pattern = re.compile(r"step_(\d+)\Z")
run_pattern = re.compile(r"[A-Za-z0-9_-]+\Z")
transfer_write_bytes = 256 * 1024
# Keep the server idle read bound above the worker's 90-second chunk deadline.
connection_timeout_seconds = 120


class UpgradeError(RuntimeError):
    pass


_identity_cache: OrderedDict[
    tuple[object, ...],
    dict[str, object],
] = OrderedDict()
_identity_lock = Lock()
_MAX_IDENTITY_CACHE_ENTRIES = 32


class _ValidationRun:
    def __init__(self) -> None:
        self.done = Event()
        self.result: dict[str, object] | None = None
        self.error: BaseException | None = None


_identity_validation_runs: dict[tuple[object, ...], _ValidationRun] = {}
_transport_digest_cache: OrderedDict[tuple[str, tuple[int, int, int, int, int]], None] = OrderedDict()
_transport_digest_lock = Lock()
_MAX_TRANSPORT_DIGEST_CACHE_ENTRIES = 128


@dataclass(frozen=True)
class _ManifestCacheEntry:
    key: tuple[str, tuple[int, int, int, int, int]]
    files: dict[str, dict[str, object]]
    chunk_records: tuple[tuple[str, int, str], ...]
    chunk_index: dict[str, tuple[int, str]]


_manifest_cache: OrderedDict[
    tuple[str, tuple[int, int, int, int, int]], _ManifestCacheEntry
] = OrderedDict()
_manifest_cache_lock = Lock()
_manifest_parse_lock = Lock()
_manifest_cache_limit = 64


def _json_object(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise UpgradeError(f"{description} is missing")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, ValueError) as exc:
        raise UpgradeError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise UpgradeError(f"{description} is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise UpgradeError(f"cannot read {path.name}") from exc
    try:
        stat_before = os.fstat(fd)
        if not stat.S_ISREG(stat_before.st_mode):
            raise UpgradeError(f"cannot read {path.name}")
        identity_before = _stat_identity_from_stat(stat_before)
        while chunk := os.read(fd, 16 * 1024 * 1024):
            digest.update(chunk)
        if _stat_identity_from_stat(os.fstat(fd)) != identity_before:
            raise UpgradeError(f"{path.name} changed during validation")
    except OSError as exc:
        raise UpgradeError(f"cannot read {path.name}") from exc
    finally:
        os.close(fd)
    return digest.hexdigest()


def _sha256_fd(fd: int, name: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    try:
        while chunk := os.pread(fd, 16 * 1024 * 1024, offset):
            digest.update(chunk)
            offset += len(chunk)
    except OSError as exc:
        raise UpgradeError(f"cannot read {name}") from exc
    return digest.hexdigest()


def _verify_transport_fd(fd: int, expected_size: int, expected_digest: str) -> None:
    identity = _fd_stat_identity(fd)
    if identity[2] != expected_size:
        raise UpgradeError("predictive transport chunk size differs")
    key = (expected_digest, identity)
    with _transport_digest_lock:
        if key in _transport_digest_cache:
            _transport_digest_cache.move_to_end(key)
            return
    actual_digest = _sha256_fd(fd, "predictive transport chunk")
    if _fd_stat_identity(fd) != identity:
        raise UpgradeError("predictive transport chunk changed during validation")
    if actual_digest != expected_digest:
        raise UpgradeError("predictive transport chunk checksum differs")
    with _transport_digest_lock:
        _transport_digest_cache[key] = None
        _transport_digest_cache.move_to_end(key)
        while len(_transport_digest_cache) > _MAX_TRANSPORT_DIGEST_CACHE_ENTRIES:
            _transport_digest_cache.popitem(last=False)


def _predictive_chunk_records(chunks: object) -> list[tuple[str, int, str]]:
    if isinstance(chunks, _ManifestCacheEntry):
        return list(chunks.chunk_records)
    if not isinstance(chunks, list) or not chunks:
        raise UpgradeError("predictive manifest has no transport chunks")
    records: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for item in chunks:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise UpgradeError("predictive manifest contains an invalid chunk")
        name = item["file"]
        if (
            Path(name).name != name
            or name in seen
            or not name.startswith(f"{adapter_weights_name}.transport-predictive-")
        ):
            raise UpgradeError("predictive manifest contains an unsafe chunk name")
        size = item.get("wire_size")
        digest = item.get("wire_sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or digest_pattern.fullmatch(digest) is None
        ):
            raise UpgradeError("predictive manifest chunk identity is invalid")
        records.append((name, size, digest))
        seen.add(name)
    return records


def _manifest_cache_entry(step_path: Path, step: int) -> _ManifestCacheEntry:
    canonical_step = step_path.resolve(strict=False)
    manifest_path = canonical_step / adapter_manifest_name
    for _attempt in range(3):
        manifest_identity = _stat_identity(manifest_path)
        cache_key = (str(canonical_step), manifest_identity)
        with _manifest_cache_lock:
            cached = _manifest_cache.get(cache_key)
            if cached is not None:
                _manifest_cache.move_to_end(cache_key)
                return cached
        with _manifest_parse_lock:
            with _manifest_cache_lock:
                cached = _manifest_cache.get(cache_key)
                if cached is not None:
                    _manifest_cache.move_to_end(cache_key)
                    return cached
            manifest = _json_object(manifest_path, f"adapter step {step} manifest")
            if _stat_identity(manifest_path) != manifest_identity:
                continue
            if manifest.get("kind") != "lora-adapter" or manifest.get("target_step") != step:
                raise UpgradeError(f"adapter step {step} manifest has the wrong identity")
            files = _file_records(manifest, f"adapter step {step} manifest")
            chunk_records = (
                tuple(_predictive_chunk_records(manifest.get("chunks")))
                if manifest.get("version") == 2
                else ()
            )
            entry = _ManifestCacheEntry(
                key=cache_key,
                files=files,
                chunk_records=chunk_records,
                chunk_index={
                    name: (size, digest) for name, size, digest in chunk_records
                },
            )
            with _manifest_cache_lock:
                cached = _manifest_cache.get(cache_key)
                if cached is not None:
                    _manifest_cache.move_to_end(cache_key)
                    return cached
                _manifest_cache[cache_key] = entry
                _manifest_cache.move_to_end(cache_key)
                while len(_manifest_cache) > _manifest_cache_limit:
                    _manifest_cache.popitem(last=False)
            return entry
    raise UpgradeError(f"adapter step {step} manifest changed during validation")


def _validate_predictive_chunks(
    directory: Path, chunks: object
) -> list[Path]:
    paths: list[Path] = []
    for name, size, digest in _predictive_chunk_records(chunks):
        if (directory / name).is_symlink() or not (directory / name).is_file():
            raise UpgradeError(f"predictive manifest chunk is missing: {name}")
        try:
            fd = os.open(
                directory / name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise UpgradeError(f"predictive manifest chunk is unavailable: {name}") from exc
        try:
            observed = os.fstat(fd)
            if not stat.S_ISREG(observed.st_mode):
                raise UpgradeError(f"predictive manifest chunk is not regular: {name}")
            _verify_transport_fd(fd, size, digest)
        finally:
            os.close(fd)
        paths.append(directory / name)
    return paths


def _stat_identity_from_stat(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise UpgradeError(f"cannot stat {path.name}") from exc
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISREG(stat_result.st_mode):
        raise UpgradeError(f"cannot stat regular file {path.name}")
    return _stat_identity_from_stat(stat_result)


def _directory_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise UpgradeError(f"cannot stat directory {path.name}") from exc
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISDIR(stat_result.st_mode):
        raise UpgradeError(f"cannot stat directory {path.name}")
    return _stat_identity_from_stat(stat_result)


def _fd_stat_identity(fd: int) -> tuple[int, int, int, int, int]:
    try:
        return _stat_identity_from_stat(os.fstat(fd))
    except OSError as exc:
        raise UpgradeError("cannot stat served adapter file") from exc


def _stable_rename_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    file_type = stat.S_IFMT(stat_result.st_mode)
    if stat.S_ISDIR(stat_result.st_mode):
        return stat_result.st_dev, stat_result.st_ino, file_type
    if stat.S_ISREG(stat_result.st_mode):
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            file_type,
        )
    return stat_result.st_dev, stat_result.st_ino, file_type


def _file_records(manifest: dict[str, object], description: str) -> dict[str, dict[str, object]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise UpgradeError(f"{description} has no file records")
    result: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise UpgradeError(f"{description} has an invalid file record")
        name = item.get("name")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or name in result
            or name not in {adapter_config_name, adapter_weights_name}
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or digest_pattern.fullmatch(digest) is None
        ):
            raise UpgradeError(f"{description} has an invalid file identity")
        result[name] = {"size": size, "sha256": digest}
    if set(result) != {adapter_config_name, adapter_weights_name}:
        raise UpgradeError(f"{description} has an incomplete file set")
    return result


def _identity_from_published_directory(
    directory: Path,
    step: int,
    *,
    cache: bool,
    weight_fd: int | None = None,
    verify_weights: bool = True,
) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise UpgradeError(f"adapter step {step} directory is missing")
    if not cache and (directory / stable_name).is_symlink():
        raise UpgradeError(f"adapter step {step} marker is a symlink")
    if not cache and not (directory / stable_name).is_file():
        raise UpgradeError(f"adapter step {step} is not stable")

    manifest_entry: _ManifestCacheEntry | None = None
    predictive_chunk_records: tuple[tuple[str, int, str], ...] = ()
    if cache:
        identity = _json_object(directory / ".identity.json", f"adapter cache step {step} identity")
        if identity.get("step") != step:
            raise UpgradeError(f"adapter cache step identity differs for step {step}")
        records = {
            adapter_config_name: {
                "size": identity.get("config_size"),
                "sha256": identity.get("config_sha256"),
            },
            adapter_weights_name: {
                "size": identity.get("weights_size"),
                "sha256": identity.get("weights_sha256"),
            },
        }
        for name, record in records.items():
            if (
                type(record["size"]) is not int
                or record["size"] <= 0
                or not isinstance(record["sha256"], str)
                or digest_pattern.fullmatch(record["sha256"]) is None
            ):
                raise UpgradeError(f"adapter cache step {step} has an invalid {name} identity")
        description = f"adapter cache step {step}"
    else:
        manifest_entry = _manifest_cache_entry(directory, step)
        records = manifest_entry.files
        predictive_chunk_records = manifest_entry.chunk_records
        description = f"adapter step {step}"

    if records[adapter_weights_name]["size"] <= 0:
        raise UpgradeError(f"{description} weights are empty")

    for name, record in records.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            raise UpgradeError(f"{description} file is incomplete: {name}")

    digest_path = directory / adapter_digest_name
    if digest_path.is_symlink() or not digest_path.is_file():
        raise UpgradeError(f"{description} weight checksum is missing")
    try:
        fd = os.open(digest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="ascii") as source:
            sidecar_digest = source.read().strip()
    except (OSError, UnicodeError) as exc:
        raise UpgradeError(f"{description} weight checksum cannot be read") from exc
    if sidecar_digest != records[adapter_weights_name]["sha256"]:
        raise UpgradeError(f"{description} weight checksum differs")
    opened_weight_stat = _fd_stat_identity(weight_fd) if weight_fd is not None else None
    identity_path = directory / (".identity.json" if cache else adapter_manifest_name)
    stat_paths = [
        directory / adapter_config_name,
        directory / adapter_weights_name,
        identity_path,
        digest_path,
    ]
    if not cache:
        stat_paths.append(directory / stable_name)
    stat_identities = tuple(_stat_identity(path) for path in stat_paths)
    if manifest_entry is not None and stat_identities[2] != manifest_entry.key[1]:
        raise UpgradeError(f"{description} manifest changed during validation")
    metadata_key = (
        step,
        cache,
        verify_weights,
        stat_identities,
        _directory_identity(directory),
        manifest_entry.key if manifest_entry is not None else None,
        opened_weight_stat,
    )
    if opened_weight_stat is not None and opened_weight_stat != _stat_identity(
        directory / adapter_weights_name
    ):
        raise UpgradeError(f"{description} weights changed before validation")

    with _identity_lock:
        validation_run = _identity_validation_runs.get(metadata_key)
        if validation_run is None:
            validation_run = _ValidationRun()
            _identity_validation_runs[metadata_key] = validation_run
            is_owner = True
        else:
            is_owner = False
    if not is_owner:
        validation_run.done.wait()
        if validation_run.error is not None:
            raise validation_run.error
        if validation_run.result is None:
            raise UpgradeError("published adapter validation returned no identity")
        return validation_run.result.copy()

    try:
        cache_key = metadata_key
        with _identity_lock:
            cached = _identity_cache.get(cache_key)
            if cached is not None:
                _identity_cache.move_to_end(cache_key)
        if cached is not None:
            validated_identity = cached.copy()
        else:
            if predictive_chunk_records:
                _validate_predictive_chunks(
                    directory,
                    manifest_entry,
                )
            actual_config_digest = _sha256(directory / adapter_config_name)
            if actual_config_digest != records[adapter_config_name]["sha256"]:
                raise UpgradeError(f"{description} configuration checksum differs")
            if verify_weights:
                actual_weights_digest = (
                    _sha256_fd(weight_fd, adapter_weights_name)
                    if weight_fd is not None
                    else _sha256(directory / adapter_weights_name)
                )
                if actual_weights_digest != records[adapter_weights_name]["sha256"]:
                    raise UpgradeError(f"{description} weight checksum differs")
            if weight_fd is not None and _fd_stat_identity(weight_fd) != opened_weight_stat:
                raise UpgradeError(f"{description} weights changed during validation")
            validated_identity = {
                "step": step,
                "config_size": records[adapter_config_name]["size"],
                "config_sha256": records[adapter_config_name]["sha256"],
                "weights_size": records[adapter_weights_name]["size"],
                "weights_sha256": records[adapter_weights_name]["sha256"],
            }
            with _identity_lock:
                _identity_cache[cache_key] = validated_identity.copy()
                _identity_cache.move_to_end(cache_key)
                while len(_identity_cache) > _MAX_IDENTITY_CACHE_ENTRIES:
                    _identity_cache.popitem(last=False)
                validated_identity = validated_identity.copy()
    except BaseException as exc:
        with _identity_lock:
            validation_run.error = exc
            _identity_validation_runs.pop(metadata_key, None)
            validation_run.done.set()
        raise
    else:
        with _identity_lock:
            validation_run.result = validated_identity.copy()
            _identity_validation_runs.pop(metadata_key, None)
            validation_run.done.set()
        return validated_identity.copy()


def _cache_directory(active_run: Path, step: int) -> Path:
    return active_run / cache_name / f"step_{step}"


def _find_base(active_run: Path, step: int) -> tuple[Path, dict[str, object]]:
    errors: list[str] = []
    published = active_run / f"step_{step}"
    try:
        return published, _identity_from_published_directory(published, step, cache=False)
    except UpgradeError as exc:
        errors.append(str(exc))
    cached = _cache_directory(active_run, step)
    try:
        return cached, _identity_from_published_directory(cached, step, cache=True)
    except UpgradeError as exc:
        errors.append(str(exc))
    raise UpgradeError(f"cannot find exact adapter base step {step}: {'; '.join(errors)}")


def _link_private_file(source: Path, destination: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | directory_flag | getattr(os, "O_NOFOLLOW", 0)
    source_fd = destination_fd = source_dir = destination_dir = None
    try:
        source_fd = os.open(source, file_flags)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise OSError("source is not a regular file")
        source_identity_before = _stat_identity_from_stat(source_stat)
        if source_stat.st_mode & 0o777 != 0o600:
            os.fchmod(source_fd, 0o600)
            source_stat = os.fstat(source_fd)
            source_identity_before = _stat_identity_from_stat(source_stat)
        source_dir = os.open(source.parent, directory_flags)
        destination_dir = os.open(destination.parent, directory_flags)
        os.link(
            source.name,
            destination.name,
            src_dir_fd=source_dir,
            dst_dir_fd=destination_dir,
            follow_symlinks=False,
        )
        destination_fd = os.open(destination.name, file_flags, dir_fd=destination_dir)
        destination_stat = os.fstat(destination_fd)
        source_identity = _stat_identity_from_stat(os.fstat(source_fd))
        if source_identity[:4] != source_identity_before[:4]:
            raise OSError("source changed while linking adapter file")
        if _stat_identity_from_stat(destination_stat) != source_identity:
            raise OSError("linked adapter file identity differs")
    except OSError as exc:
        if destination_dir is not None:
            try:
                os.unlink(destination.name, dir_fd=destination_dir)
            except FileNotFoundError:
                pass
        raise UpgradeError(f"cannot retain private adapter file {destination.name}") from exc
    finally:
        for fd in (source_fd, destination_fd, source_dir, destination_dir):
            if fd is not None:
                os.close(fd)


def _chmod_private_entry(directory: Path) -> None:
    os.chmod(directory, 0o700)
    for name in (
        adapter_config_name,
        adapter_weights_name,
        adapter_digest_name,
        ".identity.json",
    ):
        path = directory / name
        if path.exists() and not path.is_symlink():
            os.chmod(path, 0o600)


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise UpgradeError(f"cannot fsync adapter file {path.name}") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise UpgradeError(f"cannot fsync adapter directory {path.name}") from exc


def _replace_from_dirs(
    source: Path,
    destination: Path,
    expected_identity: tuple[int, int, int, int, int] | None = None,
    expected_digest: str | None = None,
    directory_validator=None,
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    flags = os.O_RDONLY | directory_flag | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_dir = os.open(source.parent, flags)
        destination_dir = os.open(destination.parent, flags)
    except OSError as exc:
        try:
            os.close(source_dir)
        except UnboundLocalError:
            pass
        raise UpgradeError("cannot open adapter publication directory") from exc
    try:
        source_stat = os.stat(source.name, dir_fd=source_dir, follow_symlinks=False)
        if stat.S_ISDIR(source_stat.st_mode):
            _replace_directory_from_dirs(
                source_dir,
                destination_dir,
                source.name,
                destination.name,
                expected_identity,
                directory_validator,
                destination.parent,
            )
        elif stat.S_ISREG(source_stat.st_mode):
            _replace_regular_from_dirs(
                source_dir,
                destination_dir,
                source.name,
                destination.name,
                expected_identity,
                expected_digest,
            )
        else:
            raise OSError("adapter publication source is not regular")
    except OSError as exc:
        raise UpgradeError("cannot publish adapter file") from exc
    finally:
        os.close(source_dir)
        os.close(destination_dir)


def _replace_directory_from_dirs(
    source_dir: int,
    destination_dir: int,
    source_name: str,
    destination_name: str,
    expected_identity: tuple[int, int, int, int, int] | None,
    validator,
    destination_parent: Path,
) -> None:
    source_stat = os.stat(source_name, dir_fd=source_dir, follow_symlinks=False)
    source_identity = _stat_identity_from_stat(source_stat)
    if expected_identity is not None and source_identity != expected_identity:
        raise OSError("adapter publication source changed")
    stable_source_identity = _stable_rename_identity(source_stat)
    candidate_name = f".{destination_name}.candidate-{os.getpid()}-{time.time_ns()}"
    backup_name = f".{destination_name}.backup-{os.getpid()}-{time.time_ns()}"
    candidate_path = destination_parent / candidate_name
    destination_exists = False
    destination_replaced = False
    try:
        os.replace(
            source_name,
            candidate_name,
            src_dir_fd=source_dir,
            dst_dir_fd=destination_dir,
        )
        candidate_stat = os.stat(candidate_name, dir_fd=destination_dir, follow_symlinks=False)
        if _stable_rename_identity(candidate_stat) != stable_source_identity:
            raise OSError("adapter publication candidate identity differs")
        if validator is not None:
            validator(candidate_path)
            validator(candidate_path)
        candidate_stat = os.stat(candidate_name, dir_fd=destination_dir, follow_symlinks=False)
        if _stable_rename_identity(candidate_stat) != stable_source_identity:
            raise OSError("adapter publication candidate changed")
        try:
            old_stat = os.stat(destination_name, dir_fd=destination_dir, follow_symlinks=False)
        except FileNotFoundError:
            old_stat = None
        if old_stat is not None:
            if not stat.S_ISDIR(old_stat.st_mode):
                raise OSError("adapter publication destination is not a directory")
            destination_exists = True
            os.replace(
                destination_name,
                backup_name,
                src_dir_fd=destination_dir,
                dst_dir_fd=destination_dir,
            )
        os.replace(
            candidate_name,
            destination_name,
            src_dir_fd=destination_dir,
            dst_dir_fd=destination_dir,
        )
        destination_replaced = True
        if validator is not None:
            validator(destination_parent / destination_name)
        destination_stat = os.stat(destination_name, dir_fd=destination_dir, follow_symlinks=False)
        if _stable_rename_identity(destination_stat) != stable_source_identity:
            raise OSError("adapter publication destination identity differs")
        if destination_exists:
            shutil.rmtree(destination_parent / backup_name)
            backup_name = ""
    except (OSError, UpgradeError) as publication_error:
        try:
            if destination_exists and backup_name:
                if destination_replaced:
                    _remove_leftover(destination_parent / destination_name)
                os.replace(
                    backup_name,
                    destination_name,
                    src_dir_fd=destination_dir,
                    dst_dir_fd=destination_dir,
                )
                backup_name = ""
            elif destination_replaced:
                _remove_leftover(destination_parent / destination_name)
        except (FileNotFoundError, OSError, UpgradeError) as restore_error:
            raise UpgradeError("cannot restore previous adapter directory") from restore_error
        raise publication_error
    finally:
        try:
            candidate_stat = os.stat(candidate_name, dir_fd=destination_dir, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(candidate_stat.st_mode):
                shutil.rmtree(candidate_path)
            else:
                os.unlink(candidate_path)
        if backup_name:
            try:
                shutil.rmtree(destination_parent / backup_name)
            except FileNotFoundError:
                pass


def _replace_regular_from_dirs(
    source_dir: int,
    destination_dir: int,
    source_name: str,
    destination_name: str,
    expected_identity: tuple[int, int, int, int, int] | None,
    expected_digest: str | None,
) -> None:
    source_fd = destination_fd = backup_fd = None
    temp_name = f".{destination_name}.publish-{os.getpid()}-{time.time_ns()}"
    backup_name = f".{destination_name}.backup-{os.getpid()}-{time.time_ns()}"
    destination_exists = False
    destination_replaced = False
    try:
        source_fd = os.open(
            source_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_dir,
        )
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise OSError("adapter publication source is not regular")
        source_identity = _stat_identity_from_stat(source_stat)
        if expected_identity is not None and source_identity != expected_identity:
            raise OSError("adapter publication source changed")
        baseline_digest = expected_digest
        if baseline_digest is None:
            baseline_digest = _sha256_fd(source_fd, "adapter publication source")
            if _stat_identity_from_stat(os.fstat(source_fd)) != source_identity:
                raise OSError("adapter publication source changed")
        try:
            destination_fd = os.open(
                destination_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=destination_dir,
            )
        except FileNotFoundError:
            pass
        else:
            destination_stat = os.fstat(destination_fd)
            if not stat.S_ISREG(destination_stat.st_mode):
                raise OSError("adapter publication destination is not regular")
            destination_exists = True
            os.link(
                destination_name,
                backup_name,
                src_dir_fd=destination_dir,
                dst_dir_fd=destination_dir,
                follow_symlinks=False,
            )
            backup_fd = os.open(
                backup_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=destination_dir,
            )
            if _stable_rename_identity(os.fstat(backup_fd)) != _stable_rename_identity(destination_stat):
                raise OSError("adapter publication backup identity differs")
        os.link(
            source_name,
            temp_name,
            src_dir_fd=source_dir,
            dst_dir_fd=destination_dir,
            follow_symlinks=False,
        )
        temp_fd = os.open(
            temp_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=destination_dir,
        )
        try:
            temp_stat = os.fstat(temp_fd)
            temp_identity = _stat_identity_from_stat(temp_stat)
            stable_temp_identity = _stable_rename_identity(temp_stat)
        finally:
            os.close(temp_fd)
        source_identity_after_link = _stat_identity_from_stat(os.fstat(source_fd))
        if temp_identity != source_identity_after_link:
            raise OSError("adapter publication source changed")
        if expected_identity is not None and source_identity_after_link[:4] != expected_identity[:4]:
            raise OSError("adapter publication source changed")
        actual_digest = _sha256_fd(source_fd, "adapter publication source")
        if _stat_identity_from_stat(os.fstat(source_fd)) != source_identity_after_link:
            raise OSError("adapter publication source changed")
        if actual_digest != baseline_digest:
            raise OSError("adapter publication source checksum differs")
        os.replace(temp_name, destination_name, src_dir_fd=destination_dir, dst_dir_fd=destination_dir)
        destination_replaced = True
        destination_stat = os.stat(destination_name, dir_fd=destination_dir, follow_symlinks=False)
        if stat.S_ISLNK(destination_stat.st_mode) or _stable_rename_identity(destination_stat) != stable_temp_identity:
            raise OSError("adapter publication destination identity differs")
        published_fd = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=destination_dir,
        )
        try:
            published_stat = os.fstat(published_fd)
            if _stable_rename_identity(published_stat) != stable_temp_identity:
                raise OSError("adapter publication destination identity differs")
            try:
                published_digest = _sha256_fd(
                    published_fd,
                    "adapter publication destination",
                )
            except UpgradeError as exc:
                raise OSError(
                    "adapter publication destination checksum cannot be read"
                ) from exc
            if published_digest != baseline_digest:
                raise OSError("adapter publication destination checksum differs")
        finally:
            os.close(published_fd)
        if destination_exists and backup_name:
            os.unlink(backup_name, dir_fd=destination_dir)
        temp_name = ""
        backup_name = ""
    except OSError:
        if temp_name:
            try:
                os.unlink(temp_name, dir_fd=destination_dir)
            except FileNotFoundError:
                pass
        if destination_exists and backup_name:
            try:
                os.replace(
                    backup_name,
                    destination_name,
                    src_dir_fd=destination_dir,
                    dst_dir_fd=destination_dir,
                )
                backup_name = ""
            except OSError:
                try:
                    os.unlink(destination_name, dir_fd=destination_dir)
                except FileNotFoundError:
                    pass
        elif destination_replaced:
            try:
                os.unlink(destination_name, dir_fd=destination_dir)
            except FileNotFoundError:
                pass
        raise
    finally:
        for fd in (source_fd, destination_fd, backup_fd):
            if fd is not None:
                os.close(fd)
        if backup_name:
            try:
                os.unlink(backup_name, dir_fd=destination_dir)
            except FileNotFoundError:
                pass


def _remove_leftover(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UpgradeError("cannot inspect predictive upgrade leftover") from exc
    try:
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        raise UpgradeError("cannot remove predictive upgrade leftover") from exc


def _scavenge_upgrade_leftovers(active_run: Path) -> None:
    cache_root = active_run / cache_name
    if cache_root.is_symlink() or (cache_root.exists() and not cache_root.is_dir()):
        raise UpgradeError("adapter cache root is not a directory")
    patterns = (re.compile(r"\.predictive-build-\d+-\d+\Z"),)
    try:
        candidates = list(active_run.iterdir())
        if cache_root.is_dir():
            candidates.extend(cache_root.iterdir())
    except OSError as exc:
        raise UpgradeError("cannot inspect predictive upgrade leftovers") from exc
    for candidate in candidates:
        if any(pattern.fullmatch(candidate.name) for pattern in patterns) or (
            candidate.parent == cache_root and re.fullmatch(r"\.pending-\d+-\d+\Z", candidate.name)
        ):
            _remove_leftover(candidate)


def _link_cache_entry(source: Path, identity: dict[str, object], destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        try:
            existing = _identity_from_published_directory(destination, int(identity["step"]), cache=True)
        except UpgradeError:
            shutil.rmtree(destination) if destination.is_dir() and not destination.is_symlink() else destination.unlink()
        else:
            if existing != identity:
                raise UpgradeError(f"adapter cache step {identity['step']} identity differs")
            _chmod_private_entry(destination)
            return
    destination.mkdir(parents=True, mode=0o700)
    os.chmod(destination, 0o700)
    for name in (adapter_config_name, adapter_weights_name):
        source_path = source / name
        target_path = destination / name
        _link_private_file(source_path, target_path)
    (destination / adapter_digest_name).write_text(f"{identity['weights_sha256']}\n", encoding="ascii")
    (destination / ".identity.json").write_text(
        json.dumps(identity, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for name in (adapter_digest_name, ".identity.json"):
        os.chmod(destination / name, 0o600)
    _chmod_private_entry(destination)
    for name in (
        adapter_config_name,
        adapter_weights_name,
        adapter_digest_name,
        ".identity.json",
    ):
        _fsync_file(destination / name)
    _fsync_directory(destination)


def _prepare_cache(
    active_run: Path,
    entries: list[tuple[Path, dict[str, object]]],
    pending: Path,
) -> None:
    pending.mkdir(parents=True, mode=0o700)
    os.chmod(pending, 0o700)
    for source, identity in entries:
        step = int(identity["step"])
        _link_cache_entry(source, identity, pending / f"step_{step}")
    _fsync_directory(pending)


def _validate_cache_candidate(
    candidate: Path, step: int, expected_identity: dict[str, object]
) -> None:
    if _identity_from_published_directory(candidate, step, cache=True) != expected_identity:
        raise UpgradeError("adapter cache changed during publication")


def _promote_cache(active_run: Path, pending: Path, newest_step: int) -> None:
    root = active_run / cache_name
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise UpgradeError("adapter cache root is not a directory")
    root.mkdir(exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)

    prepared: dict[int, tuple[Path, Path, dict[str, object]]] = {}
    for step in (newest_step, newest_step - 1):
        source = pending / f"step_{step}"
        destination = root / f"step_{step}"
        identity = _identity_from_published_directory(source, step, cache=True)
        if destination.exists() or destination.is_symlink():
            existing = _identity_from_published_directory(destination, step, cache=True)
            if existing != identity:
                raise UpgradeError(f"adapter cache step {step} identity differs")
        prepared[step] = source, destination, identity

    for step, (source, destination, _identity) in prepared.items():
        if destination.exists():
            shutil.rmtree(source)
        else:
            _replace_from_dirs(
                source,
                destination,
                _directory_identity(source),
                directory_validator=lambda candidate, step=step, identity=_identity: _validate_cache_candidate(
                    candidate, step, identity
                ),
            )
        _chmod_private_entry(destination)
        _fsync_directory(root)

    for step, (_source, destination, identity) in prepared.items():
        if _identity_from_published_directory(destination, step, cache=True) != identity:
            raise UpgradeError(f"published adapter cache step {step} identity differs")


def _prune_cache(active_run: Path, newest_step: int) -> None:
    root = active_run / cache_name
    if root.is_symlink() or not root.is_dir():
        raise UpgradeError("adapter cache root is not a directory")
    for step in (newest_step, newest_step - 1):
        _identity_from_published_directory(root / f"step_{step}", step, cache=True)
    for candidate in root.iterdir():
        match = step_pattern.fullmatch(candidate.name)
        if match is None or int(match.group(1)) >= newest_step - 1:
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()


def _install_predictive_manifest(
    target: Path,
    build: Path,
    manifest: dict[str, object],
    step: int,
) -> None:
    if (
        manifest.get("version") != 2
        or manifest.get("mode") != "delta"
        or manifest.get("target_step") != step
        or manifest.get("base_step") != step - 1
        or manifest.get("base2_step") != step - 2
        or not isinstance(manifest.get("chunks"), list)
        or not manifest["chunks"]
    ):
        raise UpgradeError("predictive manifest build returned an invalid manifest")
    chunk_names: list[str] = []
    seen_chunks: set[str] = set()
    validated_identities: dict[str, tuple[int, int, int, int, int]] = {}
    validated_digests: dict[str, str] = {}
    for item in manifest["chunks"]:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise UpgradeError("predictive manifest contains an invalid chunk")
        name = item["file"]
        if (
            Path(name).name != name
            or name in seen_chunks
            or not name.startswith(f"{adapter_weights_name}.transport-predictive-")
        ):
            raise UpgradeError("predictive manifest contains an unsafe chunk name")
        chunk = build / name
        if chunk.is_symlink() or not chunk.is_file():
            raise UpgradeError(f"predictive manifest chunk is missing: {name}")
        wire_size = item.get("wire_size")
        wire_sha256 = item.get("wire_sha256")
        if (
            type(wire_size) is not int
            or wire_size < 0
            or not isinstance(wire_sha256, str)
            or digest_pattern.fullmatch(wire_sha256) is None
        ):
            raise UpgradeError(f"predictive manifest chunk identity is invalid: {name}")
        if chunk.stat().st_size != wire_size:
            raise UpgradeError(f"predictive manifest chunk size differs: {name}")
        if _sha256(chunk) != wire_sha256:
            raise UpgradeError(f"predictive manifest chunk checksum differs: {name}")
        validated_identities[name] = _stat_identity(chunk)
        validated_digests[name] = wire_sha256
        seen_chunks.add(name)
        chunk_names.append(name)
    target_manifest = target / adapter_manifest_name
    target_digest = target / adapter_digest_name
    build_manifest = build / adapter_manifest_name
    build_digest = build / adapter_digest_name
    if (
        target_manifest.is_symlink()
        or target_digest.is_symlink()
        or build_manifest.is_symlink()
        or build_digest.is_symlink()
        or not build_manifest.is_file()
        or not build_digest.is_file()
    ):
        raise UpgradeError("predictive manifest files are not regular files")
    validated_identities[adapter_digest_name] = _stat_identity(build_digest)
    validated_identities[adapter_manifest_name] = _stat_identity(build_manifest)
    validated_digests[adapter_digest_name] = _sha256(build_digest)
    validated_digests[adapter_manifest_name] = _sha256(build_manifest)
    for artifact in build.iterdir():
        if artifact.is_symlink() or not artifact.is_file():
            raise UpgradeError("predictive build contains a non-regular artifact")
        _fsync_file(artifact)
    _fsync_directory(build)
    stale_chunks = []
    for name in chunk_names:
        target_chunk = target / name
        if target_chunk.is_symlink():
            raise UpgradeError(f"predictive target chunk is a symlink: {name}")
    for old_chunk in target.glob(f"{adapter_weights_name}.transport-predictive-*"):
        if old_chunk.name not in chunk_names:
            if old_chunk.is_symlink():
                raise UpgradeError(f"predictive stale chunk is a symlink: {old_chunk.name}")
            if old_chunk.is_file():
                stale_chunks.append(old_chunk)
    for name in chunk_names:
        _replace_from_dirs(
            build / name,
            target / name,
            validated_identities[name],
            validated_digests[name],
        )
    _replace_from_dirs(
        build_digest,
        target_digest,
        validated_identities[adapter_digest_name],
        validated_digests[adapter_digest_name],
    )
    _fsync_directory(target)
    _replace_from_dirs(
        build_manifest,
        target_manifest,
        validated_identities[adapter_manifest_name],
        validated_digests[adapter_manifest_name],
    )
    _fsync_directory(target)
    for name in (adapter_config_name, adapter_weights_name):
        for old_chunk in target.glob(f"{name}.transport-delta-*"):
            if old_chunk.is_file() and not old_chunk.is_symlink():
                old_chunk.unlink()
        for old_chunk in target.glob(f"{name}.transport-predictive-*"):
            if old_chunk in stale_chunks:
                old_chunk.unlink()
    _fsync_directory(target)


def _predictive_step_url(base_url: str, step: int) -> str:
    return f"{base_url.rstrip('/')}/step_{step}"


def _has_newer_stable_step(active_run: Path, step: int) -> bool:
    """Keep the public manifest for an older stable step unchanged."""
    try:
        candidates = active_run.iterdir()
        for candidate in candidates:
            match = step_pattern.fullmatch(candidate.name)
            if match is None or int(match.group(1)) <= step:
                continue
            stable = candidate / stable_name
            if stable.is_file() and not stable.is_symlink():
                return True
    except OSError as exc:
        raise UpgradeError("cannot inspect active adapter steps") from exc
    return False


def _upgrade_v1_step(active_run: Path, target: Path, step: int, source_base_url: str) -> None:
    lock_path = active_run / lock_name
    if lock_path.is_symlink():
        raise UpgradeError("predictive upgrade lock is a symlink")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(lock_fd, 0o600)
        lock = os.fdopen(lock_fd, "a+")
    except OSError as exc:
        raise UpgradeError("cannot open predictive upgrade lock") from exc
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _scavenge_upgrade_leftovers(active_run)
            stable = target / stable_name
            if stable.is_symlink() or not stable.is_file():
                raise UpgradeError(f"adapter step {step} is not stable")
            manifest_path = target / adapter_manifest_name
            manifest = _json_object(manifest_path, f"adapter step {step} manifest")
            if manifest.get("version") == 2:
                return
            if manifest.get("version") != 1 or manifest.get("kind") != "lora-adapter":
                raise UpgradeError(f"adapter step {step} has an unsupported manifest")
            if manifest.get("target_step") != step:
                raise UpgradeError(f"adapter step {step} manifest has the wrong target")
            target_identity = _identity_from_published_directory(target, step, cache=False)
            base1, base1_identity = _find_base(active_run, step - 1)
            base2, base2_identity = _find_base(active_run, step - 2)
            identities = (target_identity, base1_identity, base2_identity)
            if len({item["config_sha256"] for item in identities}) != 1:
                raise UpgradeError(f"adapter step {step} predecessor configurations differ")
            if len({item["weights_size"] for item in identities}) != 1:
                raise UpgradeError(f"adapter step {step} predecessor sizes differ")

            cache_root = active_run / cache_name
            pending = cache_root / f".pending-{os.getpid()}-{time.time_ns()}"
            build = active_run / f".predictive-build-{os.getpid()}-{time.time_ns()}"
            try:
                _prepare_cache(
                    active_run,
                    [
                        (target, target_identity),
                        (base1, base1_identity),
                        (base2, base2_identity),
                    ],
                    pending,
                )
                build.mkdir(parents=True, mode=0o700)
                os.chmod(build, 0o700)
                for name in (adapter_config_name, adapter_weights_name):
                    _link_private_file(target / name, build / name)
                from prime_rl.utils.adapter_delta import publish_adapter_manifest

                manifest = publish_adapter_manifest(
                    build,
                    pending / f"step_{step - 1}",
                    base2_dir=pending / f"step_{step - 2}",
                    target_url=_predictive_step_url(source_base_url, step),
                    base_url=_predictive_step_url(source_base_url, step - 1),
                    base2_url=_predictive_step_url(source_base_url, step - 2),
                    target_step=step,
                    base_step=step - 1,
                    base2_step=step - 2,
                )
                for artifact in build.iterdir():
                    if artifact.is_symlink() or not artifact.is_file():
                        raise UpgradeError("predictive build contains a non-regular artifact")
                    os.chmod(artifact, 0o600)
                _promote_cache(active_run, pending, step)
                _install_predictive_manifest(target, build, manifest, step)
                _prune_cache(active_run, step)
            finally:
                if build.exists():
                    shutil.rmtree(build)
                if pending.exists():
                    shutil.rmtree(pending)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _safe_request_path(handler: SimpleHTTPRequestHandler) -> tuple[Path, tuple[str, ...]]:
    raw_path = unquote(urlsplit(handler.path).path)
    if "\\" in raw_path:
        raise UpgradeError("request path contains a backslash")
    pieces = tuple(part for part in PurePosixPath(raw_path).parts if part not in {"/", ""})
    if any(part in {".", ".."} for part in pieces):
        raise UpgradeError("request path contains traversal")
    root = Path(handler.directory).resolve()
    path = Path(handler.translate_path(handler.path)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise UpgradeError("request path is outside the adapter root") from exc
    relative = path.relative_to(root)
    if cache_name in relative.parts or any(
        part.startswith(".predictive-") for part in relative.parts
    ):
        raise UpgradeError("private adapter state is not served")
    return path, pieces


def _byte_range(value: str | None, size: int) -> tuple[int, int, int]:
    if value is None:
        return 200, 0, max(0, size - 1)
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("only one byte range is supported")
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if match is None or size == 0:
        raise ValueError("byte range is invalid")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("byte range is invalid")
    if not first:
        count = int(last)
        if count <= 0:
            raise ValueError("byte range is invalid")
        return 206, max(0, size - count), size - 1
    start = int(first)
    if start >= size:
        raise ValueError("byte range starts after the file")
    requested_last = int(last) if last else size - 1
    if requested_last < start:
        raise ValueError("byte range ends before it starts")
    return 206, start, min(requested_last, size - 1)


def _send_range_error(handler: SimpleHTTPRequestHandler, size: int, message: str) -> None:
    body = f"<!DOCTYPE HTML>\n<html><head><title>Error response</title></head>\n<body><h1>Error response</h1>\n<p>{message}</p>\n</body></html>\n".encode(
        "utf-8"
    )
    handler.send_response(416, "Requested Range Not Satisfiable")
    handler.send_header("Content-Type", "text/html;charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Range", f"bytes */{size}")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    if handler.command != "HEAD":
        handler.wfile.write(body)


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, upgrade_path=None, source_base_url=None, **kwargs):
        self.upgrade_path = upgrade_path
        self.source_base_url = source_base_url
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(connection_timeout_seconds)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True

    def _maybe_upgrade(self):
        path, pieces = _safe_request_path(self)
        if self.upgrade_path is None:
            return path
        if cache_name in pieces or any(part.startswith(".predictive-") for part in pieces):
            raise UpgradeError("private adapter state is not served")
        prefix_length = len(self.upgrade_path)
        if len(pieces) < prefix_length + 2 or pieces[:prefix_length] != self.upgrade_path:
            return path
        match = step_pattern.fullmatch(pieces[prefix_length])
        if match is None:
            return path
        step = int(match.group(1))
        if step < 2:
            return path
        if pieces[prefix_length + 1] == stable_name:
            return path
        active_run = Path(self.directory).resolve().joinpath(*self.upgrade_path)
        if _has_newer_stable_step(active_run, step):
            return path
        target = active_run / pieces[prefix_length]
        _upgrade_v1_step(active_run, target, step, self.source_base_url)
        return path

    def _predictive_chunk_identity(
        self, step_path: Path, step: int, name: str
    ) -> tuple[int, str] | None:
        if not name.startswith(f"{adapter_weights_name}.transport-predictive-"):
            return None
        entry = _manifest_cache_entry(step_path, step)
        return entry.chunk_index.get(name)

    def send_head(self):
        self.__dict__.pop("remaining", None)
        source = None
        try:
            _request_path, pieces = _safe_request_path(self)
            path = self._maybe_upgrade()
            prefix = self.upgrade_path
            step_path = None
            step = None
            if prefix is not None:
                prefix_length = len(prefix)
                if len(pieces) > prefix_length and pieces[:prefix_length] == prefix:
                    match = step_pattern.fullmatch(pieces[prefix_length])
                    if match is not None:
                        step = int(match.group(1))
                        step_path = Path(self.directory).resolve().joinpath(
                            *pieces[: prefix_length + 1]
                        )
            chunk_identity = None
            if pieces and pieces[-1].startswith(
                f"{adapter_weights_name}.transport-predictive-"
            ):
                if step_path is None:
                    step_path = path.parent
                chunk_step = step
                if chunk_step is None:
                    step_match = step_pattern.fullmatch(step_path.name)
                    if step_match is not None:
                        chunk_step = int(step_match.group(1))
                if chunk_step is not None:
                    chunk_identity = self._predictive_chunk_identity(
                        step_path, chunk_step, pieces[-1]
                    )
                if chunk_identity is None:
                    self.send_error(404)
                    return None
            if not os.path.isdir(path):
                try:
                    source = os.fdopen(
                        os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)),
                        "rb",
                    )
                except OSError:
                    self.send_error(404)
                    return None
            if step_path is not None and step is not None:
                weight_path = step_path / adapter_weights_name
                weight_fd = source.fileno() if source is not None and path == weight_path else None
                _identity_from_published_directory(
                    step_path,
                    step,
                    cache=False,
                    weight_fd=weight_fd,
                    verify_weights=weight_fd is not None,
                )
            if chunk_identity is not None:
                _verify_transport_fd(source.fileno(), *chunk_identity)
            if os.path.isdir(path):
                if source is not None:
                    source.close()
                    source = None
                return super().send_head()
            size = os.fstat(source.fileno()).st_size
            try:
                status, start, end = _byte_range(self.headers.get("Range"), size)
            except ValueError as exc:
                source.close()
                source = None
                _send_range_error(self, size, str(exc))
                return None
            if status == 206:
                self.send_response(status)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(os.fstat(source.fileno()).st_mtime))
            self.end_headers()
            source.seek(start)
            self.remaining = end - start + 1
            return source
        except UpgradeError as exc:
            if source is not None:
                source.close()
                source = None
            self.send_error(500, str(exc))
            return None
        except Exception:
            if source is not None:
                source.close()
                source = None
            raise

    def copyfile(self, source, output):
        if not hasattr(self, "remaining"):
            return super().copyfile(source, output)
        try:
            while self.remaining:
                chunk = source.read(min(self.remaining, transfer_write_bytes))
                if not chunk:
                    self.close_connection = True
                    return
                try:
                    output.write(chunk)
                except OSError:
                    self.close_connection = True
                    return
                self.remaining -= len(chunk)
        except Exception:
            self.close_connection = True
            raise
        finally:
            source.close()


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--upgrade-run")
    parser.add_argument("--upgrade-path")
    parser.add_argument("--source-base-url")
    args = parser.parse_args()

    upgrade_path = None
    upgrade_values = (args.upgrade_run, args.upgrade_path, args.source_base_url)
    if any(value is not None for value in upgrade_values):
        if not all(value is not None for value in upgrade_values):
            parser.error("predictive upgrade requires run, path, and source URL")
        if run_pattern.fullmatch(args.upgrade_run) is None:
            parser.error("--upgrade-run contains an unsafe run name")
        expected_path = f"{args.upgrade_run}/run_{args.upgrade_run}/broadcasts"
        if args.upgrade_path != expected_path:
            parser.error("--upgrade-path is not the active run broadcast path")
        if not args.source_base_url.startswith(("http://", "https://")):
            parser.error("--source-base-url is required for predictive upgrade")
        parsed_source = urlsplit(args.source_base_url.rstrip("/"))
        if (
            parsed_source.hostname not in {"127.0.0.1", "localhost"}
            or parsed_source.username
            or parsed_source.password
            or parsed_source.query
            or parsed_source.fragment
            or parsed_source.path.rstrip("/") != f"/{expected_path}"
        ):
            parser.error("--source-base-url is not bound to --upgrade-path")
        upgrade_path = tuple(expected_path.split("/"))

    AdapterServer(
        ("127.0.0.1", args.port),
        partial(
            RangeHandler,
            directory=args.directory,
            upgrade_path=upgrade_path,
            source_base_url=args.source_base_url.rstrip("/") if args.source_base_url else None,
        ),
    ).serve_forever()


if __name__ == "__main__":
    main()
