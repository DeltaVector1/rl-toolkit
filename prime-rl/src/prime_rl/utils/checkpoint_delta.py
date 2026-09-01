"""Lossless XOR deltas for immutable sharded safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zlib
from pathlib import Path

import numpy as np


DELTA_MANIFEST = "xor-delta-v1.json"
IO_CHUNK = 16 * 1024 * 1024


def _reject_symlink_path(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            current = current.parent
            if current == current.parent:
                return
            continue
        except NotADirectoryError as exc:
            raise OSError(f"{label} has a non-directory parent: {path}") from exc
        if stat.S_ISLNK(mode):
            raise OSError(f"{label} is a symbolic link: {path}")
        if current == current.parent:
            return
        current = current.parent


def _open_directory_nofollow(path: Path, *, create: bool, label: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise OSError("checkpoint delta requires O_NOFOLLOW and O_DIRECTORY")
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | directory_flag | nofollow
    current_fd = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=current_fd)
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                if exc.errno in {getattr(os, "ELOOP", 40), 40}:
                    raise OSError(f"{label} is a symbolic link: {path}") from exc
                if exc.errno in {getattr(os, "ENOTDIR", 20), 20}:
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


def _open_directory_below(
    root_fd: int,
    root_path: Path,
    path: Path,
    label: str,
) -> int:
    """Open a directory below a held directory without using its path again."""
    absolute_root = Path(os.path.abspath(root_path))
    absolute_path = Path(os.path.abspath(path))
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise OSError(f"{label} is outside its held root: {path}") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.dup(root_fd)
    try:
        for component in relative.parts:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                if exc.errno in {
                    getattr(os, "ELOOP", 40),
                    getattr(os, "ENOTDIR", 20),
                }:
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
    label: str = "checkpoint delta path",
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise OSError("checkpoint delta requires O_NOFOLLOW")
    if dir_fd is None:
        _reject_symlink_path(Path(path), "checkpoint delta path")
    try:
        file_descriptor = os.open(path, flags | nofollow, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in {getattr(os, "ELOOP", 40), 40}:
            raise OSError(f"{label} is a symbolic link: {path}") from exc
        raise
    if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
        os.close(file_descriptor)
        raise OSError(f"{label} is not a regular file: {path}")
    return file_descriptor


def sha256_file(path: Path, *, parent_fd: int | None = None) -> str:
    digest = hashlib.sha256()
    owns_parent = parent_fd is None
    if owns_parent:
        parent_fd = _open_directory_nofollow(
            path.parent,
            create=False,
            label="checkpoint file parent",
        )
    try:
        with os.fdopen(
            _open_nofollow(
                path.name,
                os.O_RDONLY,
                dir_fd=parent_fd,
                label="checkpoint file",
            ),
            "rb",
        ) as source:
            while chunk := source.read(IO_CHUNK):
                digest.update(chunk)
    finally:
        if owns_parent:
            assert parent_fd is not None
            os.close(parent_fd)
    return digest.hexdigest()


def _read_json_entry(parent_fd: int, name: str) -> object:
    with os.fdopen(
        _open_nofollow(name, os.O_RDONLY, dir_fd=parent_fd, label="checkpoint JSON file"),
        "r",
    ) as source:
        return json.load(source)


def _write_json_atomic(
    parent_fd: int,
    temporary_name: str,
    destination_name: str,
    value: object,
) -> None:
    temporary_fd = _open_nofollow(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(temporary_fd, "w") as output:
            json.dump(value, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        existing = os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(existing.st_mode):
            raise OSError(f"checkpoint delta manifest is a symbolic link: {destination_name}")
    except FileNotFoundError:
        pass
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    try:
        os.replace(
            temporary_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise OSError("checkpoint delta inputs have different lengths")
    return np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8),
        np.frombuffer(right, dtype=np.uint8),
    ).tobytes()


def build_xor_delta_checkpoint(
    base_dir: Path,
    target_dir: Path,
    *,
    base_url: str,
    target_url: str,
) -> Path | None:
    """Build compressed byte-exact deltas when two checkpoints have the same shards."""
    base_index_path = base_dir / "model.safetensors.index.json"
    target_index_path = target_dir / "model.safetensors.index.json"
    base_fd = _open_directory_nofollow(
        base_dir,
        create=False,
        label="base checkpoint directory",
    )
    try:
        target_fd = _open_directory_nofollow(
            target_dir,
            create=False,
            label="target checkpoint directory",
        )
    except BaseException:
        os.close(base_fd)
        raise
    try:
        try:
            base_index = _read_json_entry(base_fd, base_index_path.name)
            target_index = _read_json_entry(target_fd, target_index_path.name)
        except FileNotFoundError:
            return None
        if not isinstance(base_index, dict) or not isinstance(target_index, dict):
            return None
        base_shards = sorted(set(base_index.get("weight_map", {}).values()))
        target_shards = sorted(set(target_index.get("weight_map", {}).values()))
        if (
            not target_shards
            or base_shards != target_shards
            or any(
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".safetensors")
                for filename in target_shards
            )
        ):
            return None

        try:
            existing = _read_json_entry(target_fd, DELTA_MANIFEST)
        except (FileNotFoundError, ValueError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("base_url") == base_url
            and existing.get("target_url") == target_url
        ):
            return target_dir / DELTA_MANIFEST

        files = []
        for filename in target_shards:
            base_input = target_input = temporary_fd = -1
            temporary_created = False
            try:
                base_input = _open_nofollow(filename, os.O_RDONLY, dir_fd=base_fd)
                target_input = _open_nofollow(filename, os.O_RDONLY, dir_fd=target_fd)
            except FileNotFoundError:
                for descriptor in (base_input, target_input):
                    if descriptor >= 0:
                        os.close(descriptor)
                return None
            except BaseException:
                for descriptor in (base_input, target_input):
                    if descriptor >= 0:
                        os.close(descriptor)
                raise
            try:
                base_size = os.fstat(base_input).st_size
                target_size = os.fstat(target_input).st_size
                if base_size != target_size:
                    return None
                delta_name = f"{filename}.xor.zlib"
                temporary_name = f"{delta_name}.partial"
                compressor = zlib.compressobj(level=1)
                target_digest = hashlib.sha256()
                temporary_fd = _open_nofollow(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    dir_fd=target_fd,
                )
                temporary_created = True
                with (
                    os.fdopen(base_input, "rb") as base,
                    os.fdopen(target_input, "rb") as target,
                    os.fdopen(temporary_fd, "wb") as output,
                ):
                    base_input = target_input = temporary_fd = -1
                    while target_chunk := target.read(IO_CHUNK):
                        base_chunk = base.read(len(target_chunk))
                        if len(base_chunk) != len(target_chunk):
                            raise OSError(f"base checkpoint ended early: {filename}")
                        target_digest.update(target_chunk)
                        output.write(compressor.compress(_xor_bytes(base_chunk, target_chunk)))
                    if base.read(1):
                        raise OSError(f"base checkpoint is longer than target: {filename}")
                    output.write(compressor.flush())
                try:
                    os.replace(
                        temporary_name,
                        delta_name,
                        src_dir_fd=target_fd,
                        dst_dir_fd=target_fd,
                    )
                except BaseException:
                    try:
                        os.unlink(temporary_name, dir_fd=target_fd)
                    except FileNotFoundError:
                        pass
                    raise
                finally:
                    for descriptor in (base_input, target_input, temporary_fd):
                        if descriptor >= 0:
                            os.close(descriptor)
                files.append(
                    {
                        "filename": filename,
                        "delta": delta_name,
                        "size": target_size,
                        "sha256": target_digest.hexdigest(),
                    }
                )
            except BaseException:
                for descriptor in (base_input, target_input, temporary_fd):
                    if descriptor >= 0:
                        os.close(descriptor)
                if temporary_created:
                    try:
                        os.unlink(temporary_name, dir_fd=target_fd)
                    except FileNotFoundError:
                        pass
                raise

        manifest = {
            "version": 1,
            "base_url": base_url,
            "target_url": target_url,
            "index_sha256": sha256_file(target_index_path, parent_fd=target_fd),
            "files": files,
        }
        _write_json_atomic(
            target_fd,
            f"{DELTA_MANIFEST}.partial",
            DELTA_MANIFEST,
            manifest,
        )
        return target_dir / DELTA_MANIFEST
    finally:
        os.close(target_fd)
        os.close(base_fd)


def apply_xor_delta_file(
    base_path: Path,
    delta_path: Path,
    target_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Reconstruct one target file without materializing an uncompressed delta."""
    temporary_name = f"{target_path.name}.assembling"
    base_parent_fd = _open_directory_nofollow(
        base_path.parent,
        create=False,
        label="checkpoint base parent",
    )
    common_root_path = Path(
        os.path.commonpath(
            (
                os.path.abspath(delta_path.parent),
                os.path.abspath(target_path.parent),
            )
        )
    )
    try:
        common_root_fd = _open_directory_nofollow(
            common_root_path,
            create=False,
            label="checkpoint delta and target root",
        )
    except BaseException:
        os.close(base_parent_fd)
        raise
    delta_parent_fd = target_parent_fd = -1
    separate_parent_fds = False
    try:
        target_parent_fd = _open_directory_below(
            common_root_fd,
            common_root_path,
            target_path.parent,
            label="checkpoint target parent",
        )
    except BaseException:
        os.close(common_root_fd)
        os.close(base_parent_fd)
        raise
    try:
        if delta_path.parent == target_path.parent:
            delta_parent_fd = target_parent_fd
        else:
            separate_parent_fds = True
            delta_parent_fd = _open_directory_below(
                common_root_fd,
                common_root_path,
                delta_path.parent,
                label="checkpoint delta parent",
            )
    except BaseException:
        if target_parent_fd >= 0:
            os.close(target_parent_fd)
        if delta_parent_fd >= 0 and separate_parent_fds:
            os.close(delta_parent_fd)
        os.close(common_root_fd)
        os.close(base_parent_fd)
        raise
    base_input = delta_input = temporary_output = -1
    temporary_created = False
    try:
        base_input = _open_nofollow(
            base_path.name,
            os.O_RDONLY,
            dir_fd=base_parent_fd,
            label="checkpoint base file",
        )
        delta_input = _open_nofollow(
            delta_path.name,
            os.O_RDONLY,
            dir_fd=delta_parent_fd,
            label="checkpoint delta file",
        )
        temporary_output = _open_nofollow(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            dir_fd=target_parent_fd,
            label="checkpoint temporary file",
        )
        temporary_created = True
        decompressor = zlib.decompressobj()
        digest = hashlib.sha256()
        written = 0

        with (
            os.fdopen(base_input, "rb") as base,
            os.fdopen(delta_input, "rb") as delta,
            os.fdopen(temporary_output, "wb") as target,
        ):
            base_input = delta_input = temporary_output = -1

            def consume(delta_chunk: bytes) -> None:
                nonlocal written
                base_chunk = base.read(len(delta_chunk))
                if len(base_chunk) != len(delta_chunk):
                    raise OSError(f"delta is longer than base file: {base_path.name}")
                target_chunk = _xor_bytes(base_chunk, delta_chunk)
                target.write(target_chunk)
                digest.update(target_chunk)
                written += len(target_chunk)

            while compressed := delta.read(1024 * 1024):
                pending = compressed
                while pending:
                    expanded = decompressor.decompress(pending, IO_CHUNK)
                    pending = decompressor.unconsumed_tail
                    if expanded:
                        consume(expanded)
                    elif not pending:
                        break
                while True:
                    expanded = decompressor.decompress(b"", IO_CHUNK)
                    if not expanded:
                        break
                    consume(expanded)

            tail = decompressor.flush()
            if tail:
                consume(tail)
            if base.read(1):
                raise OSError(f"delta is shorter than base file: {base_path.name}")

        if not decompressor.eof:
            raise OSError(f"incomplete compressed delta: {delta_path.name}")
        if written != expected_size:
            raise OSError(
                f"reconstructed size mismatch for {target_path.name}: {written} != {expected_size}"
            )
        if digest.hexdigest() != expected_sha256:
            raise OSError(f"reconstructed checksum mismatch for {target_path.name}")
        try:
            existing = os.stat(
                target_path.name,
                dir_fd=target_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise OSError(f"checkpoint target file is a symbolic link: {target_path}")
        os.replace(
            temporary_name,
            target_path.name,
            src_dir_fd=target_parent_fd,
            dst_dir_fd=target_parent_fd,
        )
    except BaseException:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=target_parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        for descriptor in (base_input, delta_input, temporary_output):
            if descriptor >= 0:
                os.close(descriptor)
        os.close(target_parent_fd)
        if separate_parent_fds:
            os.close(delta_parent_fd)
        os.close(common_root_fd)
        os.close(base_parent_fd)
