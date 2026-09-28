"""Fail-closed retention for consumed trainer rollout batches.

The action removes only exact, delivered ``rollouts/step_N`` directories from
one active run.  Evidence is supplied by the operator and must agree across
all configured trainer nodes.  This module does not discover hosts or make
network calls; the training_ops CLI performs the bounded host fan-out.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping


class RetentionError(RuntimeError):
    """A retention input or safety check is invalid."""


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STEP = re.compile(r"step_([0-9]+)\Z")
_RUN = re.compile(r"[A-Za-z0-9_-]+\Z")
_NODE = re.compile(r"n[0-9]+\Z")
_BATCH_NAMES = frozenset({"train_rollouts.bin", "train_rollouts.bin.delivered"})
_LOCK_NAMES = frozenset({
    ".batch-sync.lock",
    ".rollout-pull.lock",
    ".microbatch-sync.lock",
})


def _fail(message: str) -> None:
    raise RetentionError(message)


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        _fail(f"{label} cannot be inspected: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail(f"{label} is not a regular file")
    return info


def _directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        _fail(f"{label} cannot be inspected: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail(f"{label} is not a real directory")
    return info


def _canonical_run(path_value: str | Path, run_name: str | None = None) -> Path:
    try:
        path = Path(path_value)
    except TypeError as error:
        _fail("run directory is invalid")
        raise AssertionError from error
    if not path.is_absolute() or path != Path(os.path.normpath(path)) or ".." in path.parts:
        _fail("run directory must be a canonical absolute path")
    if "n79" in path.parts:
        _fail("n79 is not an allowed retention target")
    if run_name is not None and path.name != f"run_{run_name}":
        _fail("run directory is not the exact active run")
    if not re.fullmatch(r"run_[A-Za-z0-9_-]+", path.name):
        _fail("run directory must be one exact run_<name> directory")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            _fail(f"run directory ancestor is unavailable: {current}: {error}")
        if stat.S_ISLNK(info.st_mode):
            _fail(f"run directory ancestor is a symlink: {current}")
    _directory(path, "run directory")
    return path


def _digest(path: Path, label: str) -> str:
    _regular(path, label)
    result = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
    except OSError as error:
        _fail(f"{label} cannot be read: {error}")
    return result.hexdigest()


def _safe_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} is invalid")
    return value


def _load_evidence(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            _fail(f"retention evidence is invalid: {error}")
    if not isinstance(value, Mapping):
        _fail("retention evidence is not an object")
    expected = {"version", "run_name", "active_step", "required_steps", "nodes"}
    if set(value) != expected or value["version"] != 1:
        _fail("retention evidence fields are invalid")
    run_name = value["run_name"]
    if not isinstance(run_name, str) or _RUN.fullmatch(run_name) is None:
        _fail("retention evidence run name is invalid")
    active_step = _safe_int(value["active_step"], "retention active step")
    required = value["required_steps"]
    if not isinstance(required, list) or any(type(item) is not int or item < 0 for item in required):
        _fail("retention required steps are invalid")
    required_steps = sorted(set(required))
    if required_steps != required or active_step not in required_steps:
        _fail("retention required steps are not canonical or omit the active step")
    nodes = value["nodes"]
    if not isinstance(nodes, Mapping) or not nodes:
        _fail("retention node evidence is empty")
    result_nodes: dict[str, dict[str, Any]] = {}
    content_identity: dict[int, tuple[int, str, str]] = {}
    node_step_sets: dict[str, set[int]] = {}
    for node, raw in nodes.items():
        if not isinstance(node, str) or _NODE.fullmatch(node) is None or node == "n79":
            _fail("retention node identity is invalid")
        if not isinstance(raw, Mapping):
            _fail(f"retention evidence for {node} is not an object")
        expected_node = {"run_dir", "owner_uid", "owner_gid", "active_step", "required_steps", "batches"}
        if set(raw) != expected_node:
            _fail(f"retention evidence fields for {node} are invalid")
        node_run_dir = raw["run_dir"]
        if not isinstance(node_run_dir, str):
            _fail(f"retention run directory for {node} is invalid")
        _canonical_run(node_run_dir, run_name)
        owner_uid = _safe_int(raw["owner_uid"], f"retention owner uid for {node}")
        owner_gid = _safe_int(raw["owner_gid"], f"retention owner gid for {node}")
        if raw["active_step"] != active_step or raw["required_steps"] != required_steps:
            _fail(f"retention resume evidence disagrees on {node}")
        batches = raw["batches"]
        if not isinstance(batches, list) or not batches:
            _fail(f"retention batch evidence is empty on {node}")
        seen: set[int] = set()
        normalized: list[dict[str, Any]] = []
        for item in batches:
            fields = {
                "step", "size", "sha256", "delivered_sha256", "owner_uid", "owner_gid",
                "device", "inode", "mtime_ns", "dir_device", "dir_inode", "dir_mtime_ns",
            }
            if not isinstance(item, Mapping) or set(item) != fields:
                _fail(f"retention batch evidence fields are invalid on {node}")
            step = _safe_int(item["step"], "retention batch step")
            if step in seen:
                _fail(f"retention batch step is duplicated on {node}")
            seen.add(step)
            size = _safe_int(item["size"], "retention batch size", minimum=1)
            sha = item["sha256"]
            delivered_sha = item["delivered_sha256"]
            if not isinstance(sha, str) or _DIGEST.fullmatch(sha) is None or delivered_sha != sha:
                _fail(f"retention batch delivery identity is invalid on {node}")
            if item["owner_uid"] != owner_uid or item["owner_gid"] != owner_gid:
                _fail(f"retention batch ownership disagrees on {node}")
            values = {name: _safe_int(item[name], f"retention batch {name}") for name in (
                "device", "inode", "mtime_ns", "dir_device", "dir_inode", "dir_mtime_ns"
            )}
            normalized_item = {
                "step": step, "size": size, "sha256": sha, "delivered_sha256": sha,
                "owner_uid": owner_uid, "owner_gid": owner_gid, **values,
            }
            normalized.append(normalized_item)
            identity = (size, sha, sha)
            prior = content_identity.get(step)
            if prior is not None and prior != identity:
                _fail(f"retention nodes disagree on batch step {step}")
            content_identity[step] = identity
        normalized.sort(key=lambda item: item["step"])
        if not set(required_steps) <= seen:
            _fail(f"retention resume-required batch is missing on {node}")
        node_step_sets[node] = seen
        result_nodes[node] = {"run_dir": node_run_dir, "owner_uid": owner_uid, "owner_gid": owner_gid,
                              "active_step": active_step, "required_steps": required_steps, "batches": normalized}
    if len({tuple(sorted(steps)) for steps in node_step_sets.values()}) != 1:
        _fail("retention nodes disagree on the complete step inventory")
    return {"version": 1, "run_name": run_name, "active_step": active_step,
            "required_steps": required_steps, "nodes": result_nodes}


def validate_evidence(value: Mapping[str, Any] | str | Path, *, nodes: list[str] | None = None) -> dict[str, Any]:
    """Validate authoritative evidence and, when supplied, its exact node set."""
    result = _load_evidence(value)
    if nodes is not None and set(result["nodes"]) != set(nodes):
        _fail("retention evidence does not cover exactly the configured trainers")
    return result


def _tree_bytes(path: Path) -> int:
    total = 0
    try:
        for child in path.iterdir():
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode):
                _fail(f"rollout tree contains a symlink: {child}")
            if stat.S_ISDIR(info.st_mode):
                total += _tree_bytes(child)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
            else:
                _fail(f"rollout tree contains a special file: {child}")
    except OSError as error:
        _fail(f"rollout tree cannot be inspected: {error}")
    return total


def _open_fd_targets(root: Path, *, ignore_pid: int | None = None) -> list[str]:
    targets: list[str] = []
    proc = Path("/proc")
    try:
        processes = list(proc.glob("[0-9]*"))
    except OSError as error:
        _fail(f"open-file evidence cannot be inspected: {error}")
    root_text = str(root) + os.sep
    for process in processes:
        if ignore_pid is not None and process.name == str(ignore_pid):
            continue
        fd_dir = process / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError):
            # The CLI runs this module with sudo on trainer hosts. A
            # non-privileged local fixture can have unrelated root processes;
            # their descriptors are not observable in this disposable check.
            continue
        except OSError as error:
            _fail(f"open-file evidence cannot inspect {fd_dir}: {error}")
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == str(root) or target.startswith(root_text):
                targets.append(f"{process.name}:{fd.name}:{target}")
    return targets


def _lock_is_busy(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        _fail(f"lock cannot be inspected: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail(f"lock is not a regular file: {path}")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError as error:
        _fail(f"lock cannot be opened: {error}")
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)
    return False


def _check_locks(run_dir: Path) -> None:
    candidates: set[Path] = set()
    for root in (run_dir, run_dir / "rollouts"):
        try:
            candidates.update(
                path for path in root.rglob("*")
                if path.name in _LOCK_NAMES or path.name == ".lock" or path.name.endswith(".lock")
            )
        except OSError as error:
            _fail(f"lock inventory cannot be inspected: {error}")
    for name in _LOCK_NAMES:
        candidates.add(run_dir / name)
    for path in sorted(candidates):
        if path == run_dir / ".rollout-retention.lock":
            continue
        if _lock_is_busy(path):
            _fail(f"rollout writer or consumer lock is open: {path}")


def _batch_record(path: Path, item: Mapping[str, Any], owner_uid: int, owner_gid: int) -> dict[str, Any]:
    step = int(item["step"])
    info = _directory(path, f"rollout step {step}")
    try:
        entries = {child.name for child in path.iterdir()}
    except OSError as error:
        _fail(f"rollout step {step} cannot be listed: {error}")
    if entries != _BATCH_NAMES:
        unknown = sorted(entries - _BATCH_NAMES)
        _fail(f"rollout step {step} has partial or unknown files: {unknown}")
    batch = path / "train_rollouts.bin"
    delivered = path / "train_rollouts.bin.delivered"
    batch_info = _regular(batch, f"rollout batch {step}")
    delivered_info = _regular(delivered, f"rollout delivery marker {step}")
    if batch_info.st_uid != owner_uid or batch_info.st_gid != owner_gid or delivered_info.st_uid != owner_uid or delivered_info.st_gid != owner_gid:
        _fail(f"rollout step {step} ownership changed")
    digest = _digest(batch, f"rollout batch {step}")
    try:
        marker = delivered.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        _fail(f"rollout delivery marker {step} is invalid: {error}")
    if marker != digest + "\n":
        _fail(f"rollout step {step} has missing or mismatched delivery evidence")
    if (info.st_dev, info.st_ino, info.st_mtime_ns) != (item["dir_device"], item["dir_inode"], item["dir_mtime_ns"]):
        _fail(f"rollout step {step} directory identity changed")
    if batch_info.st_size != item["size"] or digest != item["sha256"]:
        _fail(f"rollout step {step} content identity changed")
    if (batch_info.st_dev, batch_info.st_ino, batch_info.st_mtime_ns) != (item["device"], item["inode"], item["mtime_ns"]):
        _fail(f"rollout step {step} batch identity changed")
    if _open_fd_targets(path):
        _fail(f"rollout step {step} has an open file descriptor")
    return {
        "step": step,
        "bytes": _tree_bytes(path),
        "sha256": digest,
        "path": str(path),
        "size": batch_info.st_size,
        "device": batch_info.st_dev,
        "inode": batch_info.st_ino,
        "mtime_ns": batch_info.st_mtime_ns,
        "dir_device": info.st_dev,
        "dir_inode": info.st_ino,
        "dir_mtime_ns": info.st_mtime_ns,
    }


def inventory(run_dir: str | Path, evidence: Mapping[str, Any] | str | Path, *, node: str | None = None) -> dict[str, Any]:
    """Return a strict, dry-run inventory for one trainer node."""
    loaded = _load_evidence(evidence)
    nodes = loaded["nodes"]
    if node is None:
        if len(nodes) != 1:
            _fail("node is required when evidence contains multiple trainers")
        node = next(iter(nodes))
    if node not in nodes:
        _fail(f"retention evidence has no node {node}")
    record = nodes[node]
    path = _canonical_run(run_dir, loaded["run_name"])
    if str(path) != record["run_dir"]:
        _fail(f"run directory disagrees with authoritative evidence on {node}")
    run_info = _directory(path, "run directory")
    if run_info.st_uid != record["owner_uid"] or run_info.st_gid != record["owner_gid"]:
        _fail(f"run directory ownership differs on {node}")
    rollouts = path / "rollouts"
    _directory(rollouts, "rollout directory")
    _check_locks(path)
    if _open_fd_targets(rollouts):
        _fail("rollout tree has an open writer or consumer file descriptor")
    expected = {int(item["step"]): item for item in record["batches"]}
    observed_steps: list[int] = []
    try:
        entries = sorted(rollouts.iterdir(), key=lambda item: item.name)
    except OSError as error:
        _fail(f"rollout directory cannot be listed: {error}")
    for entry in entries:
        if entry.is_symlink():
            _fail(f"rollout entry is a symlink: {entry}")
        match = _STEP.fullmatch(entry.name)
        if match is None:
            _fail(f"rollout directory has an unknown entry: {entry.name}")
        step = int(match.group(1))
        if step not in expected:
            _fail(f"rollout step {step} is absent from authoritative evidence")
        _batch_record(entry, expected[step], record["owner_uid"], record["owner_gid"])
        observed_steps.append(step)
    if observed_steps != sorted(expected):
        _fail(f"rollout step inventory disagrees on {node}")
    latest = sorted(expected)[-2:]
    protected = set(latest) | set(loaded["required_steps"]) | {
        step for step in expected if step >= loaded["active_step"]
    }
    selected: list[dict[str, Any]] = []
    for step in sorted(expected):
        if step in protected:
            continue
        item = _batch_record(rollouts / f"step_{step}", expected[step], record["owner_uid"], record["owner_gid"])
        selected.append(item)
    selected_bytes = sum(item["bytes"] for item in selected)
    return {
        "node": node,
        "run_name": loaded["run_name"],
        "run_dir": str(path),
        "active_step": loaded["active_step"],
        "protected_steps": sorted(protected),
        "selected": selected,
        "selected_steps": [item["step"] for item in selected],
        "selected_bytes": selected_bytes,
        "exact_bytes": selected_bytes,
        "all_steps": sorted(expected),
        "dry_run": True,
    }


def _fsync(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        _fail(f"retention directory cannot be opened for fsync: {error}")
    try:
        os.fsync(descriptor)
    except OSError as error:
        _fail(f"retention directory cannot be fsynced: {error}")
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _retention_lock(run_dir: Path):
    lock_path = run_dir / ".rollout-retention.lock"
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        _fail(f"retention lock cannot be opened: {error}")
    lock_identity = os.fstat(descriptor)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("another rollout retention action is open")
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            try:
                current = lock_path.lstat()
                if (current.st_dev, current.st_ino) == (lock_identity.st_dev, lock_identity.st_ino):
                    lock_path.unlink()
            except FileNotFoundError:
                pass


def _read_fd(fd: int) -> bytes:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    except OSError as error:
        _fail(f"quarantined file cannot be read: {error}")


def _open_file_at(directory_fd: int, name: str, label: str) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        _fail(f"{label} cannot be inspected: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(f"{label} is not a regular file")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        after = os.fstat(descriptor)
    except OSError as error:
        _fail(f"{label} cannot be opened: {error}")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        os.close(descriptor)
        _fail(f"{label} identity changed while opening")
    return descriptor, after


def _final_quarantine_check(
    quarantine: Path,
    expected: Mapping[str, Any],
    owner_uid: int,
    owner_gid: int,
) -> tuple[int, int, int, list[int]]:
    """Hold directory and file descriptors through final removal."""
    if _open_fd_targets(quarantine, ignore_pid=os.getpid()):
        _fail("quarantined rollout has an open writer or consumer descriptor")
    try:
        directory_fd = os.open(quarantine, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        _fail(f"quarantine cannot be opened: {error}")
    opened: list[int] = [directory_fd]
    try:
        directory_info = os.fstat(directory_fd)
        if (directory_info.st_dev, directory_info.st_ino, directory_info.st_mtime_ns) != (
            expected["dir_device"], expected["dir_inode"], expected["dir_mtime_ns"]
        ):
            _fail("quarantined rollout directory identity changed")
        try:
            names = set(os.listdir(directory_fd))
        except OSError as error:
            _fail(f"quarantined rollout cannot be listed: {error}")
        if names != _BATCH_NAMES:
            _fail(f"quarantined rollout has partial or unknown files: {sorted(names - _BATCH_NAMES)}")
        batch_fd, batch_info = _open_file_at(directory_fd, "train_rollouts.bin", "quarantined rollout batch")
        marker_fd, marker_info = _open_file_at(directory_fd, "train_rollouts.bin.delivered", "quarantined delivery marker")
        opened.extend((batch_fd, marker_fd))
        if any(info.st_uid != owner_uid or info.st_gid != owner_gid for info in (batch_info, marker_info)):
            _fail("quarantined rollout ownership changed")
        payload = _read_fd(batch_fd)
        digest = hashlib.sha256(payload).hexdigest()
        marker = _read_fd(marker_fd).decode("ascii")
        if marker != digest + "\n" or digest != expected["sha256"] or len(payload) != expected["size"]:
            _fail("quarantined rollout content or delivery identity changed")
        if (batch_info.st_dev, batch_info.st_ino, batch_info.st_mtime_ns) != (
            expected["device"], expected["inode"], expected["mtime_ns"]
        ):
            _fail("quarantined rollout batch identity changed")
        return directory_fd, batch_info.st_size + marker_info.st_size, batch_info.st_size, opened
    except UnicodeError as error:
        _fail(f"quarantined delivery marker is invalid: {error}")
    except Exception:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _validate_held_contents(opened: list[int], expected: Mapping[str, Any]) -> None:
    batch_fd, marker_fd = opened[1:]
    batch_info = os.fstat(batch_fd)
    marker_info = os.fstat(marker_fd)
    payload = _read_fd(batch_fd)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        marker = _read_fd(marker_fd).decode("ascii")
    except UnicodeError as error:
        _fail(f"quarantined delivery marker is invalid: {error}")
    if marker != digest + "\n" or digest != expected["sha256"] or len(payload) != expected["size"]:
        _fail("quarantined rollout content or delivery identity changed at deletion")
    if (batch_info.st_dev, batch_info.st_ino, batch_info.st_size, batch_info.st_mtime_ns) != (
        expected["device"], expected["inode"], expected["size"], expected["mtime_ns"]
    ) or marker_info.st_size != len(marker.encode("ascii")):
        _fail("quarantined rollout identity changed at deletion")


def _remove_tree_at(directory_fd: int, opened_files: list[int], expected: Mapping[str, Any]) -> int:
    """Remove one already validated tree using its held directory descriptor."""
    directory_info = os.fstat(directory_fd)
    if (directory_info.st_dev, directory_info.st_ino) != (expected["dir_device"], expected["dir_inode"]):
        _fail("quarantined rollout directory identity changed at deletion")
    _validate_held_contents([directory_fd, *opened_files], expected)
    expected_fds = {fd: os.fstat(fd) for fd in opened_files}
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        _fail(f"quarantined rollout cannot be listed at deletion: {error}")
    for name in names:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            _fail(f"quarantined entry cannot be inspected at deletion: {error}")
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            _fail(f"quarantined entry changed at deletion: {name}")
        matching = [fd for fd, held in expected_fds.items() if (held.st_dev, held.st_ino) == (info.st_dev, info.st_ino)]
        if not matching:
            _fail(f"quarantined entry identity changed at deletion: {name}")
        held = os.fstat(matching[0])
        if (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns) != (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        ):
            _fail(f"quarantined entry contents changed at deletion: {name}")
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError as error:
            _fail(f"quarantined entry cannot be removed: {error}")
    try:
        if os.listdir(directory_fd):
            _fail("quarantined rollout gained a file at deletion")
    except OSError as error:
        _fail(f"quarantined rollout cannot be checked after deletion: {error}")
    return sum(info.st_size for info in expected_fds.values())


def _delete_quarantine(
    quarantine: Path,
    expected: Mapping[str, Any],
    owner_uid: int,
    owner_gid: int,
) -> int:
    directory_fd, bytes_used, _batch_bytes, opened = _final_quarantine_check(
        quarantine, expected, owner_uid, owner_gid
    )
    try:
        parent_fd = os.open(quarantine.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        for descriptor in reversed(opened):
            os.close(descriptor)
        _fail(f"quarantine parent cannot be opened: {error}")
    try:
        # Repeat the immutable check immediately before unlinking. The held
        # descriptors make the final identity and content checks independent
        # of pathname replacement.
        if _open_fd_targets(quarantine, ignore_pid=os.getpid()):
            _fail("quarantined rollout has an open writer or consumer descriptor")
        actual = _remove_tree_at(directory_fd, opened[1:], expected)
        if actual != bytes_used:
            _fail("actual freed bytes differ from the validated quarantine")
        try:
            os.rmdir(quarantine.name, dir_fd=parent_fd)
        except OSError as error:
            _fail(f"quarantined rollout directory cannot be removed: {error}")
        return actual
    finally:
        os.close(parent_fd)
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass


def cleanup(run_dir: str | Path, evidence: Mapping[str, Any] | str | Path, *, node: str | None = None, dry_run: bool = True) -> dict[str, Any]:
    """Inventory or atomically quarantine and remove old consumed batches."""
    initial = inventory(run_dir, evidence, node=node)
    if dry_run or not initial["selected"]:
        return {**initial, "dry_run": True, "removed": [], "freed_bytes": 0}
    loaded = _load_evidence(evidence)
    path = _canonical_run(run_dir, loaded["run_name"])
    removed: list[dict[str, Any]] = []
    freed = 0
    with _retention_lock(path):
        fresh = inventory(path, loaded, node=node)
        for item in fresh["selected"]:
            source = Path(item["path"])
            if source.parent != path / "rollouts" or not _STEP.fullmatch(source.name):
                _fail("retention candidate escaped the rollout directory")
            before = source.lstat()
            expected_item = next(value for value in loaded["nodes"][fresh["node"]]["batches"] if value["step"] == item["step"])
            if (before.st_dev, before.st_ino, before.st_mtime_ns) != (expected_item["dir_device"], expected_item["dir_inode"], expected_item["dir_mtime_ns"]):
                _fail(f"rollout step {item['step']} changed before quarantine")
            quarantine = source.parent / f".retention-quarantine-{uuid.uuid4().hex}"
            try:
                os.rename(source, quarantine)
            except OSError as error:
                _fail(f"rollout step {item['step']} changed before atomic quarantine: {error}")
            _fsync(source.parent)
            try:
                _check_locks(path)
                removed_bytes = _delete_quarantine(
                    quarantine,
                    expected_item,
                    loaded["nodes"][fresh["node"]]["owner_uid"],
                    loaded["nodes"][fresh["node"]]["owner_gid"],
                )
                if removed_bytes != item["bytes"]:
                    _fail(f"rollout step {item['step']} actual bytes differ from preview")
            except Exception:
                # Keep the quarantined tree for manual recovery when any
                # identity check fails. Never restore or delete uncertain data.
                raise
            _fsync(source.parent)
            removed.append(item)
            freed += removed_bytes
    return {**initial, "dry_run": False, "removed": removed, "freed_bytes": freed,
            "exact_bytes": freed, "selected_steps": [item["step"] for item in removed]}


def build_node_evidence(run_dir: str | Path, *, node: str, run_name: str, active_step: int, required_steps: list[int]) -> dict[str, Any]:
    """Build one node evidence record for an operator's authoritative snapshot."""
    if _NODE.fullmatch(node) is None or node == "n79" or _RUN.fullmatch(run_name) is None:
        _fail("node or run name is invalid")
    path = _canonical_run(run_dir, run_name)
    _safe_int(active_step, "active step")
    required = sorted(set(required_steps))
    if required != required_steps or active_step not in required:
        _fail("required steps are not canonical or omit active step")
    info = _directory(path, "run directory")
    rollouts = path / "rollouts"
    _directory(rollouts, "rollout directory")
    batches = []
    for entry in sorted(rollouts.iterdir(), key=lambda item: item.name):
        match = _STEP.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            _fail(f"rollout inventory has an invalid entry: {entry}")
        step = int(match.group(1))
        batch = entry / "train_rollouts.bin"
        marker = entry / "train_rollouts.bin.delivered"
        batch_info = _regular(batch, f"rollout batch {step}")
        _regular(marker, f"rollout delivery marker {step}")
        digest = _digest(batch, f"rollout batch {step}")
        batches.append({
            "step": step, "size": batch_info.st_size, "sha256": digest, "delivered_sha256": digest,
            "owner_uid": info.st_uid, "owner_gid": info.st_gid,
            "device": batch_info.st_dev, "inode": batch_info.st_ino, "mtime_ns": batch_info.st_mtime_ns,
            "dir_device": entry.stat(follow_symlinks=False).st_dev,
            "dir_inode": entry.stat(follow_symlinks=False).st_ino,
            "dir_mtime_ns": entry.stat(follow_symlinks=False).st_mtime_ns,
        })
    return {"run_dir": str(path), "owner_uid": info.st_uid, "owner_gid": info.st_gid,
            "active_step": active_step, "required_steps": required, "batches": batches}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="retain consumed trainer rollout batches")
    parser.add_argument("run_dir")
    evidence_group = parser.add_mutually_exclusive_group(required=True)
    evidence_group.add_argument("--evidence")
    evidence_group.add_argument("--evidence-b64")
    parser.add_argument("--node", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        evidence = args.evidence
        if args.evidence_b64 is not None:
            evidence = json.loads(base64.b64decode(args.evidence_b64).decode("utf-8"))
        result = cleanup(args.run_dir, evidence, node=args.node, dry_run=not args.apply)
    except (RetentionError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
