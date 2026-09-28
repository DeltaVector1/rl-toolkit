"""Fail-closed retention for one trainer adapter broadcast directory.

The caller supplies a snapshot of the H200 loaded policy and the durable
receiver acknowledgements.  This module does not contact a host.  It only
removes entries after validating the local publication again under a lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class RetentionError(RuntimeError):
    """A retention input or safety check is invalid."""


_DIGEST = re.compile(r"[0-9a-f]{64}")
_STEP = re.compile(r"step_([0-9]+)\Z")
_IDENTITY_KEYS = frozenset(
    {
        "step",
        "source_url",
        "slot",
        "target_sha256",
        "target_size",
        "config_sha256",
        "config_size",
        "server_instance",
    }
)
_ACK_KEYS = frozenset(
    {
        "version",
        "run_name",
        "step",
        "source_url",
        "target_sha256",
        "target_size",
        "config_sha256",
        "config_size",
        "slot",
        "receivers",
    }
)
_REQUIRED_FILES = frozenset(
    {
        "STABLE",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.safetensors.sha256",
        "adapter_manifest.json",
    }
)
_WRITER_MARKERS = frozenset({"WRITING", ".writing", ".lock"})
_PROTECTED_NAMES = frozenset(
    {
        "checkpoints",
        "rollouts",
        ".incoming",
        "control",
        "hf-cache",
        "model",
        "trinity",
        "ZiphuAI",
        "nv-community_GLM-5.2-NVFP4",
    }
)


@dataclass(frozen=True)
class _Stable:
    path: Path
    step: int
    identity: dict[str, Any]
    stat_key: tuple[int, int, int, int]
    bytes: int
    server_instances: tuple[str, ...]


@dataclass(frozen=True)
class _Partial:
    path: Path
    step: int | None
    bytes: int
    newest_mtime_ns: int
    reason: str
    stat_key: tuple[int, int, int]


def _fail(message: str) -> None:
    raise RetentionError(message)


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        _fail(f"{label} cannot be inspected: {error}")
    if not stat.S_ISREG(info.st_mode):
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


def _safe_run_dir(value: str | Path) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        _fail("run directory is invalid")
        raise AssertionError from error
    if not path.is_absolute() or path != Path(os.path.normpath(path)) or ".." in path.parts:
        _fail("run directory must be a canonical absolute path")
    if "n79" in path.parts:
        _fail("n79 is not an allowed retention target")
    if path.name == "" or not re.fullmatch(r"run_[A-Za-z0-9_-]+", path.name):
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


def _sha256(path: Path, label: str) -> str:
    _regular(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail(f"{label} cannot be read: {error}")
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is invalid: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _identity(value: object, label: str, *, step: int | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        _fail(f"{label} identity fields are invalid")
    result = dict(value)
    if type(result["step"]) is not int or result["step"] < 0 or (step is not None and result["step"] != step):
        _fail(f"{label} step identity is invalid")
    if not isinstance(result["source_url"], str) or not result["source_url"] or any(
        character.isspace() for character in result["source_url"]
    ):
        _fail(f"{label} source URL is invalid")
    if not isinstance(result["slot"], str) or not result["slot"] or any(
        character.isspace() for character in result["slot"]
    ):
        _fail(f"{label} slot is invalid")
    for name in ("target_sha256", "config_sha256"):
        if not isinstance(result[name], str) or _DIGEST.fullmatch(result[name]) is None:
            _fail(f"{label} {name} is invalid")
    if type(result["target_size"]) is not int or result["target_size"] <= 0:
        _fail(f"{label} target size is invalid")
    if type(result["config_size"]) is not int or result["config_size"] < 0:
        _fail(f"{label} config size is invalid")
    if not isinstance(result["server_instance"], str) or re.fullmatch(r"[0-9a-f]{32}", result["server_instance"]) is None:
        _fail(f"{label} server instance is invalid")
    return result


def _core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in _IDENTITY_KEYS if name != "server_instance"}


def _load_evidence(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            _fail(f"retention evidence is invalid: {error}")
    else:
        payload = value
    if not isinstance(payload, Mapping) or set(payload) != {"version", "run_name", "h200_loaded", "acknowledged"}:
        _fail("retention evidence fields are invalid")
    if payload["version"] != 1 or not isinstance(payload["run_name"], str) or not payload["run_name"]:
        _fail("retention evidence version or run name is invalid")
    loaded = _identity(payload["h200_loaded"], "H200 loaded")
    acknowledged = payload["acknowledged"]
    if not isinstance(acknowledged, list):
        _fail("retention acknowledgement list is invalid")
    records = []
    seen_steps = set()
    for index, item in enumerate(acknowledged):
        record = _identity(item, f"acknowledgement {index}")
        if record["step"] in seen_steps:
            _fail("retention acknowledgement steps are duplicated")
        seen_steps.add(record["step"])
        if record["step"] > loaded["step"]:
            _fail("retention acknowledgement is from a future step")
        records.append(record)
    return {"version": 1, "run_name": payload["run_name"], "h200_loaded": loaded, "acknowledged": records}


def _manifest_identity(path: Path, step: int) -> tuple[dict[str, Any], set[str]]:
    manifest = _json(path / "adapter_manifest.json", "adapter manifest")
    if manifest.get("version") not in {1, 2} or manifest.get("kind") != "lora-adapter":
        _fail(f"adapter manifest has an unsupported version: {path}")
    if manifest.get("target_step") != step:
        _fail(f"adapter manifest target differs: {path}")
    model = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    model_size = _regular(model, "adapter model").st_size
    config_size = _regular(config, "adapter config").st_size
    model_sha = _sha256(model, "adapter model")
    config_sha = _sha256(config, "adapter config")
    if manifest.get("target_size") != model_size or manifest.get("target_sha256") != model_sha:
        _fail(f"adapter manifest model identity differs: {path}")
    if manifest.get("config_size") != config_size or manifest.get("config_sha256") != config_sha:
        _fail(f"adapter manifest config identity differs: {path}")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        _fail(f"adapter manifest file list is invalid: {path}")
    expected_files = {
        "adapter_model.safetensors": {"size": model_size, "sha256": model_sha},
        "adapter_config.json": {"size": config_size, "sha256": config_sha},
    }
    observed = {}
    for item in files:
        if not isinstance(item, Mapping) or item.get("name") in observed:
            _fail(f"adapter manifest file list is invalid: {path}")
        name = item.get("name")
        if name not in expected_files or item.get("size") != expected_files[name]["size"] or item.get("sha256") != expected_files[name]["sha256"]:
            _fail(f"adapter manifest file identity differs: {path}")
        observed[name] = item
    if set(observed) != set(expected_files):
        _fail(f"adapter manifest file list is incomplete: {path}")
    sidecar = path / "adapter_model.safetensors.sha256"
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        _fail(f"adapter checksum sidecar is invalid: {error}")
    if sidecar_value != model_sha:
        _fail(f"adapter checksum sidecar differs: {path}")
    extras = set()
    chunks = manifest.get("chunks", [])
    if chunks is not None:
        if not isinstance(chunks, list):
            _fail(f"adapter manifest chunks are invalid: {path}")
        for item in chunks:
            if not isinstance(item, Mapping) or not isinstance(item.get("file"), str) or not item["file"] or "/" in item["file"] or "\\" in item["file"]:
                _fail(f"adapter manifest chunk name is invalid: {path}")
            extras.add(item["file"])
    return {
        "step": step,
        "source_url": "",
        "slot": "",
        "target_sha256": model_sha,
        "target_size": model_size,
        "config_sha256": config_sha,
        "config_size": config_size,
        "server_instance": "",
    }, extras


def _ack(path: Path, *, step: int, run_name: str, identity: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = _json(path / "receivers-ready.json", "receiver acknowledgement")
    if set(value) != _ACK_KEYS or value.get("version") != 1 or value.get("run_name") != run_name or value.get("step") != step:
        _fail(f"receiver acknowledgement fields are invalid: {path}")
    for name in ("target_sha256", "target_size", "config_sha256", "config_size"):
        if value.get(name) != identity[name]:
            _fail(f"receiver acknowledgement {name} differs: {path}")
    receivers = value.get("receivers")
    if not isinstance(receivers, list) or not receivers:
        _fail(f"receiver acknowledgement records are invalid: {path}")
    records = []
    endpoints = set()
    core = {
        name: identity[name]
        for name in ("step", "target_sha256", "target_size", "config_sha256", "config_size")
    }
    for item in receivers:
        if not isinstance(item, Mapping) or set(item) != {"endpoint", "receipt"}:
            _fail(f"receiver acknowledgement records are invalid: {path}")
        endpoint = item["endpoint"]
        if not isinstance(endpoint, str) or not endpoint or endpoint in endpoints:
            _fail(f"receiver acknowledgement endpoints are invalid: {path}")
        receipt = item["receipt"]
        if not isinstance(receipt, Mapping) or set(receipt) != _IDENTITY_KEYS:
            _fail(f"receiver acknowledgement receipt is invalid: {path}")
        checked = _identity(receipt, f"receiver acknowledgement at {path}", step=step)
        checked_core = {
            name: checked[name]
            for name in ("step", "target_sha256", "target_size", "config_sha256", "config_size")
        }
        if checked_core != core or checked["source_url"] != value["source_url"] or checked["slot"] != value["slot"]:
            _fail(f"receiver acknowledgement identity differs: {path}")
        endpoints.add(endpoint)
        records.append(checked)
    return value, records


def _tree_size_and_mtime(path: Path) -> tuple[int, int]:
    total = 0
    newest = 0
    try:
        entries = [path, *path.rglob("*")]
    except OSError as error:
        _fail(f"retention tree cannot be inspected: {error}")
    for entry in entries:
        try:
            info = entry.lstat()
        except OSError as error:
            _fail(f"retention tree changed during inspection: {error}")
        newest = max(newest, info.st_mtime_ns)
        if stat.S_ISLNK(info.st_mode):
            _fail(f"retention tree contains a symlink: {entry}")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total, newest


def _stable(path: Path, *, step: int, run_name: str) -> _Stable:
    info = _directory(path, "stable adapter directory")
    stable = path / "STABLE"
    stable_info = _regular(stable, "STABLE marker")
    if stable_info.st_size != 0:
        _fail(f"STABLE marker is not empty: {path}")
    manifest_identity, extras = _manifest_identity(path, step)
    entries = set()
    try:
        for child in path.iterdir():
            child_info = child.lstat()
            if stat.S_ISLNK(child_info.st_mode):
                _fail(f"stable adapter contains a symlink: {child}")
            if stat.S_ISDIR(child_info.st_mode):
                _fail(f"stable adapter contains a directory: {child}")
            entries.add(child.name)
    except OSError as error:
        _fail(f"stable adapter cannot be listed: {error}")
    allowed = _REQUIRED_FILES | extras | {"receivers-ready.json"}
    if not _REQUIRED_FILES <= entries or not entries <= allowed:
        _fail(f"stable adapter has unknown or missing files: {path}")
    server_instances: tuple[str, ...] = ()
    if "receivers-ready.json" in entries:
        ack_value, receivers = _ack(path, step=step, run_name=run_name, identity=manifest_identity)
        manifest_identity["source_url"] = ack_value["source_url"]
        manifest_identity["slot"] = ack_value["slot"]
        manifest_identity["server_instance"] = receivers[0]["server_instance"]
        server_instances = tuple(item["server_instance"] for item in receivers)
    bytes_used, _ = _tree_size_and_mtime(path)
    return _Stable(path, step, manifest_identity, (info.st_dev, info.st_ino, info.st_mtime_ns, stable_info.st_mtime_ns), bytes_used, server_instances)


def _partial(path: Path, step: int | None, reason: str) -> _Partial:
    info = _directory(path, "partial adapter directory")
    bytes_used, newest = _tree_size_and_mtime(path)
    return _Partial(path, step, bytes_used, newest, reason, (info.st_dev, info.st_ino, info.st_mtime_ns))


def _ack_match(candidate: _Stable, records: list[dict[str, Any]]) -> bool:
    return any(_core(candidate.identity) == _core(record) and record["server_instance"] in candidate.server_instances for record in records)


def _snapshot(run_dir: Path, evidence: dict[str, Any], keep_latest: int, partial_age_seconds: float, max_entries: int, max_bytes: int, now: float) -> dict[str, Any]:
    broadcasts = run_dir / "broadcasts"
    _directory(broadcasts, "broadcast directory")
    run_name = evidence["run_name"]
    loaded = evidence["h200_loaded"]
    acknowledgements = evidence["acknowledged"]
    candidates: list[_Stable] = []
    partials: list[_Partial] = []
    invalid: list[dict[str, str]] = []
    for entry in sorted(broadcasts.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            invalid.append({"path": str(entry), "reason": "symlink"})
            continue
        match = _STEP.fullmatch(entry.name)
        if match is None:
            if entry.name in {".retention.lock"} or entry.name.startswith(".retention-quarantine-"):
                invalid.append({"path": str(entry), "reason": "retention-private"})
            else:
                invalid.append({"path": str(entry), "reason": "unknown-broadcast-entry"})
            continue
        step = int(match.group(1))
        if not entry.is_dir():
            invalid.append({"path": str(entry), "reason": "step-is-not-directory"})
            continue
        stable_path = entry / "STABLE"
        if stable_path.is_file() and not stable_path.is_symlink():
            try:
                candidates.append(_stable(entry, step=step, run_name=run_name))
            except RetentionError as error:
                invalid.append({"path": str(entry), "reason": str(error)})
            continue
        marker_names = {child.name for child in entry.iterdir() if child.name in _WRITER_MARKERS}
        reason = "writer-marker" if marker_names else "no-stable-marker"
        try:
            partials.append(_partial(entry, step, reason))
        except RetentionError as error:
            invalid.append({"path": str(entry), "reason": str(error)})
    candidates.sort(key=lambda item: item.step)
    if not any(
        item.step == loaded["step"]
        and _core(item.identity) == _core(loaded)
        and loaded["server_instance"] in item.server_instances
        for item in candidates
    ):
        _fail("H200 loaded identity has no matching local stable publication")
    keep_steps = {item.step for item in candidates[-keep_latest:]}
    acked = {item["step"]: item for item in acknowledgements}
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        record = acked.get(candidate.step)
        protected = (
            candidate.step >= loaded["step"]
            or candidate.step in keep_steps
            or record is None
            or not _ack_match(candidate, [record])
        )
        if not protected:
            eligible.append({"path": str(candidate.path), "step": candidate.step, "bytes": candidate.bytes, "identity": candidate.identity, "stat_key": candidate.stat_key})
    partial_eligible: list[dict[str, Any]] = []
    for partial in partials:
        age = max(0.0, now - partial.newest_mtime_ns / 1_000_000_000)
        names = {item.name for item in partial.path.iterdir()}
        protected = bool(names & _WRITER_MARKERS or "receivers-ready.json" in names or age < partial_age_seconds)
        if not protected:
            partial_eligible.append({"path": str(partial.path), "step": partial.step, "bytes": partial.bytes, "age_seconds": age, "newest_mtime_ns": partial.newest_mtime_ns, "reason": partial.reason, "stat_key": partial.stat_key})
    selected: list[dict[str, Any]] = []
    used_bytes = 0
    for item in [*eligible, *partial_eligible]:
        if len(selected) >= max_entries or used_bytes + item["bytes"] > max_bytes:
            continue
        selected.append(item)
        used_bytes += item["bytes"]
    return {
        "run_dir": str(run_dir),
        "loaded": loaded,
        "stable": [{"path": str(item.path), "step": item.step, "bytes": item.bytes, "identity": item.identity, "kept": item.step in keep_steps, "acknowledged": item.step in acked and _ack_match(item, [acked[item.step]])} for item in candidates],
        "partials": [{"path": str(item.path), "step": item.step, "bytes": item.bytes, "age_seconds": max(0.0, now - item.newest_mtime_ns / 1_000_000_000), "reason": item.reason, "eligible": any(value["path"] == str(item.path) for value in partial_eligible)} for item in partials],
        "invalid": invalid,
        "selected": selected,
        "selected_bytes": used_bytes,
    }


def inventory(run_dir: str | Path, evidence: Mapping[str, Any] | str | Path, *, keep_latest: int = 2, partial_age_seconds: float = 1800.0, max_entries: int = 8, max_bytes: int = 50 * 1024**3, now: float | None = None) -> dict[str, Any]:
    """Return a dry-run retention inventory for one exact run directory."""
    if type(keep_latest) is not int or not 1 <= keep_latest <= 16:
        _fail("keep_latest must be between 1 and 16")
    if not isinstance(partial_age_seconds, (int, float)) or partial_age_seconds < 0:
        _fail("partial age bound is invalid")
    if type(max_entries) is not int or not 1 <= max_entries <= 64:
        _fail("max_entries must be between 1 and 64")
    if type(max_bytes) is not int or max_bytes <= 0:
        _fail("max_bytes must be positive")
    path = _safe_run_dir(run_dir)
    loaded = _load_evidence(evidence)
    return _snapshot(path, loaded, keep_latest, float(partial_age_seconds), max_entries, max_bytes, time.time() if now is None else float(now))


def cleanup(run_dir: str | Path, evidence: Mapping[str, Any] | str | Path, *, keep_latest: int = 2, partial_age_seconds: float = 1800.0, max_entries: int = 8, max_bytes: int = 50 * 1024**3, dry_run: bool = True, now: float | None = None) -> dict[str, Any]:
    """Inventory or remove bounded, identity-checked retention candidates."""
    report = inventory(run_dir, evidence, keep_latest=keep_latest, partial_age_seconds=partial_age_seconds, max_entries=max_entries, max_bytes=max_bytes, now=now)
    if dry_run or not report["selected"]:
        report["dry_run"] = True
        report["removed"] = []
        return report
    path = _safe_run_dir(run_dir)
    evidence_value = _load_evidence(evidence)
    broadcasts = path / "broadcasts"
    lock_path = broadcasts / ".retention.lock"
    _regular(lock_path, "retention lock") if lock_path.exists() else None
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        _fail(f"retention lock cannot be opened: {error}")
    removed = []
    with os.fdopen(fd, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        fresh = inventory(path, evidence_value, keep_latest=keep_latest, partial_age_seconds=partial_age_seconds, max_entries=max_entries, max_bytes=max_bytes, now=now)
        for item in fresh["selected"]:
            source = Path(item["path"])
            if source.parent != broadcasts or not source.name.startswith("step_"):
                _fail("retention candidate escaped broadcast directory")
            before = source.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                _fail("retention candidate changed before removal")
            expected_stat = tuple(item.get("stat_key", ()))
            observed_stat = (before.st_dev, before.st_ino, before.st_mtime_ns) + ((before.st_ctime_ns,) if len(expected_stat) == 4 else ())
            if observed_stat[: len(expected_stat)] != expected_stat:
                _fail("retention candidate identity changed before removal")
            quarantine = broadcasts / f".retention-quarantine-{uuid.uuid4().hex}"
            try:
                os.rename(source, quarantine)
            except OSError as error:
                _fail(f"retention candidate changed before atomic quarantine: {error}")
            try:
                if "identity" in item:
                    checked = _stable(
                        quarantine,
                        step=int(item["step"]),
                        run_name=evidence_value["run_name"],
                    )
                    if checked.identity != item["identity"]:
                        _fail("stable retention identity changed during quarantine")
                else:
                    checked_partial = _partial(
                        quarantine,
                        item.get("step"),
                        str(item.get("reason", "partial")),
                    )
                    if (
                        checked_partial.bytes != item["bytes"]
                        or checked_partial.newest_mtime_ns != item["newest_mtime_ns"]
                    ):
                        _fail("partial retention identity changed during quarantine")
                shutil.rmtree(quarantine)
            except OSError as error:
                _fail(f"retention quarantine removal failed: {error}")
            removed.append(item)
    report["dry_run"] = False
    report["removed"] = removed
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="inventory or clean one adapter broadcast run")
    parser.add_argument("run_dir")
    parser.add_argument("evidence")
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--partial-age-seconds", type=float, default=1800.0)
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3)
    parser.add_argument("--apply", action="store_true", help="remove selected entries after revalidation")
    args = parser.parse_args(argv)
    try:
        result = cleanup(args.run_dir, args.evidence, keep_latest=args.keep_latest, partial_age_seconds=args.partial_age_seconds, max_entries=args.max_entries, max_bytes=args.max_bytes, dry_run=not args.apply)
    except RetentionError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
