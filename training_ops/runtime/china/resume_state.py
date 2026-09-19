"""Validate one adapter restart and create its explicit runtime control."""

import hashlib
import fcntl
import ctypes
import json
import os
import re
import secrets
import stat
import sys
import tomllib
from pathlib import Path


def regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"restart requires a regular file: {path}")


def digest(path: Path) -> str:
    regular(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def explicit_config(text: str, step: int) -> str:
    """Change only the two restart fields, without guessing CLI syntax."""
    if type(step) is not int or step < 0:
        raise ValueError("invalid explicit restart step")
    before = tomllib.loads(text)
    expected = dict(before)
    expected["ckpt"] = dict(before.get("ckpt", {}), resume_step=step, skip_progress=True)
    lines = text.splitlines(keepends=True)
    output = []
    inside = False
    found = False
    for line in lines:
        if re.match(r"\s*\[", line):
            inside = bool(re.fullmatch(r"\s*\[ckpt\]\s*(?:#.*)?\n?", line))
            if inside:
                found = True
                output.extend([line.rstrip("\n") + "\n", f"resume_step = {step}\nskip_progress = true\n"])
                continue
        if inside and re.match(r"\s*(resume_step|skip_progress)\s*=", line):
            continue
        output.append(line)
    if not found:
        output.append(f"\n[ckpt]\nresume_step = {step}\nskip_progress = true\n")
    result = "".join(output)
    if tomllib.loads(result) != expected:
        raise ValueError("restart config changed an unrelated setting")
    return result


def write_config(config: str, control: str, step: int) -> str:
    directory = Path(control)
    active = directory.parent
    if not directory.is_absolute() or directory.name != "control" or active.name != f"run_{active.parent.name}":
        raise ValueError("restart control directory is not exact")
    if ".." in directory.parts:
        raise ValueError("restart control directory has a parent traversal")
    rendered = explicit_config(Path(config).read_text(), step).encode()
    # Anchor each component with O_NOFOLLOW. Do not use a pathname-based
    # mkstemp/chmod after checking ancestors: that leaves a symlink race.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in directory.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        name = f"orch-restart-{secrets.token_hex(12)}.toml"
        output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
        with os.fdopen(output, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
        return str(directory / name)
    finally:
        os.close(descriptor)


def check_state(service_dir: str, step: int, role: str, adapter: str, expected: str = "", *, scan_rollouts: bool = True) -> dict:
    if type(step) is not int or step < 0 or role not in {"origin", "trainer", "orchestrator"}:
        raise ValueError("invalid explicit restart state")
    active = Path(service_dir)
    if not active.is_absolute() or active.name != f"run_{active.parent.name}":
        raise ValueError("restart run directory is not exact")
    parent = active.parent
    target = active / "broadcasts" / f"step_{step}"
    for directory in (parent, active, active / "control", active / "broadcasts", target):
        if directory.is_symlink():
            raise ValueError(f"restart directory is a symlink: {directory}")
    weight = target / "adapter_model.safetensors"
    if Path(adapter) != weight:
        raise ValueError("resume adapter must be the exact requested policy path")
    result = {"policy_step": step, "next_batch": step + 1}
    if role == "origin":
        regular(target / "adapter_manifest.json")
        manifest = json.loads((target / "adapter_manifest.json").read_text())
        if type(manifest.get("target_step")) is not int or manifest["target_step"] != step:
            raise ValueError("stable adapter manifest has a different policy step")
        actual = digest(weight)
        if manifest.get("target_sha256") != actual or manifest.get("target_size") != weight.stat().st_size:
            raise ValueError("stable adapter weight identity differs")
        config = target / "adapter_config.json"
        if manifest.get("config_sha256") != digest(config) or manifest.get("config_size") != config.stat().st_size:
            raise ValueError("stable adapter config identity differs")
        result.update({"sha256": actual, "size": weight.stat().st_size,
                       "config_sha256": manifest["config_sha256"],
                       "config_size": manifest["config_size"]})
    elif role == "trainer":
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(weight) != expected:
            raise ValueError("trainer resume adapter differs from the stable policy")
    if role in {"origin", "orchestrator"}:
        regular(target / "STABLE")
        for published in target.parent.glob("step_*/STABLE"):
            match = re.fullmatch(r"step_(\d+)", published.parent.name)
            if match and int(match[1]) > step:
                raise ValueError(f"later stable policy conflicts with restart: {published}")
    if role != "orchestrator":
        for checkpoint in (parent / "checkpoints", active / "checkpoints"):
            if checkpoint.is_symlink() or (checkpoint.exists() and any(checkpoint.iterdir())):
                raise ValueError(f"optimizer checkpoint conflicts with adapter restart: {checkpoint}")
    if not scan_rollouts:
        return result
    # Do not remove evidence. The operator must quarantine old binaries first.
    for root in (parent / "rollouts", active / "rollouts", parent / ".incoming" / "rollouts", active / ".incoming" / "rollouts"):
        if root.is_symlink():
            raise ValueError(f"restart rollout directory is a symlink: {root}")
        if not root.exists():
            continue
        for directory in root.iterdir():
            match = re.fullmatch(r"step_(\d+)", directory.name)
            if not match or int(match[1]) <= step:
                continue
            if directory.is_symlink():
                raise ValueError(f"restart rollout step is a symlink: {directory}")
            for cached in directory.rglob("*"):
                if cached.name == "train_rollouts.bin" or re.fullmatch(r"rank_\d+\.bin", cached.name):
                    raise ValueError(f"cached rollout batch blocks restart: {cached}")
    return result


def _delivered_batch(batch: Path) -> None:
    regular(batch); marker = batch.with_name(batch.name + ".delivered")  # noqa: E702
    try: regular(marker)  # noqa: E701
    except OSError as error: raise ValueError(f"batch delivery marker is missing: {marker}") from error  # noqa: E701
    expected = digest(batch)
    if marker.read_text() != expected + "\n": raise ValueError(f"batch delivery marker differs: {marker}")  # noqa: E701

def _identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns

def _tree_payload(tree: dict) -> dict:
    return {name: {"identity": list(value["identity"]), "sha256": value["sha256"]} for name, value in tree.items()}  # noqa: E501

def _cache_tree(target: Path) -> tuple[int, bool, tuple[int, int, int, int, int], dict]:
    total, binaries, sidecar, tree = 0, [], False, {}
    for path in (target, *target.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"cached rollout tree contains a symlink: {path}")
        relative = "" if path == target else path.relative_to(target).as_posix()
        if path.is_dir():
            tree[relative or "."] = {"identity": _identity(path), "sha256": ""}
            if relative and relative != "train" and not relative.startswith("train/"):
                raise ValueError(f"cached rollout tree has an unexpected directory: {path}")
            continue
        binary = relative in {"train_rollouts.bin", "train_rollouts.bin.delivered"} or re.fullmatch(r"rank_[0-9]+\.bin", relative)
        trace = relative in {"train/effective/traces.jsonl", "train/all/traces.jsonl"}
        if not binary and not trace:
            raise ValueError(f"cached rollout tree has an unexpected entry: {path}")
        regular(path); total += path.stat().st_size  # noqa: E702
        tree[relative] = {"identity": _identity(path), "sha256": digest(path)}
        if relative == "train_rollouts.bin.delivered": sidecar = True  # noqa: E701
        if binary and relative.endswith(".bin") and relative != "train_rollouts.bin.delivered":
            binaries.append(relative)
    if sidecar and "train_rollouts.bin" not in binaries:
        raise ValueError(f"cached rollout tree has a sidecar without train_rollouts.bin: {target}")
    if not binaries: return 0, False, _identity(target), tree  # noqa: E701
    if "train_rollouts.bin" in binaries and any(name.startswith("rank_") for name in binaries):
        raise ValueError(f"cached rollout tree mixes rank and train_rollouts.bin: {target}")
    if any(name.startswith("rank_") for name in binaries) and "rank_0.bin" not in binaries:
        raise ValueError(f"cached rollout tree has no rank_0.bin: {target}")
    if "train_rollouts.bin" in binaries:
        delivered = target / "train_rollouts.bin.delivered"
        if delivered.exists() or delivered.is_symlink():
            _delivered_batch(target / "train_rollouts.bin")
    return total, True, _identity(target), tree

def _future_plan(service_dir: str, step: int) -> list[dict]:
    active = exact_run(service_dir)
    parent = active.parent
    plans, roots = [], (parent / "rollouts", active / "rollouts", parent / ".incoming" / "rollouts", active / ".incoming" / "rollouts")
    for root in roots:
        if root.parent.is_symlink(): raise ValueError(f"restart rollout parent is a symlink: {root.parent}")  # noqa: E701
        if root.is_symlink():
            raise ValueError(f"restart rollout directory is a symlink: {root}")
        if not root.exists():
            continue
        for target in root.iterdir():
            match = re.fullmatch(r"step_([0-9]+)", target.name)
            if not match or int(match[1]) <= step: continue  # noqa: E701
            if target.is_symlink(): raise ValueError(f"future rollout step is a symlink: {target}")  # noqa: E701
            if not target.is_dir():
                raise ValueError(f"future rollout step is not a directory: {target}")
            size, used, identity, tree = _cache_tree(target)
            if used and int(match[1]) != step + 1:
                raise ValueError(f"later cached rollout step blocks resume: {target}")
            if used:
                plans.append({"root": root, "target": target, "step": int(match[1]), "bytes": size, "identity": identity, "tree": tree, "root_identity": _identity(root), "owner_identity": _identity(root.parent), "markers": []})
        current = root / f"step_{step}"
        if current.is_symlink():
            raise ValueError(f"restart rollout step is a symlink: {current}")
        if current.is_dir() and (current / "train_rollouts.bin").exists():
            _delivered_batch(current / "train_rollouts.bin")
    for owner in {root.parent for root in roots}:
        for marker in owner.glob(".microbatch-sync.marker*"):
            if marker.is_symlink():
                raise ValueError(f"microbatch marker is a symlink: {marker}")
            regular(marker)
            match = re.fullmatch(r"generation=rollouts/step_([0-9]+) count=([1-9][0-9]*) sha256=([a-f0-9]{64})", marker.read_text().strip())
            if not match:
                raise ValueError(f"microbatch marker is invalid: {marker}")
            marker_step = int(match[1])
            if marker_step <= step: continue  # noqa: E701
            matches = [item for item in plans if item["root"] == owner / "rollouts" and item["step"] == marker_step]
            if len(matches) != 1:
                raise ValueError(f"future marker has no exact cache: {marker}")
            target = matches[0]["target"]
            source = target / "rank_0.bin"
            if not source.is_file() or digest(source) != match[3] or int(match[2]) != len(list(target.glob("rank_*.bin"))):
                raise ValueError(f"future marker does not match rank_0.bin: {marker}")
            matches[0]["markers"].append((marker, _identity(marker)))
    return plans

def _safe_quarantine_dir(owner: Path, step: int) -> Path:
    root = owner / ".rollout-quarantine"
    for path in (root, root / f"resume-step_{step}"):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.geteuid() or stat.S_IMODE(path.stat().st_mode) != 0o700: raise ValueError(f"quarantine path is not a private directory: {path}")  # noqa: E701
        else: path.mkdir(mode=0o700)  # noqa: E701
    return root / f"resume-step_{step}"

def _fd_identity(descriptor: int) -> tuple[int, int, int, int, int]:
    info = os.fstat(descriptor); return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns  # noqa: E702

def _verify_tree(path: Path, expected: dict) -> None:
    actual = _tree_payload(_cache_tree(path)[3])
    if actual != expected: raise ValueError(f"quarantine tree identity differs: {path}")  # noqa: E701

def _write_receipt(path: Path, value: dict) -> dict:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        info = path.lstat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600 or json.loads(path.read_text()) != value:
            raise ValueError(f"quarantine receipt differs: {path}")
        return value
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())  # noqa: E702
    return value

def _verify_receipt(path: Path, value: dict, service_dir: str, step: int) -> dict:
    required = {"run", "service_dir", "resume_step", "step", "source_root", "source", "source_root_identity", "tree", "markers", "bytes", "destination"}
    if set(value) != required or value["run"] != Path(service_dir).parent.name or value["service_dir"] != service_dir or value["resume_step"] != step or value["step"] != step + 1:
        raise ValueError(f"quarantine receipt scope differs: {path}")
    destination = Path(value["destination"])
    if destination != path.with_suffix("") or destination.parent != path.parent or Path(value["source"]).name != destination.name:
        raise ValueError(f"quarantine receipt destination differs: {path}")
    source, target = Path(value["source"]), destination
    if source.exists() and target.exists() or not source.exists() and not target.exists(): raise ValueError(f"quarantine receipt has duplicate or missing source/destination: {path}")  # noqa: E701
    if source.exists() and _identity(Path(value["source_root"])) != tuple(value["source_root_identity"]): raise ValueError(f"quarantine source root changed: {path}")  # noqa: E701
    _verify_tree(source if source.exists() else target, value["tree"])
    for marker in value["markers"]:
        source_marker, destination_marker = Path(marker["source"]), Path(marker["destination"])
        if source_marker.exists() and destination_marker.exists() or not source_marker.exists() and not destination_marker.exists():
            raise ValueError(f"quarantine receipt marker state differs: {path}")
        chosen = source_marker if source_marker.exists() else destination_marker
        if _identity(chosen) != tuple(marker["identity"]) or digest(chosen) != marker["sha256"]:
            raise ValueError(f"quarantine receipt marker differs: {path}")
    return value

def _scan_quarantine(service_dir: str, step: int) -> dict[str, dict]:
    active = exact_run(service_dir)
    owners = {active.parent, active, active.parent / ".incoming", active / ".incoming"}
    source_roots = {owner / "rollouts" for owner in owners}
    receipts = {}
    for owner in owners:
        root = owner / ".rollout-quarantine" / f"resume-step_{step}"
        if not root.exists(): continue  # noqa: E701
        if root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.geteuid() or stat.S_IMODE(root.stat().st_mode) != 0o700: raise ValueError(f"quarantine path is not a private directory: {root}")  # noqa: E701
        targets, markers = [], []
        for path in root.iterdir():
            if path.name.endswith(".json"):
                if not re.fullmatch(r"step_[0-9]+\.json", path.name) or path.is_symlink(): raise ValueError(f"quarantine receipt is unsafe: {path}")  # noqa: E701
                value = json.loads(path.read_text())
                if Path(value.get("source_root", "")) not in source_roots or Path(value.get("source", "")).parent != Path(value.get("source_root", "")) or value.get("source", "") in receipts: raise ValueError(f"quarantine receipt source differs: {path}")  # noqa: E701
                receipts[value.get("source", "")] = _verify_receipt(path, value, service_dir, step)
            elif path.name.startswith("step_"): targets.append(path)  # noqa: E701
            elif path.name.startswith(".microbatch-sync.marker"): markers.append(path)  # noqa: E701
            else:
                raise ValueError(f"quarantine has an unexpected entry: {path}")
        allowed = {marker.get("destination") for value in receipts.values() for marker in value["markers"]}
        if any(not any(value.get("destination") == str(target) for value in receipts.values()) for target in targets) or any(str(marker) not in allowed for marker in markers): raise ValueError(f"quarantine entry has no exact receipt: {root}")  # noqa: E701
    return receipts

def _move_entry(source: Path, destination: Path, source_parent: tuple, destination_parent: tuple, tree: dict | None = None) -> None:
    source_root = open_directory(source.parent); destination_root = open_directory(destination.parent)  # noqa: E702
    try:
        if _identity(source.parent) != source_parent or _fd_identity(source_root) != source_parent or _identity(destination.parent) != destination_parent or _fd_identity(destination_root) != destination_parent or tree is not None and _tree_payload(_cache_tree(source)[3]) != tree:
            raise ValueError(f"quarantine parent changed before rename: {source}")
        move_no_replace(source.name, destination.name, src_dir_fd=source_root, dst_dir_fd=destination_root)
    finally:
        os.close(source_root); os.close(destination_root)  # noqa: E702

def _publisher_locks(service_dir: str) -> list[int]:
    active = exact_run(service_dir)
    locks = []
    try:
        for owner in (active.parent, active, active.parent / ".incoming", active / ".incoming"):
            if owner.is_symlink(): raise ValueError(f"publisher lock owner is a symlink: {owner}")  # noqa: E701
            for name in (".batch-sync.lock", ".microbatch-sync.lock"):
                lock = owner / name
                if lock.is_symlink(): raise ValueError(f"publisher lock is a symlink: {lock}")  # noqa: E701
                try: descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW); locks.append(descriptor)  # noqa: E701,E702
                except FileNotFoundError: continue  # noqa: E701
                if not stat.S_ISREG(os.fstat(descriptor).st_mode) or _fd_identity(descriptor) != (expected := _identity(lock)): raise ValueError(f"publisher lock identity differs: {lock}")  # noqa: E701
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _identity(lock) != expected or _fd_identity(descriptor) != expected: raise ValueError(f"publisher lock changed during lock: {lock}")  # noqa: E701
    except BaseException:
        for descriptor in locks: os.close(descriptor)  # noqa: E701
        raise
    return locks


def quarantine_rollouts(service_dir: str, step: int, role: str, adapter: str, expected: str = "") -> dict:
    locks = _publisher_locks(service_dir)
    try:
        check_state(service_dir, step, role, adapter, expected, scan_rollouts=False)
        receipts = _scan_quarantine(service_dir, step)
        plans, moved = _future_plan(service_dir, step), []
        for item in plans:
            owner = item["root"].parent; destination = _safe_quarantine_dir(owner, step) / item["target"].name  # noqa: E702
            receipt_path = destination.with_name(destination.name + ".json")
            source_present = item["target"].exists() or item["target"].is_symlink()
            value = receipts.get(str(item["target"]), {"run": Path(service_dir).parent.name, "service_dir": service_dir, "resume_step": step, "step": item["step"], "source_root": str(item["root"]), "source": str(item["target"]), "source_root_identity": list(item["root_identity"]), "tree": _tree_payload(item["tree"]), "markers": [{"source": str(marker), "destination": str(destination.parent / marker.name), "identity": list(identity), "sha256": digest(marker)} for marker, identity in item["markers"]], "bytes": item["bytes"], "destination": str(destination)})
            _write_receipt(receipt_path, value)
            for marker in value["markers"]:
                source_marker, marker_destination = Path(marker["source"]), Path(marker["destination"])
                if source_marker.exists() or source_marker.is_symlink():
                    if _identity(source_marker) != tuple(marker["identity"]): raise ValueError(f"microbatch marker changed before quarantine: {source_marker}")  # noqa: E701
                    if marker_destination.exists() or marker_destination.is_symlink(): raise ValueError(f"quarantine marker destination exists: {marker_destination}")  # noqa: E701
                    _move_entry(source_marker, marker_destination, _identity(source_marker.parent), _identity(marker_destination.parent))
                elif not marker_destination.is_file(): raise ValueError(f"quarantine marker disappeared: {source_marker}")  # noqa: E701
            if item["target"].exists() or item["target"].is_symlink():
                if _identity(item["target"]) != item["identity"] or _tree_payload(_cache_tree(item["target"])[3]) != value["tree"]: raise ValueError(f"cached rollout changed before quarantine: {item['target']}")  # noqa: E701
                if destination.exists() or destination.is_symlink(): raise ValueError(f"quarantine destination already exists: {destination}")  # noqa: E701
                _move_entry(item["target"], destination, item["root_identity"], _identity(destination.parent), value["tree"])
            elif not destination.is_dir() or destination.is_symlink():
                raise ValueError(f"quarantine source disappeared: {item['target']}")
            _verify_receipt(receipt_path, value, service_dir, step)
            if source_present: moved.append((str(destination), item["bytes"]))  # noqa: E701
        result = check_state(service_dir, step, role, adapter, expected)
        result.update({"quarantined": [path for path, _ in moved], "bytes": sum(size for _, size in moved)})
        return result
    finally:
        for descriptor in locks:
            os.close(descriptor)


def exact_run(service_dir: str) -> Path:
    active = Path(service_dir)
    if not active.is_absolute() or active.name != f"run_{active.parent.name}" or ".." in active.parts:
        raise ValueError("repair run directory is not exact")
    for path in (active, *active.parents):
        if path.is_symlink():
            raise ValueError("repair run has a symlink ancestor")
    return active


def latest_policy(service_dir: str, launch_step: int) -> dict:
    if type(launch_step) is not int or launch_step < 0:
        raise ValueError("invalid launch step")
    active = exact_run(service_dir)
    root = active / "broadcasts"
    if root.is_symlink():
        raise ValueError("broadcasts is a symlink")
    steps = []
    for path in root.iterdir():
        if not re.fullmatch(r"step_(0|[1-9][0-9]*)", path.name):
            continue
        if path.is_symlink():
            raise ValueError("policy directory is a symlink")
        stable = path / "STABLE"
        if stable.exists() or stable.is_symlink():
            regular(stable)
            steps.append(int(path.name[5:]))
    if not steps or max(steps) < launch_step:
        raise ValueError("stable policy precedes trainer launch")
    step = max(steps)
    result = {"step": step}
    if step > launch_step:
        batch = active / "rollouts" / f"step_{step}" / "train_rollouts.bin"
        for path in batch.parents:
            if path.is_symlink():
                raise ValueError("completed batch has a symlink ancestor")
        if batch.stat().st_size <= 0:
            raise ValueError("completed trainer batch is empty")
        result["batch_sha256"] = digest(batch)
    return result


def completed_batch(service_dir: str, step: int, expected: str) -> dict:
    if type(step) is not int or step < 0:
        raise ValueError("invalid completed step")
    active = exact_run(service_dir)
    root = active / "rollouts" / f"step_{step}"
    trace = root / "train/effective/traces.jsonl"
    for path in (trace, *trace.parents):
        if path.is_symlink():
            raise ValueError("completed rollout has a symlink")
    regular(trace)
    count = 0
    with trace.open() as stream:
        for line in stream:
            if not isinstance(json.loads(line), dict):
                raise ValueError("effective rollout record is invalid")
            count += 1
    if not count or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("completed effective rollout evidence is missing")
    if digest(root / "train_rollouts.bin") != expected:
        raise ValueError("completed batch differs from trainer batch")
    delivered = root / "train_rollouts.bin.delivered"
    regular(delivered)
    if delivered.read_text() != expected + "\n":
        raise ValueError("completed batch has no exact delivery marker")
    return {"step": step, "records": count, "batch_sha256": expected}


def move_no_replace(source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as error:
        raise ValueError("atomic no-replace rename is unavailable") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(src_dir_fd, os.fsencode(source), dst_dir_fd, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def partial_tree(descriptor: int, prefix: str = ".") -> dict:
    """Read every allowed entry relative to pinned, no-follow directories."""
    allowed = {"train": True, "train/all": True, "train/all/traces.jsonl": False}
    result = {prefix: os.fstat(descriptor)}
    for name in os.listdir(descriptor):
        relative = name if prefix == "." else f"{prefix}/{name}"
        if relative not in allowed:
            raise ValueError("quarantined rollout has an unexpected entry")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if allowed[relative]:
            flags |= os.O_DIRECTORY
        child = os.open(name, flags, dir_fd=descriptor)
        try:
            info = os.fstat(child)
            if allowed[relative]:
                result.update(partial_tree(child, relative))
            elif not stat.S_ISREG(info.st_mode):
                raise ValueError("quarantined trace is not regular")
            else:
                result[relative] = info
        finally:
            os.close(child)
    return result


def quarantine_partial(root: Path, target: Path, observed: dict) -> str:
    directory = open_directory(root)
    private = None
    destination = root / f"quarantined-step_{target.name[5:]}-{secrets.token_hex(16)}"
    child_name = target.name
    location = destination / child_name
    try:
        if os.stat(target.name, dir_fd=directory, follow_symlinks=False) != observed["."]:
            raise ValueError("partial rollout changed before quarantine")
        # mkdir is exclusive. A collision never grants use of an old directory.
        os.mkdir(destination.name, mode=0o700, dir_fd=directory)
        private = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
        info = os.fstat(private)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("quarantine directory is not private")
        move_no_replace(target.name, child_name, src_dir_fd=directory, dst_dir_fd=private)
        try:
            moved = os.open(child_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=private)
            try:
                current = partial_tree(moved)
            finally:
                os.close(moved)
            def identity(value):
                return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns
            if {name: identity(value) for name, value in current.items()} != {name: identity(value) for name, value in observed.items()}:
                raise ValueError("partial rollout changed during quarantine")
            linked = os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("private quarantine directory moved")
        except (OSError, ValueError) as error:
            # Never replace a new active step or return a substituted inode.
            moved_info = os.stat(child_name, dir_fd=private, follow_symlinks=False)
            original = observed["."]
            if (moved_info.st_dev, moved_info.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError(f"quarantine identity changed; preserved at {location}") from error
            try:
                move_no_replace(child_name, target.name, src_dir_fd=private, dst_dir_fd=directory)
            except OSError as rollback_error:
                raise ValueError(f"quarantine rollback refused; preserved at {location}: {rollback_error}") from error
            raise ValueError(f"quarantine validation failed; restored {target}: {error}") from error
        return str(location)
    finally:
        if private is not None:
            os.close(private)
        os.close(directory)


def partial_rollout(service_dir: str, step: int, *, quarantine=False) -> dict:
    if type(step) is not int or step < 0:
        raise ValueError("invalid partial rollout step")
    active = exact_run(service_dir)
    root = active / "rollouts"
    if root.is_symlink():
        raise ValueError("rollouts is a symlink")
    target = root / f"step_{step + 1}"
    if not root.exists():
        return {"partial": False}
    for path in root.iterdir():
        match = re.fullmatch(r"step_([0-9]+)", path.name)
        if match and int(match[1]) > step and path != target:
            raise ValueError("unexpected later rollout directory")
    if not target.exists() and not target.is_symlink():
        return {"partial": False}
    allowed = {"train", "train/all", "train/all/traces.jsonl"}
    observed = {}
    for path in (target, *target.rglob("*")):
        if path.is_symlink() or (path != target and path.relative_to(target).as_posix() not in allowed):
            raise ValueError("next rollout is not an exact trace-only partial")
        if not path.is_dir():
            regular(path)
        relative = path.relative_to(target).as_posix()
        if path.is_dir() == (relative == "train/all/traces.jsonl"):
            raise ValueError("partial rollout entry has the wrong type")
        observed[relative] = path.lstat()
    # The existing publisher owns this lock for its full lifetime. Keep it
    # locked across the final validation and rename; never stop a publisher.
    lock = active / ".batch-sync.lock"
    descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("publisher lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for proc in Path("/proc").iterdir():
            if not proc.name.isdecimal() or int(proc.name) == os.getpid():
                continue
            try:
                environment = (proc / "environ").read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            if f"SOURCE_RUN_DIR={active}".encode() in environment:
                raise ValueError("run publisher is still present")
        if lock.stat().st_ino != os.fstat(descriptor).st_ino:
            raise ValueError("publisher lock changed")
        current = {path.relative_to(target).as_posix(): path.lstat() for path in (target, *target.rglob("*"))}
        if current != observed:
            raise ValueError("partial rollout changed during validation")
        result = {"partial": True}
        if quarantine:
            result["quarantined"] = quarantine_partial(root, target, observed)
        return result
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    try:
        if sys.argv[1] == "latest":
            print(json.dumps(latest_policy(sys.argv[2], int(sys.argv[3]))))
        elif sys.argv[1] == "completed":
            print(json.dumps(completed_batch(sys.argv[2], int(sys.argv[3]), sys.argv[4])))
        elif sys.argv[1] in {"partial", "quarantine"}:
            if sys.argv[1] == "quarantine" and len(sys.argv) >= 7:
                print(json.dumps(quarantine_rollouts(*[sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]]), sort_keys=True))
            else:
                print(json.dumps(partial_rollout(sys.argv[2], int(sys.argv[3]), quarantine=sys.argv[1] == "quarantine")))
        elif sys.argv[1] == "policy":
            print(json.dumps(check_state(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], scan_rollouts=False), sort_keys=True))
        elif sys.argv[1] == "rollout":
            print(json.dumps(quarantine_rollouts(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]), sort_keys=True))
        elif sys.argv[1] == "config":
            print(explicit_config(Path(sys.argv[2]).read_text(), int(sys.argv[3])), end="")
        elif sys.argv[1] == "write-config":
            print(write_config(sys.argv[2], sys.argv[3], int(sys.argv[4])))
        else:
            value = check_state(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
            print(json.dumps(value, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"adapter restart refused: {error}")
