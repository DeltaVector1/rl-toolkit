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


def check_state(service_dir: str, step: int, role: str, adapter: str, expected: str = "") -> dict:
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
    if role != "orchestrator":
        for checkpoint in (parent / "checkpoints", active / "checkpoints"):
            if checkpoint.is_symlink() or (checkpoint.exists() and any(checkpoint.iterdir())):
                raise ValueError(f"optimizer checkpoint conflicts with adapter restart: {checkpoint}")
    if role in {"origin", "orchestrator"}:
        regular(target / "STABLE")
        for published in target.parent.glob("step_*/STABLE"):
            match = re.fullmatch(r"step_(\d+)", published.parent.name)
            if match and int(match[1]) > step:
                raise ValueError(f"later stable policy conflicts with restart: {published}")
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
        result["sha256"] = actual
        result["size"] = weight.stat().st_size
        result["config_sha256"] = manifest["config_sha256"]
        result["config_size"] = manifest["config_size"]
    elif role == "trainer":
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(weight) != expected:
            raise ValueError("trainer resume adapter differs from the stable policy")
    return result


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
            print(json.dumps(partial_rollout(sys.argv[2], int(sys.argv[3]), quarantine=sys.argv[1] == "quarantine")))
        elif sys.argv[1] == "config":
            print(explicit_config(Path(sys.argv[2]).read_text(), int(sys.argv[3])), end="")
        elif sys.argv[1] == "write-config":
            print(write_config(sys.argv[2], sys.argv[3], int(sys.argv[4])))
        else:
            value = check_state(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
            print(json.dumps(value, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"adapter restart refused: {error}")
