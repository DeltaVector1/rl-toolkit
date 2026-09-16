"""Drain one run's publisher group and inherited publication lock."""

import fcntl
import os
import signal
import stat
import sys
import time
from pathlib import Path


def process(pid: int):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        return root, fields[0], int(fields[2]), fields[19]
    except (OSError, IndexError, ValueError):
        return None


def matches(pid: int, directory: Path, script: str, lock_identity, environment="SOURCE_RUN_DIR") -> bool:
    root = Path("/proc") / str(pid)
    try:
        if f"{environment}={directory}".encode() not in (root / "environ").read_bytes().split(b"\0"):
            return False
        if script.encode() in (root / "cmdline").read_bytes().split(b"\0"):
            return True
        info = (root / "fd/9").stat()
        return lock_identity == (info.st_dev, info.st_ino)
    except OSError:
        return False


def snapshot():
    return {int(path.name): process(int(path.name)) for path in Path("/proc").iterdir() if path.name.isdecimal()}


def live_members(records, group):
    return {pid: record for pid, record in records.items() if record and record[1] != "Z" and record[2] == group}


def signal_publishers(records, directory, script, lock_identity, stopped, environment="SOURCE_RUN_DIR"):
    candidates = {
        record[2] for pid, record in records.items()
        if record and record[1] != "Z" and record[2] not in stopped
        and matches(pid, directory, script, lock_identity, environment)
    }
    plans = {}
    # Validate every candidate group before signalling any process. A matching
    # child alone never authorizes a signal to an unrelated group leader.
    for group in candidates:
        if group <= 1 or group == os.getpgrp():
            raise ValueError("publisher has no isolated process group")
        members = live_members(records, group)
        leader = records.get(group)
        if leader is None and (Path("/proc") / str(group)).exists():
            raise ValueError(f"publisher group {group} has an unreadable leader")
        if leader and leader[1] != "Z" and (leader[2] != group or not matches(group, directory, script, lock_identity, environment)):
            raise ValueError(f"publisher group {group} has an unknown leader")
        for pid, record in members.items():
            if process(pid) != record or not matches(pid, directory, script, lock_identity, environment):
                raise ValueError(f"publisher group {group} has an unknown member {pid}")
        plans[group] = members
    current = snapshot()
    for group, members in plans.items():
        if live_members(current, group) != members:
            raise ValueError(f"publisher group {group} changed during validation")
        for pid, record in members.items():
            if process(pid) != record or not matches(pid, directory, script, lock_identity, environment):
                raise ValueError(f"publisher member {pid} changed during validation")
    final = snapshot()
    for group, members in plans.items():
        if live_members(final, group) != members:
            raise ValueError(f"publisher group {group} changed before signalling")
    for group, members in plans.items():
        try:
            if group in members:
                os.killpg(group, signal.SIGTERM)
            else:
                # A missing leader never authorizes a group signal. Check
                # each orphan's start time and exact identity immediately
                # before signalling only that PID.
                for pid, record in members.items():
                    if process(pid) != record or not matches(pid, directory, script, lock_identity, environment):
                        raise ValueError(f"orphan publisher {pid} changed before signalling")
                    os.kill(pid, signal.SIGTERM)
            stopped.add(group)
        except ProcessLookupError:
            pass


def drain(directory: str, script: str, timeout: float = 30.0, *, microbatch=False) -> None:
    root = Path(directory)
    if not root.is_absolute() or root.name != f"run_{root.parent.name}" or not Path(script).is_absolute():
        raise ValueError("publisher drain requires exact absolute run and script paths")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError(f"publisher drain refuses symlink: {path}")
    # Trainers have used both paths. Drain only this exact run, never other
    # runs found by a wildcard or a PID file.
    if microbatch:
        for directory in (root.parent, root):
            drain_directory(directory, script, timeout, "run_dir", ".microbatch-sync.lock")
    else:
        drain_directory(root, script, timeout, "SOURCE_RUN_DIR", ".batch-sync.lock")


def drain_directory(root, script, timeout, environment, lock_name):
    lock = root / lock_name
    descriptor = None
    try:
        try:
            descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            pass
        identity = None
        if descriptor is not None:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("publisher lock is not a regular file")
            identity = (info.st_dev, info.st_ino)
        groups = set()
        deadline = time.monotonic() + timeout
        while True:
            records = snapshot()
            signal_publishers(records, root, script, identity, groups, environment)
            alive = any(record and record[1] != "Z" and record[2] in groups for record in records.values())
            unlocked = descriptor is None
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    unlocked = True
                except BlockingIOError:
                    unlocked = False
            # The group must be gone as well as the lock free: a child can
            # close fd 9 before completing its final remote publication.
            if not alive and unlocked:
                if descriptor is not None and lock.stat().st_ino != identity[1]:
                    raise ValueError("publication lock changed during drain")
                print(f"publisher drained: {root}; groups={len(groups)}; lock=free")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"publisher group or publication lock did not drain: {root}")
            time.sleep(0.1)
    finally:
        if descriptor is not None:
            os.close(descriptor)


if __name__ == "__main__":
    try:
        if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--microbatch"):
            raise ValueError("expected exact run directory, publisher script, and optional --microbatch")
        drain(sys.argv[1], sys.argv[2], microbatch=len(sys.argv) == 4)
    except (OSError, ValueError) as error:
        raise SystemExit(f"publisher drain refused: {error}")
