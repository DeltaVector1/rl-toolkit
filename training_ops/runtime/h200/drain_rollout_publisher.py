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
        return root, fields[0], int(fields[2]), fields[19], int(fields[3])
    except (OSError, IndexError, ValueError):
        return None


def same_run(pid: int, directory: Path, environment: str) -> bool:
    try:
        values = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return f"{environment}={directory}".encode() in values
    except OSError:
        return False


def matches(pid: int, directory: Path, script: str, lock_identity, environment="SOURCE_RUN_DIR") -> bool:
    root = Path("/proc") / str(pid)
    try:
        if not same_run(pid, directory, environment):
            return False
        if script.encode() in (root / "cmdline").read_bytes().split(b"\0"):
            return True
        info = (root / "fd/9").stat()
        return lock_identity == (info.st_dev, info.st_ino)
    except OSError:
        return False


def snapshot():
    return {int(path.name): process(int(path.name)) for path in Path("/proc").iterdir() if path.name.isdecimal()}


def is_live(record):
    return bool(record) and record[1] not in {"Z", "X", "x"}


def live_members(records, group):
    return {pid: record for pid, record in records.items() if is_live(record) and record[2] == group}


def stable_identity(record):
    return (record[3], record[2], record[4]) if is_live(record) else None


def same_member(pid, record, group):
    current = process(pid)
    if current is None and pid == group and (Path("/proc") / str(pid)).exists():
        raise ValueError(f"publisher group {group} has an unreadable leader")
    if not is_live(current) or not is_live(record) or current[3] != record[3]:
        return False
    if current[4] != group:
        raise ValueError(f"publisher group {group} has an unknown member {pid}")
    return stable_identity(current) == stable_identity(record)


def same_members(records, members, group):
    if group in members and records.get(group) is None and (Path("/proc") / str(group)).exists():
        raise ValueError(f"publisher group {group} has an unreadable leader")
    current = live_members(records, group)
    for pid, record in current.items():
        if pid in members and record[3] != members[pid][3]:
            return False
        if record[4] != group:
            raise ValueError(f"publisher group {group} has an unknown member {pid}")
    return {pid: stable_identity(record) for pid, record in current.items()} == {
        pid: stable_identity(record) for pid, record in members.items()
    }


def pending_publishers(records, directory, environment, observed_groups):
    # A child can receive SIGTERM and close fd 9 before this drain starts.
    # Its exact run environment is enough to wait for completion, but never
    # enough to authorize a signal. Keep its group in the barrier even if a
    # later exec removes that environment or the original child exits first.
    observed_groups.update(
        record[2] for pid, record in records.items()
        if pid != os.getpid() and is_live(record)
        and same_run(pid, directory, environment)
    )
    return any(is_live(record) and record[2] in observed_groups for record in records.values())


def signal_publishers(records, directory, script, lock_identity, stopped, environment="SOURCE_RUN_DIR"):
    """Rescan changed plans; never signal a changed target or later targets."""
    candidates = {
        record[2] for pid, record in records.items()
        if is_live(record) and record[2] not in stopped
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
        isolated = is_live(leader)
        if isolated and (leader[2] != group or leader[4] != group):
            raise ValueError(f"publisher group {group} has an unknown leader")
        if isolated and not same_member(group, leader, group):
            return False
        if isolated and not matches(group, directory, script, lock_identity, environment):
            raise ValueError(f"publisher group {group} has an unknown leader")
        for pid, record in members.items():
            # An exact session leader owns its same-session children, including
            # ssh children that close fd 9 and no longer name the script.
            owned = record[4] == group and (isolated or matches(pid, directory, script, lock_identity, environment))
            if not same_member(pid, record, group):
                return False
            if not owned:
                raise ValueError(f"publisher group {group} has an unknown member {pid}")
        plans[group] = members
    for _ in range(2):
        current = snapshot()
        if any(
            is_live(record) and record[2] not in plans and record[2] not in stopped
            and matches(pid, directory, script, lock_identity, environment)
            for pid, record in current.items()
        ):
            return False
        for group, members in plans.items():
            if not same_members(current, members, group):
                return False
            for pid, record in members.items():
                if not same_member(pid, record, group):
                    return False
                exact = pid == group or group not in members
                if exact and not matches(pid, directory, script, lock_identity, environment):
                    raise ValueError(f"publisher member {pid} changed exact ownership during validation")
    # Check all orphan targets together before sending any group or PID signal.
    for group, members in plans.items():
        if group not in members:
            for pid, record in members.items():
                if not same_member(pid, record, group):
                    return False
                if not matches(pid, directory, script, lock_identity, environment):
                    raise ValueError(f"orphan publisher {pid} changed exact ownership before signalling")
    for group, members in plans.items():
        try:
            if group in members:
                # Earlier signals may have changed this later group's members.
                if not same_members(snapshot(), members, group):
                    return False
                for pid, record in members.items():
                    if not same_member(pid, record, group):
                        return False
                if not matches(group, directory, script, lock_identity, environment):
                    raise ValueError(f"publisher leader {group} changed exact ownership before signal")
                if not same_members(snapshot(), members, group) or not same_member(group, members[group], group):
                    return False
                os.killpg(group, signal.SIGTERM)
            else:
                # A missing leader never authorizes a group signal. Only the
                # exact orphan PIDs in the fully rechecked plan may be signalled.
                for pid, record in members.items():
                    if not same_member(pid, record, group):
                        return False
                    if not matches(pid, directory, script, lock_identity, environment):
                        raise ValueError(f"orphan publisher {pid} changed exact ownership before signal")
                    if not same_member(pid, record, group):
                        return False
                    os.kill(pid, signal.SIGTERM)
            stopped.add(group)
        except ProcessLookupError:
            return False
    return True


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
        observed_groups = set()
        deadline = time.monotonic() + timeout
        while True:
            records = snapshot()
            pending = pending_publishers(records, root, environment, observed_groups)
            if not signal_publishers(records, root, script, identity, groups, environment):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"publisher membership did not settle: {root}")
                time.sleep(0.1)
                continue
            alive = any(is_live(record) and record[2] in groups for record in records.values())
            unlocked = descriptor is None
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    unlocked = True
                except BlockingIOError:
                    unlocked = False
            # The group must be gone as well as the lock free: a child can
            # close fd 9 before completing its final remote publication.
            if not alive and not pending and unlocked:
                # Check once more after the lock probe so a child born during
                # the first /proc scan cannot bypass the completion barrier.
                if pending_publishers(snapshot(), root, environment, observed_groups):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"publisher group did not drain: {root}")
                    continue
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
