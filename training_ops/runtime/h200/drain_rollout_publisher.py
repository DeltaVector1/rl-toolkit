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


def matches(pid: int, directory: Path, script: str, lock_identity) -> bool:
    root = Path("/proc") / str(pid)
    try:
        if f"SOURCE_RUN_DIR={directory}".encode() not in (root / "environ").read_bytes().split(b"\0"):
            return False
        if script.encode() in (root / "cmdline").read_bytes().split(b"\0"):
            return True
        info = (root / "fd/9").stat()
        return lock_identity == (info.st_dev, info.st_ino)
    except OSError:
        return False


def drain(directory: str, script: str, timeout: float = 30.0) -> None:
    root = Path(directory)
    if not root.is_absolute() or root.name != f"run_{root.parent.name}" or not Path(script).is_absolute():
        raise ValueError("publisher drain requires exact absolute run and script paths")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError(f"publisher drain refuses symlink: {path}")
    lock = root / ".batch-sync.lock"
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
            snapshot = {int(path.name): process(int(path.name)) for path in Path("/proc").iterdir() if path.name.isdecimal()}
            for pid, record in snapshot.items():
                if record is None or record[1] == "Z" or not matches(pid, root, script, identity):
                    continue
                group = record[2]
                if group <= 1 or group == os.getpgrp():
                    raise ValueError("publisher has no isolated process group")
                if group not in groups:
                    # Recheck start time, group and exact run/lock identity
                    # immediately before signalling; a PID file is not proof.
                    if process(pid) != record or not matches(pid, root, script, identity):
                        continue
                    groups.add(group)
                    try:
                        os.killpg(group, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            alive = any(record and record[1] != "Z" and record[2] in groups for record in snapshot.values())
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
        drain(sys.argv[1], sys.argv[2])
    except (OSError, ValueError) as error:
        raise SystemExit(f"publisher drain refused: {error}")
