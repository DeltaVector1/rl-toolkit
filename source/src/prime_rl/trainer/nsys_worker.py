"""Select one torchrun worker for application-controlled Nsight profiling."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import NoReturn


_TRAINER_MODULE = "prime_rl.trainer.rl.train"
_NSYS_ROOT = Path("/opt/nvidia/nsight-systems")


def _unsigned(name: str, value: str) -> int:
    if not value.isdecimal():
        raise RuntimeError(f"{name} must be an unsigned integer")
    return int(value)


def _exec_python() -> NoReturn:
    os.execv(
        sys.executable,
        [sys.executable, "-m", _TRAINER_MODULE, *sys.argv[1:]],
    )


def _exec_nsys() -> NoReturn:
    binary_value = os.environ.get("PRIME_NSYS_BIN", "")
    version = os.environ.get("PRIME_NSYS_VERSION", "")
    output = os.environ.get("PRIME_NSYS_OUTPUT", "")
    session = os.environ.get("PRIME_NSYS_SESSION", "")
    binary = Path(binary_value)
    if (
        not binary.is_absolute()
        or _NSYS_ROOT not in binary.parents
        or not binary.is_file()
        or not os.access(binary, os.X_OK)
        or re.search(r"\b[0-9]+\.[0-9]+\.[0-9]+\b", version) is None
        or not Path(output).is_absolute()
        or re.fullmatch(r"[A-Za-z0-9_.-]+", session) is None
    ):
        raise RuntimeError(
            "Nsight binary/path/version evidence is invalid: "
            f"path={binary_value!r} version={version!r}"
        )
    if os.environ.get("PRIME_CUDA_PROFILE_SECONDS", ""):
        raise RuntimeError("Nsight capture does not accept an external duration")
    command = [
        str(binary),
        "profile",
        f"--session-new={session}",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--event-sample=none",
        "--cpuctxsw=none",
        "--backtrace=none",
        "--cudabacktrace=none",
        "--cuda-memory-usage=false",
        "--flush-on-cudaprofilerstop=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--stop-on-exit=true",
        "--kill=none",
        "--force-overwrite=true",
        "-o",
        output,
        sys.executable,
        "-m",
        _TRAINER_MODULE,
        *sys.argv[1:],
    ]
    os.execv(binary, command)


def main() -> None:
    rank = _unsigned("RANK", os.environ.get("RANK", ""))
    target = _unsigned(
        "PRIME_CUDA_PROFILE_RANK",
        os.environ.get("PRIME_CUDA_PROFILE_RANK", "24"),
    )
    if os.environ.get("PRIME_CUDA_PROFILE", "0") != "1" or rank != target:
        _exec_python()
        return
    if target != 24:
        raise RuntimeError("Nsight capture worker is pinned to rank 24")
    _exec_nsys()


if __name__ == "__main__":
    main()
