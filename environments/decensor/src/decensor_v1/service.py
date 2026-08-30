from __future__ import annotations

import asyncio
import fcntl
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from decensor_v1.scoring import EvaluationResult


_judge_slot_directory = Path(tempfile.gettempdir()) / f"decensor-v1-judge-{os.getuid()}"


def _judge_capacity(config: Any | None) -> int:
    capacity = getattr(config, "max_concurrent_requests", None)
    if capacity is None:
        capacity = getattr(config, "max_concurrent", None)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise RuntimeError("judge request capacity is not configured")
    return capacity


def _try_acquire_slot(path: Path) -> int | None:
    _judge_slot_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


async def _acquire_judge_slot(capacity: int) -> int:
    paths = [
        _judge_slot_directory / f"slot-{slot}.lock"
        for slot in range(capacity)
    ]
    while True:
        for path in paths:
            descriptor = _try_acquire_slot(path)
            if descriptor is not None:
                return descriptor
        await asyncio.sleep(0.01)


def _release_judge_slot(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@asynccontextmanager
async def judge_request_slot(config: Any | None):
    descriptor = await _acquire_judge_slot(_judge_capacity(config))
    try:
        yield
    finally:
        _release_judge_slot(descriptor)


async def request_judge_score(
    name: str,
    request: Callable[[], Awaitable[Any]],
    *,
    judge_config: Any | None = None,
    attempt_count: int = 3,
    attempt_timeout_seconds: float = 180.0,
    total_timeout_seconds: float = 550.0,
) -> EvaluationResult:
    attempts: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(total_timeout_seconds):
            for attempt_number in range(1, attempt_count + 1):
                started_at = time.monotonic()
                try:
                    async with judge_request_slot(judge_config):
                        response = await asyncio.wait_for(
                            request(),
                            timeout=attempt_timeout_seconds,
                        )
                except Exception as error:
                    attempts.append(
                        {
                            "name": name,
                            "attempt": attempt_number,
                            "status": "request_failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "elapsed_seconds": time.monotonic() - started_at,
                        }
                    )
                else:
                    attempt_record = {
                        "name": name,
                        "attempt": attempt_number,
                        "status": (
                            "scored"
                            if response.parsed is not None
                            else "invalid_judge_response"
                        ),
                        "text": response.text or "",
                        "parsed": response.parsed,
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                    events = getattr(response, "events", None)
                    if events is not None:
                        attempt_record["events"] = events
                    attempts.append(attempt_record)
                    if response.parsed is not None:
                        return EvaluationResult(
                            status="scored",
                            score=response.parsed,
                            details={"attempts": attempts},
                        )
                if attempt_number < attempt_count:
                    await asyncio.sleep(0.5 * attempt_number)
    except TimeoutError:
        attempts.append(
            {
                "name": name,
                "attempt": None,
                "status": "total_timeout",
            }
        )
    return EvaluationResult(
        status="unscored_service_failure",
        details={"attempts": attempts},
    )
