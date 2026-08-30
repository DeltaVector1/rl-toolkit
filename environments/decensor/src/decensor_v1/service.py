from __future__ import annotations

import asyncio
import fcntl
import os
import platform
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from decensor_v1.scoring import EvaluationResult


try:
    _judge_scope = f"pidns-{os.stat('/proc/self/ns/pid').st_ino}"
except OSError:
    _judge_scope = os.environ.get("HOSTNAME") or platform.node() or "default"
_judge_scope = "".join(
    character if character.isalnum() or character in "-_." else "_"
    for character in _judge_scope
)
_judge_slot_directory = (
    Path(tempfile.gettempdir()) / f"decensor-v1-judge-{os.getuid()}-{_judge_scope}"
)
_judge_capacity_file = "capacity"
_judge_capacity_lock_file = "capacity.lock"


def _safe_service_error(error: object) -> str:
    message = " ".join(str(error).split())
    message = re.sub(
        r"(?i)(authorization|api[-_ ]?key|token)(\s*[:=]\s*)([^, }]+)",
        r"\1\2[removed]",
        message,
    )
    return message[:400]


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


def _ensure_judge_capacity(capacity: int) -> None:
    _judge_slot_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = _judge_slot_directory / _judge_capacity_lock_file
    capacity_path = _judge_slot_directory / _judge_capacity_file
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        configured = capacity_path.read_text().strip() if capacity_path.exists() else ""
        if configured:
            try:
                configured_capacity = int(configured)
            except ValueError as error:
                raise RuntimeError(
                    "judge request capacity metadata is invalid"
                ) from error
            if configured_capacity != capacity:
                raise RuntimeError(
                    "judge request capacity differs from the shared worker capacity"
                )
        else:
            capacity_path.write_text(f"{capacity}\n")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


async def _acquire_judge_slot(capacity: int) -> int:
    _ensure_judge_capacity(capacity)
    paths = [_judge_slot_directory / f"slot-{slot}.lock" for slot in range(capacity)]
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


def judge_failure_summary(name: str, outcome: EvaluationResult) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    attempts = outcome.details.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    for record in attempts:
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if not isinstance(status, str):
            status = "unknown"
        failure_types: list[str] = []
        error_message: str | None = None
        if status == "request_failed":
            error_type = record.get("error_type")
            if isinstance(error_type, str) and error_type:
                failure_types.append(error_type)
            error_message = record.get("error")
            if isinstance(error_message, str) and error_message:
                error_message = _safe_service_error(error_message)
            else:
                error_message = None
        elif status == "invalid_judge_response":
            events = record.get("events", [])
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    if event.get("status") != "invalid_judge_response":
                        continue
                    error_type = event.get("error_type")
                    if (
                        isinstance(error_type, str)
                        and error_type
                        and error_type not in failure_types
                    ):
                        failure_types.append(error_type)
        elif status == "total_timeout":
            failure_types.append("TimeoutError")
        if not failure_types:
            failure_types.append(status)
        elapsed_seconds = record.get("elapsed_seconds")
        if isinstance(elapsed_seconds, (int, float)) and not isinstance(
            elapsed_seconds, bool
        ):
            elapsed_seconds = round(float(elapsed_seconds), 3)
        else:
            elapsed_seconds = None
        attempt = record.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            attempt = None
        summaries.append(
            {
                "attempt": attempt,
                "status": status,
                "failure_types": failure_types,
                "error": error_message,
                "elapsed_seconds": elapsed_seconds,
            }
        )
    return {"name": name, "attempts": summaries}


def format_judge_failure_summary(summaries: list[dict[str, Any]]) -> str:
    review_messages: list[str] = []
    for summary in summaries:
        name = summary.get("name", "unknown_review")
        attempt_messages: list[str] = []
        for attempt in summary.get("attempts", []):
            attempt_number = attempt.get("attempt")
            attempt_label = (
                str(attempt_number) if attempt_number is not None else "unknown"
            )
            failure_types = ",".join(attempt.get("failure_types", ["unknown"]))
            elapsed_seconds = attempt.get("elapsed_seconds")
            elapsed_label = (
                f"{float(elapsed_seconds):.1f}s"
                if isinstance(elapsed_seconds, (int, float))
                and not isinstance(elapsed_seconds, bool)
                else "unknown duration"
            )
            attempt_messages.append(
                f"attempt {attempt_label}: {failure_types} ({elapsed_label})"
                + (f" — {attempt['error']}" if attempt.get("error") else "")
            )
        review_messages.append(
            f"{name}: " + ("; ".join(attempt_messages) or "no attempt record")
        )
    return "; ".join(review_messages) or "no failure details"


async def request_judge_score(
    name: str,
    request: Callable[[], Awaitable[Any]],
    *,
    attempt_count: int = 3,
    total_timeout_seconds: float = 550.0,
) -> EvaluationResult:
    attempts: list[dict[str, Any]] = []
    request_started_at = time.monotonic()
    try:
        async with asyncio.timeout(total_timeout_seconds):
            for attempt_number in range(1, attempt_count + 1):
                attempt_started_at = time.monotonic()
                try:
                    response = await request()
                except Exception as error:
                    attempts.append(
                        {
                            "name": name,
                            "attempt": attempt_number,
                            "status": "request_failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "elapsed_seconds": time.monotonic() - attempt_started_at,
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
                        "elapsed_seconds": time.monotonic() - attempt_started_at,
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
                "error_type": "TimeoutError",
                "elapsed_seconds": time.monotonic() - request_started_at,
            }
        )
    return EvaluationResult(
        status="unscored_service_failure",
        details={"attempts": attempts},
    )
