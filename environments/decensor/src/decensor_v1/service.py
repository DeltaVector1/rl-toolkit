from __future__ import annotations

import asyncio
import fcntl
import os
import platform
import re
import tempfile
import threading
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
_judge_capacity_owner_prefix = "worker-"
_judge_capacity_owner_suffix = ".lock"
_judge_capacity_state_lock = threading.RLock()
_judge_slot_descriptors_lock = threading.RLock()
_judge_slot_descriptors: set[int] = set()
_judge_capacity_owner_descriptor: int | None = None
_judge_capacity_owner_path: Path | None = None
_judge_capacity_owner_pid: int | None = None
_judge_capacity_owner_value: int | None = None


def safe_service_error(error: object) -> str:
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
    with _judge_slot_descriptors_lock:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        _judge_slot_descriptors.add(descriptor)
    return descriptor


def _close_inherited_capacity_owner() -> None:
    global _judge_capacity_owner_descriptor
    global _judge_capacity_owner_path
    global _judge_capacity_owner_pid
    global _judge_capacity_owner_value

    if _judge_capacity_owner_descriptor is not None:
        os.close(_judge_capacity_owner_descriptor)
    _judge_capacity_owner_descriptor = None
    _judge_capacity_owner_path = None
    _judge_capacity_owner_pid = None
    _judge_capacity_owner_value = None


def _prepare_judge_fork() -> None:
    _judge_capacity_state_lock.acquire()
    _judge_slot_descriptors_lock.acquire()


def _release_judge_fork_locks() -> None:
    _judge_slot_descriptors_lock.release()
    _judge_capacity_state_lock.release()


def _reset_judge_capacity_after_fork() -> None:
    global _judge_capacity_state_lock
    global _judge_slot_descriptors_lock

    _close_inherited_capacity_owner()
    for descriptor in _judge_slot_descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _judge_slot_descriptors.clear()
    _judge_capacity_state_lock = threading.RLock()
    _judge_slot_descriptors_lock = threading.RLock()


os.register_at_fork(
    before=_prepare_judge_fork,
    after_in_parent=_release_judge_fork_locks,
    after_in_child=_reset_judge_capacity_after_fork,
)


def _capacity_owner_path() -> Path:
    return _judge_slot_directory / (
        f"{_judge_capacity_owner_prefix}{os.getpid()}"
        f"{_judge_capacity_owner_suffix}"
    )


def _locked_capacity_owner(path: Path) -> tuple[bool, int | None]:
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except FileNotFoundError:
        return False, None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                value = int(os.read(descriptor, 4096).decode().strip())
            except (OSError, UnicodeError, ValueError):
                value = None
            return True, value
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False, None
    finally:
        os.close(descriptor)


def _active_capacity_owners(
    owner_path: Path,
) -> tuple[set[int], bool]:
    capacities: set[int] = set()
    unknown_capacity = False
    for path in _judge_slot_directory.glob(
        f"{_judge_capacity_owner_prefix}*{_judge_capacity_owner_suffix}"
    ):
        if path == owner_path:
            continue
        active, value = _locked_capacity_owner(path)
        if active:
            if value is None or value < 1:
                unknown_capacity = True
            else:
                capacities.add(value)
        else:
            path.unlink(missing_ok=True)
    return capacities, unknown_capacity


def _active_judge_slots() -> bool:
    for path in _judge_slot_directory.glob("slot-*.lock"):
        active, _ = _locked_capacity_owner(path)
        if active:
            return True
    return False


def _ensure_judge_capacity(capacity: int) -> None:
    global _judge_capacity_owner_descriptor
    global _judge_capacity_owner_path
    global _judge_capacity_owner_pid
    global _judge_capacity_owner_value

    _judge_slot_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner_path = _capacity_owner_path()
    with _judge_capacity_state_lock:
        if (
            _judge_capacity_owner_pid != os.getpid()
            or _judge_capacity_owner_path != owner_path
        ):
            _close_inherited_capacity_owner()
        if _judge_capacity_owner_value is not None:
            if _judge_capacity_owner_value != capacity:
                raise RuntimeError(
                    "judge request capacity differs from the shared worker capacity"
                )
            return

        lock_path = _judge_slot_directory / _judge_capacity_lock_file
        capacity_path = _judge_slot_directory / _judge_capacity_file
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        owner_descriptor: int | None = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            active_capacities, unknown_capacity = _active_capacity_owners(owner_path)
            configured = (
                capacity_path.read_text().strip() if capacity_path.exists() else ""
            )
            configured_capacity: int | None = None
            if configured:
                try:
                    configured_capacity = int(configured)
                except ValueError:
                    if unknown_capacity or _active_judge_slots():
                        raise RuntimeError(
                            "judge request capacity metadata is invalid"
                        )

            if unknown_capacity:
                raise RuntimeError(
                    "judge request capacity metadata is invalid"
                )
            if active_capacities and any(
                owner_capacity != capacity for owner_capacity in active_capacities
            ):
                raise RuntimeError(
                    "judge request capacity differs from the shared worker capacity"
                )
            if not active_capacities and _active_judge_slots():
                if configured_capacity is None:
                    raise RuntimeError(
                        "active judge requests have unknown worker capacity"
                    )
                if configured_capacity != capacity:
                    raise RuntimeError(
                        "judge request capacity differs from the shared worker capacity"
                    )

            owner_descriptor = os.open(owner_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(owner_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("judge capacity owner is already active") from error
            os.ftruncate(owner_descriptor, 0)
            os.write(owner_descriptor, f"{capacity}\n".encode())
            capacity_path.write_text(f"{capacity}\n")
            _judge_capacity_owner_descriptor = owner_descriptor
            _judge_capacity_owner_path = owner_path
            _judge_capacity_owner_pid = os.getpid()
            _judge_capacity_owner_value = capacity
            owner_descriptor = None
        finally:
            if owner_descriptor is not None:
                os.close(owner_descriptor)
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
    with _judge_slot_descriptors_lock:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            _judge_slot_descriptors.discard(descriptor)


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
                error_message = safe_service_error(error_message)
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
                            "error": safe_service_error(error),
                            "elapsed_seconds": time.monotonic() - attempt_started_at,
                        }
                    )
                    if attempt_number < attempt_count:
                        await asyncio.sleep(0.5 * attempt_number)
                    continue
                failure_status = getattr(response, "failure_status", None)
                attempt_record = {
                    "name": name,
                    "attempt": attempt_number,
                    "status": (
                        "scored"
                        if response.parsed is not None
                        else "invalid_judge_response"
                        if failure_status == "invalid_model_response"
                        else "request_failed"
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
    status = (
        "invalid_model_response"
        if any(record.get("status") == "invalid_judge_response" for record in attempts)
        else "unscored_service_failure"
    )
    return EvaluationResult(status=status, details={"attempts": attempts})
