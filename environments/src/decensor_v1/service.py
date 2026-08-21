from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from decensor_v1.scoring import EvaluationResult


async def request_judge_score(
    name: str,
    request: Callable[[], Awaitable[Any]],
    *,
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
