from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from ornith_curriculum_v1.models import (
    ExternalCallAttempt,
    ExternalCallResult,
)


external_call_attempt_count = 3


async def request_external_model(
    request: Callable[[], Awaitable[object]],
    validate: Callable[[object], object | None],
    *,
    attempt_timeout_seconds: float = 180.0,
    total_timeout_seconds: float = 550.0,
) -> ExternalCallResult:
    attempts: list[ExternalCallAttempt] = []
    try:
        async with asyncio.timeout(total_timeout_seconds):
            for attempt_number in range(1, external_call_attempt_count + 1):
                started_at = time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        request(),
                        timeout=attempt_timeout_seconds,
                    )
                except Exception as error:
                    attempts.append(
                        ExternalCallAttempt(
                            attempt=attempt_number,
                            status="request_failed",
                            elapsed_seconds=time.monotonic() - started_at,
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    )
                else:
                    try:
                        value = validate(response)
                    except Exception as error:
                        attempts.append(
                            ExternalCallAttempt(
                                attempt=attempt_number,
                                status="invalid_response",
                                elapsed_seconds=time.monotonic() - started_at,
                                error_type=type(error).__name__,
                                error=str(error),
                                response_text=str(response),
                            )
                        )
                        value = None
                    else:
                        attempts.append(
                            ExternalCallAttempt(
                                attempt=attempt_number,
                                status=(
                                    "completed"
                                    if value is not None
                                    else "invalid_response"
                                ),
                                elapsed_seconds=time.monotonic() - started_at,
                                response_text=str(response),
                            )
                        )
                    if value is not None:
                        return ExternalCallResult(
                            status="completed",
                            value=value,
                            attempts=tuple(attempts),
                        )
                if attempt_number < external_call_attempt_count:
                    await asyncio.sleep(0.5 * attempt_number)
    except TimeoutError:
        pass
    invalid_model_response = (
        len(attempts) == external_call_attempt_count
        and all(attempt.status == "invalid_response" for attempt in attempts)
    )
    return ExternalCallResult(
        status=(
            "invalid_model_response"
            if invalid_model_response
            else "unscored_service_failure"
        ),
        value=None,
        attempts=tuple(attempts),
    )
