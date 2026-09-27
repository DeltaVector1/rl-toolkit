from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decensor_experimental_v1 import agentic_review


def _logged_payloads(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    records = [
        record
        for record in caplog.records
        if record.name == agentic_review.__name__
        and record.levelno == logging.INFO
        and record.message.startswith("decensor_review_failure ")
    ]
    return [
        json.loads(record.message.removeprefix("decensor_review_failure "))
        for record in records
    ]


def _logged_payload(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    payloads = _logged_payloads(caplog)
    assert len(payloads) == 1
    return payloads[0]


def _review_source() -> agentic_review.ReviewSource:
    return agentic_review.ReviewSource(
        response_format="plaintext",
        documents={
            "turn_1_user_context": "User request.",
            "turn_1_answer": "Answer.",
            "turn_1_reasoning": "Reasoning.",
        },
        turns=(
            agentic_review.ReviewTurn(
                turn=1,
                user_document="turn_1_user_context",
                reasoning_document="turn_1_reasoning",
                answer_document="turn_1_answer",
            ),
        ),
    )


def _completion(tool_name: str, arguments: str) -> SimpleNamespace:
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name=tool_name, arguments=arguments)
            )
        ],
        content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")],
        usage=None,
    )


class _FakeClient:
    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = iter(responses)

    def with_options(self, **_kwargs):
        return self

    @property
    def chat(self):
        return SimpleNamespace(completions=self)

    async def create(self, **_kwargs):
        return next(self._responses)

    async def close(self):
        return None


def _review_config() -> SimpleNamespace:
    return SimpleNamespace(
        model="review-model",
        timeout_seconds=20.0,
        max_concurrent_requests=1,
        seed=0,
        max_tokens=32,
        reasoning_effort=None,
        min_p=None,
        disable_thinking=False,
        clear_thinking=False,
    )


@pytest.mark.parametrize("phase", ["visible_review", "reasoning_review"])
def test_invalid_tool_arguments_emit_bounded_evidence(
    caplog: pytest.LogCaptureFixture, phase: str
):
    raw_arguments = '{"payload":"do-not-log-this", "token":"secret-value"}'
    with caplog.at_level(logging.INFO, logger=agentic_review.__name__):
        agentic_review.log_review_failure(
            phase=phase,
            attempt=3,
            error=ValueError("invalid value supplied: secret-value"),
            raw_arguments=raw_arguments,
            finish_reason="length",
            truncated=True,
            failure_status="invalid_model_response",
        )

    payload = _logged_payload(caplog)
    assert payload["phase"] == phase
    assert payload["attempt"] == 3
    assert payload["exception_class"] == "ValueError"
    assert payload["failure_status"] == "invalid_model_response"
    assert payload["raw_arguments_chars"] == len(raw_arguments)
    assert payload["raw_arguments_bytes"] == len(raw_arguments.encode("utf-8"))
    assert payload["raw_arguments_sha256"] == hashlib.sha256(
        raw_arguments.encode("utf-8")
    ).hexdigest()
    assert payload["finish_reason"] == "length"
    assert payload["truncated"] is True
    serialized = caplog.records[0].message
    assert "do-not-log-this" not in serialized
    assert "secret-value" not in serialized


def test_task_review_failure_logs_available_identifiers(
    caplog: pytest.LogCaptureFixture,
):
    class Trace:
        task_id = "task-42"
        group_id = "group-7"

    with caplog.at_level(logging.INFO, logger=agentic_review.__name__):
        agentic_review.log_task_review_failure(
            Trace(),
            failed_reviews=["turn_1_review"],
            failure_status="unscored_service_failure",
        )

    payload = _logged_payload(caplog)
    assert payload["phase"] == "task_review_failure"
    assert payload["failure_status"] == "unscored_service_failure"
    assert payload["task_id"] == "task-42"
    assert payload["group_id"] == "group-7"
    assert payload["failed_reviews"] == ["turn_1_review"]
    assert payload["raw_arguments_chars"] == 0
    assert payload["raw_arguments_bytes"] == 0
    assert payload["finish_reason"] is None
    assert payload["truncated"] is None


def test_invalid_visible_arguments_are_logged_by_review_runner(
    caplog: pytest.LogCaptureFixture,
):
    import verifiers.v1.clients.base as client_base

    raw_arguments = '{"secret":"visible-secret"}'
    client = _FakeClient(
        [_completion("submit_visible_review", raw_arguments) for _ in range(3)]
    )
    with (
        mock.patch.object(client_base, "build_async_openai", return_value=client),
        caplog.at_level(logging.INFO, logger=agentic_review.__name__),
    ):
        response = asyncio.run(
            agentic_review.run_agentic_review(_review_config(), _review_source())
        )

    assert response.failure_status == "invalid_model_response"
    payloads = _logged_payloads(caplog)
    assert len(payloads) == 3
    assert {payload["phase"] for payload in payloads} == {"visible_review"}
    assert all(payload["raw_arguments_chars"] == len(raw_arguments) for payload in payloads)
    assert all("visible-secret" not in record.message for record in caplog.records)


def test_invalid_reasoning_arguments_are_logged_by_review_runner(
    caplog: pytest.LogCaptureFixture,
):
    import verifiers.v1.clients.base as client_base

    visible_arguments = json.dumps(
        {
            "turn": 1,
            "refusal_level": 1,
            "refusal_reason": "The answer fulfills the core request.",
            "request_outcome": "fulfilled",
            "fulfilled_separate_requested_part": False,
            "exact_measurement_or_quote_requested": False,
            "exact_measurement_or_quote_provided": False,
            "explicitly_refused_original_request": False,
            "silently_replaced_original_request": False,
            "silent_replacement_reason": "",
            "refusal_answer_evidence_ids": [],
            "visible_answer_is_clear": "pass",
            "visible_answer_is_clear_reason": "",
            "visible_answer_is_clear_evidence_ids": [],
            "meta_commentary": "absent",
            "meta_commentary_reason": "",
            "meta_commentary_evidence_ids": [],
            "markdown_quality": "not_applicable",
            "markdown_quality_reason": "",
            "markdown_quality_evidence_ids": [],
        }
    )
    raw_arguments = '{"secret":"reasoning-secret"}'
    client = _FakeClient(
        [_completion("submit_visible_review", visible_arguments)]
        + [_completion("submit_reasoning_review", raw_arguments) for _ in range(3)]
    )
    with (
        mock.patch.object(client_base, "build_async_openai", return_value=client),
        caplog.at_level(logging.INFO, logger=agentic_review.__name__),
    ):
        response = asyncio.run(
            agentic_review.run_agentic_review(_review_config(), _review_source())
        )

    assert response.failure_status == "invalid_model_response"
    payloads = _logged_payloads(caplog)
    assert len(payloads) == 3
    assert {payload["phase"] for payload in payloads} == {"reasoning_review"}
    assert all(payload["raw_arguments_chars"] == len(raw_arguments) for payload in payloads)
    assert all("reasoning-secret" not in record.message for record in caplog.records)
