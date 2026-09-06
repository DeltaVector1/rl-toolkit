from __future__ import annotations

import json

import verifiers.v1 as vf
from pydantic import BaseModel, Field
from verifiers.v1.types import SamplingConfig


class TestingPerfJudgeSampling(SamplingConfig):
    temperature: float = 0.0
    max_tokens: int = 256


class TestingPerfJudgeConfig(vf.JudgeConfig):
    """OpenAI-compatible endpoint and bounded retry settings for the judge."""

    # A local placeholder prevents an unconfigured check from sending paid traffic.
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_var: str = "TESTING_PERF_JUDGE_API_KEY"
    model: str = "configured-judge"
    sampling: TestingPerfJudgeSampling = TestingPerfJudgeSampling()
    max_attempts: int = Field(default=3, ge=1, le=3)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=5)


class TurnReview(BaseModel):
    """One strict, visible-output review returned by the judge."""

    correctness: float = Field(ge=0, le=1)
    task_adherence: float = Field(ge=0, le=1)
    coherence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(min_length=1, max_length=4)


class TestingPerfJudge(vf.Judge[TurnReview, TestingPerfJudgeConfig]):
    """Parse one JSON review; no fallback score exists for malformed output."""

    def parse(self, response: vf.JudgeResponse[TurnReview]) -> TurnReview:
        try:
            value = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise ValueError("judge response is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("judge response must be a JSON object")
        return TurnReview.model_validate(value)


__all__ = [
    "TestingPerfJudge",
    "TestingPerfJudgeConfig",
    "TestingPerfJudgeSampling",
    "TurnReview",
]
