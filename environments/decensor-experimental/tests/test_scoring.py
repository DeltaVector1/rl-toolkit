from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decensor_experimental_v1.scoring import ScoringConfig, Verdicts, score_decensor
from decensor_v1.scoring import score_decensor as score_original


def _score(
    *,
    content: str = " ".join(["Answer"] * 100),
    refusal: int = 1,
    response_format: str = "plaintext",
    **updates,
):
    verdicts = Verdicts(
        refusal=refusal,
        visible_answer_is_clear=1,
        reasoning_has_safety_or_policy_discussion=0,
        reasoning_trace_is_clear=1,
        meta_commentary=0,
        markdown_present=0,
        markdown_is_correct=1,
    )
    verdicts = replace(verdicts, **updates)
    return score_decensor(
        "Explain the requested steps.",
        [{"role": "assistant", "content": content, "reasoning": "I will answer directly."}],
        verdicts,
        response_format=response_format,
        word_requirement="",
        is_truncated=False,
    )


def test_refusal_primary_formula_keeps_all_quality_components():
    result = _score(
        refusal=2,
        visible_answer_is_clear=0,
        reasoning_has_safety_or_policy_discussion=1,
        meta_commentary=1,
    )

    assert result.status == "scored"
    assert result.details["reward_scheme"] == "refusal_primary_quality_tiebreak_v1"
    assert result.details["secondary_quality"] == pytest.approx(0.0001)
    assert result.details["quality_bonus"] == pytest.approx(0.01 * 0.0001)
    assert result.reward == pytest.approx(0.8 + 0.01 * 0.0001)
    assert result.details["final_reward_formula"] == "base_reward + 0.01 * secondary_quality"
    assert result.details["secondary_quality_formula"].count("*") == 5


def test_rejected_response_still_has_zero_reward():
    result = _score(content="")
    assert result.status == "invalid_model_response"
    assert result.reward == 0.0
    assert result.details["final_reward"] == 0.0
    assert "secondary_quality" not in result.details


def test_missing_judge_score_is_unscored_without_reward():
    result = _score(refusal=None)
    assert result.status == "unscored_service_failure"
    assert result.reward is None
    assert result.details["missing_judge_scores"] == ["refusal"]


def test_lower_refusal_wins_with_worst_quality():
    lower_refusal = _score(
        refusal=1,
        visible_answer_is_clear=0,
        reasoning_has_safety_or_policy_discussion=1,
        meta_commentary=1,
    )
    higher_refusal = _score(refusal=2)

    assert lower_refusal.reward > higher_refusal.reward


def test_equal_refusal_uses_secondary_quality_as_tie_break():
    best = _score(refusal=2)
    worst = _score(
        refusal=2,
        visible_answer_is_clear=0,
        reasoning_has_safety_or_policy_discussion=1,
        meta_commentary=1,
    )

    assert best.reward > worst.reward
    assert best.details["base_reward"] == worst.details["base_reward"]


def test_unclear_reasoning_gets_no_bonus_but_keeps_refusal_base():
    result = _score(refusal=3, reasoning_trace_is_clear=0)
    assert result.details["secondary_quality"] == 0.0
    assert result.details["quality_bonus"] == 0.0
    assert result.reward == pytest.approx(result.details["base_reward"])


def test_original_decensor_scoring_is_unchanged():
    experimental = _score(refusal=2)
    original = score_original(
        "Explain the requested steps.",
        [{"role": "assistant", "content": " ".join(["Answer"] * 100), "reasoning": "I will answer directly."}],
        Verdicts(
            refusal=2,
            visible_answer_is_clear=1,
            reasoning_has_safety_or_policy_discussion=0,
            reasoning_trace_is_clear=1,
            meta_commentary=0,
            markdown_present=0,
            markdown_is_correct=1,
        ),
        response_format="plaintext",
        word_requirement="",
        is_truncated=False,
    )

    assert original.details["final_reward_formula"].startswith("base_reward *")
    assert original.reward == pytest.approx(0.8)
    assert experimental.reward == pytest.approx(0.81)
