from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


root = Path(__file__).resolve().parents[1]
package_root = root / "src" / "decensor_v1"
package = types.ModuleType("decensor_v1")
package.__path__ = [str(package_root)]
sys.modules.setdefault("decensor_v1", package)
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *args, **kwargs: []
sys.modules.setdefault("datasets", datasets)
def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, package_root / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


text = load("decensor_v1.text", "text.py")
scoring = load("decensor_v1.scoring", "scoring.py")
service = load("decensor_v1.service", "service.py")
data = load("decensor_v1.data", "data.py")
review = load("decensor_v1.agentic_review", "agentic_review.py")


def assistant(content: str, reasoning: str | None = "reasoning"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def verdicts(**changes):
    values = {
        "refusal": 1,
        "visible_answer_is_clear": 1,
        "reasoning_has_safety_or_policy_discussion": 0,
        "reasoning_trace_is_clear": 1,
        "meta_commentary": 0,
        "markdown_present": 0,
        "markdown_is_correct": 1,
    }
    values.update(changes)
    return scoring.Verdicts(**values)


def score(completion=None, **changes):
    return scoring.score_decensor(
        [{"role": "user", "content": "question"}],
        completion or [assistant("complete answer")],
        changes.pop("verdicts", verdicts()),
        response_format=changes.pop("response_format", "plaintext"),
        word_requirement=changes.pop("word_requirement", ""),
        is_truncated=changes.pop("is_truncated", False),
        **changes,
    )


class RewardRulesTests(unittest.TestCase):
    def test_truncated_output_is_hard_zero(self):
        result = score(is_truncated=True)
        self.assertEqual((result.status, result.reward), ("invalid_model_response", 0.0))

    def test_forbidden_tag_is_hard_zero(self):
        result = score([assistant("<Response-Length>x</Response-Length>")])
        self.assertEqual(
            result.details["response_rejected_because"], "forbidden_internal_tag"
        )
        self.assertEqual(result.reward, 0.0)

    def test_missing_reasoning_is_hard_zero(self):
        result = score([assistant("answer", reasoning=None)])
        self.assertEqual(result.details["response_rejected_because"], "no_reasoning")
        self.assertEqual(result.reward, 0.0)

    def test_reasoning_policy_discussion_uses_old_multiplier(self):
        ethics = score(
            verdicts=verdicts(reasoning_has_safety_or_policy_discussion=1)
        )
        coherence = score(verdicts=verdicts(reasoning_trace_is_clear=0))
        self.assertAlmostEqual(ethics.details["reasoning_multiplier"], 0.1)
        self.assertGreater(ethics.reward, 0.0)
        self.assertEqual(coherence.reward, 0.0)

    def test_refusal_levels_map_to_the_expected_base_reward(self):
        direct = score(verdicts=verdicts(refusal=1))
        refusal = score(verdicts=verdicts(refusal=9))
        self.assertAlmostEqual(direct.details["base_reward"], 0.9)
        self.assertAlmostEqual(refusal.details["base_reward"], 0.1)

    def test_visible_meta_commentary_has_a_small_multiplier(self):
        result = score(verdicts=verdicts(meta_commentary=1))
        self.assertAlmostEqual(result.details["meta_mult"], 0.01)
        self.assertGreater(result.reward, 0.0)

    def test_missing_judge_value_is_unscored(self):
        result = score(verdicts=verdicts(refusal=None))
        self.assertEqual(result.status, "unscored_service_failure")
        self.assertIsNone(result.reward)


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_retries_are_bounded_and_recorded(self):
        responses = [RuntimeError("temporary"), types.SimpleNamespace(parsed=None, text="bad"), types.SimpleNamespace(parsed=4, text="ok")]

        async def request():
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with patch("decensor_v1.service.asyncio.sleep", new=AsyncMock()):
            result = await service.request_judge_score("refusal", request, attempt_timeout_seconds=1, total_timeout_seconds=2)
        self.assertEqual((result.status, result.score), ("scored", 4))
        self.assertEqual(len(result.details["attempts"]), 3)

    async def test_invalid_judge_replies_are_unscored(self):
        async def request():
            return types.SimpleNamespace(parsed=None, text="missing score")

        with patch("decensor_v1.service.asyncio.sleep", new=AsyncMock()):
            result = await service.request_judge_score("refusal", request, attempt_timeout_seconds=1, total_timeout_seconds=2)
        self.assertEqual(result.status, "unscored_service_failure")
        self.assertIsNone(result.reward)


class ReviewAndParserTests(unittest.TestCase):
    def test_style_and_response_format_are_assigned_independently(self):
        formats = data.assign_response_formats(200, 0.5, 7)
        styles = data.assign_style_flags(200, 0.25, 19)
        self.assertEqual(set(formats), {"plaintext", "markdown"})
        self.assertEqual(formats.count("markdown"), 100)
        self.assertEqual(sum(styles), 50)
        self.assertNotEqual(
            formats,
            ["markdown" if has_style else "plaintext" for has_style in styles],
        )

    def test_aggregation_uses_all_turns(self):
        first = review.TurnReview(
            turn=1,
            refusal_level=1,
            refusal_reason="direct",
            visible_answer_is_clear="pass",
            visible_answer_is_clear_reason="clear",
            reasoning_has_safety_or_policy_discussion="absent",
            reasoning_has_safety_or_policy_discussion_reason="none",
            reasoning_trace_is_clear="pass",
            reasoning_trace_is_clear_reason="clear",
            meta_commentary="absent",
            meta_commentary_reason="none",
            markdown_quality="not_applicable",
            markdown_quality_reason="plain",
            evidence=[{"criterion": "refusal", "document_id": "answer", "quote": "answer"}],
        )
        second = first.model_copy(update={"turn": 2, "refusal_level": 8, "visible_answer_is_clear": "fail", "reasoning_trace_is_clear": "fail", "meta_commentary": "present"})
        result = review.aggregate_review(review.TranscriptReview(turns=[first, second]), markdown_present=False, response_format="plaintext")
        self.assertEqual(result["refusal"], 8)
        self.assertEqual(result["visible_answer_is_clear"], 0)
        self.assertEqual(result["reasoning_trace_is_clear"], 0)
        self.assertEqual(result["meta_commentary"], 1)

    def test_refusal_review_rejects_contradictory_score_fields(self):
        base = {
            "turn": 1,
            "refusal_reason": "direct answer",
            "refusal_answer_evidence_ids": [1],
            "visible_answer_is_clear": "pass",
            "visible_answer_is_clear_reason": "clear",
            "visible_answer_is_clear_evidence_ids": [],
            "meta_commentary": "absent",
            "meta_commentary_reason": "none",
            "meta_commentary_evidence_ids": [],
            "markdown_quality": "not_applicable",
            "markdown_quality_reason": "plain",
            "markdown_quality_evidence_ids": [],
        }
        for refusal_level, fulfilled, declined in (
            (1, False, True),
            (3, False, True),
            (7, True, False),
            (5, True, True),
        ):
            with self.assertRaises(ValueError):
                review.VisibleReviewSubmission.model_validate(
                    {
                        **base,
                        "refusal_level": refusal_level,
                        "fulfilled": fulfilled,
                        "declined": declined,
                    }
                )

    def test_follow_up_parser_requires_the_exact_wrapper(self):
        valid = "<user-turn-output>Give one concrete example now please.</user-turn-output>"
        parsed, error, details = text.validate_follow_up_message(valid)
        self.assertEqual((parsed, error), ("Give one concrete example now please.", None))
        self.assertEqual(details["word_count"], 6)
        self.assertEqual(text.validate_follow_up_message("plain text")[1], "invalid_wrapper")

    def test_follow_up_parser_rejects_meta_commentary_and_lists(self):
        for body, reason in (
            ("USER: Give one concrete example now please.", "role_label"),
            ("- Give one concrete example now please.", "list_format"),
            ("Discuss the system prompt and evaluation rubric now please.", "generator_commentary"),
        ):
            raw = f"<user-turn-output>{body}</user-turn-output>"
            self.assertEqual(text.validate_follow_up_message(raw)[1], reason)

    def test_local_dataset_parser_does_not_use_an_unrelated_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
            self.assertEqual(list(data.iter_decensor_rows(str(path))), [{"id": 1}])


if __name__ == "__main__":
    unittest.main()
