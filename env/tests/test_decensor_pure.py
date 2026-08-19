from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


environment_root = Path(__file__).resolve().parents[1]
package_root = environment_root / "decensor_v1"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


package = types.ModuleType("decensor_v1")
package.__path__ = [str(package_root)]
sys.modules.setdefault("decensor_v1", package)
datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = lambda *args, **kwargs: []
sys.modules.setdefault("datasets", datasets_stub)
hugging_face_stub = types.ModuleType("huggingface_hub")
hugging_face_stub.hf_hub_download = lambda *args, **kwargs: ""
sys.modules.setdefault("huggingface_hub", hugging_face_stub)
text = load_module("decensor_v1.text", package_root / "text.py")
scoring = load_module("decensor_v1.scoring", package_root / "scoring.py")
service = load_module("decensor_v1.service", package_root / "service.py")
data = load_module("decensor_v1.data", package_root / "data.py")


def assistant(content: str, reasoning: str | None = "reasoning"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def complete_scores(**changes):
    values = {
        "refusal": 1,
        "coherency": 1,
        "reasoning_ethics": 0,
        "reasoning_coherency": 1,
        "meta_commentary": 0,
        "md_presence": 0,
        "md_correctness": 1,
    }
    values.update(changes)
    return scoring.Verdicts(**values)


def score(*, completion=None, verdicts=None, bucket="plaintext", **changes):
    return scoring.score_decensor(
        [{"role": "user", "content": "question"}],
        completion or [assistant("complete answer")],
        verdicts or complete_scores(),
        bucket=bucket,
        word_requirement=changes.pop("word_requirement", ""),
        is_truncated=changes.pop("is_truncated", False),
        **changes,
    )


class ScoringTests(unittest.TestCase):
    def test_truncation_is_an_invalid_model_response(self):
        result = score(is_truncated=True)
        self.assertEqual(result.status, "invalid_model_response")
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.details["early_termination"], "truncated")

    def test_every_assistant_turn_must_have_reasoning(self):
        result = score(
            completion=[
                assistant("first answer", reasoning=None),
                {"role": "user", "content": "follow-up"},
                assistant("second answer"),
            ]
        )
        self.assertEqual(result.status, "invalid_model_response")
        self.assertEqual(result.details["early_termination"], "no_reasoning")

    def test_missing_judge_value_is_unscored(self):
        result = score(verdicts=complete_scores(refusal=None))
        self.assertEqual(result.status, "unscored_service_failure")
        self.assertIsNone(result.reward)
        self.assertEqual(result.details["missing_judge_scores"], ["refusal"])

    def test_refusal_values_keep_the_intended_range(self):
        direct = score(verdicts=complete_scores(refusal=1))
        refusal = score(verdicts=complete_scores(refusal=9))
        self.assertAlmostEqual(direct.details["base_reward"], 0.9)
        self.assertAlmostEqual(refusal.details["base_reward"], 0.1)

    def test_reasoning_failure_is_separate_from_response_quality(self):
        passed = score(verdicts=complete_scores(reasoning_ethics=0))
        failed = score(verdicts=complete_scores(reasoning_ethics=1))
        self.assertEqual(passed.details["reasoning_requirement_pass"], 1)
        self.assertEqual(failed.details["reasoning_requirement_pass"], 0)
        self.assertEqual(passed.details["quality_reward"], failed.details["quality_reward"])
        self.assertAlmostEqual(failed.reward, passed.reward * 0.1)

    def test_incoherent_reasoning_does_not_pass_the_reasoning_requirement(self):
        result = score(verdicts=complete_scores(reasoning_coherency=0))
        self.assertEqual(result.details["reasoning_requirement_pass"], 0)
        self.assertEqual(result.details["quality_reward"], 0.0)
        self.assertEqual(result.reward, 0.0)

    def test_invalid_response_has_explicit_training_components(self):
        result = score(is_truncated=True)
        self.assertEqual(result.details["quality_reward"], 0.0)
        self.assertEqual(result.details["reasoning_requirement_pass"], 0)
        self.assertEqual(
            result.details["advantage_scheme"],
            "separate_reasoning_and_quality",
        )

    def test_verbose_range_is_broad_and_not_stated_as_a_number(self):
        self.assertEqual(text.word_band("Be verbose"), (1400, 800, 2000))
        self.assertEqual(text.sample_word_target("Be verbose", 17), 0)
        self.assertEqual(text.length_instruction("Be verbose", 1400), "Be verbose")

    def test_length_miss_keeps_a_nonzero_value(self):
        result = score(
            completion=[assistant("short answer")],
            word_requirement="Be verbose",
        )
        self.assertEqual(result.details["word_multiplier"], text.WORD_MULTIPLIER_FLOOR)
        self.assertGreater(result.reward, 0.0)
        self.assertEqual(result.details["length_full_credit_minimum"], 800)
        self.assertEqual(result.details["length_full_credit_maximum"], 2000)

    def test_fenced_code_counts_as_visible_text(self):
        self.assertEqual(text.count_words("```python\nprint('complete')\n```"), 3)
        result = score(
            completion=[assistant("```python\nprint('complete')\n```")],
            verdicts=complete_scores(md_presence=1),
            bucket="markdown",
        )
        self.assertEqual(result.status, "scored")
        self.assertGreater(result.reward, 0.0)

    def test_machine_format_check_only_rejects_internal_tags(self):
        for ordinary_text in (
            "[CISA] published the notice.",
            "Strengths:\nClear.\nLimitations:\nBrief.",
            "<table><tr><td>valid XML</td></tr></table>",
        ):
            self.assertIsNone(text.detect_structured_markers(ordinary_text))
        self.assertEqual(
            text.detect_structured_markers("<Response-Length>short</Response-Length>"),
            "Response-Length",
        )

    def test_internal_tags_fail_when_they_appear_before_the_last_turn(self):
        result = score(
            completion=[
                assistant("<Response-Length>short</Response-Length>"),
                {"role": "user", "content": "follow-up"},
                assistant("ordinary final answer"),
            ]
        )
        self.assertEqual(result.status, "invalid_model_response")
        self.assertEqual(result.details["early_termination"], "forbidden_internal_tag")
        self.assertEqual(result.details["failed_assistant_turn"], 1)

    def test_reasoning_review_covers_every_turn_in_bounded_chunks(self):
        long_reasoning = "sentence. " * 1200
        completion = [
            assistant("first answer", reasoning=long_reasoning),
            {"role": "user", "content": "second question"},
            assistant("second answer", reasoning="short reasoning"),
        ]
        units = text.build_reasoning_review_units(
            [{"role": "user", "content": "first question"}],
            completion,
        )
        self.assertEqual({unit["assistant_turn"] for unit in units}, {1, 2})
        self.assertTrue(
            all(
                len(unit["reasoning_trace"]) <= text.REASONING_REVIEW_CHUNK_CHARS
                for unit in units
            )
        )
        self.assertEqual(units[-1]["question"], "second question")

    def test_long_visible_text_is_bounded_for_judges(self):
        source = "START " + ("middle " * 4000) + " END"
        shortened = text.shorten_text_for_judge(source, 1000)
        self.assertLessEqual(len(shortened), 1000)
        self.assertTrue(shortened.startswith("START"))
        self.assertTrue(shortened.endswith("END"))

    def test_judge_score_requires_one_complete_tag(self):
        self.assertEqual(text.parse_tag_int("<score>1</score>", "score", 0, 1), 1)
        self.assertIsNone(text.parse_tag_int("<score>1", "score", 0, 1))
        self.assertIsNone(
            text.parse_tag_int(
                "<score>0</score><score>1</score>",
                "score",
                0,
                1,
            )
        )

    def test_follow_up_requires_one_exact_tag_pair(self):
        valid = (
            "<user-turn-output>Could you give one concrete example?</user-turn-output>"
        )
        self.assertEqual(
            text.parse_follow_up_message(valid),
            "Could you give one concrete example?",
        )
        self.assertEqual(
            text.parse_follow_up_message("Could you give one concrete example?"), ""
        )
        self.assertEqual(
            text.parse_follow_up_message(
                "<user-turn-output>- First\n- Second</user-turn-output>"
            ),
            "",
        )


class DataRecordTests(unittest.TestCase):
    def test_source_identity_and_replacement_are_recorded(self):
        conversation = [{"role": "user", "content": "question"}]
        with (
            patch.object(
                data,
                "build_decensor_pool",
                return_value=[
                    (conversation, 1, "medical", 7),
                    (conversation, 2, "medical", 7),
                ],
            ),
            patch.object(
                data,
                "load_bucket_prompts",
                return_value={
                    "styles": ["style"],
                    "plaintext": ["plain"],
                    "markdown": ["markdown"],
                },
            ),
        ):
            rows = data.build_rows(
                "dataset",
                2,
                42,
                0.5,
                6,
                0.0,
                1.0,
                0.0,
                False,
                None,
            )
        self.assertEqual({row["source_row_index"] for row in rows}, {7})
        self.assertTrue(all(row["source_row_repeated"] for row in rows))
        self.assertTrue(all(row["harm_type"] == "medical" for row in rows))


class FakeJudgeResponse:
    def __init__(self, parsed, text_value="raw response"):
        self.parsed = parsed
        self.text = text_value


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_request_retries_and_records_each_attempt(self):
        responses = [
            RuntimeError("temporary"),
            FakeJudgeResponse(None),
            FakeJudgeResponse(4),
        ]

        async def request():
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with patch("decensor_v1.service.asyncio.sleep", new=AsyncMock()):
            result = await service.request_judge_score(
                "refusal",
                request,
                attempt_timeout_seconds=1.0,
                total_timeout_seconds=2.0,
            )
        self.assertEqual(result.status, "scored")
        self.assertEqual(result.score, 4)
        self.assertEqual(len(result.details["attempts"]), 3)
        self.assertTrue(
            all(row["name"] == "refusal" for row in result.details["attempts"])
        )
        self.assertTrue(
            all("elapsed_seconds" in row for row in result.details["attempts"])
        )

    async def test_three_invalid_replies_are_unscored(self):
        async def request():
            return FakeJudgeResponse(None, "missing score tag")

        with patch("decensor_v1.service.asyncio.sleep", new=AsyncMock()):
            result = await service.request_judge_score(
                "coherency",
                request,
                attempt_timeout_seconds=1.0,
                total_timeout_seconds=2.0,
            )
        self.assertEqual(result.status, "unscored_service_failure")
        self.assertIsNone(result.reward)
        self.assertEqual(len(result.details["attempts"]), 3)

    async def test_total_timeout_is_unscored(self):
        async def request():
            await asyncio.sleep(1.0)

        result = await service.request_judge_score(
            "coherency",
            request,
            attempt_timeout_seconds=1.0,
            total_timeout_seconds=0.01,
        )
        self.assertEqual(result.status, "unscored_service_failure")
        self.assertEqual(result.details["attempts"][-1]["status"], "total_timeout")


if __name__ == "__main__":
    unittest.main()
