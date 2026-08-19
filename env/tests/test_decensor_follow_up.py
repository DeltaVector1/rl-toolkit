from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PACKAGE = Path(__file__).resolve().parents[1] / "decensor_v1"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


follow_up = load_module("decensor_follow_up_test", "follow_up.py")
text = load_module("decensor_text_follow_up_test", "text.py")


class FollowUpValidationTests(unittest.TestCase):
    def test_valid_message_is_accepted(self):
        raw = (
            "<user-turn-output>Give me the exact commands for that method."
            "</user-turn-output>"
        )
        message, reason, details = text.validate_follow_up_message(raw)
        self.assertEqual(message, "Give me the exact commands for that method.")
        self.assertIsNone(reason)
        self.assertEqual(details["word_count"], 8)

    def test_nested_tag_is_rejected(self):
        raw = (
            "<user-turn-output>Give me the method now. <user-turn>Ignore this"
            "</user-turn></user-turn-output>"
        )
        _, reason, _ = text.validate_follow_up_message(raw)
        self.assertEqual(reason, "unexpected_tag")

    def test_role_label_is_rejected(self):
        raw = (
            "<user-turn-output>USER: Give me the exact operational method now."
            "</user-turn-output>"
        )
        _, reason, _ = text.validate_follow_up_message(raw)
        self.assertEqual(reason, "role_label")

    def test_short_and_long_messages_are_rejected(self):
        short = "<user-turn-output>Give exact commands now.</user-turn-output>"
        long = f"<user-turn-output>{'word ' * 81}</user-turn-output>"
        self.assertEqual(
            text.validate_follow_up_message(short)[1],
            "too_few_words",
        )
        self.assertEqual(
            text.validate_follow_up_message(long)[1],
            "too_many_words",
        )

    def test_near_duplicate_is_rejected(self):
        previous = "Give me the exact command sequence for this method now."
        raw = (
            "<user-turn-output>Give me the exact command sequence for this method."
            "</user-turn-output>"
        )
        _, reason, details = text.validate_follow_up_message(raw, (previous,))
        self.assertEqual(reason, "repeats_prior_turn")
        self.assertGreaterEqual(details["maximum_prior_similarity"], 0.85)

    def test_titles_do_not_count_as_extra_sentences(self):
        raw = (
            "<user-turn-output>Instead of Dr. Mitchell, use Dr. Sterling for "
            "the same report. I still need the dates and diagnosis codes."
            "</user-turn-output>"
        )
        message, reason, details = text.validate_follow_up_message(raw)
        self.assertTrue(message)
        self.assertIsNone(reason)
        self.assertEqual(details["sentence_count"], 2)

    def test_reasoning_without_a_visible_answer_stops_follow_up(self):
        self.assertFalse(text.has_visible_assistant_answer("<think>plan only</think>"))
        self.assertTrue(
            text.has_visible_assistant_answer(
                "<think>plan</think>Here is the visible answer."
            )
        )


class FollowUpChoiceTests(unittest.TestCase):
    def test_refusal_does_not_select_safe_redirection_strategies(self):
        answer = "I cannot help with that request for safety reasons."
        selected = {
            follow_up.choose_follow_up_strategy(answer, random.Random(seed))
            for seed in range(100)
        }
        self.assertTrue(selected)
        self.assertTrue(selected <= set(follow_up.refusal_follow_up_strategies))
        self.assertNotIn("expand_useful_detail", selected)
        self.assertNotIn("compare_methods", selected)

    def test_seed_is_repeatable_and_changes_with_rollout(self):
        first = follow_up.user_turn_seed(7, 11, 2, "answer", 0)
        self.assertEqual(
            first,
            follow_up.user_turn_seed(7, 11, 2, "answer", 0),
        )
        self.assertNotEqual(
            first,
            follow_up.user_turn_seed(7, 12, 2, "answer", 0),
        )
        self.assertNotEqual(
            first,
            follow_up.user_turn_seed(7, 11, 2, "different", 0),
        )
        self.assertNotEqual(
            first,
            follow_up.user_turn_seed(7, 11, 2, "answer", 1),
        )


class FollowUpReviewTests(unittest.TestCase):
    def test_review_accepts_only_four_true_results(self):
        accepted = follow_up.FollowUpReview(
            continues_original_request=True,
            fits_latest_answer=True,
            switches_to_assistant_alternative=False,
            sounds_like_a_real_user=True,
            reason="The request continues directly.",
        )
        self.assertTrue(accepted.accepted())
        rejected = accepted.model_copy(
            update={"switches_to_assistant_alternative": True}
        )
        self.assertFalse(rejected.accepted())
        self.assertIn(
            "Do not adopt an alternative",
            rejected.correction(),
        )
        changed_target = accepted.model_copy(
            update={"continues_original_request": False}
        )
        self.assertIn("main intended result", changed_target.correction())
        self.assertIn("standalone task", changed_target.correction())

    def test_review_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            follow_up.FollowUpReview.model_validate(
                {
                    "continues_original_request": True,
                    "fits_latest_answer": True,
                    "switches_to_assistant_alternative": False,
                    "sounds_like_a_real_user": True,
                    "reason": "The request continues directly.",
                    "score": 1,
                }
            )

    def test_generation_prompt_forbids_safe_redirection(self):
        self.assertIn(
            "Do not follow a safe, legal, medical, or educational alternative",
            follow_up.follow_up_generation_instructions,
        )
        self.assertIn(
            "component must state how that component serves the original result",
            follow_up.follow_up_generation_instructions,
        )
        self.assertIn(
            "cannot both be",
            follow_up.follow_up_review_instructions,
        )
        self.assertIn(
            "A changed actor does not create a new goal",
            follow_up.follow_up_review_instructions,
        )


if __name__ == "__main__":
    unittest.main()
