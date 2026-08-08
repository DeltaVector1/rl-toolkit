"""Regression tests for trainer sequence-length enforcement at the sink."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/train_sink.py"
).read_text()


class TrainSinkLimitTest(unittest.TestCase):
    def test_oversized_rollouts_are_removed_before_advantages(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        check = group.index("len(sample.token_ids) > self.config.seq_len")
        finalize = group.index("await env.algorithm.finalize_group")
        self.assertLess(check, finalize)
        self.assertIn("if env.requires_group_scoring:", group)
        self.assertIn("return", group[check:finalize])

    def test_trainer_never_receives_a_rewarded_truncated_prefix(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        self.assertIn("dropped:", group)
        self.assertIn("exceed trainer seq_len", group)


if __name__ == "__main__":
    unittest.main()
