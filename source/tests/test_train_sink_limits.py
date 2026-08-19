"""Regression tests for trainer sequence-length enforcement at the sink."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/train_sink.py"
).read_text()


class TrainSinkLimitTest(unittest.TestCase):
    def test_policy_restart_discards_partial_group_without_error_accounting(self):
        method = SOURCE.split("    def abandon_groups", 1)[1].split(
            "\n    async def process_rollout", 1
        )[0]
        self.assertIn("self.pending_group_episodes.pop", method)
        self.assertIn("self.pending_groups.pop", method)
        self.assertNotIn("consecutive_errored_groups", method)
        add = SOURCE.split("    async def add", 1)[1].split(
            "\n    def abandon_groups", 1
        )[0]
        self.assertGreaterEqual(add.count("group_id in self.abandoned_groups"), 2)

    def test_oversized_rollouts_are_removed_before_advantages(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        check = group.index("len(sample.token_ids) > self.config.seq_len")
        finalize = group.index("await env.algorithm.finalize_group")
        self.assertLess(check, finalize)
        self.assertIn('r.filter_results = {"trainer_seq_len_group": True}', group)
        self.assertIn("return", group[check:finalize])

    def test_trainer_never_receives_a_rewarded_truncated_prefix(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        self.assertIn("dropped training payload:", group)
        self.assertIn('f"trainer seq_len {self.config.seq_len}', group)
        self.assertIn("r.samples = []", group)

    def test_oversized_member_rejects_complete_group(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        self.assertIn('r.filter_results = {"trainer_seq_len_group": True}', group)
        self.assertNotIn("oversized_fillers", group)

    def test_failed_rollout_rejects_complete_group_without_a_placeholder(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        self.assertIn('r.filter_results = {"errored_group": True}', group)
        self.assertIn("MAX_CONSECUTIVE_ERRORED_GROUPS", group)
        self.assertNotIn("rejected_fillers", group)

    def test_zero_advantage_group_is_replenished_after_advantage_assignment(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        finalize = group.index("await env.algorithm.finalize_group")
        reject = group.index('r.filter_results = {"zero_advantage_group": True}')
        append = group.index("self.pending_batch.append(r)")
        self.assertLess(finalize, reject)
        self.assertLess(reject, append)
        self.assertIn("MAX_CONSECUTIVE_ZERO_ADVANTAGE_GROUPS", group)
        self.assertIn("if not any(r.is_trainable for r in survivors)", group)

    def test_unrelated_rejection_breaks_zero_advantage_streak(self):
        group = SOURCE.split("    async def process_group", 1)[1].split(
            "\n    def process_batch", 1
        )[0]
        errored = group.index("if num_errored > 0:")
        oversized = group.index("if oversized:")
        untrainable = group.index("if any(not r.trainable for r in survivors):")
        zero_advantage = group.index("if not any(r.is_trainable for r in survivors):")
        for start, end in (
            (errored, oversized),
            (oversized, untrainable),
            (untrainable, zero_advantage),
        ):
            self.assertIn("self.consecutive_zero_advantage_groups = 0", group[start:end])


if __name__ == "__main__":
    unittest.main()
