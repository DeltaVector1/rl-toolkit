"""Regression tests for trainer sequence-length enforcement at the sink."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/train_sink.py"
).read_text()


class TrainSinkLimitTest(unittest.TestCase):
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
        self.assertIn("if not any(r.is_trainable for r in survivors)", group)


if __name__ == "__main__":
    unittest.main()
