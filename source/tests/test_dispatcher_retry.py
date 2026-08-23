"""Regression tests for retry classification in the edge dispatcher."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/dispatcher.py"
).read_text()


class DispatcherRetryTest(unittest.TestCase):
    def test_harness_read_timeout_is_transient(self):
        method = SOURCE.split(
            "    def _is_transient_policy_failure", 1
        )[1].split("\n    async def ", 1)[0]
        self.assertIn('"HarnessError"', method)
        self.assertIn('"readtimeout"', method)
        self.assertIn("return True", method)

    def test_retry_unpins_the_group_client(self):
        handler = SOURCE.split(
            "    async def handle_completed_rollout", 1
        )[1].split("\n    def _is_transient_policy_failure", 1)[0]
        self.assertIn("group.pinned_client = None", handler)
        self.assertIn("group.rollouts_to_schedule += 1", handler)


if __name__ == "__main__":
    unittest.main()
