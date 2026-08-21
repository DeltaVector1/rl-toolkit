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

    def test_harness_wall_timeout_is_terminal(self):
        handler = SOURCE.split(
            "    async def handle_completed_rollout", 1
        )[1].split("\n    def _is_transient_policy_failure", 1)[0]
        retry_condition = handler.split(
            "if (", 1
        )[1].split(
            "):\n            identity = client_identity", 1
        )[0]
        self.assertNotIn("harness_timeout", retry_condition)
        self.assertIn("A wall-clock harness timeout is a terminal rollout result", handler)

if __name__ == "__main__":
    unittest.main()
