"""Regression tests for retry classification in the edge dispatcher."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/dispatcher.py"
).read_text()


class DispatcherRetryTest(unittest.TestCase):
    def test_policy_update_restarts_incomplete_live_groups_without_failed_markers(self):
        pending = SOURCE.split("async def on_version_pending", 1)[1].split(
            "async def on_new_version", 1
        )[0]
        fill = SOURCE.split("async def fill_inflight", 1)[1].split(
            "async def", 1
        )[0]

        self.assertIn("self.weight_update_pending = True", pending)
        self.assertIn("group.rollouts_to_schedule = group.target_rollouts", pending)
        self.assertIn("group.policy_version_at_start = step", pending)
        self.assertIn("self.groups[uuid.uuid4()] = group", pending)
        self.assertNotIn("await self.drop_group", pending)
        self.assertIn("if self.weight_update_pending", fill)

    def test_policy_update_only_waits_for_version_pinned_eval(self):
        pending = SOURCE.split("async def on_version_pending", 1)[1].split(
            "async def on_new_version", 1
        )[0]
        self.assertIn('while any(meta.kind == "eval"', pending)

    def test_abandoned_queued_groups_are_identifiable(self):
        self.assertIn("def is_abandoned_group", SOURCE)
        self.assertIn("group_id in self.abandoned_group_ids", SOURCE)

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

    def test_missing_policy_model_is_transient(self):
        method = SOURCE.split(
            "    def _is_transient_policy_failure", 1
        )[1].split("\n    async def ", 1)[0]
        self.assertIn("missing_policy_model", method)
        self.assertIn('"does not exist"', method)
        self.assertIn('"404"', method)

    def test_vllm_response_metadata_nan_is_transient(self):
        method = SOURCE.split(
            "    def _is_transient_policy_failure", 1
        )[1].split("\n    async def ", 1)[0]
        self.assertIn("response_metadata_nan", method)
        self.assertIn(
            '"out of range float values are not json compliant" in message',
            method,
        )
        self.assertIn('"nan" in message', method)

    def test_user_simulator_connection_failure_is_retried_without_unpinning(self):
        method = SOURCE.split(
            "    def _is_transient_user_simulator_failure", 1
        )[1].split("\n    async def ", 1)[0]
        handler = SOURCE.split(
            "    async def handle_completed_rollout", 1
        )[1].split("\n    def _is_transient_policy_failure", 1)[0]
        self.assertIn('error.type != "UserError"', method)
        self.assertIn('"user simulator failed"', method)
        self.assertIn('"connection"', method)
        self.assertIn("transient_user_simulator_failure", handler)
        self.assertIn("if not transient_user_simulator_failure", handler)
        self.assertIn("group.rollouts_to_schedule += 1", handler)


if __name__ == "__main__":
    unittest.main()
