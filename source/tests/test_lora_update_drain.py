"""Regression checks for request-free LoRA worker collectives."""

from pathlib import Path
import unittest


CLIENT = (
    Path(__file__).resolve().parents[1] / "src/prime_rl/utils/client.py"
).read_text()
SERVER = (
    Path(__file__).resolve().parents[1] / "src/prime_rl/inference/vllm/server.py"
).read_text()
WATCHER = (
    Path(__file__).resolve().parents[1] / "src/prime_rl/orchestrator/watcher.py"
).read_text()
ORCHESTRATOR = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/orchestrator.py"
).read_text()
DISPATCHER = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/dispatcher.py"
).read_text()


class LoRAUpdateDrainTest(unittest.TestCase):
    def test_remote_adapter_is_staged_before_version_pending(self):
        apply = WATCHER.split("    async def apply_policy_update", 1)[1].split(
            "\n    async def reconcile_current_policy", 1
        )[0]
        self.assertLess(
            apply.index("await stage_lora_adapter"),
            apply.index("await observer.on_version_pending"),
        )

    def test_batch_boundary_is_bounded_and_precedes_group_restart(self):
        apply = WATCHER.split("    async def apply_policy_update", 1)[1].split(
            "\n    async def reconcile_current_policy", 1
        )[0]
        self.assertLess(
            apply.index("await wait_for_boundary(next_step)"),
            apply.index("await observer.on_version_pending(next_step)"),
        )
        boundary = ORCHESTRATOR.split(
            "    async def wait_for_policy_update_boundary", 1
        )[1].split("\n    async def on_new_version", 1)[0]
        self.assertIn("self.config.max_off_policy_steps < 1", boundary)
        self.assertIn("required_version", boundary)
        self.assertIn("if current == 0", boundary)
        self.assertIn("asyncio.wait_for", boundary)
        self.assertIn("POLICY_UPDATE_BATCH_WAIT_TIMEOUT_S", boundary)
        self.assertIn("inflight_train_count", boundary)
        self.assertIn("pending_train_schedule_count", boundary)
        self.assertNotIn("tail_start", boundary)
        self.assertIn("POLICY_UPDATE_BATCH_WAIT_TIMEOUT_S = 1200", ORCHESTRATOR)

    def test_orchestrator_waits_before_dispatcher_stops_fresh_groups(self):
        setup = ORCHESTRATOR.split("self.watcher = WeightWatcher(", 1)[1].split(
            "\n        )", 1
        )[0]
        self.assertIn("observers=[self, self.dispatcher]", setup)

    def test_completed_batch_skips_redundant_overflow_grace(self):
        boundary = ORCHESTRATOR.split(
            "    async def wait_for_policy_update_boundary", 1
        )[1].split("\n    async def on_new_version", 1)[0]
        grace = DISPATCHER.split(
            "    async def wait_for_policy_update_boundary", 1
        )[1].split("\n    def is_abandoned_group", 1)[0]
        self.assertIn("self.dispatcher.mark_batch_boundary_complete(step)", boundary)
        self.assertIn("step in self.batch_boundary_complete_steps", grace)
        self.assertIn("self.batch_boundary_complete_steps.discard(step)", grace)
        self.assertLess(
            grace.index("step in self.batch_boundary_complete_steps"),
            grace.index("live_groups ="),
        )

    def test_incomplete_group_grace_is_bounded_and_blocks_only_fresh_groups(self):
        grace = DISPATCHER.split(
            "    async def wait_for_policy_update_boundary", 1
        )[1].split("\n    def is_abandoned_group", 1)[0]
        scheduling = DISPATCHER.split("    async def try_schedule", 1)[1].split(
            "\n    def next_fresh_group", 1
        )[0]
        self.assertIn("POLICY_GROUP_DRAIN_GRACE_S = 60", DISPATCHER)
        self.assertIn("completed < smallest_group", grace)
        self.assertIn("deadline = monotonic() + POLICY_GROUP_DRAIN_GRACE_S", grace)
        self.assertIn("self.policy_group_drain_pending = True", grace)
        self.assertIn("self.policy_group_drain_pending", scheduling)
        # Existing groups are considered before the fresh-group block.
        self.assertLess(scheduling.index("for gid, group"), scheduling.index("policy_group_drain_pending"))

    def test_stage_endpoint_does_not_call_engine_collective(self):
        endpoint = SERVER.split('@router.post("/stage_lora_adapter")', 1)[1].split(
            "\n\n@router.", 1
        )[0]
        self.assertIn("asyncio.to_thread(stage_lora_adapter_files", endpoint)
        self.assertNotIn("collective_rpc", endpoint)

    def test_lora_update_pauses_before_load_and_resumes_in_finally(self):
        update = CLIENT.split("async def update_weights(", 1)[1].split(
            "\ndef _is_retryable_lora_error", 1
        )[0]
        branch = update.split("if lora_name is not None", 1)[1].split("else:", 1)[0]

        pause = branch.index("await _pause_engines")
        load = branch.index("await load_lora_adapter")
        finally_clause = branch.index("finally:")
        resume = branch.index("await _resume_engines")

        self.assertLess(pause, load)
        self.assertLess(load, finally_clause)
        self.assertLess(finally_clause, resume)

    def test_pause_aborts_server_requests_before_collective(self):
        pause = CLIENT.split("async def _pause_engines", 1)[1].split(
            "async def _resume_engines", 1
        )[0]
        self.assertIn('"mode": "abort"', pause)


if __name__ == "__main__":
    unittest.main()
