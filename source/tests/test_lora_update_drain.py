"""Regression checks for LoRA updates and worker collectives."""

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
