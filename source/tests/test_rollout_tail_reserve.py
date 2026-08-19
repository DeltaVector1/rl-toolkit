"""Regression checks for group-safe rollout tail handling."""

from pathlib import Path
import unittest


ORCHESTRATOR = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/orchestrator.py"
).read_text()
SERVER = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/inference/vllm/server.py"
).read_text()


class RolloutTailReserveTest(unittest.TestCase):
    def test_budget_keeps_multiple_complete_groups_as_overflow(self):
        update = ORCHESTRATOR.split("    def update_rollout_budget", 1)[1].split(
            "\n    def update_dispatch_gate", 1
        )[0]
        self.assertIn("ROLLOUT_TAIL_SPARE_GROUPS = 32", ORCHESTRATOR)
        self.assertIn("spare = group * ROLLOUT_TAIL_SPARE_GROUPS", update)
        self.assertIn("self.config.batch_size + spare", update)

    def test_engine_pause_honors_abort_mode(self):
        pause = SERVER.split('@router.post("/pause")', 1)[1].split(
            '@router.post("/resume")', 1
        )[0]
        self.assertIn('request.query_params.get("mode", "abort")', pause)
        self.assertIn("pause_generation(mode=mode", pause)


if __name__ == "__main__":
    unittest.main()
