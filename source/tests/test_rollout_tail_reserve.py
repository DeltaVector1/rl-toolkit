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
        self.assertIn("budget = rollout_budget(", update)
        self.assertIn("replica_count=len(self.policy_inference.train_clients)", update)

    def test_tail_refills_one_group_per_replica(self):
        module_path = Path(__file__).resolve().parents[1] / "src/prime_rl/orchestrator/orchestrator.py"
        source = module_path.read_text()
        function = source.split("def rollout_budget(", 1)[1].split("\n\n", 1)[0]
        namespace = {"ROLLOUT_TAIL_SPARE_GROUPS": 32}
        exec("def rollout_budget(" + function, namespace)
        budget = namespace["rollout_budget"](
            batch_size=1024,
            current=1008,
            buffered=538,
            live=1,
            group_size=16,
            max_inflight=128,
            replica_count=8,
        )
        self.assertEqual(budget, 128)

    def test_fresh_batch_keeps_normal_budget(self):
        module_path = Path(__file__).resolve().parents[1] / "src/prime_rl/orchestrator/orchestrator.py"
        source = module_path.read_text()
        function = source.split("def rollout_budget(", 1)[1].split("\n\n", 1)[0]
        namespace = {"ROLLOUT_TAIL_SPARE_GROUPS": 32}
        exec("def rollout_budget(" + function, namespace)
        budget = namespace["rollout_budget"](
            batch_size=1024,
            current=0,
            buffered=0,
            live=0,
            group_size=16,
            max_inflight=128,
            replica_count=8,
        )
        self.assertEqual(budget, 1536)

    def test_engine_pause_honors_abort_mode(self):
        pause = SERVER.split('@router.post("/pause")', 1)[1].split(
            '@router.post("/resume")', 1
        )[0]
        self.assertIn('request.query_params.get("mode", "abort")', pause)
        self.assertIn("pause_generation(mode=mode", pause)


if __name__ == "__main__":
    unittest.main()
