"""Regression check for the resume-time policy-update scheduling gate."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/orchestrator.py"
).read_text()


class OrchestratorResumeGateTest(unittest.TestCase):
    def test_resume_update_closes_dispatch_before_dispatcher_task_creation(self):
        start = SOURCE.split("    async def start(self)", 1)[1].split(
            "\n    async def", 1
        )[0]
        gate = start.index("self.dispatcher.weight_update_pending = True")
        dispatcher_task = start.index("asyncio.create_task(self.dispatcher.start()")
        self.assertLess(gate, dispatcher_task)


if __name__ == "__main__":
    unittest.main()
