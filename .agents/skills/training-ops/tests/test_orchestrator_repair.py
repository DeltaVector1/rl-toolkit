import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class OrchestratorRepairTest(unittest.TestCase):
    def test_safety_probe_does_not_require_generation(self):
        script = training_ops._orchestrator_resume_safety_script([8000], 900)
        self.assertIn("/health", script)
        self.assertIn("/liveness", script)
        self.assertIn("/metrics", script)
        self.assertIn("/staging_status", script)
        self.assertIn("staging exceeds the bounded limit", script)
        self.assertNotIn("/v1/completions", script)

    def test_stopped_orchestrator_is_started_after_all_guards_pass(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=0, reachable=True)],
            batch="batch 0/1024; 0 inflight",
        )
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=120):
            ssh_scripts.append((script, timeout))
            if "configured orchestrator container state is unknown" in script:
                return 0, "stopped\n"
            if "docker start" in script:
                return 0, "started exact orchestrator container glm52-decensor-orchestrator\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="x85"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy", return_value=True) as deploy:
            with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
                result = training_ops._repair_orchestrator()

        self.assertIn("started exact orchestrator", result)
        deploy.assert_called_once_with("prime-rl/src/prime_rl/utils/client.py")
        self.assertIn("/health", ssh_scripts[1][0])
        self.assertIn("/liveness", ssh_scripts[1][0])
        self.assertIn("/staging_status", ssh_scripts[1][0])
        self.assertIn("sudo docker start", ssh_scripts[-1][0])

    def test_busy_trainer_is_refused_before_deployment(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=80, reachable=True)],
        )
        with mock.patch.object(training_ops, "current_run", return_value="x85"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy") as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "idle or waiting"):
                training_ops._repair_orchestrator()
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
