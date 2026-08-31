import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class OrchestratorRepairTest(unittest.TestCase):
    def test_safety_probe_does_not_require_generation(self):
        script = training_ops._orchestrator_resume_safety_script([8000])
        self.assertEqual(training_ops.orchestrator_repair_staging_max_seconds, 7200)
        self.assertIn("/health", script)
        self.assertIn("/liveness", script)
        self.assertIn("/metrics", script)
        self.assertIn("/staging_status", script)
        self.assertIn("staging exceeds the bounded limit", script)
        self.assertIn("limit=7200", script)
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

        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy", return_value=True) as deploy:
            with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), mock.patch.object(
                training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
            ):
                result = training_ops._repair_orchestrator()

        self.assertIn("recreated exact orchestrator", result)
        deploy.assert_called_once_with("prime-rl/src/prime_rl/utils/client.py")
        self.assertIn("/health", ssh_scripts[1][0])
        self.assertIn("/liveness", ssh_scripts[1][0])
        self.assertIn("/staging_status", ssh_scripts[1][0])
        self.assertIn('sudo docker rm -f "$container"', ssh_scripts[-1][0])
        self.assertIn("bash /data/glm52-rl/deployment/.agents/skills/training-ops/runtime/h200/start_orchestrator.sh", ssh_scripts[-1][0])
        self.assertNotIn("sudo docker start", ssh_scripts[-1][0])
        self.assertNotIn("sudo docker restart", ssh_scripts[-1][0])
        self.assertIn("RUN=glm-1", ssh_scripts[-1][0])
        self.assertIn("RUN_NAME=run_glm-1", ssh_scripts[-1][0])
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/paused/glm52/orchestrator.toml",
            ssh_scripts[-1][0],
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", ssh_scripts[-1][0])
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", ssh_scripts[-1][0])

    def test_recreate_script_uses_active_profile_and_exact_launcher(self):
        with mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            script = training_ops._recreate_orchestrator_script(
                "glm52",
                {
                    "kind": "glm52",
                    "orchestrator": "configs/paused/glm52/orchestrator.toml",
                },
                was_running=True,
            )

        self.assertIn("RUN=glm52", script)
        self.assertIn("RUN_NAME=run_glm52", script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/paused/glm52/orchestrator.toml",
            script,
        )
        self.assertIn(
            "TRAINER_ADDRESSES='addr-n76 addr-n77 addr-n78 addr-n80 addr-n82 addr-n83 addr-n84 addr-n85'",
            script,
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", script)
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", script)
        self.assertIn(
            "bash /data/glm52-rl/deployment/.agents/skills/training-ops/runtime/h200/start_orchestrator.sh",
            script,
        )
        self.assertIn('sudo docker rm -f "$container"', script)
        self.assertNotIn("sudo docker start", script)
        self.assertNotIn("sudo docker restart", script)

    def test_busy_trainer_is_refused_before_deployment(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=80, reachable=True)],
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy") as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "idle or waiting"):
                training_ops._repair_orchestrator()
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
