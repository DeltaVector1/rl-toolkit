import unittest
from unittest.mock import patch

import training_ops


class EdgeOperationsTest(unittest.TestCase):
    def test_status_parser_keeps_metric_names_with_digits(self):
        self.assertEqual(training_ops._kv("FAILS5M 3\n"), {"FAILS5M": "3"})

    def test_blocked_host_is_refused_before_ssh(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._ssh("n79", "true")

    def test_current_run_uses_trainer_log_path_when_run_env_is_absent(self):
        command = (
            "--role=trainer\n"
            "--log-dir=/data/glm52-rl/runs/muse19/run_muse19/logs/trainer/torchrun\n"
        )
        with patch.object(training_ops, "_ssh", side_effect=[(1, ""), (0, command)]):
            self.assertEqual(training_ops.current_run(), "muse19")

    def test_start_uses_edge_launcher_and_never_starts_china_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            if command.startswith("test -f "):
                return 0, ""
            return 0, "done\n"

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(training_ops.time, "sleep"),
        ):
            training_ops.start("x99", mode="cold", supervisors=False)

        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and "scripts/h200/start_orchestrator.sh" in command
                and "ORCH_CONFIG=/data/glm52-rl/deployment/configs/current/orchestrator.toml" in command
                for host, command in calls
            )
        )
        self.assertFalse(
            any(
                host == training_ops.trainer_head
                and "cluster.sh orchestrator" in command
                for host, command in calls
            )
        )

    def test_stop_removes_edge_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
        ):
            training_ops.stop()

        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and training_ops.orchestrator_container in command
                for host, command in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
