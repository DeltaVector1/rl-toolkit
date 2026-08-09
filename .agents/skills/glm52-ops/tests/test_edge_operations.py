import unittest
from unittest.mock import patch

import glm52_ops


class EdgeOperationsTest(unittest.TestCase):
    def test_start_uses_edge_launcher_and_never_starts_china_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            if command.startswith("test -f "):
                return 0, ""
            return 0, "done\n"

        with (
            patch.object(glm52_ops, "_ssh", side_effect=fake_ssh),
            patch.object(glm52_ops, "_fan", return_value={}),
            patch.object(glm52_ops.time, "sleep"),
        ):
            glm52_ops.start("x99", mode="cold", supervisors=False)

        self.assertTrue(
            any(
                host == glm52_ops.EDGE_ORCH
                and "scripts/h200/start_orchestrator.sh" in command
                and "CONFIG_TAG=.x99h" in command
                for host, command in calls
            )
        )
        self.assertFalse(
            any(
                host == glm52_ops.HEAD
                and "cluster.sh orchestrator" in command
                for host, command in calls
            )
        )

    def test_start_refuses_without_edge_config(self):
        with patch.object(glm52_ops, "_ssh", return_value=(1, "")):
            with self.assertRaises(glm52_ops.Refused):
                glm52_ops.start("x99", mode="cold", supervisors=False)

    def test_stop_removes_edge_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        with (
            patch.object(glm52_ops, "_ssh", side_effect=fake_ssh),
            patch.object(glm52_ops, "_fan", return_value={}),
        ):
            glm52_ops.stop()

        self.assertTrue(
            any(
                host == glm52_ops.EDGE_ORCH
                and glm52_ops.ORCH_CT in command
                for host, command in calls
            )
        )


    def test_busy_trainer_is_not_reported_as_log_stalled(self):
        status = glm52_ops.Status(
            trainer_up=True,
            orch_up=True,
            trainer_log_age=1800,
            orch_log_age=1,
            tunnels_ok=48,
            judge="200",
            adapter="200",
            supervisor_run=True,
            supervisor_tunnel=True,
            nodes=[
                glm52_ops.NodeState(
                    host="n82",
                    gpu_util=99,
                    mem_free_gb=400,
                )
            ],
        )

        warnings = glm52_ops._warnings(status)

        self.assertFalse(
            any("trainer has written nothing" in warning for warning in warnings)
        )


if __name__ == "__main__":
    unittest.main()
