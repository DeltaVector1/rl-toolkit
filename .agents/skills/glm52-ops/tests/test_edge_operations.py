import unittest
import inspect
import subprocess
from unittest.mock import patch

import glm52_ops


class EdgeOperationsTest(unittest.TestCase):
    def test_ssh_cannot_wait_for_interactive_credentials(self):
        with patch.object(glm52_ops.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "ok"
            run.return_value.stderr = ""

            self.assertEqual(glm52_ops._ssh("n82", "true"), (0, "ok"))

        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        command = run.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertIn("NumberOfPasswordPrompts=0", command)
        self.assertIn("RequestTTY=no", command)

    def test_successful_ssh_does_not_mix_diagnostic_stderr_into_data(self):
        with patch.object(glm52_ops.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = "ControlSocket already exists\n"

            self.assertEqual(glm52_ops._ssh("n82", "true"), (0, ""))

    def test_start_uses_eight_transport_rounds_for_step8_headroom(self):
        self.assertIn(
            "PRIME_INT4_COMBINE_ROUNDS=8",
            inspect.getsource(glm52_ops.start),
        )

    def test_edge_probe_reports_the_batch_publisher_without_self_matching(self):
        self.assertIn("BATCH_SYNC", glm52_ops._EDGE_ORCH_PROBE)
        self.assertIn("[s]ync_rollout_batches.sh", glm52_ops._EDGE_ORCH_PROBE)
        self.assertIn("TRAINER_FAIL_WD", glm52_ops._EDGE_ORCH_PROBE)
        self.assertIn("[t]rainer_failure_watchdog.sh", glm52_ops._EDGE_ORCH_PROBE)

    def test_engine_probe_uses_recent_loaded_metrics_not_probe_zero_window(self):
        self.assertIn("docker logs --since 60s", glm52_ops._ENGINE_PROBE)
        self.assertIn("if ($1>m)m=$1", glm52_ops._ENGINE_PROBE)
        self.assertNotIn("docker logs --tail 20 __ENGINE__", glm52_ops._ENGINE_PROBE)

    def test_explicit_engines_probe_does_not_depend_on_trainer_mode(self):
        outputs = {
            host: (0, "GPU 0 8\nGEN 200 123\nRUNNING 0\nTPUT 1.5\nWD 1\n")
            for host in glm52_ops.ENGINE_HOSTS
        }
        with (
            patch.object(glm52_ops, "current_run", return_value="muse-test"),
            patch.object(glm52_ops, "_fan", return_value=outputs),
            patch.object(glm52_ops, "status") as status,
        ):
            states = glm52_ops.engines()

        status.assert_not_called()
        self.assertEqual(set(states), set(glm52_ops.ENGINE_HOSTS))
        self.assertTrue(all(state.healthy for state in states.values()))

    def test_active_engine_watchdog_suppresses_racing_manual_repair_advice(self):
        status = glm52_ops.Status(
            orch_up=True,
            batch_sync=True,
            engines=[
                glm52_ops.EngineState(
                    host="h200b", gen_code="000", watchdog=True
                )
            ],
        )

        warnings = glm52_ops._warnings(status)

        self.assertTrue(any("watchdog is active" in warning for warning in warnings))
        self.assertFalse(any("Run repair('engine:h200b')" in warning for warning in warnings))

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

    def test_missing_worker_names_are_reported(self):
        status = glm52_ops.Status(
            trainer_up=False,
            nodes=[
                glm52_ops.NodeState(host="n76", reachable=True, container=""),
                glm52_ops.NodeState(host="n82", reachable=True, container="Up 1 hour"),
            ],
        )

        warnings = glm52_ops._warnings(status)

        self.assertTrue(any("n76" in warning for warning in warnings))

    def test_trainer_only_status_does_not_report_rollout_components_as_failed(self):
        status = glm52_ops.Status(
            mode="trainer-only",
            trainer_up=True,
            tunnels_ok=0,
            orch_up=False,
            judge="000",
            adapter="000",
            supervisor_run=False,
            supervisor_tunnel=False,
            engines=[glm52_ops.EngineState(host="h200b", gen_code="502")],
        )

        warnings = glm52_ops._warnings(status)

        self.assertEqual(warnings, [])
        self.assertIn("rollout components not required", status.report())

    def test_completed_qualification_is_not_reported_as_failed(self):
        status = glm52_ops.Status(
            mode="trainer-only",
            trainer_up=False,
            trainer_finished=True,
            trainer_steps=20,
            nodes=[
                glm52_ops.NodeState(
                    host="n82", container="", mem_free_gb=400, data_free_gb=400
                )
            ],
        )

        self.assertEqual(glm52_ops._warnings(status), [])
        self.assertIn("trainer finished", status.report())

    def test_failed_final_checkpoint_and_low_disk_are_explicit(self):
        status = glm52_ops.Status(
            mode="trainer-only",
            trainer_failed=True,
            trainer_steps=20,
            nodes=[
                glm52_ops.NodeState(
                    host="n82", mem_free_gb=400, data_free_gb=0, data_use_percent=100
                )
            ],
        )

        warnings = glm52_ops._warnings(status)
        self.assertTrue(any("failed after 20" in warning for warning in warnings))
        self.assertTrue(any("0G free on /data" in warning for warning in warnings))
        self.assertIn("trainer FAILED", status.report())

    def test_qualify_refuses_a_non_fake_config(self):
        with self.assertRaises(glm52_ops.Refused):
            glm52_ops.qualify("muse7")


if __name__ == "__main__":
    unittest.main()
