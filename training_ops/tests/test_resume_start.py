import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import training_ops


class ResumeStartTest(unittest.TestCase):
    def run_start(self, *, mode="restart", resume=True, failure=None):
        calls = []

        def ssh(host, command, timeout=60):
            calls.append((host, command))
            if failure is not None and failure(host, command):
                return 1, "test failure"
            return 0, "ready"

        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            edge = root / "orch.toml"
            edge.write_text(
                f'output_dir = "{training_ops.run_root}/alpha/run_alpha"\n'
                '[weight_broadcast]\nsource_url = "http://localhost:8100/alpha"\n'
            )
            patches = {
                "repo_root": root,
                "_config_dir": mock.Mock(return_value={"trainer": "trainer.toml", "orchestrator": "orch.toml"}),
                "_start_dependency_inventory": mock.Mock(return_value=[]),
                "_deploy_dependencies": mock.Mock(),
                "_ssh": ssh,
                "_fan": lambda nodes, function: {host: function(host) for host in nodes},
                "_remote_ok": lambda host, command, _label: ssh(host, command),
                "_initial_adapter_cache_identity": mock.Mock(return_value=None),
                "_cold_start_cleanup": mock.Mock(),
                "_cold_adapter_cache_cleanup": mock.Mock(),
                "_repair_supervisors": mock.Mock(),
            }
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(training_ops, name, value))
            stack.enter_context(mock.patch.object(training_ops.time, "sleep"))
            stack.enter_context(mock.patch("builtins.print"))
            try:
                training_ops.start(
                    "alpha", mode=mode, nodes=["n82", "n76"], supervisors=False,
                    resume_adapter=f"{training_ops.run_root}/alpha/saved.safetensors" if resume else None,
                    resume_step=1 if resume else None,
                )
            except Exception as error:
                return calls, error
        return calls, None

    def test_resume_starts_trainer_before_orchestrator_without_broadcast_wait(self):
        calls, error = self.run_start()
        self.assertIsNone(error)
        commands = [command for _, command in calls]
        control = [i for i, command in enumerate(commands) if command.startswith("test -f ") and "control/orch.toml" in command]
        trainer = [i for i, command in enumerate(commands) if command.endswith(" trainer")]
        orch = next(i for i, command in enumerate(commands) if command.endswith(training_ops.orchestrator_launcher))
        self.assertEqual(len(control), 2)
        self.assertLess(max(control), min(trainer))
        self.assertLess(max(trainer), orch)
        self.assertFalse(any("fresh adapter" in command or ".trainer-restart-started" in command for command in commands))

    def test_missing_or_empty_control_refuses_before_any_write(self):
        calls, error = self.run_start(failure=lambda host, command: host == "n76" and "control/orch.toml" in command)
        self.assertIsInstance(error, training_ops.Refused)
        self.assertIn("non-empty control file", str(error))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(command.startswith("test -f ") and "test -s " in command for _, command in calls))

    def test_trainer_failure_cleans_selected_services_and_does_not_launch_orchestrator(self):
        calls, error = self.run_start(failure=lambda host, command: host == "n76" and command.endswith(" trainer"))
        self.assertIn("trainer start failed", str(error))
        self.assertFalse(any(command.endswith(training_ops.orchestrator_launcher) for _, command in calls))
        self.assertEqual(
            {host for host, command in calls if command == f"sudo docker rm -f {training_ops.trainer_container}"},
            {"n82", "n76"},
        )
        self.assertIn(("n82", f"sudo docker rm -f {training_ops.adapter_container}"), calls)

    def test_orchestrator_failure_cleans_trainer_and_preserves_original_error(self):
        calls, error = self.run_start(failure=lambda _host, command: command.endswith(training_ops.orchestrator_launcher) or command == f"sudo docker rm -f {training_ops.trainer_container}")
        self.assertIn("edge orchestrator start failed", str(error))
        self.assertEqual(len(error.__notes__), 2)
        self.assertIn(("n82", f"sudo docker rm -f {training_ops.adapter_container}"), calls)

    def test_adapter_failure_does_not_launch_trainer_or_orchestrator(self):
        calls, error = self.run_start(failure=lambda _host, command: command.endswith(" adapter-http"))
        self.assertIn("adapter server start failed", str(error))
        self.assertFalse(any(command.endswith(" trainer") or command.endswith(training_ops.orchestrator_launcher) for _, command in calls))
        self.assertIn(("n82", f"sudo docker rm -f {training_ops.adapter_container}"), calls)

    def test_cold_order_stays_orchestrator_control_trainer(self):
        calls, error = self.run_start(mode="cold", resume=False)
        self.assertIsNone(error)
        commands = [command for _, command in calls]
        orch = next(i for i, command in enumerate(commands) if command.endswith(training_ops.orchestrator_launcher))
        control = next(i for i, command in enumerate(commands) if "orchestrator control file did not arrive" in command)
        trainer = next(i for i, command in enumerate(commands) if command.endswith(" trainer"))
        self.assertLess(orch, control)
        self.assertLess(control, trainer)

    def test_restart_without_adapter_keeps_fresh_broadcast_wait(self):
        calls, error = self.run_start(resume=False)
        self.assertIsNone(error)
        commands = [command for _, command in calls]
        trainer = next(i for i, command in enumerate(commands) if command.endswith(" trainer"))
        broadcast = next(i for i, command in enumerate(commands) if "fresh adapter ready" in command)
        orch = next(i for i, command in enumerate(commands) if command.endswith(training_ops.orchestrator_launcher))
        self.assertLess(trainer, broadcast)
        self.assertLess(broadcast, orch)


if __name__ == "__main__":
    unittest.main()
