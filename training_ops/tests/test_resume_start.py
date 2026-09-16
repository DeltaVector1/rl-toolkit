import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import training_ops


class ResumeStartTest(unittest.TestCase):
    def run_start(self, *, mode="restart", resume=True, failure=None, control_files=None, nodes=None, state_failure=None, changed_origin=False):
        calls = []
        origin_checks = 0
        if control_files is None:
            control_files = {training_ops.trainer_head: "saved control"}

        def ssh(host, command, timeout=60):
            calls.append((host, command))
            if failure is not None and failure(host, command):
                return 1, "test failure"
            if command.startswith("test -f ") and "control/orch.toml" in command:
                self.assertIn(" && test -s ", command)
                return (0, "") if control_files.get(host) else (1, "")
            return 0, "ready"

        def check_state(host, directory, step, role, adapter, digest=""):
            nonlocal origin_checks
            calls.append((host, f"check restart {role} step={step}"))
            if role == state_failure:
                raise training_ops.Refused(f"test {role} state failure")
            if role == "origin":
                origin_checks += 1
            return {"policy_step": step, "next_batch": step + 1, "sha256": "a" * 64, "config_sha256": "b" * 64 if changed_origin and origin_checks > 1 else "c" * 64}

        def remote_ok(host, command, _label, **kwargs):
            code, output = ssh(host, command)
            if code:
                raise RuntimeError(output)

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
                "_deploy_dependencies": mock.Mock(side_effect=lambda _files: calls.append(("deployment", "deploy dependencies"))),
                "_ssh": ssh,
                "_check_adapter_restart": check_state,
                "_ssh_address": mock.Mock(return_value="127.0.0.1"),
                "_fan": lambda nodes, function: {host: function(host) for host in nodes},
                "_remote_ok": remote_ok,
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
                    "alpha", mode=mode, nodes=nodes or ["n82", "n76"], supervisors=False,
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
        self.assertEqual(len(control), 1)
        self.assertEqual(calls[control[0]][0], training_ops.trainer_head)
        self.assertLess(control[0], commands.index("deploy dependencies"))
        self.assertLess(max(control), min(trainer))
        self.assertLess(max(trainer), orch)
        self.assertIn("adapter_resume_step=1 ", commands[orch])
        checks = [i for i, command in enumerate(commands) if command.startswith("check restart trainer")]
        self.assertEqual(len(checks), 2)
        self.assertLess(max(checks), min(trainer))
        drain = next(i for i, command in enumerate(commands) if "drain_rollout_publisher.py" in command)
        final_origin = max(i for i, command in enumerate(commands) if command.startswith("check restart origin"))
        self.assertLess(drain, final_origin)
        self.assertLess(final_origin, min(checks))
        self.assertFalse(any("pkill -f '[s]ync_rollout_batches.sh'" in command for command in commands))
        self.assertFalse(any("fresh adapter" in command or ".trainer-restart-started" in command for command in commands))

    def test_head_only_saved_control_accepts_missing_peer_control(self):
        calls, error = self.run_start(
            control_files={training_ops.trainer_head: "saved control"},
            nodes=training_ops.trainer_hosts,
        )
        self.assertIsNone(error)
        control_hosts = [host for host, command in calls if command.startswith("test -f ") and "control/orch.toml" in command]
        self.assertEqual(control_hosts, [training_ops.trainer_head])
        self.assertEqual(
            {host for host, command in calls if command.endswith(" trainer")},
            set(training_ops.trainer_hosts),
        )

    def test_missing_or_empty_head_control_refuses_before_any_write(self):
        for control_files in ({"n76": "peer control"}, {"n82": "", "n76": "peer control"}):
            with self.subTest(control_files=control_files):
                calls, error = self.run_start(control_files=control_files)
                self.assertIsInstance(error, training_ops.Refused)
                self.assertIn("non-empty control file", str(error))
                self.assertIn(f"on {training_ops.trainer_head}", str(error))
                self.assertEqual(len(calls), 1)
                host, command = calls[0]
                self.assertEqual(host, training_ops.trainer_head)
                self.assertTrue(command.startswith("test -f ") and "test -s " in command)

    def test_adapter_restart_without_head_refuses_before_host_actions(self):
        calls, error = self.run_start(nodes=["n76"])
        self.assertIsInstance(error, training_ops.Refused)
        self.assertIn(f"trainer head {training_ops.trainer_head} in nodes", str(error))
        self.assertEqual(calls, [])

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
        calls, error = self.run_start(mode="cold", resume=False, control_files={})
        self.assertIsNone(error)
        commands = [command for _, command in calls]
        self.assertFalse(any(command.startswith("test -f ") and "control/orch.toml" in command for command in commands))
        orch = next(i for i, command in enumerate(commands) if command.endswith(training_ops.orchestrator_launcher))
        control = next(i for i, command in enumerate(commands) if "orchestrator control file did not arrive" in command)
        trainer = next(i for i, command in enumerate(commands) if command.endswith(" trainer"))
        self.assertLess(orch, control)
        self.assertLess(control, trainer)

    def test_restart_without_adapter_keeps_fresh_broadcast_wait(self):
        calls, error = self.run_start(resume=False, control_files={})
        self.assertIsNone(error)
        commands = [command for _, command in calls]
        self.assertFalse(any(command.startswith("test -f ") and "control/orch.toml" in command for command in commands))
        trainer = next(i for i, command in enumerate(commands) if command.endswith(" trainer"))
        broadcast = next(i for i, command in enumerate(commands) if "fresh adapter ready" in command)
        orch = next(i for i, command in enumerate(commands) if command.endswith(training_ops.orchestrator_launcher))
        self.assertLess(trainer, broadcast)
        self.assertLess(broadcast, orch)

    def test_unstable_origin_refuses_before_deployment_or_start(self):
        calls, error = self.run_start(state_failure="origin")
        self.assertIsInstance(error, training_ops.Refused)
        self.assertFalse(any(command == "deploy dependencies" or command.endswith(" trainer") for _, command in calls))

    def test_cached_trainer_batch_refuses_before_any_start(self):
        calls, error = self.run_start(state_failure="trainer")
        self.assertIsInstance(error, training_ops.Refused)
        self.assertFalse(any(command.endswith(" trainer") or command.endswith(training_ops.orchestrator_launcher) for _, command in calls))

    def test_orchestrator_state_is_checked_before_deployment(self):
        calls, error = self.run_start(state_failure="orchestrator")
        self.assertIsInstance(error, training_ops.Refused)
        self.assertFalse(any(command == "deploy dependencies" for _, command in calls))

    def test_publisher_drain_failure_never_scans_final_state_or_starts(self):
        calls, error = self.run_start(failure=lambda _host, command: "drain_rollout_publisher.py" in command)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(any(command.startswith("check restart trainer") or command.endswith(" trainer") for _, command in calls))

    def test_origin_mutation_after_first_check_refuses_start(self):
        calls, error = self.run_start(changed_origin=True)
        self.assertIsInstance(error, training_ops.Refused)
        self.assertIn("identity changed", str(error))
        self.assertFalse(any(command.endswith(" trainer") or command.endswith(training_ops.orchestrator_launcher) for _, command in calls))


if __name__ == "__main__":
    unittest.main()
