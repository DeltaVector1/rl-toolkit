import ast
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


def _load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    values = dict(namespace or {})
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), values)
    return values[name]


class OrchestratorRepairTest(unittest.TestCase):
    def test_adapter_tunnels_uses_explicit_config_dir_and_tunnels_only(self):
        ssh_calls = []

        def fake_ssh(host, script, timeout=120):
            ssh_calls.append((host, script, timeout))
            return 0, "adapter tunnel reconciliation complete\n"

        with mock.patch.object(training_ops, "deploy", return_value=True) as deploy, mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_adapter_tunnels("configs/general-agent")

        self.assertEqual(result, "adapter tunnel reconciliation complete")
        deploy.assert_called_once_with(
            ".agents/skills/training-ops/runtime/h200/start_orchestrator.sh",
            nodes=[training_ops.orchestrator_host],
        )
        self.assertEqual(len(ssh_calls), 1)
        host, script, timeout = ssh_calls[0]
        self.assertEqual(host, training_ops.orchestrator_host)
        self.assertEqual(timeout, 180)
        self.assertIn("RUN=glm-perf-3 RUN_NAME=run_glm-perf-3", script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/general-agent/orchestrator.toml",
            script,
        )
        self.assertIn("TRAINER_ADDRESSES='addr-n76 addr-n77 addr-n78 addr-n80 addr-n82 addr-n83 addr-n84 addr-n85'", script)
        self.assertTrue(script.endswith("--tunnels-only"))

    def test_adapter_tunnels_does_not_fall_back_from_trainer_only_run(self):
        with mock.patch.object(
            training_ops,
            "_config_for_run",
            return_value={"trainer": "configs/general-agent/trainer.toml"},
        ), mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"):
            with self.assertRaisesRegex(training_ops.Refused, "no orchestrator config"):
                training_ops._repair_adapter_tunnels()

    def test_tunnels_only_exits_before_service_start(self):
        launcher = (
            Path(__file__).resolve().parents[4]
            / ".agents/skills/training-ops/runtime/h200/start_orchestrator.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        mode_start = source.index("if [ \"$tunnels_only\" = 1 ]; then")
        docker_start = source.index("sudo docker", mode_start)
        self.assertIn("unknown launcher argument", source)
        self.assertIn("adapter tunnel reconciliation did not establish", source)
        self.assertLess(mode_start, docker_start)

    def test_orchestrator_prepares_broadcast_steps_for_appuser(self):
        launcher = (
            Path(__file__).resolve().parents[4]
            / ".agents/skills/training-ops/runtime/h200/start_orchestrator.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        start = source.index('broadcast_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME/broadcasts"')
        end = source.index('for p in "${ports[@]}"', start)
        preflight = source[start:end]
        self.assertIn(
            'sudo install -d -o 1000 -g 1000 -m 0755 "$broadcast_dir" "$broadcast_dir/step_0"',
            preflight,
        )
        self.assertIn('[[ "$step_dir" =~ /step_[0-9]+$ ]]', preflight)
        self.assertIn('[ ! -L "$step_dir" ]', preflight)
        self.assertIn('sudo install -d -o 1000 -g 1000 -m 0755 "$step_dir"', preflight)
        self.assertIn("sudo -u '#1000' test -w \"$step_dir\"", preflight)
        self.assertIn("sudo -u '#1000' test -w \"$broadcast_dir\"", preflight)
        self.assertNotIn("chown -R", preflight)
        self.assertLess(source.index('broadcast_dir="'), source.index('sudo docker run -d --name "$ORCH_CT"'))

    def test_watch_idle_message_requests_transfer_check(self):
        source = (
            Path(__file__).resolve().parents[1] / "src/training_ops/__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn("check adapter transfer progress and logs", source)
        self.assertNotIn("Read references/errors.md and repair before waiting further", source)

    def test_null_harness_bridge_disables_hidden_sdk_deadline_and_retry(self):
        root = Path(__file__).resolve().parents[4]
        harness_path = root / "prime-rl/deps/verifiers/verifiers/v1/harnesses/null/harness.py"
        launcher = (
            root / ".agents/skills/training-ops/runtime/h200/start_orchestrator.sh"
        ).read_text(encoding="utf-8")
        event = asyncio.Event()
        client = SimpleNamespace(close=mock.AsyncMock())

        async def pending(**_kwargs):
            await event.wait()

        create = mock.AsyncMock(side_effect=pending)
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        client_factory = mock.Mock(return_value=client)

        launch = _load_function(
            harness_path,
            "launch",
            {
                "APIStatusError": type("APIStatusError", (Exception,), {}),
                "AsyncOpenAI": client_factory,
                "ProgramResult": lambda **kwargs: kwargs,
                "message_to_wire": lambda message: message,
            },
        )
        harness = SimpleNamespace(resolve_prompt=lambda _data: ("system", "prompt"))
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                asyncio.wait_for(
                    launch(
                        harness,
                        SimpleNamespace(model="mock"),
                        SimpleNamespace(task=SimpleNamespace(data={})),
                        None,
                        "http://127.0.0.1",
                        "secret",
                        {},
                    ),
                    timeout=0.05,
                )
            )
        self.assertEqual(create.await_count, 1)
        client_factory.assert_called_once()
        client_kwargs = client_factory.call_args.kwargs
        self.assertIsNone(client_kwargs["timeout"])
        self.assertEqual(client_kwargs["max_retries"], 0)
        self.assertEqual(client.close.await_count, 1)
        harness_mount = (
            '-v "$deployment/prime-rl/deps/verifiers/verifiers/v1/'
            'harnesses/null/harness.py":/app/deps/verifiers/verifiers/v1/'
            'harnesses/null/harness.py:ro \\\n'
        )
        self.assertIn(harness_mount, launcher)
        docker_start = launcher.index('sudo docker run -d --name "$ORCH_CT"')
        entrypoint = launcher.index(
            '--entrypoint /app/.venv/bin/orchestrator', docker_start
        )
        self.assertLess(docker_start, launcher.index(harness_mount))
        self.assertLess(launcher.index(harness_mount), entrypoint)

    def test_harness_timeout_retries_only_after_failed_policy_call(self):
        root = Path(__file__).resolve().parents[4]
        dispatcher_path = root / "prime-rl/src/prime_rl/orchestrator/dispatcher.py"
        check = _load_function(
            dispatcher_path, "_is_transient_policy_failure", {"Rollout": object}
        )
        probe = SimpleNamespace(policy=SimpleNamespace(model_name="policy"))
        rollout = lambda typ, msg, calls=(): SimpleNamespace(
            error=SimpleNamespace(type=typ, message=msg), calls=list(calls)
        )
        transient = SimpleNamespace(model="policy", error=SimpleNamespace())
        self.assertFalse(check(probe, rollout("HarnessError", "API timeout")))
        self.assertTrue(check(probe, rollout("HarnessError", "API timeout", [transient])))
        self.assertTrue(
            check(probe, rollout("ProviderError", "connection timeout", [transient]))
        )

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

    def test_terminal_stage_states_do_not_require_an_age(self):
        script = training_ops._orchestrator_resume_safety_script([8000])
        with tempfile.TemporaryDirectory() as temporary:
            fake_curl = Path(temporary) / "curl"
            fake_curl.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  */health*|*/liveness*) printf '200' ;;\n"
                "  */metrics*) printf 'vllm:num_requests_running 0\\n' ;;\n"
                "  */staging_status*) printf '%s' \"$STAGING_RESPONSE\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{temporary}:{os.environ['PATH']}",
            }
            for response in (
                '{"status":"idle","age_seconds":0}',
                '{"status":"ready"}',
                '{"status":"failed","error":"transfer failed"}',
                '{"status":"staging","age_seconds":7200}',
            ):
                environment["STAGING_RESPONSE"] = response
                result = subprocess.run(
                    ["bash", "-c", script],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, response)

            environment["STAGING_RESPONSE"] = '{"status":"staging","age_seconds":7201}'
            result = subprocess.run(
                ["bash", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("staging exceeds the bounded limit", result.stderr)

    def test_safety_probe_rejects_unknown_stage_state(self):
        script = training_ops._orchestrator_resume_safety_script([8000])
        self.assertIn("idle|ready|failed", script)
        self.assertIn('echo "unknown staging status on $port"', script)
        self.assertIn("staging age is missing", script)

    def test_orchestrator_state_reports_absent_container(self):
        script = training_ops._orchestrator_state_script(
            "glm52-decensor-orchestrator"
        )
        self.assertIn("if name=$(sudo docker inspect", script)
        self.assertIn("echo absent", script)

    def test_absent_orchestrator_recreation_checks_that_it_stays_absent(self):
        with mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            script = training_ops._recreate_orchestrator_script(
                "glm-perf-3",
                {
                    "kind": "glm52",
                    "orchestrator": "configs/general-agent/orchestrator.toml",
                },
                was_running=False,
                observed_state="absent",
            )
        self.assertIn("[ absent = absent ]", script)
        self.assertIn('[ -z "$name" ]', script)
        self.assertIn("start_orchestrator.sh", script)

    def test_absent_orchestrator_is_recreated_after_status_reports_down(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=0, reachable=True)],
            batch="batch 112/1024; 0 inflight",
        )
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=120):
            ssh_scripts.append(script)
            if "echo absent" in script:
                return 0, "absent\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy", return_value=True), mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_orchestrator()

        self.assertIn("recreated exact orchestrator", result)
        self.assertIn("[ absent = absent ]", ssh_scripts[-1])
        self.assertIn("start_orchestrator.sh", ssh_scripts[-1])

    def test_stopped_orchestrator_is_started_after_all_guards_pass(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=0, reachable=True)],
            batch="batch 112/1024; 128 inflight",
        )
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=120):
            ssh_scripts.append((script, timeout))
            if "configured orchestrator container state is unknown" in script:
                return 0, "stopped\n"
            if "docker start" in script:
                return 0, "started exact orchestrator container glm52-decensor-orchestrator\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
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
        self.assertIn("RUN=glm-perf-3", ssh_scripts[-1][0])
        self.assertIn("RUN_NAME=run_glm-perf-3", ssh_scripts[-1][0])
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/general-agent/orchestrator.toml",
            ssh_scripts[-1][0],
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", ssh_scripts[-1][0])
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", ssh_scripts[-1][0])

    def test_running_orchestrator_is_refused_without_a_down_status(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=True,
            nodes=[training_ops.NodeState(host="n82", gpu_util=0, reachable=True)],
            batch="batch 112/1024; 128 inflight",
        )

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "_ssh", return_value=(0, "running\n")), mock.patch.object(
            training_ops, "deploy"
        ) as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "healthy"):
                training_ops._repair_orchestrator()

        deploy.assert_not_called()

    def test_failed_running_orchestrator_can_be_recreated(self):
        self.assertIn("ORCH_FAILED", training_ops.orchestrator_probe)
        before = training_ops.Status(
            trainer_up=True,
            orch_failed=True,
            nodes=[training_ops.NodeState(host="n82", gpu_util=0, reachable=True)],
            batch="batch 1024/1024; 0 inflight",
        )

        def fake_ssh(_host, script, timeout=120):
            if "configured orchestrator container state is unknown" in script:
                return 0, "running\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy", return_value=True), mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_orchestrator()

        self.assertIn("recreated exact orchestrator", result)

    def test_recreate_script_uses_active_config_and_exact_launcher(self):
        with mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            script = training_ops._recreate_orchestrator_script(
                "glm52",
                {
                    "kind": "glm52",
                    "orchestrator": "configs/general-agent/orchestrator.toml",
                },
                was_running=True,
            )

        self.assertIn("RUN=glm52", script)
        self.assertIn("RUN_NAME=run_glm52", script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/general-agent/orchestrator.toml",
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
        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "deploy") as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "idle or waiting"):
                training_ops._repair_orchestrator()
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
