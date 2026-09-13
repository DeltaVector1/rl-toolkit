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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    def test_engine_repair_accepts_explicit_config_without_current_run(self):
        host = training_ops.inference_hosts[0]
        ssh_calls = []

        def fake_ssh(ssh_host, script, timeout=120):
            ssh_calls.append((ssh_host, script, timeout))
            return 0, "issued\n" if len(ssh_calls) == 2 else "watchdog okay\n"

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), mock.patch.object(
            training_ops,
            "current_run",
            side_effect=AssertionError("explicit engine repair must not inspect current run"),
        ), mock.patch.object(
            training_ops,
            "_config_for_run",
            side_effect=AssertionError("explicit engine repair must not scan runs"),
        ):
            result = training_ops.repair(f"engine:{host}:configs/general-agent")

        self.assertEqual(result, "issued")
        self.assertEqual(len(ssh_calls), 3)
        expected_config = f"{training_ops.remote_repo}/configs/general-agent/inference.toml"
        self.assertTrue(all(expected_config in script for _, script, _ in ssh_calls))

    def test_engine_repair_rejects_missing_explicit_config_before_ssh(self):
        with mock.patch.object(training_ops, "_ssh") as ssh:
            with self.assertRaisesRegex(training_ops.Refused, "no trainer.toml"):
                training_ops.repair("engine:h200a:configs/missing")
        ssh.assert_not_called()

    def test_engine_repair_without_config_keeps_current_run_selection(self):
        host = training_ops.inference_hosts[0]
        ssh_calls = []
        selected = {"inference": "configs/general-agent/inference.toml"}

        def fake_ssh(ssh_host, script, timeout=120):
            ssh_calls.append((ssh_host, script, timeout))
            return 0, "issued\n" if len(ssh_calls) == 2 else "watchdog okay\n"

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), mock.patch.object(
            training_ops, "current_run", return_value="legacy-run"
        ), mock.patch.object(
            training_ops, "_config_for_run", return_value=selected
        ) as config_for_run:
            result = training_ops.repair(f"engine:{host}")

        self.assertEqual(result, "issued")
        config_for_run.assert_called_once_with("legacy-run")
        self.assertEqual(len(ssh_calls), 3)

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
            "training_ops/runtime/h200/start_orchestrator.sh",
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
        self.assertIn("TRAINER_ADDRESSES='addr-n82 addr-n76 addr-n77 addr-n78 addr-n80 addr-n83 addr-n84 addr-n85'", script)
        self.assertTrue(script.endswith("--tunnels-only"))
        self.assertNotIn("--restart-tunnels", script)
        self.assertIn("TUNNEL_REPAIR_DEADLINE_S=150 timeout --signal=TERM --kill-after=5s 155s", script)

    def test_adapter_tunnels_reset_is_the_explicit_full_reset(self):
        ssh_calls = []

        def fake_ssh(host, script, timeout=120):
            ssh_calls.append((host, script, timeout))
            return 0, "adapter tunnel reconciliation complete\n"

        with mock.patch.object(training_ops, "deploy", return_value=True), mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_adapter_tunnels("configs/general-agent", reset=True)

        self.assertEqual(result, "adapter tunnel reconciliation complete")
        self.assertEqual(len(ssh_calls), 1)
        self.assertTrue(ssh_calls[0][1].endswith("--restart-tunnels --tunnels-only"))

    def test_repair_dispatch_keeps_reset_explicit(self):
        with mock.patch.object(
            training_ops, "_repair_adapter_tunnels", return_value="ok"
        ) as repair_tunnels:
            self.assertEqual(training_ops.repair("adapter-tunnels:configs/current"), "ok")
            repair_tunnels.assert_called_once_with("configs/current")
            repair_tunnels.reset_mock()
            self.assertEqual(training_ops.repair("adapter-tunnels-reset:configs/current"), "ok")
            repair_tunnels.assert_called_once_with("configs/current", reset=True)

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
            Path(__file__).resolve().parents[2]
            / "training_ops/runtime/h200/start_orchestrator.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        mode_start = source.index("if [ \"$tunnels_only\" = 1 ]; then")
        docker_start = source.index("sudo docker", mode_start)
        self.assertIn("unknown launcher argument", source)
        self.assertIn("adapter tunnel reconciliation did not establish", source)
        self.assertIn('if [ "$shape" != cleanup ] && [ "$destination_port" != "$relay_port" ]', source)
        self.assertIn('body != b"source URL is not configured\\n"', source)
        self.assertIn('china_tunnel_remove_orphan_children "${forwards[@]}"', source)
        self.assertIn('old adapter tunnel ports did not clear for $marker', source)
        self.assertIn('china_tunnel_remove_orphan_children "${china_legacy_cleanup_ports[@]}"', source)
        self.assertIn('china_tunnel_remove_orphan_children "${china_legacy_cleanup_ports[@]}" || return 1', source)
        self.assertIn('china_tunnel_remove_orphan_children || repair_fail', source)
        self.assertIn('china_tunnel_remove_all_owned', source)
        self.assertNotIn('done < <(ps -eo pid=,args=)', source)
        legacy_cleanup = source.split('peer_tunnel_remove_legacy() {', 1)[1].split('\n}\n', 1)[0]
        self.assertIn('process_output=$(ps -eo pid=,args=) || return 1', legacy_cleanup)
        self.assertIn('done <<<"$process_output"', legacy_cleanup)
        self.assertIn('adapter_tunnel_readiness() {', source)
        self.assertIn('tunnel_repair_deadline=$((SECONDS + tunnel_repair_deadline_s))', source)
        self.assertIn('repair_timeout "$probe_seconds"', source)
        self.assertIn('repair_sleep 1 || return 1', source)
        self.assertIn('tunnel_repair_deadline_s=${TUNNEL_REPAIR_DEADLINE_S:-150}', source)
        self.assertIn('adapter_tunnel_exact_snapshot()', source)
        self.assertIn('adapter_tunnel_local_snapshot()', source)
        self.assertIn('adapter_tunnel_remote_snapshot()', source)
        self.assertIn('rollback_failed=0', source)
        self.assertIn('status=125', source)
        self.assertIn('process_start_time()', source)
        self.assertIn('process_identity_matches()', source)
        self.assertIn('rollback refused reused PID', source)
        self.assertIn('direct_ssh_count=0', source)
        self.assertIn('peer_tunnel_children_unchanged()', source)
        self.assertIn('print(f"RANGE {first} {last} {last - first + 1}")', source)
        self.assertNotIn('readarray -t lora_source_ports < <(python3', source)
        self.assertIn('repair_register_tunnel', source)
        self.assertIn('repair_signal() {', source)
        self.assertIn("trap 'repair_signal' TERM HUP INT", source)
        self.assertIn('if ! repair_cleanup_created_tunnels; then status=125; fi', source)
        readiness_function = source.index('adapter_tunnel_readiness()')
        final_start = source.index('for _ in $(seq 1 10); do', readiness_function)
        final_block = source[final_start:source.index('echo "adapter tunnel final local/remote snapshots failed"', final_start)]
        self.assertEqual(final_block.count('adapter_tunnel_local_snapshot'), 2)
        self.assertEqual(final_block.count('adapter_tunnel_remote_snapshot'), 2)
        self.assertLess(
            final_block.index('adapter_tunnel_local_snapshot'),
            final_block.index('adapter_tunnel_remote_snapshot'),
        )
        self.assertLess(
            final_block.index('adapter_tunnel_remote_snapshot'),
            final_block.rindex('adapter_tunnel_local_snapshot'),
        )
        readiness_call = source.index('if ! adapter_tunnel_readiness; then')
        self.assertLess(readiness_call, docker_start)
        self.assertLess(mode_start, docker_start)

    def test_orchestrator_prepares_broadcast_steps_for_appuser(self):
        launcher = (
            Path(__file__).resolve().parents[2]
            / "training_ops/runtime/h200/start_orchestrator.sh"
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
            Path(__file__).resolve().parents[1] / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn("check adapter transfer progress and logs", source)
        self.assertNotIn("Read references/errors.md and repair before waiting further", source)

    def test_null_harness_bridge_disables_hidden_sdk_deadline_and_retry(self):
        root = Path(__file__).resolve().parents[2]
        harness_path = root / "prime-rl/deps/verifiers/verifiers/v1/harnesses/null/harness.py"
        launcher = (
            root / "training_ops/runtime/h200/start_orchestrator.sh"
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
        root = Path(__file__).resolve().parents[2]
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

    def test_trainer_failure_watchdog_probe_requires_exact_identity(self):
        probe = training_ops.orchestrator_probe
        start = probe.index("trainer_fail_wd=0")
        end = probe.index('echo "ORCH_LAST', start)
        block = probe[start:end]
        expected = {
            "RUN": "run",
            "RUN_NAME": "run_run",
            "TRAINER_ADDRESSES": "10.92.20.82 10.92.20.76 10.92.20.77",
            "TRAINER_CT": "trainer",
            "ORCH_CT": "orchestrator",
        }

        def run_probe(entries):
            with tempfile.TemporaryDirectory() as temporary:
                proc_root = Path(temporary) / "proc"
                script_path = "/data/deployment/trainer_failure_watchdog.sh"
                script = block.replace("/proc/", f"{proc_root}/")
                script = script.replace(
                    "__TRAINER_WATCHDOG_SCRIPT__", script_path
                )
                for key, value in expected.items():
                    placeholder = "__ORCH__" if key == "ORCH_CT" else f"__{key}__"
                    script = script.replace(placeholder, value)
                for pid, argv, environment, process_group in entries:
                    process = proc_root / str(pid)
                    process.mkdir(parents=True)
                    (process / "cmdline").write_bytes(
                        b"\0".join(value.encode() for value in argv) + b"\0"
                    )
                    (process / "environ").write_bytes(
                        b"\0".join(value.encode() for value in environment) + b"\0"
                    )
                    (process / "stat").write_text(
                        f"{pid} (bash) S 1 {process_group} 1 0 0 0 0 0\n",
                        encoding="ascii",
                    )
                result = subprocess.run(
                    ["bash", "-c", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(
                    line
                    for line in result.stdout.splitlines()
                    if line.startswith("TRAINER_FAIL_WD ")
                )
                return int(line.rsplit(" ", 1)[1])

        def valid_environment():
            return [f"{key}={value}" for key, value in expected.items()]

        def valid_process(pid, command=("bash",), environment=None, process_group=None):
            return (
                pid,
                (*command, "/data/deployment/trainer_failure_watchdog.sh"),
                valid_environment() if environment is None else environment,
                pid if process_group is None else process_group,
            )

        self.assertEqual(
            run_probe(
                [
                    valid_process(100, ("bash",)),
                    valid_process(101, ("/bin/bash",), process_group=100),
                    valid_process(102, ("/usr/bin/bash",), process_group=100),
                ]
            ),
            1,
        )
        config_order_environment = valid_environment()
        config_order_environment[2] = "TRAINER_ADDRESSES=10.92.20.76 10.92.20.77 10.92.20.82"
        self.assertEqual(
            run_probe([valid_process(106, environment=config_order_environment)]),
            0,
        )
        self.assertEqual(
            run_probe(
                [
                    (
                        103,
                        ("bash", "/data/deployment/trainer_failure_watchdog.sh", "extra"),
                        valid_environment(),
                        103,
                    ),
                    (
                        104,
                        ("bash", "-c", "echo /data/deployment/trainer_failure_watchdog.sh"),
                        valid_environment(),
                        104,
                    ),
                    (105, ("sleep", "1"), valid_environment(), 105),
                ]
            ),
            0,
        )
        for key in expected:
            mismatched = valid_environment()
            mismatched[list(expected).index(key)] = f"{key}=wrong"
            self.assertEqual(
                run_probe([valid_process(200, ("bash",), mismatched)]),
                0,
                key,
            )
        for valid_count in (0, 1, 2):
            entries = [
                valid_process(300 + index, ("bash",))
                for index in range(valid_count)
            ]
            count = run_probe(entries)
            self.assertEqual(count, valid_count)
            self.assertEqual(count == 1, valid_count == 1)

    def test_trainer_addresses_match_head_first_watchdog_launch_order(self):
        with mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            self.assertEqual(
                training_ops._trainer_addresses(["n76", "n82", "n77"]),
                "addr-n82 addr-n76 addr-n77",
            )

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
        self.assertIn("bash /data/glm52-rl/deployment/training_ops/runtime/h200/start_orchestrator.sh", ssh_scripts[-1][0])
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
            "TRAINER_ADDRESSES='addr-n82 addr-n76 addr-n77 addr-n78 addr-n80 addr-n83 addr-n84 addr-n85'",
            script,
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", script)
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", script)
        self.assertIn(
            "bash /data/glm52-rl/deployment/training_ops/runtime/h200/start_orchestrator.sh",
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
