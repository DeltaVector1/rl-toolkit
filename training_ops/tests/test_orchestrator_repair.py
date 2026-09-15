import ast
import asyncio
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import time
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
        self.assertEqual(deploy.call_args_list, [
            mock.call("configs/general-agent/orchestrator.toml", nodes=[training_ops.orchestrator_host]),
            mock.call("training_ops/runtime/h200/start_orchestrator.sh", nodes=[training_ops.orchestrator_host]),
        ])
        self.assertEqual(len(ssh_calls), 1)
        host, script, timeout = ssh_calls[0]
        self.assertEqual(host, training_ops.orchestrator_host)
        self.assertEqual(timeout, 330)
        self.assertIn("RUN=glm-perf-3 RUN_NAME=run_glm-perf-3", script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/general-agent/orchestrator.toml",
            script,
        )
        self.assertIn("TRAINER_ADDRESSES='addr-n82 addr-n76 addr-n77 addr-n78 addr-n80 addr-n83 addr-n84 addr-n85'", script)
        self.assertTrue(script.endswith("--tunnels-only"))
        self.assertNotIn("--restart-tunnels", script)
        self.assertIn("TUNNEL_REPAIR_DEADLINE_S=300 timeout --signal=TERM --kill-after=5s 305s", script)

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

    def test_adapter_tunnels_stops_if_selected_config_cannot_be_verified(self):
        with mock.patch.object(training_ops, "deploy", return_value=False) as deploy, mock.patch.object(
            training_ops, "_ssh"
        ) as ssh, mock.patch.object(training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"):
            with self.assertRaisesRegex(RuntimeError, "orchestrator.toml"):
                training_ops._repair_adapter_tunnels("configs/general-agent")
        deploy.assert_called_once_with("configs/general-agent/orchestrator.toml", nodes=[training_ops.orchestrator_host])
        ssh.assert_not_called()

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
        docker_start = source.index('sudo docker run -d --name "$ORCH_CT"', mode_start)
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
        self.assertIn('adapter_relay_start() {', source)
        self.assertIn('adapter relay unavailable', source)
        self.assertIn('source URL is not configured', source)
        self.assertIn('headers={"Host": f"127.0.0.1:{port}"}', source)
        self.assertIn('first wrong adapter relay:', source)
        self.assertIn('body = response.read(257)', source)
        self.assertIn('body[:256]', source)
        self.assertIn('truncated={truncated}', source)
        self.assertIn("[ \"$relay_probe\" = $'source URL is not configured\\n' ]", source)
        self.assertIn("relay_status=${relay_probe: -3}", source)
        self.assertIn('Host: 127.0.0.1:${lora_source_ports[0]}', source)
        self.assertIn('tunnel_repair_deadline=$((SECONDS + tunnel_repair_deadline_s))', source)
        self.assertIn('repair_timeout "$probe_seconds"', source)
        self.assertIn('repair_sleep 1 || return 1', source)
        self.assertIn('tunnel_repair_deadline_s=${TUNNEL_REPAIR_DEADLINE_S:-300}', source)
        self.assertIn('adapter_tunnel_exact_snapshot()', source)
        self.assertIn('adapter_tunnel_local_snapshot()', source)
        self.assertIn('adapter_tunnel_remote_snapshot()', source)
        self.assertIn('adapter_tunnel_stability_snapshot()', source)
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
        final_block = source[final_start:source.index('echo "adapter tunnel final stability check failed"', final_start)]
        self.assertEqual(final_block.count('adapter_tunnel_stability_snapshot'), 1)
        self.assertNotIn('adapter_tunnel_local_snapshot', final_block)
        readiness_call = source.index('if ! adapter_tunnel_readiness; then')
        self.assertLess(readiness_call, docker_start)
        self.assertLess(mode_start, docker_start)

    def test_relay_check_precedes_prevalidated_tunnel_fast_path(self):
        launcher = (
            Path(__file__).resolve().parents[2]
            / "training_ops/runtime/h200/start_orchestrator.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        relay_check = source.index('if ! adapter_relay_start; then')
        fast_path = source.index('tunnels_prevalidated=0')
        self.assertLess(relay_check, fast_path)
        self.assertEqual(source.count('if ! adapter_relay_start; then'), 1)

    def test_relay_reuse_requires_configured_source_identity(self):
        launcher = (
            Path(__file__).resolve().parents[2]
            / "training_ops/runtime/h200/start_orchestrator.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        relay_start = source.index("adapter_relay_start() {")
        relay_end = source.index("\n}\n\nchina_legacy_cleanup_required", relay_start)
        relay_block = source[relay_start:relay_end]
        self.assertIn(
            'relay_identity=$(sudo docker inspect "$relay_container"', relay_block
        )
        self.assertIn(
            '--format \'{{index .Config.Labels "com.glm52.adapter-relay.source-url"}}\'',
            relay_block,
        )
        self.assertIn('[ "$relay_identity" = "$relay_source_url" ]', relay_block)
        self.assertIn(
            '--label "com.glm52.adapter-relay.source-url=$relay_source_url"',
            relay_block,
        )

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


class SourcePortProbeTest(unittest.TestCase):
    def test_source_probe_runs_before_peer_changes_and_is_not_repeated(self):
        script = Path(__file__).resolve().parents[1] / "runtime/h200/start_orchestrator.sh"
        source = script.read_text()
        for name in ("adapter_tunnel_local_snapshot", "adapter_tunnel_readiness"):
            function = source.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]
            self.assertNotIn("china_source_ports_ready", function)
            self.assertNotIn("china_tunnel_connection", function)
        phase = "china_lane_size=2\n" + source.split("china_lane_size=2\n", 1)[1]
        phase = phase.split('if [ "$secondary_enabled" = 1 ]; then', 1)[0]
        for probe_status in (0, 1):
            fixtures = f'''
lora_source_ports=(18100 18101)
adapter_port=8000
tunnels_only=1
china_tunnel_connection() {{ echo china; }}
china_source_ports_ready() {{ echo source; return {probe_status}; }}
peer_tunnel() {{ echo peer; }}
repair_fail() {{ exit 1; }}
'''
            result = subprocess.run(["bash", "-c", fixtures + phase], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, probe_status, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["china", "source"] + ([] if probe_status else ["peer"]))

    def test_production_probe_uses_fixed_real_seconds(self):
        script = Path(__file__).resolve().parents[1] / "runtime/h200/start_orchestrator.sh"
        section = script.read_text().split("china_source_ports_ready() {", 1)[1].split("\ntunnel_command_has()", 1)[0]
        wrapper, body = section.split("<<'PY'\n", 1)
        self.assertIn("local probe_seconds=40", wrapper)
        self.assertNotIn("repair_timeout", wrapper)
        self.assertIn("budget = 40.0", body)
        self.assertNotIn("scale", body)
        self.assertIn("enumerate(((32, 2, 10), (4, 5, 12), (2, 5, 12)), 1)", body)
        self.assertIn("asyncio.timeout(socket_seconds)", body)
        self.assertIn("asyncio.wait(tasks, timeout=stage_seconds)", body)
        self.assertIn("asyncio.wait_for(check_ports(), timeout=budget)", body)
        result = subprocess.run(
            ["bash", "-c", "china_source_ports_ready() {" + section + "\n" + r'''
repair_timeout() { echo 'unexpected repair timeout call' >&2; echo 5; }
ps() { echo 'bash source # china-tunnel-lora-0'; }
timeout() { printf '%s\n' "$*" >&2; cat >/dev/null; echo 2; }
lora_source_ports=(18100 18101)
china_source_ports_ready
'''], capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.strip(), "--signal=TERM --kill-after=1s 40 python3 - 18100 18101")

    def run_probe(self, fail_until, budget=40, delay=False, stall_first=None, response=None):
        script = Path(__file__).resolve().parents[1] / "runtime/h200/start_orchestrator.sh"
        body = script.read_text().split("china_source_ports_ready() {", 1)[1]
        body = body.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        if budget != 40:
            fixture_stages = tuple((workers, seconds, stage * budget / 40)
                                   for workers, seconds, stage in ((32, 2, 10), (4, 5, 12), (2, 5, 12)))
            body = body.replace("budget = 40.0", f"budget = {budget!r}")
            body = body.replace("((32, 2, 10), (4, 5, 12), (2, 5, 12))", repr(fixture_stages))
            body = body.replace("asyncio.sleep(0.25)", f"asyncio.sleep({0.25 * budget / 40!r})")
        calls = {}
        closed = []

        class Reader:
            def __init__(self, port, attempt):
                self.port, self.attempt = port, attempt

            async def readuntil(self, separator):
                if delay or (self.port == stall_first and self.attempt == 1):
                    await asyncio.sleep(60)
                if self.attempt <= fail_until.get(self.port, 0):
                    raise TimeoutError()
                return response or b"HTTP/1.1 200 OK\r\nServer: SimpleHTTP/0.6\r\n\r\n"

        class Writer:
            def __init__(self, port):
                self.port = port

            def write(self, data):
                assert data.startswith(b"HEAD / HTTP/1.1\r\n")

            async def drain(self):
                pass

            def close(self):
                closed.append(self.port)

        async def connect(host, port, limit):
            calls[port] = calls.get(port, 0) + 1
            return Reader(port, calls[port]), Writer(port)

        stdout, stderr = io.StringIO(), io.StringIO()
        started = time.monotonic()
        status = 0
        with mock.patch("asyncio.open_connection", connect), mock.patch.object(
            sys, "argv", ["probe", "18100", "18107"]
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(body, str(script), "exec"), {})
            except SystemExit as error:
                status = error.code
        self.assertEqual(len(closed), sum(calls.values()))
        return status, calls, stdout.getvalue(), stderr.getvalue(), time.monotonic() - started

    def test_retries_only_ports_that_failed_the_previous_attempt(self):
        status, calls, output, errors, _ = self.run_probe({18101: 1, 18105: 2})
        self.assertEqual(status, 0)
        self.assertEqual(output, "8\n")
        self.assertEqual(calls[18101], 2)
        self.assertEqual(calls[18105], 3)
        self.assertTrue(all(count == 1 for port, count in calls.items() if port not in {18101, 18105}))
        self.assertRegex(errors, r"attempt=1 elapsed=[\d.]+s failed=2")
        self.assertRegex(errors, r"attempt=2 elapsed=[\d.]+s failed=1")

    def test_returns_only_persistent_failures(self):
        status, calls, output, errors, _ = self.run_probe({18101: 1, 18105: 99})
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertEqual(calls[18101], 2)
        self.assertEqual(calls[18105], 3)
        self.assertEqual(errors.splitlines()[-1], "persistent adapter source failures: 18105")

    def test_socket_and_retries_obey_the_whole_probe_deadline(self):
        status, _, output, errors, elapsed = self.run_probe(
            dict.fromkeys(range(18100, 18108), 99), budget=0.6, delay=True
        )
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("stage timeout", errors)
        self.assertIn("attempt=3", errors)
        self.assertLess(elapsed, 1)

    def test_hung_first_stage_does_not_consume_retry_budget(self):
        status, calls, output, errors, elapsed = self.run_probe({}, budget=0.6, stall_first=18101)
        self.assertEqual(status, 0)
        self.assertEqual(output, "8\n")
        self.assertEqual(calls[18101], 2)
        self.assertTrue(all(count == 1 for port, count in calls.items() if port != 18101))
        self.assertIn("18101:stage timeout", errors)
        self.assertRegex(errors, r"attempt=2 elapsed=[\d.]+s failed=0")
        self.assertLess(elapsed, 1)

    def test_rejects_wrong_http_status_or_server(self):
        for response, reason in (
            (b"HTTP/1.1 403 Forbidden\r\nServer: SimpleHTTP/0.6\r\n\r\n", "status="),
            (b"HTTP/1.1 200 OK\r\nServer: unrelated\r\n\r\n", "server="),
        ):
            with self.subTest(response=response):
                status, calls, output, errors, _ = self.run_probe({}, budget=0.6, response=response)
                self.assertEqual(status, 1)
                self.assertEqual(output, "")
                self.assertTrue(all(count == 3 for count in calls.values()))
                self.assertIn(reason, errors)


class TunnelRepairBoundsTest(unittest.TestCase):
    source = (Path(__file__).resolve().parents[2] / "training_ops/runtime/h200/start_orchestrator.sh").read_text()

    def function(self, name):
        return name + "() {" + self.source.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"

    def shell(self, functions, body):
        return subprocess.run(
            ["bash", "-uc", "\n".join(self.function(name) for name in functions) + body],
            text=True, capture_output=True, timeout=3,
        )

    def peer_config(self, peers, urls):
        program = self.source.split('if ! peer_settings=$(python3 - "$nodes_file" "$config_file" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            nodes = Path(directory) / "nodes.toml"
            config = Path(directory) / "orchestrator.toml"
            nodes.write_text('[hosts]\ninference=' + repr(peers) + '\norchestrator="h200a"\n'
                '[inference.h200b]\nssh_host="192.0.2.1"\nssh_port=20099\nssh_user="root"\n'
                'peer_ssh_identity_file="/data/test/key"\napi_tunnel_port=18049\n')
            config.write_text('[model.client]\nbase_url=' + repr(urls) + '\n')
            return subprocess.run([sys.executable, "-c", program, str(nodes), str(config)], text=True, capture_output=True)

    def test_peer_config_requires_exact_url_before_tunnels_change(self):
        for urls in (["http://127.0.0.1:8000/v1"], ["http://127.0.0.1:18049/v1/extra"]):
            result = self.peer_config(["h200a", "h200b"], urls)
            self.assertNotEqual(result.returncode, 0)
            for text in ("h200b", "nodes.toml", "orchestrator.toml", "http://127.0.0.1:18049/v1", "no tunnels changed"):
                self.assertIn(text, result.stderr)

    def test_peer_config_supports_single_host_and_enabled_peer(self):
        single = self.peer_config(["h200a"], ["http://127.0.0.1:8000/v1"])
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertEqual(single.stdout, "none - 0 0 - 0 0\n")
        dual = self.peer_config(["h200a", "h200b"], ["http://127.0.0.1:18049/v1"])
        self.assertEqual(dual.returncode, 0, dual.stderr)
        self.assertTrue(dual.stdout.endswith("18049 1\n"))

    def test_all_comma_separated_missing_lanes_are_repaired_once(self):
        result = self.shell(["peer_repair_missing_remote_lanes"], '''
lora_source_ports=({18100..18107}); relay_port=18200
peer_missing_remote_ports=18100,18101,18104,18107,18104
pgrep_matches() { return 0; }
peer_tunnel_connection() { echo "$1"; }
peer_repair_missing_remote_lanes
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(result.stdout.splitlines()), ["lora-0", "lora-2", "lora-3"])

    def test_invalid_missing_ports_fail_before_any_repair(self):
        for value in ("18100,,18104", "18100,", "18100+1", "18100,19000", "variable", ""):
            result = self.shell(["peer_repair_missing_remote_lanes"], f'''
lora_source_ports=({{18100..18107}}); relay_port=18200
peer_missing_remote_ports='{value}'
pgrep_matches() {{ echo unexpected; }}
peer_tunnel_connection() {{ echo unexpected; }}
peer_repair_missing_remote_lanes
''')
            self.assertNotEqual(result.returncode, 0, value)
            self.assertEqual(result.stdout, "", value)

    def test_fractional_sleep_is_valid_and_still_checks_deadline(self):
        result = self.shell(["repair_sleep"], '''
repair_remaining() { return 0; }
sleep() { echo "$1"; }
repair_sleep 0.1 || exit 1
repair_remaining() { return 1; }
if repair_sleep 0.1; then exit 1; fi
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0.1\n")

    def test_many_stuck_groups_share_one_cleanup_budget(self):
        program = self.function("repair_cleanup_remaining").replace("20000000", "100000") + self.function("repair_cleanup_call") + '''
repair_total_deadline=0; repair_cleanup_spent_us=0; repair_cleanup_active_us=0
repair_cleanup_completed=0; calls=0
stuck_group() {
  calls=$((calls + 1))
  while repair_cleanup_remaining; do sleep 0.01; done
  return 1
}
for group in {1..100}; do repair_cleanup_call stuck_group || :; done
echo "$calls:$repair_cleanup_completed"
'''
        started = time.monotonic()
        result = subprocess.run(["bash", "-uc", program], text=True, capture_output=True, timeout=2)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1:1\n")
        self.assertIn("stage=stuck_group completed=1", result.stderr)

    def test_successful_cleanup_keeps_one_shared_counter(self):
        result = self.shell(["repair_cleanup_remaining", "repair_cleanup_call"], '''
repair_total_deadline=0; repair_cleanup_spent_us=0; repair_cleanup_active_us=0
repair_cleanup_completed=0
stop_group() { sleep 0.01; }
for group in {1..4}; do repair_cleanup_call stop_group || exit 1; done
[ "$repair_cleanup_spent_us" -ge 40000 ] || exit 1
echo "$repair_cleanup_completed"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "4\n")

    def test_cleanup_call_rejects_success_returned_after_deadline(self):
        result = self.shell(["repair_cleanup_remaining", "repair_cleanup_call"], '''
repair_total_deadline=0; repair_cleanup_limit_us=10000
repair_cleanup_spent_us=0; repair_cleanup_active_us=0; repair_cleanup_completed=0
late_success() { sleep 0.03; return 0; }
repair_cleanup_call late_success
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("time limit reached: stage=late_success", result.stderr)

    def test_incomplete_cleanup_reports_125_within_real_twenty_second_limit(self):
        functions = ("repair_cleanup_remaining", "repair_cleanup_call", "tunnel_wait_group_exit", "repair_exit")
        program = "\n".join(self.function(name) for name in functions) + '''
repair_total_deadline=0; repair_cleanup_limit_us=20000000
repair_cleanup_spent_us=30000000; repair_cleanup_active_us=0; repair_cleanup_completed=64
repair_stage=remote-readiness; repair_started_seconds=$SECONDS; repair_committed=0
tunnel_live_group() { echo '4242 1'; }
tunnel_stop_identity_matches() { return 0; }
kill() { return 0; }
repair_cleanup_all() {
  local status=0
  for group in 4242 4243 4244; do
    tunnel_wait_group_exit "$group" || status=1
  done
  return "$status"
}
repair_release_lock() { return 0; }
repair_exit 7
'''
        started = time.monotonic()
        result = subprocess.run(["bash", "-uc", program], text=True, capture_output=True, timeout=21)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 20)
        self.assertLessEqual(elapsed, 20.5)
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn("cause_status=7 stage=remote-readiness", result.stderr)
        self.assertIn("cleanup=failed", result.stderr)
        self.assertIn("migration_stops=64 rollback_stops=1", result.stderr)
        print(f"ordinary cleanup deadline: {elapsed:.3f}s, status125, cause_status7")

    def test_exit_keeps_original_failure_stage_and_cleanup_result(self):
        for cleanup_status, status, label in ((0, 1, "passed"), (1, 125, "failed")):
            result = self.shell(["repair_exit"], f'''
repair_stage=remote-readiness; repair_started_seconds=$SECONDS; repair_committed=0
repair_cleanup_call() {{ return {cleanup_status}; }}
repair_release_lock() {{ return 0; }}
repair_exit 1
''')
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertIn(f"status={status} cause_status=1 stage=remote-readiness", result.stderr)
            self.assertIn(f"elapsed=0s cleanup={label}", result.stderr)
            self.assertIn("status=1 stage=remote-readiness elapsed=0s cleanup=starting", result.stderr.splitlines()[0])

    def test_production_cleanup_reserve_does_not_raise_total_limit(self):
        preamble = self.source.split("\nrepair_cleanup_remaining() {", 1)[0]
        result = subprocess.run(["bash", "-uc", preamble + '\necho "$tunnel_repair_deadline:$repair_total_deadline"', "bash", "--tunnels-only"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "270:300\n")

    def test_explicit_reset_uses_existing_total_deadline_for_cleanup(self):
        preamble = self.source.split("\nrepair_cleanup_remaining() {", 1)[0]
        result = subprocess.run(["bash", "-uc", preamble + '\necho "$tunnel_repair_deadline:$repair_total_deadline:$repair_cleanup_limit_us"', "bash", "--tunnels-only", "--restart-tunnels"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "270:300:270000000\n")

    def test_reset_exit_uses_reserved_cleanup_time(self):
        result = self.shell(["repair_exit"], '''
repair_stage=peer-lane-0; repair_started_seconds=$SECONDS; repair_committed=0
repair_cleanup_limit_us=150000000; repair_cleanup_spent_us=30000000
repair_exit_cleanup_limit_us=30000000
repair_cleanup_call() { echo "$repair_cleanup_limit_us"; }
repair_release_lock() { return 0; }
repair_exit 1
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "30000000\n")

    def test_normal_migration_can_use_work_budget_without_using_rollback_budget(self):
        result = self.shell(["repair_cleanup_remaining", "repair_cleanup_call", "repair_exit"], '''
repair_stage=peer-legacy; repair_started_seconds=$SECONDS; repair_committed=0
tunnel_repair_deadline=$((SECONDS+130)); repair_total_deadline=$((SECONDS+150))
repair_cleanup_limit_us=130000000; repair_cleanup_spent_us=30000000
repair_exit_cleanup_limit_us=30000000
repair_cleanup_active_us=0; repair_cleanup_completed=63
stop_group() { sleep 0.01; }
repair_cleanup_call stop_group || exit 9
repair_cleanup_all() {
  [ "$repair_cleanup_spent_us" = 0 ] && [ "$repair_cleanup_completed" = 0 ] || return 1
  [ "$repair_cleanup_limit_us" = 30000000 ] || return 1
  sleep 0.01
}
repair_release_lock() { return 0; }
repair_exit 7
''')
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("cleanup=passed migration_stops=64 rollback_stops=1", result.stderr)

    def test_rollback_respects_real_remaining_total_time(self):
        program = "\n".join(self.function(name) for name in ("repair_cleanup_remaining", "repair_cleanup_call", "repair_exit")) + '''
SECONDS=148; repair_started_seconds=0; repair_total_deadline=150; tunnel_repair_deadline=130
repair_stage=local-readiness; repair_committed=0
repair_cleanup_spent_us=30000000; repair_cleanup_completed=64; repair_cleanup_active_us=0
repair_cleanup_limit_us=130000000
repair_cleanup_all() {
  while repair_cleanup_remaining; do sleep 0.05; done
  return 1
}
repair_release_lock() { return 0; }
repair_exit 7
'''
        started = time.monotonic()
        result = subprocess.run(["bash", "-uc", program], text=True, capture_output=True, timeout=3)
        elapsed = time.monotonic() - started
        self.assertLessEqual(elapsed, 2.5)
        self.assertGreater(elapsed, 1)
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn("elapsed=150s", result.stderr)
        self.assertIn("migration_stops=64 rollback_stops=1", result.stderr)
        print(f"rollback with two seconds left: {elapsed:.3f}s, total clock150s")

    def test_non_tunnels_mode_keeps_bounded_cleanup(self):
        preamble = self.source.split("\nrepair_cleanup_remaining() {", 1)[0]
        result = subprocess.run(["bash", "-uc", preamble + '\necho "$repair_total_deadline:$repair_cleanup_limit_us"'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0:20000000\n")

    def test_api_loss_during_readiness_repairs_once_and_rechecks(self):
        for failed_check in ("listener", "supervisor"):
            result = self.shell(["adapter_tunnel_readiness"], f'''
peer_enabled=1; secondary_enabled=0; peer_api_port=18049; repaired=0; checks=0
repair_remaining() {{ return 0; }}
repair_sleep() {{ return 0; }}
sleep() {{ return 0; }}
china_tunnel_supervisors_ready() {{ return 0; }}
peer_tunnel_supervisors_ready() {{ return 0; }}
peer_tunnel_api_listener_owner() {{ checks=$((checks+1)); [ '{failed_check}' != listener ] || [ "$repaired" = 1 ]; }}
peer_tunnel_api_supervisor_ready() {{ [ '{failed_check}' != supervisor ] || [ "$repaired" = 1 ]; }}
peer_tunnel_connection() {{ repaired=$((repaired+1)); [ "$1" = api ]; }}
peer_remote_ports_ready() {{ return 0; }}
peer_tunnel_current_children_ready() {{ return 0; }}
adapter_tunnel_local_snapshot() {{ return 0; }}
adapter_tunnel_remote_snapshot() {{ return 0; }}
adapter_tunnel_stability_snapshot() {{ return 0; }}
adapter_tunnel_readiness || exit 1
echo "$repaired:$checks"
''')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "1:2\n")

    def test_unknown_api_owner_or_child_does_not_trigger_repair(self):
        for failed_check in ("listener", "supervisor"):
            result = self.shell(["adapter_tunnel_readiness"], f'''
peer_enabled=1; secondary_enabled=0; peer_api_port=18049
repair_remaining() {{ return 0; }}
china_tunnel_supervisors_ready() {{ return 0; }}
peer_tunnel_supervisors_ready() {{ return 0; }}
peer_tunnel_api_listener_owner() {{ [ '{failed_check}' != listener ] || return 2; }}
peer_tunnel_api_supervisor_ready() {{ return 2; }}
peer_tunnel_connection() {{ echo unexpected; }}
adapter_tunnel_readiness
''')
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertIn("no repair started", result.stderr)


if __name__ == "__main__":
    unittest.main()
