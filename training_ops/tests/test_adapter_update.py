import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops

adapter_server_path = (
    Path(__file__).resolve().parents[1] / "runtime" / "china" / "adapter_server.py"
)
adapter_server_spec = importlib.util.spec_from_file_location(
    "training_ops_adapter_server", adapter_server_path
)
assert adapter_server_spec is not None and adapter_server_spec.loader is not None
adapter_server = importlib.util.module_from_spec(adapter_server_spec)
adapter_server_spec.loader.exec_module(adapter_server)


class AdapterUpdateTest(unittest.TestCase):
    def test_start_inventory_deploys_nodes_config_to_participants(self):
        selected = {
            "trainer": "configs/general-agent/trainer.toml",
            "orchestrator": "configs/general-agent/orchestrator.toml",
        }
        with mock.patch.object(training_ops, "_check_host", side_effect=lambda host: host), mock.patch.object(
            training_ops, "mounted_files", return_value=[]
        ):
            inventory = dict(
                training_ops._start_dependency_inventory(selected, training_ops.trainer_hosts)
            )
        self.assertEqual(
            set(inventory["configs/nodes.toml"]),
            set(training_ops.trainer_hosts + training_ops.inference_hosts),
        )

    def test_inference_launcher_forwards_mtp_settings_before_docker(self):
        launcher_path = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "h200"
            / "start_inference.sh"
        )
        launcher = launcher_path.read_text(encoding="utf-8")
        start = launcher.index("<<'PY'\n") + len("<<'PY'\n")
        end = launcher.index("\nPY\n)", start)
        reader = launcher[start:end]
        nodes_path = Path(training_ops.repo_root) / "configs/nodes.toml"
        nodes = nodes_path.read_text(encoding="utf-8")

        def read_settings(host, config_text=nodes):
            with tempfile.NamedTemporaryFile("w", suffix=".toml") as stream:
                stream.write(config_text)
                stream.flush()
                return subprocess.run(
                    [sys.executable, "-", stream.name, host],
                    input=reader,
                    capture_output=True,
                    text=True,
                    check=False,
                )

        a = read_settings("h200a")
        b = read_settings("h200b")
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        a_extra = json.loads(a.stdout.splitlines()[-1])
        b_extra = json.loads(b.stdout.splitlines()[-1])
        self.assertEqual(
            a_extra["speculative_config"],
            {"method": "mtp", "num_speculative_tokens": 1},
        )
        self.assertEqual(
            b_extra["speculative_config"],
            {"method": "mtp", "num_speculative_tokens": 1},
        )
        self.assertEqual(a_extra["max_num_batched_tokens"], 32768)
        self.assertEqual(a_extra["compilation_config"], {"max_cudagraph_capture_size": 256})
        self.assertIn("runtime_output=$(python3", launcher)
        self.assertLess(launcher.index("runtime_output=$(python3"), launcher.index("install -d"))
        self.assertLess(launcher.index("runtime_output=$(python3"), launcher.index("docker update"))

        invalid_configs = (
            nodes.replace("mtp_tokens = 1\nports = [8000]", "ports = [8000]", 1),
            nodes.replace("mtp_tokens = 1", "mtp_tokens = true", 1),
            nodes.replace("mtp_tokens = 1", "mtp_tokens = -1", 1),
        )
        for config_text in invalid_configs:
            result = read_settings("h200a", config_text)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mtp_tokens", result.stderr)

    def test_config_uses_the_active_glm_run_config(self):
        config = training_ops._adapter_update_config("glm-perf-3")
        self.assertEqual(config["kind"], "glm52")
        self.assertEqual(config["engine_container"], "glm52-decensor-h200")
        self.assertEqual(config["source_url"], "http://127.0.0.1:18100/glm-perf-3")
        self.assertEqual(config["ports"], [8000])
        self.assertEqual(config["run_marker"], "/glm-perf-3/")
        self.assertEqual(config["service_marker"], "/run_glm-perf-3/")

    def test_adapter_server_launcher_uses_active_glm_settings(self):
        script = training_ops._adapter_server_launch_script(
            "glm-perf-3", training_ops._adapter_update_config("glm-perf-3")
        )
        self.assertIn("adapter-http", script)
        self.assertIn("RUN=glm-perf-3", script)
        self.assertIn("ADAPTER_PREDICTIVE_RUN=glm-perf-3", script)
        self.assertIn(
            "ADAPTER_PREDICTIVE_BROADCAST_PATH=glm-perf-3/run_glm-perf-3/broadcasts", script
        )
        self.assertIn("ADAPTER_PREDICTIVE_SOURCE_URL=http://127.0.0.1:18100/glm-perf-3", script)
        self.assertIn("node_id=82", script)
        self.assertNotIn("cluster.sh trainer", script)

    def test_adapter_server_repair_requires_reported_failure_unless_reload(self):
        before = training_ops.Status(trainer_up=True, adapter="200", orch_up=False)
        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "_adapter_update_config", return_value={"kind": "glm52"}
        ), mock.patch.object(training_ops, "status", return_value=before), mock.patch.object(
            training_ops, "deploy"
        ) as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "adapter server is healthy"):
                training_ops._repair_adapter_server()
        deploy.assert_not_called()

    def test_adapter_server_repair_does_not_start_trainer(self):
        before = training_ops.Status(trainer_up=True, adapter="000", orch_up=False)
        config = training_ops._adapter_update_config("glm-perf-3")
        ssh_scripts = []

        def ssh(_host, script, timeout=120):
            ssh_scripts.append(script)
            return 0, "adapter server ready on port 8100\n"

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "_adapter_update_config", return_value=config
        ), mock.patch.object(training_ops, "status", return_value=before), mock.patch.object(
            training_ops, "deploy", return_value=True
        ), mock.patch.object(training_ops, "_ssh", side_effect=ssh):
            result = training_ops._repair_adapter_server()

        self.assertIn("reloaded active GLM adapter server", result)
        self.assertEqual(len(ssh_scripts), 2)
        self.assertIn("adapter-http", ssh_scripts[0])
        self.assertNotIn("trainer >/dev/null", ssh_scripts[0])

    def test_glm_adapter_server_gets_predictive_upgrade_settings(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "china"
            / "cluster.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ADAPTER_PREDICTIVE_UPGRADE", launcher)
        self.assertIn("ADAPTER_PREDICTIVE_RUN", launcher)
        self.assertIn("ADAPTER_PREDICTIVE_BROADCAST_PATH", launcher)
        self.assertIn("ADAPTER_PREDICTIVE_SOURCE_URL", launcher)
        install = (
            'sudo install -d -o 1000 -g 1000 -m 0755 \\\n'
            '        "$broadcast_root/$predictive_run" \\\n'
            '        "$broadcast_root/$predictive_run/run_$predictive_run" \\\n'
            '        "$active_broadcasts"'
        )
        self.assertIn(install, launcher)
        self.assertLess(
            launcher.index(install),
            launcher.index('sudo docker run -d --name "$adapter_container"'),
        )
        self.assertIn('/broadcast/"$predictive_broadcast_path":rw', launcher)
        self.assertNotIn('/broadcast/"$predictive_run":rw', launcher)
        self.assertIn('--upgrade-path "$predictive_broadcast_path"', launcher)
        self.assertIn('/run_$predictive_run/broadcasts"', launcher)
        self.assertIn("prime_rl/utils/adapter_delta.py", launcher)

    def test_server_upgrades_only_the_exact_active_broadcast_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "glm-perf-3" / "run_glm-perf-3" / "broadcasts"
            target = active / "step_2"
            target.mkdir(parents=True)
            handler = adapter_server.RangeHandler.__new__(adapter_server.RangeHandler)
            handler.directory = temporary
            handler.upgrade_path = ("glm-perf-3", "run_glm-perf-3", "broadcasts")
            handler.source_base_url = (
                "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts"
            )
            handler.path = "/glm-perf-3/run_glm-perf-3/broadcasts/step_2/adapter_manifest.json"

            with mock.patch.object(adapter_server, "_upgrade_v1_step") as upgrade:
                path = handler._maybe_upgrade()
            self.assertEqual(path, target / "adapter_manifest.json")
            upgrade.assert_called_once_with(
                active,
                target,
                2,
                "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts",
            )
            self.assertEqual(
                adapter_server._predictive_step_url(handler.source_base_url, 2),
                "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts/step_2",
            )

            handler.path = "/glm-perf-3/step_2/adapter_manifest.json"
            with mock.patch.object(adapter_server, "_upgrade_v1_step") as upgrade:
                handler._maybe_upgrade()
            upgrade.assert_not_called()

            handler.path = "/glm-perf-3/run_glm-perf-3/broadcasts/step_2/adapter_manifest.json"
            with mock.patch.object(
                adapter_server,
                "_upgrade_v1_step",
                side_effect=adapter_server.UpgradeError("missing exact base"),
            ):
                with self.assertRaisesRegex(adapter_server.UpgradeError, "missing exact base"):
                    handler._maybe_upgrade()

    def test_server_accepts_delta_as_the_public_predictive_manifest_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            build = root / "build"
            target.mkdir()
            build.mkdir()
            chunk_name = "adapter_model.safetensors.transport-predictive-00000.bin"
            (target / "adapter_manifest.json").write_text("old", encoding="utf-8")
            (target / "adapter_model.safetensors.sha256").write_text("old", encoding="ascii")
            (build / chunk_name).write_bytes(b"delta")
            (build / "adapter_manifest.json").write_text("new", encoding="utf-8")
            (build / "adapter_model.safetensors.sha256").write_text("new", encoding="ascii")
            manifest = {
                "version": 2,
                "mode": "delta",
                "target_step": 2,
                "base_step": 1,
                "base2_step": 0,
                "chunks": [{"file": chunk_name, "wire_size": 5}],
            }

            adapter_server._install_predictive_manifest(target, build, manifest, 2)
            self.assertEqual((target / "adapter_manifest.json").read_text(), "new")
            with self.assertRaises(adapter_server.UpgradeError):
                adapter_server._install_predictive_manifest(
                    target,
                    build,
                    {**manifest, "mode": "predictive"},
                    2,
                )

    def test_cache_keeps_the_old_base_until_both_new_entries_validate(self):
        def write_entry(path, step, value):
            path.mkdir(parents=True)
            config = f"config-{value}".encode()
            weights = f"weights-{value}".encode()
            config_digest = hashlib.sha256(config).hexdigest()
            weights_digest = hashlib.sha256(weights).hexdigest()
            (path / "adapter_config.json").write_bytes(config)
            (path / "adapter_model.safetensors").write_bytes(weights)
            (path / "adapter_model.safetensors.sha256").write_text(
                f"{weights_digest}\n", encoding="ascii"
            )
            (path / ".identity.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "config_size": len(config),
                        "config_sha256": config_digest,
                        "weights_size": len(weights),
                        "weights_sha256": weights_digest,
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temporary:
            active = Path(temporary)
            cache = active / adapter_server.cache_name
            write_entry(cache / "step_0", 0, "zero")
            write_entry(cache / "step_1", 1, "one")
            pending = cache / ".pending-test"
            write_entry(pending / "step_2", 2, "two")
            write_entry(pending / "step_1", 1, "different")

            with self.assertRaisesRegex(adapter_server.UpgradeError, "identity differs"):
                adapter_server._promote_cache(active, pending, 2)
            self.assertTrue((cache / "step_0").is_dir())
            self.assertFalse((cache / "step_2").exists())

            bad = pending / "step_1"
            for child in bad.iterdir():
                child.unlink()
            bad.rmdir()
            write_entry(bad, 1, "one")
            adapter_server._promote_cache(active, pending, 2)
            self.assertTrue((cache / "step_0").is_dir())
            self.assertTrue((cache / "step_1").is_dir())
            self.assertTrue((cache / "step_2").is_dir())
            adapter_server._prune_cache(active, 2)
            self.assertFalse((cache / "step_0").exists())

    def test_stage_log_requires_a_transfer_timeout(self):
        good = "Failed to apply policy update 4\nReadTimeout /stage_lora_adapter"
        self.assertEqual(
            training_ops._validate_adapter_stage_log(good),
            "Failed to apply policy update 4",
        )
        current = (
            "Policy update 4 failed on attempt 3/3; stopping\n"
            "TimeoutError: adapter delta staging exceeded its time limit"
        )
        self.assertEqual(
            training_ops._validate_adapter_stage_log(current),
            "Policy update 4 failed on attempt 3/3; stopping",
        )
        with self.assertRaises(training_ops.Refused):
            training_ops._validate_adapter_stage_log(
                "Failed to apply policy update 4\nadapter request failed"
            )

    def test_stage_success_after_failure_refuses_repair(self):
        log = (
            "Failed to apply policy update 4\nReadTimeout stage_lora_adapter\n"
            "Staged policy v4 on every inference host"
        )
        with self.assertRaisesRegex(training_ops.Refused, "already succeeded"):
            training_ops._validate_adapter_stage_log(log)

    def test_terminal_stage_status_accepts_exact_timeout_and_manifest_500(self):
        config = training_ops._adapter_update_config("glm-perf-3")
        broadcast = training_ops.AdapterBroadcast(
            path="/data/glm52-rl/runs/glm-perf-3/run_glm-perf-3/broadcasts/step_6",
            step=6,
        )
        source = "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts/step_6"
        cache = (
            "/data/glm52-rl/lora-cache/"
            + hashlib.sha256(source.encode()).hexdigest()
        )
        timeout = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "elapsed_seconds": 298.297,
            "error": "TimeoutError: adapter delta staging exceeded its time limit",
        }
        manifest_500 = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "elapsed_seconds": 299.102,
            "error": "OSError: adapter manifest request failed with HTTP status 500",
        }
        results = {
            "h200a": (0, json.dumps(timeout)),
            "h200b": (0, json.dumps(manifest_500)),
        }
        with mock.patch.object(
            training_ops,
            "_latest_adapter_broadcast",
            return_value=broadcast,
        ), mock.patch.object(training_ops, "_fan", return_value=results):
            evidence = training_ops._adapter_stage_status_evidence("glm-perf-3", config, "")
        self.assertIn(
            "policy update 6 failed on h200a (TimeoutError: adapter delta staging exceeded its time limit), "
            "h200b (OSError: adapter manifest request failed with HTTP status 500)",
            evidence,
        )

    def test_terminal_stage_status_refuses_stale_unknown_and_success(self):
        config = training_ops._adapter_update_config("glm-perf-3")
        broadcast = training_ops.AdapterBroadcast(
            path="/data/glm52-rl/runs/glm-perf-3/run_glm-perf-3/broadcasts/step_6",
            step=6,
        )
        source = "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts/step_6"
        cache = (
            "/data/glm52-rl/lora-cache/"
            + hashlib.sha256(source.encode()).hexdigest()
        )
        ready = {
            "status": "ready",
            "source_url": source,
            "path": cache,
            "elapsed_seconds": 262.336,
        }
        failed = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "elapsed_seconds": 298.297,
            "error": "TimeoutError: adapter delta staging exceeded its time limit",
        }

        def check(records, pattern, log=""):
            with mock.patch.object(
                training_ops,
                "_latest_adapter_broadcast",
                return_value=broadcast,
            ), mock.patch.object(training_ops, "_fan", return_value=records):
                with self.assertRaisesRegex(training_ops.Refused, pattern):
                    training_ops._adapter_stage_status_evidence("glm-perf-3", config, log)

        stale = dict(failed, source_url=source + "-old")
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(stale))},
            "stale or mismatched",
        )
        unknown = dict(failed, status="staging")
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(unknown))},
            "not terminal",
        )
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(ready))},
            "succeeded on every",
        )
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(failed))},
            "already succeeded",
            log="Staged policy v6 on every inference host",
        )
        unrelated = dict(
            failed,
            error="OSError: adapter manifest request failed with HTTP status 404",
        )
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(unrelated))},
            "not an allowed transfer failure",
        )

    def test_stage_status_reader_rejects_symlinks(self):
        script = training_ops._adapter_stage_status_script(
            "/data/glm52-rl/lora-cache/.adapter-stage-status.json"
        )
        self.assertIn("os.lstat(path)", script)
        self.assertIn("stat.S_ISREG", script)
        self.assertIn("O_NOFOLLOW", script)

    def test_idle_script_requires_health_and_request_metric(self):
        script = training_ops._adapter_idle_script([8000])
        self.assertIn("/health", script)
        self.assertIn("/metrics", script)
        self.assertIn("vllm:num_requests_running", script)
        self.assertIn("!found || bad || total != 0", script)

    def test_engine_state_requires_exact_identity_and_reports_restart_need(self):
        script = training_ops._adapter_engine_state_script(
            "glm52-decensor-h200",
            [8000],
            "/glm52/",
            "/run_glm52/",
            "/data/glm52-rl/runs/glm52/run_glm52",
        )
        self.assertIn('name=$(sudo docker inspect', script)
        self.assertIn('configured inference container identity is not exact', script)
        self.assertIn('staging_process_matches', script)
        self.assertIn('adapter staging remains', script)
        self.assertIn('exit 10', script)

    def test_glm_restart_wait_is_configured_for_model_load(self):
        script = training_ops._restart_adapter_engine_script(
            "glm52-decensor-h200", 8000
        )
        self.assertIn("seq 1 900", script)
        with self.assertRaisesRegex(training_ops.Refused, "integer >= 600"):
            training_ops._restart_adapter_engine_script(
                "glm52-decensor-h200", 8000, startup_timeout_seconds=180
            )

    def test_orchestrator_state_script_requires_exact_stopped_container(self):
        script = training_ops._orchestrator_state_script(
            "glm52-decensor-orchestrator"
        )
        self.assertIn("docker inspect", script)
        self.assertIn('configured orchestrator container identity is not exact', script)
        self.assertIn("false) echo stopped", script)
        self.assertNotIn("docker start", script)
        self.assertNotIn("docker restart", script)

    def test_stopped_orchestrator_is_started_after_guarded_repair(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1
                ),
            ],
        )
        fan_results = iter(
            [
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
                {"h200a": (0, "stopped 0"), "h200b": (0, "stopped 0")},
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {
                    "h200a": (0, "restarted exact inference container"),
                    "h200b": (0, "restarted exact inference container"),
                },
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
            ]
        )
        ssh_scripts = []

        def ssh(_host, script, timeout=120):
            ssh_scripts.append(script)
            if "configured orchestrator container state is unknown" in script:
                return 0, "stopped\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(
            training_ops, "_adapter_stage_evidence", return_value="Failed to apply policy update 4"
        ), mock.patch.object(
            training_ops, "_fan", side_effect=lambda *_args, **_kwargs: next(fan_results)
        ), mock.patch.object(training_ops, "deploy", return_value=True) as deploy, mock.patch.object(
            training_ops, "_ssh", side_effect=ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_adapter_update()

        self.assertIn("recreated exact orchestrator", result)
        resume_script = ssh_scripts[-1]
        self.assertIn('sudo docker rm -f "$container"', resume_script)
        self.assertIn(
            "bash /data/glm52-rl/deployment/training_ops/runtime/h200/start_orchestrator.sh",
            resume_script,
        )
        self.assertIn("RUN=glm-perf-3", resume_script)
        self.assertIn("RUN_NAME=run_glm-perf-3", resume_script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/general-agent/orchestrator.toml",
            resume_script,
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", resume_script)
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", resume_script)
        self.assertNotIn("sudo docker start", resume_script)
        self.assertNotIn("sudo docker restart", resume_script)
        self.assertIn(
            mock.call(
                "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py",
                nodes=["h200a", "h200b"],
            ),
            deploy.call_args_list,
        )
        self.assertIn(
            mock.call(
                "prime-rl/src/prime_rl/inference/relay.py",
                nodes=["h200a"],
            ),
            deploy.call_args_list,
        )
        self.assertIn(
            mock.call(
                "prime-rl/src/prime_rl/utils/checkpoint_delta.py",
                nodes=["h200a", "h200b"],
            ),
            deploy.call_args_list,
        )
        self.assertIn(
            mock.call(
                "training_ops/runtime/h200/start_orchestrator.sh",
                nodes=["h200a"],
            ),
            deploy.call_args_list,
        )
        self.assertEqual(deploy.call_count, 6)

    def test_stopped_status_refuses_if_container_is_running(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                )
            ],
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "_ssh", return_value=(0, "running\n")), mock.patch.object(
            training_ops, "_adapter_stage_evidence"
        ) as evidence, mock.patch.object(training_ops, "deploy") as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "stopped or absent exact orchestrator"):
                training_ops._repair_adapter_update()
        evidence.assert_not_called()
        deploy.assert_not_called()

    def test_absent_orchestrator_is_recreated_after_inference_reload(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1
                ),
            ],
        )
        fan_results = iter(
            [
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
                {"h200a": (0, "stopped 0"), "h200b": (0, "stopped 0")},
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {
                    "h200a": (0, "restarted exact inference container"),
                    "h200b": (0, "restarted exact inference container"),
                },
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
            ]
        )
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=120):
            ssh_scripts.append(script)
            if "configured orchestrator container state is unknown" in script:
                return 0, "absent\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(
            training_ops, "_adapter_stage_evidence", return_value="Failed to apply policy update 4"
        ), mock.patch.object(
            training_ops, "_fan", side_effect=lambda *_args, **_kwargs: next(fan_results)
        ), mock.patch.object(training_ops, "deploy", return_value=True), mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_adapter_update()

        self.assertIn("recreated exact orchestrator", result)
        self.assertIn("[ absent = absent ]", ssh_scripts[-1])
        self.assertIn('[ -z "$name" ]', ssh_scripts[-1])

    def test_repair_refuses_while_trainer_is_busy(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=80)],
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "_ssh") as ssh, mock.patch.object(
            training_ops, "_adapter_stage_evidence"
        ) as evidence:
            with self.assertRaisesRegex(training_ops.Refused, "trainer to be waiting"):
                training_ops._repair_adapter_update()
        ssh.assert_not_called()
        evidence.assert_not_called()

    def test_engine_repair_uses_active_run_config(self):
        calls = []

        def ssh(_host, script, timeout=120):
            calls.append(script)
            return 0, "issued"

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "_ssh", side_effect=ssh
        ):
            result = training_ops._repair_engine("h200a")

        self.assertEqual(result, "issued")
        self.assertTrue(any("glm52-decensor-h200" in call for call in calls))
        self.assertTrue(any("start_inference.sh" in call for call in calls))
        self.assertTrue(any("ENGINE_CONTAINER=glm52-decensor-h200" in call for call in calls))

    def test_healthy_engines_are_reloaded_to_import_receiver_changes(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=True,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1
                ),
            ],
        )
        fan_results = iter(
            [
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
                {"h200a": (0, "stopped 0"), "h200b": (0, "stopped 0")},
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {
                    "h200a": (0, "restarted exact inference container"),
                    "h200b": (0, "restarted exact inference container"),
                },
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
            ]
        )

        fan_call = 0

        def fan(hosts, function, **_kwargs):
            nonlocal fan_call
            result = next(fan_results)
            if fan_call == 3:
                for host in hosts:
                    function(host)
            fan_call += 1
            return result

        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(
            training_ops, "_adapter_stage_evidence", return_value="Failed to apply policy update 4"
        ), mock.patch.object(
            training_ops, "_fan", side_effect=fan
        ), mock.patch.object(training_ops, "deploy", return_value=True) as deploy, mock.patch.object(
            training_ops, "_ssh", return_value=(0, "")
        ), mock.patch.object(
            training_ops, "_restart_adapter_engine_script"
        ) as restart_script:
            result = training_ops._repair_adapter_update()

        self.assertIn("reloaded exact GLM inference containers", result)
        self.assertEqual(restart_script.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in restart_script.call_args_list},
            {"glm52-decensor-h200"},
        )
        self.assertEqual(deploy.call_count, 6)

    def test_prefetch_script_only_signals_exact_curl_or_timeout(self):
        script = training_ops._adapter_prefetch_stop_script(
            "/glm-perf-3/", "/run_glm-perf-3/", "/data/glm52-rl/runs/glm-perf-3/run_glm-perf-3"
        )
        self.assertIn("case \"${exe##*/}\" in curl|timeout", script)
        self.assertIn("stage_lora_adapter", script)
        self.assertIn("SOURCE_RUN_DIR=", script)
        self.assertNotIn("pkill", script)
        self.assertNotIn("kill -- -", script)

    def test_repair_refuses_before_any_mutation_when_trainer_is_down(self):
        unhealthy = training_ops.Status(trainer_up=False, orch_up=True)
        with mock.patch.object(training_ops, "current_run", return_value="glm-perf-3"), mock.patch.object(
            training_ops, "status", return_value=unhealthy
        ), mock.patch.object(training_ops, "_adapter_stage_evidence") as evidence, mock.patch.object(
            training_ops, "deploy"
        ) as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "healthy trainer"):
                training_ops._repair_adapter_update()
        evidence.assert_not_called()
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
