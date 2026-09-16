import ast
import hashlib
import io
import importlib.util
import json
import re
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


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    shared_path = training_ops.repo_root / "prime-rl/src/prime_rl/utils/checkpoint_delta.py"
    shared_constant = next(
        assignment.value
        for assignment in ast.parse(shared_path.read_text(encoding="utf-8")).body
        if isinstance(assignment, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "adapter_stage_timeout_seconds"
                for target in assignment.targets)
    )
    namespace = {"adapter_stage_timeout_seconds": ast.literal_eval(shared_constant)}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class AdapterUpdateTest(unittest.TestCase):
    def test_adapter_stage_policy_matches_all_layers(self):
        repository = Path(__file__).resolve().parents[2]
        receiver_limit = _load_function(
            repository / "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py",
            "adapter_stage_limit_seconds",
        )
        client_limit = _load_function(
            repository / "prime-rl/src/prime_rl/utils/client.py",
            "adapter_stage_limit_seconds",
        )
        layers = (
            ("training ops", training_ops._adapter_stage_limit_seconds, training_ops.Refused),
            ("receiver", receiver_limit, OSError),
            ("client", client_limit, RuntimeError),
        )
        for step, expected in ((1, 600), (2, 600), (3, 600), (4, 600), (99, 600)):
            with self.subTest(step=step):
                self.assertEqual(
                    [int(limit(step)) for _, limit, _ in layers],
                    [expected, expected, expected],
                )
        for name, limit, error_type in layers:
            with self.subTest(layer=name):
                with self.assertRaises(error_type):
                    limit(0)
        receiver_source = (
            repository / "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py"
        ).read_text(encoding="utf-8")
        client_source = (
            repository / "prime-rl/src/prime_rl/utils/client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("full_adapter_stage_timeout_s = 1800.0", receiver_source)
        self.assertIn("LORA_STAGE_TOTAL_TIMEOUT_S = 1800.0", client_source)

    def test_adapter_server_uses_short_bounded_writes(self):
        payload = b"x" * (adapter_server.transfer_write_bytes * 3 + 17)
        writes = []

        class Output:
            def write(self, value):
                if len(value) > adapter_server.transfer_write_bytes:
                    raise AssertionError("adapter server write exceeded its bound")
                writes.append(value)

        handler = adapter_server.RangeHandler.__new__(adapter_server.RangeHandler)
        handler.remaining = len(payload)
        handler.copyfile(io.BytesIO(payload), Output())

        self.assertEqual(b"".join(writes), payload)
        self.assertEqual(handler.remaining, 0)
        self.assertEqual(adapter_server.connection_timeout_seconds, 120)

    def test_adapter_server_closes_on_disconnected_output(self):
        class Output:
            def write(self, _value):
                raise ConnectionAbortedError("peer closed")

        payload = b"x" * adapter_server.transfer_write_bytes
        handler = adapter_server.RangeHandler.__new__(adapter_server.RangeHandler)
        handler.remaining = len(payload)
        source = io.BytesIO(payload)
        handler.copyfile(source, Output())

        self.assertTrue(handler.close_connection)
        self.assertEqual(handler.remaining, len(payload))
        self.assertTrue(source.closed)

    def test_adapter_server_propagates_source_read_failure(self):
        class Source:
            closed = False

            def read(self, _size):
                raise OSError("source read failed")

            def close(self):
                self.closed = True

        handler = adapter_server.RangeHandler.__new__(adapter_server.RangeHandler)
        handler.remaining = adapter_server.transfer_write_bytes
        source = Source()
        with self.assertRaisesRegex(OSError, "source read failed"):
            handler.copyfile(source, io.BytesIO())

        self.assertTrue(handler.close_connection)
        self.assertTrue(source.closed)

    def test_start_inventory_deploys_nodes_config_to_participants(self):
        selected = {
            "trainer": "configs/decensor/trainer.toml",
            "orchestrator": "configs/decensor/orchestrator.toml",
            "inference": "configs/decensor/inference.toml",
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
        self.assertEqual(
            set(inventory["configs/decensor/inference.toml"]),
            set(training_ops.inference_hosts),
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
        expected_source_ports = [
            str(port)
            for offset in range(2)
            for port in range(18100 + offset, 18228, 2)
        ]
        for result in (a, b):
            source_ports = result.stdout.splitlines()[0].split(",")
            self.assertEqual(source_ports, expected_source_ports)
            self.assertEqual(len(source_ports), 128)
            self.assertEqual(len(set(source_ports)), 128)
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
        self.assertIn(
            "inference_config=${inference_config:?set inference_config to the selected run config}",
            launcher,
        )
        self.assertNotIn("configs/general-agent/inference.toml", launcher)
        self.assertIn('@ "$inference_config"', launcher)
        self.assertIn("download_streams=${PRIME_LORA_DOWNLOAD_STREAMS-112}", launcher)
        self.assertIn('-e PRIME_LORA_DOWNLOAD_STREAMS="$download_streams"', launcher)
        self.assertIn("-e PRIME_LORA_LANE_TIMEOUT_S=90", launcher)
        self.assertIn("-e PRIME_LORA_CHUNK_TIMEOUT_S=90", launcher)
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

    def test_inference_launcher_validates_download_stream_override(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "h200"
            / "start_inference.sh"
        ).read_text(encoding="utf-8")
        start = launcher.index("download_streams=${PRIME_LORA_DOWNLOAD_STREAMS-112}")
        end = launcher.index("runtime_output=$(python3", start)
        validation = launcher[start:end]

        for value in ("32", "48", "96"):
            result = subprocess.run(
                ["bash", "-c", f"PRIME_LORA_DOWNLOAD_STREAMS={value}\n{validation}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        default_result = subprocess.run(
            ["bash", "-c", f"unset PRIME_LORA_DOWNLOAD_STREAMS\n{validation}"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(default_result.returncode, 0, default_result.stderr)

        for value in ("", "0", "129", "999999999999999999999", "not-an-integer"):
            assignment = (
                "PRIME_LORA_DOWNLOAD_STREAMS=''\n"
                if value == ""
                else f"PRIME_LORA_DOWNLOAD_STREAMS={value}\n"
            )
            result = subprocess.run(
                ["bash", "-c", f"{assignment}{validation}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, value)

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

    def test_trainer_mounts_all_config_modules_read_only(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "china"
            / "cluster.sh"
        ).read_text(encoding="utf-8")
        start = launcher.index("trainer_docker_args=(")
        end = launcher.index("    trainer_filtered_args=()", start)
        mounts = [
            line.strip()
            for line in launcher[start:end].splitlines()
            if "prime-rl-configs/src/prime_rl/configs/" in line
        ]
        expected = [
            (
                f'-v "$deployment/prime-rl/packages/prime-rl-configs/src/'
                f'prime_rl/configs/{module}":/app/packages/prime-rl-configs/src/'
                f'prime_rl/configs/{module}:ro \\'
            )
            for module in ("shared.py", "trainer.py", "orchestrator.py")
        ]
        self.assertEqual(mounts, expected)

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
                "chunks": [
                    {
                        "file": chunk_name,
                        "wire_size": 5,
                        "wire_sha256": hashlib.sha256(b"delta").hexdigest(),
                    }
                ],
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

    def test_transport_failure_classifier_is_exact_and_fail_closed(self):
        accepted = (
            "ConnectionResetError: connection reset by peer",
            "ConnectionResetError: [Errno 104] connection reset by peer",
            "ConnectionRefusedError: connection refused",
            "RemoteDisconnected: remote end closed connection without response",
            "BrokenPipeError: [Errno 32] broken pipe",
            "TimeoutError: adapter delta staging exceeded its time limit",
            "TimeoutError: adapter persistent request exceeded its stage time limit",
            "TimeoutError: adapter download exceeded its time limit",
            "OSError: adapter manifest request failed with HTTP status 500",
            "OSError: adapter persistent request failed",
            "adapter persistent request failed",
            "adapter network request failed",
            "_PersistentTransportError: adapter network read failed",
            "adapter response ended early",
            "OSError: adapter response read timeout setup failed",
            "adapter manifest is missing; transfer mode is unknown",
            "_PersistentTransportError: adapter response ended during read",
            "OSError: adapter response body ended early",
            "incomplete adapter transport chunk",
            "_PersistentTransportError: none of 8 adapter source lanes responded",
            "OSError: adapter persistent download retries exhausted",
            "_PersistentTransportError: adapter persistent download retries exhausted",
        )
        rejected = (
            "OSError: adapter persistent request failed while reading the cache",
            "OSError: adapter response body ended early; checksum differs",
            "OSError: adapter checksum mismatch",
            "OSError: adapter digest differs",
            "OSError: adapter identity is invalid",
            "OSError: adapter semantic validation failed",
            "OSError: adapter cache lock is busy",
            "OSError: adapter manifest race",
            "OSError: incomplete adapter file",
            "OSError: adapter invalid proof",
            "OSError: adapter response body is too short",
            "OSError: adapter manifest request failed with HTTP status 429",
            "OSError: adapter manifest request failed with HTTP status 503",
            "OSError: adapter manifest request failed with HTTP status 400",
            "OSError: adapter manifest request failed with HTTP status 401",
            "OSError: adapter manifest request failed with HTTP status 403",
            "OSError: adapter source returned HTTP status 503",
            "OSError: adapter source retry budget exhausted",
            "stage timeout: adapter delta staging exceeded its time limit",
            "adapter request failed; ConnectionResetError: connection reset by peer",
            "adapter manifest request failed with HTTP status 500",
            "OSError: adapter manifest request failed with HTTP status 500; cache lock held",
        )
        for error in accepted:
            with self.subTest(accepted=error):
                self.assertIsNotNone(training_ops._adapter_transport_failure_reason(error))
        for error in rejected:
            with self.subTest(rejected=error):
                self.assertIsNone(training_ops._adapter_transport_failure_reason(error))

    def test_stage_evidence_requires_structured_status_over_misleading_logs(self):
        config = training_ops._adapter_update_config("glm-perf-3")
        log = (
            "Policy update 4 failed on attempt 2/2; stopping\n"
            "_PersistentTransportError: adapter response ended during read\n"
        )
        status_refusals = (
            "cannot read exact adapter stage status on h200a",
            "adapter stage status on h200a is stale or mismatched",
            "adapter stage failure on h200a is not an allowed transfer failure",
        )
        for refusal in status_refusals:
            with self.subTest(refusal=refusal), mock.patch.object(
                training_ops, "_ssh", return_value=(0, log)
            ), mock.patch.object(
                training_ops,
                "_adapter_stage_status_evidence",
                side_effect=training_ops.Refused(refusal),
            ) as status_evidence:
                with self.assertRaisesRegex(training_ops.Refused, re.escape(refusal)):
                    training_ops._adapter_stage_evidence("glm-perf-3", config)
                status_evidence.assert_called_once_with("glm-perf-3", config, log)

    def test_terminal_stage_status_accepts_exact_transport_errors(self):
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
            "policy_step": 6,
            "stage_timeout_seconds": 600,
            "acceptance_target_seconds": 300,
            "elapsed_seconds": 298.297,
            "error": "TimeoutError: adapter persistent request exceeded its stage time limit",
        }
        manifest_500 = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "policy_step": 6,
            "stage_timeout_seconds": 600,
            "acceptance_target_seconds": 300,
            "elapsed_seconds": 299.102,
            "error": "_PersistentTransportError: adapter persistent download retries exhausted",
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
            "policy update 6 failed on h200a (TimeoutError: adapter persistent request exceeded its stage time limit), "
            "h200b (_PersistentTransportError: adapter persistent download retries exhausted)",
            evidence,
        )

    def test_step_one_terminal_status_uses_600_limit_and_reports_target_overrun(self):
        config = training_ops._adapter_update_config("glm-perf-3")
        broadcast = training_ops.AdapterBroadcast(
            path="/data/glm52-rl/runs/glm-perf-3/run_glm-perf-3/broadcasts/step_1",
            step=1,
        )
        source = "http://127.0.0.1:18100/glm-perf-3/run_glm-perf-3/broadcasts/step_1"
        cache = "/data/glm52-rl/lora-cache/" + hashlib.sha256(source.encode()).hexdigest()
        failed = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "policy_step": 1,
            "stage_timeout_seconds": 600,
            "acceptance_target_seconds": 300,
            "elapsed_seconds": 600.0,
            "error": "_PersistentTransportError: adapter persistent request failed",
        }
        with mock.patch.object(
            training_ops,
            "_latest_adapter_broadcast",
            return_value=broadcast,
        ), mock.patch.object(
            training_ops,
            "_fan",
            return_value={"h200a": (0, json.dumps(failed)), "h200b": (0, json.dumps(failed))},
        ):
            evidence = training_ops._adapter_stage_status_evidence("glm-perf-3", config, "")
        self.assertIn("operational limit 600s", evidence)
        self.assertIn("300s acceptance target exceeded on h200a, h200b", evidence)

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
            "policy_step": 6,
            "stage_timeout_seconds": 600,
            "acceptance_target_seconds": 300,
            "elapsed_seconds": 262.336,
        }
        failed = {
            "status": "failed",
            "source_url": source,
            "path": cache,
            "policy_step": 6,
            "stage_timeout_seconds": 600,
            "acceptance_target_seconds": 300,
            "elapsed_seconds": 298.297,
            "error": "_PersistentTransportError: adapter persistent request failed",
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
        wrong_limit = dict(failed, stage_timeout_seconds=960)
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(wrong_limit))},
            "wrong operational limit",
        )
        overdue = dict(failed, elapsed_seconds=600.1)
        check(
            {"h200a": (0, json.dumps(ready)), "h200b": (0, json.dumps(overdue))},
            "duration on h200b is invalid",
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

    def test_orchestrator_control_wait_uses_engine_startup_budget(self):
        source_path = Path(training_ops.__file__)
        source = source_path.read_text(encoding="utf-8")
        start = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "start"
        )
        wait = next(
            node
            for node in ast.walk(start)
            if isinstance(node, ast.FunctionDef)
            and node.name == "wait_for_orchestrator_control"
        )
        wait_source = ast.get_source_segment(source, wait)
        self.assertIsNotNone(wait_source)
        assert wait_source is not None
        self.assertIn(
            "seq 1 {adapter_update_engine_startup_timeout_seconds}",
            wait_source,
        )
        self.assertNotIn("seq 1 60", wait_source)
        self.assertIn(
            "timeout=adapter_update_engine_startup_timeout_seconds + 30",
            wait_source,
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
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
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
                "prime-rl/src/prime_rl/inference/vllm/server.py",
                nodes=["h200a", "h200b"],
            ),
            deploy.call_args_list,
        )
        for path in (
            "prime-rl/src/prime_rl/orchestrator/orchestrator.py",
            "prime-rl/src/prime_rl/orchestrator/watcher.py",
        ):
            self.assertIn(
                mock.call(path, nodes=["h200a"]),
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
        self.assertEqual(deploy.call_count, 10)

    def test_stopped_status_refuses_if_container_is_running(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
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
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
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
        self.assertTrue(
            any(
                "inference_config=/data/glm52-rl/deployment/configs/general-agent/inference.toml"
                in call
                for call in calls
            )
        )

    def test_watchdog_cleanup_uses_engine_identity_not_script_path(self):
        script = training_ops._engine_watchdog_identity_script(
            "h200a",
            "glm52-decensor-h200",
            "/data/glm52-rl/deployment/training_ops/runtime/h200/engine_watchdog.sh",
            "/data/glm52-rl/deployment/training_ops/runtime/h200/start_inference.sh",
            "/models/glm52-fp8",
            600,
            ensure_running=True,
        )

        self.assertIn(
            r"bash\ */engine_watchdog.sh|/bin/bash\ */engine_watchdog.sh",
            script,
        )
        self.assertIn("grep -Fxq ENGINE_HOST=h200a", script)
        self.assertIn("grep -Fxq ENGINE_CONTAINER=glm52-decensor-h200", script)
        self.assertIn("STARTUP_GRACE=600", script)
        self.assertIn(
            "START=/data/glm52-rl/deployment/training_ops/runtime/h200/start_inference.sh",
            script,
        )
        self.assertIn("canonical engine watchdog already running", script)
        self.assertIn("kill -TERM -- \"-$pid\"", script)
        self.assertNotIn("pkill", script)
        self.assertNotIn("pgrep", script)

    def test_supervisor_repair_reconciles_stale_watchdogs(self):
        calls = []

        def ssh(host, script, timeout=120):
            calls.append((host, script, timeout))
            return 0, "reloaded exact engine watchdog(s) without restarting an engine\n"

        with mock.patch.object(
            training_ops,
            "current_run",
            side_effect=["glm-decensor-difficulty-8", "glm-decensor-difficulty-6"],
        ) as current_run, mock.patch.object(
            training_ops,
            "_config_for_run",
            return_value={"inference": "configs/current/inference.toml"},
        ) as config_for_run, mock.patch.object(
            training_ops, "deploy", return_value=True
        ), mock.patch.object(
            training_ops, "_ssh", side_effect=ssh
        ), mock.patch("builtins.print") as printer:
            result = training_ops._repair_supervisors()

        current_run.assert_called_once_with()
        config_for_run.assert_called_once_with("glm-decensor-difficulty-8")
        printer.assert_called_once_with(result)
        self.assertIn("h200a: reloaded exact engine watchdog", result)
        self.assertEqual({host for host, _, _ in calls}, {"h200a", "h200b"})
        self.assertEqual(len(calls), 4)
        for host in ("h200a", "h200b"):
            host_calls = [
                (script, timeout)
                for called_host, script, timeout in calls
                if called_host == host
            ]
            self.assertEqual(len(host_calls), 2)
            self.assertTrue(all(timeout == 180 for _, timeout in host_calls))
            cleanup_script, cleanup_timeout = host_calls[0]
            repair_script, repair_timeout = host_calls[1]
            self.assertEqual(cleanup_timeout, 180)
            self.assertEqual(repair_timeout, 180)
            self.assertIn(f"grep -Fxq ENGINE_HOST={host}", cleanup_script)
            self.assertIn("grep -Fxq ENGINE_CONTAINER=glm52-decensor-h200", cleanup_script)
            self.assertIn('for pid in "${matches[@]}"; do stop_watchdog "$pid"; done', cleanup_script)
            self.assertNotIn("WATCHDOG_IDENTITY=engine-watchdog:", cleanup_script)
            self.assertIn("PORT=8000", repair_script)
            self.assertIn(
                f"WATCHDOG_IDENTITY=engine-watchdog:{host}:glm-decensor-difficulty-8:8000",
                repair_script,
            )
            self.assertIn(f"env ENGINE_HOST={host}", repair_script)
            self.assertIn('nohup setsid bash "$watchdog_script"', repair_script)
            self.assertIn("stop_watchdog", repair_script)
            self.assertNotIn("docker restart", "\n".join(script for script, _ in host_calls))
            self.assertNotIn("docker rm -f", "\n".join(script for script, _ in host_calls))

    def test_supervisor_repair_rejects_different_config(self):
        with mock.patch.object(training_ops, "current_run", return_value="glm-decensor-difficulty-8"), mock.patch.object(
            training_ops,
            "_config_dir",
            return_value={"inference": "configs/general-agent/inference.toml"},
        ), mock.patch.object(
            training_ops,
            "_config_for_run",
            return_value={"inference": "configs/current/inference.toml"},
        ), mock.patch.object(training_ops, "_ssh") as ssh:
            with self.assertRaisesRegex(training_ops.Refused, "does not match active run"):
                training_ops._repair_supervisors("configs/general-agent")
        ssh.assert_not_called()

    def test_healthy_engines_are_reloaded_to_import_receiver_changes(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=True,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1,
                    container=True, api_listener=True, watchdog=True
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
        self.assertEqual(deploy.call_count, 10)

    def test_prefetch_script_only_signals_exact_curl_or_timeout(self):
        script = training_ops._adapter_prefetch_stop_script(
            "/glm-perf-3/", "/run_glm-perf-3/", "/data/glm52-rl/runs/glm-perf-3/run_glm-perf-3"
        )
        self.assertIn("case \"${exe##*/}\" in curl|timeout", script)
        self.assertIn("stage_lora_adapter", script)
        self.assertIn("SOURCE_RUN_DIR=", script)
        self.assertNotIn("pkill", script)
        self.assertNotIn("kill -- -", script)

    def test_engine_watchdog_defers_each_new_container_load(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "h200"
            / "engine_watchdog.sh"
        ).read_text()
        self.assertIn("{{.Id}} {{.State.StartedAt}}", script)
        self.assertIn("engine container is loading; probe deferred", script)

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
