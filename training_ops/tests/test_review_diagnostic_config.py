import copy
import hashlib
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


root = Path(__file__).resolve().parents[2]
diagnostic = root / "configs/trinity-decensor-review-diagnostic"
production = root / "configs/trinity-decensor-experimental-fixed"


def load_config(directory: Path, filename: str) -> dict:
    with (directory / filename).open("rb") as stream:
        return tomllib.load(stream)


def without_diagnostic_identity(kind: str, config: dict) -> dict:
    result = copy.deepcopy(config)
    if kind == "trainer":
        result.pop("output_dir")
        result.pop("max_steps")
        result["swanlab"].pop("project")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        for tag in (
            "50-step",
            "batch32",
            "review-diagnostic",
            "diag-d29a5958",
        ):
            if tag in result["swanlab"]["tags"]:
                result["swanlab"]["tags"].remove(tag)
        result["file_monitor"].pop("filename")
    elif kind == "orchestrator":
        result.pop("output_dir")
        result.pop("max_steps")
        result.pop("batch_size")
        result.pop("max_inflight_rollouts")
        result["swanlab"].pop("project")
        result["swanlab"].pop("name")
        result["swanlab"].pop("group")
        for tag in (
            "50-step",
            "batch32",
            "review-diagnostic",
            "diag-d29a5958",
        ):
            if tag in result["swanlab"]["tags"]:
                result["swanlab"]["tags"].remove(tag)
        result["model"]["lora"].pop("name")
        result["weight_broadcast"].pop("source_url")
        result["weight_broadcast"].pop("source_dir")
        result["train"]["env"][0]["env"]["taskset"]["id"] = "decensor-experimental-v1"
    return result


class ReviewDiagnosticConfigTest(unittest.TestCase):
    def test_profile_is_parseable_and_resolves_as_one_unique_run(self):
        selected = training_ops._config_dir(
            "configs/trinity-decensor-review-diagnostic",
            require_orchestrator=True,
            require_inference=True,
        )
        self.assertEqual(
            training_ops._selected_run_name(selected),
            "trinity-decensor-experimental-fixed-diag-d29a5958",
        )
        self.assertEqual(
            training_ops._config_for_run(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                config_dir="configs/trinity-decensor-review-diagnostic",
            ),
            selected,
        )
        for filename in ("trainer.toml", "orchestrator.toml", "inference.toml"):
            self.assertIsInstance(load_config(diagnostic, filename), dict)

    def test_profile_keeps_training_contract_and_uses_batch32(self):
        diagnostic_trainer = load_config(diagnostic, "trainer.toml")
        diagnostic_orchestrator = load_config(diagnostic, "orchestrator.toml")
        diagnostic_inference = load_config(diagnostic, "inference.toml")
        production_trainer = load_config(production, "trainer.toml")
        production_orchestrator = load_config(production, "orchestrator.toml")
        production_inference = load_config(production, "inference.toml")

        self.assertEqual(
            without_diagnostic_identity("trainer", diagnostic_trainer),
            without_diagnostic_identity("trainer", production_trainer),
        )
        self.assertEqual(
            without_diagnostic_identity("orchestrator", diagnostic_orchestrator),
            without_diagnostic_identity("orchestrator", production_orchestrator),
        )
        self.assertEqual(
            {key: value for key, value in diagnostic_inference.items() if key != "env_vars"},
            {key: value for key, value in production_inference.items() if key != "env_vars"},
        )

        self.assertEqual(diagnostic_trainer["max_steps"], 1)
        self.assertEqual(diagnostic_orchestrator["max_steps"], 1)
        self.assertEqual(diagnostic_orchestrator["batch_size"], 32)
        self.assertEqual(diagnostic_orchestrator["group_size"], 16)
        self.assertEqual(diagnostic_orchestrator["max_inflight_rollouts"], 32)
        self.assertEqual(diagnostic_orchestrator["pool_size"], 4)
        self.assertEqual(
            diagnostic_orchestrator["train"]["env"][0]["env"]["taskset"]["id"],
            "decensor-experimental-v1",
        )
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["pool"]["num_workers"], 16)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["taskset"]["task"]["judge"]["max_concurrent_requests"], 48)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["taskset"]["task"]["user"]["max_concurrent_requests"], 48)
        self.assertEqual(diagnostic_orchestrator["optim"]["lr"], 8e-6)
        self.assertEqual(diagnostic_trainer["optim"]["lr"], 8e-6)
        self.assertEqual(diagnostic_trainer["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(diagnostic_trainer["model"]["seq_len"], 32768)
        self.assertEqual(diagnostic_orchestrator["seq_len"], 32768)
        self.assertEqual(diagnostic_orchestrator["train"]["sampling"]["max_completion_tokens"], 16384)
        self.assertEqual(diagnostic_orchestrator["train"]["env"][0]["env"]["agent"]["max_output_tokens"], 16384)
        self.assertEqual(diagnostic_trainer["model"]["quantization"]["type"], "fp8")
        self.assertFalse(diagnostic_trainer["model"]["quantization"]["dequantize_checkpoint"])

    def test_all_diagnostic_identities_are_unique_and_production_is_untouched(self):
        diagnostic_trainer = load_config(diagnostic, "trainer.toml")
        diagnostic_orchestrator = load_config(diagnostic, "orchestrator.toml")
        diagnostic_inference = load_config(diagnostic, "inference.toml")
        production_trainer = load_config(production, "trainer.toml")
        production_orchestrator = load_config(production, "orchestrator.toml")
        production_inference = load_config(production, "inference.toml")

        self.assertNotEqual(diagnostic_trainer["output_dir"], production_trainer["output_dir"])
        self.assertNotEqual(diagnostic_orchestrator["output_dir"], production_orchestrator["output_dir"])
        self.assertNotEqual(diagnostic_trainer["swanlab"]["name"], production_trainer["swanlab"]["name"])
        self.assertNotEqual(diagnostic_orchestrator["swanlab"]["group"], production_orchestrator["swanlab"]["group"])
        self.assertNotEqual(diagnostic_orchestrator["model"]["lora"]["name"], production_orchestrator["model"]["lora"]["name"])
        self.assertNotEqual(diagnostic_orchestrator["weight_broadcast"]["source_url"], production_orchestrator["weight_broadcast"]["source_url"])
        self.assertEqual(
            diagnostic_inference["env_vars"]["PRIME_LORA_CACHE_DIR"],
            production_inference["env_vars"]["PRIME_LORA_CACHE_DIR"],
        )
        self.assertEqual(diagnostic_inference["max_loras"], 1)
        self.assertEqual(diagnostic_inference["max_cpu_loras"], 2)
        self.assertIn(
            "production receipt",
            (diagnostic / "inference.toml").read_text(encoding="utf-8"),
        )
        self.assertEqual(diagnostic_orchestrator["ckpt"]["resume_step"], -1)
        self.assertTrue(diagnostic_orchestrator["ckpt"]["skip_progress"])
        self.assertTrue(diagnostic_orchestrator["output_dir"].endswith("/run_trinity-decensor-experimental-fixed-diag-d29a5958"))
        self.assertEqual(
            diagnostic_orchestrator["weight_broadcast"]["source_dir"],
            diagnostic_trainer["output_dir"],
        )
        self.assertIn("batch32", diagnostic_orchestrator["swanlab"]["tags"])
        self.assertIn("review-diagnostic", diagnostic_orchestrator["swanlab"]["tags"])
        for filename in ("trainer.toml", "orchestrator.toml"):
            config = load_config(diagnostic, filename)
            tags = config["swanlab"]["tags"]
            for tag in tags:
                self.assertIsInstance(tag, str)
                self.assertTrue(tag)
                self.assertLessEqual(len(tag), 20)

        for path in diagnostic.glob("*.toml"):
            self.assertNotIn("n79", path.read_text(encoding="utf-8").lower())
        self.assertEqual(production_trainer["output_dir"], "/data/glm52-rl/runs/trinity-decensor-experimental-fixed")
        self.assertEqual(production_orchestrator["batch_size"], 1024)
        self.assertEqual(production_orchestrator["max_inflight_rollouts"], 256)
        self.assertEqual(production_inference["env_vars"]["PRIME_LORA_CACHE_DIR"], "/data/glm52-rl/lora-cache")

    def test_training_ops_accepts_supported_environment_selector(self):
        selected = training_ops._config_dir(
            "configs/trinity-decensor-review-diagnostic",
            require_orchestrator=True,
            require_inference=True,
        )
        with mock.patch.object(training_ops, "mounted_files", return_value=[]):
            inventory = dict(
                training_ops._start_dependency_inventory(selected, ["n82"])
            )
        self.assertIn(
            "environments/decensor-experimental/src/decensor_experimental_v1/scoring.py",
            inventory,
        )

    def test_file_monitors_and_swanlab_identities_are_resolved_and_unique(self):
        diagnostic_trainer = load_config(diagnostic, "trainer.toml")
        diagnostic_orchestrator = load_config(diagnostic, "orchestrator.toml")
        production_trainer = load_config(production, "trainer.toml")
        production_orchestrator = load_config(production, "orchestrator.toml")

        for diagnostic_config, production_config in (
            (diagnostic_trainer, production_trainer),
            (diagnostic_orchestrator, production_orchestrator),
        ):
            diagnostic_monitor = Path(diagnostic_config["output_dir"]) / diagnostic_config["file_monitor"]["filename"]
            production_monitor = Path(production_config["output_dir"]) / production_config["file_monitor"]["filename"]
            self.assertNotEqual(diagnostic_monitor, production_monitor)
            for field in ("project", "name", "group"):
                self.assertNotEqual(
                    diagnostic_config["swanlab"][field],
                    production_config["swanlab"][field],
                )
        self.assertNotEqual(
            diagnostic_trainer["swanlab"]["tags"],
            production_trainer["swanlab"]["tags"],
        )
        self.assertNotEqual(
            diagnostic_orchestrator["swanlab"]["tags"],
            production_orchestrator["swanlab"]["tags"],
        )
        self.assertIn(
            "diag-d29a5958",
            diagnostic_trainer["swanlab"]["tags"],
        )
        self.assertIn(
            "diag-d29a5958",
            diagnostic_orchestrator["swanlab"]["tags"],
        )

    def test_diagnostic_has_no_resume_or_cached_rollout_inputs(self):
        forbidden = (
            "initial_adapter_path",
            "resume_adapter",
            "cached_rollout",
            "rollout_batch",
            "train_rollouts.bin",
        )
        for path in diagnostic.glob("*.toml"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, path.name)

    def test_fresh_start_is_fresh_and_skips_cold_paths(self):
        calls = []
        events = []

        def ssh(host, command, timeout=60):
            calls.append((host, command))
            if "FRESH_START_PRECHECK_OK" in command:
                events.append(("preflight", host))
                return 0, "FRESH_START_PRECHECK_OK"
            if command.endswith(" adapter-http"):
                events.append(("adapter", host))
            elif command.endswith("/runtime/orchestrator.sh"):
                events.append(("orchestrator", host))
            elif "orchestrator control file did not arrive" in command:
                events.append(("control", host))
            elif command.endswith(" trainer"):
                events.append(("trainer", host))
            return 0, "ready"

        def remote_ok(host, command, _label, **kwargs):
            calls.append((host, command))
            events.append(("remote", host, command))
            return "ready"

        selected = {
            "kind": "general-agent",
            "trainer": "configs/trinity-decensor-review-diagnostic/trainer.toml",
            "orchestrator": "configs/trinity-decensor-review-diagnostic/orchestrator.toml",
            "inference": "configs/trinity-decensor-review-diagnostic/inference.toml",
        }
        cold_cleanup = mock.Mock()
        cache_cleanup = mock.Mock()
        transport = mock.Mock()
        real_preflight = training_ops._fresh_start_preflight
        transport.side_effect = lambda run_dir, restart_route=False: events.append(
            ("transport", run_dir, restart_route)
        )
        def fresh_preflight(*args, **kwargs):
            events.append(("preflight_phase",))
            return real_preflight(*args, **kwargs)

        with mock.patch.object(training_ops, "_config_dir", return_value=selected), \
            mock.patch.object(training_ops, "_start_dependency_inventory", return_value=[]), \
            mock.patch.object(training_ops, "_deploy_dependencies", side_effect=lambda _files: events.append(("deploy",))), \
            mock.patch.object(training_ops, "_orchestrator_image_preflight", side_effect=lambda: events.append(("image",)) or "sha256:" + "d" * 64), \
            mock.patch.object(training_ops, "_fresh_start_preflight", side_effect=fresh_preflight), \
            mock.patch.object(training_ops, "_ssh", side_effect=ssh), \
            mock.patch.object(training_ops, "_ssh_address", return_value="127.0.0.1"), \
            mock.patch.object(training_ops, "_fan", side_effect=lambda hosts, function: {host: function(host) for host in hosts}), \
            mock.patch.object(training_ops, "_remote_ok", side_effect=remote_ok), \
            mock.patch.object(training_ops, "_cold_start_cleanup", cold_cleanup), \
            mock.patch.object(training_ops, "_cold_adapter_cache_cleanup", cache_cleanup), \
            mock.patch.object(training_ops, "_start_trainer_pull", transport), \
            mock.patch.object(training_ops, "_repair_supervisors"), \
            mock.patch.object(training_ops.time, "sleep") as sleep_mock, \
            mock.patch.object(training_ops, "trainer_launcher", "/runtime/cluster.sh"), \
            mock.patch.object(training_ops, "orchestrator_launcher", "/runtime/orchestrator.sh"), \
            mock.patch("builtins.print"):
            training_ops.start(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                mode="fresh",
                nodes=["n82"],
                supervisors=False,
                config_dir="configs/trinity-decensor-review-diagnostic",
            )

        cold_cleanup.assert_not_called()
        cache_cleanup.assert_not_called()
        preflight_phases = [index for index, event in enumerate(events) if event[0] == "preflight_phase"]
        self.assertEqual(len(preflight_phases), 2)
        deploy_index = next(index for index, event in enumerate(events) if event[0] == "deploy")
        image_index = next(index for index, event in enumerate(events) if event[0] == "image")
        self.assertLess(preflight_phases[0], deploy_index)
        self.assertLess(deploy_index, image_index)
        self.assertLess(image_index, preflight_phases[1])
        transport.assert_called_once_with(
            load_config(diagnostic, "orchestrator.toml")["output_dir"],
            restart_route=False,
        )
        commands = [command for _host, command in calls]
        self.assertFalse(any("start_inference.sh" in command for command in commands))
        self.assertTrue(any(command.endswith(" adapter-http") for command in commands))
        self.assertTrue(any(command.endswith(" trainer") for command in commands))
        self.assertTrue(any(command.endswith("/runtime/orchestrator.sh") for command in commands))
        self.assertTrue(any("trinity-decensor-experimental-fixed-diag-d29a5958" in command for command in commands))
        for command in commands:
            if command.endswith((" adapter-http", " trainer", "/runtime/orchestrator.sh")):
                self.assertIn("FRESH_START_NO_REPLACE=1", command)
        adapter_index = next(i for i, event in enumerate(events) if event[0] == "adapter")
        transport_index = next(i for i, event in enumerate(events) if event[0] == "transport")
        orch_index = next(i for i, event in enumerate(events) if event[0] == "orchestrator")
        control_index = next(i for i, event in enumerate(events) if event[0] == "control")
        trainer_index = next(i for i, event in enumerate(events) if event[0] == "trainer")
        self.assertLess(adapter_index, transport_index)
        self.assertLess(transport_index, orch_index)
        self.assertLess(orch_index, control_index)
        self.assertLess(control_index, trainer_index)
        self.assertLess(preflight_phases[1], adapter_index)
        orch_command = next(command for command in commands if command.endswith("/runtime/orchestrator.sh"))
        self.assertIn("TRAINER_WATCHDOG_WAIT_FOR_INITIAL_HEALTH=1", orch_command)
        self.assertFalse(any("pkill" in command or "docker rm -f" in command for command in commands))
        sleep_mock.assert_not_called()

    def test_fresh_preflight_refuses_each_remote_gate_failure(self):
        selected = {
            "trainer": "configs/trinity-decensor-review-diagnostic/trainer.toml",
            "orchestrator": "configs/trinity-decensor-review-diagnostic/orchestrator.toml",
            "inference": "configs/trinity-decensor-review-diagnostic/inference.toml",
        }
        failures = (
            "FRESH_START_PATH_PRESENT path=/data/glm52-rl/runs/example",
            "FRESH_START_PATH_SYMLINK path=/data/glm52-rl/runs/example",
            "FRESH_START_CACHE_PRESENT path=/data/glm52-rl/lora-cache/f81b26589e09c5c655d8778664fcc641ce8be39bb19ac7f75f2a6fa9a3f28928",
            "FRESH_START_CACHE_SYMLINK path=/data/glm52-rl/lora-cache/f81b26589e09c5c655d8778664fcc641ce8be39bb19ac7f75f2a6fa9a3f28928",
            "FRESH_START_CONTAINER_ACTIVE name=glm52-decensor-trainer",
        )
        for failure in failures:
            with self.subTest(failure=failure):
                with mock.patch.object(
                    training_ops,
                    "_ssh",
                    return_value=(23, failure),
                ) as ssh:
                    with self.assertRaises(training_ops.Refused) as raised:
                        training_ops._fresh_start_preflight(
                            "trinity-decensor-experimental-fixed-diag-d29a5958",
                            selected,
                            ["n82"],
                            "/data/glm52-rl/runs/trinity-decensor-experimental-fixed-diag-d29a5958/run_trinity-decensor-experimental-fixed-diag-d29a5958",
                        )
                self.assertIn(failure, str(raised.exception))
                self.assertTrue(ssh.called)

    def test_fresh_preflight_script_checks_exact_paths_cache_and_containers(self):
        selected = {
            "trainer": "configs/trinity-decensor-review-diagnostic/trainer.toml",
            "orchestrator": "configs/trinity-decensor-review-diagnostic/orchestrator.toml",
            "inference": "configs/trinity-decensor-review-diagnostic/inference.toml",
        }
        with mock.patch.object(
            training_ops,
            "_ssh",
            return_value=(0, "FRESH_START_PRECHECK_OK"),
        ) as ssh, mock.patch.object(
            training_ops,
            "_fan",
            side_effect=lambda hosts, function: {
                host: function(host) for host in hosts
            },
        ):
            training_ops._fresh_start_preflight(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                selected,
                ["n82"],
                "/data/glm52-rl/runs/trinity-decensor-experimental-fixed-diag-d29a5958/run_trinity-decensor-experimental-fixed-diag-d29a5958",
            )
        scripts = {
            call.args[0]: call.args[1]
            for call in ssh.call_args_list
        }
        self.assertEqual(set(scripts), {"n82", "h200b"})
        script = scripts["h200b"]
        self.assertIn("[ ! -e \"$path\" ]", script)
        self.assertIn("[ ! -L \"$path\" ]", script)
        self.assertIn("docker ps", script)
        self.assertIn(training_ops.trainer_container, script)
        self.assertIn(training_ops.orchestrator_container, script)
        self.assertIn(training_ops.adapter_container, script)
        self.assertIn("FRESH_START_PRECHECK_OK", script)
        source_url = "http://127.0.0.1:18100/trinity-decensor-experimental-fixed-diag-d29a5958"
        cache_source = source_url + "/broadcasts/step_0"
        self.assertEqual(
            hashlib.sha256(cache_source.encode()).hexdigest(),
            "f81b26589e09c5c655d8778664fcc641ce8be39bb19ac7f75f2a6fa9a3f28928",
        )
        self.assertIn(hashlib.sha256(cache_source.encode()).hexdigest(), script)
        self.assertNotIn(hashlib.sha256(source_url.encode()).hexdigest(), script)
        self.assertNotIn(hashlib.sha256(cache_source.encode()).hexdigest(), scripts["n82"])

    def test_fresh_preflight_checks_implicit_adapter_host_for_custom_nodes(self):
        selected = {
            "trainer": "configs/trinity-decensor-review-diagnostic/trainer.toml",
            "orchestrator": "configs/trinity-decensor-review-diagnostic/orchestrator.toml",
            "inference": "configs/trinity-decensor-review-diagnostic/inference.toml",
        }
        with mock.patch.object(
            training_ops,
            "_ssh",
            return_value=(0, "FRESH_START_PRECHECK_OK"),
        ) as ssh, mock.patch.object(
            training_ops,
            "_fan",
            side_effect=lambda hosts, function: {
                host: function(host) for host in hosts
            },
        ):
            training_ops._fresh_start_preflight(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                selected,
                ["n76"],
                "/data/glm52-rl/runs/trinity-decensor-experimental-fixed-diag-d29a5958/run_trinity-decensor-experimental-fixed-diag-d29a5958",
            )
        checked_hosts = {call.args[0] for call in ssh.call_args_list}
        self.assertIn("n76", checked_hosts)
        self.assertIn(training_ops.trainer_head, checked_hosts)
        self.assertIn(training_ops.orchestrator_host, checked_hosts)

    def test_fresh_launchers_have_fail_closed_no_replace_guards(self):
        cluster = (root / "training_ops/runtime/china/cluster.sh").read_text()
        orchestrator = (root / "training_ops/runtime/h200/start_orchestrator.sh").read_text()
        self.assertIn('fresh_start_no_replace=${FRESH_START_NO_REPLACE:-0}', cluster)
        self.assertIn('fresh_start_no_replace=${FRESH_START_NO_REPLACE:-0}', orchestrator)
        self.assertIn('fresh start refuses existing trainer container', cluster)
        self.assertIn('fresh start refuses existing adapter container', cluster)
        self.assertIn('fresh_no_replace_container "$relay_container"', orchestrator)
        self.assertIn('fresh_no_replace_container "$ORCH_CT"', orchestrator)
        self.assertIn('ORCH_CONTAINER_ID="$orchestrator_container_id"', orchestrator)
        self.assertIn('ORCH_LAUNCH_IDENTITY="$orchestrator_launch_identity"', orchestrator)
        self.assertIn('PUBLISHER_SCRIPT="$deployment/training_ops/runtime/h200/sync_rollout_batches.sh"', orchestrator)
        self.assertIn('if [ "$fresh_start_no_replace" = 0 ]; then', cluster)
        self.assertIn('if [ "$fresh_start_no_replace" = 0 ]; then', orchestrator)
        self.assertIn("pkill -f '[t]rainer_failure_watchdog.sh'", orchestrator)
        watchdog = (root / "training_ops/runtime/h200/trainer_failure_watchdog.sh").read_text()
        self.assertNotIn("pkill -f '[s]ync_rollout_batches.sh'", watchdog)
        self.assertIn("publisher_start_time", watchdog)
        self.assertIn("SOURCE_RUN_DIR", watchdog)

    def test_live_cleanup_and_no_clobber_contract_is_explicit(self):
        plan = (root / "ideas.md").read_text(encoding="utf-8")
        for marker in (
            "snapshot the full production staging receipt",
            "server identity",
            "max_loras=1",
            "max_cpu_loras=2",
            "same server instance and engine identity",
            "exactly one verified production receipt",
            "no second receipt",
            "running=0",
            "fail closed",
            "no-clobber gate",
            "executable gate, not only text",
            "use `test ! -e`",
            'test ! -e "$path"',
            "run parent, service, output, broadcast",
            "control, rollout, and checkpoint paths",
            "archive the absent results before",
            "Abort if any path exists",
            "Do not unload production before the run",
            "POST to the root `/remove_lora_adapter` endpoint",
            "status, name, and ID",
            "`/staging_status`",
            "Never run raw `rm`",
            "cache files remain engine-managed",
            "byte-for-byte unchanged",
            "production start/restore path",
            "Do not improvise a raw API restore",
        ):
            self.assertIn(marker, plan)

        trainer = load_config(diagnostic, "trainer.toml")
        orchestrator = load_config(diagnostic, "orchestrator.toml")
        run_dir = Path(orchestrator["output_dir"])
        diagnostic_paths = {
            Path(trainer["output_dir"]),
            run_dir,
            run_dir / "broadcasts",
            run_dir / "control",
            run_dir / "rollouts",
        }
        self.assertEqual(len(diagnostic_paths), 5)
        self.assertNotIn(
            Path("/data/glm52-rl/runs/trinity-decensor-experimental-fixed"),
            diagnostic_paths,
        )


if __name__ == "__main__":
    unittest.main()
