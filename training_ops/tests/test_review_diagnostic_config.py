import copy
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
            "run-trinity-decensor-experimental-fixed-diag-d29a5958",
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
            "run-trinity-decensor-experimental-fixed-diag-d29a5958",
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

        for path in diagnostic.glob("*.toml"):
            self.assertNotIn("n79", path.read_text(encoding="utf-8").lower())
        self.assertEqual(production_trainer["output_dir"], "/data/glm52-rl/runs/trinity-decensor-experimental-fixed")
        self.assertEqual(production_orchestrator["batch_size"], 1024)
        self.assertEqual(production_orchestrator["max_inflight_rollouts"], 256)
        self.assertEqual(production_inference["env_vars"]["PRIME_LORA_CACHE_DIR"], "/data/glm52-rl/lora-cache")

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
            "run-trinity-decensor-experimental-fixed-diag-d29a5958",
            diagnostic_trainer["swanlab"]["tags"],
        )
        self.assertIn(
            "run-trinity-decensor-experimental-fixed-diag-d29a5958",
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

    def test_restart_without_resume_is_fresh_and_skips_cold_paths(self):
        calls = []

        def ssh(host, command, timeout=60):
            calls.append((host, command))
            return 0, "ready"

        def remote_ok(host, command, _label, **kwargs):
            calls.append((host, command))
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
        with mock.patch.object(training_ops, "_config_dir", return_value=selected), \
            mock.patch.object(training_ops, "_start_dependency_inventory", return_value=[]), \
            mock.patch.object(training_ops, "_deploy_dependencies"), \
            mock.patch.object(training_ops, "_orchestrator_image_preflight", return_value="sha256:" + "d" * 64), \
            mock.patch.object(training_ops, "_ssh", side_effect=ssh), \
            mock.patch.object(training_ops, "_ssh_address", return_value="127.0.0.1"), \
            mock.patch.object(training_ops, "_fan", side_effect=lambda hosts, function: {host: function(host) for host in hosts}), \
            mock.patch.object(training_ops, "_remote_ok", side_effect=remote_ok), \
            mock.patch.object(training_ops, "_cold_start_cleanup", cold_cleanup), \
            mock.patch.object(training_ops, "_cold_adapter_cache_cleanup", cache_cleanup), \
            mock.patch.object(training_ops, "_start_trainer_pull", transport), \
            mock.patch.object(training_ops, "_repair_supervisors"), \
            mock.patch.object(training_ops.time, "sleep"), \
            mock.patch.object(training_ops, "trainer_launcher", "/runtime/cluster.sh"), \
            mock.patch.object(training_ops, "orchestrator_launcher", "/runtime/orchestrator.sh"), \
            mock.patch("builtins.print"):
            training_ops.start(
                "trinity-decensor-experimental-fixed-diag-d29a5958",
                mode="restart",
                nodes=["n82"],
                supervisors=False,
                config_dir="configs/trinity-decensor-review-diagnostic",
            )

        cold_cleanup.assert_not_called()
        cache_cleanup.assert_not_called()
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
