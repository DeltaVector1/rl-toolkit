import ast
import re
import unittest
from unittest.mock import patch

import training_ops


class InferenceQualificationTest(unittest.TestCase):
    def test_prime_rl_local_import_closure_is_deployed_and_read_only_mounted(self):
        spec = training_ops._inference_qualification_spec("glm52-mtp-h200")
        closure = training_ops._prime_rl_local_import_closure()
        self.assertEqual(
            set(closure),
            set(training_ops._inference_prime_rl_entry_modules)
            | {
                "prime_rl._compat",
                "prime_rl.configs.orchestrator",
                "prime_rl.configs.shared",
                "prime_rl.configs.trainer",
                "prime_rl.inference.vllm.adapter_validation",
                "prime_rl.model_identity",
                "prime_rl.trainer.distributed.expert_parallel",
                "prime_rl.trainer.distributed.partial_reshard",
                "prime_rl.trainer.distributed.pipeline_parallel",
                "prime_rl.trainer.fp8_checkpoint",
                "prime_rl.trainer.lora",
                "prime_rl.trainer.model",
                "prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention",
                "prime_rl.trainer.models.kernels.fp8_indexer",
                "prime_rl.trainer.models.kernels.sparse_mla_bwd",
                "prime_rl.trainer.models.kernels.sparse_mla_fwd",
                "prime_rl.trainer.models.layers.checkpointing",
                "prime_rl.trainer.models.layers.fp8_ada",
                "prime_rl.trainer.models.layers.lora.multi_linear",
                "prime_rl.trainer.models.layers.lora.native_fp8_moe",
                "prime_rl.trainer.models.layers.moe",
                "prime_rl.trainer.models.layers.norms",
                "prime_rl.trainer.models.layers.rotary_emb",
                "prime_rl.trainer.mxfp4",
                "prime_rl.trainer.parallel_dims",
                "prime_rl.trainer.runs",
                "prime_rl.trainer.weights",
                "prime_rl.utils.cp",
                "prime_rl.utils.vlm",
            },
        )
        inventory = set(spec["deploy_files"])
        self.assertTrue(set(closure.values()) <= inventory)
        launcher = (training_ops.repo_root / spec["launcher"]).read_text()
        for path in closure.values():
            self.assertRegex(
                launcher,
                rf'\$deployment/{re.escape(path)}"?:[^\n]*:ro\s*\\',
                path,
            )

    def test_overlay_inspected_launchers_are_deployed_to_h200a(self):
        spec = training_ops._inference_qualification_spec("glm52-mtp-h200")
        overlay_preflight = (
            training_ops.repo_root / "scripts/h200/flashmla_overlay_preflight.py"
        )
        inspected_launchers = set(
            re.findall(
                r'deployment / "(scripts/h200/[A-Za-z0-9_.-]+\.sh)"',
                overlay_preflight.read_text(),
            )
        )

        self.assertTrue(inspected_launchers)
        self.assertTrue(
            inspected_launchers <= set(spec["deploy_files"]),
            inspected_launchers - set(spec["deploy_files"]),
        )
        self.assertEqual(spec["host"], "h200a")
        self.assertEqual(spec["host"], training_ops.inference_hosts[0])

    def test_all_preflight_local_imports_are_deployed(self):
        spec = training_ops._inference_qualification_spec("glm52-mtp-h200")
        preflight = training_ops.repo_root / "scripts/glm52_h200_mtp_preflight.py"
        tree = ast.parse(preflight.read_text())
        imported = {
            f"source/src/{node.module.replace('.', '/')}.py"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("prime_rl.")
        }

        self.assertTrue(imported)
        self.assertTrue(imported <= set(spec["deploy_files"]))

    def test_mtp_source_inventory_matches_launcher_mounts(self):
        spec = training_ops._inference_qualification_spec("glm52-mtp-h200")
        launcher = (training_ops.repo_root / spec["launcher"]).read_text()
        inventory = {
            path for path in spec["deploy_files"] if path.startswith("source/")
        }
        mounted = set(re.findall(r'-v "\$deployment/([^" ]+)":', launcher))
        mounted.update(re.findall(r'source=\$deployment/([^,]+),', launcher))

        launcher_sources = {path for path in mounted if path.startswith("source/")}
        self.assertTrue(inventory <= launcher_sources, inventory - launcher_sources)
        self.assertIn(
            '-v "$deployment/source/src/prime_rl/glm52_routing.py":'
            "/app/.venv/lib/python3.12/site-packages/prime_rl/glm52_routing.py:ro",
            launcher,
        )

    def test_registered_inventory_and_deploy_files_are_inference_scoped(self):
        spec = training_ops._inference_qualification_spec("glm52-mtp-h200")
        self.assertEqual(spec["host"], training_ops.inference_hosts[0])
        self.assertIn(
            "configs/qualification/glm52-mtp-h200.toml", spec["deploy_files"]
        )
        self.assertIn(
            "scripts/glm52_h200_mtp_preflight.py", spec["deploy_files"]
        )
        self.assertIn(
            "patches/vllm/0006-v026-mtp-slot-mapping-46908.patch",
            spec["deploy_files"],
        )
        self.assertIn(
            "patches/vllm/0007-v026-mtp-slot-mapping-47013.patch",
            spec["deploy_files"],
        )

    def test_unregistered_and_blocked_profiles_refuse_before_ssh(self):
        with patch.object(training_ops, "_ssh") as ssh:
            with self.assertRaises(training_ops.Refused):
                training_ops.qualify_inference("not-registered")
            ssh.assert_not_called()
        blocked = next(iter(training_ops.blocked_hosts))
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        spec["host"] = blocked
        with patch.dict(training_ops.operations["qualification"], {"blocked": spec}):
            with self.assertRaises(training_ops.Refused):
                training_ops._inference_qualification_spec("blocked")

    def test_start_is_foreground_and_checks_ports_before_deploy(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        deployed = []
        remote_calls = []

        def remote(host, command, label, timeout=120):
            remote_calls.append((host, command, label, timeout))
            if label == "inference qualification launch":
                return "mtp-qualification-container:" + "a" * 64 + "\n"
            return ""

        with (
            patch.object(training_ops, "_inference_qualification_spec", return_value=spec),
            patch.object(training_ops, "_inference_qualification_ports", return_value=[8000, 8100]),
            patch.object(
                training_ops,
                "_inference_qualification_identity",
                return_value={"status": "absent"},
            ),
            patch.object(training_ops, "_assert_live_services_are_down") as guard,
            patch.object(training_ops, "_inference_qualification_port_conflicts"),
            patch.object(training_ops, "deploy", side_effect=lambda path, nodes: deployed.append((path, nodes)) or True),
            patch.object(training_ops, "_remote_ok", side_effect=remote),
        ):
            result = training_ops.qualify_inference("glm52-mtp-h200")

        guard.assert_called_once_with([], include_inference=True)
        self.assertEqual(result["container_id"], "a" * 64)
        self.assertEqual([nodes for _path, nodes in deployed], [[spec["host"]]] * len(spec["deploy_files"]))
        launch = next(command for _host, command, label, _timeout in remote_calls if label == "inference qualification launch")
        self.assertNotIn("nohup", launch)
        self.assertNotIn("setsid", launch)
        self.assertNotIn(" &", launch)
        self.assertIn("start_mtp_qualification.sh", launch)

    def test_port_conflict_is_fail_closed_before_deploy(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        with (
            patch.object(training_ops, "_inference_qualification_spec", return_value=spec),
            patch.object(training_ops, "_inference_qualification_ports", return_value=[8000, 8100]),
            patch.object(
                training_ops,
                "_inference_qualification_identity",
                return_value={"status": "absent"},
            ),
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(
                training_ops,
                "_inference_qualification_port_conflicts",
                side_effect=training_ops.Refused("port conflict"),
            ),
            patch.object(training_ops, "deploy") as deploy,
        ):
            with self.assertRaisesRegex(training_ops.Refused, "port conflict"):
                training_ops.qualify_inference("glm52-mtp-h200")
        deploy.assert_not_called()

    def test_live_service_guard_is_fail_closed_before_deploy(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        with (
            patch.object(training_ops, "_inference_qualification_spec", return_value=spec),
            patch.object(
                training_ops,
                "_assert_live_services_are_down",
                side_effect=training_ops.Refused("GPU owner is active"),
            ),
            patch.object(training_ops, "deploy") as deploy,
        ):
            with self.assertRaisesRegex(training_ops.Refused, "GPU owner"):
                training_ops.qualify_inference("glm52-mtp-h200")
        deploy.assert_not_called()

    def test_inference_files_cannot_deploy_to_a_trainer_host(self):
        path = training_ops.glm52_mtp_h200_qualification["profile"]
        with self.assertRaisesRegex(training_ops.Refused, "non-inference host"):
            training_ops.deploy(path, nodes=[training_ops.trainer_head])

    def test_registration_requires_referenced_files_and_unique_inventory(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        spec["deploy_files"] = [
            path
            for path in spec["deploy_files"]
            if path != spec["launcher"]
        ]
        with patch.dict(
            training_ops.operations["qualification"], {"missing-launcher": spec}
        ):
            with self.assertRaises(training_ops.Refused):
                training_ops._inference_qualification_spec("missing-launcher")
        duplicate = dict(training_ops.glm52_mtp_h200_qualification)
        duplicate["deploy_files"] = list(duplicate["deploy_files"]) + [
            duplicate["profile"]
        ]
        with patch.dict(
            training_ops.operations["qualification"], {"duplicate": duplicate}
        ):
            with self.assertRaises(training_ops.Refused):
                training_ops._inference_qualification_spec("duplicate")

    def test_status_and_cleanup_require_exact_container_identity(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        record = {
            "host": spec["host"],
            "container": spec["container"],
            "status": "running",
            "restart_count": 0,
            "image": training_ops.inference_image,
            "command": ["@", f"{training_ops.remote_repo}/{spec['profile']}"],
            "restart_policy": "no",
            "init": True,
        }
        with patch.object(training_ops, "_inference_qualification_spec", return_value=spec), patch.object(
            training_ops, "_inference_qualification_identity", return_value=record
        ), patch.object(training_ops, "_remote_ok", return_value="removed\n") as remote:
            self.assertEqual(training_ops.status_inference_qualification("glm52-mtp-h200")["status"], "running")
            self.assertEqual(training_ops.stop_inference_qualification("glm52-mtp-h200"), "removed")
        self.assertEqual(remote.call_args.args[0], spec["host"])
        self.assertIn("docker rm -f", remote.call_args.args[1])

    def test_exited_exact_identity_is_reported_and_removed(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        record = {
            "host": spec["host"],
            "container": spec["container"],
            "status": "exited",
            "restart_count": 1,
            "image": training_ops.inference_image,
            "command": ["@", f"{training_ops.remote_repo}/{spec['profile']}"],
            "restart_policy": "no",
            "init": True,
        }
        with (
            patch.object(training_ops, "_inference_qualification_spec", return_value=spec),
            patch.object(training_ops, "_inference_qualification_identity", return_value=record),
            patch.object(training_ops, "_remote_ok", return_value="removed\n") as remote,
        ):
            self.assertEqual(
                training_ops.status_inference_qualification("glm52-mtp-h200")["status"],
                "exited",
            )
            self.assertEqual(training_ops.stop_inference_qualification("glm52-mtp-h200"), "removed")
        cleanup = remote.call_args.args[1]
        self.assertIn("--filter 'name=^/glm52-optims-mtp-h200$'", cleanup)

    def test_unknown_identity_is_refused_before_cleanup(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        record = {
            "host": spec["host"],
            "container": spec["container"],
            "status": "exited",
            "restart_count": 0,
            "image": "unexpected:image",
            "command": ["unexpected"],
            "restart_policy": "no",
            "init": True,
        }
        with (
            patch.object(training_ops, "_inference_qualification_spec", return_value=spec),
            patch.object(training_ops, "_inference_qualification_identity", return_value=record),
            patch.object(training_ops, "_remote_ok") as remote,
        ):
            with self.assertRaisesRegex(training_ops.Refused, "identity differs"):
                training_ops.stop_inference_qualification("glm52-mtp-h200")
        remote.assert_not_called()

    def test_status_parser_rejects_malformed_remote_identity(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        with patch.object(training_ops, "_inference_qualification_spec", return_value=spec), patch.object(
            training_ops, "_ssh", return_value=(0, "malformed\n")
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid record"):
                training_ops.status_inference_qualification("glm52-mtp-h200")


if __name__ == "__main__":
    unittest.main()
