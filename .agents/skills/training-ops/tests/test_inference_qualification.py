import unittest
from unittest.mock import patch

import training_ops


class InferenceQualificationTest(unittest.TestCase):
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
            patch.object(training_ops, "_inference_qualification_port_conflicts"),
            patch.object(training_ops, "deploy", side_effect=lambda path, nodes: deployed.append((path, nodes)) or True),
            patch.object(training_ops, "_remote_ok", side_effect=remote),
        ):
            result = training_ops.qualify_inference("glm52-mtp-h200")

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

    def test_status_parser_rejects_malformed_remote_identity(self):
        spec = dict(training_ops.glm52_mtp_h200_qualification)
        with patch.object(training_ops, "_inference_qualification_spec", return_value=spec), patch.object(
            training_ops, "_ssh", return_value=(0, "malformed\n")
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid record"):
                training_ops.status_inference_qualification("glm52-mtp-h200")


if __name__ == "__main__":
    unittest.main()
