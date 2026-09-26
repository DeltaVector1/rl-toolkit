import json
import os
import unittest
from unittest import mock

import training_ops


APPROVED_SHA256 = "102b14539a2aa7e4b807967f9c9caeaaa974373391a21602ce45b24414fb7863"
IDENTITY = {
    "sha256": APPROVED_SHA256,
    "size": 16446,
    "mode": "644",
}
FULL_TRAINERS = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]


class TrinityTrainerLossPreflightTest(unittest.TestCase):
    def test_identity_requires_measured_size_and_mode(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(training_ops.Refused, "loss identity"):
                training_ops._trinity_trainer_loss_identity()

    def test_inspection_command_is_no_gpu_no_network_and_exactly_bound(self):
        script = training_ops._trinity_trainer_loss_inspection_script(
            'prime-rl:bad"; echo INJECTED; #', IDENTITY
        )
        self.assertIn("docker run --rm --pull=never --network none --gpus none", script)
        self.assertIn("--entrypoint /bin/bash", script)
        self.assertIn("/app/src/prime_rl/trainer/rl/loss.py", script)
        self.assertIn(APPROVED_SHA256, script)
        self.assertIn("sys.version_info[:3] == (3, 12, 3)", script)
        self.assertIn("2.11.0+cu128", script)
        self.assertIn("prime-rl", script)
        self.assertIn("setup_rl_loss_fn", script)
        self.assertIn("inspect.signature", script)
        self.assertIn("docker ps -aq", script)
        self.assertIn("image='prime-rl:bad\"; echo INJECTED; #'", script)

    def test_preflight_checks_full_configured_set_not_launch_subset(self):
        calls = []

        def fake_ssh(host, script, timeout=120):
            calls.append((host, script, timeout))
            return 0, "TRINITY_TRAINER_LOSS_PREFLIGHT_OK\n"

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            training_ops._trinity_trainer_loss_preflight(
                ["n82"], expected_identity=IDENTITY
            )

        self.assertEqual([host for host, _, _ in calls], FULL_TRAINERS)
        self.assertTrue(all(timeout == 180 for _, _, timeout in calls))

    def test_probe_runs_by_captured_immutable_id_not_mutable_tag(self):
        script = training_ops._trinity_trainer_loss_inspection_script(
            training_ops.trainer_image, IDENTITY
        )
        self.assertIn(
            '--entrypoint /bin/bash "$image_id" -c',
            script,
        )
        self.assertNotIn(
            '--entrypoint /bin/bash "$image" -c',
            script,
        )
        self.assertIn('image_id=$(sudo docker image inspect', script)

    def test_identity_rejects_wrong_measured_size_or_mode_before_ssh(self):
        for field, value in (("size", 16445), ("mode", "640")):
            with self.subTest(field=field):
                wrong = dict(IDENTITY)
                wrong[field] = value
                with mock.patch.object(training_ops, "_ssh") as ssh:
                    with self.assertRaisesRegex(training_ops.Refused, "identity"):
                        training_ops._trinity_trainer_loss_preflight(
                            ["n82"], expected_identity=wrong
                        )
                ssh.assert_not_called()

    def test_preflight_rejects_one_host_failure(self):
        def fake_ssh(host, script, timeout=120):
            return (1, "module mismatch") if host == "n76" else (0, "ok")

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            with self.assertRaisesRegex(training_ops.Refused, "n76"):
                training_ops._trinity_trainer_loss_preflight(
                    ["n82", "n76"], expected_identity=IDENTITY
                )

    def test_identity_json_accepts_the_measured_sha_and_requires_fields(self):
        with mock.patch.dict(
            os.environ,
            {"PRIME_RL_TRAINER_LOSS_IDENTITY": json.dumps(IDENTITY)},
            clear=True,
        ):
            self.assertEqual(
                training_ops._trinity_trainer_loss_identity(), IDENTITY
            )

    def test_start_calls_loss_preflight_before_dependency_deploy(self):
        events = []
        selected = {
            "trainer": "configs/general-agent/trainer.toml",
            "orchestrator": "configs/general-agent/orchestrator.toml",
            "inference": "configs/general-agent/inference.toml",
            "kind": "trinity",
        }
        with mock.patch.object(training_ops, "_config_dir", return_value=selected), mock.patch.object(
            training_ops, "_validate_trinity_afmoe_overlay"
        ), mock.patch.object(
            training_ops,
            "_trinity_trainer_loss_preflight",
            side_effect=lambda nodes: events.append("loss"),
        ), mock.patch.object(
            training_ops,
            "_start_dependency_inventory",
            side_effect=lambda selected, nodes: events.append("inventory") or [],
        ), mock.patch.object(
            training_ops,
            "_deploy_dependencies",
            side_effect=lambda files: events.append("deploy"),
        ), mock.patch.object(
            training_ops, "_trinity_trainer_preflight", side_effect=lambda nodes: events.append("base")
        ), mock.patch.object(training_ops, "_orchestrator_image_preflight", return_value="sha256:" + "a" * 64), mock.patch.object(
            training_ops, "_ssh", return_value=(1, "missing")
        ):
            with self.assertRaises(training_ops.Refused):
                training_ops.start("preflight-test", nodes=["n82"], supervisors=False)
        self.assertEqual(events[:2], ["loss", "inventory"])

    def test_qualify_calls_loss_preflight_before_trainer_launch(self):
        events = []

        def fake_ssh(host, script, timeout=120):
            if "docker ps" in script:
                return 0, ""
            events.append("launch")
            return 0, "ready"

        with mock.patch.object(
            training_ops,
            "_trinity_trainer_loss_preflight",
            side_effect=lambda nodes: events.append("loss"),
        ), mock.patch.object(
            training_ops,
            "_trainer_dependency_inventory",
            side_effect=lambda config, nodes: events.append("inventory") or [],
        ), mock.patch.object(
            training_ops,
            "_deploy_dependencies",
            side_effect=lambda files: events.append("deploy"),
        ), mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), mock.patch(
            "builtins.print"
        ):
            training_ops.qualify(
                "loss-preflight-test",
                nodes=["n82"],
                config_dir="configs/trinity-h200-post-update-qualification",
            )

        self.assertEqual(events[:3], ["loss", "inventory", "deploy"])
        self.assertIn("launch", events)


if __name__ == "__main__":
    unittest.main()
