import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import training_ops


class DeployTest(unittest.TestCase):
    def test_adapter_bucket_folder_uses_hyphenated_muse_run_name(self):
        self.assertEqual(training_ops._adapter_bucket_run_folder("muse18"), "muse-18-run")
        self.assertEqual(training_ops._adapter_bucket_run_folder("muse-19"), "muse-19-run")

    def test_adapter_broadcast_parser_requires_one_run_tree(self):
        run_base = "/data/glm52-rl/runs/muse18"
        parsed = training_ops._parse_adapter_broadcasts(
            "\n".join(
                [
                    f"{run_base}/run_muse18/broadcasts/step_0",
                    f"{run_base}/run_muse18/broadcasts/step_10",
                ]
            ),
            run_base,
        )
        self.assertEqual([item.step for item in parsed], [0, 10])
        with self.assertRaises(RuntimeError):
            training_ops._parse_adapter_broadcasts(
                "/data/glm52-rl/runs/muse17/run_muse17/broadcasts/step_1",
                run_base,
            )

    def test_archive_refuses_to_remove_latest_live_adapter(self):
        run = "muse18"
        run_base = f"{training_ops.run_root}/{run}"
        stable = (
            f"{run_base}/run_muse18/broadcasts/step_10\n"
            f"{run_base}/run_muse18/broadcasts/step_11\n"
        )
        with patch.object(training_ops, "_local_command"), patch.object(
            training_ops, "_ssh"
        ) as ssh:
            ssh.side_effect = [
                (0, stable),
                (0, "STABLE\nadapter_config.json\nadapter_model.safetensors\n"),
                (0, "STABLE\nadapter_config.json\nadapter_model.safetensors\n"),
            ]
            with self.assertRaises(training_ops.Refused):
                training_ops.archive_adapters(run, [11])

    def test_archive_reads_the_active_policy_step(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(0, "Updated weights to step 25 in 1.2s\n"),
        ):
            self.assertEqual(training_ops._confirmed_active_policy_step("muse19"), 25)

    def test_archive_removes_source_only_after_verified_upload(self):
        run = "muse18"
        run_base = f"{training_ops.run_root}/{run}"
        stable_path = f"{run_base}/run_muse18/broadcasts/step_10"
        stable = (
            stable_path + "\n"
            + f"{run_base}/run_muse18/broadcasts/step_11\n"
        )
        expected = ["STABLE", "adapter_config.json", "adapter_model.safetensors"]
        digest = hashlib.sha256(b"adapter").hexdigest()
        events = []

        def fake_ssh(host, command, timeout=120):
            if command.startswith("test -d "):
                return 0, stable
            events.append("cleanup")
            return 0, ""

        with (
            patch.object(training_ops, "_local_command", side_effect=lambda *args, **kwargs: events.append("auth")),
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(
                training_ops,
                "_remote_broadcast_files",
                side_effect=[expected, expected, expected],
            ),
            patch.object(training_ops, "_confirmed_active_policy_step", return_value=11),
            patch.object(training_ops, "_remote_sha256", return_value=digest),
            patch.object(
                training_ops,
                "_download_adapter_file",
                side_effect=lambda remote, local: (local.write_bytes(b"adapter"), events.append("download")),
            ),
            patch.object(training_ops, "_archive_one_file", side_effect=lambda *args: events.append("verified")),
        ):
            self.assertEqual(training_ops.archive_adapters(run, [10]), [10])

        self.assertLess(events.index("verified"), events.index("cleanup"))

    def test_deploy_routes_runtime_config_to_inference_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "configs" / "runtime" / "test.toml"
            local.parent.mkdir(parents=True)
            local.write_text("value = 1\n")
            want = hashlib.md5(local.read_bytes()).hexdigest()
            copied_to = []

            def fake_run(command, **kwargs):
                if command[0] == "scp":
                    copied_to.append(command[-1].split(":", 1)[0])
                return Mock(returncode=0)

            def fake_ssh(host, command, timeout=120):
                if command.startswith("md5sum "):
                    return 0, want
                return 0, ""

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(training_ops.deploy("configs/runtime/test.toml"))

            self.assertCountEqual(copied_to, training_ops.inference_hosts)

    def test_deploy_selects_only_configured_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "configs" / "runtime" / "test.toml"
            local.parent.mkdir(parents=True)
            local.write_text("value = 1\n")
            want = hashlib.md5(local.read_bytes()).hexdigest()
            copied_to = []

            def fake_run(command, **kwargs):
                if command[0] == "scp":
                    copied_to.append(command[-1])
                return Mock(returncode=0)

            def fake_ssh(host, command, timeout=120):
                if command.startswith("md5sum "):
                    return 0, want
                return 0, ""

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(
                    training_ops.deploy("configs/runtime/test.toml", nodes=["n82"])
                )
                self.assertTrue(
                    training_ops.deploy(
                        "configs/runtime/test.toml",
                        nodes=["n82"],
                        repository="glm52-qualification",
                    )
                )

            self.assertEqual(
                copied_to,
                [
                    f"n82:{training_ops.paths['remote_repository']}/configs/runtime/test.toml",
                    f"n82:{training_ops.paths['glm52_qualification_repository']}/configs/runtime/test.toml",
                ],
            )

    def test_deploy_rejects_unknown_or_missing_repository(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._repository_root("arbitrary")
        with patch.object(training_ops, "paths", {"remote_repository": "/safe"}):
            with self.assertRaises(training_ops.Refused):
                training_ops._repository_root("glm52-qualification")

    def test_deploy_rejects_blocked_host_before_copy(self):
        with patch.object(training_ops.subprocess, "run") as run:
            with self.assertRaises(training_ops.Refused):
                training_ops.deploy(
                    "configs/runtime/glm52-inference.toml",
                    nodes=["n79"],
                    repository="glm52-qualification",
                )
        run.assert_not_called()

    def test_glm_qualification_uses_isolated_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            config = repo / "configs" / "qualification" / "glm.toml"
            config.parent.mkdir(parents=True)
            config.write_text("[data.fake]\ncount = 1\n")
            deploys = []
            launches = []

            def fake_deploy(path, nodes=None, repository="default"):
                deploys.append((path, repository))
                return True

            def fake_ssh(host, command, timeout=120):
                if "docker ps" in command:
                    return 0, ""
                launches.append(command)
                return 0, ""

            def fake_fan(hosts, function, workers=None):
                return {host: function(host) for host in hosts}

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(
                    training_ops,
                    "_profile",
                    return_value={"kind": "glm52", "trainer": "configs/qualification/glm.toml"},
                ),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops, "_fan", side_effect=fake_fan),
                patch.object(training_ops, "deploy", side_effect=fake_deploy),
                patch.object(training_ops.time, "sleep"),
            ):
                training_ops.qualify("glm-q1", nodes=["n82"])

            self.assertEqual(
                deploys,
                [
                    ("scripts/china/cluster.sh", "glm52-qualification"),
                    ("configs/qualification/glm.toml", "glm52-qualification"),
                ],
            )
            self.assertTrue(
                any(
                    f"bash {training_ops.paths['glm52_qualification_repository']}/scripts/china/cluster.sh trainer"
                    in command
                    for command in launches
                )
            )

if __name__ == "__main__":
    unittest.main()
