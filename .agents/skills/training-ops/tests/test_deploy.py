import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import training_ops


class DeployTest(unittest.TestCase):
    def test_mount_preflight_refuses_missing_local_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(training_ops, "repo_root", Path(directory)),
                patch.object(training_ops, "_mounts_in", return_value={"source/missing.py"}),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "missing locally"):
                    training_ops.trainer_mount_preflight(["n82"])

    def test_mount_preflight_refuses_stale_remote_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source" / "mounted.py"
            source.parent.mkdir(parents=True)
            source.write_text("current\n")
            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={"n82": (0, "source/mounted.py\tdeadbeef\n")},
                ),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "preflight failed"):
                    training_ops.trainer_mount_preflight(["n82"])

    def test_mount_preflight_accepts_matching_remote_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source" / "mounted.py"
            source.parent.mkdir(parents=True)
            source.write_text("current\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={"n82": (0, f"source/mounted.py\t{digest}\n")},
                ),
            ):
                self.assertEqual(
                    training_ops.trainer_mount_preflight(["n82"]),
                    {"source/mounted.py": digest},
                )

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
            want = hashlib.sha256(local.read_bytes()).hexdigest()
            copied_to = []

            def fake_run(command, **kwargs):
                if command[0] == "scp":
                    copied_to.append(command[-1].split(":", 1)[0])
                return Mock(returncode=0)

            def fake_ssh(host, command, timeout=120):
                if command.startswith("test -f "):
                    return 0, want
                return 0, ""

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(training_ops.deploy("configs/runtime/test.toml"))

            self.assertCountEqual(copied_to, training_ops.inference_hosts)

if __name__ == "__main__":
    unittest.main()
