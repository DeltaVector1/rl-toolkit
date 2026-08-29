import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class ColdCleanupTest(unittest.TestCase):
    def test_cleanup_is_limited_to_one_run_and_reports_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = training_ops.run_root
            try:
                training_ops.run_root = str(Path(temporary) / "runs")
                target = Path(training_ops.run_root) / "alpha" / "run_alpha"
                stale = target / "rollouts" / "step_1" / "train_rollouts.bin"
                stale.parent.mkdir(parents=True)
                stale.write_bytes(b"stale")
                trainer_rollout = Path(training_ops.run_root) / "alpha" / "rollouts" / "step_1" / "rank_0.bin"
                trainer_rollout.parent.mkdir(parents=True)
                trainer_rollout.write_bytes(b"stale micro-batch")
                (target / "control").mkdir()
                (target / "control" / "orch.toml.delivered").write_text("old")
                (target / "broadcasts").mkdir()
                (target / "broadcasts" / "step_0").write_text("old")
                (target / "checkpoints").mkdir()
                (target / "checkpoints" / "optimizer.pt").write_text("old")
                (target / ".incoming").mkdir()
                (target / "hf-cache").mkdir()
                (target / ".batch-sync.pid").write_text("123")
                other = Path(training_ops.run_root) / "beta" / "run_beta"
                (other / "rollouts").mkdir(parents=True)
                (other / "rollouts" / "train_rollouts.bin").write_text("keep")
                model = Path(temporary) / "model"
                model.write_text("keep")

                calls = []

                def fake_ssh(host, script, timeout=0):
                    calls.append((host, script, timeout))
                    result = subprocess.run(
                        ["bash", "-c", script], capture_output=True, text=True
                    )
                    return result.returncode, result.stdout + result.stderr

                output = []
                with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), mock.patch(
                    "builtins.print", side_effect=output.append
                ):
                    training_ops._cold_start_cleanup("alpha", ["n76"])

                self.assertEqual({host for host, _, _ in calls}, {"n76", "h200a"})
                self.assertTrue(all(timeout == 300 for _, _, timeout in calls))
                self.assertTrue(target.exists())
                self.assertFalse(stale.exists())
                self.assertFalse(trainer_rollout.exists())
                self.assertTrue(other.exists())
                self.assertEqual(model.read_text(), "keep")
                self.assertTrue(any("train_rollouts.bin" in line for line in output))
            finally:
                training_ops.run_root = old_root

    def test_invalid_run_and_blocked_host_are_refused_without_ssh(self):
        with mock.patch.object(training_ops, "_ssh") as ssh:
            with self.assertRaises(training_ops.Refused):
                training_ops._cold_start_cleanup("../escape", ["n76"])
            with self.assertRaises(training_ops.Refused):
                training_ops._cold_start_cleanup("alpha", ["n79"])
            ssh.assert_not_called()

    def test_cleanup_stops_only_a_matching_microbatch_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = training_ops.run_root
            old_runtime = training_ops.runtime_root
            try:
                root = Path(temporary)
                training_ops.run_root = str(root / "runs")
                training_ops.runtime_root = str(
                    Path(__file__).resolve().parents[1] / "runtime"
                )
                stale_dir = Path(training_ops.run_root) / "stale" / "run_stale"
                (stale_dir / "rollouts").mkdir(parents=True)
                process = subprocess.Popen(
                    [
                        "bash",
                        str(Path(training_ops.runtime_root) / "china" / "sync_microbatches.sh"),
                    ],
                    env={
                        **os.environ,
                        "run_dir": str(stale_dir),
                        "nodes": "10.92.20.76",
                        "interval": "60",
                    },
                )
                try:
                    pid_file = stale_dir / ".microbatch-sync.pid"
                    for _ in range(50):
                        if pid_file.exists():
                            break
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())

                    calls = []

                    def fake_ssh(host, script, timeout=0):
                        calls.append((host, script, timeout))
                        result = subprocess.run(
                            ["bash", "-c", script], capture_output=True, text=True
                        )
                        return result.returncode, result.stdout + result.stderr

                    with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
                        training_ops._cold_start_cleanup("alpha", ["n76"])

                    process.wait(timeout=3)
                    self.assertIsNotNone(process.returncode)
                    self.assertEqual({host for host, _, _ in calls}, {"n76", "h200a"})
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=3)
            finally:
                training_ops.run_root = old_root
                training_ops.runtime_root = old_runtime

    def test_cleanup_does_not_kill_an_unrelated_process_with_a_stale_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = training_ops.run_root
            try:
                training_ops.run_root = str(Path(temporary) / "runs")
                stale_dir = Path(training_ops.run_root) / "stale" / "run_stale"
                stale_dir.mkdir(parents=True)
                process = subprocess.Popen(["sleep", "60"])
                try:
                    (stale_dir / ".microbatch-sync.pid").write_text(
                        str(process.pid), encoding="utf-8"
                    )

                    def fake_ssh(host, script, timeout=0):
                        result = subprocess.run(
                            ["bash", "-c", script], capture_output=True, text=True
                        )
                        return result.returncode, result.stdout + result.stderr

                    with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
                        training_ops._cold_start_cleanup("alpha", ["n76"])

                    self.assertIsNone(process.poll())
                finally:
                    process.terminate()
                    process.wait(timeout=3)
            finally:
                training_ops.run_root = old_root

    def test_cleanup_finds_matching_process_after_pid_file_is_unlinked(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = training_ops.run_root
            old_runtime = training_ops.runtime_root
            try:
                root = Path(temporary)
                training_ops.run_root = str(root / "runs")
                training_ops.runtime_root = str(
                    Path(__file__).resolve().parents[1] / "runtime"
                )
                stale_dir = Path(training_ops.run_root) / "stale" / "run_stale"
                (stale_dir / "rollouts").mkdir(parents=True)
                process = subprocess.Popen(
                    [
                        "bash",
                        str(Path(training_ops.runtime_root) / "china" / "sync_microbatches.sh"),
                    ],
                    env={
                        **os.environ,
                        "run_dir": str(stale_dir),
                        "nodes": "10.92.20.76",
                        "interval": "60",
                    },
                )
                try:
                    pid_file = stale_dir / ".microbatch-sync.pid"
                    lock_file = stale_dir / ".microbatch-sync.lock"
                    for _ in range(50):
                        if pid_file.exists() and lock_file.exists():
                            break
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())
                    pid_file.unlink()

                    def fake_ssh(host, script, timeout=0):
                        result = subprocess.run(
                            ["bash", "-c", script], capture_output=True, text=True
                        )
                        return result.returncode, result.stdout + result.stderr

                    with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
                        training_ops._cold_start_cleanup("alpha", ["n76"])

                    process.wait(timeout=3)
                    self.assertIsNotNone(process.returncode)
                    self.assertFalse(pid_file.exists())
                    self.assertFalse(lock_file.exists())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=3)
            finally:
                training_ops.run_root = old_root
                training_ops.runtime_root = old_runtime


if __name__ == "__main__":
    unittest.main()
