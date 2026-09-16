import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


class ColdCleanupTest(unittest.TestCase):
    def test_adapter_cache_cleanup_is_bounded_and_lock_aware(self):
        script = training_ops._adapter_cache_cleanup_script()
        self.assertIn("cache=/data/glm52-rl/lora-cache", script)
        self.assertIn('flock -n "$lock"', script)
        self.assertIn("find '$path' -xdev -depth -delete", script)
        self.assertIn("adapter-staging.lock) continue", script)
        self.assertIn("unknown adapter cache entry", script)
        self.assertNotIn("rm -rf", script)

    def test_adapter_cache_cleanup_preserves_exact_initial_identity(self):
        target = b"model"
        config = b"{}\n"
        identity = {
            "target_sha256": hashlib.sha256(target).hexdigest(),
            "target_size": len(target),
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "config_size": len(config),
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            matching = cache / ("a" * 64)
            matching.mkdir()
            (matching / "READY").write_text("ready\n", encoding="ascii")
            (matching / "adapter_model.safetensors").write_bytes(target)
            (matching / "adapter_config.json").write_bytes(config)
            (matching / "adapter_model.safetensors.sha256").write_text(
                identity["target_sha256"] + "\n", encoding="ascii"
            )
            (matching / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_url": "http://old/step_0",
                        "files": {
                            "adapter_model.safetensors": {
                                "size": len(target),
                                "sha256": identity["target_sha256"],
                            },
                            "adapter_config.json": {
                                "size": len(config),
                                "sha256": identity["config_sha256"],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            stale = cache / ("b" * 64)
            stale.mkdir()
            (stale / "READY").write_text("ready\n", encoding="ascii")
            malformed = cache / ("c" * 64)
            malformed.mkdir()
            old_cache = training_ops.paths["adapter_cache"]
            training_ops.paths["adapter_cache"] = str(cache)
            try:
                script = training_ops._adapter_cache_cleanup_script(identity).replace(
                    "sudo ", ""
                )
                result = subprocess.run(
                    ["bash", "-c", script], capture_output=True, text=True
                )
            finally:
                training_ops.paths["adapter_cache"] = old_cache
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(matching.exists())
            self.assertFalse(stale.exists())
            self.assertFalse(malformed.exists())
            self.assertIn("preserving matching adapter cache", result.stdout)

    def test_adapter_cache_cleanup_requires_idle_hosts(self):
        idle = {
            "h200a": (0, "port=8000 running=0"),
            "h200b": (0, "port=8000 running=0"),
        }
        cleared = {
            "h200a": (0, "removed=2 free_gib=100"),
            "h200b": (0, "removed=1 free_gib=500"),
        }
        with mock.patch.object(
            training_ops, "_fan", side_effect=[idle, cleared]
        ), mock.patch("builtins.print") as output:
            training_ops._cold_adapter_cache_cleanup()
        self.assertEqual(output.call_count, 2)

        with mock.patch.object(
            training_ops,
            "_fan",
            return_value={"h200a": (34, "active requests"), "h200b": (0, "idle")},
        ):
            with self.assertRaisesRegex(training_ops.Refused, "requires idle"):
                training_ops._cold_adapter_cache_cleanup()

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
                (target / "run-20260101_000000-old").mkdir()
                (target / "rollout-metrics.jsonl").write_text("old")
                (target / "final_summary.json").write_text("old")
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
                phases = ["stop" if "stop_only=1" in script else "delete" for _, script, _ in calls]
                self.assertEqual(len(phases), 4)
                self.assertLess(max(i for i, phase in enumerate(phases) if phase == "stop"), min(i for i, phase in enumerate(phases) if phase == "delete"))
                self.assertTrue(target.exists())
                self.assertFalse(stale.exists())
                self.assertFalse(trainer_rollout.exists())
                self.assertFalse((target / "run-20260101_000000-old").exists())
                self.assertFalse((target / "rollout-metrics.jsonl").exists())
                self.assertFalse((target / "final_summary.json").exists())
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
                stale_dir = Path(training_ops.run_root) / "alpha" / "run_alpha"
                publisher_dir = stale_dir.parent
                (publisher_dir / "rollouts").mkdir(parents=True)
                process = subprocess.Popen(
                    [
                        "bash",
                        str(Path(training_ops.runtime_root) / "china" / "sync_microbatches.sh"),
                    ],
                    env={
                        **os.environ,
                        "run_dir": str(publisher_dir),
                        "nodes": "10.92.20.76",
                        "interval": "60",
                    },
                )
                try:
                    pid_file = publisher_dir / ".microbatch-sync.pid"
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

    def test_cleanup_scans_only_requested_publisher_dirs_after_pid_file_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = training_ops.run_root
            old_runtime = training_ops.runtime_root
            processes = []
            try:
                root = Path(temporary)
                training_ops.run_root = str(root / "runs")
                training_ops.runtime_root = str(
                    Path(__file__).resolve().parents[1] / "runtime"
                )
                sync_script = Path(training_ops.runtime_root) / "china" / "sync_microbatches.sh"

                def start_publisher(run_dir):
                    run_dir = Path(run_dir)
                    (run_dir / "rollouts").mkdir(parents=True)
                    process = subprocess.Popen(
                        ["bash", str(sync_script)],
                        env={
                            **os.environ,
                            "run_dir": str(run_dir),
                            "nodes": "10.92.20.76",
                            "interval": "60",
                        },
                    )
                    processes.append(process)
                    pid_file = run_dir / ".microbatch-sync.pid"
                    for _ in range(50):
                        if pid_file.exists():
                            break
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists())
                    return process, pid_file

                requested = Path(training_ops.run_root) / "alpha"
                requested_process, requested_pid_file = start_publisher(requested)
                requested_pid_file.unlink()
                foreign_parent_process, _ = start_publisher(
                    Path(training_ops.run_root) / "beta"
                )
                foreign_service_process, foreign_service_pid_file = start_publisher(
                    Path(training_ops.run_root) / "gamma" / "run_gamma"
                )
                # Keep a valid foreign PID file in place. The cleanup must not
                # follow the old cross-run PID-file glob.
                foreign_service_pid_file.write_text(
                    str(foreign_service_process.pid), encoding="ascii"
                )

                def fake_ssh(host, script, timeout=0):
                    result = subprocess.run(
                        ["bash", "-c", script], capture_output=True, text=True
                    )
                    return result.returncode, result.stdout + result.stderr

                with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
                    training_ops._cold_start_cleanup("alpha", ["n76"])

                requested_process.wait(timeout=3)
                self.assertIsNotNone(requested_process.returncode)
                self.assertIsNone(foreign_parent_process.poll())
                self.assertIsNone(foreign_service_process.poll())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=3)
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
                stale_dir = Path(training_ops.run_root) / "alpha" / "run_alpha"
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
