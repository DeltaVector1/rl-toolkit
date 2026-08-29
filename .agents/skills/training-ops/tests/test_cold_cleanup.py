import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
