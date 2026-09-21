import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
import training_ops.tests  # noqa: E402,F401 - installs the local-only test guard.
import training_ops  # noqa: E402


puller = root / "training_ops/runtime/china/pull_rollout_batches.sh"


class RolloutPullTest(unittest.TestCase):
    def tools(self, directory):
        tools = Path(directory) / "tools"
        tools.mkdir()
        (tools / "ssh").write_text(
            "#!/usr/bin/env bash\nset -eu\nlast=${!#}\n"
            "case \" $* \" in *'@'*) exec bash -c \"$last\" ;; esac\n"
            "printf '%s\\n' \"$TRAINER_CT\"\n"
        )
        (tools / "rsync").write_text(
            "#!/usr/bin/env bash\nset -eu\na=(\"$@\"); "
            "src=${a[${#a[@]}-2]}; dst=${a[${#a[@]}-1]}; "
            "[[ $src = *:* ]] && src=${src#*:}; [[ $dst = *:* ]] && dst=${dst#*:}; "
            "mkdir -p \"$(dirname \"$dst\")\"; cp -- \"$src\" \"$dst\"\n"
        )
        for path in tools.iterdir():
            path.chmod(0o755)
        return {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"}

    def environment(self, source, destination, base):
        return {
            **base,
            "ROLLOUT_TRANSFER_DIRECTION": "trainer_pull",
            "RUN": "alpha",
            "TRAINER_ADDRESSES": "10.92.20.82",
            "TRAINER_CT": training_ops.trainer_container,
            "ORCH_CT": training_ops.orchestrator_container,
            "SOURCE_RUN_DIR": str(source),
            "DEST_RUN_DIR": str(destination),
            "SOURCE_HOST": "192.0.2.1",
            "SOURCE_PORT": "20099",
            "SOURCE_USER": "root",
            "SOURCE_IDENTITY_FILE": str(Path(source).parent / "key"),
            "SOURCE_KNOWN_HOSTS_FILE": str(Path(source).parent / "known_hosts"),
        }

    def test_generated_start_status_and_stop_use_the_exact_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.tools(directory)
            old_script = training_ops.rollout_pull_script
            commands = []
            try:
                training_ops.rollout_pull_script = str(puller)
                commands = [training_ops._rollout_pull_command(str(run_dir), action) for action in ("start", "status", "stop")]
                self.assertNotIn("china-trainer", "\n".join(commands))
                for command in commands[:2]:
                    result = subprocess.run(["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                pid_file = run_dir / ".rollout-pull.pid"
                self.assertTrue(pid_file.is_file())
                pid_file.unlink()
                result = subprocess.run(["bash", "-c", commands[2]], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotEqual(subprocess.run(["bash", "-c", commands[1]], env=environment).returncode, 0)
                self.assertEqual(subprocess.run(["flock", "-n", str(run_dir / ".rollout-pull.lock"), "-c", "true"]).returncode, 0)
            finally:
                if commands:
                    subprocess.run(["bash", "-c", commands[2]], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_pull_is_checksummed_resumable_and_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            files = {
                "control/orch.toml": b"control",
                "broadcasts/step_1/receivers-ready.json": b"ready",
                "rollouts/step_1/train_rollouts.bin": b"rollout-batch",
            }
            for relative, content in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            old = time.time() - 3
            os.utime(source / "control/orch.toml", (old, old))
            stable = destination / "broadcasts/step_2/STABLE"
            stable.parent.mkdir(parents=True)
            stable.write_bytes(b"stable")
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(ONCE="1", PARALLEL_THRESHOLD="1", PARALLEL_STREAMS="2")

            for _ in range(2):
                result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
            for relative, content in files.items():
                self.assertEqual((destination / relative).read_bytes(), content)
                digest = hashlib.sha256(content).hexdigest()
                self.assertEqual((source / f"{relative}.delivered").read_text().strip(), digest)
            remote_step = source / "broadcasts/step_2"
            self.assertEqual((remote_step / "STABLE").read_bytes(), b"stable")
            self.assertFalse(remote_step.is_symlink())
            self.assertEqual((remote_step.stat().st_uid, remote_step.stat().st_gid), (1000, 1000))
            self.assertEqual(remote_step.stat().st_mode & 0o777, 0o755)
            probe = remote_step / "uid-1000-write-probe"
            probe.write_bytes(b"writable")
            probe.unlink()
            self.assertTrue((source / ".trainer-health-lease").read_text().strip().isdigit())

            control = source / "control/orch.toml"
            control.write_bytes(b"new-control")
            os.utime(control, (old, old))
            result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((destination / "control/orch.toml").read_bytes(), b"control")

            (source / "control/orch.toml.delivered").unlink()
            (destination / "control/orch.toml").unlink()
            result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((destination / "control/orch.toml").read_bytes(), b"new-control")

            rollout = source / "rollouts/step_1/train_rollouts.bin"
            (source / "rollouts/step_1/train_rollouts.bin.delivered").unlink()
            rollout.write_bytes(b"different-rollout")
            result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                (destination / "rollouts/step_1/train_rollouts.bin").read_bytes(),
                b"rollout-batch",
            )

    def test_stable_refuses_an_existing_step_with_unexpected_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            step = source / "broadcasts/step_3"
            step.mkdir(parents=True)
            step.chmod(0o700)
            stable = destination / "broadcasts/step_3/STABLE"
            stable.parent.mkdir(parents=True)
            stable.write_bytes(b"stable")
            environment = self.environment(source, destination, self.tools(directory))
            environment["ONCE"] = "1"
            result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(step.stat().st_mode & 0o777, 0o700)
            self.assertFalse((step / "STABLE").exists())


if __name__ == "__main__":
    unittest.main()
