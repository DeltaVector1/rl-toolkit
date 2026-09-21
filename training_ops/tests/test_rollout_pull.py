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
            try:
                training_ops.rollout_pull_script = str(puller)
                commands = [training_ops._rollout_pull_command(str(run_dir), action) for action in ("start", "status", "stop")]
                self.assertNotIn("china-trainer", "\n".join(commands))
                for command in commands:
                    result = subprocess.run(["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
            finally:
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
            stable = destination / "broadcasts/step_1/STABLE"
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
            self.assertEqual((source / "broadcasts/step_1/STABLE").read_bytes(), b"stable")
            self.assertTrue((source / ".trainer-health-lease").read_text().strip().isdigit())

            (source / "rollouts/step_1/train_rollouts.bin").write_bytes(b"changed")
            result = subprocess.run(["bash", str(puller), "run"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((destination / "rollouts/step_1/train_rollouts.bin").read_bytes(), files["rollouts/step_1/train_rollouts.bin"])


if __name__ == "__main__":
    unittest.main()
