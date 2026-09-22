import hashlib
import os
from pathlib import Path
import signal
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
    def tools(self, directory, *, fake_rsync=True):
        tools = Path(directory) / "tools"
        tools.mkdir()
        (tools / "ssh").write_text(
            "#!/usr/bin/env bash\nset -eu\nlast=${!#}\n"
            "if [[ \" $* \" == *\" rsync --server \"* ]]; then "
            "if [ -n \"${SSH_ARG_CAPTURE:-}\" ]; then printf '%s\\n' '-- invocation --' >>\"$SSH_ARG_CAPTURE\"; printf '%s\\n' \"$@\" >>\"$SSH_ARG_CAPTURE\"; fi; "
            "if [ \"${RSYNC_SERVE:-0}\" = 1 ]; then args=(\"$@\"); for ((i=0; i<${#args[@]}; i++)); do "
            "if [[ ${args[i]} == rsync && ${args[i+1]:-} == --server ]]; then "
            "printf -v command '%s ' \"${args[@]:i+1}\"; exec bash -c \"exec /usr/bin/rsync $command\"; fi; done; fi; exit 1; fi\n"
            "[ -z \"${SSH_DELAY:-}\" ] || sleep \"$SSH_DELAY\"\n"
            "[ -n \"${SSH_FAIL:-}\" ] && exit 1 || true\n"
            "if [ \"${SSH_FAIL_STABLE:-0}\" = 1 ] && [[ \"$last\" == */STABLE* ]]; then exit 1; fi\n"
            "if [ \"${SSH_FAIL_STABLE_ATTEMPTS:-0}\" -gt 0 ] && [[ \"$last\" == */STABLE* ]]; then state=${SSH_FAIL_STABLE_STATE:?}; count=0; [ -f \"$state\" ] && count=$(cat \"$state\"); if [ \"$count\" -lt \"$SSH_FAIL_STABLE_ATTEMPTS\" ]; then printf '%s' $((count + 1)) >\"$state\"; exit 1; fi; fi\n"
            "if [ \"${SSH_FAIL_LISTING:-0}\" = 1 ] && [[ \"$last\" == *receivers-ready.json* ]]; then exit 1; fi\n"
            "case \" $* \" in *'@'*) exec bash -c \"$last\" ;; esac\n"
            "printf '%s\\n' \"$TRAINER_CT\"\n"
        )
        if fake_rsync:
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

    def test_start_waits_for_delayed_readiness_and_cleans_up_failed_child(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.tools(directory)
            old_script = training_ops.rollout_pull_script
            commands = []
            try:
                training_ops.rollout_pull_script = str(puller)
                command = training_ops._rollout_pull_command(str(run_dir), "start")
                commands = [training_ops._rollout_pull_command(str(run_dir), action) for action in ("stop", "status")]
                environment["SSH_DELAY"] = "3"
                started = subprocess.run(
                    ["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                self.assertTrue((run_dir / ".rollout-pull.pid").is_file())
                self.assertEqual(
                    subprocess.run(["bash", "-c", commands[1]], env=environment, capture_output=True).returncode,
                    0,
                )
                stopped = subprocess.run(
                    ["bash", "-c", commands[0]], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)

                environment["SSH_FAIL"] = "1"
                failed = subprocess.run(
                    ["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
                self.assertEqual(
                    subprocess.run(
                        ["flock", "-n", str(run_dir / ".rollout-pull.lock"), "-c", "true"],
                        env=environment,
                    ).returncode,
                    0,
                )
                self.assertNotEqual(
                    subprocess.run(["bash", "-c", commands[1]], env=environment, capture_output=True).returncode,
                    0,
                )
            finally:
                if commands:
                    subprocess.run(["bash", "-c", commands[0]], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_lock_loser_preserves_owner_pid_and_owner_signal_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            control = source / "control/orch.toml"
            control.parent.mkdir(parents=True)
            control.write_bytes(b"control")
            os.utime(control, (time.time() - 3, time.time() - 3))
            environment = self.environment(source, destination, self.tools(directory))
            environment["INTERVAL"] = "1"
            pid_file = destination / ".rollout-pull.pid"
            wrappers = set(Path("/tmp").glob("rollout-pull-ssh.*"))
            owner = subprocess.Popen(
                ["setsid", "bash", str(puller), "run"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(100):
                    if pid_file.is_file():
                        break
                    time.sleep(0.05)
                self.assertTrue(pid_file.is_file())
                owner_pid = pid_file.read_text()
                self.assertEqual(owner_pid.strip(), str(owner.pid))

                loser_environment = {**environment, "ONCE": "1"}
                loser = subprocess.run(
                    ["bash", str(puller), "run"],
                    env=loser_environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(loser.returncode, 0, loser.stderr)
                self.assertEqual(pid_file.read_text(), owner_pid)
                status = subprocess.run(["bash", str(puller), "status"], env=environment, capture_output=True, text=True)
                self.assertEqual(status.returncode, 0, f"owner={owner.poll()} stdout={status.stdout} stderr={status.stderr}")
                owner_wrappers = set(Path("/tmp").glob("rollout-pull-ssh.*")) - wrappers
                self.assertEqual(len(owner_wrappers), 1)

                owner.send_signal(signal.SIGTERM)
                owner.communicate(timeout=10)
                self.assertFalse(pid_file.exists())
                self.assertEqual(set(Path("/tmp").glob("rollout-pull-ssh.*")), wrappers)
            finally:
                if owner.poll() is None:
                    owner.terminate()
                    owner.communicate(timeout=10)

    def test_setup_failure_before_lock_preserves_existing_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            tools = Path(directory) / "tools"
            environment = self.environment(source, destination, self.tools(directory))
            (tools / "mktemp").write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $* == *rollout-pull-ssh.* ]]; then exit 1; fi\n"
                "exec /usr/bin/mktemp \"$@\"\n"
            )
            (tools / "mktemp").chmod(0o755)
            destination.mkdir(parents=True)
            pid_file = destination / ".rollout-pull.pid"
            pid_file.write_text("existing-owner\n")
            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(pid_file.read_text(), "existing-owner\n")
            self.assertEqual(
                subprocess.run(
                    ["flock", "-n", str(destination / ".rollout-pull.lock"), "-c", "true"],
                    env=environment,
                ).returncode,
                0,
            )

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

    def test_actual_rsync_ssh_wrapper_preserves_adversarial_options(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source root/alpha/run_alpha"
            destination = base / "destination root/alpha/run_alpha"
            files = {
                "control/orch.toml": b"control",
                "broadcasts/step_1/receivers-ready.json": b"ready",
                "rollouts/step_1/train_rollouts.bin": b"rollout-batch-data",
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
            capture = base / "ssh argv"
            identity = base / "id space 'single' \"double\" \\slash;$(touch marker)"
            known_hosts = base / "known space 'single' \"double\" \\slash;$(touch marker2)"
            environment = self.environment(source, destination, self.tools(directory, fake_rsync=False))
            environment.update(
                ONCE="1",
                SSH_ARG_CAPTURE=str(capture),
                RSYNC_SERVE="1",
                PARALLEL_THRESHOLD="10",
                PARALLEL_STREAMS="2",
                SOURCE_IDENTITY_FILE=str(identity),
                SOURCE_KNOWN_HOSTS_FILE=str(known_hosts),
            )

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(capture.exists(), result.stderr)
            blocks = []
            for line in capture.read_text().splitlines():
                if line == "-- invocation --":
                    blocks.append([])
                else:
                    blocks[-1].append(line)
            self.assertEqual(len(blocks), 5)
            expected_options = [
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                "-o",
                "GlobalKnownHostsFile=/dev/null",
                "-o",
                "UpdateHostKeys=no",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=8",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
                "-i",
                str(identity),
                "-p",
                "20099",
            ]
            for actual in blocks:
                self.assertEqual(actual[: len(expected_options)], expected_options)
                self.assertEqual(actual[len(expected_options) : len(expected_options) + 3], ["-l", "root", "192.0.2.1"])
                self.assertEqual(actual[len(expected_options) + 3 : len(expected_options) + 5], ["rsync", "--server"])
            for relative, content in files.items():
                self.assertEqual((destination / relative).read_bytes(), content)
            self.assertEqual((source / "broadcasts/step_2/STABLE").read_bytes(), b"stable")
            self.assertFalse((base / "marker").exists())
            self.assertFalse((base / "marker2").exists())

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

    def test_stable_push_precedes_rollout_listing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            stable = destination / "broadcasts/step_4/STABLE"
            stable.parent.mkdir(parents=True)
            stable.write_bytes(b"stable")
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(ONCE="1", SSH_FAIL_LISTING="1")

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((source / "broadcasts/step_4/STABLE").read_bytes(), b"stable")

    def test_stable_push_retries_transient_failure_before_rollout_pull(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            (source / "rollouts/step_1").mkdir(parents=True)
            (source / "rollouts/step_1/train_rollouts.bin").write_bytes(b"rollout")
            stable = destination / "broadcasts/step_4/STABLE"
            stable.parent.mkdir(parents=True)
            stable.write_bytes(b"stable")
            state = base / "stable-failures"
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                ONCE="1",
                SSH_FAIL_STABLE_ATTEMPTS="2",
                SSH_FAIL_STABLE_STATE=str(state),
            )

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((source / "broadcasts/step_4/STABLE").read_bytes(), b"stable")
            self.assertEqual(
                (destination / "rollouts/step_1/train_rollouts.bin").read_bytes(),
                b"rollout",
            )

    def test_persistent_stable_push_failure_blocks_rollout_pull(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            (source / "rollouts/step_1").mkdir(parents=True)
            (source / "rollouts/step_1/train_rollouts.bin").write_bytes(b"rollout")
            stable = destination / "broadcasts/step_4/STABLE"
            stable.parent.mkdir(parents=True)
            stable.write_bytes(b"stable")
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(ONCE="1", SSH_FAIL_STABLE="1")

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((source / "broadcasts/step_4/STABLE").exists())
            self.assertFalse((destination / "rollouts/step_1/train_rollouts.bin").exists())


if __name__ == "__main__":
    unittest.main()
