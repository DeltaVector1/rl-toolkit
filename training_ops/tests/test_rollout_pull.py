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
    def setUp(self):
        # These are SSH-backend regression tests, independent of live selection.
        previous = training_ops.rollout_transfer_backend
        training_ops.rollout_transfer_backend = "ssh"
        self.addCleanup(setattr, training_ops, "rollout_transfer_backend", previous)

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
            "if [ -n \"${SSH_LISTING_DELAY:-}\" ] && [[ \"$last\" == *receivers-ready.json* ]]; then sleep \"$SSH_LISTING_DELAY\"; fi\n"
            "if [ \"${SSH_FAIL_STABLE:-0}\" = 1 ] && [[ \"$last\" == */STABLE* ]]; then exit 1; fi\n"
            "if [ \"${SSH_READ_STDIN:-0}\" = 1 ] && [[ \" $* \" != *\" -n \"* ]]; then cat >/dev/null; fi\n"
            "if [ \"${SSH_FAIL_STABLE_ATTEMPTS:-0}\" -gt 0 ] && [[ \"$last\" == */STABLE* ]]; then state=${SSH_FAIL_STABLE_STATE:?}; count=0; [ -f \"$state\" ] && count=$(cat \"$state\"); if [ \"$count\" -lt \"$SSH_FAIL_STABLE_ATTEMPTS\" ]; then printf '%s' $((count + 1)) >\"$state\"; exit 1; fi; fi\n"
            "if [ -n \"${SSH_MISSING_TRAINER:-}\" ] && [[ \" $* \" == *\" $SSH_MISSING_TRAINER \"* ]] && [[ \"$last\" == *\"docker ps\"* ]]; then printf '%s\\n' missing; exit 0; fi\n"
            "if [ \"${SSH_FAIL_LISTING:-0}\" = 1 ] && [[ \"$last\" == *receivers-ready.json* ]]; then exit 1; fi\n"
            "case \" $* \" in *'@'*) exec bash -c \"$last\" ;; esac\n"
            "printf '%s\\n' \"$TRAINER_CT\"\n"
        )
        if fake_rsync:
            (tools / "rsync").write_text(
                "#!/usr/bin/env bash\nset -eu\na=(\"$@\"); "
                "src=${a[${#a[@]}-2]}; dst=${a[${#a[@]}-1]}; "
                "[[ $src = *:* ]] && src=${src#*:}; [[ $dst = *:* ]] && dst=${dst#*:}; "
                "if [ -n \"${RSYNC_ATTEMPTS_LOG:-}\" ]; then printf '%s\\n' \"$src\" >>\"$RSYNC_ATTEMPTS_LOG\"; fi; "
                "if [ -n \"${RSYNC_FAIL_SOURCE:-}\" ] && [[ \"$src\" == *\"$RSYNC_FAIL_SOURCE\"* ]] && [ ! -e \"${RSYNC_FAIL_MARKER:?}\" ]; then "
                "if (set -C; : >\"$RSYNC_FAIL_MARKER\") 2>/dev/null; then exit 1; fi; fi; "
                "mkdir -p \"$(dirname \"$dst\")\"; "
                "if [ -n \"${RSYNC_SLEEP_ONCE_MARKER:-}\" ] && [ ! -e \"$RSYNC_SLEEP_ONCE_MARKER\" ]; then "
                "(set -C; : >\"$RSYNC_SLEEP_ONCE_MARKER\") 2>/dev/null || true; "
                "sleep \"${RSYNC_SLEEP:-0}\"; fi; "
                "while [ -n \"${RSYNC_BLOCK_FILE:-}\" ] && [ -e \"$RSYNC_BLOCK_FILE\" ]; do sleep 0.1; done; "
                "cp -- \"$src\" \"$dst\"\n"
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
                commands = {
                    action: training_ops._rollout_pull_command(str(run_dir), action)
                    for action in ("start", "status", "state", "stop", "pause", "resume")
                }
                self.assertNotIn("china-trainer", "\n".join(commands.values()))
                for action in ("start", "status", "state"):
                    result = subprocess.run(["bash", "-c", commands[action]], env=environment, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if action == "state":
                        self.assertEqual(result.stdout.strip(), "running")
                pid_file = run_dir / ".rollout-pull.pid"
                self.assertTrue(pid_file.is_file())
                token = "a" * 32
                result = subprocess.run(["bash", "-c", f"{commands['pause']} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "paused")
                self.assertFalse(pid_file.exists())
                self.assertEqual((run_dir / ".rollout-pull-paused").read_text().splitlines()[:2], [token, str(run_dir)])
                blocked = subprocess.run(["bash", "-c", commands["start"]], env=environment, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(blocked.returncode, 0)
                wrong_token = subprocess.run(["bash", "-c", f"{commands['resume']} {'b' * 32}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(wrong_token.returncode, 0)
                self.assertTrue((run_dir / ".rollout-pull-paused").exists())
                result = subprocess.run(["bash", "-c", f"{commands['resume']} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "restored")
                self.assertTrue(pid_file.is_file())
                result = subprocess.run(["bash", "-c", commands["stop"]], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotEqual(subprocess.run(["bash", "-c", commands["status"]], env=environment).returncode, 0)
                result = subprocess.run(["bash", "-c", commands["state"]], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "stopped")
                self.assertEqual(subprocess.run(["flock", "-n", str(run_dir / ".rollout-pull.lock"), "-c", "true"]).returncode, 0)
            finally:
                if commands:
                    subprocess.run(["bash", "-c", commands["stop"]], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_state_reports_stopped_without_creating_a_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                command = training_ops._rollout_pull_command(str(run_dir), "state")
                result = subprocess.run(["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "stopped")
                self.assertFalse((run_dir / ".rollout-pull.lock").exists())
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
                self.assertFalse((run_dir / ".rollout-pull-control.lock").exists())
            finally:
                training_ops.rollout_pull_script = old_script

    def test_pause_of_stopped_puller_holds_barrier_and_does_not_start_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            old_script = training_ops.rollout_pull_script
            token = "c" * 32
            try:
                training_ops.rollout_pull_script = str(puller)
                pause = training_ops._rollout_pull_command(str(run_dir), "pause")
                start = training_ops._rollout_pull_command(str(run_dir), "start")
                resume = training_ops._rollout_pull_command(str(run_dir), "resume")
                result = subprocess.run(["bash", "-c", f"{pause} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout.strip()), (0, "paused"), result.stderr)
                self.assertEqual((run_dir / ".rollout-pull-paused").read_text().splitlines()[2:5], ["stopped", "0", "no"])
                blocked = subprocess.run(["bash", "-c", start], env=environment, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(blocked.returncode, 0)
                result = subprocess.run(["bash", "-c", f"{resume} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout.strip()), (0, "already-stopped"), result.stderr)
                self.assertFalse((run_dir / ".rollout-pull-paused").exists())
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
            finally:
                training_ops.rollout_pull_script = old_script

    def test_partial_pause_failure_keeps_owned_receipt_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            old_script = training_ops.rollout_pull_script
            token = "d" * 32
            try:
                training_ops.rollout_pull_script = str(puller)
                start = training_ops._rollout_pull_command(str(run_dir), "start")
                pause = training_ops._rollout_pull_command(str(run_dir), "pause")
                resume = training_ops._rollout_pull_command(str(run_dir), "resume")
                result = subprocess.run(["bash", "-c", start], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                lock = run_dir / ".rollout-pull.lock"
                saved_lock = run_dir / ".saved-rollout-pull.lock"
                lock.rename(saved_lock)
                lock.symlink_to(saved_lock.name)
                result = subprocess.run(["bash", "-c", f"{pause} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                receipt = (run_dir / ".rollout-pull-paused").read_text().splitlines()
                self.assertEqual(receipt[0:4], [token, str(run_dir), "running", receipt[3]])
                self.assertEqual(receipt[4], "yes")
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
                lock.unlink()
                saved_lock.rename(lock)
                result = subprocess.run(["bash", "-c", f"{resume} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout.strip()), (0, "restored"), result.stderr)
                self.assertTrue((run_dir / ".rollout-pull.pid").is_file())
            finally:
                cleanup = training_ops._rollout_pull_command(str(run_dir), "stop")
                subprocess.run(["bash", "-c", cleanup], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_pause_does_not_restore_a_puller_stopped_by_another_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            old_script = training_ops.rollout_pull_script
            token = "e" * 32
            race_marker = Path(directory) / "other-stop-won"
            hook = Path(directory) / "bash_env"
            hook.write_text(
                "kill() {\n"
                "  if [ \"${RACE_KILL_GROUP:-}\" -gt 1 ] && [ \"$#\" -eq 2 ] &&\n"
                "    [ \"$1\" = -- ] && [ \"$2\" = \"-$RACE_KILL_GROUP\" ] &&\n"
                "    (set -C; : >\"$RACE_MARKER\") 2>/dev/null; then\n"
                "    /bin/kill -TERM -- \"-$RACE_KILL_GROUP\" 2>/dev/null || true\n"
                "    for _ in $(seq 1 50); do\n"
                "      ps -eo pgid=,stat= | awk -v group=\"$RACE_KILL_GROUP\" '$1 == group && $2 !~ /^Z/ {found=1} END {exit !found}' || break\n"
                "      sleep 0.1\n"
                "    done\n"
                "  fi\n"
                "  builtin kill \"$@\"\n"
                "}\n"
            )
            try:
                training_ops.rollout_pull_script = str(puller)
                start = training_ops._rollout_pull_command(str(run_dir), "start")
                pause = training_ops._rollout_pull_command(str(run_dir), "pause")
                resume = training_ops._rollout_pull_command(str(run_dir), "resume")
                result = subprocess.run(["bash", "-c", start], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                recorded_pid = (run_dir / ".rollout-pull.pid").read_text().strip()
                environment.update(
                    BASH_ENV=str(hook),
                    RACE_KILL_GROUP=recorded_pid,
                    RACE_MARKER=str(race_marker),
                )
                result = subprocess.run(["bash", "-c", f"{pause} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout.strip()), (0, "paused"), result.stderr)
                self.assertTrue(race_marker.is_file())
                receipt = (run_dir / ".rollout-pull-paused").read_text().splitlines()
                self.assertEqual(receipt[2:5], ["running", recorded_pid, "no"])
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
                result = subprocess.run(["bash", "-c", f"{resume} {token}"], env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout.strip()), (0, "already-stopped"), result.stderr)
                self.assertFalse((run_dir / ".rollout-pull-paused").exists())
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
            finally:
                cleanup = training_ops._rollout_pull_command(str(run_dir), "stop")
                subprocess.run(["bash", "-c", cleanup], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_stop_is_idempotent_with_stale_lock_and_pid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            (run_dir / ".rollout-pull.lock").touch()
            (run_dir / ".rollout-pull.pid").write_text("orphaned-owner\n")
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stop_command = training_ops._rollout_pull_command(str(run_dir), "stop")
                for _ in range(2):
                    stopped = subprocess.run(
                        ["bash", "-c", stop_command],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.assertFalse((run_dir / ".rollout-pull.pid").exists())
                self.assertTrue((run_dir / ".rollout-pull.lock").exists())
            finally:
                training_ops.rollout_pull_script = old_script

    def test_stop_accepts_unresolvable_lock_owner_but_preserves_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            tools = Path(directory) / "tools"
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            lock_file = run_dir / ".rollout-pull.lock"
            lock_file.touch()
            (run_dir / ".rollout-pull.pid").write_text("unresolvable-owner\n")
            (tools / "flock").write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"$#\" -eq 2 ] && [ \"$1\" = -x ] && [ \"$2\" = 7 ]; then exec /usr/bin/flock \"$@\"; fi\n"
                "exit \"${FAKE_FLOCK_STATUS:-0}\"\n"
            )
            (tools / "lslocks").write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '- '\"$FAKE_LOCK_PATH\"\n"
            )
            for path in (tools / "flock", tools / "lslocks"):
                path.chmod(0o755)
            environment.update(
                FAKE_FLOCK_STATUS="75",
                FAKE_LOCK_PATH=str(lock_file),
            )
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stop_command = training_ops._rollout_pull_command(str(run_dir), "stop")
                stopped = subprocess.run(
                    ["bash", "-c", stop_command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.assertTrue((run_dir / ".rollout-pull.pid").exists())
            finally:
                training_ops.rollout_pull_script = old_script

    def test_stop_rejects_malformed_lock_owner_records(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            tools = Path(directory) / "tools"
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            lock_file = run_dir / ".rollout-pull.lock"
            lock_file.touch()
            pid_file = run_dir / ".rollout-pull.pid"
            pid_file.write_text("other-run-owner\n")
            (tools / "flock").write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"$#\" -eq 2 ] && [ \"$1\" = -x ] && [ \"$2\" = 7 ]; then exec /usr/bin/flock \"$@\"; fi\n"
                "exit 75\n"
            )
            (tools / "lslocks").write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$FAKE_LOCK_RECORD\"\n"
            )
            for path in (tools / "flock", tools / "lslocks"):
                path.chmod(0o755)
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stop_command = training_ops._rollout_pull_command(str(run_dir), "stop")
                for record in (
                    f"unknown {lock_file}",
                    f"? {lock_file}",
                    str(lock_file),
                ):
                    environment["FAKE_LOCK_RECORD"] = record
                    stopped = subprocess.run(
                        ["bash", "-c", stop_command],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(stopped.returncode, 0, record)
                    self.assertTrue(pid_file.exists(), record)
            finally:
                training_ops.rollout_pull_script = old_script

    def test_stop_accepts_dead_numeric_lock_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            tools = Path(directory) / "tools"
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            lock_file = run_dir / ".rollout-pull.lock"
            lock_file.touch()
            pid_file = run_dir / ".rollout-pull.pid"
            pid_file.write_text("dead-owner\n")
            (tools / "flock").write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"$#\" -eq 2 ] && [ \"$1\" = -x ] && [ \"$2\" = 7 ]; then exec /usr/bin/flock \"$@\"; fi\n"
                "exit 75\n"
            )
            (tools / "lslocks").write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"999999999 $FAKE_LOCK_PATH\"\n"
            )
            for path in (tools / "flock", tools / "lslocks"):
                path.chmod(0o755)
            environment["FAKE_LOCK_PATH"] = str(lock_file)
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stopped = subprocess.run(
                    ["bash", "-c", training_ops._rollout_pull_command(str(run_dir), "stop")],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.assertTrue(pid_file.exists())
            finally:
                training_ops.rollout_pull_script = old_script

    def test_stop_holds_lock_while_removing_pid_against_new_owner_race(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            tools = Path(directory) / "tools"
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            lock_file = run_dir / ".rollout-pull.lock"
            pid_file = run_dir / ".rollout-pull.pid"
            pid_file.write_text("old-owner\n")
            race_marker = Path(directory) / "race-started"
            flock_log = Path(directory) / "flock-log"
            (tools / "flock").write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "if [ \"$#\" -eq 2 ] && [ \"$1\" = -x ] && [ \"$2\" = 7 ]; then exec /usr/bin/flock \"$@\"; fi\n"
                "printf '%s\\n' \"$*\" >>\"$FAKE_FLOCK_LOG\"\n"
                "if [ \"${FAKE_RACE:-0}\" = 1 ] && [ ! -e \"$FAKE_RACE_MARKER\" ]; then\n"
                "  : >\"$FAKE_RACE_MARKER\"\n"
                "  ( /usr/bin/flock \"$FAKE_LOCK_PATH\" -c \"printf '%s\\\\n' raced > '$FAKE_PID_PATH'\" ) &\n"
                "fi\n"
                "exec /usr/bin/flock \"$@\"\n"
            )
            (tools / "flock").chmod(0o755)
            environment.update(
                FAKE_RACE="1",
                FAKE_RACE_MARKER=str(race_marker),
                FAKE_FLOCK_LOG=str(flock_log),
                FAKE_LOCK_PATH=str(lock_file),
                FAKE_PID_PATH=str(pid_file),
            )
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stopped = subprocess.run(
                    ["bash", "-c", training_ops._rollout_pull_command(str(run_dir), "stop")],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                for _ in range(100):
                    if pid_file.exists() and pid_file.read_text() == "raced\n":
                        break
                    time.sleep(0.05)
                self.assertTrue(pid_file.exists())
                self.assertEqual(pid_file.read_text(), "raced\n")
                self.assertEqual(len(flock_log.read_text().splitlines()), 1)
            finally:
                training_ops.rollout_pull_script = old_script

    def test_stop_fails_closed_for_live_lock_owner_and_probe_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.environment(run_dir, run_dir, self.tools(directory))
            lock_file = run_dir / ".rollout-pull.lock"
            owner = subprocess.Popen(["flock", "-n", str(lock_file), "-c", "sleep 10"])
            try:
                for _ in range(100):
                    owners = subprocess.run(
                        ["lslocks", "-n", "-o", "PID,PATH", "--notruncate"],
                        capture_output=True,
                        text=True,
                    )
                    if str(lock_file) in owners.stdout:
                        break
                    time.sleep(0.05)
                self.assertIn(str(lock_file), owners.stdout)
                (run_dir / ".rollout-pull.pid").write_text("other-run-owner\n")
                old_script = training_ops.rollout_pull_script
                try:
                    training_ops.rollout_pull_script = str(puller)
                    stop_command = training_ops._rollout_pull_command(str(run_dir), "stop")
                    stopped = subprocess.run(
                        ["bash", "-c", stop_command],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(stopped.returncode, 0)
                    self.assertIsNone(owner.poll())
                    self.assertTrue((run_dir / ".rollout-pull.pid").exists())
                finally:
                    training_ops.rollout_pull_script = old_script
            finally:
                owner.terminate()
                owner.wait(timeout=10)

            tools = Path(directory) / "tools"
            (tools / "flock").write_text(
                "#!/usr/bin/env bash\n"
                "exit 2\n"
            )
            (tools / "flock").chmod(0o755)
            environment["FAKE_FLOCK_STATUS"] = "2"
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                stopped = subprocess.run(
                    ["bash", "-c", training_ops._rollout_pull_command(str(run_dir), "stop")],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(stopped.returncode, 0)
                self.assertTrue((run_dir / ".rollout-pull.pid").exists())
            finally:
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

    def test_stop_kills_health_lease_companion(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "alpha/run_alpha"
            run_dir.mkdir(parents=True)
            environment = self.tools(directory)
            old_script = training_ops.rollout_pull_script
            try:
                training_ops.rollout_pull_script = str(puller)
                command = training_ops._rollout_pull_command(str(run_dir), "start")
                stop_command = training_ops._rollout_pull_command(str(run_dir), "stop")
                started = subprocess.run(
                    ["bash", "-c", command], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                health = subprocess.run(
                    ["pgrep", "-f", "^rollout-pull-health -c health_lease_loop$"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(health.returncode, 0, health.stderr)

                stopped = subprocess.run(
                    ["bash", "-c", stop_command], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                for _ in range(20):
                    health = subprocess.run(
                        ["pgrep", "-f", "^rollout-pull-health -c health_lease_loop$"],
                        capture_output=True,
                        text=True,
                    )
                    if health.returncode != 0:
                        break
                    time.sleep(0.1)
                self.assertNotEqual(health.returncode, 0, health.stdout)
            finally:
                subprocess.run(["bash", "-c", stop_command], env=environment, capture_output=True, timeout=10)
                training_ops.rollout_pull_script = old_script

    def test_background_jobs_do_not_hold_lock_after_parent_group_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            rollout = source / "rollouts/step_1/train_rollouts.bin"
            rollout.parent.mkdir(parents=True)
            rollout.write_bytes(bytes(range(256)))
            block = base / "rsync-block"
            block.touch()
            attempts = base / "rsync-attempts"
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                PARALLEL_THRESHOLD="1",
                PARALLEL_STREAMS="1",
                PULL_PART_TIMEOUT="300",
                RSYNC_BLOCK_FILE=str(block),
                RSYNC_ATTEMPTS_LOG=str(attempts),
            )
            old_script = training_ops.rollout_pull_script
            training_ops.rollout_pull_script = str(puller)
            stop_command = training_ops._rollout_pull_command(str(destination), "stop")
            start_command = training_ops._rollout_pull_command(str(destination), "start")
            owner = subprocess.Popen(
                ["setsid", "bash", str(puller), "run"],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(100):
                    if attempts.is_file() and attempts.read_text().strip():
                        break
                    time.sleep(0.05)
                self.assertTrue(attempts.is_file())
                self.assertTrue(attempts.read_text().strip())
                os.killpg(owner.pid, signal.SIGTERM)
                owner.wait(timeout=10)

                self.assertEqual(
                    subprocess.run(
                        ["flock", "-n", str(destination / ".rollout-pull.lock"), "-c", "true"],
                        env=environment,
                    ).returncode,
                    0,
                )
                stopped = subprocess.run(
                    ["bash", "-c", stop_command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                started = subprocess.run(
                    ["bash", "-c", start_command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                stopped = subprocess.run(
                    ["bash", "-c", stop_command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
            finally:
                block.unlink(missing_ok=True)
                if owner.poll() is None:
                    os.killpg(owner.pid, signal.SIGKILL)
                    owner.wait(timeout=10)
                subprocess.run(
                    ["bash", "-c", stop_command],
                    env=environment,
                    capture_output=True,
                    timeout=10,
                )
                training_ops.rollout_pull_script = old_script

    def test_background_jobs_close_lock_fd_at_the_fork_boundary(self):
        source = puller.read_text()
        self.assertIn('"$part" 9>&- &', source)
        self.assertIn(") 9>&- &", source)
        self.assertIn("health_lease_loop' 9>&- &", source)
        self.assertIn('if [ "${ONCE:-0}" = 1 ]; then\n    refresh_health_lease', source)

    def test_health_lease_renews_during_blocked_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            (source / "rollouts").mkdir()
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                SSH_LISTING_DELAY="4",
                TRAINER_ADDRESSES="a b c d e f g h",
            )
            process = subprocess.Popen(
                ["bash", str(puller), "run"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                time.sleep(1)
                self.assertIsNone(process.poll())
                lease = source / ".trainer-health-lease"
                self.assertTrue(lease.is_file())
                health_status = source / ".trainer-health-status"
                self.assertTrue(health_status.read_text().startswith("healthy "))
                self.assertEqual(
                    (health_status.stat().st_uid, health_status.stat().st_gid, health_status.stat().st_mode & 0o777),
                    (1000, 1000, 0o600),
                )
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0, stderr or stdout)

    def test_missing_trainer_does_not_refresh_health_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            (source / "rollouts").mkdir()
            lease = source / ".trainer-health-lease"
            lease.write_text("old\n")
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                ONCE="1",
                SSH_MISSING_TRAINER="h",
                TRAINER_ADDRESSES="a b c d e f g h",
            )
            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(lease.read_text(), "old\n")
            self.assertTrue((source / ".trainer-health-status").read_text().startswith("missing "))

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

    def test_pull_part_timeout_is_bounded_and_retry_resumes_partial_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            rollout = source / "rollouts/step_1/train_rollouts.bin"
            rollout.parent.mkdir(parents=True)
            rollout.write_bytes(bytes(range(256)))
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                ONCE="1",
                PARALLEL_THRESHOLD="1",
                PARALLEL_STREAMS="2",
                PULL_PART_TIMEOUT="1",
                RSYNC_SLEEP="3",
                RSYNC_SLEEP_ONCE_MARKER=str(base / "rsync-sleep-once"),
            )

            started = time.monotonic()
            failed = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
            )
            elapsed = time.monotonic() - started
            self.assertNotEqual(failed.returncode, 0, failed.stderr)
            self.assertLess(elapsed, 5)
            partial_parts = destination / ".incoming/rollouts/step_1/train_rollouts.bin.parallel"
            self.assertTrue(partial_parts.is_dir())

            resumed = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual((destination / "rollouts/step_1/train_rollouts.bin").read_bytes(), rollout.read_bytes())

    def test_pull_part_timeout_validation_and_resume_flags(self):
        source = puller.read_text()
        self.assertIn("pull_part_timeout=${PULL_PART_TIMEOUT:-300}", source)
        self.assertIn(
            'timeout --signal=TERM --kill-after=10s "$pull_part_timeout" rsync "$@"',
            source,
        )
        self.assertEqual(
            source.count("pull_rsync -az --partial --append-verify --timeout=60"),
            2,
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_dir = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            environment = self.environment(source_dir, destination, self.tools(directory))
            for value in ("0", "-1", "not-an-integer", "3601"):
                environment["PULL_PART_TIMEOUT"] = value
                result = subprocess.run(
                    ["bash", str(puller), "run"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2, value)
                self.assertIn("PULL_PART_TIMEOUT", result.stderr)

    def test_failed_oldest_rollout_stops_scan_and_next_scan_retries_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            for step in (1, 2):
                rollout = source / f"rollouts/step_{step}/train_rollouts.bin"
                rollout.parent.mkdir(parents=True)
                rollout.write_bytes(bytes([step]) * 256)
            attempts = base / "rsync-attempts"
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                ONCE="1",
                PARALLEL_THRESHOLD="1",
                PARALLEL_STREAMS="2",
                RSYNC_ATTEMPTS_LOG=str(attempts),
                RSYNC_FAIL_SOURCE="step_1",
                RSYNC_FAIL_MARKER=str(base / "rsync-failed"),
            )

            failed = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
            )
            self.assertNotEqual(failed.returncode, 0, failed.stderr)
            first_scan = attempts.read_text().splitlines()
            self.assertTrue(first_scan)
            self.assertTrue(all("step_1" in attempt for attempt in first_scan))
            self.assertFalse((destination / "rollouts/step_2/train_rollouts.bin").exists())

            resumed = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertTrue(any("step_2" in attempt for attempt in attempts.read_text().splitlines()))
            for step in (1, 2):
                self.assertTrue((destination / f"rollouts/step_{step}/train_rollouts.bin").is_file())

    def test_rollout_steps_are_sorted_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            for step in (10, 2):
                rollout = source / f"rollouts/step_{step}/train_rollouts.bin"
                rollout.parent.mkdir(parents=True)
                rollout.write_bytes(bytes([step]) * 256)
            attempts = base / "rsync-attempts"
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(
                ONCE="1",
                PARALLEL_THRESHOLD="1",
                PARALLEL_STREAMS="1",
                RSYNC_ATTEMPTS_LOG=str(attempts),
            )

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=8,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            order = [
                2 if "step_2" in attempt else 10
                for attempt in attempts.read_text().splitlines()
                if "step_2" in attempt or "step_10" in attempt
            ]
            self.assertEqual(order, [2, 10])

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

    def test_remote_does_not_consume_stable_marker_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/alpha/run_alpha"
            destination = base / "destination/alpha/run_alpha"
            (source / "broadcasts").mkdir(parents=True)
            for step in range(1, 7):
                stable = destination / f"broadcasts/step_{step}/STABLE"
                stable.parent.mkdir(parents=True)
                stable.write_bytes(b"stable")
            environment = self.environment(source, destination, self.tools(directory))
            environment.update(ONCE="1", SSH_READ_STDIN="1")

            result = subprocess.run(
                ["bash", str(puller), "run"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for step in range(1, 7):
                self.assertEqual(
                    (source / f"broadcasts/step_{step}/STABLE").read_bytes(),
                    b"stable",
                )


if __name__ == "__main__":
    unittest.main()
