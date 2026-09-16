import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

spec = importlib.util.spec_from_file_location("publisher_drain", Path(__file__).parents[1] / "runtime/h200/drain_rollout_publisher.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublisherDrainTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.active = Path(self.temporary.name) / "alpha/run_alpha"
        self.active.mkdir(parents=True)
        self.script = str(Path(self.temporary.name) / "sync_rollout_batches.sh")

    def child(self, *, identity=True, publish=False, microbatch=False, parent=False, finish_delay=0.25):
        directory = self.active.parent if parent else self.active
        ready = directory / "ready"
        code = r'''
import fcntl,os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1])
fd=os.open(root/sys.argv[3],os.O_CREAT|os.O_RDWR,0o600)
os.dup2(fd,9)
if fd!=9: os.close(fd)
fcntl.flock(9,fcntl.LOCK_EX)
if os.fork(): os._exit(0)
def finish(*args):
 os.close(9)
 time.sleep(float(sys.argv[5]))
 if sys.argv[2]=='publish':
  target=root/'rollouts/step_4'/sys.argv[4]
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(b'old in-flight publication')
 os._exit(0)
signal.signal(signal.SIGTERM,finish)
(root/'ready').write_text(str(os.getpid()))
while True: time.sleep(0.1)
'''
        environment = "run_dir" if microbatch else "SOURCE_RUN_DIR"
        lock = ".microbatch-sync.lock" if microbatch else ".batch-sync.lock"
        filename = "rank_0.bin" if microbatch else "train_rollouts.bin"
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(directory), "publish" if publish else "idle", lock, filename, str(finish_delay)],
            start_new_session=True,
            env={**os.environ, environment: str(directory) if identity else "/unrelated"},
        )
        process.wait(timeout=3)
        def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists())
        return int(ready.read_text())

    def test_orphan_child_is_drained_even_after_lock_release(self):
        pid = self.child(publish=True)
        started = time.monotonic()
        publisher.drain(str(self.active), self.script, timeout=3)
        self.assertGreaterEqual(time.monotonic() - started, 0.2)
        state = publisher.process(pid)
        self.assertTrue(state is None or state[1] == "Z")
        # The following resume scan sees the completed old publication. It
        # cannot arrive later, after the trainer has passed its cache check.
        self.assertTrue((self.active / "rollouts/step_4/train_rollouts.bin").is_file())

    def test_unknown_lock_holder_fails_closed_and_is_not_killed(self):
        pid = self.child(identity=False)
        with self.assertRaises(TimeoutError):
            publisher.drain(str(self.active), self.script, timeout=0.2)
        self.assertNotEqual(publisher.process(pid)[1], "Z")

    def test_stale_pid_file_never_targets_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait(timeout=3)))
        (self.active / ".batch-sync.pid").write_text(str(unrelated.pid))
        publisher.drain(str(self.active), self.script, timeout=0.2)
        self.assertIsNone(unrelated.poll())

    def test_microbatch_orphan_finishes_before_resume_scan_in_both_run_paths(self):
        for parent in (False, True):
            with self.subTest(parent=parent):
                directory = self.active.parent if parent else self.active
                pid = self.child(microbatch=True, parent=parent, publish=True)
                started = time.monotonic()
                publisher.drain(str(self.active), self.script, timeout=3, microbatch=True)
                self.assertGreaterEqual(time.monotonic() - started, 0.2)
                state = publisher.process(pid)
                self.assertTrue(state is None or state[1] == "Z")
                # A late SIGTERM handler deliberately closes its inherited
                # lock before publishing. Resume must see that stale rank
                # file now, not after its final cached-batch scan.
                cached = directory / "rollouts/step_4/rank_0.bin"
                self.assertEqual(cached.read_bytes(), b"old in-flight publication")

    def test_microbatch_unknown_lock_holder_fails_closed_without_a_signal(self):
        pid = self.child(microbatch=True, identity=False)
        with self.assertRaises(TimeoutError):
            publisher.drain(str(self.active), self.script, timeout=0.2, microbatch=True)
        self.assertNotEqual(publisher.process(pid)[1], "Z")

    def test_microbatch_pre_signalled_orphan_cannot_publish_after_drain(self):
        for parent in (False, True):
            with self.subTest(parent=parent):
                directory = self.active.parent if parent else self.active
                pid = self.child(microbatch=True, parent=parent, publish=True, finish_delay=0.6)
                os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while (Path("/proc") / str(pid) / "fd/9").exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertFalse((Path("/proc") / str(pid) / "fd/9").exists())
                self.assertNotEqual(publisher.process(pid)[1], "Z")
                with mock.patch.object(publisher.os, "kill") as individual_signal, mock.patch.object(publisher.os, "killpg") as group_signal:
                    publisher.drain(str(self.active), self.script, timeout=3, microbatch=True)
                    individual_signal.assert_not_called()
                    group_signal.assert_not_called()
                state = publisher.process(pid)
                self.assertTrue(state is None or state[1] == "Z")
                cached = directory / "rollouts/step_4/rank_0.bin"
                self.assertEqual(cached.read_bytes(), b"old in-flight publication")
                # The final resume scan can now see and quarantine this file.
                # No child is left that could refill the path after the scan.
                cached.rename(cached.with_suffix(".quarantined"))
                time.sleep(0.1)
                self.assertFalse(cached.exists())

    def test_same_run_environment_without_ownership_refuses_without_a_signal(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
            env={**os.environ, "run_dir": str(self.active)},
        )
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait(timeout=3)))
        with mock.patch.object(publisher.os, "kill") as individual_signal, mock.patch.object(publisher.os, "killpg") as group_signal:
            with self.assertRaises(TimeoutError):
                publisher.drain(str(self.active), self.script, timeout=0.2, microbatch=True)
            individual_signal.assert_not_called()
            group_signal.assert_not_called()
        self.assertIsNone(unrelated.poll())

    def test_microbatch_stale_pid_does_not_target_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait(timeout=3)))
        for directory in (self.active.parent, self.active):
            (directory / ".microbatch-sync.pid").write_text(str(unrelated.pid))
        publisher.drain(str(self.active), self.script, timeout=0.2, microbatch=True)
        self.assertIsNone(unrelated.poll())

    def test_matching_child_cannot_authorize_unrelated_group_members(self):
        group = 987650
        records = {
            group: (Path(f"/proc/{group}"), "S", group, "100"),
            group + 1: (Path(f"/proc/{group + 1}"), "S", group, "101"),
            group + 2: (Path(f"/proc/{group + 2}"), "S", group, "102"),
        }
        for unrelated in (group, group + 2):
            with self.subTest(unrelated=unrelated), mock.patch.object(publisher, "process", side_effect=records.get), mock.patch.object(publisher, "matches", side_effect=lambda pid, *_args: pid != unrelated), mock.patch.object(publisher, "snapshot", return_value=records), mock.patch.object(publisher.os, "killpg") as group_signal, mock.patch.object(publisher.os, "kill") as individual_signal:
                with self.assertRaisesRegex(ValueError, "unknown (member|leader)"):
                    publisher.signal_publishers(records, self.active, self.script, (1, 2), set())
                group_signal.assert_not_called()
                individual_signal.assert_not_called()

    def test_all_proven_members_allow_only_the_exact_group_signal(self):
        group = 987650
        records = {pid: (Path(f"/proc/{pid}"), "S", group, str(pid)) for pid in (group, group + 1)}
        stopped = set()
        with mock.patch.object(publisher, "process", side_effect=records.get), mock.patch.object(publisher, "matches", return_value=True), mock.patch.object(publisher, "snapshot", return_value=records), mock.patch.object(publisher.os, "killpg") as group_signal, mock.patch.object(publisher.os, "kill") as individual_signal:
            publisher.signal_publishers(records, self.active, self.script, (1, 2), stopped)
        group_signal.assert_called_once_with(group, signal.SIGTERM)
        individual_signal.assert_not_called()
        self.assertEqual(stopped, {group})

    def test_orphan_signals_only_rechecked_proven_pids(self):
        group = 987650
        pid = group + 1
        records = {pid: (Path(f"/proc/{pid}"), "S", group, "101")}
        with mock.patch.object(publisher, "process", side_effect=records.get), mock.patch.object(publisher, "matches", return_value=True), mock.patch.object(publisher, "snapshot", return_value=records), mock.patch.object(publisher.os, "killpg") as group_signal, mock.patch.object(publisher.os, "kill") as individual_signal:
            publisher.signal_publishers(records, self.active, self.script, (1, 2), set())
        group_signal.assert_not_called()
        individual_signal.assert_called_once_with(pid, signal.SIGTERM)

    def test_mixed_orphan_group_signals_nobody(self):
        group = 987650
        records = {pid: (Path(f"/proc/{pid}"), "S", group, str(pid)) for pid in (group + 1, group + 2)}
        with mock.patch.object(publisher, "process", side_effect=records.get), mock.patch.object(publisher, "matches", side_effect=lambda pid, *_args: pid == group + 1), mock.patch.object(publisher, "snapshot", return_value=records), mock.patch.object(publisher.os, "killpg") as group_signal, mock.patch.object(publisher.os, "kill") as individual_signal:
            with self.assertRaisesRegex(ValueError, "unknown member"):
                publisher.signal_publishers(records, self.active, self.script, (1, 2), set())
        group_signal.assert_not_called()
        individual_signal.assert_not_called()

    def test_orphan_start_time_change_signals_nobody(self):
        group, pid = 987650, 987651
        record = (Path(f"/proc/{pid}"), "S", group, "101")
        changed = (*record[:3], "102")
        records = {pid: record}
        with mock.patch.object(publisher, "process", side_effect=[record, record, changed]), mock.patch.object(publisher, "matches", return_value=True), mock.patch.object(publisher, "snapshot", return_value=records), mock.patch.object(publisher.os, "killpg") as group_signal, mock.patch.object(publisher.os, "kill") as individual_signal:
            with self.assertRaisesRegex(ValueError, "changed before signalling"):
                publisher.signal_publishers(records, self.active, self.script, (1, 2), set())
        group_signal.assert_not_called()
        individual_signal.assert_not_called()

    def test_launcher_drain_failure_exits_before_replacement_start(self):
        source = (Path(__file__).parents[1] / "runtime/h200/start_orchestrator.sh").read_text()
        block = source.split('  publisher_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"', 1)[1].split("  publisher_pid=$!", 1)[0]
        # Preserve the exact drain command and failure operator from the
        # launcher; replace only the background launch with a marker.
        drain = block.split("  nohup setsid", 1)[0]
        program = 'deployment=/unused; publisher_dir=/unused\npython3() { return 19; }\n' + drain + '\necho replacement-started\n'
        result = subprocess.run(["bash", "-uc", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("replacement-started", result.stdout)


if __name__ == "__main__":
    unittest.main()
