import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
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

    def child(self, *, identity=True, publish=False):
        ready = self.active / "ready"
        code = r'''
import fcntl,os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1])
fd=os.open(root/'.batch-sync.lock',os.O_CREAT|os.O_RDWR,0o600)
os.dup2(fd,9)
if fd!=9: os.close(fd)
fcntl.flock(9,fcntl.LOCK_EX)
if os.fork(): os._exit(0)
def finish(*args):
 os.close(9)
 time.sleep(0.25)
 if sys.argv[2]=='publish':
  target=root/'rollouts/step_4/train_rollouts.bin'
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(b'old in-flight publication')
 os._exit(0)
signal.signal(signal.SIGTERM,finish)
(root/'ready').write_text(str(os.getpid()))
while True: time.sleep(0.1)
'''
        process = subprocess.Popen([sys.executable, "-c", code, str(self.active), "publish" if publish else "idle"], start_new_session=True, env=dict(os.environ, SOURCE_RUN_DIR=str(self.active) if identity else "/unrelated"))
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


if __name__ == "__main__":
    unittest.main()
