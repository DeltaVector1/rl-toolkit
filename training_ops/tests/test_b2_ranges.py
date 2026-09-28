import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import training_ops
from training_ops.runtime.h200 import b2_rollout as tool
import training_ops.tests  # noqa: F401 - local-only network guard
from training_ops.tests import test_b2_rollout as fixtures


fake_curl = r"""
import json,os,signal,sys,time
from pathlib import Path
case,source=sys.argv[1:3]
args=sys.argv[3:]
config=sys.stdin.read()
assert config.startswith('url = "https://') and 'temporary' in config
start,end=map(int,args[args.index('--range')+1].split('-'))
if case=='slow' or (case=='fail' and start):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    time.sleep(20)
if case=='straggler' and start==0:
    marker=Path(str(Path(source).with_suffix('.straggler')))
    if not marker.exists():
        body=Path(source).read_bytes()[start:end+1]
        Path(args[args.index('--output')+1]).write_bytes(body[:len(body)//2])
        marker.write_text('failed')
        sys.exit(28)
if case=='fail': sys.exit(22)
body=Path(source).read_bytes()[start:end+1]
if case=='short': body=body[:-1]
if case=='hash': body=bytes(len(body))
Path(args[args.index('--output')+1]).write_bytes(body)
status='200' if case=='status' else '206'
value=f'bytes {start}-{end}/{Path(source).stat().st_size}'
if case=='header': value='bytes 0-0/1'
header=f'HTTP/1.1 {status} OK\r\nContent-Range: {value}\r\n\r\n'
Path(args[args.index('--dump-header')+1]).write_bytes(header.encode())
print(f'{status}\t192.0.2.1\t1.1\t0.01\t0.02\t0.03\t{len(body)}',end='')
"""


class RangeLayoutTest(unittest.TestCase):
    def test_exact_uneven_and_small_ranges(self):
        self.assertEqual(tool.download_ranges(10, 3), [(0, 3), (4, 6), (7, 9)])
        self.assertEqual(tool.download_ranges(2, 8), [(0, 0), (1, 1)])
        for size, workers in [(0, 8), (1, 0), (1, 33), (True, 2), (2, True)]:
            with self.subTest(size=size, workers=workers), self.assertRaises(
                ValueError
            ):
                tool.download_ranges(size, workers)

    def test_overlap_gap_and_wrong_end_fail(self):
        for ranges in [[(0, 5), (5, 9)], [(0, 3), (5, 9)], [(0, 8)], []]:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                tool.validate_ranges(ranges, 10)

    def test_strict_settings_and_environment(self):
        self.assertEqual(training_ops._rollout_download_settings({}), (1, 67108864))
        for value in [True, 0, 33, 8.0, "8"]:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                training_ops._rollout_download_settings({"download_workers": value})
        for settings in [
            {"typo": 8},
            {"download_threshold_bytes": False},
            {"download_threshold_bytes": 0},
        ]:
            with self.assertRaises(RuntimeError):
                training_ops._rollout_download_settings(settings)
        with patch.object(training_ops, "rollout_download_workers", 8):
            env = training_ops._rollout_pull_environment("/data/fixture")
            self.assertEqual(env["B2_DOWNLOAD_WORKERS"], "8")
            self.assertEqual(env["B2_DOWNLOAD_THRESHOLD_BYTES"], "67108864")


class RangeReceiveTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.B2RolloutTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.value = self.fixture.publish()
        self.children = []
        self.commands = []
        self.real_popen = subprocess.Popen
        self.case = "ok"

    def spawn(self, command, **kwargs):
        self.commands.append(command)
        self.assertEqual(command[:2], ["curl", "--disable"])
        self.assertEqual(command[command.index("--retry") + 1], "0")
        self.assertEqual(command[command.index("--speed-time") + 1], "60")
        self.assertNotIn("--location", command)
        self.assertNotIn("temporary", str(command))
        self.assertTrue(kwargs["start_new_session"])
        child = self.real_popen(
            [
                sys.executable,
                "-c",
                fake_curl,
                self.case,
                str(self.fixture.source),
                *command[1:],
            ],
            **kwargs,
        )
        self.children.append(child)
        return child

    def receive(self):
        fixture = self.fixture
        with patch.object(tool.subprocess, "Popen", self.spawn):
            tool.receive(
                self.value,
                str(fixture.source),
                fixture.partial,
                fixture.final,
                download_workers=8,
                download_threshold_bytes=1,
            )

    def test_parallel_exact_publish_ack_cleanup(self):
        fixture = self.fixture
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.receive()
        self.assertEqual(len(self.children), 8)
        self.assertTrue(all(child.poll() == 0 for child in self.children))
        self.assertNotIn("temporary", log.getvalue())
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(sum(record["bytes"] for record in records), len(fixture.body))
        self.assertEqual(fixture.final.read_bytes(), fixture.body)
        self.assertEqual(fixture.final.stat().st_uid, fixture.source.stat().st_uid)
        parts = Path(str(fixture.final) + ".b2-parts")
        self.assertEqual(parts.stat().st_mode & 0o777, 0o700)
        self.assertEqual(len(list(parts.iterdir())), 24)
        pending = Path(str(fixture.final) + ".b2-ack.json")
        receipt = tool.parse_json(pending.read_text())
        Path(str(fixture.source) + ".delivered").write_text(receipt["sha256"])
        tool.acknowledge(fixture.client, "fixture", receipt, fixture.runs)
        self.assertEqual(fixture.client.deletes[0]["VersionId"], receipt["version_id"])
        tool.cleanup_parts(fixture.final, receipt)
        self.assertFalse(parts.exists())

    def test_parallel_root_publication_is_readable_by_trainer(self):
        fixture = self.fixture
        fixture.root.chmod(0o755)
        fixture.source.chmod(0o600)
        if os.geteuid() == 0:
            os.chown(fixture.source, 1000, 1000)
        self.value = fixture.publish()
        with contextlib.redirect_stderr(io.StringIO()):
            self.receive()
        self.assertEqual(fixture.final.stat().st_uid, fixture.source.stat().st_uid)
        self.assertEqual(fixture.final.parent.stat().st_mode & 0o777, 0o755)
        if os.geteuid() == 0:
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; from pathlib import Path; os.setgroups([]); os.setgid(1000); os.setuid(1000); print(len(Path(sys.argv[1]).read_bytes()))",
                    str(fixture.final),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(int(child.stdout), len(fixture.body))

    def test_faults_preserve_evidence_without_publication(self):
        for case in ["status", "header", "short", "hash"]:
            with self.subTest(case=case):
                self.case = case
                with self.assertRaises(ValueError):
                    self.receive()
                self.assertFalse(self.fixture.final.exists())
                self.assertTrue(
                    all(child.poll() is not None for child in self.children)
                )
                # Separate fresh fixture per attempt; never retry a retained directory.
                self.fixture.doCleanups()
                self.fixture.setUp()
                self.value = self.fixture.publish()

    def test_one_straggler_reconnects_only_the_failed_range(self):
        self.case = "straggler"
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.receive()
        self.assertEqual(self.fixture.final.read_bytes(), self.fixture.body)
        ranges = [
            command[command.index("--range") + 1]
            for command in self.commands
        ]
        self.assertEqual(len(ranges), 9)
        counts = {value: ranges.count(value) for value in set(ranges)}
        self.assertEqual(sorted(counts.values()), [1] * 7 + [2])
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(
            [record for record in records if record.get("retry")],
            [{"range_index": 0, "attempt": 1, "retry": True}],
        )
        self.assertFalse(any(record.get("failed") for record in records))

    def test_no_retry_after_shared_deadline(self):
        fixture = self.fixture
        fixture.final.parent.mkdir(parents=True)
        fixture.partial.parent.mkdir(parents=True)
        self.case = "straggler"
        ticks = iter([0] + [0] * 8 + [300])
        with patch.object(tool.subprocess, "Popen", self.spawn), self.assertRaises(
            TimeoutError
        ):
            tool.receive_ranges(
                self.value["url"],
                len(fixture.body),
                fixture.partial,
                fixture.final,
                8,
                clock=lambda: next(ticks),
            )
        self.assertEqual(len(self.commands), 8)
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_failure_kills_and_reaps_hung_siblings(self):
        self.case = "fail"
        started = time.monotonic()
        with self.assertRaises(ValueError):
            self.receive()
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(len(self.children), 8)
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_shared_deadline_reduces_each_child_budget(self):
        fixture = self.fixture
        fixture.final.parent.mkdir(parents=True)
        fixture.partial.parent.mkdir(parents=True)
        self.case = "slow"
        ticks = iter([0, 0, 100, 200, 301])
        with patch.object(tool.subprocess, "Popen", self.spawn), self.assertRaises(
            TimeoutError
        ):
            tool.receive_ranges(
                self.value["url"],
                len(fixture.body),
                fixture.partial,
                fixture.final,
                8,
                clock=lambda: next(ticks),
            )
        self.assertEqual(
            [
                float(command[command.index("--max-time") + 1])
                for command in self.commands
            ],
            [300, 200, 100],
        )
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_existing_parts_symlink_and_foreign_entries_refused(self):
        fixture = self.fixture
        fixture.final.parent.mkdir(parents=True)
        parts = Path(str(fixture.final) + ".b2-parts")
        parts.symlink_to(fixture.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.receive()
        self.assertEqual(self.children, [])

    def test_external_term_reaps_owned_curl_groups(self):
        self.external_term(False)

    def test_repeated_term_and_alarm_during_cleanup_reap_owned_groups(self):
        self.external_term(True)

    def test_signal_between_spawn_and_registration_reaps_child(self):
        original = self.spawn

        def interrupted_spawn(command, **kwargs):
            child = original(command, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        self.case = "slow"
        fixture = self.fixture
        with patch.object(
            tool.subprocess, "Popen", interrupted_spawn
        ), self.assertRaises(InterruptedError):
            tool.receive(
                self.value,
                str(fixture.source),
                fixture.partial,
                fixture.final,
                download_workers=8,
                download_threshold_bytes=1,
            )
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())

    def test_below_threshold_preserves_single_download(self):
        fixture = self.fixture
        with patch.object(
            tool.subprocess,
            "Popen",
            side_effect=AssertionError("unexpected parallel download"),
        ):
            tool.receive(
                self.value,
                str(fixture.source),
                fixture.partial,
                fixture.final,
                fixture.curl,
                download_workers=8,
            )
        self.assertEqual(fixture.final.read_bytes(), fixture.body)

    def external_term(self, repeat):
        fixture = self.fixture
        harness = r"""
import json,os,sys,time
from pathlib import Path
from training_ops.runtime.h200 import b2_rollout as tool
original=tool.subprocess.Popen
children=[]
def spawn(command, **kwargs):
 child=original([sys.executable,'-c',sys.argv[3],'slow',sys.argv[2],*command[1:]],**kwargs)
 children.append(child.pid)
 Path(sys.argv[1]).write_text(json.dumps(children))
 return child
tool.subprocess.Popen=spawn
original_stop=tool.stop_downloads
def stop(children):
 Path(sys.argv[1]+'.cleanup').touch()
 original_stop(children)
tool.stop_downloads=stop
value=json.loads(sys.stdin.read())
tool.receive(value,value['source'],sys.argv[4],sys.argv[5],download_workers=8,download_threshold_bytes=1)
"""
        pids = fixture.root / "owned-pids"
        process = self.real_popen(
            [
                sys.executable,
                "-c",
                harness,
                str(pids),
                str(fixture.source),
                fake_curl,
                str(fixture.partial),
                str(fixture.final),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            process.stdin.write(json.dumps(self.value))
            process.stdin.close()
            until = time.monotonic() + 5
            owned = []
            while time.monotonic() < until:
                try:
                    owned = json.loads(pids.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if len(owned) == 8:
                    break
                time.sleep(0.02)
            self.assertEqual(len(owned), 8)
            time.sleep(0.15)  # Let fake workers install their TERM-ignore handler.
            process.send_signal(signal.SIGTERM)
            if repeat:
                until = time.monotonic() + 2
                while (
                    not Path(str(pids) + ".cleanup").exists()
                    and time.monotonic() < until
                ):
                    time.sleep(0.01)
                self.assertTrue(Path(str(pids) + ".cleanup").exists())
                process.send_signal(signal.SIGTERM)
                process.send_signal(signal.SIGALRM)
            self.assertNotEqual(process.wait(timeout=4), 0)
            for pid in owned:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class DaemonSettingsTest(unittest.TestCase):
    def test_stop_owns_legacy_and_different_worker_settings(self):
        fixture = fixtures.RolloutPullTest()
        for old_settings in [
            {},
            {"B2_DOWNLOAD_WORKERS": "2", "B2_DOWNLOAD_THRESHOLD_BYTES": "128"},
        ]:
            with self.subTest(
                old_settings=old_settings
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, destination = root / "source", root / "destination"
                source.mkdir()
                destination.mkdir()
                env = fixture.environment(source, destination, fixture.tools(directory))
                env.update(
                    ROLLOUT_TRANSFER_BACKEND="b2_s3",
                    B2_ROLLOUT_PYTHON="/usr/bin/true",
                    INTERVAL="1",
                    **old_settings,
                )
                process = subprocess.Popen(
                    ["bash", str(fixtures.puller), "run"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    until = time.monotonic() + 5
                    while (
                        not (destination / ".rollout-pull.pid").exists()
                        and time.monotonic() < until
                    ):
                        time.sleep(0.02)
                    self.assertTrue((destination / ".rollout-pull.pid").exists())
                    desired = dict(
                        env,
                        B2_DOWNLOAD_WORKERS="8",
                        B2_DOWNLOAD_THRESHOLD_BYTES="67108864",
                    )
                    status = subprocess.run(
                        ["bash", str(fixtures.puller), "status"],
                        env=desired,
                        capture_output=True,
                        timeout=5,
                    )
                    self.assertNotEqual(status.returncode, 0)
                    stopped = subprocess.run(
                        ["bash", str(fixtures.puller), "stop"],
                        env=desired,
                        capture_output=True,
                        timeout=15,
                    )
                    self.assertEqual(stopped.returncode, 0, stopped.stderr)
                    process.wait(timeout=3)
                    self.assertFalse((destination / ".rollout-pull.pid").exists())
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
