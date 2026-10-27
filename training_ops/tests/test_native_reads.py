import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import training_ops
from training_ops.runtime import watch_native_rollouts as viewer


class NativeReadsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name="training", started=1, command=None):
        root = self.root / name
        ops = root / "launcher/ops"
        ops.mkdir(parents=True)
        ticks = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
        state = {"pid": os.getpid(), "start_ticks": ticks, "run": name, "started_at": started}
        (ops / "process.json").write_text(json.dumps(state))
        (ops / "manifest.json").write_text(json.dumps({"run": name, "output_dir": str(root),
            "command": command or ["/bin/bash", "-c", "exec /usr/bin/uv run --no-sync rl @ config.toml"]}))
        log = root / "logs/attempt_1/orchestrator.log"
        log.parent.mkdir(parents=True)
        log.write_text('{"time":"2026-10-27 12:00:00","level":"INFO","message":"Train batch 3/24"}\n')
        return root, state

    def metrics(self, root, rows):
        path = root / "monitors/file/metrics.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def report(self, root, state):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            viewer.status_report(root, state)
        return output.getvalue()

    def test_selects_live_training_not_newer_services_or_stale_pid(self):
        training, _ = self.fixture()
        self.fixture("router", started=5, command=["/bin/bash", "-c", "exec vllm-router --port 8000"])
        self.fixture("engine", started=6, command=["uv", "run", "inference", "@", "engine.toml"])
        old, state = self.fixture("stale", started=9)
        state["start_ticks"] = "wrong"
        (old / "launcher/ops/process.json").write_text(json.dumps(state))
        self.assertEqual(viewer.select_run(self.root)[0], training)

    def test_new_run_name_is_selected_without_code_change(self):
        self.fixture("first", started=1)
        latest, _ = self.fixture("replacement", started=2)
        self.assertEqual(viewer.select_run(self.root)[0], latest)

    def test_no_live_run_fails_without_saved_run_fallback(self):
        root, state = self.fixture()
        state["start_ticks"] = "wrong"
        (root / "launcher/ops/process.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "No live training"):
            viewer.select_run(self.root)
        self.assertEqual(viewer.select_run(self.root, root.name)[0], root)

    def test_json_log_format_and_latest_attempt(self):
        root, state = self.fixture()
        path = root / "logs/attempt_10/orchestrator.log"
        path.parent.mkdir()
        path.write_text(json.dumps({"record": {"time": {"repr": "2026-10-27 13:00:00"},
            "level": {"name": "SUCCESS"}, "message": "Step 2"}}) + "\n")
        result = subprocess.run([sys.executable, "-c", Path(viewer.__file__).read_text(),
            "--root", str(self.root), "--no-follow"], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2026-10-27 13:00:00 SUCCESS Step 2", result.stdout)
        self.assertNotIn('"record"', result.stdout)
        self.assertNotIn("Train batch 3/24", result.stdout)

    def test_stream_reads_new_line_and_reports_process_failure(self):
        root, state = self.fixture()
        path = viewer.log_path(root)
        (root / "launcher/ops/result.json").write_text('{"returncode": 7}')
        def append(_):
            with path.open("a") as stream:
                stream.write('{"time":"later","message":"Train batch 5/24"}\n')
        output = io.StringIO()
        with mock.patch.object(viewer, "alive", side_effect=[True, False]), mock.patch.object(
            viewer.time, "sleep", side_effect=append
        ), contextlib.redirect_stdout(output):
            self.assertEqual(viewer.watch(root, state), 7)
        self.assertIn("later Train batch 5/24", output.getvalue())
        self.assertIn("exit code 7", output.getvalue())

    def test_counts_unique_completed_optimizer_updates_not_policy_or_row_count(self):
        root, state = self.fixture()
        self.metrics(root, [
            {"producer": "trainer", "step": 1, "time/forward_backward": 2.0},
            {"producer": "trainer", "step": 1, "optim/grad_norm": 0.1},
            {"producer": "trainer", "step": 2, "time/forward_backward": 3.0},
            {"producer": "trainer", "step": 3, "diagnostic/parity_guard_rejected": 1},
            {"producer": "orch", "step": 9, "time/step": 4.0},
        ])
        report = self.report(root, state)
        self.assertIn("Completed optimizer updates: 2", report)
        self.assertIn("Latest completed trainer step: 2", report)
        self.assertIn("Batch progress: 3/24", report)
        self.assertIn("Rejected parity steps in read window: 3", report)

    def test_inference_only_tail_cannot_erase_a_completed_update(self):
        root, state = self.fixture()
        path = self.metrics(root, [{"producer": "trainer", "step": 1, "time/forward_backward": 1.0}])
        with path.open("a") as stream:
            for _ in range(5000):
                stream.write(json.dumps({"producer": "orch", "step": None, "inference/padding": "x" * 1024}) + "\n")
        report = self.report(root, state)
        self.assertIn("Completed optimizer updates: unknown", report)
        self.assertNotIn("Completed optimizer updates: 0", report)

    def test_missing_metrics_are_unknown(self):
        root, state = self.fixture()
        report = self.report(root, state)
        self.assertIn("Completed optimizer updates: unknown", report)
        self.assertIn("Gradient norm: unknown", report)

    def test_kl_labels_use_distinct_step25_metric_fields(self):
        root, state = self.fixture()
        self.metrics(root, [{"producer": "trainer", "step": 25,
                            "masked_mismatch_kl/mean": 0.007590143010020256,
                            "mismatch_kl/all/mean": 0.0007629550527781248,
                            "unmasked_mismatch_kl/mean": 0.0007671326166018844}])
        report = self.report(root, state)
        self.assertIn("Masked KL: 0.0075901", report)
        self.assertIn("All-token KL: 0.00076296", report)
        self.assertIn("Unmasked KL: 0.00076713", report)

    def test_failed_episodes_are_excluded_and_group_failures_are_separate(self):
        root, state = self.fixture()
        index = root / "monitors/file/traces/stream.index.jsonl"
        index.parent.mkdir(parents=True)
        rows = [
            {"kind": "train", "env": "terminal", "group": "one", "ok": False, "reward": 1, "num_errors": 0},
            {"kind": "train", "env": "terminal", "group": "one", "ok": False, "reward": 1, "num_errors": 0},
            {"kind": "train", "env": "terminal", "group": "two", "ok": True, "reward": 0.25},
            {"kind": "train", "env": "terminal", "group": "three", "ok": True, "reward": float("nan")},
            {"kind": "eval", "env": "terminal", "group": "one", "ok": True, "reward": 0.75},
        ]
        index.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = self.report(root, state)
        self.assertIn("train/terminal: mean 0.25 from 1 valid scores; failed episodes 2; groups with failed episodes 1; unknown scores 1", report)
        self.assertIn("eval/terminal: mean 0.75 from 1 valid scores", report)
        self.assertIn("Learning trend: unknown", report)


class NativeRoutingTest(unittest.TestCase):
    def invoke(self, action, arguments=None):
        with mock.patch.object(training_ops.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops._native_read_action(arguments or [], action)
        return run.call_args

    def test_status_logs_and_watch_share_active_node_without_legacy_probes(self):
        for action in ("status", "logs", "watch-rollouts"):
            with self.subTest(action=action), mock.patch.object(training_ops, "status") as legacy:
                call = self.invoke(action)
                argv = call.args[0]
                self.assertIn("root@192.222.55.85", argv)
                self.assertIn("ConnectTimeout=5", argv)
                self.assertNotIn("ConnectTimeout=25", argv)
                self.assertIn("--root", shlex.split(argv[-1]))
                legacy.assert_not_called()
                self.assertEqual(call.kwargs["timeout"], 610 if action == "watch-rollouts" else 15)

    def test_command_timeout_has_readable_failure(self):
        with mock.patch.object(training_ops.subprocess, "run", side_effect=subprocess.TimeoutExpired("ssh", 15)):
            with self.assertRaisesRegex(RuntimeError, "15 second deadline"), contextlib.redirect_stdout(io.StringIO()):
                training_ops._native_read_action([], "status")

    def test_invalid_arguments_fail_before_remote_call(self):
        for args in (["--run", "../bad"], ["--lines", "0"], ["--seconds", "601"]):
            with self.subTest(args=args), mock.patch.object(training_ops.subprocess, "run") as run:
                with self.assertRaises(training_ops.Refused):
                    training_ops._native_read_action(args, "status")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
