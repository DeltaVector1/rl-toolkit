import contextlib
import io
import json
import sys
import unittest
from unittest import mock

import training_ops


class CliTest(unittest.TestCase):
    @staticmethod
    def _rollout_result(host, *, apply=False, step=1, all_steps=None):
        candidate = {
            "step": step,
            "bytes": 4,
            "sha256": "a" * 64,
            "path": f"/tmp/run_demo/rollouts/step_{step}",
            "size": 4,
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
            "dir_device": 1,
            "dir_inode": 4,
            "dir_mtime_ns": 5,
            "dir_uid": 1000,
            "dir_gid": 1000,
        }
        result = {
            "node": host,
            "run_name": "demo",
            "run_dir": "/tmp/run_demo",
            "active_step": 4,
            "all_steps": all_steps or [0, 1, 2, 3, 4],
            "selected_steps": [step],
            "selected_bytes": 4,
            "selected": [candidate],
        }
        if apply:
            result.update({"removed": [candidate], "freed_bytes": 4})
        return result

    def test_stop_rejects_extra_arguments_without_stopping(self):
        for argument in ("--help", "unexpected"):
            with self.subTest(argument=argument), mock.patch.object(training_ops, "stop") as stop:
                with mock.patch.object(sys, "argv", ["training-ops", "stop", argument]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exited:
                        training_ops.cli()
                self.assertEqual(exited.exception.code, 2)
                stop.assert_not_called()

    def test_bare_stop_stops_once(self):
        with mock.patch.object(training_ops, "stop") as stop, mock.patch.object(
            sys, "argv", ["training-ops", "stop"]
        ):
            training_ops.cli()
        stop.assert_called_once_with()

    def test_retention_defaults_to_dry_run(self):
        with mock.patch.object(
            training_ops,
            "_broadcast_retention_cleanup",
            return_value={"dry_run": True},
        ) as cleanup, mock.patch.object(
            sys,
            "argv",
            ["training-ops", "retention", "/tmp/run_demo", "--evidence", "/tmp/evidence.json"],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        cleanup.assert_called_once_with(
            "/tmp/run_demo",
            "/tmp/evidence.json",
            keep_latest=2,
            partial_age_seconds=1800.0,
            max_entries=8,
            max_bytes=50 * 1024**3,
            dry_run=True,
        )

    def test_retention_apply_parses_bounded_options(self):
        with mock.patch.object(
            training_ops,
            "_broadcast_retention_cleanup",
            return_value={"dry_run": False},
        ) as cleanup, mock.patch.object(
            sys,
            "argv",
            [
                "training-ops", "retention", "/tmp/run_demo", "--evidence", "/tmp/evidence.json",
                "--apply", "--keep-latest", "3", "--partial-age-seconds", "60",
                "--max-entries", "2", "--max-bytes", "1024",
            ],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        cleanup.assert_called_once_with(
            "/tmp/run_demo",
            "/tmp/evidence.json",
            keep_latest=3,
            partial_age_seconds=60.0,
            max_entries=2,
            max_bytes=1024,
            dry_run=False,
        )

    def test_retention_rejects_missing_evidence_without_calling_cleanup(self):
        with mock.patch.object(training_ops, "_broadcast_retention_cleanup") as cleanup, mock.patch.object(
            sys, "argv", ["training-ops", "retention", "/tmp/run_demo"]
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        cleanup.assert_not_called()

    def test_rollout_retention_dry_run_preflights_all_trainers(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1,
            "run_name": "demo",
            "active_step": 4,
            "required_steps": [3, 4],
            "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            return {host: (0, json.dumps(self._rollout_result(host))) for host in hosts}

        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json"]):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        fan.assert_called_once()

    def test_rollout_retention_apply_runs_preflight_then_apply(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1,
            "run_name": "demo",
            "active_step": 4,
            "required_steps": [3, 4],
            "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            apply = fake_fan.calls > 0
            fake_fan.calls += 1
            return {host: (0, json.dumps(self._rollout_result(host, apply=apply))) for host in hosts}
        fake_fan.calls = 0

        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json", "--apply"]):
            with contextlib.redirect_stdout(io.StringIO()):
                training_ops.cli()
        self.assertEqual(fan.call_count, 2)

    def test_rollout_retention_cross_node_failure_blocks_apply(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1,
            "run_name": "demo",
            "active_step": 4,
            "required_steps": [3, 4],
            "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            return {host: ((1, "node failed") if host == hosts[0] else (0, json.dumps(self._rollout_result(host)))) for host in hosts}

        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json", "--apply"]), \
            contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        self.assertEqual(fan.call_count, 1)

    def test_rollout_retention_rejects_apply_candidate_change(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1, "run_name": "demo", "active_step": 4,
            "required_steps": [3, 4], "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            fake_fan.calls += 1
            step = 1 if fake_fan.calls == 1 else 2
            return {host: (0, json.dumps(self._rollout_result(host, apply=fake_fan.calls > 1, step=step))) for host in hosts}
        fake_fan.calls = 0
        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json", "--apply"]), \
            contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        self.assertEqual(fan.call_count, 2)

    def test_rollout_retention_rejects_complete_inventory_disagreement(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1, "run_name": "demo", "active_step": 4,
            "required_steps": [3, 4], "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            return {
                host: (0, json.dumps(self._rollout_result(host, all_steps=[0, 1, 2, 3, 4] if index == 0 else [0, 1, 2, 3, 5])))
                for index, host in enumerate(hosts)
            }
        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json"]), \
            contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        fan.assert_called_once()

    def test_rollout_retention_apply_subprocess_failure_is_reported(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1, "run_name": "demo", "active_step": 4,
            "required_steps": [3, 4], "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            fake_fan.calls += 1
            if fake_fan.calls == 2:
                return {hosts[0]: (1, "apply subprocess failed")}
            return {host: (0, json.dumps(self._rollout_result(host))) for host in hosts}
        fake_fan.calls = 0
        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json", "--apply"]), \
            contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                training_ops.cli()
        self.assertEqual(exited.exception.code, 2)
        self.assertEqual(fan.call_count, 2)

    def test_rollout_retention_no_candidates_is_verified_noop(self):
        nodes = [host for host in training_ops.trainer_hosts if host not in training_ops.blocked_hosts]
        evidence = {
            "version": 1, "run_name": "demo", "active_step": 4,
            "required_steps": [3, 4], "nodes": {host: {"run_dir": "/tmp/run_demo"} for host in nodes},
        }
        def fake_fan(hosts, callback):
            result = {"node": hosts[0], "run_name": "demo", "run_dir": "/tmp/run_demo", "active_step": 4, "all_steps": [3, 4], "selected_steps": [], "selected_bytes": 0, "selected": []}
            return {host: (0, json.dumps(dict(result, node=host))) for host in hosts}
        with mock.patch.object(training_ops, "_cold_run_paths", return_value=("/tmp", "/tmp/demo", "/tmp/run_demo")), \
            mock.patch.object(training_ops._rollout_retention, "validate_evidence", return_value=evidence), \
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan) as fan, \
            mock.patch.object(sys, "argv", ["training-ops", "rollout-retention", "demo", "--evidence", "/tmp/evidence.json", "--apply"]):
            with contextlib.redirect_stdout(io.StringIO()):
                result = training_ops.run("rollout-retention demo --evidence /tmp/evidence.json --apply")
        self.assertTrue(result["no_op"])
        fan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
