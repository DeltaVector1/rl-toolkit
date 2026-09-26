import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


class TrainerComputeProbeTest(unittest.TestCase):
    def run_probe(
        self,
        log,
        started_at="2026-09-08T22:00:00.000000000Z",
        running="true",
        exit_code="0",
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sudo").write_text(
                "#!/bin/sh\nexec \"$@\"\n", encoding="utf-8"
            )
            (root / "docker").write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *State.Running*) printf '%s\\n' \"$TRAINER_RUNNING\" ;;\n"
                "  *State.ExitCode*) printf '%s\\n' \"$TRAINER_EXIT_CODE\" ;;\n"
                "  inspect*) printf '%s\\n' \"$TRAINER_STARTED_AT\" ;;\n"
                "  *logs*) printf '%s\\n' \"$TRAINER_LOG\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            (root / "sudo").chmod(0o755)
            (root / "docker").chmod(0o755)
            script = (
                "strip() { sed 's/\\x1b\\[[0-9;]*m//g'; };\n"
                + training_ops.trainer_compute_probe
            )
            environment = {
                "PATH": f"{directory}:/usr/bin:/bin",
                "TRAINER_LOG": log,
                "TRAINER_STARTED_AT": started_at,
                "TRAINER_RUNNING": running,
                "TRAINER_EXIT_CODE": exit_code,
            }
            result = subprocess.run(
                ["bash", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return training_ops._kv(result.stdout)

    def run_trainer_probe(self, log, running="false", exit_code="137"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sudo").write_text(
                "#!/bin/sh\nexec \"$@\"\n", encoding="utf-8"
            )
            (root / "docker").write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *State.StartedAt*) printf '%s\\n' \"$TRAINER_STARTED_AT\" ;;\n"
                "  *trainer*State.Running*) printf '/trainer %s exited\\n' \"$TRAINER_RUNNING\" ;;\n"
                "  *adapter*State.Running*) printf '/adapter false exited 0\\n' ;;\n"
                "  *State.Running*) printf '/orch false exited\\n' ;;\n"
                "  *State.ExitCode*) printf '%s\\n' \"$TRAINER_EXIT_CODE\" ;;\n"
                "  *logs*) printf '%s\\n' \"$TRAINER_LOG\" ;;\n"
                "  inspect*) printf '\\n' ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            (root / "curl").write_text(
                "#!/bin/sh\nprintf 000\n", encoding="utf-8"
            )
            for path in (root / "sudo", root / "docker", root / "curl"):
                path.chmod(0o755)
            compute = training_ops.trainer_compute_probe.replace(
                "__TRAINER__", "trainer"
            )
            script = training_ops.trainer_probe
            script = script.replace("__TRAINER__", "trainer")
            script = script.replace("__ORCH__", "orch")
            script = script.replace("__ADAPTER__", "9999")
            script = script.replace("__ADAPTER_CONTAINER__", "adapter")
            script = script.replace("__TRAINER_COMPUTE__", compute)
            environment = {
                "PATH": f"{directory}:/usr/bin:/bin",
                "TRAINER_LOG": log,
                "TRAINER_STARTED_AT": "2026-09-08T22:00:00.000000000Z",
                "TRAINER_RUNNING": running,
                "TRAINER_EXIT_CODE": exit_code,
            }
            result = subprocess.run(
                ["bash", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return training_ops._kv(result.stdout)

    def test_parses_current_step_with_optional_pp_completion(self):
        self.assertEqual(training_ops._parse_trainer_compute("2"), (2, None))
        self.assertEqual(
            training_ops._parse_trainer_compute("2 454.89"), (2, 454.89)
        )

    def test_rejects_malformed_or_negative_compute_report(self):
        for value in ("", "2 nope", "2 -1", "2 1 extra"):
            self.assertIsNone(training_ops._parse_trainer_compute(value))

    def test_compute_limit_remains_enforced(self):
        status = training_ops.Status(
            trainer_compute_step=2,
            trainer_compute_seconds=600.01,
        )
        self.assertIn(
            "trainer compute step 2 exceeds configured upper bound 600s; operator must stop",
            training_ops._warnings(status),
        )
        active = training_ops.Status(
            trainer_compute_state="active",
            trainer_compute_step=2,
            trainer_compute_active=True,
            trainer_compute_elapsed_seconds=600.01,
        )
        self.assertIn(
            "trainer compute step 2 has been active for more than 600s; operator must stop",
            training_ops._warnings(active),
        )

    def test_running_incomplete_step_stays_active(self):
        report = self.run_probe(
            "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=1\n"
            "2026-09-08T22:04:00Z PP step: 55 micro-batches in 300.00s\n"
            "2026-09-08T22:04:01Z TRAINER_FORWARD_BACKWARD_COMPLETE step=1 seconds=301.000000\n"
            "2026-09-08T22:05:00Z TRAINER_FORWARD_BACKWARD_START step=2\n"
            "2026-09-08T22:09:50Z PP step: 55 micro-batches in 454.89s\n"
        )
        self.assertEqual(training_ops._parse_trainer_compute(report["TRAINER_COMPUTE"]), (2, None))
        self.assertEqual(report["TRAINER_COMPUTE_PP_SECONDS"], "454.89")
        self.assertEqual(report["TRAINER_COMPUTE_ACTIVE"], "1")
        self.assertEqual(report["TRAINER_COMPUTE_STATE"], "active")
        self.assertTrue(float(report["TRAINER_COMPUTE_ELAPSED_SECONDS"]) >= 0)

    def test_exited_incomplete_step_stops_timer_and_preserves_pp(self):
        report = self.run_probe(
            "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=2\n"
            "2026-09-08T22:09:50Z PP step: 55 micro-batches in 454.89s\n",
            running="false",
            exit_code="1",
        )
        self.assertEqual(report["TRAINER_COMPUTE"], "2")
        self.assertEqual(report["TRAINER_COMPUTE_PP_SECONDS"], "454.89")
        self.assertEqual(report["TRAINER_COMPUTE_ACTIVE"], "0")
        self.assertEqual(report["TRAINER_COMPUTE_ELAPSED_SECONDS"], "")
        self.assertEqual(report["TRAINER_COMPUTE_STATE"], "stopped")

    def test_exited_completed_step_keeps_authoritative_times(self):
        report = self.run_probe(
            "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=2\n"
            "2026-09-08T22:09:50Z PP step: 55 micro-batches in 454.89s\n"
            "2026-09-08T22:10:00Z TRAINER_FORWARD_BACKWARD_COMPLETE step=2 seconds=601.000000\n",
            running="false",
            exit_code="0",
        )
        self.assertEqual(training_ops._parse_trainer_compute(report["TRAINER_COMPUTE"]), (2, 601.0))
        self.assertEqual(report["TRAINER_COMPUTE_PP_SECONDS"], "454.89")
        self.assertEqual(report["TRAINER_COMPUTE_ACTIVE"], "0")
        self.assertEqual(report["TRAINER_COMPUTE_STATE"], "complete")

    def test_nonzero_exit_marks_failure_without_known_fatal_text(self):
        report = self.run_trainer_probe(
            "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=2\n"
        )
        self.assertEqual(report["TRAINER_UP"], "down")
        self.assertEqual(report["TRAINER_FAILED"], "1")

    def test_matching_completion_ignores_later_adapter_work(self):
        report = self.run_probe(
            "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=2\n"
            "2026-09-08T22:09:50Z PP step: 55 micro-batches in 454.89s\n"
            "2026-09-08T22:10:00Z TRAINER_FORWARD_BACKWARD_COMPLETE step=2 seconds=601.000000\n"
            "2026-09-08T22:10:01Z Starting broadcasting weights to inference engine via shared filesystem\n"
            "2026-09-08T22:12:00Z Saving checkpoint at step 2\n"
        )
        self.assertEqual(
            training_ops._parse_trainer_compute(report["TRAINER_COMPUTE"]),
            (2, 601.0),
        )
        self.assertEqual(report["TRAINER_COMPUTE_PP_SECONDS"], "454.89")
        self.assertEqual(report["TRAINER_COMPUTE_ACTIVE"], "0")
        self.assertEqual(report["TRAINER_COMPUTE_STATE"], "complete")
        self.assertEqual(report["TRAINER_COMPUTE_ELAPSED_SECONDS"], "")
        status = training_ops.Status(
            trainer_compute_step=2,
            trainer_compute_seconds=601.0,
            trainer_compute_pp_seconds=454.89,
        )
        self.assertIn(
            "trainer compute step 2 exceeds configured upper bound 600s; operator must stop",
            training_ops._warnings(status),
        )

    def test_missing_or_invalid_container_start_does_not_read_logs(self):
        for started_at in ("", "not-a-timestamp"):
            report = self.run_probe(
                "2026-09-08T22:00:01Z TRAINER_FORWARD_BACKWARD_START step=2\n",
                started_at=started_at,
            )
            self.assertEqual(report["TRAINER_COMPUTE_STATE"], "unknown")
            self.assertNotIn("TRAINER_COMPUTE", report)


class TrainerLogAgeWarningTest(unittest.TestCase):
    def _has_trainer_stall_warning(self, warnings):
        return any(message.startswith("trainer has written nothing") for message in warnings)

    def _status(self, **engine):
        return training_ops.Status(
            trainer_up=True,
            orch_up=True,
            trainer_log_age=training_ops.stall_seconds + 1,
            engines=[training_ops.EngineState(host="h200a", **engine)],
        )

    def test_active_exact_rollout_suppresses_trainer_log_age_warning(self):
        status = self._status(
            policy_state="ready",
            running=1,
            throughput=5.0,
        )

        self.assertFalse(
            self._has_trainer_stall_warning(training_ops._warnings(status))
        )

    def test_negative_rollout_states_keep_trainer_log_age_warning(self):
        cases = (
            ("idle", {"running": 0, "throughput": 20.0}),
            ("non-exact-policy", {"policy_state": "staging", "running": 1, "throughput": 20.0}),
            ("stalled-throughput", {"running": 1, "throughput": 4.99}),
            ("unreachable", {"reachable": False, "running": 1, "throughput": 20.0}),
        )
        for name, engine in cases:
            with self.subTest(name=name):
                self.assertTrue(
                    self._has_trainer_stall_warning(
                        training_ops._warnings(self._status(**engine))
                    )
                )

        with self.subTest(name="orchestrator-down"):
            status = self._status(policy_state="ready", running=1, throughput=20.0)
            status.orch_up = False
            self.assertTrue(
                self._has_trainer_stall_warning(training_ops._warnings(status))
            )

        with self.subTest(name="trainer-only"):
            status = self._status(policy_state="ready", running=1, throughput=20.0)
            status.mode = "trainer-only"
            self.assertTrue(
                self._has_trainer_stall_warning(training_ops._warnings(status))
            )


class ActiveRewardRowsTest(unittest.TestCase):
    def test_selects_rows_after_active_container_start(self):
        rows = [
            {"step": 1, "time": 90, "eff": 0.1},
            {"step": 1, "time": 110, "eff": 0.2},
            {"step": 2, "time": 120, "eff": 0.3},
        ]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x85", "x85", 100
        )

        self.assertEqual([row["eff"] for row in selected], [0.2, 0.3])
        self.assertEqual(excluded, 1)
        self.assertEqual(without_time, 0)

    def test_does_not_drop_rows_without_matching_active_run(self):
        rows = [{"step": 1, "eff": 0.1}]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x22", "x85", 100
        )

        self.assertIs(selected, rows)
        self.assertEqual(excluded, 0)
        self.assertEqual(without_time, 0)

    def test_reports_rows_without_timestamps_when_filtering(self):
        rows = [{"step": 1, "eff": 0.1}, {"step": 2, "time": 110, "eff": 0.2}]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x85", "x85", 100
        )

        self.assertEqual([row["step"] for row in selected], [2])
        self.assertEqual(excluded, 0)
        self.assertEqual(without_time, 1)


class StepsReaderTest(unittest.TestCase):
    def trainer_config(self, directory, output_dir, filename):
        path = Path(directory) / "trainer.toml"
        path.write_text(
            f'output_dir = "{output_dir}"\n'
            "[file_monitor]\n"
            f'filename = "{filename}"\n',
            encoding="utf-8",
        )
        return path

    def fake_steps_ssh(self, calls, metrics_path, code=0, error=""):
        def ssh(host, script, timeout=180):
            calls.append(script)
            if code:
                return code, error
            return code, json.dumps(
                {
                    "metrics_path": metrics_path,
                    "rows": [
                        {
                            "step": 2,
                            "forward_backward_seconds": 204.8,
                            "wait_seconds": 0.0,
                            "tokens_per_second": 2.0,
                        }
                    ],
                    "invalid_lines": 0,
                    "excluded_lines": 0,
                }
            )

        return ssh

    def test_uses_configured_output_and_custom_filename(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            config = self.trainer_config(
                directory,
                "/data/glm52-rl/qualifications/custom-run",
                "step-times.jsonl",
            )
            expected = "/data/glm52-rl/qualifications/custom-run/step-times.jsonl"
            with (
                mock.patch.object(
                    training_ops,
                    "_config_for_run",
                    return_value={"trainer": str(config)},
                ) as profile,
                mock.patch.object(
                    training_ops,
                    "_ssh",
                    side_effect=self.fake_steps_ssh(calls, expected),
                ),
                redirect_stdout(StringIO()) as output,
            ):
                rows = training_ops.steps("custom-run")
        self.assertEqual(rows[0]["step"], 2)
        profile.assert_called_once_with("custom-run", required=False)
        rendered = output.getvalue()
        self.assertIn(f"step source=configured path={expected}", rendered)
        self.assertIn("204.8s", rendered)
        self.assertIn(f"path={expected}", calls[0])
        self.assertNotIn("find ", calls[0])

    def test_uses_legacy_archive_search_without_profile(self):
        calls = []
        expected = "/data/glm52-rl/runs/archived/run/trainer-metrics.jsonl"
        with (
            mock.patch.object(training_ops, "_config_for_run", return_value=None),
            mock.patch.object(
                training_ops,
                "_ssh",
                side_effect=self.fake_steps_ssh(calls, expected),
            ),
            redirect_stdout(StringIO()) as output,
        ):
            training_ops.steps("archived")
        self.assertIn("step source=archive", output.getvalue())
        self.assertIn("find /data/glm52-rl/runs/archived", calls[0])
        self.assertIn("trainer-metrics.jsonl", calls[0])

    def test_rejects_malformed_or_ambiguous_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text("output_dir = [", encoding="utf-8")
            with mock.patch.object(
                training_ops,
                "_config_for_run",
                return_value={"trainer": str(config)},
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot read trainer config"):
                    training_ops.steps("bad")
        with mock.patch.object(
            training_ops,
            "_config_for_run",
            side_effect=training_ops.Refused("multiple configured trainer profiles"),
        ):
            with self.assertRaisesRegex(training_ops.Refused, "multiple configured"):
                training_ops.steps("ambiguous")

    def test_reports_missing_configured_remote_file(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            config = self.trainer_config(
                directory,
                "/data/glm52-rl/qualifications/missing",
                "custom-metrics.jsonl",
            )
            expected = "/data/glm52-rl/qualifications/missing/custom-metrics.jsonl"
            with (
                mock.patch.object(
                    training_ops,
                    "_config_for_run",
                    return_value={"trainer": str(config)},
                ),
                mock.patch.object(
                    training_ops,
                    "_ssh",
                    side_effect=self.fake_steps_ssh(
                        calls,
                        expected,
                        code=4,
                        error=f"missing trainer metrics: {expected}",
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "source=configured") as raised:
                    training_ops.steps("missing")
        message = str(raised.exception)
        self.assertIn(expected, message)
        self.assertIn("exit=4", message)
        self.assertIn("cannot read trainer steps", message)


class ParityReaderTest(unittest.TestCase):
    def test_trinity_trainer_only_profile_is_discoverable_by_run_name(self):
        selected = training_ops._config_for_run("trinity-h200-parity")

        self.assertEqual(
            selected["trainer"], "configs/trinity-h200-parity/trainer.toml"
        )
        self.assertNotIn("orchestrator", selected)
        config_path = Path(__file__).resolve().parents[2] / selected["trainer"]
        with config_path.open("rb") as stream:
            config = training_ops.tomllib.load(stream)
        self.assertEqual(Path(config["output_dir"]).name, "trinity-h200-parity")
        self.assertEqual(
            config["data"]["fake"]["reference_path"],
            "/data/glm52-rl/qualifications/trinity-h200-parity/reference.json",
        )

    def run_reader(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            result = subprocess.run(
                ["bash", "-c", training_ops._parity_reader_script(str(path))],
                capture_output=True,
                text=True,
                check=False,
            )
        return result

    def valid_record(self):
        return {
            "sequence_lengths": [2, 1],
            "position_ids": [0, 1, 0],
            "loss_mask": [True, False, True],
            "delta": [0.1, 0.2, -0.1],
            "trainer_logprobs": [-1.0, -2.0, -3.0],
            "reference_logprobs": [-1.1, -2.2, -2.9],
        }

    def test_reader_reports_record_and_selected_tokens(self):
        result = self.run_reader(self.valid_record())

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["record_tokens"], 3)
        self.assertEqual(report["selected_tokens"], 2)
        self.assertAlmostEqual(report["loss_mask_excluded_fraction"], 1 / 3)
        self.assertNotIn("masked_fraction", report)

    def test_reader_rejects_invalid_arrays_and_empty_selection(self):
        cases = []
        mismatched = self.valid_record()
        mismatched["delta"] = [0.1]
        cases.append(mismatched)
        for value in (None, float("nan"), float("inf"), float("-inf")):
            nonfinite = self.valid_record()
            nonfinite["delta"][0] = value
            cases.append(nonfinite)
        empty = self.valid_record()
        empty["loss_mask"] = [False, False, False]
        cases.append(empty)
        overflow = self.valid_record()
        overflow["delta"][0] = 1000.0
        overflow["trainer_logprobs"][0] = -1.0
        overflow["reference_logprobs"][0] = -1001.0
        inconsistent = self.valid_record()
        inconsistent["delta"][0] = 0.5
        cases.append(inconsistent)

        for record in cases:
            result = self.run_reader(record)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid parity record", result.stderr)

        result = self.run_reader(overflow)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("K3 proxy overflow", result.stderr)

        result = self.run_reader([])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("top-level JSON must be an object", result.stderr)

    def test_parity_uses_configured_direct_record(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            trainer_path = Path(directory) / "trainer.toml"
            trainer_path.write_text(
                'output_dir = "/data/glm52-rl/runs/glm-perf-1"\n'
                '[data.fake]\nreference_path = "/tmp/reference"\n',
                encoding="utf-8",
            )

            def fake_ssh(host, script, timeout=180):
                calls.append(script)
                if script.startswith("python3 - "):
                    return 0, json.dumps({"record_tokens": 3, "selected_tokens": 2})
                if "direct-parity.json" in script:
                    return 0, "PARITY 100.0 /data/glm52-rl/runs/glm-perf-1/direct-parity.json\n"
                raise AssertionError("unexpected parity probe")

            def fake_fan(hosts, function, workers=None):
                host = hosts[0]
                return {host: function(host)}

            with (
                mock.patch.object(training_ops, "_config_for_run", return_value={"trainer": str(trainer_path)}),
                mock.patch.object(training_ops, "_fan", side_effect=fake_fan),
                mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                redirect_stdout(StringIO()),
            ):
                report = training_ops.parity("glm-perf-1")

        self.assertEqual(report["source"], "configured")
        self.assertIn("-maxdepth 0", calls[0])
        self.assertIn("/data/glm52-rl/runs/glm-perf-1/direct-parity.json", calls[0])
        self.assertNotIn("parity-step", calls[0])

    def test_parity_uses_configured_full_run_records(self):
        calls = []
        profile = training_ops._config_dir("configs/general-agent")
        trainer_path = Path(__file__).resolve().parents[2] / profile["trainer"]
        with trainer_path.open("rb") as stream:
            output_dir = training_ops.tomllib.load(stream)["output_dir"]
        run = Path(output_dir).name

        def fake_ssh(host, script, timeout=180):
            calls.append(script)
            if script.startswith("python3 - "):
                return 0, json.dumps({"record_tokens": 3, "selected_tokens": 2})
            if "parity-step" in script:
                return 0, f"PARITY 100.0 {output_dir}/parity-step20.json\n"
            raise AssertionError("unexpected parity probe")

        def fake_fan(hosts, function, workers=None):
            host = hosts[0]
            return {host: function(host)}

        with (
            mock.patch.object(training_ops, "_config_for_run", return_value={"trainer": "configs/general-agent/trainer.toml", "orchestrator": "configs/general-agent/orchestrator.toml"}),
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            redirect_stdout(StringIO()),
        ):
            report = training_ops.parity(run)

        self.assertEqual(report["source"], "configured")
        self.assertIn(output_dir, calls[0])
        self.assertIn("parity-step*.json", calls[0])
        self.assertNotIn("direct-parity.json", calls[0])

    def test_parity_uses_archive_search_without_profile(self):
        calls = []

        def fake_ssh(host, script, timeout=180):
            calls.append(script)
            if script.startswith("python3 - "):
                return 0, json.dumps({"record_tokens": 3, "selected_tokens": 2})
            if "parity-step" in script:
                return 0, "PARITY 100.0 /data/glm52-rl/runs/archived/parity-step1.json\n"
            raise AssertionError("unexpected parity probe")

        def fake_fan(hosts, function, workers=None):
            host = hosts[0]
            return {host: function(host)}

        with (
            mock.patch.object(training_ops, "_config_for_run", return_value=None),
            mock.patch.object(training_ops, "_fan", side_effect=fake_fan),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            redirect_stdout(StringIO()),
        ):
            report = training_ops.parity("archived")

        self.assertEqual(report["source"], "archive")
        self.assertIn("/data/glm52-rl/runs/archived", calls[0])
        self.assertIn("parity-step*.json", calls[0])


class ActiveMetricRowsTest(unittest.TestCase):
    def test_filters_old_trainer_steps_at_active_run_boundary(self):
        rows = [
            {"step": 8, "time": 1788100109.0},
            {"step": 1, "time": 1788110000.0},
            {"step": 5, "time": 1788120000.0},
        ]

        selected, excluded, without_time = training_ops._select_active_metric_rows(
            rows, "glm52", "glm52", 1788108469.312
        )

        self.assertEqual([row["step"] for row in selected], [1, 5])
        self.assertEqual(excluded, 1)
        self.assertEqual(without_time, 0)

    def test_keeps_historical_rows_when_no_active_run_matches(self):
        rows = [{"step": 8, "time": 100.0}]

        selected, excluded, without_time = training_ops._select_active_metric_rows(
            rows, "glm52", "other-run", 200.0
        )

        self.assertIs(selected, rows)
        self.assertEqual(excluded, 0)
        self.assertEqual(without_time, 0)


class AcceptanceProbeTest(unittest.TestCase):
    def acceptance(self, trainer_rows="", rollout_rows="", orchestrator_config=""):
        return training_ops._acceptance_from_text(
            "glm52",
            "",
            orchestrator_config,
            "\n".join(json.dumps(row) for row in trainer_rows),
            "\n".join(json.dumps(row) for row in rollout_rows),
            "",
            "too few steps",
        )

    def test_trainer_timing_uses_upper_bound_and_includes_step_one(self):
        for seconds in (300.0, 600.0):
            report = self.acceptance([{"step": 1, "time/forward_backward": seconds}])
            self.assertEqual(report.trainer_compute_step, 1)
            self.assertFalse(
                any("time/forward_backward exceeds" in alarm for alarm in report.alarms)
            )

        report = self.acceptance([{"step": 1, "time/forward_backward": 601.0}])
        self.assertTrue(
            any("time/forward_backward exceeds" in alarm for alarm in report.alarms)
        )

    def test_missing_nonfinite_and_negative_timing_is_explicit(self):
        for value in (None, float("nan"), -1.0):
            row = {"step": 1}
            if value is not None:
                row["time/forward_backward"] = value
            report = self.acceptance([row])
            self.assertTrue(
                any("time/forward_backward is missing or invalid" in alarm for alarm in report.alarms),
                value,
            )

    def test_rollout_timing_uses_completed_batch_cycle(self):
        for seconds in (300.0, 720.0):
            report = self.acceptance(
                [{"step": 1, "time/forward_backward": 300.0}],
                [
                    {
                        "step": 1,
                        "time/step": seconds,
                        "train/agg/all/timing/generation/max": 900.0,
                    }
                ],
            )
            self.assertFalse(
                any("batch cycle exceeds" in alarm for alarm in report.alarms)
            )
        report = self.acceptance(
            [{"step": 1, "time/forward_backward": 300.0}],
            [
                {
                    "step": 1,
                    "time/step": 721.0,
                    "train/agg/all/timing/generation/max": 1.0,
                }
            ],
        )
        self.assertTrue(any("batch cycle exceeds" in alarm for alarm in report.alarms))
        self.assertIn("metric time/step", report.report())

    def test_rollout_generation_max_does_not_replace_batch_cycle_measurement(self):
        report = self.acceptance(
            [{"step": 1, "time/forward_backward": 300.0}],
            [{"step": 1, "train/agg/all/timing/generation/max": 721.0}],
        )
        self.assertTrue(
            any("batch cycle is missing or invalid" in alarm for alarm in report.alarms)
        )

    def test_startup_eval_overhead_is_excluded_only_at_step_one(self):
        config = "[eval]\ninterval = 10\nskip_first_step = false\n"
        rows = [
            {
                "step": 1,
                "time/step": 1156.855868612416,
                "eval/decensor-hard-eval/all/reward/mean": 0.1,
            },
            {"step": 2, "time/step": 721.0},
        ]
        report = self.acceptance(
            [{"step": 2, "time/forward_backward": 300.0}], rows, config
        )
        self.assertEqual(report.rollout_generation_startup_steps, [1])
        self.assertTrue(any("batch cycle exceeds" in alarm for alarm in report.alarms))
        self.assertIn("startup evaluation overhead excluded", report.report())
        self.assertIn("time/step is sink-to-sink batch elapsed time", report.report())
        self.assertIn("can include waiting", report.report())

    def test_current_fp8_config_does_not_require_removed_dequant_flags(self):
        report = training_ops._acceptance_from_text(
            "glm52",
            """\
[model.quantization]
type = "fp8"
dequantize_checkpoint = false
""",
            "",
            "",
            "",
            "",
            "too few steps",
        )

        self.assertTrue(report.native_fp8)

    def test_explicit_optional_dequant_flag_rejects_native_fp8(self):
        report = training_ops._acceptance_from_text(
            "glm52",
            """\
[model.quantization]
type = "fp8"
dequantize_checkpoint = false
dequantize_checkpoint_moe = true
""",
            "",
            "",
            "",
            "",
            "too few steps",
        )

        self.assertFalse(report.native_fp8)

    def test_decodes_run_boundary_metadata(self):
        decoded = training_ops._decode_acceptance_probe(
            "ACTIVE_RUN glm52\n"
            "ACTIVE_RUN_SOURCE RUN\n"
            "START_EPOCH 1788108469.312\n"
        )

        self.assertEqual(decoded["active_run"], "glm52")
        self.assertEqual(decoded["active_run_source"], "RUN")
        self.assertEqual(decoded["start_epoch"], "1788108469.312")

    def test_acceptance_ignores_old_steps_and_failures(self):
        trainer_rows = [
            {"step": 8, "time": 1788100109.0, "time/forward_backward": 700.0},
            {"step": 1, "time": 1788110000.0, "time/forward_backward": 700.0},
            {"step": 5, "time": 1788120000.0, "time/forward_backward": 700.0},
        ]
        rollout_rows = [
            {
                "step": 8,
                "time": 1788100109.0,
                "train/agg/all/has_error/mean": 1.0,
            },
            {
                "step": 1,
                "time": 1788110000.0,
                "train/agg/all/has_error/mean": 0.0,
            },
        ]

        report = training_ops._acceptance_from_text(
            "glm52",
            "",
            "",
            "\n".join(json.dumps(row) for row in trainer_rows),
            "\n".join(json.dumps(row) for row in rollout_rows),
            "",
            "too few steps",
            trainer_active_run="glm52",
            trainer_start_epoch=1788108469.312,
            rollout_active_run="glm52",
            rollout_start_epoch=1788108469.312,
        )

        self.assertEqual(report.completed_steps, 5)
        self.assertEqual(report.failures, {})

    def test_isolated_unscored_rollout_is_not_failure_evidence(self):
        rows = [{
            "step": 1,
            "time": 1000.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 1.0,
            "train/agg/all/user_sim_failed/mean": 1.0,
        }]
        logs = "timestamp=950 Rollout failed after three attempts: rollout is unscored; its complete group will be replaced"
        self.assertEqual(training_ops._failure_evidence([], [], logs), {})
        self.assertEqual(training_ops._failure_evidence([], rows, logs), {})

    def test_paired_unscored_rollout_and_dropped_group_is_not_failure(self):
        rows = [{
            "step": 2,
            "time": 2000.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 0.0025,
        }]
        logs = (
            "timestamp=1950 Rollout failed in group abc — TaskError: rollout is unscored\n"
            "Finished group | rollouts=16 (errored=1) | dropped complete group; "
            "consecutive errored groups=1/8"
        )
        self.assertEqual(training_ops._failure_evidence([], rows, logs), {})

    def test_replacement_promise_and_drop_is_one_event(self):
        rows = [{
            "step": 2,
            "time": 2000.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 0.0025,
        }]
        logs = (
            "timestamp=1950 rollout is unscored; its complete group will be replaced\n"
            "Finished group | rollouts=16 (errored=1) | dropped complete group; "
            "consecutive errored groups=1/8"
        )
        self.assertEqual(training_ops._failure_evidence([], rows, logs), {})

    def test_replacement_does_not_suppress_a_later_error_row(self):
        rows = [
            {
                "step": 2,
                "time": 2000.0,
                "time/step": 100.0,
                "train/agg/all/has_error/mean": 0.0025,
            },
            {
                "step": 3,
                "time": 3000.0,
                "time/step": 100.0,
                "train/agg/all/has_error/mean": 0.0025,
                "train/agg/all/metrics/decensor_user_sim_failed/mean": 1.0,
            },
        ]
        logs = (
            "timestamp=1950 Rollout failed in group abc — TaskError: rollout is unscored\n"
            "Finished group | rollouts=16 (errored=1) | dropped complete group; "
            "consecutive errored groups=1/8"
        )
        self.assertEqual(
            training_ops._failure_evidence([], rows, logs),
            {"environment": 1, "transport/rollout": 1},
        )

    def test_replacement_does_not_hide_unrelated_unscored_log(self):
        rows = [{
            "step": 2,
            "time": 2000.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 0.0025,
        }]
        logs = (
            "timestamp=1950 Rollout failed in group abc — TaskError: rollout is unscored\n"
            "Finished group | rollouts=16 (errored=1) | dropped complete group; "
            "consecutive errored groups=1/8\n"
            "Rollout failed in group unrelated: rollout is unscored"
        )
        self.assertEqual(
            training_ops._failure_evidence([], rows, logs),
            {"transport": 1},
        )

    def test_dated_replacement_does_not_match_a_later_date(self):
        rows = [{
            "step": 2,
            "time": 1789732850.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 0.0025,
        }]
        logs = (
            "2026-09-17T12:00:00+00:00 Rollout failed in group abc — TaskError: rollout is unscored\n"
            "2026-09-17T12:00:00+00:00 Finished group | rollouts=16 (errored=1) | "
            "dropped complete group; consecutive errored groups=1/8"
        )
        self.assertEqual(
            training_ops._failure_evidence([], rows, logs),
            {"transport/rollout": 1},
        )

    def test_unscored_text_does_not_hide_other_failure_categories(self):
        replacement = (
            "Rollout failed in group abc — TaskError: rollout is unscored\n"
            "Finished group | rollouts=16 (errored=1) | dropped complete group; "
            "consecutive errored groups=1/8\n"
        )
        rows = [{
            "step": 2,
            "time": 2000.0,
            "time/step": 100.0,
            "train/agg/all/has_error/mean": 0.0025,
        }]
        for line, category in (
            ("judge failed: rollout is unscored", "judge"),
            ("adapter load failed: rollout is unscored", "adapter"),
            ("OOM while scoring: rollout is unscored", "stability"),
        ):
            failures = training_ops._failure_evidence([], rows, replacement + line)
            self.assertIn(category, failures, line)

        failures = training_ops._failure_evidence(
            [], rows, replacement + "Rollout failed: ReadTimeout; rollout is unscored"
        )
        self.assertIn("transport", failures)

    def test_replaced_group_does_not_hide_transport_or_repeated_group_failure(self):
        replacement = (
            "rollout is unscored; its complete group will be replaced\n"
            "Rollout failed in group 4: ReadTimeout"
        )
        rows = [{"step": 1, "train/agg/all/has_error/mean": 1.0}]
        self.assertIn("transport", training_ops._failure_evidence([], rows, replacement))

        repeated = (
            "rollout is unscored; its complete group will be replaced\n"
            "Finished group | consecutive errored groups=8/8"
        )
        self.assertIn("transport", training_ops._failure_evidence([], rows, repeated))

    def test_judge_outage_remains_failure_evidence(self):
        failures = training_ops._failure_evidence(
            [], [], "scoring judge returned 503"
        )
        self.assertIn("judge", failures)
        self.assertEqual(
            training_ops._failure_evidence([], [], "judge connection healthy"), {}
        )


if __name__ == "__main__":
    unittest.main()
