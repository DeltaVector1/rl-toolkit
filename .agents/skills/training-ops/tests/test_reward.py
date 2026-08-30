import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


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
    def test_probe_uses_container_command_when_env_has_no_run(self):
        script = training_ops._acceptance_probe_script("glm52", "trainer")

        self.assertIn("container_cmd=", script)
        self.assertIn("active_run_source=log-dir", script)
        self.assertIn('docker logs --since "$started_at"', script)

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


if __name__ == "__main__":
    unittest.main()
