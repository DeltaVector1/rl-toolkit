import unittest

import training_ops


class PreBatchProgressParserTest(unittest.TestCase):
    def telemetry(self):
        return (
            "complete_groups=43 clean_groups=26 rejected_groups=17 "
            "rejected_rollouts=272 policy_invalid_rollouts=70 "
            "accepted_buffer=416 accepted_target=1024 accepted_unit=rollouts "
            "accepted_buffer_tokens=9043485 partial_groups=16 partial_rollouts=161 "
            "inflight_train=128 "
            "oldest_partial_age_seconds=219.720 reject_stale_policy=0 "
            "reject_errored_group=17 reject_trainer_seq_len_group=0 "
            "reject_untrainable_group=0 reject_zero_advantage_group=0"
        )

    def test_reproduces_snapshot_without_rollout_metrics(self):
        progress = training_ops._parse_pipeline_telemetry(
            self.telemetry(), "2026-01-01T00:00:00.000000000Z"
        )

        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertTrue(progress.available)
        self.assertEqual(progress.complete_groups, 43)
        self.assertEqual(progress.clean_groups, 26)
        self.assertEqual(progress.rejected_groups, 17)
        self.assertEqual(progress.rejected_rollouts, 272)
        self.assertEqual(progress.policy_invalid_rollouts, 70)
        self.assertEqual(progress.accepted_buffer, 416)
        self.assertEqual(progress.accepted_target, 1024)
        self.assertEqual(progress.partial_groups, 16)
        self.assertEqual(progress.partial_rollouts, 161)
        self.assertEqual(progress.inflight_train, 128)
        self.assertIsNotNone(progress.progress_age_seconds)
        self.assertEqual(progress.rejection_reasons["errored_group"], 17)

        report = training_ops.Status(prebatch=progress).report()
        self.assertIn("pre-batch progress", report)
        self.assertIn("accepted buffer=416/1024 rollouts", report)
        self.assertIn("groups complete=43 clean=26 rejected=17", report)
        self.assertIn("partial=16 groups/161 rollouts", report)
        self.assertNotIn("prompt", report.lower())
        self.assertNotIn("response", report.lower())

    def test_rejects_unstructured_or_sensitive_fields(self):
        self.assertIsNone(
            training_ops._parse_pipeline_telemetry(
                self.telemetry() + " prompt=do-not-log"
            )
        )
        self.assertIsNone(
            training_ops._parse_pipeline_telemetry(
                self.telemetry().replace("accepted_buffer=416", "accepted_buffer=-1")
            )
        )
        self.assertIsNone(
            training_ops._parse_pipeline_telemetry(
                self.telemetry() + " complete_groups=999"
            )
        )

    def test_status_without_marker_does_not_invent_progress(self):
        status = training_ops.Status()
        self.assertFalse(status.prebatch.available)
        self.assertIn("pre-batch progress: no telemetry", status.report())


if __name__ == "__main__":
    unittest.main()
