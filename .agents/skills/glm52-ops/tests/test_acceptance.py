import json
import unittest

import glm52_ops


TRAINER_CONFIG = """
[model]
seq_len = 32768

[model.quantization]
type = "fp8"
dequantize_checkpoint = false
dequantize_checkpoint_linears = false
dequantize_checkpoint_moe = false

[optim]
lr = 8e-6
max_norm = 1.0
weight_decay = 0.0
"""

ORCHESTRATOR_CONFIG = """
seq_len = 8192
batch_size = 1024
group_size = 16

[optim]
lr = 8e-6

[train.sampling]
max_completion_tokens = 16384

[[train.env]]
env.agent.max_output_tokens = 16384

[train.env.sampling]
max_completion_tokens = 16384
"""


def jsonl(*rows):
    return "\n".join(json.dumps(row) for row in rows)


class AcceptanceTest(unittest.TestCase):
    def build(
        self,
        *,
        trainer_config=TRAINER_CONFIG,
        orchestrator_config=ORCHESTRATOR_CONFIG,
        trainer_metrics=None,
        rollout_metrics=None,
        logs="batch_size=2048 validation completed",
        reward_verdict="inside the noise",
    ):
        if trainer_metrics is None:
            trainer_metrics = jsonl(
                {"step": 50, "time/forward_backward": 700.0}
            )
        if rollout_metrics is None:
            rollout_metrics = jsonl(
                {"step": 50, "train/agg/all/timing/generation/max": 600.0}
            )
        return glm52_ops._acceptance_from_text(
            "x-test",
            trainer_config,
            orchestrator_config,
            trainer_metrics,
            rollout_metrics,
            logs,
            reward_verdict,
        )

    def test_all_measured_gates_pass(self):
        report = self.build()

        self.assertTrue(report.passed)
        self.assertEqual(report.context_length, 32768)
        self.assertEqual(report.max_output_lengths, [16384])
        self.assertEqual(report.completed_steps, 50)
        self.assertIn("VERDICT: PASS", report.report())
        self.assertNotIn("\\n", report.report())

    def test_rollout_packing_length_does_not_replace_training_context(self):
        report = self.build(
            trainer_config=TRAINER_CONFIG.replace("32768", "2048")
        )

        self.assertEqual(report.context_length, 2048)
        self.assertTrue(
            any("context length is 2048" in alarm for alarm in report.alarms)
        )

    def test_timing_stability_and_failure_alarms_are_separate(self):
        report = self.build(
            trainer_metrics=jsonl(
                {
                    "step": 14,
                    "time/forward_backward": 3540.0,
                    "diagnostic/nonfinite_gradient": 1,
                }
            ),
            rollout_metrics=jsonl(
                {
                    "step": 19,
                    "train/agg/all/timing/generation/max": 1500.0,
                    "train/agg/all/has_error/mean": 0.01,
                }
            ),
            logs="adapter load timeout",
            reward_verdict="falling",
        )

        alarms = "\n".join(report.alarms)
        self.assertIn("trainer forward/backward", alarms)
        self.assertIn("active rollout generation", alarms)
        self.assertIn("stability failure evidence", alarms)
        self.assertIn("transport/rollout failure evidence", alarms)
        self.assertIn("adapter failure evidence", alarms)
        self.assertIn("reward verdict is falling", alarms)
        self.assertIn("14/50", alarms)


if __name__ == "__main__":
    unittest.main()
