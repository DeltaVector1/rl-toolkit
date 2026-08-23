import json
import unittest

import training_ops


trainer_config = """
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

orchestrator_config = """
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

muse_trainer_config = """
[model]
name = "/data/muse-glimmer/model"
seq_len = 65536
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"
cp = 4
pp = 8

[model.lora]
rank = 128
alpha = 256.0
dropout = 0.0

[optim]
lr = 8e-6
max_norm = 1.0
weight_decay = 0.0
"""

muse_orchestrator_config = orchestrator_config


def jsonl(*rows):
    return "\n".join(json.dumps(row) for row in rows)


class AcceptanceTest(unittest.TestCase):
    def build(
        self,
        *,
        trainer_config=trainer_config,
        orchestrator_config=orchestrator_config,
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
        return training_ops._acceptance_from_text(
            "x-test",
            trainer_config,
            orchestrator_config,
            trainer_metrics,
            rollout_metrics,
            logs,
            reward_verdict,
        )

    def test_all_measured_checks_pass(self):
        report = self.build()

        self.assertTrue(report.passed)
        self.assertEqual(report.context_length, 32768)
        self.assertEqual(report.max_output_lengths, [16384])
        self.assertEqual(report.completed_steps, 50)
        self.assertIn("result: pass", report.report())
        self.assertNotIn("\\n", report.report())

    def test_muse_uses_bf16_batch_context_and_one_sided_time_limits(self):
        report = training_ops._acceptance_from_text(
            "muse-test",
            muse_trainer_config,
            muse_orchestrator_config,
            jsonl({"step": 50, "time/forward_backward": 250.0}),
            jsonl({"step": 50, "train/agg/all/timing/generation/max": 300.0}),
            "batch_size=1024 validation completed",
            "inside the noise",
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.profile, "muse")
        self.assertTrue(report.precision_ok)
        self.assertIn("BF16 trainer: yes", report.report())
        self.assertNotIn("native FP8 trainer", report.report())

if __name__ == "__main__":
    unittest.main()
