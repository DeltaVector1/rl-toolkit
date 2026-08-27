"""Static guards for the GLM state-only optimizer offload qualification."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "configs/qualification/glm52-native-b2048.toml").read_text()
PRODUCTION = (ROOT / "configs/paused/trainer.glm52-production.toml").read_text()
TRAIN = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()


class GLM52OptimizerOffloadTest(unittest.TestCase):
    def test_offload_keeps_fsdp_and_parameter_paths_on_gpu(self):
        self.assertIn("optim_cpu_offload = true", CONFIG)
        self.assertIn("fsdp_cpu_offload = false", CONFIG)
        self.assertIn("cold_cpu_offload = false", CONFIG)
        self.assertIn("cpu_offload=config.model.optim_cpu_offload", TRAIN)
        self.assertIn("optim_cpu_offload = false", PRODUCTION)

    def test_largest_stage_muon_state_budget(self):
        # The largest stage owns 11 sparse layers. Each local expert has three
        # rank-32 LoRA projections. Muon allocates one BF16 momentum tensor per
        # trainable value, so this is the exact state-only GPU cut.
        rank = 32
        hidden = 6144
        expert_hidden = 2048
        experts = 256
        sparse_layers = 11
        per_projection = rank * (hidden + expert_hidden)
        routed_values = sparse_layers * experts * 3 * per_projection
        state_gib = routed_values * 2 / 2**30
        self.assertEqual(routed_values, 2_214_592_512)
        self.assertGreaterEqual(state_gib, 4.0)


if __name__ == "__main__":
    unittest.main()
