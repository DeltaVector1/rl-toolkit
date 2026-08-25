"""Regression checks for the bounded native-FP8 checkpoint mode."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTING = (
    ROOT / "source/src/prime_rl/trainer/models/layers/checkpointing.py"
).read_text()
MODEL = (ROOT / "source/src/prime_rl/trainer/model.py").read_text()
CONFIG = (
    ROOT
    / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
).read_text()


class ReentrantSharedExpertCheckpointTest(unittest.TestCase):
    def test_mode_is_explicit_in_the_config_schema(self):
        self.assertIn('"reentrant_attention_shared_expert"', CONFIG)

    def test_attention_and_shared_expert_are_reentrant(self):
        function = CHECKPOINTING.split(
            "def set_reentrant_attention_shared_expert_checkpointing", 1
        )[1].split("\n\ndef set_reentrant_attention_all_experts_checkpointing", 1)[0]
        self.assertIn("set_reentrant_attention_checkpointing(layer)", function)
        self.assertIn('checkpoint_method(\n        shared_expert,\n        "forward"', function)
        self.assertIn("use_reentrant=True", function)
        self.assertIn("preserve_rng_state=True", function)

    def test_routed_experts_are_not_checkpointed(self):
        function = CHECKPOINTING.split(
            "def set_reentrant_attention_shared_expert_checkpointing", 1
        )[1].split("\n\ndef set_reentrant_attention_all_experts_checkpointing", 1)[0]
        self.assertNotIn("_run_routed_experts", function)
        self.assertNotIn("experts,", function)

    def test_dense_layers_keep_attention_checkpointing(self):
        function = CHECKPOINTING.split(
            "def set_reentrant_attention_shared_expert_checkpointing", 1
        )[1].split("\n\ndef set_reentrant_attention_all_experts_checkpointing", 1)[0]
        self.assertIn("if shared_expert is None:\n        return", function)

    def test_model_routes_only_the_new_mode_to_the_new_helper(self):
        branch = MODEL.split(
            'elif ac_config.mode == "reentrant_attention_shared_expert":', 1
        )[1].split("elif ac_config.mode", 1)[0]
        self.assertIn(
            "set_reentrant_attention_shared_expert_checkpointing(transformer_block)",
            branch,
        )

    def test_stage_cutoff_is_disabled_by_default(self):
        self.assertIn(
            "keep_routed_expert_activations_from_pp_stage: int | None = Field(None, ge=0)",
            CONFIG,
        )

    def test_stage_cutoff_keeps_early_stages_recomputed(self):
        branch = MODEL.split(
            'elif ac_config.mode == "reentrant_attention_all_experts":', 1
        )[1].split("elif ac_config.mode", 1)[0]
        self.assertIn("if keep_routed_expert_activations:", branch)
        self.assertLess(
            branch.index("set_reentrant_attention_shared_expert_checkpointing("),
            branch.index("set_reentrant_attention_all_experts_checkpointing("),
        )

    def test_stage_cutoff_fails_outside_pipeline_range(self):
        setup = MODEL.split("def apply_ac", 1)[1].split(
            "selective_layers = 0", 1
        )[0]
        self.assertIn("pp_size <= 1", setup)
        self.assertIn("virtual_stages != 1", setup)
        self.assertIn("routed_expert_cutoff >= pp_size", setup)

if __name__ == "__main__":
    unittest.main()
