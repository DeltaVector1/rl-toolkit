"""Regression checks for the bounded native-FP8 checkpoint mode."""

from pathlib import Path
import ast
import types
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
        self.assertIn("if keep_for_layer:", branch)
        self.assertLess(
            branch.index("set_reentrant_attention_shared_expert_checkpointing("),
            branch.index("set_reentrant_attention_all_experts_checkpointing("),
        )

    def test_stage_cutoff_fails_outside_pipeline_range(self):
        helper = MODEL.split("def _routed_expert_checkpoint_policy", 1)[1].split(
            "\ndef apply_ac", 1
        )[0]
        self.assertIn("pp_size <= 1", helper)
        self.assertIn("routed_expert_cutoff >= pp_size", helper)
        self.assertIn("pp_layer_ranges", helper)

    def test_virtual_chunks_on_one_rank_get_different_policies(self):
        """The physical cutoff is mapped to the accepted global layer boundary."""
        tree = ast.parse(MODEL)
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_routed_expert_checkpoint_policy"
        )
        namespace = {"nn": types.SimpleNamespace(Module=object)}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(ROOT), "exec"), namespace)

        class Layers:
            def named_children(self):
                return [("10", object()), ("50", object())]

        language_model = types.SimpleNamespace(
            pp_physical_rank=1,
            pp_physical_size=8,
            pp_virtual_stages=2,
            pp_layer_ranges=((5, 10), (45, 50)),
            layers=Layers(),
            config=types.SimpleNamespace(
                num_hidden_layers=78,
                pp_layer_boundaries=[0, 8, 18, 28, 38, 48, 58, 69, 78],
            ),
        )
        language_model.layers.named_children = lambda: [
            ("5", object()),
            ("47", object()),
            ("48", object()),
            ("49", object()),
        ]
        policy = namespace["_routed_expert_checkpoint_policy"](language_model, 5)
        self.assertFalse(policy[5])
        self.assertFalse(policy[47])
        self.assertTrue(policy[48])
        self.assertTrue(policy[49])

        # The next virtual chunk is owned by physical rank 2. It must keep
        # layer 50 as well, proving that retention continues across chunks.
        language_model.pp_physical_rank = 2
        language_model.pp_layer_ranges = ((10, 15), (50, 55))
        language_model.layers.named_children = lambda: [
            ("10", object()),
            ("50", object()),
        ]
        policy = namespace["_routed_expert_checkpoint_policy"](language_model, 5)
        self.assertFalse(policy[10])
        self.assertTrue(policy[50])

if __name__ == "__main__":
    unittest.main()
