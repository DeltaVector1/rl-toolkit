"""Static regression checks for native-FP8 shared-expert LoRA."""

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
LORA_SOURCE = (REPO / "source/src/prime_rl/trainer/lora.py").read_text()
MODEL_SOURCE = (
    REPO
    / "source/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py"
).read_text()
SERVER_SOURCE = (
    REPO / "source/src/prime_rl/inference/vllm/server.py"
).read_text()


class FP8SharedExpertLoRATest(unittest.TestCase):
    def test_wrapper_keeps_fp8_base_and_bf16_trainable_weights(self):
        wrapper = LORA_SOURCE.split(
            "class MultiLoRAFP8CheckpointFeedForward", 1
        )[1].split("\n\ndef _find_target_modules", 1)[0]
        self.assertIn("dtype=torch.bfloat16", wrapper)
        self.assertIn("_Linear.apply(", wrapper)
        self.assertIn("getattr(self.base_layer, scale_name)", wrapper)
        self.assertNotIn(".bfloat16()", wrapper)

    def test_all_three_shared_expert_projections_are_adapted(self):
        wrapper = LORA_SOURCE.split(
            "class MultiLoRAFP8CheckpointFeedForward", 1
        )[1].split("\n\ndef _find_target_modules", 1)[0]
        self.assertIn('(\"w1\", \"w1_scale_inv\", \"gate_proj\")', wrapper)
        self.assertIn('(\"w2\", \"w2_scale_inv\", \"down_proj\")', wrapper)
        self.assertIn('(\"w3\", \"w3_scale_inv\", \"up_proj\")', wrapper)

    def test_export_uses_hf_shared_experts_name(self):
        self.assertIn(
            'replace(".mlp.shared_expert.", ".mlp.shared_experts.")',
            MODEL_SOURCE,
        )

    def test_shared_experts_does_not_enable_routed_moe_lora(self):
        server = SERVER_SOURCE.split("def server(", 1)[1]
        self.assertIn('"experts" in target.split(".")', server)
        self.assertNotIn('any("expert" in m', server)


if __name__ == "__main__":
    unittest.main()
