"""Static regression checks for native-FP8 shared-expert LoRA."""

import ast
import hashlib
from pathlib import Path
import re
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
TRAIN_PATH = REPO / "source/src/prime_rl/trainer/rl/train.py"


def qualification_inventory():
    tree = ast.parse(TRAIN_PATH.read_text())
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "_qualification_source_targets",
                    "_qualification_adapter_targets",
                }
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_adapter_inventory"
        )
    ]
    namespace = {"hashlib": hashlib, "re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return namespace["_qualification_adapter_inventory"]


def complete_glm52_adapter_shapes():
    shapes = {}
    for layer in range(78):
        for projection in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ):
            shapes[f"model.layers.{layer}.self_attn.{projection}.lora_A.weight"] = (32, 1)
            shapes[f"model.layers.{layer}.self_attn.{projection}.lora_B.weight"] = (1, 32)
    for layer in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shapes[f"model.layers.{layer}.mlp.{projection}.lora_A.weight"] = (32, 1)
            shapes[f"model.layers.{layer}.mlp.{projection}.lora_B.weight"] = (1, 32)
    for layer in range(3, 78):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shapes[f"model.layers.{layer}.mlp.shared_experts.{projection}.lora_A.weight"] = (32, 1)
            shapes[f"model.layers.{layer}.mlp.shared_experts.{projection}.lora_B.weight"] = (1, 32)
        for expert in range(256):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                shapes[f"{prefix}.lora_A.weight"] = (32, 1)
                shapes[f"{prefix}.lora_B.weight"] = (1, 32)
    return shapes


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

    def test_qualification_uses_leaf_metadata_and_exact_expert_tensor_coverage(self):
        validate = qualification_inventory()
        shapes = complete_glm52_adapter_shapes()
        inventory = validate(shapes)
        self.assertEqual(inventory["tensor_count"], 116_448)
        self.assertEqual(inventory["shared_expert_tensors"], 450)
        self.assertEqual(inventory["routed_expert_tensors"], 115_200)
        self.assertEqual(len(inventory["leaf_targets"]), 9)
        self.assertNotIn("shared_expert", inventory["leaf_targets"])

        shapes.pop("model.layers.77.mlp.experts.255.down_proj.lora_B.weight")
        with self.assertRaisesRegex(RuntimeError, "routed expert missing=1"):
            validate(shapes)


if __name__ == "__main__":
    unittest.main()
