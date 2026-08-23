import ast
import json
import tempfile
import unittest
from pathlib import Path

import tomllib

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "source/src/prime_rl/trainer/model.py"


class NativeFP8ExpertPrefetchTest(unittest.TestCase):
    def test_native_fp8_expert_prefetch_is_explicit(self):
        tree = ast.parse(MODEL.read_text())
        setup = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_fsdp"
        )
        source = ast.unparse(setup)

        self.assertEqual(
            source.count("isinstance(next_mlp.experts, FSDPModule)"),
            1,
        )
        self.assertEqual(
            source.count("isinstance(prev_mlp.experts, FSDPModule)"),
            1,
        )
        self.assertIn("prefetch_modules.append(next_mlp.experts)", source)
        self.assertIn("prefetch_modules.append(prev_mlp.experts)", source)
        self.assertEqual(source.count("config.native_fp8_expert_prefetch"), 2)

        trainer_config = (
            ROOT
            / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
        ).read_text()
        self.assertIn("native_fp8_expert_prefetch: bool = False", trainer_config)

    def test_b2048_profile_uses_64_full_context_microbatches(self):
        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        relative = operations["profiles"]["glm52-native-b2048"]["trainer"]
        with (ROOT / relative).open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["max_steps"], 2)
        self.assertEqual(
            config["data"], {"fake": {"batch_size": 64, "generate_samples": True}}
        )
        self.assertEqual(config["model"]["seq_len"], 32768)
        self.assertEqual(config["model"]["pp"], 8)
        self.assertEqual(config["model"]["cp"], 8)
        self.assertEqual(config["model"]["ep"], 8)
        self.assertIs(config["model"]["reshard_after_forward"], True)
        self.assertFalse(config["model"]["native_fp8_expert_prefetch"])

    def test_stability_profile_matches_production_shape(self):
        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        relative = operations["profiles"]["glm52-native-stability20"]["trainer"]
        self.assertTrue(relative.startswith("configs/qualification/"))
        with (ROOT / relative).open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["max_steps"], 20)
        self.assertFalse(config["enable_router_replay"])
        self.assertEqual(
            config["data"], {"fake": {"batch_size": 32, "generate_samples": True}}
        )
        self.assertEqual(config["model"]["seq_len"], 32768)
        self.assertEqual(config["model"]["pp"], 8)
        self.assertEqual(config["model"]["cp"], 8)
        self.assertEqual(config["model"]["ep"], 8)
        self.assertIs(config["model"]["reshard_after_forward"], True)
        self.assertFalse(config["model"]["native_fp8_expert_prefetch"])
        self.assertTrue(config["model"]["freeze_moe_router"])
        self.assertEqual(config["optim"]["lr"], 0.0)
        self.assertEqual(config["model"]["lora"]["rank"], 32)
        self.assertIn("experts", config["model"]["lora"]["target_modules"])
        self.assertIn("shared_expert", config["model"]["lora"]["target_modules"])

        data_source = (ROOT / "source/src/prime_rl/trainer/rl/data.py").read_text()
        self.assertIn("self.batch_counter * 1000 + micro_batch_idx", data_source)

    def test_production_keeps_rejected_features_disabled(self):
        with (ROOT / "configs/paused/trainer.glm52-production.toml").open("rb") as stream:
            production = tomllib.load(stream)
        self.assertIs(production["model"]["reshard_after_forward"], True)
        self.assertFalse(production["model"]["native_fp8_expert_prefetch"])

    def test_glm_router_keeps_bf16_gate_and_uses_h200_order(self):
        model_source = MODEL.read_text()
        moe_source = (
            ROOT / "source/src/prime_rl/trainer/models/layers/moe.py"
        ).read_text()
        token_router_source = moe_source.split("class NemotronHRouter", 1)[0]
        self.assertIn('model_type", None) == "glm_moe_dsa"', model_source)
        self.assertIn("router_param_dtype = (", model_source)
        self.assertIn("torch.bfloat16", model_source)
        self.assertIn("use_fp32_router_output = True", model_source)
        self.assertIn(
            "0.5 * torch.tanh(0.5 * scores) + 0.5",
            moe_source,
        )
        self.assertIn(
            "class _BF16RouterGemm",
            moe_source,
        )
        self.assertIn(
            'if gate_weight.requires_grad:',
            moe_source,
        )
        self.assertIn(
            "requires frozen router weights",
            moe_source,
        )
        self.assertIn(
            "torch.mm(hidden_states, gate_weight.t(), out_dtype=torch.float32)",
            moe_source,
        )
        self.assertNotIn("sorted=False", token_router_source)

        forward_source = ast.unparse(
            next(
                node
                for node in ast.parse(moe_source).body
                if isinstance(node, ast.ClassDef) and node.name == "TokenChoiceTopKRouter"
            )
        )
        self.assertIn("if router_debug_active", forward_source)
        self.assertIn("raw_router_logits = scores.to", forward_source)
        self.assertIn("del raw_router_logits", forward_source)

    def test_glm_router_bias_stays_fp32_across_setup_and_load(self):
        model_source = MODEL.read_text()
        helper = ast.unparse(
            next(
                node
                for node in ast.parse(model_source).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "preserve_glm_moe_bias_precision"
            )
        )
        self.assertIn("glm_moe_dsa", helper)
        self.assertIn("expert_bias", helper)
        self.assertIn("e_score_correction_bias", helper)
        self.assertIn("value.to(dtype=torch.float32)", helper)
        self.assertIn("GLM {name} must remain float32", helper)

        setup_source = ast.unparse(
            next(
                node
                for node in ast.parse(model_source).body
                if isinstance(node, ast.FunctionDef) and node.name == "setup_model"
            )
        )
        self.assertGreaterEqual(
            setup_source.count("preserve_glm_moe_bias_precision(model)"), 4
        )
        self.assertIn("apply_pp_layers(model, pp_rank, parallel_dims.pp)", setup_source)
        self.assertIn("setup_fsdp(model, config, parallel_dims)", setup_source)
        self.assertIn("load_dcp_from_hf(model, config, parallel_dims)", setup_source)

        bias_bytes = 78 * 256 * 4
        self.assertEqual(bias_bytes, 79872)
        self.assertLess(bias_bytes, 128 * 1024)

        load_source = ast.unparse(
            next(
                node
                for node in ast.parse(model_source).body
                if isinstance(node, ast.FunctionDef) and node.name == "load_dcp_from_hf"
            )
        )
        self.assertIn("restore_glm_moe_bias_values(model, snapshot_path)", load_source)
        self.assertIn("source.dtype != torch.float32", ast.unparse(ast.parse(model_source)))

    @unittest.skipUnless(torch is not None, "PP fingerprint validation requires PyTorch")
    def test_glm_router_bias_fingerprint_requires_all_75_routed_layers(self):
        from prime_rl.trainer.model import (
            _expected_glm_moe_layer_indices,
            _validate_glm_moe_bias_union,
        )

        model = type(
            "Model",
            (),
            {
                "config": type(
                    "Config",
                    (),
                    {"num_hidden_layers": 78, "first_k_dense_replace": 3},
                )(),
            },
        )()
        expected = _expected_glm_moe_layer_indices(model)
        self.assertEqual(expected, tuple(range(3, 78)))

        def record(pp_rank, layers):
            return {
                "pp_rank": pp_rank,
                "layers": {
                    f"model.layers.{layer}.mlp.gate.e_score_correction_bias": {
                        "layer": layer,
                        "dtype": "torch.float32",
                        "shape": [256],
                        "sha256": f"{layer:064x}",
                    }
                    for layer in layers
                },
            }

        ranges = [range(3 + 10 * rank, 3 + 10 * (rank + 1)) for rank in range(7)]
        ranges.append(range(73, 78))
        records = [record(rank, layers) for rank, layers in enumerate(ranges)]
        count, fingerprint = _validate_glm_moe_bias_union(records, expected, 8)
        self.assertEqual(count, 75)
        self.assertEqual(len(fingerprint), 64)

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _validate_glm_moe_bias_union(records[:-1], expected, 8)
        duplicate = records.copy()
        duplicate[1] = record(1, range(12, 22))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            _validate_glm_moe_bias_union(duplicate, expected, 8)

    def test_glm_router_bias_restores_after_resume_load(self):
        train_source = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
        load_position = train_source.index("ckpt_manager.load(checkpoint_step")
        restore_position = train_source.index("restore_glm_moe_bias_values(model, Path(config.model.name)")
        self.assertLess(load_position, restore_position)

    @unittest.skipUnless(torch is not None, "FP32 checkpoint restoration requires PyTorch")
    def test_glm_router_bias_restores_exact_checkpoint_values(self):
        from safetensors.torch import save_file

        from prime_rl.trainer.model import restore_glm_moe_bias_values
        from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 3
                self.mlp = MoE(MoEArgs(num_experts=4, top_k=1), dim=2, hidden_dim=4)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "model_type": "glm_moe_dsa",
                        "num_hidden_layers": 4,
                        "first_k_dense_replace": 3,
                    },
                )()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer()])

        source = torch.tensor([34.18921, 34.60598, 34.601982, 34.60798], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {"model.layers.3.mlp.gate.e_score_correction_bias": source},
                root / "model.safetensors",
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "model.layers.3.mlp.gate.e_score_correction_bias": "model.safetensors"
                        },
                    }
                )
            )
            model = Model()
            model.model.layers[0].mlp.expert_bias.copy_(source.to(torch.bfloat16).float())
            restore_glm_moe_bias_values(model, root)
            self.assertEqual(model.model.layers[0].mlp.expert_bias.dtype, torch.float32)
            torch.testing.assert_close(model.model.layers[0].mlp.expert_bias, source, rtol=0, atol=0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "BF16 router autograd test requires CUDA PyTorch",
    )
    def test_bf16_router_gemm_matches_forward_and_backward_reference(self):
        from prime_rl.trainer.models.layers.moe import _BF16RouterGemm

        device = torch.device("cuda")
        hidden_states = torch.randn(17, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        gate_weight = torch.randn(11, 32, device=device, dtype=torch.bfloat16)
        actual = _BF16RouterGemm.apply(hidden_states, gate_weight)
        reference = torch.mm(hidden_states.float(), gate_weight.float().t())
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)

        output_gradient = torch.randn_like(actual)
        (actual * output_gradient).sum().backward()
        expected_hidden_gradient = torch.mm(
            output_gradient, gate_weight.float()
        ).to(torch.bfloat16)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        torch.testing.assert_close(
            hidden_states.grad, expected_hidden_gradient, rtol=0.01, atol=0.01
        )
        self.assertIsNone(gate_weight.grad)

        trainable_weight = gate_weight.detach().requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "frozen router weights"):
            _BF16RouterGemm.apply(hidden_states.detach(), trainable_weight)


if __name__ == "__main__":
    unittest.main()
