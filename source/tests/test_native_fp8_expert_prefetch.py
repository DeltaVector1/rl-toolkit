import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomllib

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "source/src/prime_rl/trainer/model.py"


class NativeFP8ExpertPrefetchTest(unittest.TestCase):
    def test_qualification_environment_ignores_generic_trainer_identity(self):
        train_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
        tree = ast.parse(train_path.read_text())
        names = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_qualification_environment_names"
                for target in node.targets
            )
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_environment"
        )
        module = ast.Module(body=[names, function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"os": os, "re": __import__("re")}
        exec(compile(module, str(train_path), "exec"), namespace)
        read = namespace["_qualification_environment"]

        with patch.dict(
            os.environ,
            {"PRIME_TRAINER_HOST": "n82", "PRIME_TRAINER_HEAD": "n82"},
            clear=True,
        ):
            self.assertIsNone(read())
        with patch.dict(
            os.environ,
            {"PRIME_QUALIFICATION_SESSION_ID": "1" * 32},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                read()

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_qualification_input_digest_binds_tokens_lengths_and_order(self):
        train_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
        function = next(
            node
            for node in ast.parse(train_path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_batch_sha256"
        )
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"hashlib": hashlib, "json": json, "torch": torch}
        exec(compile(module, str(train_path), "exec"), namespace)
        digest = namespace["_qualification_batch_sha256"]
        batch = [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "sequence_lengths": [2, 2],
            },
            {
                "input_ids": torch.tensor([[5, 6]], dtype=torch.long),
                "sequence_lengths": [2],
            },
        ]
        original = digest(batch)
        self.assertEqual(original, digest(batch))
        changed_token = [dict(item) for item in batch]
        changed_token[0] = dict(batch[0])
        changed_token[0]["input_ids"] = batch[0]["input_ids"].clone()
        changed_token[0]["input_ids"][0, 0] = 9
        self.assertNotEqual(original, digest(changed_token))
        changed_lengths = [dict(item) for item in batch]
        changed_lengths[0] = dict(batch[0], sequence_lengths=[1, 3])
        self.assertNotEqual(original, digest(changed_lengths))
        self.assertNotEqual(original, digest(list(reversed(batch))))

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_qualification_step_records_reject_adjacent_input_reuse(self):
        train_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
        function = next(
            node
            for node in ast.parse(train_path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_step_records"
        )
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "math": __import__("math"),
            "re": __import__("re"),
            "torch": torch,
        }
        exec(compile(module, str(train_path), "exec"), namespace)
        make_records = namespace["_qualification_step_records"]
        history = []
        for step in (1, 2):
            history.extend(
                (
                    {"step": step, "data/rollout_rows": 2048},
                    {"step": step, "loss/mean": 0.5},
                    {"step": step, "optim/grad_norm": 0.8, "optim/lr": 8e-6},
                    {
                        "step": step,
                        "perf/peak_memory": 44.0,
                        "perf/forward_backward_peak_memory": 43.5,
                        "time/forward_backward": 300.0,
                    },
                )
            )
        values = {1: "1" * 64, 2: "2" * 64}
        self.assertEqual(
            [record["input_sha256"] for record in make_records(history, 2, values)],
            ["1" * 64, "2" * 64],
        )
        with self.assertRaisesRegex(RuntimeError, "reused"):
            make_records(history, 2, {1: "1" * 64, 2: "1" * 64})

    def test_qualification_launcher_binds_runtime_source_image_and_launcher(self):
        launcher = (ROOT / "scripts/china/cluster.sh").read_text()
        self.assertIn('expected_paths = set(mounted_paths)', launcher)
        self.assertIn('set(source_files) != expected_paths', launcher)
        self.assertNotIn('len(source_files) != 46', launcher)
        self.assertIn('candidate.read_bytes()', launcher)
        self.assertIn('qualification source checksum differs', launcher)
        self.assertIn('qualification launcher checksum differs', launcher)
        self.assertIn('qualification trainer image changed after session creation', launcher)
        self.assertIn('PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256=', launcher)
        self.assertIn('PRIME_QUALIFICATION_LAUNCHER_SHA256=', launcher)

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
        self.assertEqual(source.count("config.native_fp8_expert_prefetch"), 3)

        trainer_config = (
            ROOT
            / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
        ).read_text()
        self.assertIn("native_fp8_expert_prefetch: bool = False", trainer_config)

    def test_fsdp_param_comm_overlap_is_qualification_only_and_bounded(self):
        model_source = MODEL.read_text()
        trainer_config = (
            ROOT / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
        ).read_text()
        self.assertIn("fsdp_param_comm_overlap: bool = False", trainer_config)
        setup = next(
            node
            for node in ast.parse(model_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_fsdp"
        )
        setup_source = ast.unparse(setup)
        self.assertLess(
            setup_source.index("_validate_fsdp_param_comm_overlap"),
            setup_source.index("fully_shard("),
        )
        self.assertIn("prefetch_modules = prefetch_modules[:1]", setup_source)
        self.assertIn("_configure_fsdp_param_comm_overlap", setup_source)

        profile = tomllib.loads(
            (ROOT / "configs/qualification/glm52-native-fsdp-overlap.toml").read_text()
        )
        self.assertTrue(profile["model"]["fsdp_param_comm_overlap"])
        for relative in (
            "configs/paused/trainer.glm52-production.toml",
            "configs/qualification/glm52-native-b2048.toml",
            "configs/qualification/glm52-native-stability20.toml",
        ):
            current = tomllib.loads((ROOT / relative).read_text())
            self.assertFalse(current["model"].get("fsdp_param_comm_overlap", False), relative)

    def test_fsdp_param_comm_overlap_requires_supported_runtime_and_shape(self):
        tree = ast.parse(MODEL.read_text())
        selected = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_validate_fsdp_param_comm_overlap",
                "_configure_fsdp_param_comm_overlap",
            }
        ]
        module = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
            "nn": type("NN", (), {"Module": object}),
            "ModelConfig": object,
            "ParallelDims": object,
            "get_logger": lambda: type("Logger", (), {"info": lambda *_: None})(),
        }
        exec(compile(module, str(MODEL), "exec"), namespace)

        config = type(
            "Config",
            (),
            {
                "fsdp_param_comm_overlap": True,
                "reshard_after_forward": True,
                "fsdp_cpu_offload": False,
                "cold_cpu_offload": False,
                "optim_cpu_offload": False,
                "pp_schedule": "1f1b",
                "ep_comm_backend": "torch",
                "ac": type(
                    "AC",
                    (),
                    {
                        "mode": "reentrant_attention_all_experts",
                        "keep_routed_expert_activations_from_pp_stage": 5,
                    },
                )(),
            },
        )()
        dims = type(
            "Dims",
            (),
            {
                "ep_enabled": True,
                "pp": 8,
                "cp": 8,
                "ep": 8,
                "dp_replicate": 1,
                "global_ep": False,
            },
        )()
        calls = []

        class Model:
            def _set_unshard_async_op(self, value):
                calls.append(value)

        namespace["_configure_fsdp_param_comm_overlap"](Model(), config, dims, True)
        self.assertEqual(calls, [True])

        with self.assertRaisesRegex(ValueError, "native FP8"):
            namespace["_configure_fsdp_param_comm_overlap"](Model(), config, dims, False)
        with self.assertRaisesRegex(ValueError, "PP8/CP8/EP8"):
            bad_dims = type("Dims", (), {**dims.__dict__, "cp": 4})()
            namespace["_configure_fsdp_param_comm_overlap"](Model(), config, bad_dims, True)
        with self.assertRaisesRegex(RuntimeError, "FSDP2"):
            namespace["_configure_fsdp_param_comm_overlap"](object(), config, dims, True)

    def test_fsdp_param_comm_overlap_limits_nested_prefetch_to_one_target(self):
        helper = next(
            node
            for node in ast.parse(MODEL.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_set_bounded_nested_backward_prefetch"
        )
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)

        class FakeFSDP:
            def __init__(self):
                self.targets = []

            def set_modules_to_backward_prefetch(self, modules):
                self.targets.append(list(modules))

        class FakeMoE:
            pass

        namespace = {
            "MoE": FakeMoE,
            "LatentMoE": type("LatentMoE", (), {}),
            "FSDPModule": FakeFSDP,
            "nn": type("NN", (), {"Module": object}),
        }
        exec(compile(module, str(MODEL), "exec"), namespace)

        def block():
            current = type("Block", (), {})()
            current.mlp = FakeMoE()
            current.mlp.router = FakeFSDP()
            current.mlp.experts = FakeFSDP()
            current.mlp.shared_expert = FakeFSDP()
            return current

        previous, current = block(), block()
        namespace["_set_bounded_nested_backward_prefetch"](current, previous, False)
        self.assertEqual(current.mlp.router.targets, [[previous.mlp.router]])
        self.assertEqual(current.mlp.experts.targets, [[current.mlp.experts]])
        self.assertEqual(current.mlp.shared_expert.targets, [[current.mlp.shared_expert]])
        self.assertTrue(all(len(targets) == 1 for group in (current.mlp.router, current.mlp.experts, current.mlp.shared_expert) for targets in group.targets))

        namespace["_set_bounded_nested_backward_prefetch"](current, previous, True)
        self.assertEqual(current.mlp.experts.targets[-1], [previous.mlp.experts])

    def test_prefetch_pairs_stop_at_virtual_stage_boundaries(self):
        tree = ast.parse(MODEL.read_text())
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_virtual_stage_blocks", "_virtual_stage_prefetch_pairs"}
        ]
        module = ast.Module(body=helpers, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "RuntimeError": RuntimeError,
            "nn": type("NN", (), {"Module": object}),
        }
        exec(compile(module, str(MODEL), "exec"), namespace)

        class Layer:
            def __init__(self, index):
                self.index = index

        layers = {str(index): Layer(index) for index in (0, 1, 8, 9)}
        language_model = type(
            "LanguageModel",
            (),
            {"layers": layers, "pp_layer_ranges": ((0, 2), (8, 10))},
        )()
        groups, forward, backward = namespace["_virtual_stage_prefetch_pairs"](language_model)

        self.assertEqual([[layer.index for layer in group] for group in groups], [[0, 1], [8, 9]])
        self.assertEqual([(left.index, right.index) for left, right in forward], [(0, 1), (8, 9)])
        self.assertEqual([(left.index, right.index) for left, right in backward], [(9, 8), (1, 0)])

    def test_small_profile_uses_late_stage_local_recompute(self):
        with (ROOT / "configs/qualification/glm52-native-small.toml").open("rb") as stream:
            config = tomllib.load(stream)
        self.assertEqual(
            config["model"]["quantization"]["expert_activation_transport"],
            "bf16_forward_fp8_local_recompute_backward",
        )
        self.assertEqual(
            config["model"]["ac"]["keep_routed_expert_activations_from_pp_stage"],
            5,
        )

    def test_b2048_profile_uses_64_full_context_microbatches(self):
        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        relative = operations["profiles"]["glm52-native-b2048"]["trainer"]
        with (ROOT / relative).open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["max_steps"], 2)
        self.assertEqual(
            config["data"],
            {
                "fake": {
                    "batch_size": 64,
                    "generate_samples": True,
                    "samples_per_micro_batch": 32,
                    "group_size": 16,
                }
            },
        )
        self.assertEqual(
            config["data"]["fake"]["batch_size"]
            * config["data"]["fake"]["samples_per_micro_batch"],
            2048,
        )
        self.assertEqual(config["optim"]["lr"], 8e-6)
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
            config["data"],
            {
                "fake": {
                    "batch_size": 64,
                    "generate_samples": True,
                    "samples_per_micro_batch": 32,
                    "group_size": 16,
                    "export_final_adapter": True,
                }
            },
        )
        self.assertEqual(
            config["data"]["fake"]["batch_size"]
            * config["data"]["fake"]["samples_per_micro_batch"],
            2048,
        )
        self.assertEqual(config["model"]["seq_len"], 32768)
        self.assertEqual(config["model"]["pp"], 8)
        self.assertEqual(config["model"]["cp"], 8)
        self.assertEqual(config["model"]["ep"], 8)
        self.assertIs(config["model"]["reshard_after_forward"], True)
        self.assertFalse(config["model"]["native_fp8_expert_prefetch"])
        self.assertTrue(config["model"]["freeze_moe_router"])
        self.assertEqual(config["optim"]["lr"], 8e-6)
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
        self.assertIn("def _glm_router_fsdp_policy", model_source)
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
        self.assertIn(
            "apply_pp_layers(model, pp_rank, parallel_dims.pp, virtual_stages=virtual_stages)",
            setup_source,
        )
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
        self.assertIn(
            "restore_glm_moe_bias_values(model, snapshot_path, parallel_dims)",
            load_source,
        )
        load_node = next(
            node
            for node in ast.parse(model_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "load_dcp_from_hf"
        )
        restore_expressions = [
            node
            for node in load_node.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "restore_glm_moe_bias_values"
        ]
        self.assertEqual(len(restore_expressions), 1)
        lora_loop = next(
            node
            for node in ast.walk(load_node)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "module"
        )
        self.assertGreater(restore_expressions[0].lineno, lora_loop.end_lineno)
        self.assertIn("source.dtype != torch.float32", ast.unparse(ast.parse(model_source)))
        self.assertIn("_collect_live_glm_moe_bias_layers", model_source)
        self.assertIn("cache_glm_moe_bias_values", model_source)
        self.assertIn("destination differs from exact source after setup", model_source)
        self.assertIn("_glm_freeze_moe_router", model_source)
        self.assertIn("_use_static_expert_bias", model_source)
        self.assertIn("_expert_bias_updates_disabled", model_source)
        self.assertIn("_fp32_expert_bias_source_sha256", model_source)
        self.assertIn("static routing correction changed after source capture", model_source)
        setup_source = ast.unparse(
            next(
                node
                for node in ast.parse(model_source).body
                if isinstance(node, ast.FunctionDef) and node.name == "setup_model"
            )
        )
        self.assertIn("_reset_runtime_moe_buffers(model)", setup_source)
        self.assertIn("validate_glm_moe_bias_values(model, parallel_dims)", setup_source)
        moe_source = (ROOT / "source/src/prime_rl/trainer/models/layers/moe.py").read_text()
        self.assertIn("_require_static_expert_bias(self)", moe_source)
        self.assertEqual(moe_source.count("_require_static_expert_bias(self)"), 2)
        self.assertIn("selection_bias = expert_bias", moe_source)
        self.assertIn("selection_scores = selection_scores + selection_bias.to(torch.float32)", moe_source)
        self.assertIn("_move_static_expert_bias(self, x.device)", moe_source)
        self.assertIn("_move_static_expert_bias(self, x_flat.device)", moe_source)
        self.assertIn("to(device=device, dtype=torch.float32)", moe_source)
        self.assertIn('static_expert_bias = getattr(self, "_use_static_expert_bias", False)', moe_source)
        self.assertIn("if self.load_balance_coeff is not None:", moe_source)
        self.assertIn("expert_bias_is_static=static_expert_bias", moe_source)
        self.assertIn("def _glm_router_fsdp_policy", model_source)
        self.assertIn('"cast_forward_inputs" not in inspect.signature(MixedPrecisionPolicy).parameters', model_source)
        self.assertIn("policy_args[\"cast_forward_inputs\"] = not frozen_glm", model_source)
        self.assertIn("return None if frozen_glm else MixedPrecisionPolicy(**policy_args)", model_source)
        self.assertIn("if router_policy is not None:", model_source)

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
        preserve_position = train_source.index("preserve_glm_moe_bias_precision(model)", load_position - 500)
        restore_position = train_source.index(
            "restore_glm_moe_bias_values(model, base_snapshot, parallel_dims)",
            load_position,
        )
        validate_position = train_source.index("validate_glm_moe_bias_values(model, parallel_dims)", load_position)
        self.assertLess(preserve_position, load_position)
        self.assertLess(load_position, restore_position)
        self.assertLess(restore_position, validate_position)
        self.assertIn("if config.model.freeze_moe_router:", train_source[load_position:validate_position])
        self.assertIn("else:\n            cache_glm_moe_bias_values(model)", train_source[load_position:validate_position])
        self.assertIn("if not base_snapshot.is_dir():", train_source[load_position:restore_position])
        self.assertIn("Frozen GLM resume requires the local base model snapshot", train_source)

    @unittest.skipUnless(torch is not None, "FP32 checkpoint restoration requires PyTorch")
    def test_glm_router_bias_restores_exact_checkpoint_values(self):
        from safetensors.torch import save_file

        from prime_rl.trainer.model import restore_glm_moe_bias_values, validate_glm_moe_bias_values
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
                self._glm_freeze_moe_router = True
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
            model.model.layers[0].mlp.expert_bias = model.model.layers[0].mlp.expert_bias.to(torch.bfloat16)
            model.model.layers[0].mlp.expert_bias.copy_(source)
            restore_glm_moe_bias_values(model, root)
            self.assertEqual(model.model.layers[0].mlp.expert_bias.dtype, torch.bfloat16)
            torch.testing.assert_close(model.model.layers[0].mlp._fp32_expert_bias, source, rtol=0, atol=0)
            self.assertEqual(
                model.model.layers[0].mlp._fp32_expert_bias_source_sha256,
                model.model.layers[0].mlp._fp32_expert_bias_cache_sha256,
            )
            changed = source + torch.tensor([0.123, 0.0, -0.25, 0.5])
            model.model.layers[0].mlp.expert_bias.copy_(changed)
            validate_glm_moe_bias_values(model)
            torch.testing.assert_close(
                model.model.layers[0].mlp._fp32_expert_bias, source, rtol=0, atol=0
            )

    @unittest.skipUnless(torch is not None, "Frozen router behavior requires PyTorch")
    def test_frozen_router_uses_static_bias_and_stops_load_balance_updates(self):
        from prime_rl.trainer.model import cache_glm_moe_bias_values, freeze_moe_router
        from prime_rl.trainer.models.layers.moe import MoE, MoEArgs, _require_static_expert_bias

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 3
                self.mlp = MoE(
                    MoEArgs(num_experts=4, top_k=1, num_shared_experts=0),
                    dim=2,
                    hidden_dim=4,
                )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"model_type": "glm_moe_dsa"})()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer()])

        class RecordingRouter(torch.nn.Module):
            route_scale = 1.0

            def __init__(self, seen):
                super().__init__()
                self.seen = seen

            def forward(self, x, expert_bias, routed_experts=None, **kwargs):
                self.seen.append(expert_bias.detach().clone())
                count = x.shape[0]
                scores = torch.ones(count, 1, dtype=torch.float32)
                indices = torch.zeros(count, 1, dtype=torch.long)
                counts = torch.zeros(4, dtype=torch.float32)
                counts[0] = count
                return scores, indices, counts, torch.tensor(float(count))

        model = Model()
        mlp = model.model.layers[0].mlp
        source = torch.tensor([34.18921, 34.60598, 34.601982, 34.60798], dtype=torch.float32)
        mlp.expert_bias.copy_(source)
        freeze_moe_router(model)
        cache_glm_moe_bias_values(model)
        self.assertIsNone(mlp.load_balance_coeff)
        self.assertTrue(mlp._expert_bias_updates_disabled)
        self.assertNotIn("_fp32_expert_bias", mlp._buffers)
        self.assertNotIn("_fp32_expert_bias", model.state_dict())
        torch.testing.assert_close(mlp._fp32_expert_bias, source, rtol=0, atol=0)
        self.assertIsNotNone(mlp._fp32_expert_bias_validation)
        torch.testing.assert_close(_require_static_expert_bias(mlp), source, rtol=0, atol=0)
        mlp._fp32_expert_bias.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "changed after source capture"):
            cache_glm_moe_bias_values(model)
        with self.assertRaisesRegex(RuntimeError, "cache changed after validation"):
            mlp(torch.ones(1, 2, 2))
        delattr(mlp, "_fp32_expert_bias")
        with self.assertRaisesRegex(RuntimeError, "requires an FP32 bias cache"):
            mlp(torch.ones(1, 2, 2))
        mlp._fp32_expert_bias = source.clone()
        mlp._fp32_expert_bias_cache_sha256 = None
        with self.assertRaisesRegex(RuntimeError, "hashes are incomplete"):
            mlp(torch.ones(1, 2, 2))
        mlp._fp32_expert_bias_cache_sha256 = mlp._fp32_expert_bias_source_sha256
        mlp._fp32_expert_bias.copy_(source)
        cache_glm_moe_bias_values(model)
        self.assertIsNotNone(mlp._fp32_expert_bias_validation)
        torch.testing.assert_close(_require_static_expert_bias(mlp), source, rtol=0, atol=0)

        seen = []
        mlp.router = RecordingRouter(seen)
        mlp._run_routed_experts = lambda x, indices, counts, scores: torch.zeros_like(x)
        for step in range(3):
            mlp.expert_bias.add_(float(step + 1))
            mlp(torch.ones(1, 2, 2))
        self.assertEqual(len(seen), 3)
        for value in seen:
            torch.testing.assert_close(value, source, rtol=0, atol=0)
        self.assertEqual(float(mlp.tokens_per_expert.sum()), 0.0)

    @unittest.skipUnless(torch is not None, "Static bias transfer requires PyTorch")
    def test_static_bias_transfer_ignores_bf16_registered_buffer(self):
        from prime_rl.trainer.model import _glm_moe_bias_sha256
        from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
        from prime_rl.trainer.models.layers.moe import _move_static_expert_bias

        mlp = MoE(MoEArgs(num_experts=4, top_k=1), dim=2, hidden_dim=4)
        exact = torch.tensor([34.18921, 34.60598, 34.601982, 34.60798], dtype=torch.float32)
        mlp.expert_bias = mlp.expert_bias.to(torch.bfloat16)
        mlp._fp32_expert_bias = exact.clone()
        digest = _glm_moe_bias_sha256(exact)
        mlp._fp32_expert_bias_source_sha256 = digest
        mlp._fp32_expert_bias_cache_sha256 = digest
        applied = _move_static_expert_bias(mlp, torch.device("cpu"))
        torch.testing.assert_close(applied, exact, rtol=0, atol=0)
        self.assertEqual(applied.dtype, torch.float32)

    @unittest.skipUnless(torch is not None, "Static bias validation requires PyTorch")
    def test_static_bias_hashes_once_per_transfer_not_per_forward(self):
        from prime_rl.trainer.model import _glm_moe_bias_sha256
        from prime_rl.trainer.models.layers.moe import (
            MoE,
            MoEArgs,
            _move_static_expert_bias,
            _require_static_expert_bias,
        )

        mlp = MoE(MoEArgs(num_experts=4, top_k=1), dim=2, hidden_dim=4)
        exact = torch.tensor([34.18921, 34.60598, 34.601982, 34.60798], dtype=torch.float32)
        mlp._fp32_expert_bias = exact.clone()
        digest = _glm_moe_bias_sha256(exact)
        mlp._fp32_expert_bias_source_sha256 = digest
        mlp._fp32_expert_bias_cache_sha256 = digest
        real_sha256 = hashlib.sha256
        with patch("prime_rl.trainer.models.layers.moe.hashlib.sha256") as sha256:
            sha256.side_effect = real_sha256
            _move_static_expert_bias(mlp, torch.device("cpu"))
            after_transfer = sha256.call_count
            _require_static_expert_bias(mlp)
            _require_static_expert_bias(mlp)
            self.assertEqual(sha256.call_count, after_transfer)
            self.assertEqual(after_transfer, 1)

    @unittest.skipUnless(torch is not None, "Dynamic router behavior requires PyTorch")
    def test_non_frozen_router_uses_registered_bias_without_static_cache(self):
        from prime_rl.trainer.model import cache_glm_moe_bias_values
        from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 3
                self.mlp = MoE(
                    MoEArgs(num_experts=4, top_k=1, num_shared_experts=0),
                    dim=2,
                    hidden_dim=4,
                )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"model_type": "glm_moe_dsa"})()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer()])

        model = Model()
        mlp = model.model.layers[0].mlp
        model._glm_freeze_moe_router = False
        cache_glm_moe_bias_values(model)
        self.assertFalse(hasattr(mlp, "_fp32_expert_bias"))
        self.assertFalse(mlp._use_static_expert_bias)
        self.assertFalse(mlp._expert_bias_updates_disabled)

        seen = []

        class RecordingRouter(torch.nn.Module):
            route_scale = 1.0

            def forward(self, x, expert_bias, routed_experts=None, **kwargs):
                seen.append(expert_bias.detach().clone())
                count = x.shape[0]
                scores = torch.ones(count, 1, dtype=torch.float32)
                indices = torch.zeros(count, 1, dtype=torch.long)
                counts = torch.zeros(4, dtype=torch.float32)
                counts[0] = count
                return scores, indices, counts, torch.tensor(float(count))

        mlp.router = RecordingRouter()
        mlp._run_routed_experts = lambda x, indices, counts, scores: torch.zeros_like(x)
        dynamic = torch.tensor([1.25, 2.5, 3.75, 5.0], dtype=torch.float32)
        mlp.expert_bias.copy_(dynamic)
        mlp(torch.ones(1, 2, 2))
        torch.testing.assert_close(seen[0], dynamic, rtol=0, atol=0)

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
