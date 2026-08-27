"""CPU checks for the opt-in GLM-5.2 interleaved qualification profiles."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "source/src/prime_rl/trainer/distributed/pp_runner.py"


def _load_runner():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    torch.nn = nn
    torch.Tensor = object
    logger = types.ModuleType("prime_rl.utils.logger")
    logger.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)
    modules = {
        "torch": torch,
        "torch.nn": nn,
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.utils": types.ModuleType("prime_rl.utils"),
        "prime_rl.utils.logger": logger,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("candidate_pp_runner", RUNNER_PATH)
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(runner)
    return runner


def _load_config(relative: str) -> dict:
    import tomllib

    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def _action(stage: int, microbatch: int, kind: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        stage_index=stage,
        microbatch_index=microbatch,
        computation_type=types.SimpleNamespace(value=kind),
        sub_actions=None,
    )


def _complete_virtual_schedule(microbatches: int = 64) -> dict[int, list]:
    actions = {}
    for rank in range(8):
        rank_actions = []
        for stage in (rank, rank + 8):
            for microbatch in range(microbatches):
                if stage > 0:
                    rank_actions.append(_action(stage, microbatch, "RECV_F"))
                rank_actions.append(_action(stage, microbatch, "F"))
                if stage < 15:
                    rank_actions.append(_action(stage, microbatch, "RECV_B"))
                rank_actions.append(_action(stage, microbatch, "B"))
        actions[rank] = rank_actions
    return actions


class InterleavedCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_profiles_are_opt_in_and_preserve_training_invariants(self):
        import tomllib

        with (ROOT / "configs/operations.toml").open("rb") as stream:
            operations = tomllib.load(stream)
        target_modules = {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "shared_expert",
            "experts",
        }
        for name in (
            "glm52-native-interleaved-small",
            "glm52-native-interleaved-b2048",
        ):
            relative = operations["profiles"][name]["trainer"]
            config = _load_config(relative)
            model = config["model"]
            quantization = model["quantization"]
            lora = model["lora"]
            self.assertEqual(model["pp_schedule"], "interleaved_1f1b")
            self.assertEqual(model["pp"], 8)
            self.assertEqual(model["cp"], 8)
            self.assertEqual(model["ep"], 8)
            self.assertEqual(model["pp_boundary_plan"], "balanced")
            self.assertNotIn("pp_layer_boundaries", model)
            self.assertTrue(model["freeze_moe_router"])
            self.assertEqual(quantization["type"], "fp8")
            self.assertFalse(quantization["dequantize_checkpoint"])
            self.assertEqual(
                quantization["expert_activation_transport"],
                "bf16_forward_fp8_recompute_backward",
            )
            self.assertEqual(lora["rank"], 32)
            self.assertEqual(lora["alpha"], 64)
            self.assertEqual(lora["dropout"], 0.0)
            self.assertEqual(set(lora["target_modules"]), target_modules)
            self.assertEqual(config["optim"]["type"], "muon")
            self.assertEqual(config["optim"]["weight_decay"], 0.0)
            self.assertEqual(config["optim"]["max_norm"], 1.0)
            self.assertNotIn("keep_routed_expert_activations_from_pp_stage", model["ac"])
            self.assertFalse(config["enable_router_replay"])
            self.assertNotIn("rollout_cache", config.get("data", {}))

        small = _load_config(
            "configs/qualification/glm52-native-interleaved-small.toml"
        )
        full = _load_config(
            "configs/qualification/glm52-native-interleaved-b2048.toml"
        )
        self.assertEqual(small["data"]["fake"]["batch_size"], 16)
        self.assertEqual(full["data"]["fake"]["batch_size"] * full["data"]["fake"]["samples_per_micro_batch"], 2048)
        self.assertEqual(full["data"]["fake"]["group_size"], 16)
        self.assertEqual(full["optim"]["lr"], 8e-6)

    def test_virtual_schedule_has_exact_microbatch_coverage(self):
        actions = _complete_virtual_schedule()
        positions = self.runner._schedule_action_positions(actions)
        for stage in range(16):
            for microbatch in range(64):
                self.assertIn((stage, microbatch, "F"), positions)
                self.assertIn((stage, microbatch, "B"), positions)
                if stage > 0:
                    self.assertIn((stage, microbatch, "RECV_F"), positions)
                if stage < 15:
                    self.assertIn((stage, microbatch, "RECV_B"), positions)

        capacities = self.runner.derive_interleaved_transport_capacities(
            actions,
            {stage: stage % 8 for stage in range(16)},
            num_stages=16,
            num_microbatches=64,
        )
        self.assertEqual(set(capacities), set(range(16)))
        self.assertTrue(all(capacities[stage]["forward"] > 0 for stage in range(1, 16)))
        self.assertTrue(all(capacities[stage]["backward"] > 0 for stage in range(15)))
        self.assertTrue(all(capacities[stage]["sparse"] > 0 for stage in range(1, 16)))

        missing = _complete_virtual_schedule()
        missing[3] = [
            item
            for item in missing[3]
            if not (
                item.stage_index == 3
                and item.microbatch_index == 11
                and item.computation_type.value == "F"
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete compute actions"):
            self.runner.derive_interleaved_transport_capacities(
                missing,
                {stage: stage % 8 for stage in range(16)},
                num_stages=16,
                num_microbatches=64,
            )

        duplicate = _complete_virtual_schedule()
        duplicate[3].append(_action(3, 11, "F"))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.runner.derive_interleaved_transport_capacities(
                duplicate,
                {stage: stage % 8 for stage in range(16)},
                num_stages=16,
                num_microbatches=64,
            )

    def test_modeled_transport_peak_is_below_measured_memory_ceiling(self):
        # Exact capacities from the lowered 16-stage GLM schedule. Hidden and
        # sparse-cache receive buffers are the schedule-owned activation memory.
        forward = [0, 16, 16, 16, 16, 14, 12, 10, 8, 8, 8, 8, 8, 6, 4, 2]
        backward = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        sparse = [0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2]
        hidden_slot = 4096 * 6144 * 2
        sparse_slot = 4096 * 2048 * 2
        stage_bytes = [
            (fwd + bwd) * hidden_slot + cache * sparse_slot
            for fwd, bwd, cache in zip(forward, backward, sparse)
        ]
        physical_bytes = [stage_bytes[rank] + stage_bytes[rank + 8] for rank in range(8)]
        physical_gib = [value / 2**30 for value in physical_bytes]
        self.assertEqual(physical_gib, [0.484375, 1.28125, 1.28125, 1.28125, 1.28125, 1.09375, 0.953125, 0.671875])
        modeled_peak = max(physical_gib)
        self.assertLessEqual(modeled_peak, 1.28125)
        # 43.2 GiB is the measured warm 1F1B peak. Adding the entire candidate
        # ring model is conservative because the baseline already has rings.
        self.assertLessEqual(43.2 + modeled_peak, 46.0)


if __name__ == "__main__":
    unittest.main()
