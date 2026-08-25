"""CPU-only regression tests for the GLM pipeline plan.

Run with:
    python3 -m unittest source/tests/test_pipeline_parallel.py

The test imports only the pipeline planning code. It supplies small torch
stubs, so it does not need a GPU or the trainer environment.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
import types
import unittest
from unittest import mock


SOURCE = pathlib.Path(__file__).resolve().parents[1]

# This is the native GLM-5.2 IndexShare schedule from the checkpoint config.
# A pipeline stage can start only at a "full" layer. A "shared" layer needs
# cached sparse-MLA indices from the layer before it.
GLM_INDEXER_TYPES = [
    "full",
    "full",
    "full",
    "shared",
    "shared",
    "shared",
] + [kind for _ in range(18) for kind in ("full", "shared", "shared", "shared")]


class _Module:
    def __init__(self, *args, **kwargs):
        super().__init__()


class _Tensor:
    def __init__(self, name: str):
        self.name = name
        self.moves: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


def _load(path: pathlib.Path, name: str):
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = _Module
    torch.nn = nn
    torch.Tensor = _Tensor
    torch.int32 = object()
    torch.is_tensor = lambda value: isinstance(value, _Tensor)

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
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, torch


class PipelineBalanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pp, _ = _load(
            SOURCE / "src/prime_rl/trainer/distributed/pipeline_parallel.py",
            "pipeline_parallel_under_test",
        )

    def test_glm_pp8_plan_is_cache_safe_and_memory_balanced(self):
        config = types.SimpleNamespace(indexer_types=GLM_INDEXER_TYPES)
        allowed = self.pp._index_computing_layers(config)
        bounds = self.pp.balanced_boundaries(78, 8, allowed)

        self.assertEqual(bounds, [0, 6, 14, 22, 30, 42, 54, 66, 78])
        self.assertEqual(
            [end - start for start, end in zip(bounds, bounds[1:])],
            [6, 8, 8, 8, 12, 12, 12, 12],
        )
        self.assertTrue(all(GLM_INDEXER_TYPES[start] == "full" for start in bounds[:-1]))

    def test_twelve_layers_is_the_smallest_cache_safe_maximum(self):
        allowed = [i for i, kind in enumerate(GLM_INDEXER_TYPES) if kind == "full"]
        interior = [i for i in allowed if 0 < i < 78]
        best = min(
            max(end - start for start, end in zip((0, *cuts), (*cuts, 78)))
            for cuts in itertools.combinations(interior, 7)
        )
        self.assertEqual(best, 12)

    def test_current_plan_is_the_unique_minimum_of_measured_memory_model(self):
        allowed = [i for i, kind in enumerate(GLM_INDEXER_TYPES) if kind == "full"]
        interior = [i for i in allowed if 0 < i < 78]

        def maximum_cost(cuts):
            bounds = (0, *cuts, 78)
            sizes = [end - start for start, end in zip(bounds, bounds[1:])]
            return max(
                size * (1.50 + 0.137 * (8 - stage))
                for stage, size in enumerate(sizes)
            )

        ranked = sorted(
            (maximum_cost(cuts), cuts)
            for cuts in itertools.combinations(interior, 7)
        )
        self.assertAlmostEqual(ranked[0][0], 24.576)
        self.assertEqual(ranked[0][1], (6, 14, 22, 30, 42, 54, 66))
        self.assertAlmostEqual(
            next(cost for cost, _ in ranked if cost > ranked[0][0]),
            25.960,
        )

    def test_unsafe_even_boundaries_are_not_selected(self):
        even = self.pp.balanced_boundaries(78, 8, None)
        unsafe = {i for i in even[1:-1] if GLM_INDEXER_TYPES[i] == "shared"}
        self.assertEqual(unsafe, {20, 40, 60, 69})
        config = types.SimpleNamespace(indexer_types=GLM_INDEXER_TYPES)
        for rank in range(8):
            start, _ = self.pp.stage_layer_range(78, rank, 8, config)
            self.assertNotIn(start, unsafe)

    def test_impossible_cache_safe_plan_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "index-cache chain"):
            self.pp.balanced_boundaries(16, 8, [0, 8])

    def test_transferred_index_plan_accounts_for_embedding_and_head(self):
        config = types.SimpleNamespace(
            hidden_size=6144,
            vocab_size=154880,
            tie_word_embeddings=False,
        )
        self.assertEqual(
            self.pp.compute_balanced_boundaries(78, 8, config),
            [0, 8, 18, 28, 38, 48, 58, 68, 78],
        )

    def test_interleaved_plan_has_two_nonempty_ranges_per_rank(self):
        config = types.SimpleNamespace(
            hidden_size=6144,
            vocab_size=154880,
            tie_word_embeddings=False,
        )
        bounds = self.pp.compute_balanced_boundaries(78, 16, config)
        self.assertEqual(len(bounds), 17)
        ranges = [
            (bounds[rank], bounds[rank + 1])
            for rank in range(16)
        ]
        self.assertTrue(all(start < end for start, end in ranges))
        self.assertEqual(sum(end - start for start, end in ranges), 78)
        for physical_rank in range(8):
            self.assertTrue(ranges[physical_rank + 8][0] >= ranges[physical_rank][1])


class SideInputKeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner, cls.torch = _load(
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py",
            "pp_runner_under_test",
        )

    def test_side_inputs_are_consumed_by_microbatch_key(self):
        first = _Tensor("first")
        second = _Tensor("second")
        routed = _Tensor("routed")
        stage = self.runner.PipelineStageModule(
            model=object(),
            is_first=False,
            is_last=False,
        )
        stage.set_side_inputs(
            [
                {"marker": "micro-0", "value": first},
                {"marker": "micro-1", "value": second, "routed_experts": routed},
            ]
        )

        got1 = stage._next_side(1)
        got0 = stage._next_side(0)

        self.assertEqual((got0["marker"], got1["marker"]), ("micro-0", "micro-1"))
        self.assertEqual(stage._side, {})
        self.assertEqual(first.moves, [(('cuda',), {"non_blocking": True})])
        self.assertEqual(second.moves, [(('cuda',), {"non_blocking": True})])
        self.assertEqual(
            routed.moves,
            [
                (("cuda",), {"non_blocking": True}),
                ((self.torch.int32,), {}),
            ],
        )

    def test_missing_and_duplicate_side_inputs_fail_closed(self):
        stage = self.runner.PipelineStageModule(
            model=object(), is_first=False, is_last=False, stage_identity=3
        )
        stage.set_side_inputs({(3, 7): {"marker": "micro-7"}})
        self.assertEqual(stage._next_side(7)["marker"], "micro-7")
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            stage._next_side(7)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            stage._next_side(8)

    def test_interleaved_microbatch_count_is_validated(self):
        self.assertEqual(self.runner.validate_interleaved_microbatches(64, 8), 8)
        with self.assertRaisesRegex(ValueError, "at least 8"):
            self.runner.validate_interleaved_microbatches(7, 8)
        with self.assertRaisesRegex(ValueError, "multiple of the number of rounds"):
            self.runner.validate_interleaved_microbatches(13, 6)

    def test_interleaved_transport_capacity_parser_and_modulo_reuse(self):
        def action(stage, microbatch, kind):
            return types.SimpleNamespace(
                stage_index=stage,
                microbatch_index=microbatch,
                computation_type=types.SimpleNamespace(value=kind),
                sub_actions=None,
            )

        actions = {
            0: [
                action(0, 0, "F"),
                action(0, 1, "F"),
                action(0, 0, "RECV_B"),
                action(0, 0, "B"),
                action(0, 1, "RECV_B"),
                action(0, 1, "B"),
            ],
            1: [
                action(1, 0, "RECV_F"),
                action(1, 0, "F"),
                action(1, 1, "RECV_F"),
                action(1, 1, "F"),
                action(1, 0, "B"),
                action(1, 1, "B"),
            ],
        }
        capacities = self.runner.derive_interleaved_transport_capacities(
            actions,
            {0: 0, 1: 1},
            num_stages=2,
            num_microbatches=2,
        )
        self.assertEqual(
            capacities,
            {
                0: {"forward": 0, "backward": 1, "sparse": 0},
                1: {"forward": 2, "backward": 0, "sparse": 1},
            },
        )
        actions[1].append(action(1, 0, "F"))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.runner.derive_interleaved_transport_capacities(
                actions,
                {0: 0, 1: 1},
                num_stages=2,
                num_microbatches=2,
            )

    def test_glm_interleaved_ring_projection(self):
        forward_slots = [0, 16, 16, 16, 16, 14, 12, 10, 8, 8, 8, 8, 8, 6, 4, 2]
        backward_slots = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        sparse_slots = [0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2]
        hidden_slot_bytes = 4096 * 6144 * 2
        sparse_slot_bytes = 4096 * 2048 * 2
        exact_bytes = (
            (sum(forward_slots) + sum(backward_slots)) * hidden_slot_bytes
            + sum(sparse_slots) * sparse_slot_bytes
        )
        global_slot_bytes = (
            (15 * 16 + 15 * 16) * hidden_slot_bytes
            + (15 * 16) * sparse_slot_bytes
        )
        self.assertEqual(exact_bytes, 8.328125 * 2**30)
        self.assertEqual(global_slot_bytes, 26.25 * 2**30)
        self.assertLess(exact_bytes, global_slot_bytes)

    def test_sparse_cache_slot_reuse_does_not_block_the_cpu(self):
        source = (
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py"
        ).read_text()
        sparse_source = source.split("def _sparse_cache_pipeline_stage_cls", 1)[1]
        method = sparse_source.split("        def get_fwd_recv_ops", 1)[1].split(
            "        def forward_one_chunk", 1
        )[0]
        self.assertIn("event.synchronize()", method)
        self.assertNotIn("torch.cuda.current_stream().wait_event(event)", method)


if __name__ == "__main__":
    unittest.main()
