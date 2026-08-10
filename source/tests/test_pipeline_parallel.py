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
        self.dtype = object()
        self.moves: list[tuple[tuple, dict]] = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def reshape(self, *args):
        return self

    def view(self, *args):
        return self

    def float(self):
        return self

    def bitwise_and_(self, *args):
        return self

    def requires_grad_(self, *args):
        return self

    def __getitem__(self, key):
        return self

    def __mul__(self, other):
        return self

    def __add__(self, other):
        return self


def _load(path: pathlib.Path, name: str):
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = _Module
    torch.nn = nn
    torch.Tensor = _Tensor
    torch.int32 = object()
    torch.int16 = object()
    torch.float16 = object()
    torch.is_tensor = lambda value: isinstance(value, _Tensor)
    torch.nan_to_num = lambda value: value

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
        self.assertTrue(
            all(GLM_INDEXER_TYPES[start] == "full" for start in bounds[:-1])
        )

    def test_twelve_layers_is_the_smallest_cache_safe_maximum(self):
        allowed = [i for i, kind in enumerate(GLM_INDEXER_TYPES) if kind == "full"]
        interior = [i for i in allowed if 0 < i < 78]
        best = min(
            max(end - start for start, end in zip((0, *cuts), (*cuts, 78)))
            for cuts in itertools.combinations(interior, 7)
        )
        self.assertEqual(best, 12)

    def test_sparse_cache_transfer_enables_ten_layer_compute_plan(self):
        config = types.SimpleNamespace(
            indexer_types=GLM_INDEXER_TYPES,
            num_hidden_layers=78,
            pp_transfer_sparse_indices=True,
        )
        bounds = [
            self.pp.stage_layer_range(78, rank, 8, config)[0] for rank in range(8)
        ] + [78]
        self.assertEqual(bounds, [0, 9, 18, 28, 38, 48, 58, 68, 78])
        self.assertEqual(
            max(end - start for start, end in zip(bounds, bounds[1:])),
            10,
        )

    def test_current_plan_is_the_unique_minimum_of_measured_memory_model(self):
        allowed = [i for i, kind in enumerate(GLM_INDEXER_TYPES) if kind == "full"]
        interior = [i for i in allowed if 0 < i < 78]

        def maximum_cost(cuts):
            bounds = (0, *cuts, 78)
            sizes = [end - start for start, end in zip(bounds, bounds[1:])]
            return max(
                size * (1.50 + 0.137 * (8 - stage)) for stage, size in enumerate(sizes)
            )

        ranked = sorted(
            (maximum_cost(cuts), cuts) for cuts in itertools.combinations(interior, 7)
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


class SideInputFifoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner, cls.torch = _load(
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py",
            "pp_runner_under_test",
        )

    def test_side_inputs_are_consumed_once_in_forward_order(self):
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

        got0 = stage._next_side()
        got1 = stage._next_side()

        self.assertEqual((got0["marker"], got1["marker"]), ("micro-0", "micro-1"))
        self.assertIsNone(stage._side[0])
        self.assertIsNone(stage._side[1])
        self.assertEqual(first.moves, [(("cuda",), {"non_blocking": True})])
        self.assertEqual(second.moves, [(("cuda",), {"non_blocking": True})])
        self.assertEqual(
            routed.moves,
            [
                (("cuda",), {"non_blocking": True}),
                ((self.torch.int32,), {}),
            ],
        )

    def test_middle_stage_transfers_sparse_index_cache(self):
        hidden = _Tensor("hidden")
        incoming = _Tensor("incoming-cache")
        outgoing = _Tensor("outgoing-cache")
        position_ids = _Tensor("position-ids")
        seen = {}

        class Decoder:
            def __call__(self, **kwargs):
                seen.update(kwargs)
                return types.SimpleNamespace(
                    last_hidden_state=hidden,
                    past_key_values=outgoing,
                )

        stage = self.runner.PipelineStageModule(
            model=types.SimpleNamespace(model=Decoder()),
            is_first=False,
            is_last=False,
            transfer_sparse_indices=True,
        )
        stage.set_side_inputs(
            [
                {
                    "position_ids": position_ids,
                    "seq_lens": None,
                    "seq_lens_are_pre_shard": True,
                }
            ]
        )
        got = stage._forward_inner(hidden, incoming)

        self.assertEqual(got, (hidden, outgoing))
        self.assertIs(seen["inputs_embeds"], hidden)
        self.assertIs(seen["sparse_indices_cache"], incoming)

    def test_sparse_cache_transport_has_one_shared_index_group(self):
        self.assertEqual(
            self.runner.sparse_cache_transport_shape(4096, 2048),
            (1, 4096, 1, 2048),
        )


if __name__ == "__main__":
    unittest.main()
