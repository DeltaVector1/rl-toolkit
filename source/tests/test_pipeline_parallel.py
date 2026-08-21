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
H200_START = (SOURCE.parent / "scripts/h200/start_inference.sh").read_text()

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
    nn.ModuleDict = dict
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

    def test_native_mtp_is_not_enabled(self):
        self.assertNotIn('"speculative_config"', H200_START)

    def test_flashmla_stride_failure_is_documented(self):
        self.assertIn("2,415,919,104-byte stride", H200_START)
        self.assertIn("exceeds FlashMLA's int32 limit", H200_START)

class SideInputFifoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner, cls.torch = _load(
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py",
            "pp_runner_under_test",
        )

if __name__ == "__main__":
    unittest.main()
