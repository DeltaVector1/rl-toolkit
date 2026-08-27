"""Focused GLM sparse-MLA construction and interleaved PP smoke test.

The production sparse MLA and PP splitter are loaded from this checkout.  The
small test replaces only the optional CUDA sparse kernel with a shape-preserving
stub so that the actual attention projections and forward path can run on CPU.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


_TEST_ROOT = os.environ.get("GLM_TEST_ROOT")
ROOT = Path(_TEST_ROOT) if _TEST_ROOT is not None else Path(__file__).resolve().parents[2]
SPARSE_PATH = ROOT / "source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py"
PIPELINE_PATH = ROOT / "source/src/prime_rl/trainer/distributed/pipeline_parallel.py"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_production_modules():
    """Load the two production modules with only unavailable runtime pieces stubbed."""
    package_names = (
        "prime_rl",
        "prime_rl.trainer",
        "prime_rl.trainer.models",
        "prime_rl.trainer.models.kernels",
        "prime_rl.trainer.models.layers",
        "prime_rl.trainer.models.layers.lora",
        "prime_rl.utils",
    )
    modules: dict[str, types.ModuleType] = {
        name: _package(name) for name in package_names
    }

    indexer = types.ModuleType("prime_rl.trainer.models.kernels.fp8_indexer")
    indexer.fp8_indexer = lambda *args, **kwargs: None
    modules[indexer.__name__] = indexer

    lora = types.ModuleType("prime_rl.trainer.models.layers.lora.multi_linear")

    class MultiLoRALinear(nn.Module):
        pass

    lora.MultiLoRALinear = MultiLoRALinear
    lora.fused_frozen_linear_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the smoke test uses ordinary projection layers")
    )
    modules[lora.__name__] = lora

    checkpointing = types.ModuleType("prime_rl.trainer.models.layers.checkpointing")
    checkpointing.current_reentrant_block_phase = lambda: None
    modules[checkpointing.__name__] = checkpointing

    norms = types.ModuleType("prime_rl.trainer.models.layers.norms")

    class RMSNorm(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(config.hidden_size))
            self.eps = config.eps

        def forward(self, value):
            variance = value.float().square().mean(dim=-1, keepdim=True)
            return value * torch.rsqrt(variance + self.eps) * self.weight

    class LayerNorm(nn.LayerNorm):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__(dim, eps=eps)

    norms.LayerNorm = LayerNorm
    norms.RMSNorm = RMSNorm
    norms.RMSNormConfig = types.SimpleNamespace
    modules[norms.__name__] = norms

    rotary = types.ModuleType("prime_rl.trainer.models.layers.rotary_emb")
    rotary.rotate_half = lambda value: torch.stack((-value[..., 1::2], value[..., ::2]), dim=-1).flatten(-2)
    modules[rotary.__name__] = rotary

    profiler = types.ModuleType("prime_rl.trainer.phase_profiler")

    class Profiler:
        enabled = False

        def end_span(self, _span):
            pass

    profiler.get_phase_profiler = lambda: Profiler()
    profiler.phase_span = lambda *_args, **_kwargs: (lambda function: function)
    modules[profiler.__name__] = profiler

    cp = types.ModuleType("prime_rl.utils.cp")
    cp.gather_for_cp = lambda value, _group: value
    modules[cp.__name__] = cp

    logger = types.ModuleType("prime_rl.utils.logger")
    logger.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)
    modules[logger.__name__] = logger

    language_model = types.ModuleType("prime_rl.trainer.model")
    language_model.get_language_model = lambda model: model
    modules[language_model.__name__] = language_model

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        sparse_spec = importlib.util.spec_from_file_location("_glm_sparse_under_test", SPARSE_PATH)
        assert sparse_spec is not None and sparse_spec.loader is not None
        sparse = importlib.util.module_from_spec(sparse_spec)
        sys.modules[sparse_spec.name] = sparse
        with mock.patch.object(torch, "compile", lambda function, **kwargs: function):
            sparse_spec.loader.exec_module(sparse)

        pipeline_spec = importlib.util.spec_from_file_location("_glm_pipeline_under_test", PIPELINE_PATH)
        assert pipeline_spec is not None and pipeline_spec.loader is not None
        pipeline = importlib.util.module_from_spec(pipeline_spec)
        sys.modules[pipeline_spec.name] = pipeline
        pipeline_spec.loader.exec_module(pipeline)
        return sparse, pipeline
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.self_attn = attention


class _TinyLanguageModel(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.config = types.SimpleNamespace()


class _TinyContainer(nn.Module):
    """Text-model shape expected by the production get_language_model helper."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.model = _TinyLanguageModel(layers)


class GlmInterleavedAttentionTest(unittest.TestCase):
    def test_split_attention_constructs_and_runs_first_forward(self):
        sparse, pipeline = _load_production_modules()
        device = torch.device(os.environ.get("GLM_TEST_DEVICE", "cpu"))
        if device.type == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        args = sparse.SparseMlaAttentionArgs(
            hidden_size=8,
            num_attention_heads=2,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_rope_head_dim=2,
            qk_nope_head_dim=2,
            qk_head_dim=4,
            v_head_dim=2,
            attention_bias=False,
            rms_norm_eps=1e-5,
            index_n_heads=1,
            index_head_dim=4,
            index_topk=64,
            use_index_cache=False,
            skip_topk=True,
        )

        def build_model():
            return _TinyContainer(
                [
                    _AttentionLayer(sparse.GlmMoeDsaAttention(args))
                    for _ in range(4)
                ]
            )

        baseline = build_model()
        pipeline.apply_pp_layers(baseline, pp_rank=0, pp_size=2, virtual_stages=1)
        interleaved = build_model()
        pipeline.apply_pp_layers(interleaved, pp_rank=0, pp_size=2, virtual_stages=2)
        baseline.to(device)
        interleaved.to(device)

        self.assertEqual(tuple(baseline.model.layers), ("0", "1"))
        self.assertEqual(tuple(interleaved.model.layers), ("0", "2"))
        self.assertEqual(interleaved.model.pp_layer_ranges, ((0, 1), (2, 3)))

        expected_scaling = args.qk_head_dim ** -0.5
        for model in (baseline, interleaved):
            for layer in model.model.layers.values():
                attention = layer.self_attn
                self.assertEqual(attention.scaling, expected_scaling)
                self.assertEqual(attention._cp_rank, 0)
                self.assertEqual(attention._cp_world_size, 1)

        calls: list[float] = []

        def fake_sparse_mla(query, _key_value, _indices, scale):
            calls.append(scale)
            return query[..., : args.kv_lora_rank]

        hidden = torch.randn(1, 2, args.hidden_size, device=device)
        positions = (
            torch.ones(1, 2, args.qk_rope_head_dim, device=device),
            torch.zeros(1, 2, args.qk_rope_head_dim, device=device),
        )
        with mock.patch.object(sparse._SparseMLA, "apply", staticmethod(fake_sparse_mla)):
            for model in (baseline, interleaved):
                for layer in model.model.layers.values():
                    output, cache = layer.self_attn(hidden, positions)
                    self.assertEqual(tuple(output.shape), (1, 2, args.hidden_size))
                    self.assertIsNone(cache)

        self.assertEqual(calls, [expected_scaling] * 4)


if __name__ == "__main__":
    unittest.main()
