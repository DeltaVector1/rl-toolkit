"""CPU autograd and dequantization checks for the Ada FP8 training path."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FP8_ADA_PATH = REPOSITORY_ROOT / "source/src/prime_rl/trainer/models/layers/fp8_ada.py"
MULTI_LINEAR_PATH = (
    REPOSITORY_ROOT / "source/src/prime_rl/trainer/models/layers/lora/multi_linear.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_fp8_import_stubs() -> None:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except ImportError:
        triton = types.ModuleType("triton")
        triton_language = types.ModuleType("triton.language")
        triton_language.bfloat16 = object()
        triton_language.float16 = object()
        triton_language.dtype = type
        triton.language = triton_language
        sys.modules.setdefault("triton", triton)
        sys.modules.setdefault("triton.language", triton_language)

    lm_head = types.ModuleType("prime_rl.trainer.models.layers.lm_head")

    class PrimeLmOutput:
        def __init__(self, logprobs, entropy):
            self.logprobs = logprobs
            self.entropy = entropy

    def online_update(m, s, t, values):
        new_m = torch.maximum(m, values.max(dim=-1).values)
        old_factor = torch.exp(m - new_m)
        values_factor = torch.exp(values - new_m.unsqueeze(-1))
        new_s = s * old_factor + values_factor.sum(dim=-1)
        new_t = t * old_factor + (values_factor * values).sum(dim=-1)
        return new_m, new_s, new_t

    lm_head.PrimeLmOutput = PrimeLmOutput
    lm_head.cast_float_and_contiguous = lambda function: function
    lm_head._online_logsumexp_and_weighted_update = online_update
    sys.modules.setdefault("prime_rl.trainer.models.layers.lm_head", lm_head)

    custom_ops = types.SimpleNamespace(
        scaled_fp8_quant=lambda x, use_per_token_if_dynamic=True: (
            x.clone(),
            torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype),
        )
    )
    vllm = types.ModuleType("vllm")
    vllm._custom_ops = custom_ops
    sys.modules.setdefault("vllm", vllm)

    fused_moe = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    fused_moe._prepare_expert_assignment = lambda *args, **kwargs: (None, None, None)
    fused_moe.dispatch_fused_moe_kernel = lambda *args, **kwargs: None
    fused_moe.get_default_config = lambda *args, **kwargs: None
    fused_moe.moe_kernel_quantize_input = lambda *args, **kwargs: (None, None)
    sys.modules.setdefault(
        "vllm.model_executor", types.ModuleType("vllm.model_executor")
    )
    sys.modules.setdefault(
        "vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")
    )
    sys.modules.setdefault(
        "vllm.model_executor.layers.fused_moe",
        types.ModuleType("vllm.model_executor.layers.fused_moe"),
    )
    sys.modules.setdefault("vllm.model_executor.layers.fused_moe.fused_moe", fused_moe)


def _install_lora_import_stubs() -> None:
    base = types.ModuleType("prime_rl.trainer.models.layers.lora.base")

    class MultiLoRAModule(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer

    base.MultiLoRAModule = MultiLoRAModule
    base.get_lora_num_tokens = lambda: torch.tensor([0])
    base.get_multilora_scaling = lambda: torch.tensor([1.0])
    sys.modules.setdefault("prime_rl.trainer.models.layers.lora.base", base)


_install_fp8_import_stubs()
_install_lora_import_stubs()
fp8_ada = _load_module("_fp8_ada_under_test", FP8_ADA_PATH)
multi_linear = _load_module("_multi_linear_under_test", MULTI_LINEAR_PATH)


def _expanded_block_scale(scale: torch.Tensor, shape: tuple[int, int]):
    return scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[
        : shape[0], : shape[1]
    ]


class FP8AdaAutogradTest(unittest.TestCase):
    def test_dequant_weight_expands_each_128_square_block(self):
        weight = (
            torch.arange(256 * 384, dtype=torch.int32)
            .remainder(31)
            .sub(15)
            .reshape(256, 384)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.tensor([[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]], dtype=torch.float32)

        got = fp8_ada._dequant_weight(weight, scale, 128, 256, torch.float32)
        expected = weight[128:256].float() * _expanded_block_scale(
            scale[1:2], (128, 384)
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_dequant_weight_supports_partial_blocks(self):
        weight = (
            torch.arange(130 * 259, dtype=torch.int32)
            .remainder(31)
            .sub(15)
            .reshape(130, 259)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.tensor([[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]], dtype=torch.float32)
        got = fp8_ada._dequant_weight(weight, scale, 127, 130, torch.float32)
        expected = (
            weight[127:130].float() * _expanded_block_scale(scale, (130, 259))[127:130]
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_fp8_dx_contiguates_gradient_and_transposes_block_metadata(self):
        grad_storage = torch.arange(3 * 10, dtype=torch.float32).reshape(3, 10)
        grad = grad_storage[:, ::2]
        self.assertFalse(grad.is_contiguous())
        weight = torch.randn(5, 7)
        scale = torch.ones(1, 1)
        expected = grad @ weight
        seen = {}

        original_eligible = fp8_ada._native_sm89_eligible
        original_grouped = fp8_ada._grouped_mm
        fp8_ada._native_sm89_eligible = lambda *_args: (True, "")

        def fake_grouped(value, raw_weight, block_scale, offsets):
            seen["stride"] = value.stride()
            self.assertTrue(value.is_contiguous())
            torch.testing.assert_close(raw_weight[0], weight.t())
            torch.testing.assert_close(block_scale[0], scale.t())
            torch.testing.assert_close(offsets, torch.tensor([3], dtype=torch.int32))
            return F.linear(value, raw_weight[0])

        fp8_ada._grouped_mm = fake_grouped
        try:
            got = fp8_ada._linear_input_grad(
                grad, weight, scale, fqn="model.layers.7.self_attn.q_b_proj"
            )
        finally:
            fp8_ada._native_sm89_eligible = original_eligible
            fp8_ada._grouped_mm = original_grouped

        self.assertEqual(seen["stride"], (5, 1))
        torch.testing.assert_close(got, expected)

    def test_short_forward_uses_fused_w8a8_kernel(self):
        value = torch.randn(4, 3)
        weight = torch.randn(5, 3)
        scale = torch.ones(1, 1)
        sentinel = torch.full((4, 5), 23.0)

        original_native = fp8_ada._native_block_fp8_mm
        original_grouped = fp8_ada._grouped_mm

        def fail_native(*_args, **_kwargs):
            raise AssertionError("short forward used the BF16 direct kernel")

        fp8_ada._native_block_fp8_mm = fail_native
        fp8_ada._grouped_mm = lambda *_args, **_kwargs: sentinel
        try:
            got = fp8_ada._linear(value, weight, scale)
        finally:
            fp8_ada._native_block_fp8_mm = original_native
            fp8_ada._grouped_mm = original_grouped

        torch.testing.assert_close(got, sentinel)

    def test_long_forward_uses_fused_w8a8_kernel(self):
        rows = 257
        value = torch.randn(rows, 3)
        weight = torch.randn(5, 3)
        scale = torch.ones(1, 1)
        sentinel = torch.full((rows, 5), 29.0)

        original_native = fp8_ada._native_block_fp8_mm
        original_grouped = fp8_ada._grouped_mm

        def fail_native(*_args, **_kwargs):
            raise AssertionError("long SM89 forward used slower direct kernel")

        fp8_ada._native_block_fp8_mm = fail_native
        fp8_ada._grouped_mm = lambda *_args, **_kwargs: sentinel
        try:
            got = fp8_ada._linear(value, weight, scale)
        finally:
            fp8_ada._native_block_fp8_mm = original_native
            fp8_ada._grouped_mm = original_grouped

        torch.testing.assert_close(got, sentinel)

    def test_eligible_fp8_dx_uses_grouped_kernel_and_not_oracle(self):
        grad = torch.randn(2, 3)
        weight = torch.randn(3, 4)
        scale = torch.ones(1, 1)
        sentinel = torch.full((2, 4), 17.0)

        original_eligible = fp8_ada._native_sm89_eligible
        original_grouped = fp8_ada._grouped_mm
        original_oracle = fp8_ada._linear_input_grad_oracle
        fp8_ada._native_sm89_eligible = lambda *_args: (True, "")
        fp8_ada._grouped_mm = lambda *_args, **_kwargs: sentinel

        def fail_oracle(*_args, **_kwargs):
            raise AssertionError("eligible SM89 input used the oracle")

        fp8_ada._linear_input_grad_oracle = fail_oracle
        try:
            got = fp8_ada._linear_input_grad(
                grad,
                weight,
                scale,
                fqn="model.layers.11.self_attn.o_proj",
            )
        finally:
            fp8_ada._native_sm89_eligible = original_eligible
            fp8_ada._grouped_mm = original_grouped
            fp8_ada._linear_input_grad_oracle = original_oracle

        torch.testing.assert_close(got, sentinel)

    def test_fp8_linear_custom_backward_matches_dequantized_linear(self):
        torch.manual_seed(1)
        x = torch.randn(3, 256, dtype=torch.double, requires_grad=True)
        weight = torch.randn(384, 256, dtype=torch.double)
        scale = torch.rand(3, 2, dtype=torch.double) + 0.25
        upstream = torch.randn(3, 384, dtype=torch.double)

        original_linear = fp8_ada._linear

        def dequantized_linear(value, raw_weight, block_scale):
            expanded = _expanded_block_scale(
                block_scale, (raw_weight.shape[0], raw_weight.shape[1])
            )
            return F.linear(value, raw_weight * expanded)

        live_weight = weight.clone()
        live_scale = scale.clone()
        fp8_ada._linear = dequantized_linear
        try:
            got = fp8_ada._Linear.apply(
                x,
                live_weight,
                live_scale,
                "model.layers.0.self_attn.o_proj",
                None,
            )
            self.assertEqual(
                fp8_ada._LINEAR_SNAPSHOT_CACHE["model.layers.0.self_attn.o_proj"][2],
                1,
            )
            # Simulate FSDP RAF releasing the explicit forward inputs. Backward
            # must use the owned direct-context snapshots.
            live_weight.fill_(float("nan"))
            live_scale.fill_(float("nan"))
            got.backward(upstream)
        finally:
            fp8_ada._linear = original_linear
        self.assertEqual(fp8_ada._LINEAR_SNAPSHOT_CACHE, {})
        got_grad = x.grad.detach().clone()

        # Multiple 1F1B forwards share one frozen snapshot until their
        # matching backwards release the final reference.
        x0 = x.detach().clone().requires_grad_()
        x1 = x.detach().clone().requires_grad_()
        live_weight = weight.clone()
        live_scale = scale.clone()
        fp8_ada._linear = dequantized_linear
        try:
            got0 = fp8_ada._Linear.apply(
                x0,
                live_weight,
                live_scale,
                "model.layers.0.self_attn.o_proj",
                None,
            )
            got1 = fp8_ada._Linear.apply(
                x1,
                live_weight,
                live_scale,
                "model.layers.0.self_attn.o_proj",
                None,
            )
            entry = fp8_ada._LINEAR_SNAPSHOT_CACHE["model.layers.0.self_attn.o_proj"]
            self.assertEqual(entry[2], 2)
            self.assertNotEqual(entry[0].data_ptr(), live_weight.data_ptr())
            live_weight.fill_(float("nan"))
            live_scale.fill_(float("nan"))
            got0.backward(upstream)
            self.assertEqual(entry[2], 1)
            got1.backward(upstream)
        finally:
            fp8_ada._linear = original_linear
        self.assertEqual(fp8_ada._LINEAR_SNAPSHOT_CACHE, {})

        ref_x = x.detach().clone().requires_grad_()
        dequantized_linear(ref_x, weight, scale).backward(upstream)
        torch.testing.assert_close(got_grad, ref_x.grad, rtol=1e-12, atol=1e-12)

    def test_fp8_linear_snapshot_is_not_retained_without_input_gradient(self):
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        scale = torch.ones(1, 1)
        original_linear = fp8_ada._linear
        fp8_ada._linear = lambda value, raw_weight, _scale: F.linear(
            value, raw_weight
        )
        try:
            fp8_ada._Linear.apply(x, weight, scale, "probe.no_grad", None)
        finally:
            fp8_ada._linear = original_linear
        self.assertEqual(fp8_ada._LINEAR_SNAPSHOT_CACHE, {})

    def test_fp8_linear_forward_failure_releases_snapshot(self):
        x = torch.randn(3, 4, requires_grad=True)
        weight = torch.randn(5, 4)
        scale = torch.ones(1, 1)
        original_linear = fp8_ada._linear

        def fail(*_args):
            raise RuntimeError("injected forward failure")

        fp8_ada._linear = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                fp8_ada._Linear.apply(x, weight, scale, "probe.failure", None)
        finally:
            fp8_ada._linear = original_linear
        self.assertEqual(fp8_ada._LINEAR_SNAPSHOT_CACHE, {})

    def test_fp8_linear_reacquires_current_owner_parameter_storage(self):
        torch.manual_seed(37)
        x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        weight = torch.randn(5, 4, dtype=torch.double)
        scale = torch.ones(1, 1, dtype=torch.double)
        upstream = torch.randn(3, 5, dtype=torch.double)
        owner = types.SimpleNamespace(
            weight=weight.clone(),
            weight_scale_inv=scale.clone(),
        )
        stale_weight = owner.weight
        stale_scale = owner.weight_scale_inv
        original_linear = fp8_ada._linear

        def dense(value, raw_weight, _scale):
            return F.linear(value, raw_weight)

        fp8_ada._linear = dense
        try:
            got = fp8_ada._Linear.apply(
                x,
                stale_weight,
                stale_scale,
                "probe.live-owner",
                (owner, "weight", "weight_scale_inv"),
            )
            owner.weight = weight.clone()
            owner.weight_scale_inv = scale.clone()
            stale_weight.fill_(float("nan"))
            stale_scale.fill_(float("nan"))
            got.backward(upstream)
        finally:
            fp8_ada._linear = original_linear

        ref_x = x.detach().clone().requires_grad_()
        F.linear(ref_x, weight).backward(upstream)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
        self.assertEqual(fp8_ada._LINEAR_SNAPSHOT_CACHE, {})

    def test_fp8_expert_custom_backward_matches_dense_swiglu(self):
        torch.manual_seed(2)
        counts = torch.tensor([2, 3], dtype=torch.int32)
        offsets = counts.cumsum(0)
        x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(2, 6, 4, dtype=torch.double)
        w3 = torch.randn(2, 6, 4, dtype=torch.double)
        w2 = torch.randn(2, 4, 6, dtype=torch.double)
        dummy_scale = torch.ones(2, 1, 1, dtype=torch.double)
        upstream = torch.randn(5, 4, dtype=torch.double)

        original_grouped_mm = fp8_ada._grouped_mm

        def dense_grouped_mm(value, weight, scale, ends):
            outputs = []
            start = 0
            for expert, end in enumerate(ends.tolist()):
                outputs.append(F.linear(value[start:end], weight[expert]))
                start = end
            return torch.cat(outputs)

        fp8_ada._grouped_mm = dense_grouped_mm
        try:
            got = fp8_ada._Experts.apply(
                x,
                w1,
                dummy_scale,
                w2,
                dummy_scale,
                w3,
                dummy_scale,
                offsets,
                None,
            )
            got.backward(upstream)
        finally:
            fp8_ada._grouped_mm = original_grouped_mm
        got_grad = x.grad.detach().clone()

        ref_x = x.detach().clone().requires_grad_()
        pieces = []
        start = 0
        for expert, end in enumerate(offsets.tolist()):
            z1 = F.linear(ref_x[start:end], w1[expert])
            z3 = F.linear(ref_x[start:end], w3[expert])
            pieces.append(F.linear(F.silu(z1) * z3, w2[expert]))
            start = end
        torch.cat(pieces).backward(upstream)
        torch.testing.assert_close(got_grad, ref_x.grad, rtol=1e-12, atol=1e-12)

    def test_fp8_expert_reacquires_current_owner_parameter_storage(self):
        torch.manual_seed(41)
        counts = torch.tensor([2, 3], dtype=torch.int32)
        offsets = counts.cumsum(0)
        x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        base_w1 = torch.randn(2, 6, 4, dtype=torch.double)
        base_w2 = torch.randn(2, 4, 6, dtype=torch.double)
        base_w3 = torch.randn(2, 6, 4, dtype=torch.double)
        scale = torch.ones(2, 1, 1, dtype=torch.double)
        upstream = torch.randn(5, 4, dtype=torch.double)
        live = [
            base_w1.clone(),
            scale.clone(),
            base_w2.clone(),
            scale.clone(),
            base_w3.clone(),
            scale.clone(),
        ]
        owner = types.SimpleNamespace(
            w1=live[0],
            w1_scale_inv=live[1],
            w2=live[2],
            w2_scale_inv=live[3],
            w3=live[4],
            w3_scale_inv=live[5],
        )
        original_grouped_mm = fp8_ada._grouped_mm

        def dense_grouped_mm(value, weight, _scale, ends):
            outputs = []
            start = 0
            for expert, end in enumerate(ends.tolist()):
                outputs.append(F.linear(value[start:end], weight[expert]))
                start = end
            return torch.cat(outputs)

        fp8_ada._grouped_mm = dense_grouped_mm
        try:
            got = fp8_ada._Experts.apply(
                x,
                live[0],
                live[1],
                live[2],
                live[3],
                live[4],
                live[5],
                offsets,
                owner,
            )
            owner.w1 = base_w1.clone()
            owner.w1_scale_inv = scale.clone()
            owner.w2 = base_w2.clone()
            owner.w2_scale_inv = scale.clone()
            owner.w3 = base_w3.clone()
            owner.w3_scale_inv = scale.clone()
            for value in live:
                value.fill_(float("nan"))
            got.backward(upstream)
        finally:
            fp8_ada._grouped_mm = original_grouped_mm

        ref_x = x.detach().clone().requires_grad_()
        pieces = []
        start = 0
        for expert, end in enumerate(offsets.tolist()):
            z1 = F.linear(ref_x[start:end], base_w1[expert])
            z3 = F.linear(ref_x[start:end], base_w3[expert])
            pieces.append(F.linear(F.silu(z1) * z3, base_w2[expert]))
            start = end
        torch.cat(pieces).backward(upstream)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-12, atol=1e-12)

    def test_grouped_mm_pair_reuses_assignment_and_shared_input_quantization(self):
        calls = {"prepare": 0, "quantize": 0, "dispatch": 0}
        value = types.SimpleNamespace(is_cuda=True, shape=(7, 8))
        weight1 = types.SimpleNamespace(shape=(2, 16, 8))
        weight2 = types.SimpleNamespace(shape=(2, 16, 8))
        scale1, scale2, offsets = object(), object(), object()
        originals = (
            fp8_ada._prepare_grouped_mm,
            fp8_ada._quantize_grouped_mm_input,
            fp8_ada._dispatch_grouped_mm,
        )

        def fake_prepare(*args):
            calls["prepare"] += 1
            return ("config", "rows", "sorted", "experts", "padded")

        def fake_quantize(*args):
            calls["quantize"] += 1
            return "quantized", "activation-scale"

        def fake_dispatch(*args):
            calls["dispatch"] += 1
            return args[1]

        (
            fp8_ada._prepare_grouped_mm,
            fp8_ada._quantize_grouped_mm_input,
            fp8_ada._dispatch_grouped_mm,
        ) = (fake_prepare, fake_quantize, fake_dispatch)
        try:
            got = fp8_ada._grouped_mm_pair(
                value,
                weight1,
                scale1,
                value,
                weight2,
                scale2,
                offsets,
                shared_input=True,
            )
        finally:
            (
                fp8_ada._prepare_grouped_mm,
                fp8_ada._quantize_grouped_mm_input,
                fp8_ada._dispatch_grouped_mm,
            ) = originals

        self.assertEqual(got, (weight1, weight2))
        self.assertEqual(calls, {"prepare": 1, "quantize": 1, "dispatch": 2})

    def test_grouped_w2_chunking_preserves_expert_rows_and_padding(self):
        torch.manual_seed(23)
        x = torch.randn(13, 4)
        weight = torch.randn(2, 3, 4)
        scale = torch.ones(2, 1, 1)
        offsets = torch.tensor([6, 11], dtype=torch.int32)
        seen_rows = []

        original_grouped_mm = fp8_ada._grouped_mm

        def dense_grouped_mm(value, raw_weight, _scale, ends):
            seen_rows.append(value.shape[0])
            outputs = value.new_zeros(value.shape[0], raw_weight.shape[1])
            begin = 0
            for expert, finish in enumerate(ends.tolist()):
                outputs[begin:finish] = F.linear(
                    value[begin:finish], raw_weight[expert]
                )
                begin = finish
            return outputs

        fp8_ada._grouped_mm = dense_grouped_mm
        try:
            got = fp8_ada._grouped_mm_chunked(
                x,
                weight,
                scale,
                offsets,
                chunk_rows=5,
            )
        finally:
            fp8_ada._grouped_mm = original_grouped_mm

        expected = torch.cat(
            (
                F.linear(x[:6], weight[0]),
                F.linear(x[6:11], weight[1]),
                x.new_zeros(2, weight.shape[1]),
            )
        )
        self.assertEqual(seen_rows, [5, 5, 3])
        torch.testing.assert_close(got, expected)

    def test_logprob_and_entropy_backward_match_dense_softmax(self):
        torch.manual_seed(3)
        hidden = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(7, 5, dtype=torch.float32)
        scale = torch.rand(7, dtype=torch.float32) + 0.25
        labels = torch.tensor([0, 6, 2, 4])
        inv_temperature = torch.tensor([1.0, 0.7, 1.3, 0.9])
        grad_logprob = torch.tensor([0.4, -0.2, 0.7, -0.5])
        grad_entropy = torch.tensor([-0.3, 0.6, 0.2, -0.1])

        original_scaled_mm = fp8_ada._scaled_mm

        def dense_scaled_mm(value, raw_weight, row_scale):
            if row_scale is not None:
                raw_weight = raw_weight * row_scale.unsqueeze(-1)
            return F.linear(value, raw_weight)

        fp8_ada._scaled_mm = dense_scaled_mm
        try:
            logprob, entropy = fp8_ada._FP8LogProb.apply(
                hidden, weight, scale, labels, inv_temperature, 2
            )
            torch.autograd.backward((logprob, entropy), (grad_logprob, grad_entropy))
        finally:
            fp8_ada._scaled_mm = original_scaled_mm
        got_grad = hidden.grad.detach().clone()

        ref_hidden = hidden.detach().clone().requires_grad_()
        logits = F.linear(ref_hidden, weight * scale.unsqueeze(-1))
        scaled = logits * inv_temperature.unsqueeze(-1)
        log_probs = F.log_softmax(scaled, dim=-1)
        probabilities = log_probs.exp()
        ref_logprob = log_probs[torch.arange(4), labels]
        ref_entropy = -(probabilities * log_probs).sum(dim=-1)
        torch.autograd.backward(
            (ref_logprob, ref_entropy), (grad_logprob, grad_entropy)
        )
        torch.testing.assert_close(got_grad, ref_hidden.grad, rtol=2e-6, atol=2e-6)

    def test_entropy_only_backward_matches_dense_softmax(self):
        torch.manual_seed(31)
        hidden = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(6, 4, dtype=torch.float32)
        scale = torch.rand(6, dtype=torch.float32) + 0.25
        labels = torch.tensor([1, 4, 0])
        inv_temperature = torch.tensor([0.6, 1.0, 1.4])
        upstream = torch.tensor([0.7, -0.2, 0.5])

        original_scaled_mm = fp8_ada._scaled_mm

        def dense_scaled_mm(value, raw_weight, row_scale):
            if row_scale is not None:
                raw_weight = raw_weight * row_scale.unsqueeze(-1)
            return F.linear(value, raw_weight)

        fp8_ada._scaled_mm = dense_scaled_mm
        try:
            _, entropy = fp8_ada._FP8LogProb.apply(
                hidden, weight, scale, labels, inv_temperature, 2
            )
            entropy.backward(upstream)
        finally:
            fp8_ada._scaled_mm = original_scaled_mm
        got_grad = hidden.grad.detach().clone()

        ref_hidden = hidden.detach().clone().requires_grad_()
        scaled = F.linear(
            ref_hidden, weight * scale.unsqueeze(-1)
        ) * inv_temperature.unsqueeze(-1)
        probabilities = F.softmax(scaled, dim=-1)
        ref_entropy = -(probabilities * F.log_softmax(scaled, dim=-1)).sum(dim=-1)
        ref_entropy.backward(upstream)
        torch.testing.assert_close(got_grad, ref_hidden.grad, rtol=2e-6, atol=2e-6)

    def test_checkpoint_fp8_experts_permute_ep_source_major_rows(self):
        import torchtitan.experiments.kernels.moe.indices as index_ops

        # Source-major rows after EP all-to-all:
        # [src0/e0 x2, src0/e1 x1, src1/e0 x3, src1/e1 x0].
        x = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        counts = torch.tensor([2, 1, 3, 0])
        weights = tuple(torch.zeros(2, 1, 1) for _ in range(3))
        scales = tuple(torch.ones(2, 1, 1) for _ in range(3))
        captured = {}

        def fake_generate(raw_counts, local_experts, ep_size, *_args):
            torch.testing.assert_close(raw_counts, counts)
            self.assertEqual((local_experts, ep_size), (2, 2))
            return (
                torch.tensor([0, 1, 3, 4, 5, 2]),
                torch.tensor([5, 1]),
                torch.tensor([0, 5, 6]),
            )

        def fake_apply(
            permuted_x, _w1, _s1, _w2, _s2, _w3, _s3, offsets, _owner
        ):
            captured["x"] = permuted_x.clone()
            captured["offsets"] = offsets.clone()
            return permuted_x

        original_generate = index_ops.generate_permute_indices
        original_apply = fp8_ada._Experts.apply
        index_ops.generate_permute_indices = fake_generate
        fp8_ada._Experts.apply = fake_apply
        try:
            got = fp8_ada.fp8_experts(
                weights[0],
                weights[1],
                weights[2],
                scales[0],
                scales[1],
                scales[2],
                x,
                counts,
            )
        finally:
            index_ops.generate_permute_indices = original_generate
            fp8_ada._Experts.apply = original_apply

        torch.testing.assert_close(captured["x"], x[[0, 1, 3, 4, 5, 2]])
        torch.testing.assert_close(
            captured["offsets"], torch.tensor([5, 6], dtype=torch.int32)
        )
        torch.testing.assert_close(got, x)

    def test_grouped_mm_zeros_rows_after_last_expert_offset(self):
        x = torch.randn(4, 2)
        weight = torch.randn(2, 3, 2)
        scale = torch.ones(2, 1, 1)
        offsets = torch.tensor([1, 2], dtype=torch.int32)

        originals = (
            fp8_ada.moe_kernel_quantize_input,
            fp8_ada.dispatch_fused_moe_kernel,
        )

        def fake_quantize(value, *args, **kwargs):
            return value, torch.ones(value.shape[0], 1)

        def fake_dispatch(*args, **kwargs):
            args[2].fill_(7)

        fp8_ada.moe_kernel_quantize_input = fake_quantize
        fp8_ada.dispatch_fused_moe_kernel = fake_dispatch
        try:
            got = fp8_ada._grouped_mm(x, weight, scale, offsets)
        finally:
            (
                fp8_ada.moe_kernel_quantize_input,
                fp8_ada.dispatch_fused_moe_kernel,
            ) = originals

        torch.testing.assert_close(got[:2], torch.full_like(got[:2], 7))
        torch.testing.assert_close(got[2:], torch.zeros_like(got[2:]))

    def test_replace_linear_assigns_full_module_names(self):
        model = nn.Module()
        model.config = types.SimpleNamespace(
            quantization_config={"modules_to_not_convert": []}
        )
        model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        model.layers[0].o_proj = nn.Linear(128, 128, bias=False)
        model.layers[1].o_proj = nn.Linear(128, 128, bias=False)

        fp8_ada.replace_linear(model)

        self.assertEqual(model.layers[0].o_proj._fp8_fqn, "layers.0.o_proj")
        self.assertEqual(model.layers[1].o_proj._fp8_fqn, "layers.1.o_proj")


class MultiLinearAutogradTest(unittest.TestCase):
    def test_lora_mm_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "LoRA matrix mismatch"):
            multi_linear._lora_mm(torch.randn(2, 3), torch.randn(4, 5))

    def test_single_lora_passes_double_gradcheck(self):
        torch.manual_seed(4)
        inputs = (
            torch.randn(3, 4, dtype=torch.double, requires_grad=True),
            torch.randn(2, 4, dtype=torch.double, requires_grad=True),
            torch.randn(5, 2, dtype=torch.double, requires_grad=True),
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x, lora_a, lora_b: multi_linear._SingleLoRA.apply(
                    x, lora_a, lora_b, "gradcheck"
                ),
                inputs,
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_module_does_not_elementwise_clip_upstream_gradient(self):
        torch.manual_seed(5)
        base = nn.Linear(4, 5, bias=False, dtype=torch.double)
        module = multi_linear.MultiLoRALinear(
            base, rank=2, n_adapters=1, use_grouped_mm=False
        )
        with torch.no_grad():
            module.lora_A[0].normal_()
            module.lora_B[0].normal_()

        ref_base = nn.Linear(4, 5, bias=False, dtype=torch.double)
        ref_base.load_state_dict(base.state_dict())
        ref_a = module.lora_A[0].detach().clone().requires_grad_()
        ref_b = module.lora_B[0].detach().clone().requires_grad_()
        x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        upstream = torch.full((3, 5), 5.0, dtype=torch.double)

        module(x).backward(upstream)
        (ref_base(ref_x) + (ref_x @ ref_a.T) @ ref_b.T).backward(upstream)

        torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            module.lora_A[0].grad, ref_a.grad, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            module.lora_B[0].grad, ref_b.grad, rtol=1e-12, atol=1e-12
        )
        self.assertGreater(float(module.lora_B[0].grad.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
