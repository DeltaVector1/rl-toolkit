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
FP8_ADA_PATH = (
    REPOSITORY_ROOT
    / "source/src/prime_rl/trainer/models/layers/fp8_ada.py"
)
MULTI_LINEAR_PATH = (
    REPOSITORY_ROOT
    / "source/src/prime_rl/trainer/models/layers/lora/multi_linear.py"
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

    fused_moe = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    fused_moe._prepare_expert_assignment = lambda *args, **kwargs: (None, None, None)
    fused_moe.dispatch_fused_moe_kernel = lambda *args, **kwargs: None
    fused_moe.get_default_config = lambda *args, **kwargs: None
    fused_moe.moe_kernel_quantize_input = lambda *args, **kwargs: (None, None)
    sys.modules.setdefault("vllm.model_executor", types.ModuleType("vllm.model_executor"))
    sys.modules.setdefault(
        "vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")
    )
    sys.modules.setdefault(
        "vllm.model_executor.layers.fused_moe",
        types.ModuleType("vllm.model_executor.layers.fused_moe"),
    )
    sys.modules.setdefault(
        "vllm.model_executor.layers.fused_moe.fused_moe", fused_moe
    )


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
    return (
        scale.repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)[: shape[0], : shape[1]]
    )


class FP8AdaAutogradTest(unittest.TestCase):
    def test_dequant_weight_expands_each_128_square_block(self):
        weight = (
            torch.arange(256 * 384, dtype=torch.int32)
            .remainder(31)
            .sub(15)
            .reshape(256, 384)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.tensor(
            [[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]], dtype=torch.float32
        )

        got = fp8_ada._dequant_weight(weight, scale, 128, 256, torch.float32)
        expected = weight[128:256].float() * _expanded_block_scale(
            scale[1:2], (128, 384)
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

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
            got = fp8_ada._Linear.apply(x, live_weight, live_scale)
            # Simulate FSDP RAF releasing the explicit forward inputs. Backward
            # must use the owned direct-context snapshots.
            live_weight.fill_(float("nan"))
            live_scale.fill_(float("nan"))
            got.backward(upstream)
        finally:
            fp8_ada._linear = original_linear
        got_grad = x.grad.detach().clone()

        ref_x = x.detach().clone().requires_grad_()
        dequantized_linear(ref_x, weight, scale).backward(upstream)
        torch.testing.assert_close(got_grad, ref_x.grad, rtol=1e-12, atol=1e-12)

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
                x, w1, dummy_scale, w2, dummy_scale, w3, dummy_scale, offsets
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
            torch.autograd.backward(
                (logprob, entropy), (grad_logprob, grad_entropy)
            )
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
        scaled = (
            F.linear(ref_hidden, weight * scale.unsqueeze(-1))
            * inv_temperature.unsqueeze(-1)
        )
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

        def fake_apply(permuted_x, _w1, _s1, _w2, _s2, _w3, _s3, offsets):
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
        torch.testing.assert_close(captured["offsets"], torch.tensor([5, 6], dtype=torch.int32))
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


class MultiLinearAutogradTest(unittest.TestCase):
    def test_single_lora_passes_double_gradcheck(self):
        torch.manual_seed(4)
        inputs = (
            torch.randn(3, 4, dtype=torch.double, requires_grad=True),
            torch.randn(2, 4, dtype=torch.double, requires_grad=True),
            torch.randn(5, 2, dtype=torch.double, requires_grad=True),
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                multi_linear._SingleLoRA.apply,
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
