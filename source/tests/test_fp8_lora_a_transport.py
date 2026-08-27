"""CPU and distributed-fake checks for the opt-in FP8 LoRA-A route."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from prime_rl.trainer.distributed.fp8_lora_a_transport import (  # noqa: E402
    bounded_workspace_bytes,
    prepare_fp8_lora_a_transport,
    run_fp8_lora_a_experts,
    transport_bytes,
    transport_mode,
)


class RecordingCollective:
    """One-rank fake that records every payload and preserves row order."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.dtype, int, int]] = []

    def all_to_all_single(self, receive, send, output_splits, input_splits, *, group):
        self.calls.append((send.dtype, send.numel(), receive.numel()))
        if tuple(output_splits) != tuple(input_splits) or len(output_splits) != 1:
            raise AssertionError("this fake only models a one-rank route")
        receive.copy_(send)


def rowwise_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # CPU float8 conversion is deterministic. Distinct scales make an
    # accidental value/scale permutation visible in the output.
    scales = torch.arange(1, value.shape[0] + 1, device=value.device, dtype=torch.float32)
    quantized = (value.float() / scales[:, None]).round().clamp(-4, 4)
    return quantized.to(torch.float8_e4m3fn), scales


def _selected(value: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return value.index_select(0, ids)


def _base(value: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.bmm(value.unsqueeze(1), _selected(weight, ids).transpose(1, 2)).squeeze(1)


def _reference(
    x,
    scores,
    ids,
    w1,
    w2,
    w3,
    gate_a,
    gate_b,
    up_a,
    up_b,
    down_a,
    down_b,
    upstream,
):
    projected = (x.float() * scores[:, None]).to(x.dtype)
    q, scale = rowwise_quantize(projected)
    dequantized = q.to(x.dtype) * scale[:, None].to(x.dtype)
    # Match the native expert activation VJP: FP8 QDQ forward with a BF16
    # straight-through base-input gradient.
    base_input = projected + (dequantized - projected).detach()
    p_gate = torch.bmm(projected.unsqueeze(1), _selected(gate_a, ids).transpose(1, 2)).squeeze(1)
    p_up = torch.bmm(projected.unsqueeze(1), _selected(up_a, ids).transpose(1, 2)).squeeze(1)
    gate = _base(base_input, w1, ids) + torch.bmm(p_gate.unsqueeze(1), _selected(gate_b, ids).transpose(1, 2)).squeeze(1)
    up = _base(base_input, w3, ids) + torch.bmm(p_up.unsqueeze(1), _selected(up_b, ids).transpose(1, 2)).squeeze(1)
    hidden = torch.nn.functional.silu(gate) * up
    qh, sh = rowwise_quantize(hidden)
    dh = qh.to(x.dtype) * sh[:, None].to(x.dtype)
    hidden_base = hidden + (dh - hidden).detach()
    result = _base(hidden_base, w2, ids)
    hidden_low = torch.bmm(hidden.unsqueeze(1), _selected(down_a, ids).transpose(1, 2)).squeeze(1)
    result = result + torch.bmm(hidden_low.unsqueeze(1), _selected(down_b, ids).transpose(1, 2)).squeeze(1)
    loss = (result * upstream).sum()
    loss.backward()
    return result.detach(), [x.grad.detach(), scores.grad.detach(), *[v.grad.detach() for v in (gate_a, gate_b, up_a, up_b, down_a, down_b)]]


class FP8LoRAATransportTest(unittest.TestCase):
    def make_case(self):
        torch.manual_seed(7)
        rows, hidden, experts, rank, intermediate = 7, 4, 3, 2, 5
        x = torch.randn(rows, hidden, dtype=torch.bfloat16, requires_grad=True)
        scores = (torch.rand(rows, dtype=torch.float32) + 0.25).requires_grad_()
        # Repeated expert ids are intentional. The route is stable and has no
        # coalescing, so every duplicate remains an independent row.
        ids = torch.tensor([2, 0, 2, 1, 0, 1, 2], dtype=torch.int64)
        def parameter(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16, requires_grad=True)
        factors = (
            parameter(experts, rank, hidden),
            parameter(experts, intermediate, rank),
            parameter(experts, rank, hidden),
            parameter(experts, intermediate, rank),
            parameter(experts, rank, intermediate),
            parameter(experts, hidden, rank),
        )
        w1 = parameter(experts, intermediate, hidden)
        w2 = parameter(experts, hidden, intermediate)
        w3 = parameter(experts, intermediate, hidden)
        upstream = torch.randn(rows, hidden, dtype=torch.bfloat16)
        return x, scores, ids, (w1, w2, w3), factors, upstream

    def test_forward_input_lora_and_route_gradients_match_bf16_reference(self):
        case = self.make_case()
        x, scores, ids, (w1, w2, w3), factors, upstream = case
        gate_a, gate_b, up_a, up_b, down_a, down_b = factors
        ref_result, ref_grads = _reference(
            x.detach().clone().requires_grad_(),
            scores.detach().clone().requires_grad_(),
            ids,
            w1.detach(), w2.detach(), w3.detach(),
            *(value.detach().clone().requires_grad_() for value in factors),
            upstream,
        )
        fake = RecordingCollective()
        state = prepare_fp8_lora_a_transport(
            x,
            ids,
            gate_a,
            up_a,
            [ids.shape[0]],
            [ids.shape[0]],
            collective=fake,
            route_scores=scores,
            score_before_experts=True,
            quantize=rowwise_quantize,
        )
        got = run_fp8_lora_a_experts(
            state,
            ids,
            w1, w2, w3,
            gate_b, up_b, down_a, down_b,
            quantize=rowwise_quantize,
        )
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got.detach(), ref_result))
        forward_call_dtypes = [dtype for dtype, _, _ in fake.calls]
        (got * upstream).sum().backward()
        got_grads = [x.grad, scores.grad, gate_a.grad, gate_b.grad, up_a.grad, up_b.grad, down_a.grad, down_b.grad]
        for got_grad, expected in zip(got_grads, ref_grads):
            # Source-side VJPs are mathematically identical. Separate
            # all-to-all autograd nodes can add BF16 terms in a different
            # order, so allow one BF16 ulp at the largest values.
            torch.testing.assert_close(got_grad, expected, rtol=2e-2, atol=0.25)
        # Q, scales, and the two rank-sized projections are three forward
        # all-to-all calls. Values and scales must be kept paired.
        self.assertEqual(
            forward_call_dtypes,
            [torch.float8_e4m3fn, torch.float32, torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        self.assertTrue(torch.equal(state.received_scales, state.scales))
        expected_gate = torch.bmm(
            (x.float() * scores[:, None].float()).to(x.dtype).unsqueeze(1),
            gate_a.index_select(0, ids).transpose(1, 2),
        ).squeeze(1)
        self.assertTrue(torch.equal(state.received_gate_projection, expected_gate))

    def test_permutation_duplicates_and_malformed_inputs(self):
        x, scores, ids, _, factors, _ = self.make_case()
        gate_a, _, up_a, _, _, _ = factors
        fake = RecordingCollective()
        state = prepare_fp8_lora_a_transport(
            x, ids, gate_a, up_a, [ids.shape[0]], [ids.shape[0]],
            collective=fake, quantize=rowwise_quantize,
        )
        expected_gate = torch.bmm(x.unsqueeze(1), gate_a.index_select(0, ids).transpose(1, 2)).squeeze(1)
        self.assertTrue(torch.equal(state.received_gate_projection, expected_gate))
        self.assertEqual(state.source_expert_indices.tolist(), ids.tolist())
        with self.assertRaises(ValueError):
            prepare_fp8_lora_a_transport(x, ids, gate_a, up_a, [6], [6], collective=fake, quantize=rowwise_quantize)
        with self.assertRaises(ValueError):
            prepare_fp8_lora_a_transport(x, ids, gate_a, up_a, [7], [7], collective=fake, route_scores=scores[:-1], score_before_experts=True, quantize=rowwise_quantize)
        with self.assertRaises(TypeError):
            prepare_fp8_lora_a_transport(x.float(), ids, gate_a, up_a, [7], [7], collective=fake, quantize=rowwise_quantize)

    def test_mode_bytes_and_bounded_memory_are_explicit(self):
        self.assertEqual(transport_mode(), "bf16")
        self.assertEqual(transport_mode("fp8_lora_a"), "fp8_lora_a")
        with self.assertRaises(ValueError):
            transport_mode("route_cache")
        payload = transport_bytes(6144, 32)
        self.assertEqual(payload["bf16_forward"], 12288)
        self.assertEqual(payload["fp8_lora_a_forward"], 6276)
        self.assertEqual(payload["bf16_backward"], 12288)
        self.assertEqual(payload["fp8_lora_a_backward"], 12416)
        self.assertEqual(payload["dispatch_forward"], 6276)
        self.assertEqual(payload["combine_forward"], 12288)
        self.assertEqual(payload["dispatch_backward"], 12416)
        self.assertEqual(payload["combine_backward"], 12288)
        workspace = bounded_workspace_bytes(3, 5, 6144, 32)
        self.assertEqual(workspace["forward_send_bytes"], 3 * 6276)
        self.assertEqual(workspace["forward_receive_bytes"], 5 * 6276)
        self.assertEqual(workspace["forward_owner_bf16_bytes"], 5 * 12288)
        self.assertEqual(workspace["forward_round_scratch_bytes"], 0)
        rounded = bounded_workspace_bytes(3, 5, 6144, 32, rounds=2)
        self.assertEqual(rounded["forward_round_scratch_bytes"], (2 + 3) * 6144)
        self.assertEqual(rounded["backward_round_scratch_bytes"], (2 + 3) * 12288)


if __name__ == "__main__":
    unittest.main()
