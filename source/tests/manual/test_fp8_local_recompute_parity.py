"""One-GPU gradient comparison for local routed-expert recompute."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import nn
from vllm import _custom_ops as ops

from prime_rl.trainer.distributed.expert_parallel import ReceivedFP8ExpertInput
from prime_rl.trainer.models.layers.fp8_ada import (
    _InplacePermutedLoRAExperts,
    scale_routed_output,
)


def quantize_dequantize(value: torch.Tensor) -> torch.Tensor:
    quantized, scales = ops.scaled_fp8_quant(
        value,
        use_per_token_if_dynamic=True,
    )
    return quantized.to(value.dtype) * scales.to(value.dtype)


class FP8ForwardAndBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return quantize_dequantize(value)

    @staticmethod
    def backward(ctx, grad):
        return quantize_dequantize(grad)


class BF16ForwardFP8Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return quantize_dequantize(grad)


class BF16ForwardAndBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad


class TestOwner(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.base_layer = nn.Module()
        for name in (
            "w1",
            "w1_scale_inv",
            "w2",
            "w2_scale_inv",
            "w3",
            "w3_scale_inv",
        ):
            self.base_layer.register_parameter(
                name,
                nn.Parameter(state[name].clone(), requires_grad=False),
            )
        for projection in ("w1", "w2", "w3"):
            for factor in ("A", "B"):
                name = f"{projection}_lora_{factor}"
                setattr(
                    self,
                    name,
                    nn.ParameterList([nn.Parameter(state[name].clone())]),
                )

    def factors(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"{projection}_lora_{factor}")[0]
            for projection in ("w1", "w2", "w3")
            for factor in ("A", "B")
        ]


@dataclass
class RunResult:
    output: torch.Tensor
    hidden_grad: torch.Tensor
    route_grad: torch.Tensor
    factor_grads: list[torch.Tensor]


def run_bf16_forward(
    state: dict[str, torch.Tensor],
    source: torch.Tensor,
    scores: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    *,
    local: bool,
) -> torch.Tensor:
    owner = TestOwner(state)
    received = ReceivedFP8ExpertInput(
        save_for_backward=False,
        local_recompute=local,
    )
    received.prepare_from_bf16(source)
    with torch.no_grad():
        output = _InplacePermutedLoRAExperts.apply(
            source.clone(),
            owner.base_layer.w1,
            owner.base_layer.w1_scale_inv,
            owner.base_layer.w2,
            owner.base_layer.w2_scale_inv,
            owner.base_layer.w3,
            owner.base_layer.w3_scale_inv,
            *owner.factors(),
            2.0,
            permutation,
            offsets,
            owner,
            received,
            scores if local else None,
            source.shape[0],
        )
        if not local:
            output = scale_routed_output(
                output,
                scores,
                chunk_rows=source.shape[0],
            )
    return output


def make_state(
    experts: int,
    width: int,
    intermediate: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    def weight(shape):
        return (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.03
        ).to(torch.float8_e4m3fn)

    def factor(shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.01

    return {
        "w1": weight((experts, intermediate, width)),
        "w1_scale_inv": torch.ones(
            experts,
            intermediate // 128,
            width // 128,
            device="cuda",
            dtype=torch.float32,
        ),
        "w2": weight((experts, width, intermediate)),
        "w2_scale_inv": torch.ones(
            experts,
            width // 128,
            intermediate // 128,
            device="cuda",
            dtype=torch.float32,
        ),
        "w3": weight((experts, intermediate, width)),
        "w3_scale_inv": torch.ones(
            experts,
            intermediate // 128,
            width // 128,
            device="cuda",
            dtype=torch.float32,
        ),
        "w1_lora_A": factor((experts, rank, width)),
        "w1_lora_B": factor((experts, intermediate, rank)),
        "w2_lora_A": factor((experts, rank, intermediate)),
        "w2_lora_B": factor((experts, width, rank)),
        "w3_lora_A": factor((experts, rank, width)),
        "w3_lora_B": factor((experts, intermediate, rank)),
    }


def run_case(
    state: dict[str, torch.Tensor],
    source_value: torch.Tensor,
    score_value: torch.Tensor,
    upstream: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    *,
    local: bool,
) -> RunResult:
    owner = TestOwner(state)
    source = source_value.clone().requires_grad_()
    scores = score_value.clone().requires_grad_()
    received = ReceivedFP8ExpertInput(local_recompute=local)
    received.prepare_from_bf16(source.detach())
    boundary = BF16ForwardFP8Backward if local else FP8ForwardAndBackward
    dispatched = boundary.apply(source)
    output = _InplacePermutedLoRAExperts.apply(
        dispatched,
        owner.base_layer.w1,
        owner.base_layer.w1_scale_inv,
        owner.base_layer.w2,
        owner.base_layer.w2_scale_inv,
        owner.base_layer.w3,
        owner.base_layer.w3_scale_inv,
        *owner.factors(),
        2.0,
        permutation,
        offsets,
        owner,
        received,
        scores if local else None,
        source.shape[0],
    )
    if local:
        output = BF16ForwardAndBackward.apply(output)
    else:
        output = scale_routed_output(
            FP8ForwardAndBackward.apply(output),
            scores,
            chunk_rows=source.shape[0],
        )
    saved_output = output.detach().clone()
    output.backward(upstream)
    return RunResult(
        output=saved_output,
        hidden_grad=source.grad.detach().clone(),
        route_grad=scores.grad.detach().clone(),
        factor_grads=[value.grad.detach().clone() for value in owner.factors()],
    )


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = value.float() - reference.float()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": float(delta.abs().max()),
        "relative_rmse": float(delta.square().mean().sqrt() / denominator),
        "l1": float(delta.abs().sum()),
    }


def main() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this check requires one RTX 4090")
    torch.manual_seed(3107)
    experts = 2
    width = 128
    intermediate = 128
    rank = 32
    rows = 64
    state = make_state(experts, width, intermediate, rank)
    source = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda", dtype=torch.float32) * 0.8 + 0.1
    upstream = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    padding = torch.full(
        (16,),
        rows,
        device="cuda",
        dtype=torch.int64,
    )
    permutation = torch.cat(
        (
            torch.arange(0, rows, 2, device="cuda"),
            padding,
            torch.arange(1, rows, 2, device="cuda"),
            padding,
        )
    ).to(torch.int64)
    offsets = torch.tensor(
        [rows // 2 + padding.shape[0], rows + 2 * padding.shape[0]],
        device="cuda",
        dtype=torch.int32,
    )

    baseline = run_case(
        state,
        source,
        scores,
        upstream,
        permutation,
        offsets,
        local=False,
    )
    local = run_case(
        state,
        source,
        scores,
        upstream,
        permutation,
        offsets,
        local=True,
    )
    baseline_forward = run_bf16_forward(
        state,
        source,
        scores,
        permutation,
        offsets,
        local=False,
    )
    local_forward = run_bf16_forward(
        state,
        source,
        scores,
        permutation,
        offsets,
        local=True,
    )
    names = (
        "w1_lora_A",
        "w1_lora_B",
        "w2_lora_A",
        "w2_lora_B",
        "w3_lora_A",
        "w3_lora_B",
    )
    report = {
        "bf16_forward": difference(local_forward, baseline_forward),
        "checkpoint_output_not_used": difference(local.output, baseline.output),
        "hidden_grad": difference(local.hidden_grad, baseline.hidden_grad),
        "route_grad": difference(local.route_grad, baseline.route_grad),
        "lora_grad": {
            name: difference(got, expected)
            for name, got, expected in zip(
                names,
                local.factor_grads,
                baseline.factor_grads,
                strict=True,
            )
        },
    }
    print(json.dumps(report, indent=2))
    gradients = [report["hidden_grad"], report["route_grad"]]
    gradients.extend(report["lora_grad"].values())
    if report["bf16_forward"]["max_abs"] != 0 or any(
        item["relative_rmse"] > 1e-6 for item in gradients
    ):
        raise RuntimeError("local expert recompute gradients differ from baseline")


if __name__ == "__main__":
    main()
