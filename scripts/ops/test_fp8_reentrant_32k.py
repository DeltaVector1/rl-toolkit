"""Two-rank SM89 gate for 32K native-FP8 whole-block recomputation.

This is a block-level correctness and memory gate. It does not claim that the
complete 745B model fits. Run it in the trainer image on free RTX 4090 GPUs:

    torchrun --standalone --nproc-per-node=2 \
        scripts/ops/test_fp8_reentrant_32k.py

The gate keeps ``reshard_after_forward=True``. It compares eager and reentrant
input gradients, runs reentrant backward twice, rejects nonfinite values, and
reports both a conservative preflight estimate and the measured CUDA peak.
The defaults account for a 32,768-token sequence split across CP8.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True)
class ProbeShape:
    seq_len: int = 32768
    cp: int = 8
    hidden_size: int = 6144
    q_rank: int = 1536
    q_width: int = 24576
    kv_rank: int = 512

    @property
    def local_tokens(self) -> int:
        return math.ceil(self.seq_len / self.cp)

    def validate(self) -> None:
        if self.seq_len < 32768:
            raise ValueError("the 32K gate refuses seq_len values below 32768")
        for name in ("cp", "hidden_size", "q_rank", "q_width", "kv_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seq_len % self.cp:
            raise ValueError("seq_len must divide evenly across context-parallel ranks")
        if self.q_width % self.hidden_size:
            raise ValueError("q_width must be a multiple of hidden_size")
        for name in ("hidden_size", "q_rank", "q_width", "kv_rank"):
            if getattr(self, name) % 128:
                raise ValueError(f"{name} must be divisible by the FP8 block size 128")


def _linear_storage_bytes(in_features: int, out_features: int) -> int:
    fp8_weight = in_features * out_features
    fp32_scales = (in_features // 128) * (out_features // 128) * 4
    return fp8_weight + fp32_scales


def estimate_working_set_bytes(shape: ProbeShape, fsdp_world_size: int) -> dict[str, int]:
    """Conservatively account for the probe's reentrant-backward working set."""
    shape.validate()
    if fsdp_world_size < 1:
        raise ValueError("fsdp_world_size must be positive")

    h, qr, qw, kr = (
        shape.hidden_size,
        shape.q_rank,
        shape.q_width,
        shape.kv_rank,
    )
    linear_shapes = (
        (h, qr),
        (qr, qw),
        (h, kr),
        (kr, h),
        (h, h),
    )
    parameter_payload = sum(_linear_storage_bytes(i, o) for i, o in linear_shapes)
    tokens = shape.local_tokens
    bf16 = 2
    input_bytes = tokens * h * bf16
    # Count all forward intermediates, then count a similarly sized backward
    # wave. This is conservative because not all tensors are live together.
    forward_intermediates = tokens * (qr + qw + h + kr + h + h + h) * bf16

    categories = {
        "resident_fsdp_shards": math.ceil(parameter_payload / fsdp_world_size),
        "unsharded_parameters": parameter_payload,
        # Native _Linear owns snapshots during grad-enabled recompute so FSDP
        # cannot invalidate its weight and scale storage before inner backward.
        "owned_backward_snapshots": parameter_payload,
        "checkpoint_input_and_gradient": 2 * input_bytes,
        "recompute_forward_and_backward": 2 * forward_intermediates,
        "position_ids_and_offsets": tokens * 16,
    }
    categories["estimated_total"] = sum(categories.values())
    return categories


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--q-rank", type=int, default=1536)
    parser.add_argument("--q-width", type=int, default=24576)
    parser.add_argument("--kv-rank", type=int, default=512)
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        default=23.0,
        help="hard per-rank peak budget for a nominal 24 GiB RTX 4090",
    )
    parser.add_argument(
        "--safety-margin-gib",
        type=float,
        default=2.0,
        help="free space that must remain beyond the estimate and measured peak",
    )
    return parser.parse_args()


def main() -> None:
    # Imports that require the trainer image stay out of module scope so CPU
    # unit tests can validate memory accounting without vLLM or CUDA.
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from prime_rl.trainer.models.layers.checkpointing import (
        set_reentrant_block_checkpointing,
    )
    from prime_rl.trainer.models.layers.fp8_ada import FP8CheckpointLinear

    args = _arguments()
    shape = ProbeShape(
        seq_len=args.seq_len,
        cp=args.cp,
        hidden_size=args.hidden_size,
        q_rank=args.q_rank,
        q_width=args.q_width,
        kv_rank=args.kv_rank,
    )
    shape.validate()
    if args.memory_budget_gib <= 0 or args.safety_margin_gib < 0:
        raise ValueError("memory budget must be positive and margin nonnegative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    if capability != (8, 9):
        raise RuntimeError(
            f"this validation is restricted to free SM89 GPUs; detected SM{capability[0]}{capability[1]}"
        )
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("run with at least two ranks so FSDP really shards and reshards")

    estimate = estimate_working_set_bytes(shape, world_size)
    budget = int(args.memory_budget_gib * GIB)
    margin = int(args.safety_margin_gib * GIB)
    free, total = torch.cuda.mem_get_info(local_rank)
    free_tensor = torch.tensor(free, device="cuda", dtype=torch.int64)
    dist.all_reduce(free_tensor, op=dist.ReduceOp.MIN)
    minimum_free = int(free_tensor.item())
    if estimate["estimated_total"] + margin > budget:
        raise RuntimeError(
            "estimated 32K working set plus safety margin exceeds the configured "
            f"budget: {estimate['estimated_total'] / GIB:.2f} + "
            f"{args.safety_margin_gib:.2f} > {args.memory_budget_gib:.2f} GiB"
        )
    if estimate["estimated_total"] + margin > minimum_free:
        raise RuntimeError(
            "estimated 32K working set plus safety margin exceeds currently free "
            f"memory: need {(estimate['estimated_total'] + margin) / GIB:.2f} GiB, "
            f"minimum free is {minimum_free / GIB:.2f} GiB"
        )

    def make_linear(in_features: int, out_features: int, fqn: str):
        with torch.device("meta"):
            base = nn.Linear(in_features, out_features, bias=False)
        module = FP8CheckpointLinear.from_linear(base)
        module.to_empty(device=torch.device("cuda", local_rank))
        generator = torch.Generator(device="cuda").manual_seed(
            1009 + sum(map(ord, fqn))
        )
        with torch.no_grad():
            for start in range(0, out_features, 128):
                end = min(start + 128, out_features)
                values = torch.randn(
                    end - start,
                    in_features,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                module.weight[start:end].copy_(values.to(torch.float8_e4m3fn))
            module.weight_scale_inv.fill_(0.01)
        module._fp8_fqn = fqn
        return module

    class NativeFp8AttentionProbe(nn.Module):
        def __init__(self):
            super().__init__()
            h, qr, qw, kr = (
                shape.hidden_size,
                shape.q_rank,
                shape.q_width,
                shape.kv_rank,
            )
            self.q_down = make_linear(h, qr, "probe.q_down")
            self.q_up = make_linear(qr, qw, "probe.q_up")
            self.kv_down = make_linear(h, kr, "probe.kv_down")
            self.kv_up = make_linear(kr, h, "probe.kv_up")
            self.out = make_linear(h, h, "probe.out")

        def forward(self, *, hidden_states, metadata):
            q = self.q_up(torch.nn.functional.silu(self.q_down(hidden_states)))
            q = q.view(q.shape[0], -1, shape.hidden_size).mean(dim=1)
            kv = self.kv_up(torch.nn.functional.silu(self.kv_down(hidden_states)))
            # Make the nested keyword tensor an explicit checkpoint dependency
            # without changing the measured projection result.
            position_ids = metadata["position_ids"].to(q.dtype).unsqueeze(-1)
            mixed = torch.nn.functional.silu(q + kv + position_ids.mul(0))
            return self.out(mixed)

    torch.manual_seed(7)
    block = NativeFp8AttentionProbe().cuda(local_rank)
    set_reentrant_block_checkpointing(block)
    mesh = init_device_mesh("cuda", (world_size,))
    fully_shard(block, mesh=mesh, reshard_after_forward=True)

    generator = torch.Generator(device="cuda").manual_seed(29)
    template = torch.randn(
        shape.local_tokens,
        shape.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    position_ids = torch.arange(
        shape.local_tokens, device="cuda", dtype=torch.int64
    )

    def run_iteration(checkpoint_enabled: bool):
        # The checkpoint helper deliberately bypasses recompute in eval mode.
        block.train(checkpoint_enabled)
        value = template.detach().clone().requires_grad_()
        torch.cuda.reset_peak_memory_stats(local_rank)
        output = block(
            hidden_states=value,
            metadata={"position_ids": position_ids},
        )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("native-FP8 probe forward produced nonfinite values")
        output.float().square().mean().backward()
        torch.cuda.synchronize(local_rank)
        if value.grad is None or not bool(torch.isfinite(value.grad).all()):
            raise RuntimeError("native-FP8 probe backward produced a nonfinite or missing input gradient")
        if not bool(value.grad.abs().max() > 0):
            raise RuntimeError("native-FP8 probe backward produced an all-zero input gradient")
        return (
            output.detach().cpu(),
            value.grad.detach().cpu(),
            torch.cuda.max_memory_allocated(local_rank),
        )

    eager_output, eager_gradient, eager_peak = run_iteration(False)
    checkpoint_output, checkpoint_gradient, checkpoint_peak = run_iteration(True)
    repeat_output, repeat_gradient, repeat_peak = run_iteration(True)

    torch.testing.assert_close(checkpoint_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_gradient, eager_gradient, rtol=0, atol=0)
    torch.testing.assert_close(repeat_output, checkpoint_output, rtol=0, atol=0)
    torch.testing.assert_close(repeat_gradient, checkpoint_gradient, rtol=0, atol=0)

    peak = max(eager_peak, checkpoint_peak, repeat_peak)
    peak_tensor = torch.tensor(peak, device="cuda", dtype=torch.int64)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    maximum_peak = int(peak_tensor.item())
    if maximum_peak + margin > budget:
        raise RuntimeError(
            f"measured peak plus margin exceeds budget: {maximum_peak / GIB:.2f} + "
            f"{args.safety_margin_gib:.2f} > {args.memory_budget_gib:.2f} GiB"
        )

    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "device_total_gib": total / GIB,
                    "minimum_free_before_gib": minimum_free / GIB,
                    "shape": {
                        **shape.__dict__,
                        "local_tokens": shape.local_tokens,
                    },
                    "estimate_gib": {
                        key: value / GIB for key, value in estimate.items()
                    },
                    "measured_peak_gib": maximum_peak / GIB,
                    "memory_budget_gib": args.memory_budget_gib,
                    "safety_margin_gib": args.safety_margin_gib,
                    "reshard_after_forward": True,
                    "checkpoint": "reentrant whole block",
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
