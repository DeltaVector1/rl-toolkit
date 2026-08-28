"""Small single-GPU SM89 sparse MLA forward/backward comparison.

Run with the trainer environment on one RTX 4090:

    CUDA_VISIBLE_DEVICES=0 python3 scripts/ops/benchmark_sparse_mla_bwd_sm89.py

The candidate is enabled only for the second backward call.  The script
requires exact forward and dQ parity.  dKV is checked with the tolerance used
by the existing CUDA check because duplicate-index FP32 atomics do not promise
one global write order.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def _timed(callable, warmup: int, iterations: int) -> tuple[object, list[float]]:
    for _ in range(warmup):
        callable()
    torch.cuda.synchronize()
    times: list[float] = []
    result = None
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callable()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return result, times


def _summary(times: list[float]) -> tuple[float, float]:
    ordered = sorted(times)
    return statistics.median(ordered), ordered[-1]


def _case(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260828)
    batch, sequence, heads, sequence_kv, groups, topk = 1, 4, 16, 9, 1, 32
    query = torch.randn(
        batch, sequence, heads, 576, device=device, dtype=torch.bfloat16, generator=generator
    ).mul_(0.02)
    key_value = torch.randn(
        batch, sequence_kv, groups, 576, device=device, dtype=torch.bfloat16, generator=generator
    ).mul_(0.02)
    key_value[:, -1].zero_()
    route = torch.arange(topk, device=device, dtype=torch.int32).remainder(sequence_kv)
    indices = route.view(1, 1, 1, topk).expand(batch, sequence, groups, topk).contiguous()
    output_gradient = torch.randn(
        batch, sequence, heads, 512, device=device, dtype=torch.bfloat16, generator=generator
    ).mul_(1e-3)
    return query, key_value, indices, output_gradient


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise SystemExit(f"SM89 is required, found sm_{capability[0]}{capability[1]}")
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be non-negative and iterations must be positive")

    device = torch.device("cuda")
    query, key_value, indices, output_gradient = _case(device)
    scale = 576**-0.5
    os.environ["PRIME_SPARSE_BWD_DIRECT_DKV"] = "0"
    reference_output, reference_lse = sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=scale
    )
    repeated_output, repeated_lse = sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=scale
    )
    torch.cuda.synchronize()
    if not torch.equal(reference_output, repeated_output) or not torch.equal(
        reference_lse, repeated_lse
    ):
        raise RuntimeError("forward repeat is not exact")
    reference, reference_times = _timed(
        lambda: sparse_mla_bwd(
            query, key_value, reference_output, output_gradient, indices, reference_lse, sm_scale=scale
        ),
        args.warmup,
        args.iterations,
    )

    os.environ["PRIME_SPARSE_BWD_DIRECT_DKV"] = "1"
    candidate, candidate_times = _timed(
        lambda: sparse_mla_bwd(
            query, key_value, reference_output, output_gradient, indices, reference_lse, sm_scale=scale
        ),
        args.warmup,
        args.iterations,
    )
    torch.cuda.synchronize()
    if not torch.equal(reference[0], candidate[0]):
        raise RuntimeError("backward dQ is not exact")
    torch.testing.assert_close(candidate[1], reference[1], rtol=2e-2, atol=2e-3)
    if not bool(torch.isfinite(candidate[0]).all() and torch.isfinite(candidate[1]).all()):
        raise RuntimeError("candidate backward returned a nonfinite gradient")

    reference_median, reference_max = _summary(reference_times)
    candidate_median, candidate_max = _summary(candidate_times)
    print(
        f"device={torch.cuda.get_device_name()} capability=sm_{capability[0]}{capability[1]} "
        "shape=B1,S4,S_kv9,H16,G1,topk32,tail64"
    )
    print("forward_exact=1 dQ_exact=1 dKV_close=1 finite_gradients=1")
    print(
        f"reference_ms_median={reference_median:.3f} reference_ms_max={reference_max:.3f} "
        f"candidate_ms_median={candidate_median:.3f} candidate_ms_max={candidate_max:.3f}"
    )


if __name__ == "__main__":
    main()
