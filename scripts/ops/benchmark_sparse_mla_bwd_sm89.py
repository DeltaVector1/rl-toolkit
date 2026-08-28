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
from dataclasses import dataclass

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


MAX_WARMUP = 5
MAX_ITERATIONS = 20


@dataclass(frozen=True)
class Case:
    name: str
    sequence: int
    heads: int
    sequence_kv: int
    topk: int
    route_keys: int


CASES = (
    Case("smoke", sequence=4, heads=16, sequence_kv=9, topk=32, route_keys=9),
    Case(
        "head_contention",
        sequence=32,
        heads=64,
        sequence_kv=33,
        topk=128,
        route_keys=8,
    ),
)


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


def _case(device: torch.device, case: Case) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260828)
    batch, groups = 1, 1
    query = torch.randn(
        batch,
        case.sequence,
        case.heads,
        576,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value = torch.randn(
        batch,
        case.sequence_kv,
        groups,
        576,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value[:, -1].zero_()
    route = torch.arange(case.topk, device=device, dtype=torch.int32).remainder(
        case.route_keys
    )
    indices = (
        route.view(1, 1, 1, case.topk)
        .expand(batch, case.sequence, groups, case.topk)
        .contiguous()
    )
    output_gradient = torch.randn(
        batch,
        case.sequence,
        case.heads,
        512,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-3)
    return query, key_value, indices, output_gradient


def _run_case(device: torch.device, case: Case, warmup: int, iterations: int) -> None:
    query, key_value, indices, output_gradient = _case(device, case)
    scale = 576**-0.5
    os.environ["PRIME_SPARSE_BWD_REUSE_DQ_SHARED"] = "0"
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
        warmup,
        iterations,
    )

    os.environ["PRIME_SPARSE_BWD_REUSE_DQ_SHARED"] = "1"
    candidate, candidate_times = _timed(
        lambda: sparse_mla_bwd(
            query, key_value, reference_output, output_gradient, indices, reference_lse, sm_scale=scale
        ),
        warmup,
        iterations,
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
        f"case={case.name} shape=B1,S{case.sequence},S_kv{case.sequence_kv},"
        f"H{case.heads},G1,topk{case.topk},route_keys{case.route_keys},tail64"
    )
    print("forward_exact=1 dQ_exact=1 dKV_close=1 finite_gradients=1")
    print(
        f"reference_ms_median={reference_median:.3f} reference_ms_max={reference_max:.3f} "
        f"candidate_ms_median={candidate_median:.3f} candidate_ms_max={candidate_max:.3f}"
    )


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise SystemExit(f"SM89 is required, found sm_{capability[0]}{capability[1]}")
    if not 0 <= args.warmup <= MAX_WARMUP:
        raise SystemExit(f"warmup must be between 0 and {MAX_WARMUP}")
    if not 1 <= args.iterations <= MAX_ITERATIONS:
        raise SystemExit(f"iterations must be between 1 and {MAX_ITERATIONS}")

    print(f"device={torch.cuda.get_device_name()} capability=sm_89")
    device = torch.device("cuda")
    for case in CASES:
        _run_case(device, case, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
