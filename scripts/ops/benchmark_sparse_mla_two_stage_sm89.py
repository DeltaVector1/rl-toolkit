"""Bounded end-to-end benchmark for the opt-in SM89 sparse MLA reducer."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import preprocess, sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
from prime_rl.trainer.models.kernels.sparse_mla_two_stage import (
    MAX_WORKSPACE_BYTES,
    sparse_mla_dkv_two_stage,
    two_stage_memory,
)


MAX_WARMUP = 1
MAX_ITERATIONS = 3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sequence-kv", type=int, choices=(4097, 32769), default=4097)
    return parser.parse_args()


def make_inputs(sequence_kv: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    q = torch.randn(1, 4096, 64, 576, device="cuda", dtype=torch.bfloat16, generator=generator)
    kv = torch.randn(1, sequence_kv, 1, 576, device="cuda", dtype=torch.bfloat16, generator=generator)
    kv[:, -1].zero_()
    do = torch.randn(1, 4096, 64, 512, device="cuda", dtype=torch.bfloat16, generator=generator)
    indices = torch.arange(2048, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 1, 2048).expand(1, 4096, 1, 2048).contiguous()
    return q, kv, do, indices


def timed(callable) -> tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires an SM89 RTX 4090")
    if not 0 <= args.warmup <= MAX_WARMUP:
        raise SystemExit(f"warmup must be between 0 and {MAX_WARMUP}")
    if not 1 <= args.iterations <= MAX_ITERATIONS:
        raise SystemExit(f"iterations must be between 1 and {MAX_ITERATIONS}")

    memory = two_stage_memory(args.sequence_kv)
    if memory.total_bytes > MAX_WORKSPACE_BYTES:
        raise SystemExit(f"workspace bound failed: {memory.total_bytes} bytes")
    q, kv, do, indices = make_inputs(args.sequence_kv)
    scale = 576**-0.5
    os.environ["PRIME_SPARSE_BWD_TWO_STAGE"] = "0"
    output, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale)
    (baseline_dq, baseline_dkv), baseline_ms = timed(
        lambda: sparse_mla_bwd(q, kv, output, do, indices, lse, sm_scale=scale)
    )
    os.environ["PRIME_SPARSE_BWD_TWO_STAGE"] = "1"
    candidate, candidate_ms = timed(
        lambda: sparse_mla_bwd(q, kv, output, do, indices, lse, sm_scale=scale)
    )
    if not torch.equal(baseline_dq, candidate[0]):
        raise SystemExit("dQ is not exact against the baseline")
    torch.testing.assert_close(candidate[1], baseline_dkv, rtol=2e-2, atol=2e-3)
    if not bool(torch.isfinite(candidate[0]).all() and torch.isfinite(candidate[1]).all()):
        raise SystemExit("candidate returned a nonfinite gradient")
    delta_kernel = preprocess(64, 512)
    delta = delta_kernel(output, do)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    direct_candidate, _ = timed(
        lambda: sparse_mla_dkv_two_stage(
            q, kv, do, indices, lse, delta, sm_scale=scale
        )
    )
    peak_workspace = torch.cuda.max_memory_allocated() - allocated_before
    if peak_workspace > MAX_WORKSPACE_BYTES:
        raise SystemExit(
            f"measured workspace bound failed: {peak_workspace} > {MAX_WORKSPACE_BYTES} bytes"
        )
    torch.testing.assert_close(direct_candidate[0], baseline_dkv, rtol=2e-2, atol=2e-3)
    if not bool(torch.isfinite(direct_candidate[0]).all()):
        raise SystemExit("direct candidate returned a nonfinite dKV")
    times: list[float] = []
    for _ in range(args.warmup):
        sparse_mla_bwd(q, kv, output, do, indices, lse, sm_scale=scale)
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        _, elapsed = timed(
            lambda: sparse_mla_bwd(q, kv, output, do, indices, lse, sm_scale=scale)
        )
        times.append(elapsed)
    print(
        f"device={torch.cuda.get_device_name()} capability=sm_89 "
        f"shape=B1,S4096,S_kv{args.sequence_kv},H64,G1,topk2048"
    )
    print(
        f"workspace_bytes={memory.total_bytes} workspace_mib={memory.total_bytes / 2**20:.2f} "
        f"workspace_limit_bytes={MAX_WORKSPACE_BYTES} measured_workspace_bytes={peak_workspace} "
        f"dq_exact=1 dkv_close=1 finite=1"
    )
    print(
        f"baseline_ms={baseline_ms:.3f} candidate_first_ms={candidate_ms:.3f} "
        f"candidate_median_ms={statistics.median(times):.3f} candidate_max_ms={max(times):.3f}"
    )


if __name__ == "__main__":
    main()
