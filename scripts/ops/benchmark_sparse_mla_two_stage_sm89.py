"""Bounded end-to-end benchmark for the opt-in SM89 sparse MLA reducer."""

from __future__ import annotations

import argparse
import os
import signal
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
SEQUENCE_KV = 32769
TIMEOUT_SECONDS = 900


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS)
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


def _timeout_handler(_signum, _frame):
    raise TimeoutError("SM89 sparse MLA benchmark exceeded its host timeout")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires an SM89 RTX 4090")
    if not 0 <= args.warmup <= MAX_WARMUP:
        raise SystemExit(f"warmup must be between 0 and {MAX_WARMUP}")
    if not 1 <= args.iterations <= MAX_ITERATIONS:
        raise SystemExit(f"iterations must be between 1 and {MAX_ITERATIONS}")
    if not 1 <= args.timeout_seconds <= TIMEOUT_SECONDS:
        raise SystemExit(
            f"timeout-seconds must be between 1 and {TIMEOUT_SECONDS}"
        )

    memory = two_stage_memory(SEQUENCE_KV)
    if memory.total_bytes > MAX_WORKSPACE_BYTES:
        raise SystemExit(f"workspace bound failed: {memory.total_bytes} bytes")
    q, kv, do, indices = make_inputs(SEQUENCE_KV)
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
    if not bool(
        torch.isfinite(baseline_dq).all()
        and torch.isfinite(baseline_dkv).all()
        and torch.isfinite(candidate[0]).all()
        and torch.isfinite(candidate[1]).all()
    ):
        raise SystemExit("candidate returned a nonfinite gradient")
    delta_kernel = preprocess(64, 512)
    delta = delta_kernel(output, do)
    torch.cuda.synchronize()
    free_hbm_before, total_hbm = torch.cuda.mem_get_info(q.device)
    allocated_before = torch.cuda.memory_allocated(q.device)
    reserved_before = torch.cuda.memory_reserved(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)

    direct_output = None

    def direct_call():
        nonlocal direct_output
        direct_output = torch.empty_like(kv, dtype=torch.float32)
        return sparse_mla_dkv_two_stage(
            q,
            kv,
            do,
            indices,
            lse,
            delta,
            sm_scale=scale,
            output=direct_output,
        )

    direct_candidate, direct_ms = timed(
        direct_call
    )
    peak_allocated_delta = torch.cuda.max_memory_allocated(q.device) - allocated_before
    peak_reserved_delta = torch.cuda.max_memory_reserved(q.device) - reserved_before
    free_hbm_after, _ = torch.cuda.mem_get_info(q.device)
    if peak_allocated_delta > MAX_WORKSPACE_BYTES:
        raise SystemExit(
            "measured allocated-memory bound failed: "
            f"{peak_allocated_delta} > {MAX_WORKSPACE_BYTES} bytes"
        )
    if peak_reserved_delta > MAX_WORKSPACE_BYTES:
        raise SystemExit(
            "measured reserved-memory bound failed: "
            f"{peak_reserved_delta} > {MAX_WORKSPACE_BYTES} bytes"
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
        f"shape=B1,S4096,S_kv{SEQUENCE_KV},H64,G1,topk2048 "
        "route_skew=2048_keys_x_4096_routes"
    )
    print(
        f"workspace_bytes={memory.total_bytes} workspace_mib={memory.total_bytes / 2**20:.2f} "
        f"route_sort_bytes={memory.route_sort_bytes} route_filter_bytes={memory.route_filter_bytes} "
        f"reduction_bytes={memory.reduction_bytes} output_pair_bytes={memory.output_pair_bytes} "
        f"workspace_limit_bytes={MAX_WORKSPACE_BYTES} "
        f"peak_allocated_delta={peak_allocated_delta} peak_reserved_delta={peak_reserved_delta} "
        f"free_hbm_before={free_hbm_before} free_hbm_after={free_hbm_after} total_hbm={total_hbm} "
        f"dq_exact=1 dkv_close=1 finite=1"
    )
    print(
        f"baseline_ms={baseline_ms:.3f} candidate_first_ms={candidate_ms:.3f} "
        f"candidate_median_ms={statistics.median(times):.3f} candidate_max_ms={max(times):.3f} "
        f"direct_ms={direct_ms:.3f}"
    )


def main() -> None:
    args = arguments()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    try:
        run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
