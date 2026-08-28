"""Bounded comparison for the opt-in SM89 bucketed sparse MLA reducer."""

from __future__ import annotations

import argparse
import signal
import statistics

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import preprocess
from prime_rl.trainer.models.kernels.sparse_mla_bucketed import (
    MAX_WORKSPACE_BYTES,
    bucketed_memory,
    sparse_mla_dkv_bucketed,
)
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
from prime_rl.trainer.models.kernels.sparse_mla_two_stage import (
    sparse_mla_dkv_two_stage,
)


MAX_WARMUP = 1
MAX_ITERATIONS = 3
SEQUENCE_KV = 32769
TIMEOUT_SECONDS = 900
ROUTE_DISTRIBUTIONS = ("collision", "uniform-wide", "moderate-skew")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument(
        "--route-distribution",
        choices=(*ROUTE_DISTRIBUTIONS, "all"),
        default="all",
        help="route pattern to benchmark; all runs each pattern",
    )
    return parser.parse_args()


def make_inputs(sequence_kv: int, route_distribution: str) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    q = torch.randn(
        1, 4096, 64, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv = torch.randn(
        1, sequence_kv, 1, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv[:, -1].zero_()
    do = torch.randn(
        1, 4096, 64, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    route_count = 4096 * 2048
    source = torch.arange(route_count, device="cuda", dtype=torch.int32)
    query = torch.div(source, 2048, rounding_mode="floor")
    rank = source.remainder(2048)
    if route_distribution == "collision":
        keys = rank
    elif route_distribution == "uniform-wide":
        keys = (rank * 17 + query * 8191).remainder(sequence_kv - 1)
    elif route_distribution == "moderate-skew":
        hot_routes = 1536
        hot_pool = 8192
        hot = (rank * 17 + query * 8191).remainder(hot_pool)
        cold_rank = rank - hot_routes
        cold = hot_pool + (cold_rank * 17 + query * 8191).remainder(
            sequence_kv - 1 - hot_pool
        )
        keys = torch.where(rank < hot_routes, hot, cold)
    else:
        raise ValueError(f"unknown route distribution: {route_distribution}")
    indices = keys.view(1, 4096, 1, 2048).contiguous()
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
    raise TimeoutError("SM89 bucketed sparse MLA benchmark exceeded its host timeout")


def run_distribution(
    sequence_kv: int,
    route_distribution: str,
    warmup: int,
    iterations: int,
) -> tuple[str, str, str]:
    memory = bucketed_memory(sequence_kv)
    if memory.total_bytes > MAX_WORKSPACE_BYTES:
        raise SystemExit(f"workspace bound failed: {memory.total_bytes} bytes")
    q, kv, do, indices = make_inputs(sequence_kv, route_distribution)
    scale = 576**-0.5
    output, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale)
    delta_kernel = preprocess(64, 512)
    delta = delta_kernel(output, do)
    torch.cuda.synchronize()

    baseline_output = torch.empty_like(kv, dtype=torch.float32)
    candidate_output = torch.empty_like(kv, dtype=torch.float32)
    _, baseline_ms = timed(
        lambda q=q,
        kv=kv,
        do=do,
        indices=indices,
        lse=lse,
        delta=delta,
        output=baseline_output: sparse_mla_dkv_two_stage(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=output
        )
    )
    _, candidate_first_ms = timed(
        lambda q=q,
        kv=kv,
        do=do,
        indices=indices,
        lse=lse,
        delta=delta,
        output=candidate_output: sparse_mla_dkv_bucketed(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=output
        )
    )
    baseline = baseline_output.view_as(kv).to(dtype=kv.dtype)
    candidate = candidate_output.view_as(kv).to(dtype=kv.dtype)
    torch.testing.assert_close(candidate, baseline, rtol=2e-2, atol=2e-3)
    if not bool(torch.isfinite(candidate).all() and torch.isfinite(baseline).all()):
        raise SystemExit("bucketed candidate returned nonfinite dKV")

    for _ in range(warmup):
        sparse_mla_dkv_bucketed(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=candidate_output
        )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(q.device)
    reserved_before = torch.cuda.memory_reserved(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)
    times: list[float] = []
    for _ in range(iterations):
        _, elapsed = timed(
            lambda q=q,
            kv=kv,
            do=do,
            indices=indices,
            lse=lse,
            delta=delta,
            output=candidate_output: sparse_mla_dkv_bucketed(
                q, kv, do, indices, lse, delta, sm_scale=scale, output=output
            )
        )
        times.append(elapsed)
    peak_allocated_delta = torch.cuda.max_memory_allocated(q.device) - allocated_before
    peak_reserved_delta = torch.cuda.max_memory_reserved(q.device) - reserved_before
    if peak_allocated_delta > MAX_WORKSPACE_BYTES:
        raise SystemExit(
            f"measured allocated-memory bound failed: {peak_allocated_delta} > {MAX_WORKSPACE_BYTES}"
        )
    if peak_reserved_delta > MAX_WORKSPACE_BYTES:
        raise SystemExit(
            f"measured reserved-memory bound failed: {peak_reserved_delta} > {MAX_WORKSPACE_BYTES}"
        )
    messages = (
        f"device={torch.cuda.get_device_name()} capability=sm_89 "
        f"shape=B1,S4096,S_kv{sequence_kv},H64,G1,topk2048 "
        f"route_distribution={route_distribution} order=atomic",
        f"ledger_total_bytes={memory.total_bytes} ledger_total_mib={memory.total_bytes / 2**20:.2f} "
        f"route_sources_bytes={memory.route_sources_bytes} histogram_bytes={memory.histogram_bytes} "
        f"offsets_bytes={memory.offsets_bytes} cursor_bytes={memory.cursor_bytes} "
        f"prefix_temp_bytes={memory.prefix_temp_bytes} output_fp32_bytes={memory.output_fp32_bytes} "
        f"output_bf16_bytes={memory.output_bf16_bytes} workspace_limit_bytes={MAX_WORKSPACE_BYTES} "
        f"peak_allocated_delta={peak_allocated_delta} peak_reserved_delta={peak_reserved_delta} "
        "dkv_close=1 finite=1",
        f"baseline_ms={baseline_ms:.3f} candidate_first_ms={candidate_first_ms:.3f} "
        f"candidate_median_ms={statistics.median(times):.3f} candidate_max_ms={max(times):.3f}",
    )
    del q, kv, do, indices, output, lse, delta, baseline_output, candidate_output
    torch.cuda.empty_cache()
    return messages


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
        raise SystemExit(f"timeout-seconds must be between 1 and {TIMEOUT_SECONDS}")

    distributions = (
        ROUTE_DISTRIBUTIONS
        if args.route_distribution == "all"
        else (args.route_distribution,)
    )
    for route_distribution in distributions:
        for line in run_distribution(
            SEQUENCE_KV, route_distribution, args.warmup, args.iterations
        ):
            print(line, flush=True)


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
