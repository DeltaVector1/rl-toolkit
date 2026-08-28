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
from prime_rl.trainer.models.kernels.sparse_mla_two_stage import sparse_mla_dkv_two_stage


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
    raise TimeoutError("SM89 bucketed sparse MLA benchmark exceeded its host timeout")


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

    memory = bucketed_memory(SEQUENCE_KV)
    if memory.total_bytes > MAX_WORKSPACE_BYTES:
        raise SystemExit(f"workspace bound failed: {memory.total_bytes} bytes")
    q, kv, do, indices = make_inputs(SEQUENCE_KV)
    scale = 576**-0.5
    output, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale)
    delta_kernel = preprocess(64, 512)
    delta = delta_kernel(output, do)
    torch.cuda.synchronize()

    baseline_output = torch.empty_like(kv, dtype=torch.float32)
    candidate_output = torch.empty_like(kv, dtype=torch.float32)
    _, baseline_ms = timed(
        lambda: sparse_mla_dkv_two_stage(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=baseline_output
        )
    )
    _, candidate_first_ms = timed(
        lambda: sparse_mla_dkv_bucketed(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=candidate_output
        )
    )
    baseline = baseline_output.view_as(kv).to(dtype=kv.dtype)
    candidate = candidate_output.view_as(kv).to(dtype=kv.dtype)
    torch.testing.assert_close(candidate, baseline, rtol=2e-2, atol=2e-3)
    if not bool(torch.isfinite(candidate).all() and torch.isfinite(baseline).all()):
        raise SystemExit("bucketed candidate returned nonfinite dKV")

    for _ in range(args.warmup):
        sparse_mla_dkv_bucketed(
            q, kv, do, indices, lse, delta, sm_scale=scale, output=candidate_output
        )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(q.device)
    reserved_before = torch.cuda.memory_reserved(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)
    times: list[float] = []
    for _ in range(args.iterations):
        _, elapsed = timed(
            lambda: sparse_mla_dkv_bucketed(
                q, kv, do, indices, lse, delta, sm_scale=scale, output=candidate_output
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
    print(
        f"device={torch.cuda.get_device_name()} capability=sm_89 "
        f"shape=B1,S4096,S_kv{SEQUENCE_KV},H64,G1,topk2048 "
        "route_skew=2048_keys_x_4096_routes order=atomic"
    )
    print(
        f"ledger_total_bytes={memory.total_bytes} ledger_total_mib={memory.total_bytes / 2**20:.2f} "
        f"route_sources_bytes={memory.route_sources_bytes} histogram_bytes={memory.histogram_bytes} "
        f"offsets_bytes={memory.offsets_bytes} cursor_bytes={memory.cursor_bytes} "
        f"prefix_temp_bytes={memory.prefix_temp_bytes} output_fp32_bytes={memory.output_fp32_bytes} "
        f"output_bf16_bytes={memory.output_bf16_bytes} workspace_limit_bytes={MAX_WORKSPACE_BYTES} "
        f"peak_allocated_delta={peak_allocated_delta} peak_reserved_delta={peak_reserved_delta} "
        "dkv_close=1 finite=1"
    )
    print(
        f"baseline_ms={baseline_ms:.3f} candidate_first_ms={candidate_first_ms:.3f} "
        f"candidate_median_ms={statistics.median(times):.3f} candidate_max_ms={max(times):.3f}"
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
