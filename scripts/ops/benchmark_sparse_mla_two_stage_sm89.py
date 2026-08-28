"""Bounded benchmark for the opt-in SM89 two-stage sparse MLA reducer."""

from __future__ import annotations

import argparse
import statistics

import torch

from prime_rl.trainer.models.kernels.sparse_mla_two_stage import (
    MAX_WORKSPACE_BYTES,
    sparse_mla_dkv_two_stage,
    two_stage_memory,
)


MAX_WARMUP = 2
MAX_ITERATIONS = 5


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
    lse = torch.zeros(1, 4096, 64, device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(lse)
    return q, kv, do, indices, lse, delta


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
    inputs = make_inputs(args.sequence_kv)
    for _ in range(args.warmup):
        sparse_mla_dkv_two_stage(*inputs)
    torch.cuda.synchronize()
    times: list[float] = []
    first = None
    for _ in range(args.iterations):
        result, elapsed = timed(lambda: sparse_mla_dkv_two_stage(*inputs))
        if first is None:
            first = result[0]
        elif not torch.equal(first, result[0]):
            raise SystemExit("repeat output is not exact")
        times.append(elapsed)
    print(
        f"device={torch.cuda.get_device_name()} capability=sm_89 "
        f"shape=B1,S4096,S_kv{args.sequence_kv},H64,G1,topk2048"
    )
    print(
        f"workspace_bytes={memory.total_bytes} workspace_mib={memory.total_bytes / 2**20:.2f} "
        f"workspace_limit_bytes={MAX_WORKSPACE_BYTES} repeat_exact=1"
    )
    print(
        f"median_ms={statistics.median(times):.3f} max_ms={max(times):.3f} "
        f"iterations={args.iterations}"
    )


if __name__ == "__main__":
    main()
