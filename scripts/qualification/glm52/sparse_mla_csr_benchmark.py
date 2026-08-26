"""Isolated sparse MLA backward benchmark.

Run this only on an idle permitted trainer GPU in the pinned image.  It does
not start a trainer or modify a service.  The default dimensions match the
SM89 production report in ``docs/research/sparse-mla-backward-redesign.md``.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_csr import (
    DEFAULT_ENTRY_CHUNK,
    DEFAULT_KEY_TILE,
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_PARTIAL_BYTES,
    DEFAULT_QUERY_CHUNK,
    build_reverse_csr,
    estimate_partial_bytes,
    sparse_mla_dkv_csr,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--sequence-kv", type=int, default=4097)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--tail", type=int, default=64, choices=(16, 64, 128))
    parser.add_argument("--query-chunk", type=int, default=DEFAULT_QUERY_CHUNK)
    parser.add_argument("--key-tile", type=int, default=DEFAULT_KEY_TILE)
    parser.add_argument("--entry-chunk", type=int, default=DEFAULT_ENTRY_CHUNK)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def _event_time(function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _timed(function, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    return [_event_time(function) for _ in range(iterations)]


def _summary(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return median, p95


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run in the pinned PyTorch 2.11/CUDA 12.8 image")
    if args.sequence_kv <= 1 or args.sequence_kv < args.sequence:
        raise SystemExit("sequence-kv must include the final sentinel and cover sequence")

    device = torch.device("cuda")
    torch.manual_seed(7)
    width = 512 + args.tail
    q = torch.randn(args.batch, args.sequence, args.heads, width, device=device, dtype=torch.bfloat16)
    kv = torch.randn(args.batch, args.sequence_kv, args.groups, width, device=device, dtype=torch.bfloat16)
    do = torch.randn(args.batch, args.sequence, args.heads, 512, device=device, dtype=torch.bfloat16)
    o = torch.randn_like(do)
    lse = torch.randn(args.batch, args.sequence, args.heads, device=device, dtype=torch.float32)
    indices = torch.randint(
        0,
        args.sequence_kv - 1,
        (args.batch, args.sequence, args.groups, args.topk),
        device=device,
        dtype=torch.int32,
    )
    scale = width ** -0.5

    partial_bytes = estimate_partial_bytes(
        args.batch,
        args.sequence,
        args.groups,
        args.sequence_kv,
        width,
        query_chunk=args.query_chunk,
        key_tile=args.key_tile,
    )
    reverse = build_reverse_csr(
        indices,
        args.sequence_kv,
        key_tile=args.key_tile,
        max_index_bytes=DEFAULT_MAX_INDEX_BYTES,
    )
    print(f"shape B={args.batch} S={args.sequence} S_kv={args.sequence_kv} H={args.heads} G={args.groups} topk={args.topk} tail={args.tail}")
    print(f"valid_entries={reverse.valid_entries} reverse_index_bytes={reverse.index_bytes} partial_bytes={partial_bytes}")
    print(f"partial_budget={DEFAULT_MAX_PARTIAL_BYTES} entry_chunk={args.entry_chunk}")

    baseline = None
    if not args.skip_baseline:
        os.environ["PRIME_SPARSE_BWD_KEY_MAJOR_CSR"] = "0"
        baseline = sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=scale)
        baseline_times = _timed(
            lambda: sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=scale),
            args.warmup,
            args.iterations,
        )
        median, p95 = _summary(baseline_times)
        print(f"production median_ms={median:.3f} p95_ms={p95:.3f} kernel_count=3 dkv_atomics=enabled")
    else:
        print("production baseline skipped")

    # Run the candidate through the same q-major dQ-only wrapper.  The direct
    # helper below times the key-major CSR dKV work separately.
    os.environ["PRIME_SPARSE_BWD_KEY_MAJOR_CSR"] = "1"
    candidate = sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=scale)
    candidate_times = _timed(
        lambda: sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=scale),
        args.warmup,
        args.iterations,
    )
    median, p95 = _summary(candidate_times)
    print(f"candidate median_ms={median:.3f} p95_ms={p95:.3f} q_major_dq=1 csr_dkv=1 dkv_atomics=0")

    delta = torch.sum(o.float() * do.float(), dim=-1)
    csr_times = _timed(
        lambda: sparse_mla_dkv_csr(
            q,
            kv,
            do,
            indices,
            lse,
            delta,
            sm_scale=scale,
            query_chunk=args.query_chunk,
            key_tile=args.key_tile,
            entry_chunk=args.entry_chunk,
        ),
        args.warmup,
        args.iterations,
    )
    median, p95 = _summary(csr_times)
    print(f"csr_dkv_only median_ms={median:.3f} p95_ms={p95:.3f}")

    if baseline is not None:
        dq_error = (candidate[0].float() - baseline[0].float()).abs()
        dkv_error = (candidate[1].float() - baseline[1].float()).abs()
        print(f"dq_max_abs={dq_error.max().item():.6g} dkv_max_abs={dkv_error.max().item():.6g}")
    print(f"candidate_nonfinite_dq={int((~torch.isfinite(candidate[0])).sum())} candidate_nonfinite_dkv={int((~torch.isfinite(candidate[1])).sum())}")
    print("note=kernel count is logical launch count; use Nsight Compute for hardware instruction counters")


if __name__ == "__main__":
    main()
