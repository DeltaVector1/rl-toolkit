"""Bounded SM89 comparison for the 16- and 32-head backward tiles."""

from __future__ import annotations

import argparse
import os

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


_QUALIFICATION_PROFILES = {
    "glm52-native-small",
    "glm52-native-b2048",
    "glm52-native-b2048-memory",
    "glm52-native-b2048-nsys",
    "glm52-native-delta3",
    "glm52-native-stability20",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.seq_len < 1 or args.seq_len > 128:
        raise ValueError("--seq-len must be between 1 and 128")
    if args.warmup < 0 or args.warmup > 2:
        raise ValueError("--warmup must be between 0 and 2")
    if args.repeats < 1 or args.repeats > 3:
        raise ValueError("--repeats must be between 1 and 3")
    if os.environ.get("PRIME_QUALIFICATION_PROFILE") not in _QUALIFICATION_PROFILES:
        raise RuntimeError(
            "set PRIME_QUALIFICATION_PROFILE to an allowlisted GLM qualification "
            "profile before running this comparison"
        )


def _set_geometry(block_heads: int) -> None:
    if block_heads not in {16, 32}:
        raise ValueError("block_heads must be 16 or 32")
    os.environ["PRIME_SPARSE_BWD_USE_BLOCK_H32"] = "1" if block_heads == 32 else "0"
    os.environ["PRIME_SPARSE_BWD_BLOCK_HEADS"] = str(block_heads)


def _timed_backward(
    query: torch.Tensor,
    key_value: torch.Tensor,
    output: torch.Tensor,
    output_gradient: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    block_heads: int,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float], int]:
    _set_geometry(block_heads)
    for _ in range(warmup):
        sparse_mla_bwd(
            query, key_value, output, output_gradient, indices, lse, sm_scale=scale
        )
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    query_gradient = key_value_gradient = None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        query_gradient, key_value_gradient = sparse_mla_bwd(
            query, key_value, output, output_gradient, indices, lse, sm_scale=scale
        )
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    assert query_gradient is not None and key_value_gradient is not None
    if not bool(torch.isfinite(query_gradient).all()):
        raise RuntimeError(f"block_H={block_heads} produced a nonfinite query gradient")
    if not bool(torch.isfinite(key_value_gradient).all()):
        raise RuntimeError(f"block_H={block_heads} produced a nonfinite key/value gradient")
    return (
        query_gradient,
        key_value_gradient,
        elapsed_ms,
        torch.cuda.max_memory_allocated(),
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> tuple[int, float, float]:
    absolute = (left.float() - right.float()).abs()
    scale = right.float().abs().clamp_min(1e-12)
    return int((left != right).sum().item()), float(absolute.max()), float((absolute / scale).max())


def main() -> None:
    args = _arguments()
    _validate(args)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this comparison requires an SM89 CUDA device")

    batch = 1
    heads = 64
    topk = 2048
    seq_len = args.seq_len
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    query = torch.randn(
        batch, seq_len, heads, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.02)
    key_value = torch.randn(
        batch, seq_len + 1, 1, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.02)
    key_value[:, -1].zero_()
    key_positions = torch.arange(topk, device="cuda", dtype=torch.int32)
    query_positions = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.int32).unsqueeze(1)
    indices = torch.where(key_positions.unsqueeze(0) < query_positions, key_positions, seq_len)
    indices = indices.view(batch, seq_len, 1, topk).contiguous()
    scale = 576**-0.5

    output, lse = sparse_mla_fwd_interface(query, key_value, indices, sm_scale=scale)
    output_gradient = torch.randn(
        output.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(1e-4)

    baseline = _timed_backward(
        query, key_value, output, output_gradient, indices, lse, scale, 16, args.warmup, args.repeats
    )
    candidate = _timed_backward(
        query, key_value, output, output_gradient, indices, lse, scale, 32, args.warmup, args.repeats
    )
    dq_mismatch, dq_abs, dq_rel = _difference(candidate[0], baseline[0])
    dkv_mismatch, dkv_abs, dkv_rel = _difference(candidate[1], baseline[1])
    if dq_abs > 0.02 or dkv_abs > 0.02:
        raise RuntimeError(
            "block_H=32 backward parity exceeded the bounded tolerance: "
            f"dQ_abs={dq_abs:.6g} dKV_abs={dkv_abs:.6g}"
        )

    print(
        f"shape batch={batch} seq_len={seq_len} heads={heads} topk={topk} "
        f"warmup={args.warmup} repeats={args.repeats}",
        flush=True,
    )
    print(
        f"block_H=16 times_ms={','.join(f'{value:.3f}' for value in baseline[2])} "
        f"peak_allocated={baseline[3]}",
        flush=True,
    )
    print(
        f"block_H=32 times_ms={','.join(f'{value:.3f}' for value in candidate[2])} "
        f"peak_allocated={candidate[3]}",
        flush=True,
    )
    print(
        f"parity dQ_mismatch={dq_mismatch}/{candidate[0].numel()} dQ_abs={dq_abs:.6g} "
        f"dQ_rel={dq_rel:.6g} dKV_mismatch={dkv_mismatch}/{candidate[1].numel()} "
        f"dKV_abs={dkv_abs:.6g} dKV_rel={dkv_rel:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
