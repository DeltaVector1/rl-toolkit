"""Repeatability gate for the production SM89 sparse MLA kernels."""

from __future__ import annotations

import argparse
import time

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2048)
    return parser.parse_args()


def mismatch(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left != right).sum().item())


def timed(callable):
    """Return one synchronized CUDA execution and its elapsed milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = callable()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end), time.perf_counter() - wall_start


def main() -> None:
    args = arguments()
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this gate requires an SM89 GPU")

    generator = torch.Generator(device="cuda").manual_seed(20260810)
    query = torch.randn(
        1,
        args.seq_len,
        args.heads,
        576,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value = torch.randn(
        1,
        args.seq_len + 1,
        1,
        576,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value[:, -1].zero_()
    key_positions = torch.arange(args.topk, device="cuda", dtype=torch.int32)
    query_positions = torch.arange(
        1, args.seq_len + 1, device="cuda", dtype=torch.int32
    ).unsqueeze(1)
    indices = key_positions.unsqueeze(0).expand(args.seq_len, -1)
    indices = torch.where(indices < query_positions, indices, args.seq_len)
    indices = indices.view(1, args.seq_len, 1, args.topk).contiguous()
    scale = 576**-0.5

    query_reference = query.clone()
    key_value_reference = key_value.clone()
    (first, first_lse), first_forward_ms, first_forward_wall = timed(
        lambda: sparse_mla_fwd_interface(
            query, key_value, indices, sm_scale=scale
        )
    )
    (second, _), steady_forward_ms, steady_forward_wall = timed(
        lambda: sparse_mla_fwd_interface(
            query, key_value, indices, sm_scale=scale
        )
    )
    torch.cuda.synchronize()
    print(
        f"forward_repeat_mismatch={mismatch(second, first)}/{first.numel()} "
        f"first_cuda_ms={first_forward_ms:.3f} "
        f"first_wall_s={first_forward_wall:.3f} "
        f"steady_cuda_ms={steady_forward_ms:.3f} "
        f"steady_wall_s={steady_forward_wall:.3f}",
        flush=True,
    )

    output_gradient = torch.randn(
        first.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    (query_gradient, key_value_gradient), first_backward_ms, first_backward_wall = timed(
        lambda: sparse_mla_bwd(
            query,
            key_value,
            first,
            output_gradient,
            indices,
            first_lse,
            sm_scale=scale,
        )
    )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(query_gradient).all()):
        raise RuntimeError("query gradient is nonfinite")
    if not bool(torch.isfinite(key_value_gradient).all()):
        raise RuntimeError("key/value gradient is nonfinite")

    third, _ = sparse_mla_fwd_interface(query, key_value, indices, sm_scale=scale)
    torch.cuda.synchronize()
    print(
        f"post_backward_output_mismatch={mismatch(third, first)}/{first.numel()} "
        f"query_mutation={mismatch(query, query_reference)}/{query.numel()} "
        f"key_value_mutation={mismatch(key_value, key_value_reference)}/"
        f"{key_value.numel()} first_backward_cuda_ms={first_backward_ms:.3f} "
        f"first_backward_wall_s={first_backward_wall:.3f}",
        flush=True,
    )
    _, steady_backward_ms, steady_backward_wall = timed(
        lambda: sparse_mla_bwd(
            query,
            key_value,
            third,
            output_gradient,
            indices,
            first_lse,
            sm_scale=scale,
        )
    )
    fourth, _ = sparse_mla_fwd_interface(query, key_value, indices, sm_scale=scale)
    torch.cuda.synchronize()
    print(
        f"post_second_backward_output_mismatch={mismatch(fourth, first)}/"
        f"{first.numel()} query_mutation={mismatch(query, query_reference)}/"
        f"{query.numel()} key_value_mutation="
        f"{mismatch(key_value, key_value_reference)}/{key_value.numel()} "
        f"steady_backward_cuda_ms={steady_backward_ms:.3f} "
        f"steady_backward_wall_s={steady_backward_wall:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
