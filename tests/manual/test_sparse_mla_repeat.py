"""Repeatability gate for the production SM89 sparse MLA kernels."""

from __future__ import annotations

import argparse

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
    first, first_lse = sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=scale
    )
    second, _ = sparse_mla_fwd_interface(query, key_value, indices, sm_scale=scale)
    torch.cuda.synchronize()
    print(
        f"forward_repeat_mismatch={mismatch(second, first)}/{first.numel()}",
        flush=True,
    )

    output_gradient = torch.randn(
        first.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    query_gradient, key_value_gradient = sparse_mla_bwd(
        query,
        key_value,
        first,
        output_gradient,
        indices,
        first_lse,
        sm_scale=scale,
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
        f"{key_value.numel()}",
        flush=True,
    )
    sparse_mla_bwd(
        query,
        key_value,
        third,
        output_gradient,
        indices,
        first_lse,
        sm_scale=scale,
    )
    fourth, _ = sparse_mla_fwd_interface(query, key_value, indices, sm_scale=scale)
    torch.cuda.synchronize()
    print(
        f"post_second_backward_output_mismatch={mismatch(fourth, first)}/"
        f"{first.numel()} query_mutation={mismatch(query, query_reference)}/"
        f"{query.numel()} key_value_mutation="
        f"{mismatch(key_value, key_value_reference)}/{key_value.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
