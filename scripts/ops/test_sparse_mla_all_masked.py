"""CUDA regression gate for a fully masked sparse-MLA query."""

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import (
    sparse_mla_fwd_interface,
)


def main() -> None:
    query = torch.randn(
        1, 1, 64, 576, device="cuda", dtype=torch.bfloat16
    )
    # Index 1 is the zero sentinel. Index 0 is the only valid K row.
    key_value = torch.zeros(
        1, 2, 1, 576, device="cuda", dtype=torch.bfloat16
    )
    indices = torch.ones(
        1, 1, 1, 2048, device="cuda", dtype=torch.int32
    )
    scale = 256**-0.5

    output, lse = sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=scale
    )
    query_grad, key_value_grad = sparse_mla_bwd(
        query,
        key_value,
        output,
        torch.randn_like(output),
        indices,
        lse,
        sm_scale=scale,
    )

    if torch.count_nonzero(output):
        raise RuntimeError("a fully masked sparse-MLA output was not zero")
    if not bool(torch.isposinf(lse).all()):
        raise RuntimeError("a fully masked sparse-MLA LSE was not +inf")
    if not bool(
        torch.isfinite(query_grad).all()
        and torch.isfinite(key_value_grad).all()
    ):
        raise RuntimeError("a fully masked sparse-MLA gradient was nonfinite")
    print("fully masked sparse MLA: zero output and finite zero gradients")


if __name__ == "__main__":
    main()
