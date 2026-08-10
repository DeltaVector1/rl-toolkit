"""Production-shape CUDA gate for routed-expert inverse permutation memory.

Run on one free RTX 4090 in the trainer image. The row count reproduces the
426 MiB BF16 allocation that failed x60k at hidden size 6,144.
"""

from __future__ import annotations

import json

import torch

from prime_rl.trainer.models.layers import fp8_ada
from prime_rl.trainer.models.layers.fp8_ada import scale_routed_output


GIB = 1024**3
MIB = 1024**2
ROWS = 36_352
HIDDEN = 6_144
SENTINEL_REPEATS = 128


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    fp8_ada._FP8_EXPERT_ACTIVATION_OFFLOAD = True
    generator = torch.Generator(device="cuda").manual_seed(61)
    value = torch.randn(
        ROWS,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    permutation = torch.randperm(ROWS, device="cuda", generator=generator)
    indices = torch.cat(
        (
            permutation,
            torch.full(
                (SENTINEL_REPEATS,),
                ROWS,
                device="cuda",
                dtype=torch.int64,
            ),
        )
    )
    padding_rows = indices == ROWS
    permuted = value[indices.clamp_max(ROWS - 1)]
    permuted.masked_fill_(padding_rows.unsqueeze(-1), 0)
    expert_output = permuted.mul(2)

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = permuted
    out[indices] = expert_output
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - before
    if incremental_peak > 64 * MIB:
        raise RuntimeError(
            "inverse permutation allocated another routed-output-sized buffer: "
            f"incremental peak {incremental_peak / MIB:.2f} MiB"
        )

    torch.testing.assert_close(out[:ROWS], value.mul(2), rtol=0, atol=0)
    out[:ROWS].sum().backward()
    if value.grad is None or not bool(torch.isfinite(value.grad).all()):
        raise RuntimeError(
            "inverse permutation produced a missing or nonfinite gradient"
        )
    torch.testing.assert_close(
        value.grad,
        torch.full_like(value, 2),
        rtol=0,
        atol=0,
    )
    del (
        out,
        expert_output,
        permuted,
        permutation,
        indices,
        padding_rows,
        value,
    )
    torch.cuda.empty_cache()

    route_leaf = torch.randn(
        ROWS,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    route_value = route_leaf * 1
    route_scores = torch.randn(
        ROWS,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    torch.cuda.synchronize()
    route_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    route_pointer = route_value.data_ptr()
    routed = scale_routed_output(route_value, route_scores)
    torch.cuda.synchronize()
    route_forward_incremental = torch.cuda.max_memory_allocated() - route_before
    if routed.data_ptr() != route_pointer:
        raise RuntimeError("route scaling did not reuse expert-output storage")
    if route_forward_incremental > 256 * MIB:
        raise RuntimeError(
            "in-place route scaling exceeded its bounded forward budget: "
            f"incremental peak {route_forward_incremental / MIB:.2f} MiB"
        )
    upstream = torch.randn_like(routed)
    expected_output = (route_leaf.detach().float() * route_scores.detach()[:, None]).to(
        route_leaf.dtype
    )
    expected_value_grad = (upstream.float() * route_scores.detach()[:, None]).to(
        route_leaf.dtype
    )
    expected_score_grad = (upstream.float() * route_leaf.detach().float()).sum(dim=-1)
    routed.backward(upstream)
    torch.cuda.synchronize()
    route_total_peak = torch.cuda.max_memory_allocated()
    torch.testing.assert_close(routed, expected_output, rtol=0, atol=0)
    if route_leaf.grad is None or route_scores.grad is None:
        raise RuntimeError("chunked route scaling produced missing gradients")
    torch.testing.assert_close(route_leaf.grad, expected_value_grad, rtol=0, atol=0)
    torch.testing.assert_close(route_scores.grad, expected_score_grad, rtol=0, atol=0)
    if not bool(torch.isfinite(route_leaf.grad).all()) or not bool(
        torch.isfinite(route_scores.grad).all()
    ):
        raise RuntimeError("chunked route scaling produced nonfinite gradients")
    del routed, route_leaf, route_value, route_scores, upstream
    torch.cuda.empty_cache()

    dispatch_rows = 44_714
    dispatch = torch.full(
        (dispatch_rows, 1, HIDDEN),
        7,
        device="cuda",
        dtype=torch.bfloat16,
    )
    result = dispatch[:, 0]
    padding_mask = torch.arange(dispatch_rows, device="cuda") >= dispatch_rows - 128
    torch.cuda.synchronize()
    mask_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result.masked_fill_(padding_mask.unsqueeze(-1), 0)
    torch.cuda.synchronize()
    mask_incremental_peak = torch.cuda.max_memory_allocated() - mask_before
    if mask_incremental_peak > 8 * MIB:
        raise RuntimeError(
            "in-place expert padding mask allocated an output copy: "
            f"incremental peak {mask_incremental_peak / MIB:.2f} MiB"
        )
    if result.data_ptr() != dispatch.data_ptr():
        raise RuntimeError("expert padding mask did not return dispatch storage")
    if not bool((result[-128:] == 0).all()) or not bool((result[:-128] == 7).all()):
        raise RuntimeError("in-place expert padding mask changed the wrong rows")

    print(
        json.dumps(
            {
                "result": "PASS",
                "shape": [ROWS, HIDDEN],
                "single_buffer_gib": ROWS * HIDDEN * 2 / GIB,
                "incremental_unpermute_peak_mib": incremental_peak / MIB,
                "incremental_padding_mask_peak_mib": mask_incremental_peak / MIB,
                "padding_mask_buffer_gib": dispatch_rows * HIDDEN * 2 / GIB,
                "route_scale_forward_incremental_mib": route_forward_incremental / MIB,
                "route_scale_forward_backward_peak_gib": route_total_peak / GIB,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "gradient": "exact",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
