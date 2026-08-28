"""Qualification-only SM89 two-stage sparse MLA dKV reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


SM89 = (8, 9)
QUERY_TOKENS = 4096
ATTENTION_HEADS = 64
VALUE_WIDTH = 512
ROPE_WIDTH = 64
TOTAL_WIDTH = VALUE_WIDTH + ROPE_WIDTH
GROUPS = 1
TOPK = 2048
BLOCK_WIDTH = 128
ROUTE_BYTES_PER_ENTRY = 28
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class TwoStageMemory:
    sequence_kv: int
    route_sort_bytes: int
    reduction_bytes: int
    total_bytes: int


def two_stage_memory(
    sequence_kv: int,
    *,
    route_count: int = QUERY_TOKENS * TOPK,
) -> TwoStageMemory:
    """Return the conservative peak workspace bound.

    Route compaction and sorting happen before the FP32 output is allocated.
    The larger of that temporary and the reduction buffers is the bound.
    """

    if sequence_kv < 1 or route_count < 0:
        raise ValueError("two-stage memory dimensions must be non-negative")
    route_sort_bytes = route_count * ROUTE_BYTES_PER_ENTRY
    reduction_bytes = (
        route_count * 4
        + (sequence_kv + 1) * 8
        + sequence_kv * TOTAL_WIDTH * 4
    )
    return TwoStageMemory(
        sequence_kv=sequence_kv,
        route_sort_bytes=route_sort_bytes,
        reduction_bytes=reduction_bytes,
        total_bytes=max(route_sort_bytes, reduction_bytes),
    )


def supports_glm_sm89_shape(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    *,
    capability: tuple[int, int],
) -> bool:
    """Return true only for the GLM CP shard shapes used by this prototype."""

    if capability != SM89:
        return False
    if tuple(q.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS, TOTAL_WIDTH):
        return False
    if tuple(kv.shape) not in (
        (1, QUERY_TOKENS + 1, GROUPS, TOTAL_WIDTH),
        (1, 32768 + 1, GROUPS, TOTAL_WIDTH),
    ):
        return False
    if tuple(do.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS, VALUE_WIDTH):
        return False
    return tuple(indices.shape) == (1, QUERY_TOKENS, GROUPS, TOPK)


if triton is not None:

    @triton.jit
    def _two_stage_dkv_kernel(
        q,
        kv,
        do,
        lse,
        delta,
        route_positions,
        key_offsets,
        output,
        q_stride_s,
        q_stride_h,
        kv_stride_s,
        do_stride_s,
        do_stride_h,
        lse_stride_s,
        lse_stride_h,
        delta_stride_s,
        delta_stride_h,
        output_stride_s,
        sm_scale,
        BLOCK_WIDTH: tl.constexpr,
    ):
        key_id = tl.program_id(0)
        width_block = tl.program_id(1)
        width = width_block * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        output_mask = width < TOTAL_WIDTH
        route_start = tl.load(key_offsets + key_id)
        route_end = tl.load(key_offsets + key_id + 1)
        accumulator = tl.zeros((BLOCK_WIDTH,), dtype=tl.float32)

        for head in tl.range(0, ATTENTION_HEADS):
            head_partial = tl.zeros((BLOCK_WIDTH,), dtype=tl.float32)
            for route_index in tl.range(route_start, route_end):
                position = tl.load(route_positions + route_index)
                query_id = position // TOPK
                q_base = query_id * q_stride_s + head * q_stride_h
                kv_base = key_id * kv_stride_s

                score = tl.zeros((), dtype=tl.float32)
                for score_start in range(0, TOTAL_WIDTH, BLOCK_WIDTH):
                    score_width = score_start + tl.arange(0, BLOCK_WIDTH)
                    score_mask = score_width < TOTAL_WIDTH
                    q_score = tl.load(
                        q + q_base + score_width,
                        mask=score_mask,
                        other=0.0,
                    ).to(tl.float32)
                    kv_score = tl.load(
                        kv + kv_base + score_width,
                        mask=score_mask,
                        other=0.0,
                    ).to(tl.float32)
                    score += tl.sum(q_score * kv_score, axis=0)

                lse_value = tl.load(lse + query_id * lse_stride_s + head * lse_stride_h)
                probability = tl.exp2(score * (sm_scale * LOG2_E) - lse_value)
                probability = tl.where(probability == probability, probability, 0.0)

                dp_score = tl.zeros((), dtype=tl.float32)
                for score_start in range(0, VALUE_WIDTH, BLOCK_WIDTH):
                    score_width = score_start + tl.arange(0, BLOCK_WIDTH)
                    score_mask = score_width < VALUE_WIDTH
                    do_score = tl.load(
                        do + query_id * do_stride_s + head * do_stride_h + score_width,
                        mask=score_mask,
                        other=0.0,
                    ).to(tl.float32)
                    kv_score = tl.load(
                        kv + kv_base + score_width,
                        mask=score_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dp_score += tl.sum(do_score * kv_score, axis=0)
                delta_value = tl.load(delta + query_id * delta_stride_s + head * delta_stride_h)
                dp = probability * (dp_score - delta_value) * sm_scale
                dp = tl.where(dp == dp, dp, 0.0)

                q_output = tl.load(q + q_base + width, mask=output_mask, other=0.0).to(tl.float32)
                do_output = tl.load(
                    do + query_id * do_stride_s + head * do_stride_h + width,
                    mask=output_mask & (width < VALUE_WIDTH),
                    other=0.0,
                ).to(tl.float32)
                head_partial += dp * q_output + probability * do_output
            accumulator += head_partial

        tl.store(output + key_id * output_stride_s + width, accumulator, mask=output_mask)

else:
    _two_stage_dkv_kernel = None


def _sorted_route_positions(
    indices: torch.Tensor,
    sequence_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build key-major route positions with GPU tensor operations only."""

    flat = indices.reshape(-1)
    positions = torch.arange(flat.numel(), device=indices.device, dtype=torch.int32)
    valid = (flat >= 0) & (flat < sequence_kv - 1)
    keys = flat.masked_select(valid)
    positions = positions.masked_select(valid)
    del flat, valid
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    sorted_positions = positions.index_select(0, order)
    del keys, positions, order
    counts = torch.bincount(sorted_keys, minlength=sequence_kv)
    del sorted_keys
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=indices.device),
            counts.cumsum(0, dtype=torch.int64),
        )
    )
    return sorted_positions, offsets


def sparse_mla_dkv_two_stage(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    *,
    sm_scale: float | None = None,
    max_workspace_bytes: int = MAX_WORKSPACE_BYTES,
) -> tuple[torch.Tensor, TwoStageMemory]:
    """Compute BF16 dKV with one GPU reduction kernel.

    GPU sorting makes routes key-major.  The Triton kernel accumulates each
    key's FP32 head partials in route order and writes one FP32 output row.
    The result is compared with the existing TileLang path at rtol=2e-2 and
    atol=2e-3 during qualification.
    """

    tensors = (q, kv, do, indices, lse, delta)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("two-stage sparse MLA requires CUDA tensors")
    if triton is None or _two_stage_dkv_kernel is None:
        raise RuntimeError("two-stage sparse MLA requires Triton")
    capability = torch.cuda.get_device_capability(q.device)
    if not supports_glm_sm89_shape(q, kv, do, indices, capability=capability):
        raise ValueError("two-stage sparse MLA requires the exact GLM SM89 CP shape")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("two-stage sparse MLA requires contiguous tensors")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise TypeError("two-stage sparse MLA requires BF16 inputs")
    if indices.dtype != torch.int32:
        raise TypeError("two-stage sparse MLA requires int32 indices")
    if tuple(lse.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS):
        raise ValueError("two-stage sparse MLA requires GLM LSE shape")
    if tuple(delta.shape) != tuple(lse.shape):
        raise ValueError("two-stage sparse MLA requires matching Delta shape")
    if sm_scale is None:
        sm_scale = TOTAL_WIDTH**-0.5

    sequence_kv = int(kv.shape[1])
    memory = two_stage_memory(sequence_kv)
    if max_workspace_bytes > MAX_WORKSPACE_BYTES or memory.total_bytes > max_workspace_bytes:
        raise RuntimeError(
            "two-stage sparse MLA workspace exceeds the 256 MiB qualification bound: "
            f"{memory.total_bytes} > {max_workspace_bytes} bytes"
        )

    route_positions, key_offsets = _sorted_route_positions(indices, sequence_kv)
    output = torch.zeros(
        sequence_kv,
        TOTAL_WIDTH,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (sequence_kv, triton.cdiv(TOTAL_WIDTH, BLOCK_WIDTH))
    _two_stage_dkv_kernel[grid](
        q,
        kv,
        do,
        lse,
        delta,
        route_positions,
        key_offsets,
        output,
        q.stride(1),
        q.stride(2),
        kv.stride(1),
        do.stride(1),
        do.stride(2),
        lse.stride(1),
        lse.stride(2),
        delta.stride(1),
        delta.stride(2),
        output.stride(0),
        float(sm_scale),
        BLOCK_WIDTH=BLOCK_WIDTH,
        num_warps=4,
        num_stages=1,
    )
    if not torch.isfinite(output).all():
        raise RuntimeError("two-stage sparse MLA produced nonfinite dKV")
    return output.view_as(kv).to(dtype=kv.dtype), memory
