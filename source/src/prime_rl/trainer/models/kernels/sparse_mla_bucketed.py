"""Qualification-only SM89 sparse MLA dKV route buckets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prime_rl.trainer.models.kernels.sparse_mla_two_stage import (
    ATTENTION_HEADS,
    MAX_WORKSPACE_BYTES,
    QUERY_TOKENS,
    TOTAL_WIDTH,
    TOPK,
    VALUE_WIDTH,
    supports_glm_sm89_shape,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


ROUTE_SOURCE_BYTES = 4
HISTOGRAM_BYTES = 4
OFFSET_BYTES = 4
CURSOR_BYTES = 4
PREFIX_TEMP_BYTES = 4
OUTPUT_FP32_BYTES = 4
OUTPUT_BF16_BYTES = 2
BLOCK_WIDTH = 1024
BLOCK_SIZE = 256
LOG2_E = 1.4426950408889634
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class BucketedMemory:
    sequence_kv: int
    route_count: int
    route_sources_bytes: int
    histogram_bytes: int
    offsets_bytes: int
    cursor_bytes: int
    prefix_temp_bytes: int
    output_fp32_bytes: int
    output_bf16_bytes: int
    total_bytes: int


def bucketed_memory(
    sequence_kv: int,
    *,
    route_count: int = QUERY_TOKENS * TOPK,
) -> BucketedMemory:
    """Return the full conservative allocation ledger for the candidate."""

    if isinstance(sequence_kv, bool) or not isinstance(sequence_kv, int) or sequence_kv < 1:
        raise ValueError("bucketed route sequence length must be a positive integer")
    if isinstance(route_count, bool) or not isinstance(route_count, int) or route_count < 0:
        raise ValueError("bucketed route count must be a non-negative integer")
    if route_count > INT32_MAX:
        raise ValueError("bucketed route count must fit in int32")
    route_sources_bytes = route_count * ROUTE_SOURCE_BYTES
    histogram_bytes = sequence_kv * HISTOGRAM_BYTES
    offsets_bytes = (sequence_kv + 1) * OFFSET_BYTES
    cursor_bytes = sequence_kv * CURSOR_BYTES
    prefix_temp_bytes = (sequence_kv + 1) * PREFIX_TEMP_BYTES
    output_fp32_bytes = sequence_kv * TOTAL_WIDTH * OUTPUT_FP32_BYTES
    output_bf16_bytes = sequence_kv * TOTAL_WIDTH * OUTPUT_BF16_BYTES
    total_bytes = (
        route_sources_bytes
        + histogram_bytes
        + offsets_bytes
        + cursor_bytes
        + prefix_temp_bytes
        + output_fp32_bytes
        + output_bf16_bytes
    )
    return BucketedMemory(
        sequence_kv=sequence_kv,
        route_count=route_count,
        route_sources_bytes=route_sources_bytes,
        histogram_bytes=histogram_bytes,
        offsets_bytes=offsets_bytes,
        cursor_bytes=cursor_bytes,
        prefix_temp_bytes=prefix_temp_bytes,
        output_fp32_bytes=output_fp32_bytes,
        output_bf16_bytes=output_bf16_bytes,
        total_bytes=total_bytes,
    )


if triton is not None:

    @triton.jit
    def _count_route_keys(
        indices,
        counts,
        route_count,
        sequence_kv,
        BLOCK: tl.constexpr,
    ):
        route_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = route_index < route_count
        key = tl.load(indices + route_index, mask=active, other=0)
        valid = active & (key >= 0) & (key < sequence_kv - 1)
        safe_key = tl.where(valid, key, 0)
        tl.atomic_add(counts + safe_key, 1, mask=valid)


    @triton.jit
    def _scatter_route_sources(
        indices,
        cursors,
        route_sources,
        route_count,
        sequence_kv,
        BLOCK: tl.constexpr,
    ):
        source_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = source_index < route_count
        key = tl.load(indices + source_index, mask=active, other=0)
        valid = active & (key >= 0) & (key < sequence_kv - 1)
        safe_key = tl.where(valid, key, 0)
        destination = tl.atomic_add(cursors + safe_key, 1, mask=valid)
        tl.store(route_sources + destination, source_index, mask=valid)


    @triton.jit
    def _bucketed_dkv_kernel(
        q,
        kv,
        do,
        lse,
        delta,
        route_sources,
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
        TOPK: tl.constexpr,
        ATTENTION_HEADS: tl.constexpr,
        VALUE_WIDTH: tl.constexpr,
        TOTAL_WIDTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        LOG2_E: tl.constexpr,
    ):
        key_id = tl.program_id(0)
        width = tl.arange(0, BLOCK_WIDTH)
        output_mask = width < TOTAL_WIDTH
        route_start = tl.load(key_offsets + key_id)
        route_end = tl.load(key_offsets + key_id + 1)
        accumulator = tl.zeros((BLOCK_WIDTH,), dtype=tl.float32)

        for head in tl.range(0, ATTENTION_HEADS):
            head_partial = tl.zeros((BLOCK_WIDTH,), dtype=tl.float32)
            for route_index in tl.range(route_start, route_end):
                source_index = tl.load(route_sources + route_index)
                query_id = source_index // TOPK
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

                probability = probability.to(tl.bfloat16).to(tl.float32)
                dp = dp.to(tl.bfloat16).to(tl.float32)
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
    _count_route_keys = None
    _scatter_route_sources = None
    _bucketed_dkv_kernel = None


def _build_key_buckets(
    indices: torch.Tensor,
    sequence_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or _count_route_keys is None or _scatter_route_sources is None:
        raise RuntimeError("bucketed sparse MLA requires Triton")
    route_count = indices.numel()
    if route_count > INT32_MAX:
        raise ValueError("bucketed route count must fit in int32")
    counts = torch.zeros(sequence_kv, dtype=torch.int32, device=indices.device)
    grid = (triton.cdiv(route_count, BLOCK_SIZE),)
    _count_route_keys[grid](
        indices,
        counts,
        route_count,
        sequence_kv,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    offsets = torch.empty(sequence_kv + 1, dtype=torch.int32, device=indices.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    cursors = offsets[:-1].clone()
    route_sources = torch.empty(route_count, dtype=torch.int32, device=indices.device)
    _scatter_route_sources[grid](
        indices,
        cursors,
        route_sources,
        route_count,
        sequence_kv,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    del counts, cursors
    return route_sources, offsets


def sparse_mla_dkv_bucketed(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    *,
    sm_scale: float | None = None,
    max_workspace_bytes: int = MAX_WORKSPACE_BYTES,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, BucketedMemory]:
    """Compute SM89 sparse MLA dKV with histogram and scatter routes."""

    if (
        isinstance(max_workspace_bytes, bool)
        or not isinstance(max_workspace_bytes, int)
        or max_workspace_bytes <= 0
    ):
        raise ValueError("bucketed sparse MLA workspace limit must be a positive integer")
    tensors = (q, kv, do, indices, lse, delta)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("bucketed sparse MLA requires CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("bucketed sparse MLA requires same-device tensors")
    if triton is None or _bucketed_dkv_kernel is None:
        raise RuntimeError("bucketed sparse MLA requires Triton")
    capability = torch.cuda.get_device_capability(q.device)
    if not supports_glm_sm89_shape(q, kv, do, indices, capability=capability):
        raise ValueError("bucketed sparse MLA requires the exact GLM SM89 CP shape")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("bucketed sparse MLA requires contiguous tensors")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise TypeError("bucketed sparse MLA requires BF16 inputs")
    if indices.dtype != torch.int32:
        raise TypeError("bucketed sparse MLA requires int32 indices")
    if lse.dtype != torch.float32 or delta.dtype != torch.float32:
        raise TypeError("bucketed sparse MLA requires FP32 LSE and Delta")
    if tuple(lse.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS):
        raise ValueError("bucketed sparse MLA requires GLM LSE shape")
    if tuple(delta.shape) != tuple(lse.shape):
        raise ValueError("bucketed sparse MLA requires matching Delta shape")
    sequence_kv = int(kv.shape[1])
    if output is None:
        output_buffer = None
    else:
        if not output.is_cuda or output.device != q.device:
            raise ValueError("bucketed sparse MLA output must share the input device")
        if output.dtype != torch.float32:
            raise TypeError("bucketed sparse MLA output must be FP32")
        if not output.is_contiguous():
            raise ValueError("bucketed sparse MLA output must be contiguous")
        if tuple(output.shape) == tuple(kv.shape):
            output_buffer = output.view(sequence_kv, TOTAL_WIDTH)
        elif tuple(output.shape) == (sequence_kv, TOTAL_WIDTH):
            output_buffer = output
        else:
            raise ValueError("bucketed sparse MLA output has the wrong shape")
    if sm_scale is None:
        sm_scale = TOTAL_WIDTH**-0.5

    memory = bucketed_memory(sequence_kv, route_count=indices.numel())
    if max_workspace_bytes > MAX_WORKSPACE_BYTES or memory.total_bytes > max_workspace_bytes:
        raise RuntimeError(
            "bucketed sparse MLA workspace exceeds the 256 MiB qualification bound: "
            f"{memory.total_bytes} > {max_workspace_bytes} bytes"
        )

    route_sources, key_offsets = _build_key_buckets(indices, sequence_kv)
    if output_buffer is None:
        output_buffer = torch.empty(
            sequence_kv,
            TOTAL_WIDTH,
            dtype=torch.float32,
            device=q.device,
        )
    output_buffer.zero_()
    _bucketed_dkv_kernel[(sequence_kv,)](
        q,
        kv,
        do,
        lse,
        delta,
        route_sources,
        key_offsets,
        output_buffer,
        q.stride(1),
        q.stride(2),
        kv.stride(1),
        do.stride(1),
        do.stride(2),
        lse.stride(1),
        lse.stride(2),
        delta.stride(1),
        delta.stride(2),
        output_buffer.stride(0),
        float(sm_scale),
        TOPK=TOPK,
        ATTENTION_HEADS=ATTENTION_HEADS,
        VALUE_WIDTH=VALUE_WIDTH,
        TOTAL_WIDTH=TOTAL_WIDTH,
        BLOCK_WIDTH=BLOCK_WIDTH,
        LOG2_E=LOG2_E,
        num_warps=4,
        num_stages=1,
    )
    del route_sources, key_offsets
    return output_buffer.view_as(kv).to(dtype=kv.dtype), memory
