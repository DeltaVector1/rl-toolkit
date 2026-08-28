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
BLOCK_WIDTH = 1024
ROUTE_CHUNK_ENTRIES = 1 << 20
QUERY_ID_BITS = 12
QUERY_ID_MASK = QUERY_TOKENS - 1
PACKED_ROUTE_BYTES = 4
ROUTE_ORDER_BYTES = 8
# The sort implementation can use temporary storage in addition to its
# output index.  Keep this ledger conservative so the qualification bound is
# checked before a kernel launch.
ARGSORT_SCRATCH_BYTES = 64
ROUTE_FILTER_BYTES_PER_ENTRY = 4 + 1 + 4 + 4 + 4
ROUTE_SORT_BYTES_PER_ENTRY = PACKED_ROUTE_BYTES + ROUTE_ORDER_BYTES + ARGSORT_SCRATCH_BYTES
# Kept as a compatibility name for qualification reports.
ROUTE_BYTES_PER_ENTRY = ROUTE_SORT_BYTES_PER_ENTRY
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class TwoStageMemory:
    sequence_kv: int
    route_chunk_entries: int
    route_filter_bytes: int
    route_sort_bytes: int
    offset_bytes: int
    reduction_bytes: int
    output_pair_bytes: int
    total_bytes: int
    chunk_count: int


def two_stage_memory(
    sequence_kv: int,
    *,
    route_count: int = QUERY_TOKENS * TOPK,
) -> TwoStageMemory:
    """Return the conservative peak allocation bound in bytes.

    Routes are sorted in bounded chunks.  The output is FP32 while chunks are
    reduced, then one BF16 output is made after all temporary route storage is
    released.  No full route-sized sort buffer is included or needed.
    """

    if sequence_kv < 1 or route_count < 0:
        raise ValueError("two-stage memory dimensions must be non-negative")
    chunk_entries = min(route_count, ROUTE_CHUNK_ENTRIES)
    route_filter_bytes = chunk_entries * ROUTE_FILTER_BYTES_PER_ENTRY
    route_sort_bytes = chunk_entries * ROUTE_SORT_BYTES_PER_ENTRY
    offset_bytes = (sequence_kv + 1) * 8 + sequence_kv * 8
    fp32_output_bytes = sequence_kv * TOTAL_WIDTH * 4
    output_pair_bytes = sequence_kv * TOTAL_WIDTH * (4 + 2)
    reduction_bytes = (
        chunk_entries * (PACKED_ROUTE_BYTES + ROUTE_ORDER_BYTES)
        + offset_bytes
        + fp32_output_bytes
    )
    filter_peak = route_filter_bytes + PACKED_ROUTE_BYTES * chunk_entries + fp32_output_bytes
    sort_peak = route_sort_bytes + offset_bytes + fp32_output_bytes
    return TwoStageMemory(
        sequence_kv=sequence_kv,
        route_chunk_entries=chunk_entries,
        route_filter_bytes=route_filter_bytes,
        route_sort_bytes=route_sort_bytes,
        offset_bytes=offset_bytes,
        reduction_bytes=reduction_bytes,
        output_pair_bytes=output_pair_bytes,
        total_bytes=max(filter_peak, sort_peak, reduction_bytes, output_pair_bytes),
        chunk_count=(route_count + ROUTE_CHUNK_ENTRIES - 1) // ROUTE_CHUNK_ENTRIES,
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
        route_ids,
        route_order,
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
        ACCUMULATE: tl.constexpr,
        ATTENTION_HEADS: tl.constexpr,
        VALUE_WIDTH: tl.constexpr,
        TOTAL_WIDTH: tl.constexpr,
        QUERY_ID_MASK: tl.constexpr,
        LOG2_E: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
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
                source_index = tl.load(route_order + route_index)
                route_id = tl.load(route_ids + source_index)
                query_id = route_id & QUERY_ID_MASK
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

                # Match the reference BF16 shared-memory conversion.
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

        if ACCUMULATE:
            accumulator += tl.load(
                output + key_id * output_stride_s + width,
                mask=output_mask,
                other=0.0,
            )
        tl.store(output + key_id * output_stride_s + width, accumulator, mask=output_mask)

else:
    _two_stage_dkv_kernel = None


def _sorted_route_chunk(
    flat_indices: torch.Tensor,
    start: int,
    end: int,
    sequence_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one bounded key-major route chunk on the GPU."""

    chunk = flat_indices[start:end]
    positions = torch.arange(start, end, device=flat_indices.device, dtype=torch.int32)
    valid = (chunk >= 0) & (chunk < sequence_kv - 1)
    keys = chunk.masked_select(valid)
    query_positions = positions.masked_select(valid)
    del chunk, positions
    route_ids = (keys << QUERY_ID_BITS) | (query_positions // TOPK)
    del keys, query_positions, valid
    route_order = torch.argsort(route_ids, stable=True)
    counts = torch.bincount(route_ids >> QUERY_ID_BITS, minlength=sequence_kv)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=flat_indices.device),
            counts.cumsum(0, dtype=torch.int64),
        )
    )
    del counts
    return route_ids, route_order, offsets


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
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TwoStageMemory]:
    """Compute BF16 dKV with bounded GPU route sorts.

    Each route chunk is stable-sorted by key and query order.  The reduction
    kernel adds each chunk to the FP32 output in chunk order.  This gives a
    deterministic per-key reduction without a full 8M-entry sort.
    """

    if (
        isinstance(max_workspace_bytes, bool)
        or not isinstance(max_workspace_bytes, int)
        or max_workspace_bytes <= 0
    ):
        raise ValueError("two-stage sparse MLA workspace limit must be a positive integer")
    tensors = (q, kv, do, indices, lse, delta)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("two-stage sparse MLA requires CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("two-stage sparse MLA requires same-device tensors")
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
    if lse.dtype != torch.float32 or delta.dtype != torch.float32:
        raise TypeError("two-stage sparse MLA requires FP32 LSE and Delta")
    if tuple(lse.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS):
        raise ValueError("two-stage sparse MLA requires GLM LSE shape")
    if tuple(delta.shape) != tuple(lse.shape):
        raise ValueError("two-stage sparse MLA requires matching Delta shape")
    sequence_kv = int(kv.shape[1])
    if output is None:
        output_buffer = None
    else:
        if not output.is_cuda or output.device != q.device:
            raise ValueError("two-stage sparse MLA output must share the input device")
        if output.dtype != torch.float32:
            raise TypeError("two-stage sparse MLA output must be FP32")
        if not output.is_contiguous():
            raise ValueError("two-stage sparse MLA output must be contiguous")
        if tuple(output.shape) == tuple(kv.shape):
            output_buffer = output.view(sequence_kv, TOTAL_WIDTH)
        elif tuple(output.shape) == (sequence_kv, TOTAL_WIDTH):
            output_buffer = output
        else:
            raise ValueError("two-stage sparse MLA output has the wrong shape")
    if sm_scale is None:
        sm_scale = TOTAL_WIDTH**-0.5

    memory = two_stage_memory(sequence_kv)
    if max_workspace_bytes > MAX_WORKSPACE_BYTES or memory.total_bytes > max_workspace_bytes:
        raise RuntimeError(
            "two-stage sparse MLA workspace exceeds the 256 MiB qualification bound: "
            f"{memory.total_bytes} > {max_workspace_bytes} bytes"
        )

    if output_buffer is None:
        output_buffer = torch.empty(
            sequence_kv,
            TOTAL_WIDTH,
            dtype=torch.float32,
            device=q.device,
        )
    output_buffer.zero_()
    flat_indices = indices.reshape(-1)
    for start in range(0, flat_indices.numel(), ROUTE_CHUNK_ENTRIES):
        end = min(start + ROUTE_CHUNK_ENTRIES, flat_indices.numel())
        route_ids, route_order, key_offsets = _sorted_route_chunk(
            flat_indices,
            start,
            end,
            sequence_kv,
        )
        _two_stage_dkv_kernel[(sequence_kv,)](
            q,
            kv,
            do,
            lse,
            delta,
            route_ids,
            route_order,
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
            ACCUMULATE=start != 0,
            ATTENTION_HEADS=ATTENTION_HEADS,
            VALUE_WIDTH=VALUE_WIDTH,
            TOTAL_WIDTH=TOTAL_WIDTH,
            QUERY_ID_MASK=QUERY_ID_MASK,
            LOG2_E=LOG2_E,
            BLOCK_WIDTH=BLOCK_WIDTH,
            num_warps=4,
            num_stages=1,
        )
        del route_ids, route_order, key_offsets
    del flat_indices
    return output_buffer.view_as(kv).to(dtype=kv.dtype), memory
