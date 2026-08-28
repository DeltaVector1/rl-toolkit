"""Qualification-only SM89 two-stage sparse MLA dKV reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SM89 = (8, 9)
QUERY_TOKENS = 4096
ATTENTION_HEADS = 64
VALUE_WIDTH = 512
ROPE_WIDTH = 64
TOTAL_WIDTH = VALUE_WIDTH + ROPE_WIDTH
GROUPS = 1
TOPK = 2048
QUERY_CHUNK = 32
ENTRY_CHUNK = 64
HEAD_BLOCK = 64
LIVE_CONTRIBUTION_TENSORS = 6
ROUTE_BYTES_PER_ENTRY = 32
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class TwoStageMemory:
    sequence_kv: int
    partial_bytes: int
    route_bytes: int
    contribution_bytes: int
    total_bytes: int


def two_stage_memory(
    sequence_kv: int,
    *,
    query_chunk: int = QUERY_CHUNK,
    entry_chunk: int = ENTRY_CHUNK,
    head_block: int = HEAD_BLOCK,
) -> TwoStageMemory:
    """Return the conservative live FP32 workspace bound."""

    if sequence_kv < 1 or query_chunk <= 0 or entry_chunk <= 0 or head_block <= 0:
        raise ValueError("two-stage memory dimensions must be positive")
    # One reusable partial and one fixed-order accumulator are live.  Route
    # metadata uses int32/int32/int64.  The last term covers gathered inputs,
    # probabilities, and the FP32 contribution for one entry chunk.
    partial_bytes = 2 * sequence_kv * TOTAL_WIDTH * 4
    route_bytes = query_chunk * TOPK * ROUTE_BYTES_PER_ENTRY
    contribution_bytes = (
        entry_chunk * head_block * TOTAL_WIDTH * 4 * LIVE_CONTRIBUTION_TENSORS
    )
    total_bytes = partial_bytes + route_bytes + contribution_bytes
    return TwoStageMemory(
        sequence_kv=sequence_kv,
        partial_bytes=partial_bytes,
        route_bytes=route_bytes,
        contribution_bytes=contribution_bytes,
        total_bytes=total_bytes,
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


def _segment_sum(values: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.numel() == 0:
        return values[:0], keys
    starts = torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    starts[1:] = keys[1:] != keys[:-1]
    first = torch.nonzero(starts, as_tuple=False).flatten()
    ends = torch.cat((first[1:], keys.new_tensor([keys.numel()])))
    lengths = ends - first
    segment_reduce = getattr(torch, "segment_reduce", None)
    if segment_reduce is not None:
        reduced = segment_reduce(values, "sum", lengths=lengths)
    else:
        reduced = torch.stack(
            [values[int(start) : int(end)].sum(dim=0) for start, end in zip(first, ends)]
        )
    return reduced, keys.index_select(0, first)


def _route_contribution(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_start: int,
    head_count: int,
    sm_scale: float,
) -> torch.Tensor:
    query_ids = positions.to(torch.int64) // TOPK
    head_ids = torch.arange(
        head_start,
        head_start + head_count,
        dtype=torch.int64,
        device=q.device,
    )[None, :]
    q_rows = q[0, query_ids[:, None], head_ids, :].float()
    do_rows = do[0, query_ids[:, None], head_ids, :].float()
    kv_rows = kv[0, keys.to(torch.int64), 0, :].float()
    lse_rows = lse[0, query_ids[:, None], head_ids].float()
    delta_rows = delta[0, query_ids[:, None], head_ids].float()

    score = (q_rows * kv_rows[:, None, :]).sum(dim=-1)
    probabilities = torch.exp2(score * (sm_scale * LOG2_E) - lse_rows)
    probabilities = torch.where(
        torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)
    )
    dp_score = (do_rows * kv_rows[:, None, :VALUE_WIDTH]).sum(dim=-1)
    dp = probabilities * (dp_score - delta_rows) * sm_scale
    dp = torch.where(torch.isfinite(dp), dp, torch.zeros_like(dp))
    value = (dp[..., None] * q_rows[..., :VALUE_WIDTH] + probabilities[..., None] * do_rows).sum(dim=1)
    rope = (dp[..., None] * q_rows[..., VALUE_WIDTH:]).sum(dim=1)
    return torch.nan_to_num(torch.cat((value, rope), dim=-1))


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
    """Compute dKV without global scalar atomics.

    Stage one writes one reusable FP32 partial for each query chunk.  Stage
    two adds each partial to the FP32 result in query order.  Stable key sort
    preserves route order for duplicate keys.  The returned tensor is BF16;
    the FP32 comparison tolerance for this prototype is rtol=2e-2,
    atol=2e-3 against the existing TileLang path.
    """

    if not (q.is_cuda and kv.is_cuda and do.is_cuda and indices.is_cuda):
        raise ValueError("two-stage sparse MLA requires CUDA tensors")
    capability = torch.cuda.get_device_capability(q.device)
    if not supports_glm_sm89_shape(q, kv, do, indices, capability=capability):
        raise ValueError("two-stage sparse MLA requires the exact GLM SM89 CP shape")
    if not all(t.is_contiguous() for t in (q, kv, do, indices, lse, delta)):
        raise ValueError("two-stage sparse MLA requires contiguous tensors")
    if tuple(lse.shape) != (1, QUERY_TOKENS, ATTENTION_HEADS):
        raise ValueError("two-stage sparse MLA requires GLM LSE shape")
    if tuple(delta.shape) != tuple(lse.shape):
        raise ValueError("two-stage sparse MLA requires matching Delta shape")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise TypeError("two-stage sparse MLA requires BF16 inputs")
    if indices.dtype != torch.int32:
        raise TypeError("two-stage sparse MLA requires int32 indices")
    if sm_scale is None:
        sm_scale = TOTAL_WIDTH**-0.5

    memory = two_stage_memory(int(kv.shape[1]))
    if memory.total_bytes > max_workspace_bytes or max_workspace_bytes > MAX_WORKSPACE_BYTES:
        raise RuntimeError(
            "two-stage sparse MLA workspace exceeds the 256 MiB qualification bound: "
            f"{memory.total_bytes} > {max_workspace_bytes} bytes"
        )

    sequence_kv = int(kv.shape[1])
    partial = torch.zeros(
        sequence_kv, TOTAL_WIDTH, dtype=torch.float32, device=q.device
    )
    result = torch.zeros_like(partial)
    for query_start in range(0, QUERY_TOKENS, QUERY_CHUNK):
        query_end = min(query_start + QUERY_CHUNK, QUERY_TOKENS)
        partial.zero_()
        route_values = indices[0, query_start:query_end, 0].reshape(-1)
        valid = (route_values >= 0) & (route_values < sequence_kv - 1)
        positions = torch.nonzero(valid, as_tuple=False).flatten().to(torch.int32)
        keys = route_values.index_select(0, positions)
        if positions.numel():
            order = torch.argsort(keys, stable=True)
            positions = positions.index_select(0, order)
            keys = keys.index_select(0, order)
            for entry_start in range(0, int(positions.numel()), ENTRY_CHUNK):
                entry_end = min(int(positions.numel()), entry_start + ENTRY_CHUNK)
                positions_chunk = positions[entry_start:entry_end]
                keys_chunk = keys[entry_start:entry_end]
                for head_start in range(0, ATTENTION_HEADS, HEAD_BLOCK):
                    contribution = _route_contribution(
                        q,
                        kv,
                        do,
                        lse,
                        delta,
                        positions_chunk + query_start * TOPK,
                        keys_chunk,
                        head_start=head_start,
                        head_count=HEAD_BLOCK,
                        sm_scale=float(sm_scale),
                    )
                    reduced, unique_keys = _segment_sum(contribution, keys_chunk)
                    old = partial.index_select(0, unique_keys.to(torch.int64))
                    partial.index_copy_(0, unique_keys.to(torch.int64), old + reduced)
        result.add_(partial)
    return result.view_as(kv).to(dtype=kv.dtype), memory
