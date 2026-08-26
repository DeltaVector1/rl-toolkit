"""Atomic-free key-major sparse MLA dKV backward candidate.

This module is deliberately independent of TileLang.  The production
backward keeps the TileLang q-major dQ kernel and uses this module only when
``PRIME_SPARSE_BWD_KEY_MAJOR_CSR=1``.  The reverse index is built once per
backward.  Entries are stable-sorted by ``(batch, group, key tile, key)``;
the second key component is only a within-tile ordering and does not remove
duplicates.

The reducer writes one unique partial row at a time and uses fixed-order
segment reductions.  It does not call scatter_add/index_add/atomic_add.
Temporary dKV partial storage is bounded by ``max_partial_bytes``.  Reverse
index storage is accounted for separately because its size depends on the
number of valid routing entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_KEY_TILE = 16
DEFAULT_QUERY_CHUNK = 256
# 512 entries x 16 heads x 576 columns in FP32 is about 18 MiB for the
# largest live contribution tensor.  Together with the 144 MiB production
# partial at query_chunk=256 this stays below the 288 MiB partial budget.
DEFAULT_ENTRY_CHUNK = 512
DEFAULT_MAX_PARTIAL_BYTES = 288 * 1024 * 1024
DEFAULT_MAX_INDEX_BYTES = 192 * 1024 * 1024
DEFAULT_MAX_TOTAL_TEMP_BYTES = 512 * 1024 * 1024
_LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class ReverseCSR:
    """Stable key-major reverse index for one sparse MLA backward.

    ``positions`` contains flattened ``[B, S, G, topk]`` positions.  The
    original order is retained for equal keys, so duplicate routes produce
    duplicate contributions.  ``tile_ptr`` indexes rows by
    ``[batch, group, key_tile]`` and is useful for bounded query-chunk work.
    """

    positions: torch.Tensor
    keys: torch.Tensor
    key_ids: torch.Tensor
    tile_ptr: torch.Tensor
    batch: int
    sequence: int
    groups: int
    topk: int
    sequence_kv: int
    key_tile: int

    @property
    def valid_entries(self) -> int:
        return int(self.positions.numel())

    @property
    def num_key_tiles(self) -> int:
        return (self.sequence_kv + self.key_tile - 1) // self.key_tile

    @property
    def index_bytes(self) -> int:
        """Bytes held by the reverse index tensors, excluding sort scratch."""

        return sum(
            int(t.numel()) * t.element_size()
            for t in (self.positions, self.keys, self.key_ids, self.tile_ptr)
        )


def _require_indices(indices: torch.Tensor) -> tuple[int, int, int, int]:
    if indices.ndim != 4:
        raise ValueError("sparse MLA indices must have shape [B, S, G, topk]")
    if indices.dtype != torch.int32:
        raise TypeError("sparse MLA indices must be int32")
    if not indices.is_contiguous():
        raise ValueError("sparse MLA indices must be contiguous")
    return tuple(int(v) for v in indices.shape)  # type: ignore[return-value]


def estimate_partial_bytes(
    batch: int,
    sequence: int,
    groups: int,
    sequence_kv: int,
    width: int,
    *,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    key_tile: int = DEFAULT_KEY_TILE,
) -> int:
    """Return the FP32 partial-buffer size for the configured shape."""

    if query_chunk <= 0 or key_tile <= 0:
        raise ValueError("query_chunk and key_tile must be positive")
    query_chunks = (sequence + query_chunk - 1) // query_chunk
    key_tiles = (sequence_kv + key_tile - 1) // key_tile
    return batch * groups * query_chunks * key_tiles * key_tile * width * 4


def build_reverse_csr(
    indices: torch.Tensor,
    sequence_kv: int,
    *,
    key_tile: int = DEFAULT_KEY_TILE,
    max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
) -> ReverseCSR:
    """Build a stable key-major CSR index without deduplicating routes.

    Only indices in ``[0, sequence_kv - 1)`` are valid.  In particular,
    ``sequence_kv - 1`` is the zero sentinel and is discarded.  The validity
    mask is applied before indexing any KV tensor, which also makes negative
    and oversized malformed entries safe.
    """

    batch, sequence, groups, topk = _require_indices(indices)
    if sequence_kv < 1:
        raise ValueError("sequence_kv must include the final zero sentinel")
    if key_tile <= 0:
        raise ValueError("key_tile must be positive")
    if max_index_bytes <= 0:
        raise ValueError("max_index_bytes must be positive")

    flat = indices.reshape(-1)
    valid = (flat >= 0) & (flat < sequence_kv - 1)
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    # A 32-bit position is enough for the production route tensor and halves
    # reverse-index storage.  Keep int64 for unusually large synthetic cases.
    if flat.numel() <= torch.iinfo(torch.int32).max:
        positions = positions.to(torch.int32)
    keys = flat.index_select(0, positions.to(torch.int64))
    num_tiles = (sequence_kv + key_tile - 1) // key_tile

    # Exact key IDs sort by batch/group/key.  This is key-tile-major with a
    # stable within-tile key order and therefore gives fixed segment order.
    positions_i64 = positions.to(torch.int64)
    route_width = groups * topk
    batch_ids = positions_i64 // (sequence * route_width)
    sequence_ids = (positions_i64 // route_width) % sequence
    group_ids = (positions_i64 % route_width) // topk
    key_ids = ((batch_ids * groups + group_ids) * sequence_kv + keys.to(torch.int64)).to(torch.int64)
    tile_ids = ((batch_ids * groups + group_ids) * num_tiles + keys.to(torch.int64) // key_tile)
    order = torch.argsort(key_ids, stable=True)
    positions = positions.index_select(0, order)
    keys = keys.index_select(0, order)
    key_ids = key_ids.index_select(0, order)

    tile_ids = tile_ids.index_select(0, order)
    rows = batch * groups * num_tiles
    tile_ptr = torch.zeros(rows + 1, dtype=torch.int64, device=indices.device)
    if tile_ids.numel():
        counts = torch.bincount(tile_ids, minlength=rows)
        tile_ptr[1:] = torch.cumsum(counts, dim=0)

    # Do not keep sort scratch live while the dKV partial buffer is allocated.
    # The persistent reverse index below is accounted for by index_bytes.
    del flat, valid, positions_i64, batch_ids, sequence_ids, group_ids, order, tile_ids
    if "counts" in locals():
        del counts

    result = ReverseCSR(
        positions=positions,
        keys=keys,
        key_ids=key_ids,
        tile_ptr=tile_ptr,
        batch=batch,
        sequence=sequence,
        groups=groups,
        topk=topk,
        sequence_kv=sequence_kv,
        key_tile=key_tile,
    )
    if result.index_bytes > max_index_bytes:
        raise RuntimeError(
            "sparse MLA CSR reverse index exceeds the configured budget: "
            f"{result.index_bytes} > {max_index_bytes} bytes"
        )
    return result


def _segment_sum(values: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce sorted rows by equal IDs in a fixed order.

    ``ids`` is sorted and non-empty.  PyTorch 2.11 provides segment_reduce on
    CUDA.  The loop is a small compatibility fallback for CPU tests and older
    local wheels; it has the same left-to-right reduction order.
    """

    if ids.numel() == 0:
        return values[:0], ids
    starts = torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)
    starts[1:] = ids[1:] != ids[:-1]
    first = torch.nonzero(starts, as_tuple=False).flatten()
    ends = torch.cat((first[1:], ids.new_tensor([ids.numel()])))
    lengths = ends - first
    segment_reduce = getattr(torch, "segment_reduce", None)
    if segment_reduce is not None:
        reduced = segment_reduce(values, "sum", lengths=lengths)
    else:
        reduced = torch.stack([values[int(s) : int(e)].sum(dim=0) for s, e in zip(first, ends)])
    return reduced, ids.index_select(0, first)


def _entry_contribution(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    *,
    sequence: int,
    groups: int,
    topk: int,
    head_start: int,
    head_count: int,
    sm_scale: float,
) -> torch.Tensor:
    """Compute FP32 dKV rows for sorted valid routes in one bounded chunk."""

    route_width = groups * topk
    positions_i64 = positions.to(torch.int64)
    batch_ids = positions_i64 // (sequence * route_width)
    sequence_ids = (positions_i64 // route_width) % sequence
    group_ids = (positions_i64 % route_width) // topk
    head_ids = torch.arange(
        head_start, head_start + head_count, device=q.device, dtype=torch.int64
    )[None, :].expand(positions_i64.numel(), -1)

    q_rows = q[batch_ids[:, None], sequence_ids[:, None], head_ids, :].float()
    kv_rows = kv[batch_ids, keys.to(torch.int64), group_ids, :].float()
    do_rows = do[batch_ids[:, None], sequence_ids[:, None], head_ids, :].float()
    lse_rows = lse[batch_ids[:, None], sequence_ids[:, None], head_ids].float()
    delta_rows = delta[batch_ids[:, None], sequence_ids[:, None], head_ids].float()

    q_score = (q_rows * kv_rows[:, None, :]).sum(dim=-1)
    finite_lse = torch.isfinite(lse_rows)
    finite_score = torch.isfinite(q_score)
    score_exp = torch.exp2(q_score * (sm_scale * _LOG2_E) - lse_rows)
    probs = torch.where(finite_lse & finite_score, score_exp, torch.zeros_like(score_exp))
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))

    d_dim = do.shape[-1]
    dp_score = (do_rows * kv_rows[:, None, :d_dim]).sum(dim=-1)
    dp = probs * (dp_score - delta_rows) * sm_scale
    dp = torch.where(torch.isfinite(dp), dp, torch.zeros_like(dp))

    dkv_d = (dp[..., None] * q_rows[..., :d_dim] + probs[..., None] * do_rows).sum(dim=1)
    dkv_tail = (dp[..., None] * q_rows[..., d_dim:]).sum(dim=1)
    contribution = torch.cat((dkv_d, dkv_tail), dim=-1)
    return torch.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)


def sparse_mla_dkv_csr(
    q: torch.Tensor,
    kv: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    *,
    sm_scale: float | None = None,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    key_tile: int = DEFAULT_KEY_TILE,
    entry_chunk: int = DEFAULT_ENTRY_CHUNK,
    max_partial_bytes: int = DEFAULT_MAX_PARTIAL_BYTES,
    max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
    max_total_temp_bytes: int = DEFAULT_MAX_TOTAL_TEMP_BYTES,
) -> tuple[torch.Tensor, ReverseCSR, int]:
    """Return BF16 dKV from a bounded, atomic-free key-major reduction.

    The third return value is the allocated FP32 partial-buffer size.  It is
    exposed so benchmark and qualification code can report temporary memory.
    """

    if q.ndim != 4 or kv.ndim != 4 or do.ndim != 4:
        raise ValueError("q, kv, and do must be rank-4 tensors")
    batch, sequence, heads, width = (int(v) for v in q.shape)
    kv_batch, sequence_kv, groups, kv_width = (int(v) for v in kv.shape)
    if kv_batch != batch or kv_width != width:
        raise ValueError("q and kv shapes are inconsistent")
    if heads % groups != 0:
        raise ValueError("attention heads must divide evenly across KV groups")
    if tuple(do.shape[:3]) != (batch, sequence, heads) or int(do.shape[-1]) >= width:
        raise ValueError("do shape is inconsistent with q")
    if tuple(lse.shape) != (batch, sequence, heads) or tuple(delta.shape) != tuple(lse.shape):
        raise ValueError("lse and delta must have shape [B, S, H]")
    if query_chunk <= 0 or key_tile <= 0 or entry_chunk <= 0:
        raise ValueError("query_chunk, key_tile, and entry_chunk must be positive")
    if max_partial_bytes <= 0 or max_index_bytes <= 0 or max_total_temp_bytes <= 0:
        raise ValueError("temporary-memory budgets must be positive")
    if sm_scale is None:
        sm_scale = width ** -0.5

    partial_bytes = estimate_partial_bytes(
        batch,
        sequence,
        groups,
        sequence_kv,
        width,
        query_chunk=query_chunk,
        key_tile=key_tile,
    )
    if partial_bytes > max_partial_bytes:
        raise RuntimeError(
            "sparse MLA CSR partial dKV exceeds the configured budget: "
            f"{partial_bytes} > {max_partial_bytes} bytes"
        )
    reverse = build_reverse_csr(
        indices,
        sequence_kv,
        key_tile=key_tile,
        max_index_bytes=max_index_bytes,
    )
    # Conservative upper bound for the live per-entry tensors.  The q/KV
    # gathers, score, probability, and dP tensors are each bounded by this
    # factor; this gate covers their overlap with the persistent partial and
    # reverse-index buffers.
    head_block = min(16, heads // groups)
    contribution_bytes = entry_chunk * head_block * width * 4 * 3
    total_temp_bytes = partial_bytes + reverse.index_bytes + contribution_bytes
    if total_temp_bytes > max_total_temp_bytes:
        raise RuntimeError(
            "sparse MLA CSR temporary memory exceeds the configured budget: "
            f"{total_temp_bytes} > {max_total_temp_bytes} bytes"
        )
    partial = torch.zeros(
        batch,
        groups,
        (sequence + query_chunk - 1) // query_chunk,
        reverse.num_key_tiles,
        key_tile,
        width,
        dtype=torch.float32,
        device=q.device,
    )
    head_count = heads // groups
    tile_rows = batch * groups * reverse.num_key_tiles

    # Each (query chunk, tile) has a unique partial owner.  Equal-key rows in
    # a tile are reduced before the unique assignment, so no atomic update is
    # needed even when the route list contains duplicates.
    for chunk_id, query_start in enumerate(range(0, sequence, query_chunk)):
        query_end = min(sequence, query_start + query_chunk)
        for tile_row in range(tile_rows):
            start = int(reverse.tile_ptr[tile_row].item())
            end = int(reverse.tile_ptr[tile_row + 1].item())
            if start == end:
                continue
            tile_positions = reverse.positions[start:end]
            positions_i64 = tile_positions.to(torch.int64)
            route_width = groups * reverse.topk
            query_ids = (positions_i64 // route_width) % sequence
            in_chunk = (query_ids >= query_start) & (query_ids < query_end)
            selected = torch.nonzero(in_chunk, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            tile_positions = tile_positions.index_select(0, selected)
            tile_keys = reverse.keys[start:end].index_select(0, selected)
            tile_key_ids = reverse.key_ids[start:end].index_select(0, selected)
            batch_id = tile_row // (groups * reverse.num_key_tiles)
            group_id = (tile_row // reverse.num_key_tiles) % groups
            tile_id = tile_row % reverse.num_key_tiles
            partial_view = partial[batch_id, group_id, chunk_id, tile_id]

            for entry_start in range(0, int(tile_positions.numel()), entry_chunk):
                entry_end = min(int(tile_positions.numel()), entry_start + entry_chunk)
                positions_chunk = tile_positions[entry_start:entry_end]
                keys_chunk = tile_keys[entry_start:entry_end]
                key_ids_chunk = tile_key_ids[entry_start:entry_end]
                for head_start in range(0, head_count, head_block):
                    contribution = _entry_contribution(
                        q,
                        kv,
                        do,
                        lse,
                        delta,
                        positions_chunk,
                        keys_chunk,
                        sequence=sequence,
                        groups=groups,
                        topk=reverse.topk,
                        head_start=group_id * head_count + head_start,
                        head_count=min(head_block, head_count - head_start),
                        sm_scale=float(sm_scale),
                    )
                    reduced, unique_ids = _segment_sum(contribution, key_ids_chunk)
                    key_offsets = unique_ids % sequence_kv % key_tile
                    # unique_ids are strictly increasing here; indexed
                    # assignment therefore has one writer per destination.
                    partial_view[key_offsets] = partial_view[key_offsets] + reduced

    dkv_flat = torch.zeros(batch * sequence_kv * groups, width, dtype=torch.float32, device=q.device)
    for batch_id in range(batch):
        for group_id in range(groups):
            for tile_id in range(reverse.num_key_tiles):
                key_start = tile_id * key_tile
                key_end = min(sequence_kv, key_start + key_tile)
                if key_start >= key_end:
                    continue
                tile_sum = partial[batch_id, group_id, :, tile_id, : key_end - key_start].sum(dim=0)
                flat_start = (batch_id * sequence_kv + key_start) * groups + group_id
                key_ids = torch.arange(
                    flat_start,
                    flat_start + (key_end - key_start) * groups,
                    groups,
                    dtype=torch.int64,
                    device=q.device,
                )
                dkv_flat.index_copy_(0, key_ids, tile_sum)

    dkv = dkv_flat.view(batch, sequence_kv, groups, width).to(dtype=kv.dtype)
    return dkv, reverse, partial_bytes
