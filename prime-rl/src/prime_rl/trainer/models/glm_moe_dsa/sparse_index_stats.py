"""Opt-in sparse-index distribution capture for qualification runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


_OUTPUT = os.environ.get("PRIME_SPARSE_INDEX_STATS_PATH")
_MAX_PER_LAYER = int(os.environ.get("PRIME_SPARSE_INDEX_STATS_MAX_PER_LAYER", "1"))
_RANK = int(os.environ.get("RANK", "0"))
_COUNTS: dict[int, int] = {}


def summarize_sparse_indices(indices: torch.Tensor, sequence_kv: int) -> dict[str, Any]:
    if not isinstance(indices, torch.Tensor):
        raise TypeError("sparse indices must be a tensor")
    if indices.numel() == 0:
        raise ValueError("sparse indices cannot be empty")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("sparse indices must use int32 or int64")
    if not isinstance(sequence_kv, int) or sequence_kv < 2:
        raise ValueError("sequence_kv must be at least 2")

    values = indices.detach().reshape(-1).to(dtype=torch.int64)
    sentinel = values == sequence_kv - 1
    valid = (values >= 0) & (values < sequence_kv - 1)
    valid_values = values[valid]
    counts = torch.bincount(valid_values, minlength=sequence_kv - 1)
    used = counts[counts > 0].to(dtype=torch.float32).cpu()
    sentinel_count = int(sentinel.sum().item())
    invalid = int((~valid & ~sentinel).sum().item())
    entries = int(values.numel())
    if used.numel():
        quantiles = torch.quantile(used, torch.tensor([0.5, 0.95, 0.99]))
        route_values, route_counts = torch.unique(used.to(dtype=torch.int64), return_counts=True)
        route_histogram = {
            str(int(value)): int(count)
            for value, count in zip(route_values.tolist(), route_counts.tolist(), strict=True)
        }
        routes = {
            "min": float(used.min().item()),
            "mean": float(used.mean().item()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(used.max().item()),
        }
    else:
        routes = {name: 0.0 for name in ("min", "mean", "p50", "p95", "p99", "max")}
        route_histogram = {}
    return {
        "sequence_kv": sequence_kv,
        "entries": entries,
        "valid_entries": entries - invalid - sentinel_count,
        "sentinel_entries": sentinel_count,
        "invalid_entries": invalid,
        "invalid_fraction": invalid / entries,
        "invalid_percent": 100.0 * invalid / entries,
        "sentinel_fraction": sentinel_count / entries,
        "sentinel_percent": 100.0 * sentinel_count / entries,
        "unique_keys": int(used.numel()),
        "routes_per_key": routes,
        "routes_per_key_histogram": route_histogram,
    }


def record_sparse_index_tensor(
    indices: torch.Tensor,
    *,
    sequence_kv: int,
    layer_index: int,
    cp_rank: int,
    cp_world_size: int,
) -> None:
    if _OUTPUT is None or _MAX_PER_LAYER <= 0:
        return
    seen = _COUNTS.get(layer_index, 0)
    if seen >= _MAX_PER_LAYER:
        return
    _COUNTS[layer_index] = seen + 1
    record = summarize_sparse_indices(indices, sequence_kv)
    record.update(
        {
            "layer_index": layer_index,
            "rank": _RANK,
            "cp_rank": cp_rank,
            "cp_world_size": cp_world_size,
            "sample_index": seen,
        }
    )
    output = Path(f"{_OUTPUT}.rank{_RANK}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
