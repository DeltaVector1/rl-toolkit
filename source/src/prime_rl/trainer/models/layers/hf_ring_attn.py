"""Hugging Face ring-attention adapter with exact sliding-window semantics."""

from __future__ import annotations

from functools import wraps

import torch.distributed as dist
from ring_flash_attn import substitute_hf_flash_attn as _upstream_substitute
from ring_flash_attn.adapters import hf_adapter


def _transformers_window_for_upstream(value: int | None) -> int | None:
    """Convert Transformers' total causal window to the old adapter boundary.

    Current Transformers passes a total window of ``N`` and calls FlashAttention
    with inclusive boundaries ``(N - 1, N - 1)``.  The installed ring adapter
    passes ``(N, N)``.  Give that adapter ``N - 1`` so it emits the exact
    current-Transformers boundary.
    """

    return None if value is None else max(0, value - 1)


def _exact_window_wrapper(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if kwargs.get("sliding_window") is not None:
            kwargs["sliding_window"] = _transformers_window_for_upstream(
                kwargs["sliding_window"]
            )
        elif len(args) > 9 and args[9] is not None:
            args = list(args)
            args[9] = _transformers_window_for_upstream(args[9])
            args = tuple(args)
        return function(*args, **kwargs)

    return wrapped


def substitute_hf_flash_attn(
    process_group: dist.ProcessGroup, heads_k_stride: int
) -> None:
    """Install ring attention with the current HF sliding-window boundary."""

    original_create = hf_adapter.create_ring_flash_attention_forward

    def create_exact(group, stride):
        return [
            _exact_window_wrapper(function)
            for function in original_create(group, stride)
        ]

    hf_adapter.create_ring_flash_attention_forward = create_exact
    try:
        _upstream_substitute(process_group, heads_k_stride)
    finally:
        hf_adapter.create_ring_flash_attention_forward = original_create
