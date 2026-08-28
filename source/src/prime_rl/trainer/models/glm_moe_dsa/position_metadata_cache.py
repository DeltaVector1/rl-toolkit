"""Bounded cache for immutable GLM position metadata."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

import torch


DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _tensor_revision(value: torch.Tensor) -> int:
    """Return the in-place mutation counter used by PyTorch tensors."""

    return int(getattr(value, "_version", 0))


def _value_signature(value):
    if isinstance(value, torch.Tensor):
        if value.numel() <= 1024:
            return (
                "tensor_values",
                str(value.device),
                str(value.dtype),
                tuple(value.shape),
                tuple(value.detach().cpu().reshape(-1).tolist()),
            )
        return (
            "tensor",
            str(value.device),
            str(value.dtype),
            tuple(value.shape),
            _tensor_revision(value),
            value.data_ptr(),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_value_signature(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _value_signature(item)) for key, item in value.items()))
    return (type(value).__name__, repr(value))


def rotary_revision(rotary_embedding, explicit_revision: int = 0) -> tuple:
    """Describe mutable rotary state that can change the returned values."""

    state = [
        ("module", id(rotary_embedding)),
        ("explicit", int(explicit_revision)),
    ]
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(rotary_embedding, name, None)
        if isinstance(value, torch.Tensor):
            state.append((name, _value_signature(value)))
    for name in ("rope_type", "max_seq_len_cached", "original_max_seq_len", "rope_init_fn"):
        value = getattr(rotary_embedding, name, None)
        state.append((name, _value_signature(value)))
    state.append(
        (
            "attention_scaling",
            _value_signature(getattr(rotary_embedding, "attention_scaling", None)),
        )
    )
    config = getattr(rotary_embedding, "config", None)
    state.append(("config", id(config), repr(getattr(config, "__dict__", config))))
    return tuple(state)


@dataclass(frozen=True)
class PositionMetadata:
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ks: torch.Tensor
    ke: torch.Tensor


@dataclass(frozen=True)
class _CacheKey:
    group_id: int | None
    cp_rank: int
    cp_world_size: int
    position_shape: tuple[int, ...]
    position_dtype: str
    position_device: str
    position_revision: int
    hidden_shape: tuple[int, ...]
    hidden_dtype: str
    hidden_device: str
    rotary_revision: tuple


@dataclass
class _Entry:
    key: _CacheKey
    position_ids: torch.Tensor
    metadata: PositionMetadata
    bytes: int


class PositionMetadataCache:
    """A small exact-value cache.  A zero capacity disables it."""

    def __init__(self, max_entries: int = 0, max_bytes: int = DEFAULT_MAX_BYTES):
        if max_entries < 0:
            raise ValueError(f"max_entries must be >= 0, got {max_entries}")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: list[_Entry] = []
        self._bytes = 0
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    @staticmethod
    def make_key(
        *,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        cp_group,
        cp_rank: int,
        cp_world_size: int,
        rotary_state: tuple,
        source_position_ids: torch.Tensor | None = None,
    ) -> _CacheKey:
        return _CacheKey(
            group_id=id(cp_group) if cp_group is not None else None,
            cp_rank=int(cp_rank),
            cp_world_size=int(cp_world_size),
            position_shape=tuple(position_ids.shape),
            position_dtype=str(position_ids.dtype),
            position_device=str(position_ids.device),
            position_revision=_tensor_revision(
                source_position_ids if source_position_ids is not None else position_ids
            ),
            hidden_shape=tuple(hidden_states.shape),
            hidden_dtype=str(hidden_states.dtype),
            hidden_device=str(hidden_states.device),
            rotary_revision=rotary_state,
        )

    def get(self, key: _CacheKey, position_ids: torch.Tensor) -> PositionMetadata | None:
        if not self.enabled:
            return None
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry.key != key:
                    continue
                # The snapshot catches changes on another CP rank.  The
                # revision in the key catches normal in-place changes here.
                if not torch.equal(entry.position_ids, position_ids):
                    continue
                if index:
                    self._entries.append(self._entries.pop(index))
                return entry.metadata
        return None

    def put(
        self,
        key: _CacheKey,
        position_ids: torch.Tensor,
        metadata: PositionMetadata,
    ) -> None:
        if not self.enabled:
            return
        snapshot = position_ids.detach().clone()
        stored = PositionMetadata(
            position_embeddings=tuple(
                value.detach().contiguous().clone()
                for value in metadata.position_embeddings
            ),
            ks=metadata.ks.detach().contiguous().clone(),
            ke=metadata.ke.detach().contiguous().clone(),
        )
        entry_bytes = sum(
            _tensor_bytes(value)
            for value in (*stored.position_embeddings, stored.ks, stored.ke, snapshot)
        )
        if entry_bytes > self.max_bytes:
            return
        with self._lock:
            while self._entries and (
                len(self._entries) >= self.max_entries or self._bytes + entry_bytes > self.max_bytes
            ):
                evicted = self._entries.pop(0)
                self._bytes -= evicted.bytes
            self._entries.append(_Entry(key, snapshot, stored, entry_bytes))
            self._bytes += entry_bytes

    def get_or_compute(
        self,
        *,
        key: _CacheKey,
        position_ids: torch.Tensor,
        compute: Callable[[], PositionMetadata],
    ) -> PositionMetadata:
        cached = self.get(key, position_ids)
        if cached is not None:
            return cached
        metadata = compute()
        self.put(key, position_ids, metadata)
        return metadata


__all__ = [
    "DEFAULT_MAX_BYTES",
    "PositionMetadata",
    "PositionMetadataCache",
    "rotary_revision",
]
