"""Opt-in EP8 LoRA-A replica lifecycle.

EP partitions expert-major LoRA-A on axis zero. This helper all-gathers one
stable full table for the current step, then reduce-scatters each full dA as
soon as its layer backward finishes. The optimizer continues to own the
local shard. Full tables and full gradients are never optimizer state.

The core has no Torch import at module load. NumPy arrays and fake collectives
can exercise the state machine on CPU; ``TorchCollective`` uses the existing
EP process group in production.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence


MIB = 1024**2
GIB = 1024**3
DEFAULT_EP_SIZE = 8
DEFAULT_LOCAL_SHARD_BYTES = 24 * MIB
DEFAULT_FULL_REPLICA_BYTES = 192 * MIB
DEFAULT_LAYER_PEAK_BUDGET_BYTES = int(2.2 * GIB)
DEFAULT_TOTAL_PEAK_BUDGET_BYTES = 46 * GIB


class ReplicaManagerError(RuntimeError):
    """Base error for the fail-closed manager."""


class ReplicaManagerDisabledError(ReplicaManagerError):
    """The feature was not explicitly enabled."""


class ReplicaLifecycleError(ReplicaManagerError):
    """An operation was made outside its valid step/layer state."""


class ReplicaValidationError(ReplicaManagerError, ValueError):
    """A shard, table, or gradient has an invalid layout."""


class StaleReplicaError(ReplicaManagerError):
    """A table or gradient belongs to an old policy version."""


class ReplicaMemoryError(ReplicaManagerError, MemoryError):
    """A projected allocation crosses a hard memory limit."""


class ReplicaCollectiveError(ReplicaManagerError):
    """A collective failed and the active step was aborted."""


@dataclass(frozen=True)
class ReplicaMemoryAccounting:
    replica_live_bytes: int
    full_gradient_live_bytes: int
    projected_peak_bytes: int
    peak_projected_bytes: int
    layer_peak_budget_bytes: int
    total_peak_budget_bytes: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "replica_live_bytes": self.replica_live_bytes,
            "full_gradient_live_bytes": self.full_gradient_live_bytes,
            "projected_peak_bytes": self.projected_peak_bytes,
            "peak_projected_bytes": self.peak_projected_bytes,
            "layer_peak_budget_bytes": self.layer_peak_budget_bytes,
            "total_peak_budget_bytes": self.total_peak_budget_bytes,
            "replica_live_mib": self.replica_live_bytes / MIB,
            "projected_peak_gib": self.projected_peak_bytes / GIB,
        }


@dataclass(frozen=True)
class ReplicaCollectiveStats:
    all_gather_calls: int
    reduce_scatter_calls: int
    all_gather_local_bytes: int
    all_gather_full_bytes: int
    all_gather_wire_bytes: int
    reduce_scatter_full_bytes: int
    reduce_scatter_local_bytes: int
    reduce_scatter_wire_bytes: int
    replica_lifetime_steps: int
    replica_peak_bytes: int
    full_gradient_peak_bytes: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


@dataclass
class _Layer:
    owner: Any
    local: Any
    version: Any
    shape: tuple[int, ...]
    local_bytes: int
    replica: Any = None
    replica_version: Any = None
    replica_bytes: int = 0
    full_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GateUpReplica:
    gate: Any
    up: Any
    version: Any


def _local(value: Any) -> Any:
    method = getattr(value, "to_local", None)
    return method() if callable(method) else value


def _detach(value: Any) -> Any:
    method = getattr(value, "detach", None)
    return method() if callable(method) else value


def _shape(value: Any) -> tuple[int, ...]:
    value = _local(value)
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        return (len(value),) + (_shape(value[0]) if value else ())
    raise ReplicaValidationError("tensor must expose shape")


def _nbytes(value: Any) -> int:
    value = _local(value)
    if getattr(value, "nbytes", None) is not None:
        return int(value.nbytes)
    numel = getattr(value, "numel", None)
    size = getattr(value, "element_size", None)
    if callable(numel) and callable(size):
        return int(numel()) * int(size())
    raise ReplicaValidationError("tensor must expose nbytes or numel/element_size")


def _contiguous(value: Any) -> bool:
    value = _local(value)
    method = getattr(value, "is_contiguous", None)
    if callable(method):
        return bool(method())
    flags = getattr(value, "flags", None)
    return bool(getattr(flags, "c_contiguous", True))


def _clone(value: Any) -> Any:
    value = _local(value)
    method = getattr(value, "clone", None)
    if callable(method):
        return method()
    method = getattr(value, "copy", None)
    if callable(method):
        return method()
    return copy.deepcopy(value)


def _empty_like(value: Any, shape: tuple[int, ...]) -> Any:
    value = _local(value)
    method = getattr(value, "new_empty", None)
    if callable(method):
        return method(shape)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return np.empty(shape, dtype=value.dtype)
    except ImportError:
        pass
    raise ReplicaValidationError("cannot allocate collective output for tensor type")


def _concat(values: Sequence[Any]) -> Any:
    first = _local(values[0])
    if type(first).__module__.startswith("torch"):
        import torch  # type: ignore

        return torch.cat([_local(value) for value in values], dim=0)
    try:
        import numpy as np  # type: ignore

        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=0)
    except ImportError:
        pass
    if isinstance(first, list):
        result: list[Any] = []
        for value in values:
            result.extend(copy.deepcopy(value))
        return result
    if isinstance(first, tuple):
        result_tuple: tuple[Any, ...] = ()
        for value in values:
            result_tuple += tuple(copy.deepcopy(value))
        return result_tuple
    raise ReplicaValidationError("collective needs all_gather_into_tensor for this tensor type")


def _split(value: Any, parts: int) -> list[Any]:
    shape = _shape(value)
    if shape[0] % parts:
        raise ReplicaValidationError("full table rows must divide EP size")
    rows = shape[0] // parts
    local = _local(value)
    if type(local).__module__.startswith("torch"):
        import torch  # type: ignore

        return list(torch.chunk(local, parts, dim=0))
    try:
        import numpy as np  # type: ignore

        if isinstance(local, np.ndarray):
            return list(np.split(local, parts, axis=0))
    except ImportError:
        pass
    return [value[index * rows : (index + 1) * rows] for index in range(parts)]


def _call(method: Callable[..., Any], *args: Any, group: Any = None) -> Any:
    if group is None:
        return method(*args)
    try:
        return method(*args, group=group)
    except TypeError as first:
        try:
            return method(*args, group)
        except TypeError:
            raise first


class TorchCollective:
    """Adapter for the existing Torch EP process group."""

    def __init__(self, group: Any = None):
        try:
            import torch.distributed as dist  # type: ignore
        except ImportError as exc:  # pragma: no cover - Torch-less CI
            raise RuntimeError("TorchCollective requires torch.distributed") from exc
        self.dist, self.group = dist, group

    def all_gather_into_tensor(self, output: Any, local: Any) -> None:
        self.dist.all_gather_into_tensor(output, _local(local), group=self.group)

    def reduce_scatter(self, full: Any) -> Any:
        world = self.dist.get_world_size(self.group)
        shape = (_shape(full)[0] // world,) + _shape(full)[1:]
        output = _empty_like(full, shape)
        self.dist.reduce_scatter_tensor(output, full, group=self.group)
        return output


class LoRAAReplicaManager:
    """Manage one-step full gate/up LoRA-A tables without changing FSDP."""

    def __init__(
        self,
        *,
        ep_size: int = DEFAULT_EP_SIZE,
        collective: Any = None,
        group: Any = None,
        enabled: bool = False,
        layer_peak_budget_bytes: int = DEFAULT_LAYER_PEAK_BUDGET_BYTES,
        total_peak_budget_bytes: int = DEFAULT_TOTAL_PEAK_BUDGET_BYTES,
    ) -> None:
        if not isinstance(ep_size, int) or isinstance(ep_size, bool) or ep_size < 1:
            raise ValueError("ep_size must be a positive integer")
        if layer_peak_budget_bytes <= 0 or total_peak_budget_bytes <= 0:
            raise ValueError("memory budgets must be positive")
        if layer_peak_budget_bytes > total_peak_budget_bytes:
            raise ValueError("layer budget cannot exceed total budget")
        if enabled and ep_size > 1 and collective is None:
            raise ValueError("enabled EP replicas require a collective")
        self.ep_size = ep_size
        self.collective = collective
        self.group = group
        self.enabled = bool(enabled)
        self.layer_budget = layer_peak_budget_bytes
        self.total_budget = total_peak_budget_bytes
        self._layers: dict[Hashable, _Layer] = {}
        self._step: Hashable | None = None
        self._full_grad_live = 0
        self._peak_projected = 0
        self._replica_peak = 0
        self._lifetime_steps = 0
        self._ag_calls = self._rs_calls = 0
        self._ag_local = self._ag_full = self._ag_wire = 0
        self._rs_full = self._rs_local = self._rs_wire = 0
        self._full_grad_peak = 0

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ReplicaManagerDisabledError("LoRA-A replicas are disabled")

    def _require_step(self) -> None:
        self._require_enabled()
        if self._step is None:
            raise ReplicaLifecycleError("begin_step is required")

    def register_layer(self, layer_id: Hashable, local_shard: Any, *, version: Any = 0) -> None:
        """Register the existing FSDP/EP local shard; never replace it."""

        self._require_enabled()
        owner, local = local_shard, _local(local_shard)
        shape = _shape(local)
        if len(shape) < 2 or not shape[0] or any(dim <= 0 for dim in shape[1:]):
            raise ReplicaValidationError("LoRA-A shard must be a non-empty expert-major table")
        if not _contiguous(local):
            raise ReplicaValidationError("LoRA-A shard must be contiguous")
        previous = self._layers.get(layer_id)
        if previous is not None and previous.replica is not None and previous.owner is not owner:
            raise ReplicaLifecycleError("cannot replace a shard while its replica is live")
        if previous is not None and previous.shape != shape:
            raise ReplicaValidationError("LoRA-A shard shape changed")
        if previous is not None and previous.version != version:
            raise StaleReplicaError(f"layer {layer_id!r} version changed without optimizer update")
        self._layers[layer_id] = _Layer(
            owner=owner,
            local=local,
            version=version,
            shape=shape,
            local_bytes=_nbytes(local),
            replica=previous.replica if previous else None,
            replica_version=previous.replica_version if previous else None,
            replica_bytes=previous.replica_bytes if previous else 0,
            full_shape=previous.full_shape if previous else None,
        )

    register_parameter = register_layer

    def begin_step(self, step_id: Hashable) -> None:
        self._require_enabled()
        if self._step is not None:
            raise ReplicaLifecycleError(f"step {self._step!r} is still active")
        if any(layer.replica is not None for layer in self._layers.values()):
            self._clear()
            raise ReplicaLifecycleError("stale replica remained before step start")
        self._step = step_id

    start_step = begin_step

    def _layer(self, layer_id: Hashable) -> _Layer:
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise ReplicaLifecycleError(f"unknown LoRA-A layer {layer_id!r}") from exc

    def _preflight(self, layer: _Layer, layer_id: Hashable) -> int:
        full_bytes = layer.local_bytes * self.ep_size
        layer_peak = full_bytes * 2
        projected = self._replica_live + full_bytes
        self._peak_projected = max(self._peak_projected, projected + full_bytes)
        if layer_peak > self.layer_budget:
            raise ReplicaMemoryError(f"layer {layer_id!r} exceeds full-replica + full-dA budget")
        if projected + full_bytes >= self.total_budget:
            raise ReplicaMemoryError("projected LoRA-A replica peak reaches total budget")
        return full_bytes

    @property
    def _replica_live(self) -> int:
        return sum(layer.replica_bytes for layer in self._layers.values() if layer.replica is not None)

    def replica(self, layer_id: Hashable, local_shard: Any | None = None, *, version: Any = None) -> Any:
        """All-gather one stable expert-major table and retain it for the step."""

        self._require_step()
        layer = self._layer(layer_id)
        if local_shard is not None:
            self.register_layer(layer_id, local_shard, version=layer.version if version is None else version)
            layer = self._layer(layer_id)
        if version is not None and version != layer.version:
            raise StaleReplicaError(f"layer {layer_id!r} replica version is stale")
        if layer.replica is not None:
            if layer.replica_version != layer.version:
                self._clear()
                raise StaleReplicaError(f"layer {layer_id!r} replica is stale")
            return layer.replica
        try:
            full_bytes = self._preflight(layer, layer_id)
            full = self._gather(layer.local)
            expected = (layer.shape[0] * self.ep_size,) + layer.shape[1:]
            if _shape(full) != expected or _nbytes(full) != full_bytes:
                raise ReplicaValidationError(f"all-gather returned invalid table for {layer_id!r}")
            layer.replica, layer.replica_version = full, layer.version
            layer.replica_bytes, layer.full_shape = full_bytes, expected
            self._replica_peak = max(self._replica_peak, self._replica_live)
            self._ag_calls += 1
            self._ag_local += layer.local_bytes
            self._ag_full += full_bytes
            self._ag_wire += layer.local_bytes * (self.ep_size - 1)
            return full
        except BaseException as exc:
            self._abort()
            if isinstance(exc, ReplicaManagerError):
                raise
            raise ReplicaCollectiveError(f"all-gather failed for {layer_id!r}") from exc

    get_replica = replica
    full_table = replica

    def register_gate_up_layer(
        self,
        layer_id: Hashable,
        gate_local: Any,
        up_local: Any,
        *,
        version: Any = 0,
    ) -> None:
        """Register gate and up A tables under one layer name."""

        self.register_layer((layer_id, "gate"), gate_local, version=version)
        try:
            self.register_layer((layer_id, "up"), up_local, version=version)
        except BaseException:
            self._layers.pop((layer_id, "gate"), None)
            raise

    register_gate_up = register_gate_up_layer

    def gate_up_replica(
        self,
        layer_id: Hashable,
        gate_local: Any | None = None,
        up_local: Any | None = None,
        *,
        version: Any = None,
    ) -> GateUpReplica:
        if (gate_local is None) != (up_local is None):
            raise ReplicaValidationError("gate and up local shards must be supplied together")
        if gate_local is not None:
            self.register_gate_up_layer(
                layer_id, gate_local, up_local, version=0 if version is None else version
            )
        gate = self.replica((layer_id, "gate"), version=version)
        up = self.replica((layer_id, "up"), version=version)
        return GateUpReplica(gate, up, self._layer((layer_id, "gate")).version)

    get_gate_up_replica = gate_up_replica

    def _gather(self, local: Any) -> Any:
        if self.ep_size == 1:
            return _clone(local)
        if self.collective is None:
            raise ReplicaCollectiveError("EP collective is not configured")
        method = getattr(self.collective, "all_gather_into_tensor", None)
        if callable(method):
            output = _empty_like(local, (_shape(local)[0] * self.ep_size,) + _shape(local)[1:])
            _call(method, output, local, group=self.group)
            return output
        method = getattr(self.collective, "all_gather", None)
        if not callable(method):
            raise ReplicaCollectiveError("collective has no all_gather operation")
        try:
            result = _call(method, local, group=self.group)
        except TypeError:
            outputs = [_empty_like(local, _shape(local)) for _ in range(self.ep_size)]
            result = _call(method, outputs, local, group=self.group)
            result = outputs if result is None else result
        if result is None:
            raise ReplicaCollectiveError("all_gather returned no output")
        if not isinstance(result, (list, tuple)):
            return _clone(result)
        if len(result) != self.ep_size:
            raise ReplicaValidationError("all_gather returned the wrong rank count")
        if any(_shape(part) != _shape(local) for part in result):
            raise ReplicaValidationError("all_gather changed local shard shape")
        return _concat(result)

    def reduce_scatter_gradient(
        self,
        layer_id: Hashable,
        full_gradient: Any,
        *,
        version: Any = None,
        target: Any | None = None,
    ) -> Any:
        """Reduce-scatter one source-computed full dA; retain local dA only."""

        self._require_step()
        try:
            layer = self._layer(layer_id)
            if version is not None and version != layer.version:
                raise StaleReplicaError(f"layer {layer_id!r} gradient version is stale")
            if layer.replica is None or layer.full_shape is None:
                raise ReplicaLifecycleError(f"layer {layer_id!r} has no live replica")
            full_gradient = _detach(_local(full_gradient))
            if _shape(full_gradient) != layer.full_shape or _nbytes(full_gradient) != layer.replica_bytes:
                raise ReplicaValidationError(f"full dA shape/bytes are invalid for {layer_id!r}")
            projected = self._replica_live + layer.replica_bytes
            if layer.replica_bytes * 2 > self.layer_budget or projected >= self.total_budget:
                raise ReplicaMemoryError(f"full dA for {layer_id!r} exceeds memory budget")
            self._full_grad_live = layer.replica_bytes
            self._full_grad_peak = max(self._full_grad_peak, self._full_grad_live)
            local_gradient = self._reduce_scatter(full_gradient, layer)
            if _shape(local_gradient) != layer.shape:
                raise ReplicaValidationError(f"reduce-scatter returned invalid local dA for {layer_id!r}")
            if target is not None:
                add = getattr(target, "add_", None)
                if callable(add):
                    add(local_gradient)
                else:
                    target[...] += local_gradient
            self._rs_calls += 1
            self._rs_full += layer.replica_bytes
            self._rs_local += layer.local_bytes
            self._rs_wire += layer.replica_bytes - layer.local_bytes
            return local_gradient
        except BaseException as exc:
            self._abort()
            if isinstance(exc, ReplicaManagerError):
                raise
            raise ReplicaCollectiveError(f"reduce-scatter failed for {layer_id!r}") from exc
        finally:
            self._full_grad_live = 0

    reduce_scatter_grad = reduce_scatter_gradient

    def reduce_scatter_gate_up_gradients(
        self,
        layer_id: Hashable,
        gate_full: Any,
        up_full: Any,
        *,
        version: Any = None,
        gate_target: Any | None = None,
        up_target: Any | None = None,
    ) -> tuple[Any, Any]:
        # Each call releases its full dA before the next reduction starts.
        gate = self.reduce_scatter_gradient(
            (layer_id, "gate"), gate_full, version=version, target=gate_target
        )
        up = self.reduce_scatter_gradient(
            (layer_id, "up"), up_full, version=version, target=up_target
        )
        return gate, up

    reduce_scatter_gate_up = reduce_scatter_gate_up_gradients

    def _reduce_scatter(self, full: Any, layer: _Layer) -> Any:
        if self.ep_size == 1:
            return _clone(full)
        if self.collective is None:
            raise ReplicaCollectiveError("EP collective is not configured")
        method = getattr(self.collective, "reduce_scatter", None)
        if not callable(method):
            raise ReplicaCollectiveError("collective has no reduce_scatter operation")
        try:
            result = _call(method, full, group=self.group)
        except TypeError as first:
            output = _empty_like(full, layer.shape)
            try:
                result = _call(method, output, _split(full, self.ep_size), group=self.group)
            except TypeError:
                raise first
            result = output if result is None else result
        if result is None:
            raise ReplicaCollectiveError("reduce_scatter returned no output")
        return result

    def source_compute_and_reduce_scatter(
        self,
        layer_id: Hashable,
        source_compute: Callable[[Any], Any],
        *,
        version: Any = None,
        target: Any | None = None,
    ) -> Any:
        """Run source dA computation and reduce-scatter at the layer boundary."""

        try:
            full = source_compute(self.replica(layer_id, version=version))
            return self.reduce_scatter_gradient(layer_id, full, version=version, target=target)
        except BaseException:
            self._abort()
            raise

    compute_and_reduce = source_compute_and_reduce_scatter

    def mark_optimizer_step(self, versions: Mapping[Hashable, Any] | None = None) -> None:
        """Invalidate replicas immediately after the optimizer updates local A."""

        self._require_step()
        if self._full_grad_live:
            raise ReplicaLifecycleError("full dA is still live")
        versions = versions or {}
        pair_bases = {
            key[0]
            for key in self._layers
            if isinstance(key, tuple) and len(key) == 2
        }
        unknown = {key for key in versions if key not in self._layers and key not in pair_bases}
        if unknown:
            raise ReplicaLifecycleError("optimizer version contains an unknown layer")
        next_versions: dict[Hashable, Any] = {}
        try:
            for layer_id, layer in self._layers.items():
                value = versions.get(layer_id)
                if value is None and isinstance(layer_id, tuple) and layer_id[0] in versions:
                    value = versions[layer_id[0]]
                if value is None:
                    if not isinstance(layer.version, int) or isinstance(layer.version, bool):
                        raise StaleReplicaError(f"next version required for {layer_id!r}")
                    value = layer.version + 1
                if value == layer.version:
                    raise StaleReplicaError(f"optimizer did not advance {layer_id!r}")
                next_versions[layer_id] = value
        except BaseException:
            self._abort()
            raise
        for layer_id, layer in self._layers.items():
            self._invalidate(layer)
            layer.version = next_versions[layer_id]

    after_optimizer_step = mark_optimizer_step

    def optimizer_step(
        self,
        optimizer: Any,
        *,
        versions: Mapping[Hashable, Any] | None = None,
    ) -> None:
        """Run the existing local-shard optimizer and invalidate replicas."""

        self._require_step()
        try:
            optimizer.step()
            self.mark_optimizer_step(versions)
        except BaseException:
            self._abort()
            raise

    def assert_optimizer_owns_local_shards(self, optimizer: Any) -> None:
        groups = getattr(optimizer, "param_groups", None)
        if groups is None:
            raise ReplicaLifecycleError("optimizer has no param_groups")
        ids = {id(parameter) for group in groups for parameter in group.get("params", ())}
        for layer_id, layer in self._layers.items():
            if id(layer.owner) not in ids and id(layer.local) not in ids:
                raise ReplicaLifecycleError(f"optimizer does not own local A for {layer_id!r}")
            if layer.replica is not None and id(layer.replica) in ids:
                raise ReplicaLifecycleError(f"optimizer owns full replica for {layer_id!r}")

    def end_step(self) -> None:
        self._require_step()
        if self._full_grad_live:
            self._abort()
            raise ReplicaLifecycleError("cannot end step while full dA is live")
        self._clear()
        self._lifetime_steps += 1
        self._step = None

    finish_step = end_step

    def release(self) -> None:
        """Release step replicas while keeping the step lifecycle open."""

        self._require_step()
        if self._full_grad_live:
            self._abort()
            raise ReplicaLifecycleError("cannot release while full dA is live")
        self._clear()

    release_replicas = release

    def _invalidate(self, layer: _Layer) -> None:
        layer.replica = layer.replica_version = None
        layer.replica_bytes = 0
        layer.full_shape = None

    def _clear(self) -> None:
        for layer in self._layers.values():
            self._invalidate(layer)

    def _abort(self) -> None:
        self._clear()
        self._full_grad_live = 0
        self._step = None

    def abort_step(self) -> None:
        self._require_enabled()
        self._abort()

    @contextmanager
    def step(self, step_id: Hashable):
        self.begin_step(step_id)
        try:
            yield self
        except BaseException:
            self._abort()
            raise
        else:
            self.end_step()

    @property
    def active(self) -> bool:
        return self._step is not None

    @property
    def memory(self) -> ReplicaMemoryAccounting:
        live = self._replica_live
        return ReplicaMemoryAccounting(
            replica_live_bytes=live,
            full_gradient_live_bytes=self._full_grad_live,
            projected_peak_bytes=live + self._full_grad_live,
            peak_projected_bytes=self._peak_projected,
            layer_peak_budget_bytes=self.layer_budget,
            total_peak_budget_bytes=self.total_budget,
        )

    accounting = memory

    @property
    def collective_stats(self) -> ReplicaCollectiveStats:
        return ReplicaCollectiveStats(
            all_gather_calls=self._ag_calls,
            reduce_scatter_calls=self._rs_calls,
            all_gather_local_bytes=self._ag_local,
            all_gather_full_bytes=self._ag_full,
            all_gather_wire_bytes=self._ag_wire,
            reduce_scatter_full_bytes=self._rs_full,
            reduce_scatter_local_bytes=self._rs_local,
            reduce_scatter_wire_bytes=self._rs_wire,
            replica_lifetime_steps=self._lifetime_steps,
            replica_peak_bytes=self._replica_peak,
            full_gradient_peak_bytes=self._full_grad_peak,
        )

    stats = collective_stats


LoRAAReplicaManager = LoRAAReplicaManager
EP8LoRAAReplicaManager = LoRAAReplicaManager

__all__ = [
    "DEFAULT_EP_SIZE",
    "DEFAULT_FULL_REPLICA_BYTES",
    "DEFAULT_LAYER_PEAK_BUDGET_BYTES",
    "DEFAULT_LOCAL_SHARD_BYTES",
    "DEFAULT_TOTAL_PEAK_BUDGET_BYTES",
    "EP8LoRAAReplicaManager",
    "GateUpReplica",
    "GIB",
    "LoRAAReplicaManager",
    "ReplicaCollectiveError",
    "ReplicaCollectiveStats",
    "ReplicaLifecycleError",
    "ReplicaManagerDisabledError",
    "ReplicaManagerError",
    "ReplicaMemoryAccounting",
    "ReplicaMemoryError",
    "ReplicaValidationError",
    "StaleReplicaError",
    "TorchCollective",
]
