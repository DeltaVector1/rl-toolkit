"""Opt-in, low-overhead CUDA phase timing for the GLM trainer.

The profiler is inactive unless ``PRIME_PHASE_PROFILE=1``.  It records CUDA
events only on selected steps and resolves them with one device
synchronization at the end of that step.  It does not change execution order,
streams, or tensor values.

The separate Nsight Systems CUDA+NVTX range is application-controlled.  It
starts on one selected post-warmup step and stops after that step's backward
pass, before optimizer work.

CUDA OOM diagnostics are independent of phase timing.  They collect small
rank-local records on the error path and do not issue distributed operations.
"""

from __future__ import annotations

import heapq
import json
import os
import statistics
import tempfile
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch

PHASES = (
    "attention_forward",
    "attention_recompute",
    "attention_backward",
    "routed_expert_base_forward",
    "routed_expert_base_recompute",
    "routed_expert_base_backward",
    "lora_forward",
    "lora_backward",
    "sparse_mla_backward",
    "fsdp_collective",
    "fsdp_wait",
    "pp_send",
    "pp_recv",
    "pp_wait",
    "optimizer",
    "unclassified_compute",
)
_PHASE_SET = frozenset(PHASES)
_PROFILE_REQUESTED = os.environ.get("PRIME_PHASE_PROFILE", "0") == "1"
PROFILE_REQUESTED = _PROFILE_REQUESTED
_EVENT_PAIR_COST_US = 20.0
_CUDA_CAPTURE_ACTIVE = False
_CUDA_CAPTURE_API = None

# CUDA memory diagnostics are intentionally much smaller than the full PyTorch
# allocator snapshot.  A snapshot is only requested from explicitly selected
# ranks and only while a selected step is running.  Every rank still keeps the
# cheap OOM path armed so a local failure does not depend on a collective.
_CUDA_MEMORY_CAPTURE_MAX_ENTRIES = 4096
_CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES = 256 * 1024
_CUDA_MEMORY_CAPTURE_MAX_SEGMENTS = 64
_CUDA_MEMORY_CAPTURE_MAX_BLOCKS = 256
_CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES = 256
_CUDA_MEMORY_CAPTURE_MAX_ERROR_CHARS = 512


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_text(value: Any, limit: int = 160) -> str:
    try:
        return str(value)[:limit]
    except BaseException:
        return type(value).__name__[:limit]


def _compact_frame(frames: Any) -> dict[str, Any]:
    frame = frames[0] if isinstance(frames, (list, tuple)) and frames else {}
    if not isinstance(frame, dict):
        return {"frame": _safe_text(frame)}
    return {
        "filename": _safe_text(frame.get("filename", "<unknown>")),
        "line": _safe_int(frame.get("line", 0)),
        "name": _safe_text(frame.get("name", "")),
    }


def _compact_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep a bounded allocator summary from a CUDA memory snapshot."""
    segments = snapshot.get("segments", ())
    segment_heap = []
    block_heap = []
    segment_count = 0
    active_block_count = 0
    for segment in segments if isinstance(segments, (list, tuple)) else ():
        if not isinstance(segment, dict):
            continue
        segment_count += 1
        segment_row = {
            "address": _safe_int(segment.get("address")),
            "total_size": _safe_int(segment.get("total_size")),
            "allocated_size": _safe_int(segment.get("allocated_size")),
            "active_size": _safe_int(segment.get("active_size")),
            "segment_type": _safe_text(segment.get("segment_type", ""), 32),
        }
        segment_item = (
            segment_row["active_size"],
            segment_row["total_size"],
            segment_count,
            segment_row,
        )
        if len(segment_heap) < _CUDA_MEMORY_CAPTURE_MAX_SEGMENTS:
            heapq.heappush(segment_heap, segment_item)
        else:
            heapq.heappushpop(segment_heap, segment_item)
        blocks = segment.get("blocks", ())
        for block in blocks if isinstance(blocks, (list, tuple)) else ():
            if not isinstance(block, dict):
                continue
            state = _safe_text(block.get("state", ""), 32)
            if not state.startswith("active"):
                continue
            active_block_count += 1
            block_row = {
                "address": _safe_int(block.get("address")),
                "size": _safe_int(block.get("size")),
                "requested_size": _safe_int(block.get("requested_size")),
                "state": state,
                "frame": _compact_frame(block.get("frames", ())),
            }
            block_item = (
                block_row["size"],
                block_row["address"],
                active_block_count,
                block_row,
            )
            if len(block_heap) < _CUDA_MEMORY_CAPTURE_MAX_BLOCKS:
                heapq.heappush(block_heap, block_item)
            else:
                heapq.heappushpop(block_heap, block_item)

    traces = snapshot.get("device_traces", ())
    trace_rows = deque(maxlen=_CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES)
    if isinstance(traces, (list, tuple)):
        for device_trace in traces:
            if not isinstance(device_trace, (list, tuple)):
                continue
            trace_rows.extend(
                device_trace[-_CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES :]
            )
    compact_traces = []
    for trace in trace_rows:
        if not isinstance(trace, dict):
            continue
        compact_traces.append(
            {
                "action": _safe_text(trace.get("action", ""), 32),
                "addr": _safe_int(trace.get("addr")),
                "size": _safe_int(trace.get("size")),
                "stream": _safe_int(trace.get("stream")),
                "device_free": _safe_int(trace.get("device_free")),
                "frame": _compact_frame(trace.get("frames", ())),
            }
        )

    segment_rows = [item[3] for item in sorted(segment_heap, reverse=True)]
    block_rows = [item[3] for item in sorted(block_heap, reverse=True)]
    return {
        "segment_count": segment_count,
        "active_block_count": active_block_count,
        "trace_count": sum(
            len(trace) for trace in traces if isinstance(trace, (list, tuple))
        )
        if isinstance(traces, (list, tuple))
        else 0,
        "largest_segments": segment_rows,
        "largest_active_blocks": block_rows,
        "trace_tail": compact_traces,
    }


def _parse_unsigned_env(name: str, *, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if not value.isdecimal():
        raise ValueError(f"{name} must be an unsigned integer")
    return int(value)


def _parse_rank_selector(value: str) -> frozenset[int]:
    """Parse the optional allocator-history rank selector.

    The selector is deliberately separate from artifact collection.  An empty
    selector means that every rank writes lightweight OOM evidence but no rank
    enables allocator history.  ``all`` is accepted only as an explicit opt-in.
    """
    value = value.strip()
    if not value:
        return frozenset()
    if value.lower() == "all":
        return frozenset({-1})
    ranks = []
    for part in value.split(","):
        part = part.strip()
        if not part.isdecimal():
            raise ValueError(
                "PRIME_CUDA_MEMORY_CAPTURE_RANK must contain unsigned integers"
            )
        ranks.append(int(part))
    return frozenset(ranks)


def _rank_artifact_path(base: Path, rank: int) -> Path:
    """Return a unique path for one rank without changing the configured base."""
    if base.suffix:
        return base.with_name(f"{base.stem}.rank_{rank}{base.suffix}")
    return base / f"rank_{rank}.json"


def _safe_stat(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _cuda_memory_stats() -> dict[str, int | float]:
    """Read a small set of allocator/driver counters without synchronizing CUDA."""
    stats: dict[str, int | float] = {}
    cuda = getattr(torch, "cuda", None)
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        function = getattr(cuda, name, None)
        if not callable(function):
            continue
        try:
            value = _safe_stat(function())
        except BaseException:
            continue
        if value is not None:
            stats[name] = value

    memory_stats = getattr(cuda, "memory_stats", None)
    if callable(memory_stats):
        try:
            raw = memory_stats()
        except BaseException:
            raw = {}
        for key in (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
            "inactive_split_bytes.all.current",
            "num_alloc_retries",
            "num_ooms",
        ):
            value = _safe_stat(raw.get(key)) if hasattr(raw, "get") else None
            if value is not None:
                stats[key] = value

    mem_get_info = getattr(cuda, "mem_get_info", None)
    if callable(mem_get_info):
        try:
            free_bytes, total_bytes = mem_get_info()
            free = _safe_stat(free_bytes)
            total = _safe_stat(total_bytes)
            if free is not None:
                stats["driver_free_bytes"] = free
            if total is not None:
                stats["driver_total_bytes"] = total
        except BaseException:
            pass
    return stats


def _nvml_memory_stats() -> dict[str, int | float | str]:
    """Read optional NVML counters; importing NVML is deferred to the OOM path."""
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        return {"available": False}

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        index = _parse_unsigned_env("LOCAL_RANK", default=0) or 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result: dict[str, int | float | str] = {"available": True}
        for name in ("free", "total", "used"):
            value = _safe_stat(getattr(info, name, None))
            if value is not None:
                result[f"{name}_bytes"] = value
        utilization = getattr(pynvml, "nvmlDeviceGetUtilizationRates", None)
        if callable(utilization):
            rates = utilization(handle)
            for name in ("gpu", "memory"):
                value = _safe_stat(getattr(rates, name, None))
                if value is not None:
                    result[f"{name}_utilization_pct"] = value
        return result
    except BaseException as error:
        return {"available": False, "error": type(error).__name__[:64]}
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except BaseException:
                pass


class CudaMemoryCapture:
    """Arm a rank-local OOM diagnostic without distributed or CUDA sync calls.

    Every rank is armed when the feature is enabled.  The optional allocator
    history is enabled only on ``PRIME_CUDA_MEMORY_CAPTURE_RANK`` (or an
    explicit rank list), and Python stack recording requires the separate
    ``PRIME_CUDA_MEMORY_CAPTURE_STACKS=1`` opt-in.  A normal step end only
    disables history; it never writes a diagnostic artifact.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._active = False
        self._history_active = False
        self._history_error: str | None = None
        self._written = False
        self._rank: int | None = None
        self._step: int | None = None
        self._target_step = 1
        self._base_path: Path | None = None
        self._artifact_path: Path | None = None
        self._refresh_configuration()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def artifact_path(self) -> Path | None:
        return self._artifact_path

    def _refresh_configuration(self) -> bool:
        self.enabled = os.environ.get("PRIME_CUDA_MEMORY_CAPTURE", "0") == "1"
        if not self.enabled:
            return False
        self._rank = _parse_unsigned_env("RANK")
        if self._rank is None:
            return False
        self._target_step = _parse_unsigned_env(
            "PRIME_CUDA_MEMORY_CAPTURE_STEP", default=1
        ) or 1
        output = os.environ.get("PRIME_CUDA_MEMORY_CAPTURE_PATH", "")
        if not output:
            raise ValueError(
                "PRIME_CUDA_MEMORY_CAPTURE_PATH is required when CUDA memory capture is enabled"
            )
        self._base_path = Path(output)
        self._artifact_path = _rank_artifact_path(self._base_path, self._rank)
        return True

    def _history_selected(self) -> bool:
        if self._rank is None or self._step != self._target_step:
            return False
        selected = _parse_rank_selector(
            os.environ.get("PRIME_CUDA_MEMORY_CAPTURE_RANK", "")
        )
        return -1 in selected or self._rank in selected

    def _start_allocator_history(self) -> None:
        memory = getattr(getattr(torch, "cuda", None), "memory", None)
        record_history = getattr(memory, "_record_memory_history", None)
        snapshot = getattr(memory, "_snapshot", None)
        if not callable(record_history) or not callable(snapshot):
            self._history_error = "allocator_history_api_unavailable"
            return
        kwargs: dict[str, Any] = {
            "enabled": "all",
            "context": "all",
            "max_entries": _CUDA_MEMORY_CAPTURE_MAX_ENTRIES,
            "clear_history": True,
        }
        if os.environ.get("PRIME_CUDA_MEMORY_CAPTURE_STACKS", "0") == "1":
            kwargs["stacks"] = "python"
        try:
            record_history(**kwargs)
        except BaseException as error:
            self._history_error = type(error).__name__[:64]
            try:
                record_history(enabled=None)
            except BaseException:
                pass
            return
        self._history_active = True

    def _stop_allocator_history(self) -> None:
        if not self._history_active:
            return
        memory = getattr(getattr(torch, "cuda", None), "memory", None)
        record_history = getattr(memory, "_record_memory_history", None)
        if callable(record_history):
            try:
                record_history(enabled=None)
            except BaseException:
                pass
        self._history_active = False

    def begin_step(self, step: int) -> bool:
        """Arm this rank for one runtime step and optionally start history."""
        if self._active:
            raise RuntimeError("CUDA memory capture step was not ended")
        if self._written or not self._refresh_configuration():
            return False
        self._step = int(step)
        self._active = True
        self._history_error = None
        if self._history_selected():
            self._start_allocator_history()
        return True

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        oom_types = tuple(
            candidate
            for candidate in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
            )
            if isinstance(candidate, type)
        )
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text = str(current).lower()
            if oom_types and isinstance(current, oom_types):
                return "cpu" not in text
            if "cuda" in text and "out of memory" in text:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _encode_record(record: dict[str, Any]) -> bytes:
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def _bounded_payload(cls, record: dict[str, Any]) -> bytes:
        """Encode a record without ever returning a payload above the limit.

        Allocator snapshots are nested under ``allocator.snapshot``.  Build a
        small metadata record first so an arbitrary nested snapshot is never
        copied into an oversized fallback candidate.  A compact snapshot from
        ``_compact_memory_snapshot`` is retained when it fits; otherwise the
        fixed minimal record is used.
        """

        def small_mapping(value: Any) -> dict[str, int | float | str | bool]:
            if not isinstance(value, dict):
                return {}
            result: dict[str, int | float | str | bool] = {}
            for key, item in list(value.items())[:32]:
                safe_value = _safe_stat(item)
                if safe_value is None:
                    if isinstance(item, bool):
                        safe_value = item
                    elif isinstance(item, str):
                        safe_value = _safe_text(item, 128)
                    else:
                        continue
                result[_safe_text(key, 96)] = safe_value
            return result

        allocator = record.get("allocator")
        nested_snapshot = allocator.get("snapshot") if isinstance(allocator, dict) else None
        top_level_snapshot = record.get("snapshot")
        snapshot = nested_snapshot if nested_snapshot is not None else top_level_snapshot

        base: dict[str, Any] = {
            "format": _safe_text(record.get("format", "prime-cuda-memory-capture-v2"), 64),
            "rank": _safe_int(record.get("rank")),
            "step": _safe_int(record.get("step")),
            "status": _safe_text(record.get("status", "cuda_oom"), 32),
            "error": {
                "type": _safe_text(
                    record.get("error", {}).get("type", "")
                    if isinstance(record.get("error"), dict)
                    else "",
                    64,
                ),
                "message": _safe_text(
                    record.get("error", {}).get("message", "")
                    if isinstance(record.get("error"), dict)
                    else "",
                    _CUDA_MEMORY_CAPTURE_MAX_ERROR_CHARS,
                ),
            },
            "memory": small_mapping(record.get("memory")),
            "nvml": small_mapping(record.get("nvml")),
            "allocator": small_mapping(allocator),
        }

        def fit(candidate: dict[str, Any]) -> bytes | None:
            try:
                payload = cls._encode_record(candidate)
            except BaseException:
                return None
            return payload if len(payload) <= _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES else None

        # Normal lightweight records are already small.  Try the compact
        # allocator summary only when it has the shape produced by our
        # compactor; arbitrary user-provided objects stay out of the encoder.
        compact_snapshot = isinstance(snapshot, dict) and {
            "segment_count",
            "active_block_count",
            "trace_count",
            "largest_segments",
            "largest_active_blocks",
            "trace_tail",
        }.issuperset(snapshot)
        if compact_snapshot:
            # Add rows one at a time and retain only candidates that already
            # fit.  This avoids constructing an oversized JSON payload when a
            # compact summary is still too large for the artifact budget.
            def compact_row(key: str, item: Any) -> dict[str, Any] | None:
                if not isinstance(item, dict):
                    return None
                if key == "largest_segments":
                    return {
                        "address": _safe_int(item.get("address")),
                        "total_size": _safe_int(item.get("total_size")),
                        "allocated_size": _safe_int(item.get("allocated_size")),
                        "active_size": _safe_int(item.get("active_size")),
                        "segment_type": _safe_text(item.get("segment_type", ""), 32),
                    }
                if key == "largest_active_blocks":
                    row = {
                        "address": _safe_int(item.get("address")),
                        "size": _safe_int(item.get("size")),
                        "requested_size": _safe_int(item.get("requested_size")),
                        "state": _safe_text(item.get("state", ""), 32),
                    }
                else:
                    row = {
                        "action": _safe_text(item.get("action", ""), 32),
                        "addr": _safe_int(item.get("addr")),
                        "size": _safe_int(item.get("size")),
                        "stream": _safe_int(item.get("stream")),
                        "device_free": _safe_int(item.get("device_free")),
                    }
                frame = item.get("frame")
                if isinstance(frame, dict):
                    row["frame"] = {
                        "filename": _safe_text(frame.get("filename", "<unknown>")),
                        "line": _safe_int(frame.get("line", 0)),
                        "name": _safe_text(frame.get("name", "")),
                    }
                elif frame is not None:
                    row["frame"] = {"frame": _safe_text(frame)}
                return row

            snapshot_candidate: dict[str, Any] = {}
            for key in ("segment_count", "active_block_count", "trace_count"):
                value = _safe_stat(snapshot.get(key))
                if value is not None:
                    snapshot_candidate[key] = value
            omitted = False
            for key, limit in (
                ("largest_segments", _CUDA_MEMORY_CAPTURE_MAX_SEGMENTS),
                ("largest_active_blocks", _CUDA_MEMORY_CAPTURE_MAX_BLOCKS),
                ("trace_tail", _CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES),
            ):
                values = snapshot.get(key)
                rows: list[Any] = []
                if isinstance(values, (list, tuple)):
                    for item in values:
                        if len(rows) >= limit:
                            omitted = True
                            break
                        row = compact_row(key, item)
                        if row is None:
                            omitted = True
                            continue
                        candidate_snapshot = dict(snapshot_candidate)
                        candidate_snapshot[key] = [*rows, row]
                        candidate = dict(base)
                        candidate["allocator"] = dict(base["allocator"])
                        candidate["allocator"]["snapshot"] = candidate_snapshot
                        if fit(candidate) is None:
                            omitted = True
                            continue
                        rows.append(row)
                elif values is not None:
                    omitted = True
                snapshot_candidate[key] = rows
            with_snapshot = dict(base)
            with_snapshot["allocator"] = dict(base["allocator"])
            with_snapshot["allocator"]["snapshot"] = snapshot_candidate
            if omitted:
                with_snapshot["truncated"] = True
            payload = fit(with_snapshot)
            if payload is not None:
                return payload

        without_snapshot = dict(base)
        if snapshot is not None:
            without_snapshot["truncated"] = True
        payload = fit(without_snapshot)
        if payload is not None:
            return payload

        reduced = dict(without_snapshot)
        reduced["truncated"] = True
        reduced.pop("memory", None)
        reduced.pop("nvml", None)
        reduced["allocator"] = {
            key: value
            for key, value in base["allocator"].items()
            if key != "snapshot"
        }
        payload = fit(reduced)
        if payload is not None:
            return payload

        minimal = {
            "format": "prime-cuda-memory-capture-v2",
            "rank": _safe_int(record.get("rank")),
            "step": _safe_int(record.get("step")),
            "status": "cuda_oom",
            "truncated": True,
            "error": {
                "type": _safe_text(
                    record.get("error", {}).get("type", "")
                    if isinstance(record.get("error"), dict)
                    else "",
                    64,
                ),
                "message": _safe_text(
                    record.get("error", {}).get("message", "")
                    if isinstance(record.get("error"), dict)
                    else "",
                    128,
                ),
            },
        }
        payload = fit(minimal)
        if payload is not None:
            return payload
        # This literal is a final invariant even if an encoder is replaced by
        # a test double or returns an unexpected value.
        return b'{"format":"prime-cuda-memory-capture-v2","status":"cuda_oom","truncated":true}'

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _finish_oom(self, error: BaseException) -> bool:
        if self._written:
            return False
        # Refresh here as well as in begin_step.  The outer train() handler can
        # see an OOM during setup, before the first runtime step is entered.
        if not self._refresh_configuration():
            return False
        if self._rank is None:
            return False
        if self._artifact_path is None:
            if self._base_path is None:
                return False
            self._artifact_path = _rank_artifact_path(self._base_path, self._rank)

        memory = getattr(getattr(torch, "cuda", None), "memory", None)
        snapshot_function = getattr(memory, "_snapshot", None)
        allocator: dict[str, Any] = {"history_enabled": self._history_active}
        if self._history_active and callable(snapshot_function):
            try:
                allocator["snapshot"] = _compact_memory_snapshot(snapshot_function())
            except BaseException as snapshot_error:
                allocator["error"] = type(snapshot_error).__name__[:64]
        elif self._history_error is not None:
            allocator["error"] = self._history_error

        record = {
            "format": "prime-cuda-memory-capture-v2",
            "rank": self._rank,
            "step": self._step if self._step is not None else self._target_step,
            "status": "cuda_oom",
            "error": {
                "type": type(error).__name__[:64],
                "message": _safe_text(error, _CUDA_MEMORY_CAPTURE_MAX_ERROR_CHARS),
            },
            "memory": _cuda_memory_stats(),
            "nvml": _nvml_memory_stats(),
            "allocator": allocator,
        }
        self._stop_allocator_history()
        try:
            self._atomic_write(self._artifact_path, self._bounded_payload(record))
        except BaseException:
            # Diagnostics cannot replace the original OOM.  Mark this attempt
            # consumed so a later teardown path does not repeat expensive work.
            self._written = True
            self._active = False
            return False
        self._written = True
        self._active = False
        return True

    def end_step(self) -> bool:
        """Stop optional history; successful steps produce no artifact."""
        if not self._active:
            return False
        self._stop_allocator_history()
        self._active = False
        return True

    def dump_on_oom(self, error: BaseException) -> bool:
        """Write one rank-local artifact for a CUDA OOM, without collectives."""
        if not self._is_cuda_oom(error):
            return False
        return self._finish_oom(error)


_MEMORY_CAPTURE = CudaMemoryCapture()


def get_cuda_memory_capture() -> CudaMemoryCapture:
    return _MEMORY_CAPTURE


def _cuda_capture_step() -> int:
    value = os.environ.get("PRIME_CUDA_PROFILE_STEPS", "2")
    if not value.isdecimal() or int(value) < 2:
        raise ValueError(
            "PRIME_CUDA_PROFILE_STEPS must be one unsigned integer >= 2"
        )
    return int(value)


def start_cuda_capture(step: int) -> bool:
    """Start an opt-in Nsight CUDA-profiler range on one global rank."""
    global _CUDA_CAPTURE_ACTIVE, _CUDA_CAPTURE_API
    target = os.environ.get("PRIME_CUDA_PROFILE_RANK", "")
    if os.environ.get("PRIME_CUDA_PROFILE", "0") != "1" or not target:
        return False
    if not target.isdecimal() or not os.environ.get("RANK", "").isdecimal():
        raise ValueError("PRIME_CUDA_PROFILE_RANK and RANK must be unsigned integers")
    if int(os.environ["RANK"]) != int(target):
        return False
    if int(step) != _cuda_capture_step():
        return False
    if _CUDA_CAPTURE_ACTIVE:
        raise RuntimeError("CUDA profiler capture is already active")
    api = torch.cuda.cudart()
    result = api.cudaProfilerStart()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStart failed with error {result}")
    try:
        torch.cuda.nvtx.range_push(f"prime_nsys_training_step_{int(step)}")
    except BaseException:
        api.cudaProfilerStop()
        raise
    _CUDA_CAPTURE_API = api
    _CUDA_CAPTURE_ACTIVE = True
    return True


def stop_cuda_capture() -> bool:
    """End the one opt-in Nsight CUDA-profiler range."""
    global _CUDA_CAPTURE_ACTIVE, _CUDA_CAPTURE_API
    if not _CUDA_CAPTURE_ACTIVE:
        return False
    api = _CUDA_CAPTURE_API
    error = None
    try:
        torch.cuda.nvtx.range_pop()
    except BaseException as caught:
        error = caught
    try:
        result = api.cudaProfilerStop()
        if result not in (None, 0) and error is None:
            error = RuntimeError(f"cudaProfilerStop failed with error {result}")
    except BaseException as caught:
        if error is None:
            error = caught
    finally:
        _CUDA_CAPTURE_ACTIVE = False
        _CUDA_CAPTURE_API = None
    if error is not None:
        raise error
    return True


def _parse_steps(value: str) -> frozenset[int]:
    if not value:
        return frozenset({1})
    steps = frozenset(int(part) for part in value.split(",") if part)
    if not steps or min(steps) < 1:
        raise ValueError("PRIME_PHASE_PROFILE_STEPS must contain positive integers")
    return steps


def _parse_positive(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


class PhaseProfiler:
    """Collect sampled CUDA intervals for one trainer rank."""

    def __init__(self) -> None:
        self.enabled = _PROFILE_REQUESTED
        self.steps = _parse_steps(os.environ.get("PRIME_PHASE_PROFILE_STEPS", "1"))
        self.sample_every = _parse_positive(
            os.environ.get("PRIME_PHASE_PROFILE_SAMPLE_EVERY", "8"),
            "PRIME_PHASE_PROFILE_SAMPLE_EVERY",
        )
        self._active = False
        self._step = None
        self._step_start = None
        self._calls: dict[str, int] = {}
        self._spans: list[tuple[str, Any, Any, str | None]] = []
        self._stack: list[str] = []

    @property
    def active(self) -> bool:
        return self._active

    def begin_step(self, step: int) -> None:
        if not self.enabled:
            return
        if self._active:
            raise RuntimeError("phase profiler step was not ended")
        self._step = int(step)
        self._active = self._step in self.steps
        if not self._active:
            return
        self._calls.clear()
        self._spans.clear()
        self._stack.clear()
        try:
            self._step_start = torch.cuda.Event(enable_timing=True)
            self._step_start.record(torch.cuda.current_stream())
        except (AttributeError, AssertionError, RuntimeError):
            self._step_start = None

    def start_span(self, phase: str, *, stream=None):
        """Start a sampled phase without synchronizing the device."""
        if not self._active:
            return None
        if phase not in _PHASE_SET:
            raise ValueError(f"unknown phase {phase!r}")
        self._calls[phase] = self._calls.get(phase, 0) + 1
        sampled = self._calls[phase] == 1 or self._calls[phase] % self.sample_every == 0
        parent = self._stack[-1] if self._stack else None
        self._stack.append(phase)
        if not sampled:
            return (False, phase)
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream or torch.cuda.current_stream())
        except (AttributeError, AssertionError, RuntimeError):
            return (False, phase)
        return (True, phase, parent, start, end, stream)

    def end_span(self, timer) -> None:
        if timer is None:
            return
        phase = timer[1]
        if not self._stack or self._stack[-1] != phase:
            raise RuntimeError("phase profiler spans closed out of order")
        self._stack.pop()
        if not timer[0]:
            return
        _, _, parent, _start, end, stream = timer
        end.record(stream or torch.cuda.current_stream())
        self._spans.append((phase, timer[3], end, parent))

    @contextmanager
    def span(self, phase: str, *, stream=None) -> Iterator[None]:
        timer = self.start_span(phase, stream=stream)
        try:
            yield
        finally:
            self.end_span(timer)

    def end_step(self, *, rank: int = 0, stage: int = 0) -> dict[str, Any] | None:
        """Synchronize and resolve one measured step.

        This is the only synchronization performed by this profiler.  Sampled
        spans are expanded by ``sample_every`` and are labeled as estimates.
        """
        if not self._active:
            return None
        if self._stack:
            raise RuntimeError("phase profiler has open spans at step end")
        try:
            # The end event must be recorded before the one synchronization.
            step_end = torch.cuda.Event(enable_timing=True)
            step_end.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            step_ms = self._step_start.elapsed_time(step_end) if self._step_start else 0.0
        except (AttributeError, AssertionError, RuntimeError):
            step_ms = 0.0

        inclusive: dict[str, float] = {phase: 0.0 for phase in PHASES}
        children: dict[str, float] = {phase: 0.0 for phase in PHASES}
        sampled: dict[str, int] = {phase: 0 for phase in PHASES}
        calls: dict[str, int] = {phase: self._calls.get(phase, 0) for phase in PHASES}
        for phase, start, end, parent in self._spans:
            elapsed = start.elapsed_time(end) * self.sample_every
            inclusive[phase] += elapsed
            sampled[phase] += 1
            if parent in children:
                children[parent] += elapsed
        categories: dict[str, dict[str, float | int | bool]] = {}
        classified_sum = 0.0
        for phase in PHASES:
            exclusive = max(0.0, inclusive[phase] - children[phase])
            if phase != "unclassified_compute":
                classified_sum += exclusive
            categories[phase] = {
                "calls": calls[phase],
                "sampled": sampled[phase],
                "estimated": self.sample_every > 1,
                "inclusive_ms": inclusive[phase],
                "exclusive_ms": exclusive,
            }
        classified = min(step_ms, classified_sum)
        categories["unclassified_compute"]["inclusive_ms"] = max(0.0, step_ms - classified)
        categories["unclassified_compute"]["exclusive_ms"] = categories[
            "unclassified_compute"
        ]["inclusive_ms"]
        result = {
            "rank": int(rank),
            "stage": int(stage),
            "step": int(self._step),
            "step_ms": step_ms,
            "sample_every": self.sample_every,
            "classified_sum_ms": classified_sum,
            "modeled_overhead_fraction": self.modeled_overhead_fraction(
                self._calls, step_ms, sample_every=self.sample_every
            ),
            "categories": categories,
            "assertions": {
                "known_categories_only": True,
                "nested_intervals_subtracted": True,
                "single_end_sync": True,
                "classified_ms_le_step_ms": classified <= step_ms + 1e-3,
                "overlap_possible": classified_sum > step_ms + 1e-3,
            },
        }
        self._active = False
        self._step_start = None
        self._spans.clear()
        self._calls.clear()
        return result

    @staticmethod
    def modeled_overhead_fraction(
        calls: dict[str, int],
        step_ms: float,
        *,
        sample_every: int = 8,
        event_pair_cost_us: float = _EVENT_PAIR_COST_US,
    ) -> float:
        """Estimate event-record overhead for a measured step.

        The estimate charges a conservative fixed cost for each sampled event
        pair.  It is a gate for profiling configuration, not a replacement for
        a clean-versus-profiled GPU measurement.
        """
        if step_ms <= 0:
            return 0.0
        if sample_every < 1:
            raise ValueError("sample_every must be positive")
        sampled_pairs = sum(
            count and (1 + (count - 1) // sample_every) for count in calls.values()
        )
        return sampled_pairs * event_pair_cost_us / 1000.0 / step_ms

    @staticmethod
    def aggregate(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Return min/median/max per category across rank and stage reports."""
        if not reports:
            return {"reports": 0, "categories": {}}
        categories = {}
        for phase in PHASES:
            values = [
                float(report.get("categories", {}).get(phase, {}).get("exclusive_ms", 0.0))
                for report in reports
            ]
            categories[phase] = {
                "min_ms": min(values),
                "median_ms": statistics.median(values),
                "max_ms": max(values),
            }
        step_values = [float(report.get("step_ms", 0.0)) for report in reports]
        return {
            "reports": len(reports),
            "step_ms": {
                "min": min(step_values),
                "median": statistics.median(step_values),
                "max": max(step_values),
            },
            "categories": categories,
        }

    @classmethod
    def format_aggregate(cls, reports: Sequence[dict[str, Any]]) -> str:
        summary = cls.aggregate(reports)
        parts = [
            f"[phase-profile] reports={summary['reports']} "
            f"step_ms(min/median/max)={summary.get('step_ms', {}).get('min', 0.0):.1f}/"
            f"{summary.get('step_ms', {}).get('median', 0.0):.1f}/"
            f"{summary.get('step_ms', {}).get('max', 0.0):.1f}"
        ]
        for phase in PHASES:
            value = summary["categories"].get(phase)
            if value is not None:
                parts.append(
                    f"{phase}={value['min_ms']:.1f}/{value['median_ms']:.1f}/{value['max_ms']:.1f}ms"
                )
        return " ".join(parts)


_PROFILER = PhaseProfiler()


def get_phase_profiler() -> PhaseProfiler:
    return _PROFILER


def phase_span(phase: str | Callable[[], str]):
    """Decorate a function with a phase span while keeping disabled calls cheap."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not _PROFILE_REQUESTED or not _PROFILER.active:
                return function(*args, **kwargs)
            name = phase() if callable(phase) else phase
            with _PROFILER.span(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator


def _pp_phase(description, p2p_ops) -> str:
    text = f"{description!s} {p2p_ops!s}".lower()
    if "recv" in text and "send" not in text:
        return "pp_recv"
    if "send" in text and "recv" not in text:
        return "pp_send"
    return "pp_recv" if "recv" in text else "pp_send"


def install_pp_phase_hooks() -> bool:
    if not _PROFILER.enabled:
        return False
    try:
        from torch.distributed.pipelining import schedules
        original_batch = schedules._batch_p2p
        original_wait = schedules._wait_batch_p2p
    except (AttributeError, ImportError):
        return False
    if getattr(schedules, "_prime_phase_profile_installed", False):
        return True

    @wraps(original_batch)
    def traced_batch(p2p_ops, desc=None):
        with _PROFILER.span(_pp_phase(desc, p2p_ops)):
            return original_batch(p2p_ops, desc=desc)

    @wraps(original_wait)
    def traced_wait(work):
        with _PROFILER.span("pp_wait"):
            return original_wait(work)

    schedules._batch_p2p = traced_batch
    schedules._wait_batch_p2p = traced_wait
    schedules._prime_phase_profile_installed = True
    return True


def install_fsdp_phase_hooks() -> bool:
    if not _PROFILER.enabled:
        return False
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except (AttributeError, ImportError):
        return False
    if getattr(FSDPParamGroup, "_prime_phase_profile_installed", False):
        return True
    hooks = (
        ("unshard", "fsdp_collective"),
        ("wait_for_unshard", "fsdp_wait"),
        ("_wait_all_gather_streams_on_event", "fsdp_wait"),
    )
    for method_name, phase in hooks:
        original = getattr(FSDPParamGroup, method_name, None)
        if original is None:
            continue

        @wraps(original)
        def traced(self, *args, _original=original, _phase=phase, **kwargs):
            with _PROFILER.span(_phase):
                return _original(self, *args, **kwargs)

        setattr(FSDPParamGroup, method_name, traced)
    original_prefetch = getattr(FSDPParamGroup, "_prefetch_unshard", None)
    if original_prefetch is not None:
        @wraps(original_prefetch)
        def traced_prefetch(target_fsdp_param_group, pass_type):
            with _PROFILER.span("fsdp_collective"):
                return original_prefetch(target_fsdp_param_group, pass_type)

        FSDPParamGroup._prefetch_unshard = staticmethod(traced_prefetch)
    FSDPParamGroup._prime_phase_profile_installed = True
    return True
