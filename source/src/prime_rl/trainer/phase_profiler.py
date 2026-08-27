"""Opt-in, low-overhead CUDA phase timing for the GLM trainer.

The profiler is inactive unless ``PRIME_PHASE_PROFILE=1``.  It records CUDA
events only on selected steps and resolves them with one device
synchronization at the end of that step.  It does not change execution order,
streams, or tensor values.

The separate Nsight Systems CUDA+NVTX range is application-controlled.  It
starts on one selected post-warmup step and stops after that step's backward
pass, before optimizer work.
"""

from __future__ import annotations

import heapq
import json
import os
import statistics
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

_CUDA_MEMORY_CAPTURE_MAX_ENTRIES = 4096
_CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES = 1 << 20
_CUDA_MEMORY_CAPTURE_MAX_SEGMENTS = 64
_CUDA_MEMORY_CAPTURE_MAX_BLOCKS = 256
_CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES = 256


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_text(value: Any, limit: int = 160) -> str:
    return str(value)[:limit]


def _compact_frame(frames: Any) -> dict[str, Any]:
    frame = frames[0] if frames else {}
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


class CudaMemoryCapture:
    """Capture one bounded CUDA allocator snapshot for rank 0, step 1."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("PRIME_CUDA_MEMORY_CAPTURE", "0") == "1"
        self._active = False
        self._captured = False
        self._path: Path | None = None

    @property
    def active(self) -> bool:
        return self._active

    def begin_step(self, step: int) -> bool:
        if not self.enabled:
            return False
        if int(step) != 1 or os.environ.get("RANK", "") != "0":
            return False
        if self._active:
            raise RuntimeError("CUDA memory capture is already active")
        if self._captured:
            return False
        output = os.environ.get("PRIME_CUDA_MEMORY_CAPTURE_PATH", "")
        if not output:
            raise ValueError(
                "PRIME_CUDA_MEMORY_CAPTURE_PATH is required when CUDA memory capture is enabled"
            )
        memory = getattr(torch.cuda, "memory", None)
        record_history = getattr(memory, "_record_memory_history", None)
        snapshot = getattr(memory, "_snapshot", None)
        if not callable(record_history) or not callable(snapshot):
            raise RuntimeError(
                "CUDA memory capture requires the pinned PyTorch memory history APIs"
            )
        try:
            record_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=_CUDA_MEMORY_CAPTURE_MAX_ENTRIES,
                clear_history=True,
            )
            self._path = Path(output)
        except BaseException:
            try:
                record_history(enabled=None)
            except BaseException:
                pass
            raise
        self._active = True
        self._captured = True
        return True

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        return (
            oom_type is not None
            and isinstance(error, oom_type)
            and "cuda" in str(error).lower()
        )

    def _finish(self, status: str, error: BaseException | None = None) -> bool:
        if not self._active:
            return False
        memory = getattr(torch.cuda, "memory", None)
        snapshot = getattr(memory, "_snapshot", None)
        record_history = getattr(memory, "_record_memory_history", None)
        path = self._path
        diagnostic_error = None
        try:
            if not callable(snapshot) or path is None:
                raise RuntimeError("CUDA memory capture was not initialized")
            record = {
                "format": "prime-cuda-memory-capture-v1",
                "rank": 0,
                "step": 1,
                "status": status,
                "snapshot": _compact_memory_snapshot(snapshot()),
            }
            if error is not None:
                record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:512],
                }
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
            payload_bytes = payload.encode("utf-8")
            if len(payload_bytes) > _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES:
                record = {
                    "format": "prime-cuda-memory-capture-v1",
                    "rank": 0,
                    "step": 1,
                    "status": status,
                    "truncated": True,
                }
                payload_bytes = json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            if len(payload_bytes) > _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES:
                raise RuntimeError("CUDA memory capture output exceeds its fixed limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload_bytes)
        except BaseException as caught:
            diagnostic_error = caught
        finally:
            if callable(record_history):
                try:
                    record_history(enabled=None)
                except BaseException as caught:
                    if diagnostic_error is None:
                        diagnostic_error = caught
            self._path = None
            self._active = False
        if diagnostic_error is not None:
            raise diagnostic_error
        return True

    def end_step(self) -> bool:
        return self._finish("end")

    def dump_on_oom(self, error: BaseException) -> bool:
        if not self._active or not self._is_cuda_oom(error):
            return False
        return self._finish("cuda_oom", error)


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
