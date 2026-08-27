"""Opt-in, low-overhead CUDA phase timing for the GLM trainer.

The profiler is inactive unless ``PRIME_PHASE_PROFILE=1``.  It records CUDA
events only on selected steps and resolves them with one device
synchronization at the end of that step.  It does not change execution order,
streams, or tensor values.
"""

from __future__ import annotations

import os
import statistics
from contextlib import contextmanager
from functools import wraps
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


def _cuda_capture_step_set() -> frozenset[int]:
    value = os.environ.get("PRIME_CUDA_PROFILE_STEPS", "1")
    steps = frozenset(int(part) for part in value.split(",") if part)
    if not steps or min(steps) < 1:
        raise ValueError("PRIME_CUDA_PROFILE_STEPS must contain positive integers")
    return steps


def start_cuda_capture(step: int) -> bool:
    """Start an opt-in Nsight CUDA-profiler range on one global rank."""
    global _CUDA_CAPTURE_ACTIVE, _CUDA_CAPTURE_API
    target = os.environ.get("PRIME_CUDA_PROFILE_RANK", "")
    if os.environ.get("PRIME_CUDA_PROFILE", "0") != "1" or not target:
        return False
    if int(os.environ.get("RANK", "-1")) != int(target):
        return False
    if int(step) not in _cuda_capture_step_set():
        return False
    if _CUDA_CAPTURE_ACTIVE:
        raise RuntimeError("CUDA profiler capture is already active")
    _CUDA_CAPTURE_API = torch.cuda.cudart()
    result = _CUDA_CAPTURE_API.cudaProfilerStart()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStart failed with error {result}")
    torch.cuda.nvtx.range_push(f"prime_nsys_training_step_{int(step)}")
    _CUDA_CAPTURE_ACTIVE = True
    return True


def stop_cuda_capture() -> bool:
    """End the one opt-in Nsight CUDA-profiler range."""
    global _CUDA_CAPTURE_ACTIVE, _CUDA_CAPTURE_API
    if not _CUDA_CAPTURE_ACTIVE:
        return False
    torch.cuda.nvtx.range_pop()
    result = _CUDA_CAPTURE_API.cudaProfilerStop()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStop failed with error {result}")
    _CUDA_CAPTURE_ACTIVE = False
    _CUDA_CAPTURE_API = None
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
