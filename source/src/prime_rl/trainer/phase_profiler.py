"""Opt-in, low-overhead CUDA phase timing for the GLM trainer.

The profiler is inactive unless ``PRIME_PHASE_PROFILE=1``.  It records CUDA
events only on selected steps and resolves them with one device
synchronization at the end of that step.  It does not change execution order,
streams, or tensor values.

The separate Nsight Systems CUDA+NVTX range is inactive unless
``PRIME_CUDA_PROFILE=1``.  It starts on one selected post-warmup step and
stops after a bounded wall-clock duration.
"""

from __future__ import annotations

import os
import re
import statistics
import threading
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
_CUDA_CAPTURE_REQUESTED = os.environ.get("PRIME_CUDA_PROFILE", "0") == "1"
_EVENT_PAIR_COST_US = 20.0


def _parse_cuda_capture_steps(value: str) -> frozenset[int]:
    """Parse one capture step after one warmup step."""
    if re.fullmatch(r"(?:[2-9]|[1-9][0-9]+)", value) is None:
        raise ValueError(
            "PRIME_CUDA_PROFILE_STEPS must be one unsigned integer >= 2 (after warmup)"
        )
    return frozenset({int(value)})


def _parse_cuda_capture_duration(value: str) -> int:
    """Return a strict capture duration in seconds, from 30 through 120."""
    if re.fullmatch(r"(?:[3-9][0-9]|1[01][0-9]|120)", value) is None:
        raise ValueError(
            "PRIME_CUDA_PROFILE_SECONDS must be an integer from 30 through 120"
        )
    return int(value)


class _CudaCaptureController:
    """Control one CUDA-profiler range without changing trainer computation."""

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int = -1,
        target_rank: int = -1,
        steps: frozenset[int] = frozenset(),
        duration_seconds: int = 60,
        cuda_api_factory: Callable[[], Any] | None = None,
        nvtx: Any | None = None,
        timer_factory: Callable[[float, Callable[[], None]], threading.Timer]
        | None = None,
    ) -> None:
        if enabled:
            if target_rank < 0:
                raise ValueError("PRIME_CUDA_PROFILE_RANK must be non-negative")
            if len(steps) != 1 or min(steps) < 2:
                raise ValueError(
                    "PRIME_CUDA_PROFILE_STEPS must contain one step >= 2 (after warmup)"
                )
            duration_seconds = _parse_cuda_capture_duration(str(duration_seconds))
        self.enabled = enabled
        self.rank = rank
        self.target_rank = target_rank
        self.steps = steps
        self.duration_seconds = duration_seconds
        self._cuda_api_factory = cuda_api_factory
        self._nvtx = nvtx
        self._cuda_api = None
        self._active = False
        self._capture_used = False
        self._nvtx_active = False
        self._cuda_stopped = False
        self._timer: threading.Timer | None = None
        self._timer_factory = timer_factory or threading.Timer
        self._timer_error: BaseException | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "_CudaCaptureController":
        enabled = os.environ.get("PRIME_CUDA_PROFILE", "0") == "1"
        if not enabled:
            return cls(enabled=False)
        target_value = os.environ.get("PRIME_CUDA_PROFILE_RANK", "")
        if not target_value:
            raise ValueError(
                "PRIME_CUDA_PROFILE_RANK is required when CUDA profiling is enabled"
            )
        if re.fullmatch(r"[0-9]+", target_value) is None:
            raise ValueError("PRIME_CUDA_PROFILE_RANK must be an unsigned integer")
        try:
            target_rank = int(target_value)
            rank = int(os.environ.get("RANK", "-1"))
        except ValueError as error:
            raise ValueError("CUDA profiler ranks must be integers") from error
        if target_rank < 0:
            raise ValueError("PRIME_CUDA_PROFILE_RANK must be non-negative")
        steps = _parse_cuda_capture_steps(
            os.environ.get("PRIME_CUDA_PROFILE_STEPS", "2")
        )
        duration_seconds = _parse_cuda_capture_duration(
            os.environ.get("PRIME_CUDA_PROFILE_SECONDS", "60")
        )
        return cls(
            enabled=True,
            rank=rank,
            target_rank=target_rank,
            steps=steps,
            duration_seconds=duration_seconds,
            cuda_api_factory=torch.cuda.cudart,
            nvtx=torch.cuda.nvtx,
        )

    @property
    def active(self) -> bool:
        return self._active

    @contextmanager
    def step_scope(self, step: int) -> Iterator[None]:
        """Stop and clean up a selected step on normal or error exit."""
        self.start(step)
        try:
            yield
        finally:
            self.stop()

    def start(self, step: int) -> bool:
        """Start the single selected range on the configured global rank."""
        if not self.enabled or self.rank != self.target_rank:
            return False
        if self._capture_used or int(step) not in self.steps:
            return False
        if self._active or self._nvtx_active:
            raise RuntimeError("CUDA profiler capture is already active")
        if self._cuda_api_factory is None or self._nvtx is None:
            raise RuntimeError("CUDA profiler capture dependencies are unavailable")
        api = self._cuda_api_factory()
        result = api.cudaProfilerStart()
        if result not in (None, 0):
            raise RuntimeError(f"cudaProfilerStart failed with error {result}")
        try:
            self._nvtx.range_push(f"prime_nsys_training_step_{int(step)}")
        except BaseException:
            # Do not leave the CUDA profiler running if the NVTX marker failed.
            api.cudaProfilerStop()
            raise
        self._cuda_api = api
        self._active = True
        self._nvtx_active = True
        self._cuda_stopped = False
        self._timer_error = None
        self._capture_used = True
        timer = self._timer_factory(self.duration_seconds, self._timer_stop)
        timer.daemon = True
        self._timer = timer
        try:
            timer.start()
        except BaseException:
            self._timer = None
            self._active = False
            try:
                self._nvtx.range_pop()
            finally:
                self._nvtx_active = False
                api.cudaProfilerStop()
            raise
        return True

    def _timer_stop(self) -> None:
        """Stop collection from the timer; NVTX is popped by the owner thread."""
        with self._lock:
            if not self._active or self._cuda_stopped:
                return
            api = self._cuda_api
            try:
                result = api.cudaProfilerStop() if api is not None else 1
                if result not in (None, 0):
                    raise RuntimeError(f"cudaProfilerStop failed with error {result}")
            except BaseException as error:
                self._timer_error = error
            finally:
                self._cuda_stopped = True
                self._active = False

    def stop(self) -> bool:
        """Cancel and join the timer, then stop and close the range exactly once."""
        timer = self._timer
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
            timer.join()
        with self._lock:
            if not self._active and not self._nvtx_active:
                return False
            api = self._cuda_api
            if api is None or self._nvtx is None:
                raise RuntimeError("CUDA profiler capture state is incomplete")
            error = self._timer_error
            self._timer = None
            self._timer_error = None
            try:
                if self._nvtx_active:
                    self._nvtx.range_pop()
                    self._nvtx_active = False
            except BaseException as caught:
                error = error or caught
            try:
                if not self._cuda_stopped:
                    result = api.cudaProfilerStop()
                    if result not in (None, 0):
                        raise RuntimeError(f"cudaProfilerStop failed with error {result}")
            except BaseException as caught:
                error = error or caught
            finally:
                self._cuda_stopped = True
                self._cuda_api = None
                self._active = False
                self._nvtx_active = False
        if error is not None:
            raise error
        return True


_CUDA_CAPTURE_CONTROLLER: _CudaCaptureController | None = None


def _cuda_capture_controller() -> _CudaCaptureController:
    global _CUDA_CAPTURE_CONTROLLER
    if _CUDA_CAPTURE_CONTROLLER is None:
        _CUDA_CAPTURE_CONTROLLER = _CudaCaptureController.from_environment()
    return _CUDA_CAPTURE_CONTROLLER


def start_cuda_capture(step: int) -> bool:
    """Start the opt-in Nsight range on one rank after the warmup step."""
    if not _CUDA_CAPTURE_REQUESTED:
        return False
    return _cuda_capture_controller().start(step)


def stop_cuda_capture() -> bool:
    """Retain the old API for callers that need an explicit emergency stop."""
    if not _CUDA_CAPTURE_REQUESTED:
        return False
    return _cuda_capture_controller().stop()


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
