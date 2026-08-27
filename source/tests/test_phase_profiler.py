import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")
from prime_rl.trainer import phase_profiler
from prime_rl.trainer.phase_profiler import (
    _CUDA_MEMORY_CAPTURE_MAX_BLOCKS,
    _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES,
    _CUDA_MEMORY_CAPTURE_MAX_SEGMENTS,
    _CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES,
    PHASES,
    CudaMemoryCapture,
    PhaseProfiler,
    get_cuda_memory_capture,
)


class PhaseProfilerTest(unittest.TestCase):
    def test_cuda_memory_capture_accessor_is_available_at_startup(self):
        self.assertIs(phase_profiler.get_cuda_memory_capture, get_cuda_memory_capture)
        self.assertIs(get_cuda_memory_capture(), get_cuda_memory_capture())
        self.assertIsInstance(get_cuda_memory_capture(), CudaMemoryCapture)

    def test_cuda_memory_capture_is_selected_bounded_and_does_not_sync(self):
        class FakeMemory:
            history_calls = []
            snapshot_calls = 0

            @classmethod
            def _record_memory_history(cls, **kwargs):
                cls.history_calls.append(kwargs)

            @classmethod
            def _snapshot(cls):
                cls.snapshot_calls += 1
                return {
                    "segments": [
                        {
                            "address": index,
                            "total_size": 4096,
                            "allocated_size": 2048,
                            "active_size": 2048,
                            "segment_type": "large",
                            "blocks": [
                                {
                                    "address": index,
                                    "size": 2048,
                                    "requested_size": 1024,
                                    "state": "active_allocated",
                                    "frames": [
                                        {
                                            "filename": "/x/model.py",
                                            "line": index,
                                            "name": "forward",
                                        }
                                    ],
                                }
                            ],
                        }
                        for index in range(_CUDA_MEMORY_CAPTURE_MAX_SEGMENTS + 5)
                    ],
                    "device_traces": [
                        [
                            {
                                "action": "alloc",
                                "addr": index,
                                "size": 2048,
                                "stream": 0,
                                "frames": [],
                            }
                            for index in range(_CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES + 5)
                        ]
                    ],
                }

        def unexpected_sync():
            raise AssertionError("memory capture must not synchronize CUDA")

        fake_cuda = types.SimpleNamespace(memory=FakeMemory, synchronize=unexpected_sync)
        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            phase_profiler.torch, "cuda", fake_cuda, create=True
        ), patch.dict(
            "os.environ",
            {
                "PRIME_CUDA_MEMORY_CAPTURE": "1",
                "PRIME_CUDA_MEMORY_CAPTURE_PATH": f"{output_dir}/capture.json",
                "RANK": "0",
            },
        ):
            capture = CudaMemoryCapture()
            self.assertFalse(capture.begin_step(2))
            self.assertEqual(FakeMemory.history_calls, [])
            self.assertTrue(capture.begin_step(1))
            self.assertTrue(capture.active)
            self.assertEqual(
                FakeMemory.history_calls,
                [
                    {
                        "enabled": "all",
                        "context": "all",
                        "stacks": "python",
                        "max_entries": 4096,
                        "clear_history": True,
                    }
                ],
            )
            self.assertEqual(FakeMemory.snapshot_calls, 0)
            self.assertTrue(capture.end_step())
            self.assertFalse(capture.active)
            self.assertFalse(capture.begin_step(1))
            self.assertEqual(FakeMemory.snapshot_calls, 1)
            self.assertEqual(FakeMemory.history_calls[-1], {"enabled": None})
            record = json.loads(
                (pathlib.Path(output_dir) / "capture.json").read_text()
            )
            self.assertEqual(record["status"], "end")
            self.assertLessEqual(
                (pathlib.Path(output_dir) / "capture.json").stat().st_size,
                _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES,
            )
            compact = record["snapshot"]
            self.assertLessEqual(
                len(compact["largest_segments"]), _CUDA_MEMORY_CAPTURE_MAX_SEGMENTS
            )
            self.assertLessEqual(
                len(compact["largest_active_blocks"]), _CUDA_MEMORY_CAPTURE_MAX_BLOCKS
            )
            self.assertLessEqual(
                len(compact["trace_tail"]), _CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES
            )

    def test_cuda_memory_capture_dumps_only_cuda_oom_or_explicit_end(self):
        class FakeMemory:
            history_calls = []
            snapshot_calls = 0

            @classmethod
            def _record_memory_history(cls, **kwargs):
                cls.history_calls.append(kwargs)

            @classmethod
            def _snapshot(cls):
                cls.snapshot_calls += 1
                return {"segments": [], "device_traces": [[]]}

        class FakeOOM(Exception):
            pass

        fake_cuda = types.SimpleNamespace(memory=FakeMemory)
        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            phase_profiler.torch, "cuda", fake_cuda, create=True
        ), patch.object(
            phase_profiler.torch, "OutOfMemoryError", FakeOOM, create=True
        ), patch.dict(
            "os.environ",
            {
                "PRIME_CUDA_MEMORY_CAPTURE": "1",
                "PRIME_CUDA_MEMORY_CAPTURE_PATH": f"{output_dir}/capture.json",
                "RANK": "0",
            },
        ):
            capture = CudaMemoryCapture()
            self.assertTrue(capture.begin_step(1))
            self.assertFalse(capture.dump_on_oom(RuntimeError("other failure")))
            self.assertTrue(capture.active)
            self.assertEqual(FakeMemory.snapshot_calls, 0)
            self.assertTrue(capture.dump_on_oom(FakeOOM("CUDA out of memory")))
            self.assertFalse(capture.active)
            record = json.loads(
                (pathlib.Path(output_dir) / "capture.json").read_text()
            )
            self.assertEqual(record["status"], "cuda_oom")
            self.assertEqual(record["error"]["message"], "CUDA out of memory")

        with patch.dict(
            "os.environ",
            {"PRIME_CUDA_MEMORY_CAPTURE": "1", "RANK": "1"},
        ):
            capture = CudaMemoryCapture()
            self.assertFalse(capture.begin_step(1))

    def test_cuda_capture_lifecycle_is_single_use_and_step_selected(self):
        class CudaApi:
            def __init__(self):
                self.starts = 0
                self.stops = 0

            def cudaProfilerStart(self):
                self.starts += 1
                return 0

            def cudaProfilerStop(self):
                self.stops += 1
                return 0

        class Nvtx:
            def __init__(self):
                self.events = []

            def range_push(self, name):
                self.events.append(("push", name))

            def range_pop(self):
                self.events.append(("pop", None))

        api = CudaApi()
        nvtx = Nvtx()
        cuda = types.SimpleNamespace(cudart=lambda: api, nvtx=nvtx)
        with patch.object(phase_profiler.torch, "cuda", cuda, create=True), patch.dict(
            "os.environ",
            {
                "PRIME_CUDA_PROFILE": "1",
                "PRIME_CUDA_PROFILE_RANK": "24",
                "PRIME_CUDA_PROFILE_STEPS": "2",
                "RANK": "24",
            },
        ):
            self.assertFalse(phase_profiler.start_cuda_capture(1))
            self.assertTrue(phase_profiler.start_cuda_capture(2))
            self.assertTrue(phase_profiler.stop_cuda_capture())
            self.assertFalse(phase_profiler.stop_cuda_capture())
        self.assertEqual(api.starts, 1)
        self.assertEqual(api.stops, 1)
        self.assertEqual(nvtx.events, [("push", "prime_nsys_training_step_2"), ("pop", None)])

    def test_aggregate_reports_min_median_max_without_category_double_count(self):
        reports = [
            {
                "rank": 0,
                "stage": 0,
                "step_ms": 100.0,
                "categories": {
                    phase: {"exclusive_ms": 10.0 if phase == "optimizer" else 0.0}
                    for phase in PHASES
                },
            },
            {
                "rank": 1,
                "stage": 1,
                "step_ms": 120.0,
                "categories": {
                    phase: {"exclusive_ms": 20.0 if phase == "optimizer" else 0.0}
                    for phase in PHASES
                },
            },
            {
                "rank": 2,
                "stage": 2,
                "step_ms": 110.0,
                "categories": {
                    phase: {"exclusive_ms": 15.0 if phase == "optimizer" else 0.0}
                    for phase in PHASES
                },
            },
        ]

        summary = PhaseProfiler.aggregate(reports)

        self.assertEqual(summary["reports"], 3)
        self.assertEqual(summary["step_ms"], {"min": 100.0, "median": 110.0, "max": 120.0})
        self.assertEqual(
            summary["categories"]["optimizer"],
            {"min_ms": 10.0, "median_ms": 15.0, "max_ms": 20.0},
        )
        self.assertEqual(set(summary["categories"]), set(PHASES))
        self.assertLess(
            PhaseProfiler.modeled_overhead_fraction(
                {phase: 200_000 for phase in PHASES},
                600_000.0,
                sample_every=8,
            ),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
