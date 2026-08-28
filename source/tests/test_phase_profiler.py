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
                "PRIME_CUDA_MEMORY_CAPTURE_RANK": "3",
                "PRIME_CUDA_MEMORY_CAPTURE_STEP": "7",
                "PRIME_CUDA_MEMORY_CAPTURE_STACKS": "1",
                "RANK": "3",
            },
        ):
            capture = CudaMemoryCapture()
            self.assertTrue(capture.begin_step(7))
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
            class FakeOOM(Exception):
                pass

            with patch.object(phase_profiler.torch, "OutOfMemoryError", FakeOOM, create=True):
                self.assertTrue(capture.dump_on_oom(FakeOOM("CUDA out of memory")))
            self.assertFalse(capture.active)
            self.assertEqual(FakeMemory.snapshot_calls, 1)
            self.assertEqual(FakeMemory.history_calls[-1], {"enabled": None})
            artifact = pathlib.Path(output_dir) / "capture.rank_3.json"
            record = json.loads(
                artifact.read_text()
            )
            self.assertEqual(record["status"], "cuda_oom")
            self.assertEqual(record["rank"], 3)
            self.assertEqual(record["step"], 7)
            self.assertLessEqual(
                artifact.stat().st_size,
                _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES,
            )
            compact = record["allocator"]["snapshot"]
            self.assertLessEqual(
                len(compact["largest_segments"]), _CUDA_MEMORY_CAPTURE_MAX_SEGMENTS
            )
            self.assertLessEqual(
                len(compact["largest_active_blocks"]), _CUDA_MEMORY_CAPTURE_MAX_BLOCKS
            )
            self.assertLessEqual(
                len(compact["trace_tail"]), _CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES
            )

    def test_cuda_memory_capture_writes_non_rank0_oom_with_lightweight_stats(self):
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

        class FakeCuda:
            memory = FakeMemory

            @staticmethod
            def memory_stats():
                return {
                    "allocated_bytes.all.current": 123,
                    "reserved_bytes.all.current": 456,
                    "num_ooms": 1,
                }

            @staticmethod
            def memory_allocated():
                return 123

            @staticmethod
            def memory_reserved():
                return 456

            @staticmethod
            def max_memory_allocated():
                return 789

            @staticmethod
            def max_memory_reserved():
                return 987

            @staticmethod
            def mem_get_info():
                return 1000, 2000

            @staticmethod
            def synchronize():
                raise AssertionError("memory diagnostics must not synchronize CUDA")

        class FakeOOM(Exception):
            pass

        fake_cuda = FakeCuda
        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            phase_profiler.torch, "cuda", fake_cuda, create=True
        ), patch.object(
            phase_profiler.torch, "OutOfMemoryError", FakeOOM, create=True
        ), patch.dict(
            "os.environ",
            {
                "PRIME_CUDA_MEMORY_CAPTURE": "1",
                "PRIME_CUDA_MEMORY_CAPTURE_PATH": f"{output_dir}/capture.json",
                "PRIME_CUDA_MEMORY_CAPTURE_RANK": "0",
                "PRIME_CUDA_MEMORY_CAPTURE_STEP": "23",
                "RANK": "17",
            },
        ):
            capture = CudaMemoryCapture()
            self.assertTrue(capture.begin_step(23))
            self.assertFalse(capture.dump_on_oom(RuntimeError("other failure")))
            self.assertTrue(capture.active)
            self.assertEqual(FakeMemory.snapshot_calls, 0)
            self.assertTrue(capture.dump_on_oom(FakeOOM("CUDA out of memory")))
            self.assertFalse(capture.active)
            artifact = pathlib.Path(output_dir) / "capture.rank_17.json"
            record = json.loads(
                artifact.read_text()
            )
            self.assertEqual(record["status"], "cuda_oom")
            self.assertEqual(record["rank"], 17)
            self.assertEqual(record["step"], 23)
            self.assertEqual(record["error"]["message"], "CUDA out of memory")
            self.assertFalse(record["allocator"]["history_enabled"])
            self.assertEqual(record["memory"]["num_ooms"], 1)
            self.assertEqual(FakeMemory.snapshot_calls, 0)

    def test_cuda_memory_capture_end_step_does_not_write_success_artifact(self):
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

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            phase_profiler.torch,
            "cuda",
            types.SimpleNamespace(memory=FakeMemory),
            create=True,
        ), patch.dict(
            "os.environ",
            {
                "PRIME_CUDA_MEMORY_CAPTURE": "1",
                "PRIME_CUDA_MEMORY_CAPTURE_PATH": f"{output_dir}/capture.json",
                "PRIME_CUDA_MEMORY_CAPTURE_RANK": "0",
                "PRIME_CUDA_MEMORY_CAPTURE_STEP": "4",
                "RANK": "0",
            },
        ):
            capture = CudaMemoryCapture()
            self.assertTrue(capture.begin_step(4))
            self.assertTrue(capture.end_step())
            self.assertFalse((pathlib.Path(output_dir) / "capture.rank_0.json").exists())
            self.assertEqual(FakeMemory.snapshot_calls, 0)
            self.assertEqual(FakeMemory.history_calls[-1], {"enabled": None})

    def test_cuda_memory_capture_nested_snapshot_is_bounded(self):
        # This models a compact allocator summary that is still larger than
        # the per-rank limit.  The real summary is nested under allocator.
        oversized_snapshot = {
            "segment_count": 1,
            "active_block_count": 1,
            "trace_count": 1,
            "largest_segments": [],
            "largest_active_blocks": [],
            "trace_tail": ["x" * 1024 for _ in range(280)],
        }
        record = {
            "format": "prime-cuda-memory-capture-v2",
            "rank": 17,
            "step": 23,
            "status": "cuda_oom",
            "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"},
            "memory": {"num_ooms": 1},
            "nvml": {"available": False},
            "allocator": {"history_enabled": True, "snapshot": oversized_snapshot},
        }
        payload = CudaMemoryCapture._bounded_payload(record)
        self.assertLessEqual(len(payload), _CUDA_MEMORY_CAPTURE_MAX_OUTPUT_BYTES)
        reduced = json.loads(payload)
        self.assertTrue(reduced["truncated"])
        retained = reduced["allocator"].get("snapshot")
        if retained is not None:
            self.assertLessEqual(
                len(retained["trace_tail"]), _CUDA_MEMORY_CAPTURE_MAX_TRACE_ENTRIES
            )

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

    def test_phase_records_are_disabled_without_a_qualification_path(self):
        with patch.dict("os.environ", {}, clear=True):
            profiler = PhaseProfiler()
        report = {"rank": 4, "stage": 2, "step": 3}
        self.assertFalse(profiler.write_report(report))

    def test_phase_record_path_requires_qualification(self):
        with tempfile.TemporaryDirectory() as output_dir, patch.dict(
            "os.environ",
            {"PRIME_PHASE_PROFILE_RECORD_PATH": f"{output_dir}/phase.json"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "qualification profile"):
                PhaseProfiler()

    def test_phase_record_is_atomic_rank_stage_file_and_bounded(self):
        report = {
            "rank": 17,
            "stage": 5,
            "step": 23,
            "step_ms": 123.5,
            "sample_every": 8,
            "classified_sum_ms": 100.0,
            "modeled_overhead_fraction": 0.01,
            "categories": {
                phase: {
                    "calls": 4,
                    "sampled": 1,
                    "estimated": True,
                    "inclusive_ms": 10.0,
                    "exclusive_ms": 9.0,
                }
                for phase in PHASES
            },
        }
        with tempfile.TemporaryDirectory() as output_dir, patch.dict(
            "os.environ",
            {
                "PRIME_QUALIFICATION_PROFILE": "glm52-native-b2048-nsys",
                "PRIME_PHASE_PROFILE_RECORD_PATH": f"{output_dir}/phase.json",
            },
            clear=True,
        ):
            profiler = PhaseProfiler()
            self.assertTrue(profiler.write_report(report))
            artifact = pathlib.Path(output_dir) / "phase.rank_17.step_23.json"
            self.assertTrue(artifact.is_file())
            self.assertFalse(list(pathlib.Path(output_dir).glob("*.tmp")))
            self.assertLessEqual(
                artifact.stat().st_size,
                phase_profiler._PHASE_PROFILE_MAX_OUTPUT_BYTES,
            )
            record = json.loads(artifact.read_text())
            self.assertEqual(
                {record[key] for key in ("rank", "stage", "step")},
                {17, 5, 23},
            )
            self.assertEqual(set(record["categories"]), set(PHASES))

    def test_phase_record_payload_remains_bounded_for_untrusted_values(self):
        oversized_integer = 10**5000
        report = {
            "rank": oversized_integer,
            "stage": oversized_integer,
            "step": oversized_integer,
            "step_ms": 10**309,
            "classified_sum_ms": -(10**309),
            "modeled_overhead_fraction": float("inf"),
            "categories": {
                phase: {
                    "calls": oversized_integer,
                    "sampled": oversized_integer,
                    "inclusive_ms": "x" * 10**7,
                    "exclusive_ms": float("nan"),
                }
                for phase in PHASES
            },
        }
        payload = PhaseProfiler._bounded_record_payload(report)
        self.assertLessEqual(
            len(payload), phase_profiler._PHASE_PROFILE_MAX_OUTPUT_BYTES
        )
        record = json.loads(payload)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["rank"], phase_profiler._PHASE_PROFILE_MAX_INTEGER)
        self.assertEqual(record["stage"], phase_profiler._PHASE_PROFILE_MAX_INTEGER)
        self.assertEqual(record["step"], phase_profiler._PHASE_PROFILE_MAX_INTEGER)
        self.assertEqual(record["step_ms"], 0.0)
        self.assertEqual(record["classified_sum_ms"], 0.0)
        self.assertEqual(record["modeled_overhead_fraction"], 0.0)
        self.assertEqual(
            record["categories"]["pp_wait"]["calls"],
            phase_profiler._PHASE_PROFILE_MAX_INTEGER,
        )
        self.assertEqual(record["categories"]["pp_wait"]["inclusive_ms"], 0.0)
        self.assertEqual(record["categories"]["pp_wait"]["exclusive_ms"], 0.0)
        json.dumps(record, allow_nan=False)

    def test_phase_record_payload_uses_valid_minimal_fallback_on_encoding_error(self):
        class EncodingFailure(Exception):
            pass

        with tempfile.TemporaryDirectory() as output_dir, patch.dict(
            "os.environ",
            {
                "PRIME_QUALIFICATION_PROFILE": "glm52-native-b2048-nsys",
                "PRIME_PHASE_PROFILE_RECORD_PATH": f"{output_dir}/phase.json",
            },
            clear=True,
        ):
            profiler = PhaseProfiler()
            with patch.object(
                phase_profiler.json,
                "dumps",
                side_effect=EncodingFailure("injected encoding error"),
            ):
                self.assertTrue(
                    profiler.write_report({"rank": 1, "stage": 2, "step": 3})
                )
            artifact = pathlib.Path(output_dir) / "phase.rank_1.step_3.json"
            self.assertTrue(artifact.is_file())
            payload = artifact.read_bytes()

        self.assertEqual(payload, phase_profiler._PHASE_PROFILE_MINIMAL_PAYLOAD)
        self.assertLessEqual(
            len(payload), phase_profiler._PHASE_PROFILE_MAX_OUTPUT_BYTES
        )
        self.assertEqual(
            json.loads(payload),
            {"format": "prime-phase-profile-v1", "status": "truncated"},
        )


if __name__ == "__main__":
    unittest.main()
