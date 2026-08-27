import unittest
import importlib.util
import sys
import types
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")
from prime_rl.trainer import phase_profiler
from prime_rl.trainer.phase_profiler import PHASES, PhaseProfiler


class PhaseProfilerTest(unittest.TestCase):
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
