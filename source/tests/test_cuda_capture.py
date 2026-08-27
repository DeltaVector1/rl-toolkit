import unittest
import importlib.util
import importlib.machinery
import sys
import types

if importlib.util.find_spec("torch") is None:
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    sys.modules["torch"] = fake_torch

from prime_rl.trainer.phase_profiler import (
    _CudaCaptureController,
    _parse_cuda_capture_microbatches,
    _parse_cuda_capture_steps,
)


class _FakeCudaApi:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def cudaProfilerStart(self):
        self.starts += 1
        return 0

    def cudaProfilerStop(self):
        self.stops += 1
        return 0


class _FakeNvtx:
    def __init__(self) -> None:
        self.pushed = []
        self.popped = 0

    def range_push(self, value):
        self.pushed.append(value)

    def range_pop(self):
        self.popped += 1


class CudaCaptureTest(unittest.TestCase):
    def test_capture_starts_and_stops_once_at_three_unique_microbatches(self):
        api = _FakeCudaApi()
        nvtx = _FakeNvtx()
        controller = _CudaCaptureController(
            enabled=True,
            rank=24,
            target_rank=24,
            steps=frozenset({2}),
            microbatch_limit=3,
            cuda_api_factory=lambda: api,
            nvtx=nvtx,
        )

        self.assertTrue(controller.start(2))
        self.assertFalse(controller.microbatch_complete(0))
        self.assertFalse(controller.microbatch_complete(1))
        self.assertTrue(controller.microbatch_complete(2))
        self.assertFalse(controller.microbatch_complete(2))
        self.assertFalse(controller.stop())
        self.assertEqual(api.starts, 1)
        self.assertEqual(api.stops, 1)
        self.assertEqual(nvtx.pushed, ["prime_nsys_training_step_2"])
        self.assertEqual(nvtx.popped, 1)

    def test_capture_limit_and_step_fail_closed(self):
        for value in ("0", "1", "5", "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parse_cuda_capture_microbatches(value)
        for value in ("", "0", "1", "1,2", "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parse_cuda_capture_steps(value)


if __name__ == "__main__":
    unittest.main()
