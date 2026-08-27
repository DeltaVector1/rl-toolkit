import importlib.machinery
import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    sys.modules["torch"] = fake_torch

from prime_rl.trainer.phase_profiler import (
    _CudaCaptureController,
    _parse_cuda_capture_duration,
    _parse_cuda_capture_steps,
)


class _FakeCudaApi:
    def __init__(self, stop_result=0):
        self.starts = 0
        self.stops = 0
        self.stop_result = stop_result

    def cudaProfilerStart(self):
        self.starts += 1
        return 0

    def cudaProfilerStop(self):
        self.stops += 1
        return self.stop_result


class _StartErrorCudaApi(_FakeCudaApi):
    def cudaProfilerStart(self):
        self.starts += 1
        return 1


class _FakeNvtx:
    def __init__(self):
        self.pushed = []
        self.popped = 0

    def range_push(self, value):
        self.pushed.append(value)

    def range_pop(self):
        self.popped += 1


class _FakeTimer:
    def __init__(self, duration, callback):
        self.duration = duration
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.joined = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def join(self):
        self.joined = True

    def fire(self):
        self.callback()


class CudaCaptureTest(unittest.TestCase):
    def _controller(self, api=None, nvtx=None, duration=60):
        timers = []

        def timer_factory(seconds, callback):
            timer = _FakeTimer(seconds, callback)
            timers.append(timer)
            return timer

        controller = _CudaCaptureController(
            enabled=True,
            rank=24,
            target_rank=24,
            steps=frozenset({2}),
            duration_seconds=duration,
            cuda_api_factory=lambda: api or _FakeCudaApi(),
            nvtx=nvtx or _FakeNvtx(),
            timer_factory=timer_factory,
        )
        return controller, timers

    def test_capture_starts_and_stops_once(self):
        api = _FakeCudaApi()
        nvtx = _FakeNvtx()
        controller, timers = self._controller(api, nvtx)

        self.assertTrue(controller.start(2))
        self.assertEqual(timers[0].duration, 60)
        timers[0].fire()
        self.assertFalse(controller.active)
        self.assertTrue(controller.stop())
        self.assertFalse(controller.stop())
        self.assertEqual(api.starts, 1)
        self.assertEqual(api.stops, 1)
        self.assertEqual(nvtx.pushed, ["prime_nsys_training_step_2"])
        self.assertEqual(nvtx.popped, 1)

    def test_strict_step_and_duration_parsers(self):
        for value in ("", " 2", "2 ", "+2", "0", "1", "1,2", "2,", ",2"):
            with self.subTest(kind="step", value=value):
                with self.assertRaises(ValueError):
                    _parse_cuda_capture_steps(value)
        for value in ("", " 30", "30 ", "+30", "29", "121", "30,60"):
            with self.subTest(kind="duration", value=value):
                with self.assertRaises(ValueError):
                    _parse_cuda_capture_duration(value)

    def test_wrong_rank_and_step_do_not_call_cuda_api(self):
        api = _FakeCudaApi()
        controller = _CudaCaptureController(
            enabled=True,
            rank=23,
            target_rank=24,
            steps=frozenset({2}),
            duration_seconds=60,
            cuda_api_factory=lambda: api,
            nvtx=_FakeNvtx(),
        )
        self.assertFalse(controller.start(2))
        controller.rank = 24
        self.assertFalse(controller.start(1))
        self.assertEqual(api.starts, 0)

    def test_timer_stop_api_error_is_cleaned_up(self):
        api = _FakeCudaApi(stop_result=1)
        nvtx = _FakeNvtx()
        controller, timers = self._controller(api, nvtx)
        controller.start(2)
        timers[0].fire()
        with self.assertRaises(RuntimeError):
            controller.stop()
        self.assertFalse(controller.active)
        self.assertEqual(api.stops, 1)
        self.assertEqual(nvtx.popped, 1)

    def test_start_api_error_does_not_start_timer(self):
        api = _StartErrorCudaApi()
        controller, timers = self._controller(api)
        with self.assertRaises(RuntimeError):
            controller.start(2)
        self.assertEqual(timers, [])
        self.assertFalse(controller.active)
        self.assertFalse(controller.stop())

    def test_step_scope_stops_on_short_batch_and_error(self):
        for error in (False, True):
            api = _FakeCudaApi()
            controller, timers = self._controller(api)
            with (self.assertRaises(RuntimeError) if error else nullcontext()):
                with controller.step_scope(2):
                    if error:
                        raise RuntimeError("test error")
            self.assertFalse(controller.active)
            self.assertTrue(timers[0].cancelled)
            self.assertTrue(timers[0].joined)
            self.assertEqual(api.stops, 1)

    def test_launcher_uses_bounded_timer_capture(self):
        source = Path(__file__).parents[2] / "scripts/china/cluster.sh"
        text = source.read_text()
        for argument in (
            "PRIME_CUDA_PROFILE_SECONDS",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--event-sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--flush-on-cudaprofilerstop=true",
            "--kill=none",
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, text)
        self.assertIn(
            'echo "Nsight disabled on n$node; capture owner is n78 rank 24"',
            text,
        )
        self.assertIn('[ "$capture_rank" = "24" ]', text)

    def test_capture_has_no_schedule_or_microbatch_control(self):
        profiler = Path(__file__).parents[1] / "src/prime_rl/trainer/phase_profiler.py"
        runner = Path(__file__).parents[1] / "src/prime_rl/trainer/distributed/pp_runner.py"
        train = Path(__file__).parents[1] / "src/prime_rl/trainer/rl/train.py"
        self.assertNotIn("microbatch_complete", profiler.read_text())
        self.assertNotIn("prime_cuda_capture", runner.read_text())
        self.assertNotIn("cuda_capture_step", train.read_text())


if __name__ == "__main__":
    unittest.main()
