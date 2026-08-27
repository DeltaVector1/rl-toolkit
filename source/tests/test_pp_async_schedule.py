"""CPU/fake-work checks for the opt-in asynchronous 1F1B transport."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


SOURCE = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE / "src/prime_rl/trainer/distributed/async_schedule.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "async_schedule_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeOp:
    def __init__(self, kind: str, microbatch: int, tensor=None):
        self.kind = kind
        self.microbatch = microbatch
        self.tensor = tensor


class FakeTransport:
    def __init__(self, module, log):
        self.module = module
        self.log = log
        self.state = module.AsyncP2PState(
            self._post_receive,
            self._post_send,
            self._wait_receive,
            self._wait_send,
        )

    def receive_window(self, _direction):
        return 2

    def _post_receive(self, ops):
        token = tuple((op.kind, op.microbatch) for op in ops)
        self.log.append(("post_recv", token))
        return token

    def _post_send(self, ops):
        token = tuple((op.kind, op.microbatch) for op in ops)
        self.log.append(("post_send", token))
        return token

    def _wait_receive(self, token):
        self.log.append(("wait_recv", token))

    def _wait_send(self, token):
        self.log.append(("wait_send", token))

    def prepost_receive(self, direction, microbatch, ops, *, buffer=None):
        self.state.prepost_receive(direction, microbatch, ops, buffer=buffer)

    def wait_receive(self, direction, microbatch):
        self.state.wait_receive(direction, microbatch)

    def release_receive(self, direction, microbatch):
        self.state.release_receive(direction, microbatch)

    def post_send(self, direction, microbatch, ops):
        self.state.post_send(direction, microbatch, ops)

    def drain(self):
        self.state.drain()

    def cleanup(self):
        self.log.append(("cleanup",))
        self.state.cleanup()


class ReferenceTransport:
    """No-op transport with the same driver-facing interface."""

    def __init__(self):
        self.received = set()

    def prepost_receive(self, direction, microbatch, _ops, *, buffer=None):
        self.received.add((direction, microbatch))

    def wait_receive(self, direction, microbatch):
        assert (direction, microbatch) in self.received

    def release_receive(self, direction, microbatch):
        self.received.remove((direction, microbatch))

    def post_send(self, _direction, _microbatch, _ops):
        pass

    def drain(self):
        assert not self.received

    def cleanup(self):
        self.received.clear()


class FakeStage:
    stage_index = 1
    is_first = False
    is_last = False

    def __init__(self, log, fail_forward=None):
        self.log = log
        self.fail_forward = fail_forward
        self.outputs = {}
        self.gradients = []
        self._fwd_buffers = [object() for _ in range(8)]
        self._bwd_buffers = [object() for _ in range(8)]

    def get_fwd_recv_ops(self, microbatch):
        return [FakeOp("fwd_recv", microbatch, self._fwd_buffers[microbatch])]

    def get_bwd_recv_ops(self, microbatch):
        return [FakeOp("bwd_recv", microbatch, self._bwd_buffers[microbatch])]

    def forward_one_chunk(self, microbatch, args, _kwargs, save_forward_output=True):
        self.log.append(("forward", microbatch))
        if microbatch == self.fail_forward:
            raise RuntimeError("fake forward failure")
        value = args * 3 + 1
        self.outputs[microbatch] = value
        return value

    def get_fwd_send_ops(self, microbatch):
        return [FakeOp("fwd_send", microbatch, self.outputs[microbatch])]

    def backward_one_chunk(self, microbatch, loss=None, last_backward=False):
        self.log.append(("backward", microbatch))
        self.gradients.append((microbatch, loss, last_backward))

    def get_bwd_send_ops(self, microbatch):
        return [FakeOp("bwd_send", microbatch, self.gradients[-1])]

    def perform_reduce_grad(self, scale):
        self.log.append(("reduce", scale))


class FakeSchedule:
    _n_microbatches = 5
    _num_microbatches = 5
    _num_stages = 4
    scale_grads = False

    def __init__(self, stage):
        self._stage = stage
        self.losses = []
        self.updated_losses = None

    def _maybe_compute_loss(self, _stage, output, target_mbs, microbatch):
        self.losses.append(output + target_mbs[microbatch])

    def _maybe_get_loss(self, _stage, microbatch):
        return self.losses[microbatch]

    def _update_losses(self, _stage, losses):
        self.updated_losses = list(self.losses)
        if losses is not None:
            losses.extend(self.losses)


class AsyncP2PStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_window_order_lifetime_and_exact_memory(self):
        log = []
        transport = FakeTransport(self.module, log)
        first_buffer = object()
        transport.prepost_receive("fwd", 0, [FakeOp("recv", 0, first_buffer)])
        transport.prepost_receive("fwd", 1, [FakeOp("recv", 1, object())])
        with self.assertRaisesRegex(RuntimeError, "window exceeded"):
            transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        transport.wait_receive("fwd", 0)
        with self.assertRaisesRegex(RuntimeError, "still owned"):
            transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        transport.release_receive("fwd", 0)
        transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        self.assertEqual(transport.state.max_pending_receives["fwd"], 2)
        self.assertEqual(
            log[:2], [("post_recv", (("recv", 0),)), ("post_recv", (("recv", 1),))]
        )
        self.assertEqual(self.module.estimate_extra_buffer_bytes((1, 4096, 6144), 2), 0)
        self.assertEqual(
            self.module.estimate_extra_buffer_bytes(
                (1, 4096, 6144),
                2,
                sparse_shape=(1, 4096, 2048),
                sparse_element_size=2,
                slots_per_direction=2,
            ),
            268435456,
        )
        transport.cleanup()

    def test_async_driver_matches_1f1b_values_and_gradients(self):
        async_log = []
        async_stage = FakeStage(async_log)
        async_schedule = FakeSchedule(async_stage)
        async_transport = FakeTransport(self.module, async_log)
        targets = [10, 20, 30, 40, 50]
        self.module.run_1f1b_async_step(
            async_schedule,
            async_transport,
            list(range(5)),
            [{} for _ in range(5)],
            targets,
            [],
            True,
        )

        reference_stage = FakeStage([])
        reference_schedule = FakeSchedule(reference_stage)
        reference_transport = ReferenceTransport()
        self.module.run_1f1b_async_step(
            reference_schedule,
            reference_transport,
            list(range(5)),
            [{} for _ in range(5)],
            targets,
            [],
            True,
        )
        self.assertEqual(async_stage.outputs, reference_stage.outputs)
        self.assertEqual(async_stage.gradients, reference_stage.gradients)
        self.assertEqual(
            async_schedule.updated_losses, reference_schedule.updated_losses
        )
        self.assertEqual(
            [item for item in async_log if item[0] in ("forward", "backward")],
            [
                ("forward", 0),
                ("forward", 1),
                ("forward", 2),
                ("backward", 0),
                ("forward", 3),
                ("backward", 1),
                ("forward", 4),
                ("backward", 2),
                ("backward", 3),
                ("backward", 4),
            ],
        )
        self.assertLess(
            async_log.index(("wait_recv", (("fwd_recv", 0),))),
            async_log.index(("forward", 0)),
        )
        self.assertLess(
            async_log.index(("post_recv", (("fwd_recv", 1),))),
            async_log.index(("forward", 0)),
        )
        self.assertEqual(
            async_transport.state.max_pending_receives, {"fwd": 2, "bwd": 2}
        )
        self.assertLessEqual(async_transport.state.max_pending_sends, 2)

    def test_error_cleans_all_outstanding_work(self):
        log = []
        stage = FakeStage(log, fail_forward=1)
        schedule = FakeSchedule(stage)
        transport = FakeTransport(self.module, log)
        with self.assertRaisesRegex(RuntimeError, "fake forward failure"):
            self.module.run_1f1b_async_step(
                schedule,
                transport,
                list(range(5)),
                [{} for _ in range(5)],
                [0] * 5,
                [],
                True,
            )
        self.assertIn(("cleanup",), log)
        self.assertFalse(transport.state._receives["fwd"])
        self.assertFalse(transport.state._receives["bwd"])
        self.assertFalse(transport.state._sends)

    def test_schedule_name_is_explicitly_opt_in(self):
        runner_source = (
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py"
        ).read_text()
        config_source = (
            SOURCE / "packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
        ).read_text()
        self.assertIn('schedule_name == "1f1b_async"', runner_source)
        self.assertIn('"1f1b_async"', config_source)
        self.assertNotIn('pp_schedule = "1f1b_async"', runner_source)

    def test_cuda_transport_uses_explicit_stream_dependencies(self):
        source = MODULE_PATH.read_text()
        self.assertIn("torch.cuda.Stream(device=stage.device)", source)
        self.assertIn("with self._torch.cuda.stream(self._stream)", source)
        self.assertIn("self._stream.wait_event(ready)", source)
        self.assertIn("current_stream().wait_event(pending.event)", source)


if __name__ == "__main__":
    unittest.main()

