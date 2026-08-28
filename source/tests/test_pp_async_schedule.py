"""CPU/fake-work checks for the opt-in asynchronous 1F1B transport."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


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
    def __init__(self, kind: str, microbatch: int, tensor=None, peer=None):
        self.kind = kind
        self.microbatch = microbatch
        self.tensor = tensor
        self.peer = peer


class FakeCuda:
    def __init__(self, log):
        self.log = log
        self.compute_stream = FakeComputeStream(log)
        self.streams = []
        self.events = []

    def Stream(self, *, device):
        stream = FakeCudaStream(self.log, device, len(self.streams))
        self.streams.append(stream)
        return stream

    def Event(self):
        event = FakeCudaEvent(self.log, len(self.events))
        self.events.append(event)
        return event

    def current_stream(self):
        return self.compute_stream

    @contextmanager
    def stream(self, stream):
        self.log.append(("stream_enter", stream.index))
        yield
        self.log.append(("stream_exit", stream.index))


class FakeCudaStream:
    def __init__(self, log, device, index):
        self.log = log
        self.device = device
        self.index = index
        self.log.append(("stream_create", index, device))

    def wait_event(self, event):
        self.log.append(("stream_wait_event", self.index, event.index))

    def synchronize(self):
        self.log.append(("stream_sync", self.index))


class FakeComputeStream:
    index = -1

    def __init__(self, log):
        self.log = log

    def wait_event(self, event):
        self.log.append(("stream_wait_event", self.index, event.index))


class FakeCudaEvent:
    def __init__(self, log, index):
        self.log = log
        self.index = index
        self.log.append(("event_create", index))

    def record(self, stream):
        self.log.append(("event_record", self.index, stream))


class FakeCudaWork:
    def __init__(self, log, index):
        self.log = log
        self.index = index

    def wait(self):
        self.log.append(("work_wait", self.index))


class FakeCudaDist(types.ModuleType):
    def __init__(self, log):
        super().__init__("torch.distributed")
        self.log = log
        self.calls = 0

    def batch_isend_irecv(self, ops):
        self.log.append(("batch_isend_irecv", self.calls, tuple(ops)))
        work = FakeCudaWork(self.log, self.calls)
        self.calls += 1
        return [work]


def make_cuda_transport(module, log):
    cuda = FakeCuda(log)
    dist = FakeCudaDist(log)
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.cuda = cuda
    torch.distributed = dist
    stage = types.SimpleNamespace(
        device="cuda:0",
        receive_sparse_indices=False,
        _forward_slot_count=2,
        _backward_slot_count=2,
    )
    modules = {"torch": torch, "torch.distributed": dist}
    with mock.patch.dict(sys.modules, modules):
        transport = module.CudaAsyncP2P(stage)
    return transport, cuda


class FakeTransport:
    def __init__(self, module, log, *, fwd_window=2, bwd_window=2):
        self.module = module
        self.log = log
        self.state = module.AsyncP2PState(
            self._post_receive,
            self._post_send,
            self._wait_receive,
            self._wait_send,
            max_window={"fwd": fwd_window, "bwd": bwd_window},
        )
        self.fwd_window = fwd_window
        self.bwd_window = bwd_window
        self.groups = []
        self.max_groups = 0

    def receive_window(self, direction):
        return self.fwd_window if direction == "fwd" else self.bwd_window

    def receive_capacity_available(self, direction):
        return self.state.receive_capacity_available(direction)

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
        return self.state.prepost_receive(direction, microbatch, ops, buffer=buffer)

    def wait_receive(self, direction, microbatch):
        self.state.wait_receive(direction, microbatch)

    def release_receive(self, direction, microbatch):
        self.state.release_receive(direction, microbatch)

    def release_receive_buffer(self, direction, microbatch, buffer_index):
        self.state.release_receive_buffer(direction, microbatch, buffer_index)

    def post_send(self, direction, microbatch, ops):
        self.state.post_send(direction, microbatch, ops)

    def post_group(self, description, ops):
        op_list = tuple(ops)
        if not op_list:
            return None
        if len(self.groups) >= self.module.MAX_P2P_WINDOW:
            raise RuntimeError("fake grouped P2P bound exceeded")
        token = (description, op_list)
        self.groups.append(token)
        self.max_groups = max(self.max_groups, len(self.groups))
        self.log.append(
            (
                "post_group",
                description,
                tuple((op.kind, op.microbatch) for op in op_list),
            )
        )
        return token

    def wait_group(self, token):
        if token is None:
            return
        self.groups.remove(token)
        self.log.append(("wait_group", token[0]))

    def drain(self):
        assert not self.groups
        self.state.drain()

    def cleanup(self):
        self.log.append(("cleanup",))
        self.groups.clear()
        self.state.cleanup()


class ReferenceTransport:
    """No-op transport with the same driver-facing interface."""

    def __init__(self):
        self.received = set()

    def prepost_receive(self, direction, microbatch, _ops, *, buffer=None):
        self.received.add((direction, microbatch))
        return True

    def wait_receive(self, direction, microbatch):
        assert (direction, microbatch) in self.received

    def release_receive(self, direction, microbatch):
        self.received.remove((direction, microbatch))

    def release_receive_buffer(self, direction, microbatch, _buffer_index):
        # Reference transport has no buffer ownership model.
        del direction, microbatch

    def post_send(self, _direction, _microbatch, _ops):
        pass

    def post_group(self, _description, ops):
        op_list = tuple(ops)
        return op_list or None

    def wait_group(self, _token):
        pass

    def drain(self):
        assert not self.received

    def cleanup(self):
        self.received.clear()


class FakeStage:
    is_first = False
    is_last = False

    def __init__(self, log, fail_forward=None, *, stage_index=1, sparse=False):
        self.log = log
        self.fail_forward = fail_forward
        self.stage_index = stage_index
        self.receive_sparse_indices = sparse
        self.outputs = {}
        self.gradients = []
        self._fwd_buffers = [object() for _ in range(8)]
        self._bwd_buffers = [object() for _ in range(8)]
        self._cache_buffers = [object()]

    def get_fwd_recv_ops(self, microbatch):
        ops = [FakeOp("fwd_recv", microbatch, self._fwd_buffers[microbatch % 8])]
        if self.receive_sparse_indices:
            ops.append(FakeOp("cache_recv", microbatch, self._cache_buffers[0]))
        return ops

    def get_bwd_recv_ops(self, microbatch):
        return [FakeOp("bwd_recv", microbatch, self._bwd_buffers[microbatch % 8])]

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


class PeerStage(FakeStage):
    def __init__(self, log, stage_index, sparse):
        super().__init__(log, stage_index=stage_index, sparse=sparse)
        self.is_first = stage_index == 0
        self.is_last = stage_index == 7

    def get_fwd_recv_ops(self, microbatch):
        if self.is_first:
            return []
        ops = [
            FakeOp(
                "fwd", microbatch, self._fwd_buffers[microbatch], self.stage_index - 1
            )
        ]
        if self.receive_sparse_indices:
            ops.append(
                FakeOp(
                    "cache", microbatch, self._cache_buffers[0], self.stage_index - 1
                )
            )
        return ops

    def get_fwd_send_ops(self, microbatch):
        if self.is_last:
            return []
        ops = [FakeOp("fwd", microbatch, peer=self.stage_index + 1)]
        if self.receive_sparse_indices:
            ops.append(FakeOp("cache", microbatch, peer=self.stage_index + 1))
        return ops

    def get_bwd_recv_ops(self, microbatch):
        if self.is_last:
            return []
        return [
            FakeOp(
                "bwd", microbatch, self._bwd_buffers[microbatch], self.stage_index + 1
            )
        ]

    def get_bwd_send_ops(self, microbatch):
        if self.is_first:
            return []
        return [FakeOp("bwd", microbatch, peer=self.stage_index - 1)]


class EdgeTraceTransport(FakeTransport):
    def __init__(self, module, rank, trace):
        self.rank = rank
        self.trace = trace
        super().__init__(module, [], fwd_window=1, bwd_window=1)

    def _post_receive(self, ops):
        op_list = tuple(ops)
        if op_list:
            self.trace.append(
                (
                    self.rank,
                    op_list[0].peer,
                    "receive",
                    tuple((op.kind, op.microbatch) for op in op_list),
                )
            )
        return op_list

    def _post_send(self, ops):
        op_list = tuple(ops)
        if op_list:
            self.trace.append(
                (
                    self.rank,
                    op_list[0].peer,
                    "send",
                    tuple((op.kind, op.microbatch) for op in op_list),
                )
            )
        return op_list

    def post_group(self, description, ops):
        op_list = tuple(ops)
        if not op_list:
            return None
        by_peer = {}
        for op in op_list:
            by_peer.setdefault(op.peer, []).append(op)
        for peer, peer_ops in by_peer.items():
            self.trace.append(
                (
                    self.rank,
                    peer,
                    description,
                    tuple((op.kind, op.microbatch) for op in peer_ops),
                )
            )
        token = (description, op_list)
        self.groups.append(token)
        self.max_groups = max(self.max_groups, len(self.groups))
        return token


class FakeSchedule:
    _num_stages = 4
    scale_grads = False

    def __init__(self, stage, n_microbatches=5, *, num_stages=4):
        self._stage = stage
        self._n_microbatches = n_microbatches
        self._num_microbatches = n_microbatches
        self._num_stages = num_stages
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
        self.assertFalse(
            transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        )
        transport.wait_receive("fwd", 0)
        self.assertFalse(
            transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        )
        transport.release_receive("fwd", 0)
        self.assertTrue(
            transport.prepost_receive("fwd", 2, [FakeOp("recv", 2, first_buffer)])
        )
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
            async_log.index(("post_group", "fwd_recv", (("fwd_recv", 0),))),
            async_log.index(("forward", 0)),
        )
        self.assertLessEqual(async_transport.max_groups, 2)

    def test_stage_one_to_six_64_microbatch_wraparound_and_sparse_ring(self):
        """All non-edge PP8 stages preserve 1F1B order across ring wraparound."""
        for stage_index in range(1, 7):
            async_stage = FakeStage([], stage_index=stage_index)
            async_schedule = FakeSchedule(async_stage, 64, num_stages=8)
            async_transport = FakeTransport(self.module, [], fwd_window=2, bwd_window=1)
            self.module.run_1f1b_async_step(
                async_schedule,
                async_transport,
                list(range(64)),
                [{} for _ in range(64)],
                [0] * 64,
                [],
                True,
            )
            reference_stage = FakeStage([], stage_index=stage_index)
            reference_schedule = FakeSchedule(reference_stage, 64, num_stages=8)
            self.module.run_1f1b_async_step(
                reference_schedule,
                ReferenceTransport(),
                list(range(64)),
                [{} for _ in range(64)],
                [0] * 64,
                [],
                True,
            )
            self.assertEqual(async_stage.outputs, reference_stage.outputs)
            self.assertEqual(async_stage.gradients, reference_stage.gradients)
            self.assertLessEqual(async_transport.state.max_pending_receives["fwd"], 2)
            self.assertLessEqual(async_transport.state.max_pending_receives["bwd"], 1)

        sparse_log = []
        sparse_stage = FakeStage(sparse_log, stage_index=3, sparse=True)
        sparse_schedule = FakeSchedule(sparse_stage, 64, num_stages=8)
        sparse_transport = FakeTransport(
            self.module, sparse_log, fwd_window=1, bwd_window=1
        )
        self.module.run_1f1b_async_step(
            sparse_schedule,
            sparse_transport,
            list(range(64)),
            [{} for _ in range(64)],
            [0] * 64,
            [],
            True,
        )
        self.assertLessEqual(sparse_transport.max_groups, 2)
        cache_posts = [
            item
            for item in sparse_log
            if item[0] == "post_group" and any(op[0] == "cache_recv" for op in item[2])
        ]
        self.assertEqual(len(cache_posts), 64)
        for left, right in zip(cache_posts, cache_posts[1:]):
            self.assertLessEqual(
                sparse_log.index(left) + 1,
                sparse_log.index(right),
            )

    def test_dense_and_sparse_pp8_groups_match_on_every_edge(self):
        def normalized(rank, peer, ops):
            messages = []
            for kind, microbatch in ops:
                forward = kind in ("fwd", "cache")
                sends = (forward and peer == rank + 1) or (
                    kind == "bwd" and peer == rank - 1
                )
                source, target = (rank, peer) if sends else (peer, rank)
                messages.append((source, target, kind, microbatch))
            return tuple(messages)

        for sparse in (False, True):
            traces = {}
            for rank in range(8):
                trace = []
                stage = PeerStage([], rank, sparse)
                schedule = FakeSchedule(stage, 8, num_stages=8)
                transport = EdgeTraceTransport(self.module, rank, trace)
                self.module.run_1f1b_async_step(
                    schedule,
                    transport,
                    list(range(8)),
                    [{} for _ in range(8)],
                    [0] * 8,
                    [],
                    True,
                )
                self.assertLessEqual(transport.max_groups, 2)
                traces[rank] = trace

            for left in range(7):
                right = left + 1
                left_calls = [item for item in traces[left] if item[1] == right]
                right_calls = [item for item in traces[right] if item[1] == left]
                self.assertEqual(len(left_calls), len(right_calls))
                for left_call, right_call in zip(left_calls, right_calls):
                    self.assertEqual(
                        set(normalized(left, right, left_call[3])),
                        set(normalized(right, left, right_call[3])),
                    )

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

    def test_sparse_receive_lookahead_is_one(self):
        stage = FakeStage([], sparse=True)
        self.assertEqual(self.module.receive_window_for_stage(stage, "fwd"), 1)
        stage.receive_sparse_indices = False
        stage._forward_slot_count = 2
        self.assertEqual(self.module.receive_window_for_stage(stage, "fwd"), 2)

    def test_live_stage_ring_slot_is_deferred(self):
        stage = FakeStage([])
        stage._forward_slot_by_key = {
            (stage.stage_index, 0): 0,
            (stage.stage_index, 8): 0,
        }
        stage._forward_slot_owner = [(stage.stage_index, 0)]
        stage._forward_events = [None]
        self.assertFalse(self.module._stage_receive_slot_available(stage, "fwd", 8))
        stage._forward_events[0] = object()
        self.assertTrue(self.module._stage_receive_slot_available(stage, "fwd", 8))

    def test_async_topology_guard_is_fail_closed(self):
        runner_source = (
            SOURCE / "src/prime_rl/trainer/distributed/pp_runner.py"
        ).read_text()
        self.assertIn("validate_async_topology", runner_source)
        self.assertIn('"cutoff": 5', runner_source)
        self.assertIn('"pp_limit_receive_buffers": True', runner_source)
        self.assertIn("if not limit_receive_buffers or pp_size != 8", runner_source)

    def test_cuda_transport_uses_explicit_stream_dependencies(self):
        source = MODULE_PATH.read_text()
        self.assertIn("torch.cuda.Stream(device=stage.device)", source)
        self.assertIn("with self._torch.cuda.stream(self._stream)", source)
        self.assertIn("self._stream.wait_event(ready)", source)
        self.assertIn("work.wait()", source)

    def test_cuda_transport_keeps_event_alive_and_drains_stream(self):
        log = []
        transport, cuda = make_cuda_transport(self.module, log)
        init_record = next(i for i, item in enumerate(log) if item[0] == "event_record")
        init_wait = next(
            i for i, item in enumerate(log) if item[0] == "stream_wait_event"
        )
        self.assertLess(init_record, init_wait)

        transport.prepost_receive("fwd", 0, ["receive"])
        transport.post_send("fwd", 0, ["send"])
        first_batch = next(
            i for i, item in enumerate(log) if item[0] == "batch_isend_irecv"
        )
        self.assertLess(init_wait, first_batch)
        send_token = transport._state._sends[0][2]
        self.assertIs(send_token.event, cuda.events[1])
        transport.wait_receive("fwd", 0)
        transport.release_receive("fwd", 0)
        transport.drain()

        waits = [i for i, item in enumerate(log) if item[0] == "work_wait"]
        stream_sync = next(i for i, item in enumerate(log) if item[0] == "stream_sync")
        self.assertTrue(waits)
        self.assertTrue(all(index < stream_sync for index in waits))
        transport.cleanup()
        self.assertEqual(
            [item for item in log if item[0] == "stream_sync"], [("stream_sync", 0)]
        )

    def test_cuda_transport_finishes_old_stream_before_next_step(self):
        log = []
        first, _cuda = make_cuda_transport(self.module, log)
        first.post_send("fwd", 0, ["step_one"])
        first.drain()
        first_sync = log.index(("stream_sync", 0))

        second, _cuda = make_cuda_transport(self.module, log)
        second_create = len(log) - 1 - log[::-1].index(("stream_create", 0, "cuda:0"))
        self.assertLess(first_sync, second_create)
        second.cleanup()

    def test_cuda_group_posts_one_batch_and_connects_both_streams(self):
        log = []
        transport, cuda = make_cuda_transport(self.module, log)
        pending = transport.post_group("paired", ["send", "receive"])
        self.assertIsNotNone(pending)
        batches = [item for item in log if item[0] == "batch_isend_irecv"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][2], ("send", "receive"))
        transport.wait_group(pending)
        self.assertIn(("stream_wait_event", 0, pending.event.index), log)
        self.assertIn(
            ("stream_wait_event", cuda.compute_stream.index, pending.completed.index),
            log,
        )
        transport.drain()


if __name__ == "__main__":
    unittest.main()
