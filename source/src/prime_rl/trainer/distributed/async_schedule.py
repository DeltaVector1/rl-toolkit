"""Opt-in asynchronous P2P transport for the single-stage 1F1B schedule.

The driver below keeps the compute schedule identical to PyTorch's
``Schedule1F1B``.  It changes only when the P2P work is posted and waited:
receives are posted up to two micro-batches ahead, and sends are posted on a
dedicated CUDA stream after an event from the compute stream.  Receive buffers
are owned by ``PipelineStage``; this module allocates no activation buffers.

This module does not alter ``torch.distributed`` or the installed PyTorch
classes.  The subclass is created lazily by ``make_schedule_1f1b_async`` so
the state-machine driver can be tested without importing CUDA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable


MAX_P2P_WINDOW = 2


@dataclass
class _PendingReceive:
    key: tuple[str, int]
    token: Any
    buffer: Any = None


class AsyncP2PState:
    """Bounded P2P bookkeeping shared by CUDA and fake-work transports.

    ``post_receive`` and ``post_send`` are callbacks.  A callback receives the
    operation list and returns an opaque work token.  The wait callbacks own
    the backend-specific completion operation.  The state object enforces the
    two-entry receive window and two-entry send queue, detects duplicate keys,
    and drains all work if the schedule raises.
    """

    def __init__(
        self,
        post_receive: Callable[[list[Any]], Any],
        post_send: Callable[[list[Any]], Any],
        wait_receive: Callable[[Any], None],
        wait_send: Callable[[Any], None],
        *,
        max_window: int | dict[str, int] = MAX_P2P_WINDOW,
    ) -> None:
        if isinstance(max_window, dict):
            windows = {
                direction: int(max_window.get(direction, MAX_P2P_WINDOW))
                for direction in ("fwd", "bwd")
            }
        else:
            windows = {"fwd": int(max_window), "bwd": int(max_window)}
        if any(not 1 <= window <= MAX_P2P_WINDOW for window in windows.values()):
            raise ValueError(
                f"P2P windows must be between 1 and {MAX_P2P_WINDOW}, got {windows}"
            )
        self._post_receive = post_receive
        self._post_send = post_send
        self._wait_receive = wait_receive
        self._wait_send = wait_send
        self.max_window = windows
        self.send_window = MAX_P2P_WINDOW
        self._receives: dict[str, dict[int, _PendingReceive]] = {
            "fwd": {},
            "bwd": {},
        }
        self._completed_receives: dict[str, dict[int, _PendingReceive]] = {
            "fwd": {},
            "bwd": {},
        }
        self._receive_order: dict[str, deque[int]] = {
            "fwd": deque(),
            "bwd": deque(),
        }
        self._received_keys: set[tuple[str, int]] = set()
        self._receive_buffer_owner: dict[int, tuple[str, int]] = {}
        self._sends: deque[tuple[str, int, Any]] = deque()
        self.max_pending_receives = {"fwd": 0, "bwd": 0}
        self.max_pending_sends = 0
        self.closed = False

    @staticmethod
    def _key(direction: str, microbatch: int) -> tuple[str, int]:
        if direction not in ("fwd", "bwd"):
            raise ValueError(f"unknown P2P direction {direction!r}")
        return direction, int(microbatch)

    def prepost_receive(
        self,
        direction: str,
        microbatch: int,
        ops: Iterable[Any],
        *,
        buffer: Any = None,
    ) -> None:
        """Post one receive and retain its token until the tensor is used."""
        key = self._key(direction, microbatch)
        pending = self._receives[direction]
        if key[1] in pending or key in self._received_keys:
            raise RuntimeError(f"duplicate {direction} receive for microbatch {key[1]}")
        if len(pending) >= self.max_window[direction]:
            raise RuntimeError(
                f"{direction} receive window exceeded {self.max_window[direction]} entries"
            )
        op_list = list(ops)
        if buffer is None:
            # ``dist.P2POp`` has a public ``tensor`` field.  Record every
            # receive tensor so a fake backend and the CUDA path both reject
            # reuse while an earlier receive still owns the storage.
            buffer_ids = tuple(
                id(tensor)
                for tensor in (getattr(op, "tensor", None) for op in op_list)
                if tensor is not None
            )
        elif buffer:
            if isinstance(buffer, int):
                buffer_ids = (buffer,)
            elif isinstance(buffer, (tuple, list, set)):
                buffer_ids = tuple(id(item) for item in buffer)
            else:
                buffer_ids = (id(buffer),)
        else:
            buffer_ids = ()
        if buffer_ids:
            for buffer_id in buffer_ids:
                owner = self._receive_buffer_owner.get(buffer_id)
                if owner is not None:
                    raise RuntimeError(
                        f"receive buffer is still owned by {owner[0]}:{owner[1]}"
                    )
                self._receive_buffer_owner[buffer_id] = key
        token = self._post_receive(op_list) if op_list else None
        pending[key[1]] = _PendingReceive(key, token, buffer_ids)
        self._receive_order[direction].append(key[1])
        self.max_pending_receives[direction] = max(
            self.max_pending_receives[direction], len(pending)
        )

    def wait_receive(self, direction: str, microbatch: int) -> None:
        """Wait for a receive immediately before its stage tensor is read."""
        key = self._key(direction, microbatch)
        pending = self._receives[direction].pop(key[1], None)
        if pending is None:
            raise RuntimeError(f"{direction} receive {key[1]} was not preposted")
        try:
            if pending.token is not None:
                self._wait_receive(pending.token)
        finally:
            self._completed_receives[direction][key[1]] = pending
            self._received_keys.add(key)
            try:
                self._receive_order[direction].remove(key[1])
            except ValueError:
                pass

    def release_receive(self, direction: str, microbatch: int) -> None:
        """Release receive storage after the stage no longer reads it.

        Hidden-state receives remain live through their microbatch backward;
        this is later than ``wait_receive``.  Keeping this separate from the
        communication completion is what makes a bounded receive ring safe.
        """
        key = self._key(direction, microbatch)
        item = self._completed_receives[direction].pop(key[1], None)
        if item is None:
            raise RuntimeError(f"{direction} receive {key[1]} was not completed")
        for buffer_id in item.buffer or ():
            if self._receive_buffer_owner.get(buffer_id) == key:
                del self._receive_buffer_owner[buffer_id]

    def post_send(
        self,
        direction: str,
        microbatch: int,
        ops: Iterable[Any],
    ) -> None:
        """Post a send after compute, waiting only when the bounded queue is full."""
        op_list = list(ops)
        if not op_list:
            return
        while len(self._sends) >= self.send_window:
            _direction, _microbatch, token = self._sends.popleft()
            self._wait_send(token)
        token = self._post_send(op_list)
        self._sends.append((direction, int(microbatch), token))
        self.max_pending_sends = max(self.max_pending_sends, len(self._sends))

    def drain(self) -> None:
        """Complete normal end-of-step work and reject forgotten receives."""
        forgotten = [
            f"{direction}:{mb}"
            for direction, pending in self._receives.items()
            for mb in pending
        ]
        forgotten.extend(
            f"{direction}:{mb}"
            for direction, pending in self._completed_receives.items()
            for mb in pending
        )
        if forgotten:
            raise RuntimeError(f"unwaited P2P receives at schedule end: {forgotten}")
        while self._sends:
            _direction, _microbatch, token = self._sends.popleft()
            self._wait_send(token)

    def cleanup(self) -> None:
        """Drain outstanding work after an exception, preserving the original error."""
        if self.closed:
            return
        self.closed = True
        # Cleanup must make a best effort.  A communication error here must not
        # replace the forward/backward exception that caused schedule teardown.
        for direction in ("fwd", "bwd"):
            pending = self._receives[direction]
            for microbatch in list(self._receive_order[direction]):
                item = pending.pop(microbatch, None)
                if item is None:
                    continue
                if item.token is not None:
                    try:
                        self._wait_receive(item.token)
                    except BaseException:
                        pass
                for buffer_id in item.buffer or ():
                    if self._receive_buffer_owner.get(buffer_id) == item.key:
                        del self._receive_buffer_owner[buffer_id]
            self._receive_order[direction].clear()
            for item in self._completed_receives[direction].values():
                for buffer_id in item.buffer or ():
                    if self._receive_buffer_owner.get(buffer_id) == item.key:
                        del self._receive_buffer_owner[buffer_id]
            self._completed_receives[direction].clear()
        while self._sends:
            _direction, _microbatch, token = self._sends.popleft()
            try:
                self._wait_send(token)
            except BaseException:
                pass


class _CudaPending:
    __slots__ = ("works", "event")

    def __init__(self, works: list[Any], event: Any = None) -> None:
        self.works = works
        self.event = event


class CudaAsyncP2P:
    """NCCL P2P implementation used by the opt-in schedule."""

    def __init__(self, stage: Any, *, max_window: int | None = None) -> None:
        import torch
        import torch.distributed as dist

        self._torch = torch
        self._dist = dist
        self._stream = torch.cuda.Stream(device=stage.device)
        if max_window is None:
            max_window = {
                "fwd": min(
                    MAX_P2P_WINDOW,
                    max(
                        1,
                        int(
                            getattr(
                                stage,
                                "_forward_slot_count",
                                MAX_P2P_WINDOW,
                            )
                        ),
                    ),
                ),
                "bwd": min(
                    MAX_P2P_WINDOW,
                    max(
                        1,
                        int(
                            getattr(
                                stage,
                                "_backward_slot_count",
                                MAX_P2P_WINDOW,
                            )
                        ),
                    ),
                ),
            }
        self._state = AsyncP2PState(
            self._post_receive,
            self._post_send,
            self._wait_receive,
            self._wait_send,
            max_window=max_window,
        )

    @property
    def max_pending_receives(self) -> dict[str, int]:
        return dict(self._state.max_pending_receives)

    @property
    def max_pending_sends(self) -> int:
        return self._state.max_pending_sends

    def receive_window(self, direction: str) -> int:
        return self._state.max_window[direction]

    def _post_receive(self, ops: list[Any]) -> _CudaPending:
        with self._torch.cuda.stream(self._stream):
            works = self._dist.batch_isend_irecv(ops)
            event = self._torch.cuda.Event()
            event.record(self._stream)
        return _CudaPending(list(works), event)

    def _post_send(self, ops: list[Any]) -> _CudaPending:
        # The send tensor is produced on the default compute stream.  Make the
        # communication stream wait for that exact compute work before NCCL
        # reads the tensor.
        ready = self._torch.cuda.Event()
        ready.record(self._torch.cuda.current_stream())
        with self._torch.cuda.stream(self._stream):
            self._stream.wait_event(ready)
            works = self._dist.batch_isend_irecv(ops)
        return _CudaPending(list(works))

    def _wait_receive(self, pending: _CudaPending) -> None:
        # Work.wait surfaces transport errors.  The event adds an explicit
        # compute-stream dependency, including for backends whose Work.wait is
        # host-only.
        for work in pending.works:
            work.wait()
        self._torch.cuda.current_stream().wait_event(pending.event)

    @staticmethod
    def _wait_send(pending: _CudaPending) -> None:
        for work in pending.works:
            work.wait()

    def prepost_receive(
        self, direction: str, microbatch: int, ops: Iterable[Any], *, buffer=None
    ) -> None:
        self._state.prepost_receive(direction, microbatch, ops, buffer=buffer)

    def wait_receive(self, direction: str, microbatch: int) -> None:
        self._state.wait_receive(direction, microbatch)

    def release_receive(self, direction: str, microbatch: int) -> None:
        self._state.release_receive(direction, microbatch)

    def post_send(self, direction: str, microbatch: int, ops: Iterable[Any]) -> None:
        self._state.post_send(direction, microbatch, ops)

    def drain(self) -> None:
        self._state.drain()

    def cleanup(self) -> None:
        self._state.cleanup()


def _run_1f1b_async_step(
    schedule: Any,
    transport: Any,
    arg_mbs: list,
    kwarg_mbs: list,
    target_mbs: list | None,
    losses: list | None,
    return_outputs: bool,
) -> None:
    """Run the exact PyTorch 1F1B action order with asynchronous transport.

    The only additional operations are receive preposts.  Every stage forward
    and backward call, loss callback, and final gradient reduction is in the
    same order as ``torch.distributed.pipelining.Schedule1F1B``.
    """
    stage = schedule._stage
    n_microbatches = schedule._n_microbatches
    warmup_chunks = min(n_microbatches, schedule._num_stages - stage.stage_index)
    fwd_mb_index = 0
    bwd_mb_index = 0
    posted_fwd: set[int] = set()
    posted_bwd: set[int] = set()

    def post_fwd(microbatch: int) -> None:
        if stage.is_first or microbatch in posted_fwd:
            return
        ops = stage.get_fwd_recv_ops(microbatch)
        transport.prepost_receive("fwd", microbatch, ops)
        posted_fwd.add(microbatch)

    def post_bwd(microbatch: int) -> None:
        if stage.is_last or microbatch in posted_bwd:
            return
        ops = stage.get_bwd_recv_ops(microbatch)
        transport.prepost_receive("bwd", microbatch, ops)
        posted_bwd.add(microbatch)

    def prime_fwd(next_mb: int) -> None:
        if stage.is_first:
            return
        window = int(
            getattr(transport, "receive_window", lambda _direction: MAX_P2P_WINDOW)(
                "fwd"
            )
        )
        for microbatch in range(next_mb, min(n_microbatches, next_mb + window)):
            if microbatch not in posted_fwd:
                post_fwd(microbatch)

    def prime_bwd(next_mb: int) -> None:
        if stage.is_last:
            return
        window = int(
            getattr(transport, "receive_window", lambda _direction: MAX_P2P_WINDOW)(
                "bwd"
            )
        )
        for microbatch in range(next_mb, min(n_microbatches, next_mb + window)):
            if microbatch not in posted_bwd:
                post_bwd(microbatch)

    # Warmup.
    for _ in range(warmup_chunks):
        prime_fwd(fwd_mb_index)
        if not stage.is_first:
            transport.wait_receive("fwd", fwd_mb_index)
        output = stage.forward_one_chunk(
            fwd_mb_index,
            arg_mbs[fwd_mb_index],
            kwarg_mbs[fwd_mb_index],
            save_forward_output=return_outputs,
        )
        transport.post_send("fwd", fwd_mb_index, stage.get_fwd_send_ops(fwd_mb_index))
        schedule._maybe_compute_loss(stage, output, target_mbs, fwd_mb_index)
        fwd_mb_index += 1

    # 1B1F steady state.  Keep the same backward-first ordering as PyTorch.
    while True:
        prime_bwd(bwd_mb_index)
        if not stage.is_last:
            transport.wait_receive("bwd", bwd_mb_index)
        loss = schedule._maybe_get_loss(stage, bwd_mb_index)
        stage.backward_one_chunk(
            bwd_mb_index,
            loss=loss,
            last_backward=bwd_mb_index == n_microbatches - 1,
        )
        if not stage.is_first:
            transport.release_receive("fwd", bwd_mb_index)
        if not stage.is_last:
            transport.release_receive("bwd", bwd_mb_index)
        transport.post_send("bwd", bwd_mb_index, stage.get_bwd_send_ops(bwd_mb_index))
        bwd_mb_index += 1

        if fwd_mb_index == n_microbatches:
            break

        prime_fwd(fwd_mb_index)
        if not stage.is_first:
            transport.wait_receive("fwd", fwd_mb_index)
        output = stage.forward_one_chunk(
            fwd_mb_index,
            arg_mbs[fwd_mb_index],
            kwarg_mbs[fwd_mb_index],
            save_forward_output=return_outputs,
        )
        transport.post_send("fwd", fwd_mb_index, stage.get_fwd_send_ops(fwd_mb_index))
        schedule._maybe_compute_loss(stage, output, target_mbs, fwd_mb_index)
        fwd_mb_index += 1

    # Cooldown.
    while bwd_mb_index < n_microbatches:
        prime_bwd(bwd_mb_index)
        if not stage.is_last:
            transport.wait_receive("bwd", bwd_mb_index)
        loss = schedule._maybe_get_loss(stage, bwd_mb_index)
        stage.backward_one_chunk(
            bwd_mb_index,
            loss=loss,
            last_backward=bwd_mb_index == n_microbatches - 1,
        )
        if not stage.is_first:
            transport.release_receive("fwd", bwd_mb_index)
        if not stage.is_last:
            transport.release_receive("bwd", bwd_mb_index)
        transport.post_send("bwd", bwd_mb_index, stage.get_bwd_send_ops(bwd_mb_index))
        bwd_mb_index += 1

    transport.drain()
    schedule._update_losses(stage, losses)
    stage.perform_reduce_grad(n_microbatches if schedule.scale_grads else 1)


def run_1f1b_async_step(
    schedule: Any,
    transport: Any,
    arg_mbs: list,
    kwarg_mbs: list,
    target_mbs: list | None,
    losses: list | None,
    return_outputs: bool,
) -> None:
    """Run one step and clean transport state on both success and failure."""
    try:
        _run_1f1b_async_step(
            schedule,
            transport,
            arg_mbs,
            kwarg_mbs,
            target_mbs,
            losses,
            return_outputs,
        )
    finally:
        transport.cleanup()


def make_schedule_1f1b_async() -> type:
    """Create the local opt-in schedule subclass from installed PyTorch."""
    from torch.distributed.pipelining import Schedule1F1B

    class Schedule1F1BAsync(Schedule1F1B):
        """Schedule1F1B with bounded asynchronous P2P transport."""

        def _step_microbatches(
            self,
            arg_mbs=None,
            kwarg_mbs=None,
            target_mbs=None,
            losses=None,
            return_outputs=True,
        ):
            arg_mbs, kwarg_mbs = self._check_inputs(
                arg_mbs, kwarg_mbs, target_mbs, losses
            )
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])
            transport = CudaAsyncP2P(self._stage)
            try:
                run_1f1b_async_step(
                    self,
                    transport,
                    arg_mbs,
                    kwarg_mbs,
                    target_mbs,
                    losses,
                    return_outputs,
                )
            finally:
                transport.cleanup()

    return Schedule1F1BAsync


def estimate_extra_buffer_bytes(
    hidden_shape: Iterable[int],
    hidden_element_size: int,
    *,
    sparse_shape: Iterable[int] | None = None,
    sparse_element_size: int = 0,
    slots_per_direction: int = 0,
) -> int:
    """Return exact bytes for optional *additional* transport buffers.

    The production wrapper uses existing ``PipelineStage`` receive buffers and
    passes ``slots_per_direction=0``.  The bound remains explicit: an alternate
    backend may request at most two additional slots in each direction.
    """
    if not 0 <= slots_per_direction <= MAX_P2P_WINDOW:
        raise ValueError(f"slots_per_direction must be between 0 and {MAX_P2P_WINDOW}")
    if hidden_element_size < 0 or sparse_element_size < 0:
        raise ValueError("element sizes cannot be negative")

    def numel(shape: Iterable[int]) -> int:
        result = 1
        for dimension in shape:
            if int(dimension) < 0:
                raise ValueError("buffer dimensions cannot be negative")
            result *= int(dimension)
        return result

    per_slot = numel(hidden_shape) * int(hidden_element_size)
    if sparse_shape is not None:
        per_slot += numel(sparse_shape) * int(sparse_element_size)
    return 2 * int(slots_per_direction) * per_slot

