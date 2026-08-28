"""Eight-GPU CUDA qualification for the optional asynchronous PP8 schedule."""

from __future__ import annotations

import hashlib
import json
import os
import time
import types
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record

from prime_rl.trainer import phase_profiler
from prime_rl.trainer.distributed import async_schedule, pp_runner


stages = 8
microbatches = 8
steps = 2
rows = 2
columns = 4
learning_rate = 1e-4
junction_sparse_flags = (
    (False, True),
    (True, False),
    (False, True),
    (True, False),
    (False, True),
    (True, False),
    (False, True),
    (True, False),
)
journal_max_bytes = 256 * 1024
journal_max_records = 2048
journal_max_record_bytes = 1024


class RankJournal:
    def __init__(self, local_rank: int, output_root: Path = Path("/output")):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= os.O_NOFOLLOW
        self.path = output_root / f"rank-{int(local_rank)}.journal.jsonl"
        self.fd = os.open(self.path, flags, 0o600)
        self.bytes_written = 0
        self.records_written = 0
        self.mode = "startup"
        self.step = -1

    def write(self, event: str, **fields) -> None:
        if self.records_written >= journal_max_records:
            raise RuntimeError("rank journal record limit reached")
        record = {
            "time_ns": time.monotonic_ns(),
            "mode": self.mode,
            "step": self.step,
            "event": str(event),
            **fields,
        }
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if len(payload) > journal_max_record_bytes:
            raise RuntimeError("rank journal record is too large")
        if self.bytes_written + len(payload) > journal_max_bytes:
            raise RuntimeError("rank journal byte limit reached")
        offset = 0
        while offset < len(payload):
            offset += os.write(self.fd, payload[offset:])
        os.fsync(self.fd)
        self.bytes_written += len(payload)
        self.records_written += 1

    def close(self) -> None:
        if self.fd < 0:
            return
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = -1


def _p2p_summary(ops) -> list[dict[str, object]]:
    result = []
    for op in ops:
        function = getattr(op, "op", None)
        result.append(
            {
                "op": getattr(function, "__name__", type(function).__name__),
                "peer": int(getattr(op, "group_peer", getattr(op, "peer", -1))),
            }
        )
    return result


@contextmanager
def journal_native_transport(journal: RankJournal):
    from torch.distributed.pipelining import schedules

    original_batch = schedules._batch_p2p
    original_wait = schedules._wait_batch_p2p

    def traced_batch(p2p_ops, desc=None):
        summary = _p2p_summary(p2p_ops)
        journal.write("p2p_post_before", description=str(desc or ""), ops=summary)
        work = original_batch(p2p_ops, desc=desc)
        journal.write("p2p_post_after", description=str(desc or ""), count=len(work))
        return work

    def traced_wait(work):
        journal.write("p2p_wait_before", count=len(work))
        result = original_wait(work)
        journal.write("p2p_wait_after", count=len(work))
        return result

    schedules._batch_p2p = traced_batch
    schedules._wait_batch_p2p = traced_wait
    try:
        yield
    finally:
        schedules._batch_p2p = original_batch
        schedules._wait_batch_p2p = original_wait


class AffineStage(nn.Module):
    transfer_sparse_indices = False

    def __init__(self, stage_index: int):
        super().__init__()
        self.stage_identity = stage_index
        self.is_last = stage_index == stages - 1
        self.scale = nn.Parameter(torch.tensor(1.0 + stage_index * 0.01))
        self.bias = nn.Parameter(torch.tensor(stage_index * 0.001))
        self._active_microbatch: int | None = None
        self._microbatch_step: int | None = None
        self._microbatch_records: list[dict[str, object]] = []
        self._current_record: dict[str, object] | None = None

    def begin_microbatch_step(self, step: int) -> None:
        if self._active_microbatch is not None or self._current_record is not None:
            raise AssertionError("the previous dummy microbatch step is still active")
        if step != len(self._microbatch_records):
            raise AssertionError(f"unexpected dummy step {step}")
        self._microbatch_step = int(step)
        self._current_record = {"step": int(step), "set": [], "clear": []}

    def set_active_microbatch(self, microbatch_id: int) -> None:
        if self._current_record is None or self._microbatch_step is None:
            raise AssertionError("the dummy microbatch step was not started")
        if self._active_microbatch is not None:
            raise AssertionError(
                f"dummy stage {self.stage_identity} already has active microbatch "
                f"{self._active_microbatch}"
            )
        microbatch_id = int(microbatch_id)
        seen = self._current_record["set"]
        if not 0 <= microbatch_id < microbatches or microbatch_id in seen:
            raise AssertionError(f"invalid dummy microbatch {microbatch_id}")
        self._active_microbatch = microbatch_id
        seen.append(microbatch_id)

    def clear_active_microbatch(self) -> None:
        if self._current_record is None or self._active_microbatch is None:
            raise AssertionError("the dummy stage has no active microbatch")
        self._current_record["clear"].append(self._active_microbatch)
        self._active_microbatch = None

    def assert_microbatch_step_complete(self) -> None:
        if self._current_record is None or self._microbatch_step is None:
            raise AssertionError("the dummy microbatch step was not started")
        expected = list(range(microbatches))
        if self._active_microbatch is not None:
            raise AssertionError("the dummy stage still has an active microbatch")
        if sorted(self._current_record["set"]) != expected:
            raise AssertionError("the dummy stage did not set every microbatch")
        if self._current_record["clear"] != self._current_record["set"]:
            raise AssertionError("the dummy stage did not clear every microbatch")
        self._microbatch_records.append(self._current_record)
        self._current_record = None
        self._microbatch_step = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = hidden * self.scale + self.bias
        return output.sum() if self.is_last else output


class SparseAffineStage(AffineStage):
    transfer_sparse_indices = True

    def __init__(self, stage_index: int):
        super().__init__(stage_index)
        self.receive_sparse_indices, self.send_sparse_indices = junction_sparse_flags[
            stage_index
        ]
        self._cache_received: list[tuple[int, int]] = []
        self._cache_sent: list[tuple[int, int]] = []

    def _cache(self, source_stage: int, microbatch: int, device) -> torch.Tensor:
        values = torch.arange(8, device=device, dtype=torch.int16)
        values.add_(source_stage * 31 + microbatch * 7 + 1)
        return values.view(torch.float16).view(1, 2, 1, 4)

    def _cache_source(self) -> int:
        for stage in range(self.stage_identity - 1, -1, -1):
            if junction_sparse_flags[stage][1]:
                return stage
        raise AssertionError(f"stage {self.stage_identity} has no cache source")

    def forward(self, hidden: torch.Tensor, cache: torch.Tensor | None = None):
        microbatch = self._active_microbatch
        if microbatch is None:
            raise AssertionError("sparse stage ran without an active microbatch")
        if self.receive_sparse_indices:
            if cache is None:
                raise AssertionError(
                    f"stage {self.stage_identity} did not receive its sparse cache"
                )
            expected = self._cache(self._cache_source(), microbatch, cache.device)
            if not torch.equal(cache.view(torch.int16), expected.view(torch.int16)):
                raise AssertionError(
                    f"stage {self.stage_identity} received the wrong cache for "
                    f"microbatch {microbatch}"
                )
            self._cache_received.append((self.stage_identity, microbatch))
        elif self.stage_identity == 0 and cache is not None:
            raise AssertionError("stage 0 must not receive a sparse cache")
        elif self.stage_identity != 0 and (cache is None or cache.numel() != 1):
            raise AssertionError(
                f"stage {self.stage_identity} expected a one-element cache placeholder"
            )

        output = hidden * self.scale + self.bias
        if self.is_last:
            if not self.receive_sparse_indices:
                raise AssertionError("the last stage must receive the sparse cache")
            return output.sum()
        if self.send_sparse_indices:
            cache_out = self._cache(self.stage_identity, microbatch, hidden.device)
            self._cache_sent.append((self.stage_identity, microbatch))
        else:
            cache_out = torch.zeros(1, dtype=torch.float16, device=hidden.device)
        return output, cache_out

    def assert_cache_lifecycle(self) -> None:
        expected = list(range(microbatches)) * steps
        if (
            self.receive_sparse_indices
            and [mb for _stage, mb in self._cache_received] != expected
        ):
            raise AssertionError(
                f"stage {self.stage_identity} did not consume every sparse cache"
            )
        if not self.receive_sparse_indices and self._cache_received:
            raise AssertionError(
                f"stage {self.stage_identity} consumed an unexpected cache"
            )
        if (
            self.send_sparse_indices
            and [mb for _stage, mb in self._cache_sent] != expected
        ):
            raise AssertionError(
                f"stage {self.stage_identity} did not produce every sparse cache"
            )
        if not self.send_sparse_indices and self._cache_sent:
            raise AssertionError(
                f"stage {self.stage_identity} produced an unexpected cache"
            )


class WorkLifecycle:
    def __init__(self, journal=None):
        self.journal = journal
        self.created = 0
        self.waited = 0
        self.retired = 0
        self._pending: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._live: dict[int, object] = {}
        self._waited: set[int] = set()

    def _write(self, event: str, work_id: int, description: str) -> None:
        if self.journal is not None:
            self.journal.write(
                event,
                work_id=work_id,
                description=description,
            )

    def create(self, pending, description: str) -> None:
        if pending is None:
            return
        pending_id = id(pending)
        if pending_id in self._pending:
            raise AssertionError("NCCL pending group was recorded twice")
        work_ids = []
        for work in pending.works:
            work_id = self.created
            self.created += 1
            self._live[work_id] = work
            work_ids.append(work_id)
            self._write("async_work_created", work_id, description)
        self._pending[pending_id] = (description, tuple(work_ids))

    def mark_waited(self, pending) -> None:
        if pending is None:
            return
        record = self._pending.get(id(pending))
        if record is None:
            raise AssertionError("NCCL pending group was not recorded")
        description, work_ids = record
        for work_id in work_ids:
            if work_id in self._waited:
                raise AssertionError(f"NCCL work {work_id} was waited twice")
            self._waited.add(work_id)
            self.waited += 1
            self._write("async_work_waited", work_id, description)

    def retire(self, pending) -> None:
        if pending is None:
            return
        record = self._pending.pop(id(pending), None)
        if record is None:
            raise AssertionError("NCCL pending group was not recorded")
        description, work_ids = record
        for work_id in work_ids:
            if work_id not in self._waited or work_id not in self._live:
                raise AssertionError(f"NCCL work {work_id} was not waited")
            del self._live[work_id]
            self._waited.remove(work_id)
            self.retired += 1
            self._write("async_work_retired", work_id, description)

    def assert_drained(self) -> None:
        if any(not work.is_completed() for work in self._live.values()):
            raise AssertionError("NCCL work remained active after drain")
        if self._pending or self._live or self._waited:
            raise AssertionError("NCCL work remained active after drain")
        if self.created != self.waited or self.waited != self.retired:
            raise AssertionError("NCCL work lifecycle counts differ after drain")

    def counts(self) -> dict[str, int]:
        return {
            "created": self.created,
            "waited": self.waited,
            "retired": self.retired,
            "live": len(self._live),
        }


class AuditedTransport(async_schedule.CudaAsyncP2P):
    instances: list["AuditedTransport"] = []
    journal: RankJournal | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifecycle = WorkLifecycle(self.journal)
        self.drained = False
        self.instances.append(self)

    def _post_receive(self, ops):
        if self.journal is not None:
            self.journal.write("async_receive_post_before", ops=_p2p_summary(ops))
        pending = super()._post_receive(ops)
        self.lifecycle.create(pending, "receive")
        if self.journal is not None:
            self.journal.write("async_receive_post_after", count=len(pending.works))
        return pending

    def _post_send(self, ops):
        if self.journal is not None:
            self.journal.write("async_send_post_before", ops=_p2p_summary(ops))
        pending = super()._post_send(ops)
        self.lifecycle.create(pending, "send")
        if self.journal is not None:
            self.journal.write("async_send_post_after", count=len(pending.works))
        return pending

    def _wait_receive(self, pending):
        if self.journal is not None:
            self.journal.write("async_receive_wait_before", count=len(pending.works))
        result = super()._wait_receive(pending)
        self.lifecycle.mark_waited(pending)
        self.lifecycle.retire(pending)
        if self.journal is not None:
            self.journal.write("async_receive_wait_after", count=len(pending.works))
        return result

    def _wait_send(self, pending):
        if self.journal is not None:
            self.journal.write("async_send_wait_before", count=len(pending.works))
        result = super()._wait_send(pending)
        self.lifecycle.mark_waited(pending)
        self.lifecycle.retire(pending)
        if self.journal is not None:
            self.journal.write("async_send_wait_after", count=len(pending.works))
        return result

    def post_group(self, description, ops):
        op_list = list(ops)
        if self.journal is not None:
            self.journal.write(
                "async_group_post_before",
                description=description,
                ops=_p2p_summary(op_list),
            )
        pending = super().post_group(description, op_list)
        self.lifecycle.create(pending, description)
        if self.journal is not None:
            self.journal.write(
                "async_group_post_after",
                description=description,
                count=0 if pending is None else len(pending.works),
            )
        return pending

    def wait_group(self, pending):
        if self.journal is not None:
            self.journal.write(
                "async_group_wait_before",
                count=0 if pending is None else len(pending.works),
            )
        result = super().wait_group(pending)
        self.lifecycle.mark_waited(pending)
        self.lifecycle.retire(pending)
        if self.journal is not None:
            self.journal.write("async_group_wait_after")
        return result

    def drain(self):
        if self.journal is not None:
            self.journal.write("async_drain_before")
        super().drain()
        state = self._state
        if any(state._receives.values()) or any(state._completed_receives.values()):
            raise AssertionError("receive work remained after drain")
        if state._sends:
            raise AssertionError("send work remained after drain")
        if not self._stream_synchronized or not self._stream.query():
            raise AssertionError("communication stream remained active after drain")
        self.lifecycle.assert_drained()
        self.drained = True
        if self.journal is not None:
            self.journal.write("async_drain_after")


def fixed_input(step: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(
        microbatches * rows * columns,
        device=device,
        dtype=torch.float32,
    )
    return (
        values.view(microbatches, rows, columns).add_(step).div_(1000).requires_grad_()
    )


def record_actions(stage, actions: list[tuple[str, int]], journal: RankJournal) -> None:
    forward = stage.forward_one_chunk
    backward = stage.backward_one_chunk

    def forward_one_chunk(microbatch, *args, **kwargs):
        journal.write("action_before", action="forward", microbatch=int(microbatch))
        actions.append(("F", int(microbatch)))
        result = forward(microbatch, *args, **kwargs)
        journal.write("action_after", action="forward", microbatch=int(microbatch))
        return result

    def backward_one_chunk(microbatch, *args, **kwargs):
        journal.write("action_before", action="backward", microbatch=int(microbatch))
        actions.append(("B", int(microbatch)))
        result = backward(microbatch, *args, **kwargs)
        journal.write("action_after", action="backward", microbatch=int(microbatch))
        return result

    stage.forward_one_chunk = types.MethodType(
        lambda _stage, microbatch, *args, **kwargs: forward_one_chunk(
            microbatch, *args, **kwargs
        ),
        stage,
    )
    stage.backward_one_chunk = types.MethodType(
        lambda _stage, microbatch, *args, **kwargs: backward_one_chunk(
            microbatch, *args, **kwargs
        ),
        stage,
    )


def run_mode(
    mode: str,
    rank: int,
    device: torch.device,
    journal: RankJournal,
    *,
    sparse: bool = False,
) -> dict:
    schedule_name = "1f1b_async" if mode.startswith("1f1b_async") else mode
    is_async = schedule_name == "1f1b_async"
    journal.mode = mode
    journal.step = -1
    journal.write("schedule_build_before", rank=rank)
    module = (SparseAffineStage(rank) if sparse else AffineStage(rank)).to(device)
    example = torch.empty((1, rows, columns), device=device)
    cache_example = None
    placeholder_example = None
    if sparse:
        cache_example = torch.empty((1, 2, 1, 4), dtype=torch.float16, device=device)
        placeholder_example = torch.empty((1,), dtype=torch.float16, device=device)
    if rank == 0:
        example_input = example
    elif sparse:
        example_input = (
            example,
            cache_example if module.receive_sparse_indices else placeholder_example,
        )
    else:
        example_input = example
    if module.is_last:
        example_output = torch.empty((), device=device)
    elif sparse:
        example_output = (
            example,
            cache_example if module.send_sparse_indices else placeholder_example,
        )
    else:
        example_output = example
    schedule = pp_runner.build_schedule(
        module,
        pp_rank=rank,
        pp_size=stages,
        n_microbatches=microbatches,
        pp_group=dist.group.WORLD,
        schedule_name=schedule_name,
        example_input=example_input,
        example_output=example_output,
        limit_receive_buffers=True,
        async_topology={
            "pp_limit_receive_buffers": True,
            "pp": 8,
            "cp": 8,
            "ep": 8,
            "base_schedule": "1f1b",
            "cutoff": 5,
        }
        if is_async
        else None,
    )
    journal.write("schedule_build_after", rank=rank)
    actions = []
    record_actions(schedule._stage, actions, journal)
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    reports = []
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)

    for step in range(steps):
        journal.step = step
        journal.write("step_before")
        module.begin_microbatch_step(step)
        optimizer.zero_grad(set_to_none=True)
        inputs = fixed_input(step, device)
        losses = [] if module.is_last else None
        action_start = len(actions)
        transport_start = len(AuditedTransport.instances)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with journal_native_transport(journal):
            if rank == 0:
                schedule.step(inputs, return_outputs=False)
            elif rank == stages - 1:
                schedule.step(
                    target=torch.zeros(microbatches, device=device),
                    losses=losses,
                    return_outputs=False,
                )
            else:
                schedule.step(return_outputs=False)
        journal.write("step_schedule_after")
        module.assert_microbatch_step_complete()
        end.record()
        end.synchronize()

        step_actions = actions[action_start:]
        expected = [("F", index) for index in range(microbatches)]
        expected += [("B", index) for index in range(microbatches)]
        if sorted(step_actions) != sorted(expected):
            raise AssertionError(f"{mode} action set differs at step {step}")
        if module.scale.grad is None or module.bias.grad is None:
            raise AssertionError(f"{mode} has a missing gradient at step {step}")
        if not torch.isfinite(module.scale.grad) or not torch.isfinite(
            module.bias.grad
        ):
            raise AssertionError(f"{mode} has a nonfinite gradient at step {step}")
        if is_async:
            created = AuditedTransport.instances[transport_start:]
            if len(created) != 1 or not created[0].drained:
                raise AssertionError(f"async transport did not drain at step {step}")
            work_lifecycle = created[0].lifecycle.counts()
            if not (
                work_lifecycle["created"]
                == work_lifecycle["waited"]
                == work_lifecycle["retired"]
                and work_lifecycle["live"] == 0
            ):
                raise AssertionError(f"async work lifecycle differs at step {step}")
        else:
            work_lifecycle = None

        report = {
            "step": step,
            "actions": step_actions,
            "losses": [float(value.detach()) for value in losses or ()],
            "input_grad": inputs.grad.detach().cpu().tolist() if rank == 0 else None,
            "scale_grad": float(module.scale.grad),
            "bias_grad": float(module.bias.grad),
            "cuda_ms": start.elapsed_time(end),
            "microbatch_lifecycle": module._microbatch_records[-1],
            "work_lifecycle": work_lifecycle,
        }
        optimizer.step()
        report["scale_after_update"] = float(module.scale)
        report["bias_after_update"] = float(module.bias)
        reports.append(report)
        journal.write("step_barrier_before")
        dist.barrier()
        journal.write("step_barrier_after")
        journal.write("step_after")

    if sparse:
        module.assert_cache_lifecycle()
        cache_lifecycle = {
            "received": len(module._cache_received),
            "sent": len(module._cache_sent),
            "steps": steps,
            "microbatches": microbatches,
        }
    else:
        cache_lifecycle = None
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device) - before,
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return {
        "steps": reports,
        "peak_memory_bytes_max": int(peak),
        "sparse_cache": sparse,
        "sparse_cache_lifecycle": cache_lifecycle,
        "all_async_nccl_work_drained": not is_async
        or all(item.drained for item in AuditedTransport.instances),
    }


def assert_parity(baseline: dict, candidate: dict) -> None:
    if len(baseline["steps"]) != steps or len(candidate["steps"]) != steps:
        raise AssertionError("both schedules must complete exactly two steps")
    for old, new in zip(baseline["steps"], candidate["steps"]):
        if old["actions"] != new["actions"]:
            raise AssertionError(f"action order differs at step {old['step']}")
        for field in (
            "losses",
            "input_grad",
            "scale_grad",
            "bias_grad",
            "scale_after_update",
            "bias_after_update",
            "microbatch_lifecycle",
        ):
            if old[field] != new[field]:
                raise AssertionError(f"{field} differs at step {old['step']}")


def _process_group_timeout() -> timedelta:
    raw = os.environ.get("PRIME_PROCESS_GROUP_TIMEOUT_SECONDS")
    if raw is None or not raw.isdecimal() or int(raw) < 1:
        raise RuntimeError(
            "PRIME_PROCESS_GROUP_TIMEOUT_SECONDS must be a positive integer"
        )
    return timedelta(seconds=int(raw))


def _process_group_device_id(local_rank: int) -> int:
    if os.environ.get("PRIME_PROCESS_GROUP_DEVICE_ID") != "local_rank":
        raise RuntimeError(
            "PRIME_PROCESS_GROUP_DEVICE_ID must be configured as local_rank"
        )
    return local_rank


@record
def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != stages:
        raise AssertionError("this qualification requires exactly eight ranks")
    if torch.cuda.device_count() != stages:
        raise AssertionError("this qualification requires exactly eight visible GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    journal = RankJournal(local_rank)
    try:
        journal.write("cuda_device_before", local_rank=local_rank)
        torch.cuda.set_device(local_rank)
        journal.write("cuda_device_after", local_rank=local_rank)
        journal.write("group_init_before")
        dist.init_process_group(
            "nccl",
            timeout=_process_group_timeout(),
            device_id=_process_group_device_id(local_rank),
        )
        rank = dist.get_rank()
        journal.write("group_init_after", rank=rank)
        baseline = run_mode("1f1b", rank, torch.device("cuda", local_rank), journal)
        journal.write("mode_barrier_before", mode="1f1b")
        dist.barrier()
        journal.write("mode_barrier_after", mode="1f1b")
        original = async_schedule.CudaAsyncP2P
        async_schedule.CudaAsyncP2P = AuditedTransport
        AuditedTransport.journal = journal
        try:
            candidate = run_mode(
                "1f1b_async", rank, torch.device("cuda", local_rank), journal
            )
            assert_parity(baseline, candidate)
            journal.write("mode_barrier_before", mode="1f1b_async")
            dist.barrier()
            journal.write("mode_barrier_after", mode="1f1b_async")
            sparse = run_mode(
                "1f1b_async_sparse",
                rank,
                torch.device("cuda", local_rank),
                journal,
                sparse=True,
            )
        finally:
            AuditedTransport.journal = None
            async_schedule.CudaAsyncP2P = original
        assert_parity(baseline, sparse)
        local = {
            "rank": rank,
            "baseline": baseline,
            "async": candidate,
            "async_sparse": sparse,
        }
        rank_reports = [None] * stages if rank == 0 else None
        dist.gather_object(local, rank_reports, dst=0)
        if rank == 0:
            result = {
                "status": "pass",
                "ranks": dist.get_world_size(),
                "steps": steps,
                "microbatches": microbatches,
                "action_loss_grad_update_parity": True,
                "all_async_nccl_work_drained": all(
                    item["async"]["all_async_nccl_work_drained"]
                    for item in rank_reports
                ),
                "all_sparse_cache_nccl_work_drained": all(
                    item["async_sparse"]["all_async_nccl_work_drained"]
                    for item in rank_reports
                ),
                "rank_reports": rank_reports,
                "async_schedule_sha256": hashlib.sha256(
                    Path(async_schedule.__file__).read_bytes()
                ).hexdigest(),
                "pp_runner_sha256": hashlib.sha256(
                    Path(pp_runner.__file__).read_bytes()
                ).hexdigest(),
                "phase_profiler_sha256": hashlib.sha256(
                    Path(phase_profiler.__file__).read_bytes()
                ).hexdigest(),
                "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            Path("/output/result.json").write_text(
                json.dumps(result, sort_keys=True) + "\n"
            )
    finally:
        if dist.is_initialized():
            journal.write("group_destroy_before")
            dist.destroy_process_group()
            journal.write("group_destroy_after")
        journal.close()


if __name__ == "__main__":
    main()
