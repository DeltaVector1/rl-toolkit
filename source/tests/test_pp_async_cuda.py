"""Eight-GPU CUDA qualification for the optional asynchronous PP8 schedule."""

from __future__ import annotations

import hashlib
import json
import os
import time
import types
from contextlib import contextmanager
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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = hidden * self.scale + self.bias
        return output.sum() if self.is_last else output


class AuditedTransport(async_schedule.CudaAsyncP2P):
    instances: list["AuditedTransport"] = []
    journal: RankJournal | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work = []
        self.drained = False
        self.instances.append(self)

    def _post_receive(self, ops):
        if self.journal is not None:
            self.journal.write("async_receive_post_before", ops=_p2p_summary(ops))
        pending = super()._post_receive(ops)
        if self.journal is not None:
            self.journal.write("async_receive_post_after", count=len(pending.works))
        self.work.extend(pending.works)
        return pending

    def _post_send(self, ops):
        if self.journal is not None:
            self.journal.write("async_send_post_before", ops=_p2p_summary(ops))
        pending = super()._post_send(ops)
        if self.journal is not None:
            self.journal.write("async_send_post_after", count=len(pending.works))
        self.work.extend(pending.works)
        return pending

    def _wait_receive(self, pending):
        if self.journal is not None:
            self.journal.write("async_receive_wait_before", count=len(pending.works))
        result = super()._wait_receive(pending)
        if self.journal is not None:
            self.journal.write("async_receive_wait_after", count=len(pending.works))
        return result

    def _wait_send(self, pending):
        if self.journal is not None:
            self.journal.write("async_send_wait_before", count=len(pending.works))
        result = super()._wait_send(pending)
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
        if pending is not None:
            self.work.extend(pending.works)
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
        if any(not item.is_completed() for item in self.work):
            raise AssertionError("NCCL work remained active after drain")
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


def run_mode(mode: str, rank: int, device: torch.device, journal: RankJournal) -> dict:
    journal.mode = mode
    journal.step = -1
    journal.write("schedule_build_before", rank=rank)
    module = AffineStage(rank).to(device)
    example = torch.empty((1, rows, columns), device=device)
    schedule = pp_runner.build_schedule(
        module,
        pp_rank=rank,
        pp_size=stages,
        n_microbatches=microbatches,
        pp_group=dist.group.WORLD,
        schedule_name=mode,
        example_input=example,
        example_output=torch.empty((), device=device) if module.is_last else example,
        limit_receive_buffers=True,
        async_topology={
            "pp_limit_receive_buffers": True,
            "pp": 8,
            "cp": 8,
            "ep": 8,
            "base_schedule": "1f1b",
            "cutoff": 5,
        }
        if mode == "1f1b_async"
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
        if mode == "1f1b_async":
            created = AuditedTransport.instances[transport_start:]
            if len(created) != 1 or not created[0].drained:
                raise AssertionError(f"async transport did not drain at step {step}")

        report = {
            "step": step,
            "actions": step_actions,
            "losses": [float(value.detach()) for value in losses or ()],
            "input_grad": inputs.grad.detach().cpu().tolist() if rank == 0 else None,
            "scale_grad": float(module.scale.grad),
            "bias_grad": float(module.bias.grad),
            "cuda_ms": start.elapsed_time(end),
        }
        optimizer.step()
        report["scale_after_update"] = float(module.scale)
        report["bias_after_update"] = float(module.bias)
        reports.append(report)
        journal.write("step_barrier_before")
        dist.barrier()
        journal.write("step_barrier_after")
        journal.write("step_after")

    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device) - before,
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return {
        "steps": reports,
        "peak_memory_bytes_max": int(peak),
        "all_async_nccl_work_drained": mode != "1f1b_async"
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
        ):
            if old[field] != new[field]:
                raise AssertionError(f"{field} differs at step {old['step']}")


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
        dist.init_process_group("nccl")
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
        finally:
            AuditedTransport.journal = None
            async_schedule.CudaAsyncP2P = original
        assert_parity(baseline, candidate)
        local = {"rank": rank, "baseline": baseline, "async": candidate}
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
