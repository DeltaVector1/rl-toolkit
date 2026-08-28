"""Eight-GPU CUDA qualification for the optional asynchronous PP8 schedule."""

from __future__ import annotations

import hashlib
import json
import os
import types
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work = []
        self.drained = False
        self.instances.append(self)

    def _post_receive(self, ops):
        pending = super()._post_receive(ops)
        self.work.extend(pending.works)
        return pending

    def _post_send(self, ops):
        pending = super()._post_send(ops)
        self.work.extend(pending.works)
        return pending

    def drain(self):
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


def fixed_input(step: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(
        microbatches * rows * columns,
        device=device,
        dtype=torch.float32,
    )
    return values.view(microbatches, rows, columns).add_(step).div_(1000).requires_grad_()


def record_actions(stage, actions: list[tuple[str, int]]) -> None:
    forward = stage.forward_one_chunk
    backward = stage.backward_one_chunk

    def forward_one_chunk(microbatch, *args, **kwargs):
        actions.append(("F", int(microbatch)))
        return forward(microbatch, *args, **kwargs)

    def backward_one_chunk(microbatch, *args, **kwargs):
        actions.append(("B", int(microbatch)))
        return backward(microbatch, *args, **kwargs)

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


def run_mode(mode: str, rank: int, device: torch.device) -> dict:
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
    actions = []
    record_actions(schedule._stage, actions)
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    reports = []
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        inputs = fixed_input(step, device)
        losses = [] if module.is_last else None
        action_start = len(actions)
        transport_start = len(AuditedTransport.instances)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if rank == 0:
            schedule.step(inputs, return_outputs=False)
        else:
            schedule.step(
                target=torch.zeros(microbatches, device=device),
                losses=losses,
                return_outputs=False,
            )
        end.record()
        end.synchronize()

        step_actions = actions[action_start:]
        expected = [("F", index) for index in range(microbatches)]
        expected += [("B", index) for index in range(microbatches)]
        if sorted(step_actions) != sorted(expected):
            raise AssertionError(f"{mode} action set differs at step {step}")
        if module.scale.grad is None or module.bias.grad is None:
            raise AssertionError(f"{mode} has a missing gradient at step {step}")
        if not torch.isfinite(module.scale.grad) or not torch.isfinite(module.bias.grad):
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
        dist.barrier()

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
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    try:
        baseline = run_mode("1f1b", rank, torch.device("cuda", local_rank))
        dist.barrier()
        original = async_schedule.CudaAsyncP2P
        async_schedule.CudaAsyncP2P = AuditedTransport
        try:
            candidate = run_mode("1f1b_async", rank, torch.device("cuda", local_rank))
        finally:
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
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
