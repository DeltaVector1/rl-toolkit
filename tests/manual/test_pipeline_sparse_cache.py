"""Two-rank gate for hidden-state plus sparse-index pipeline transport."""

import os

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.distributed.pp_runner import build_schedule


class FirstStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0, device="cuda"))

    def forward(self, value):
        hidden = value * self.scale
        indices = torch.tensor(
            [0, 1, 2047, 2048, 32639, 32640, 32767, 32768],
            device=value.device,
            dtype=torch.int32,
        ).view_as(value)
        transport = indices.to(torch.int16).view(torch.float16)
        return hidden, transport


class LastStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0, device="cuda"))

    def forward(self, hidden, cache):
        expected = torch.tensor(
            [0, 1, 2047, 2048, 32639, 32640, 32767, 32768],
            device=cache.device,
            dtype=torch.int32,
        ).view_as(cache)
        decoded = cache.view(torch.int16).to(torch.int32).bitwise_and_(0xFFFF)
        if not torch.equal(decoded, expected):
            raise RuntimeError("pipeline sparse-index cache changed in transport")
        return hidden.sum() * self.scale


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    example = torch.empty(1, 8, device="cuda")
    cache_example = torch.empty(1, 8, device="cuda", dtype=torch.float16)
    module = FirstStage() if rank == 0 else LastStage()
    module.transfer_sparse_indices = True
    schedule = build_schedule(
        module,
        pp_rank=rank,
        pp_size=2,
        n_microbatches=4,
        pp_group=dist.group.WORLD,
        schedule_name="1f1b",
        example_input=example if rank == 0 else (example, cache_example),
        example_output=(example, cache_example)
        if rank == 0
        else torch.empty((), device="cuda"),
    )

    if rank == 0:
        print("rank0_schedule_start", flush=True)
        schedule.step(torch.ones(4, 8, device="cuda"), return_outputs=False)
    else:
        print("rank1_schedule_start", flush=True)
        schedule.step(target=torch.zeros(4, device="cuda"), return_outputs=False)
    print(f"rank{rank}_schedule_done", flush=True)

    stage = schedule._stage
    if rank == 0:
        if any(len(recv_infos) != 1 for recv_infos in stage.grad_recv_info.values()):
            raise RuntimeError("the discrete cache allocated a backward receive buffer")
    else:
        if any(
            recv_infos[1].buffer.requires_grad
            for recv_infos in stage.args_recv_info.values()
        ):
            raise RuntimeError("the discrete cache receive buffer requires a gradient")
        cache_buffers = {
            recv_infos[1].buffer.data_ptr()
            for recv_infos in stage.args_recv_info.values()
        }
        if len(cache_buffers) > 2:
            raise RuntimeError("the discrete cache allocated one buffer per microbatch")

    if module.scale.grad is None or not bool(torch.isfinite(module.scale.grad)):
        raise RuntimeError(f"rank {rank} did not receive a finite parameter gradient")
    print(f"rank{rank}_gradient_done", flush=True)
    dist.barrier()
    if rank == 0:
        print("pipeline_sparse_cache=PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
