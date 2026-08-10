"""Two-rank gate for hidden-state plus sparse-index pipeline transport."""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B


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
        transport = indices.to(torch.int16).view(torch.float16).requires_grad_()
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
        # PipelineStage requires every input to participate in backward. The
        # sparse indices are discrete, so use a zero-valued dependency only.
        anchor = torch.nan_to_num(cache.reshape(-1)[0].float()) * 0
        hidden = hidden + anchor.to(hidden.dtype)
        return hidden.sum() * self.scale


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    example = torch.empty(1, 8, device="cuda")
    cache_example = torch.empty(1, 8, device="cuda", dtype=torch.float16)
    module = FirstStage() if rank == 0 else LastStage()
    stage = PipelineStage(
        module,
        stage_index=rank,
        num_stages=2,
        device=torch.device("cuda"),
        group=dist.group.WORLD,
        input_args=example if rank == 0 else (example, cache_example),
        output_args=(example, cache_example) if rank == 0 else torch.empty((), device="cuda"),
    )
    schedule = Schedule1F1B(
        stage,
        n_microbatches=4,
        loss_fn=lambda output, _target: output,
        scale_grads=False,
    )

    if rank == 0:
        print("rank0_schedule_start", flush=True)
        schedule.step(torch.ones(4, 8, device="cuda"), return_outputs=False)
    else:
        print("rank1_schedule_start", flush=True)
        schedule.step(target=torch.zeros(4, device="cuda"), return_outputs=False)
    print(f"rank{rank}_schedule_done", flush=True)

    if module.scale.grad is None or not bool(torch.isfinite(module.scale.grad)):
        raise RuntimeError(f"rank {rank} did not receive a finite parameter gradient")
    print(f"rank{rank}_gradient_done", flush=True)
    dist.barrier()
    if rank == 0:
        print("pipeline_sparse_cache=PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
