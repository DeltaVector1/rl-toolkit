"""Two-rank gate for hidden-state plus sparse-index pipeline transport."""

import os

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.distributed.pp_runner import build_schedule


BOUNDARY_INDICES = [0, 1, 2047, 2048, 32639, 32640, 32767, 32768]


def make_indices(device, cache_shape):
    indices = torch.zeros(cache_shape, device=device, dtype=torch.int32)
    indices.flatten()[: len(BOUNDARY_INDICES)] = torch.tensor(
        BOUNDARY_INDICES,
        device=device,
        dtype=torch.int32,
    )
    return indices


class FirstStage(nn.Module):
    def __init__(self, cache_shape):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0, device="cuda"))
        self.cache_shape = cache_shape

    def forward(self, value):
        hidden = value * self.scale
        indices = make_indices(value.device, self.cache_shape)
        transport = indices.to(torch.int16).view(torch.float16)
        return hidden, transport


class LastStage(nn.Module):
    def __init__(self, cache_shape):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0, device="cuda"))
        self.cache_shape = cache_shape

    def forward(self, hidden, cache):
        expected = make_indices(cache.device, self.cache_shape)
        decoded = cache.view(torch.int16).to(torch.int32).bitwise_and_(0xFFFF)
        if not torch.equal(decoded, expected):
            raise RuntimeError("pipeline sparse-index cache changed in transport")
        return hidden.sum() * self.scale


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    production_cache = os.environ.get("PRODUCTION_CACHE", "") == "1"
    n_microbatches = 95 if production_cache else 4
    hidden_shape = (1, 1) if production_cache else (1, 8)
    cache_shape = (1, 4096, 1, 2048) if production_cache else (1, 8)
    example = torch.empty(hidden_shape, device="cuda")
    cache_example = torch.empty(cache_shape, device="cuda", dtype=torch.float16)
    module = FirstStage(cache_shape) if rank == 0 else LastStage(cache_shape)
    module.transfer_sparse_indices = True
    schedule = build_schedule(
        module,
        pp_rank=rank,
        pp_size=2,
        n_microbatches=n_microbatches,
        pp_group=dist.group.WORLD,
        schedule_name="1f1b",
        example_input=example if rank == 0 else (example, cache_example),
        example_output=(example, cache_example)
        if rank == 0
        else torch.empty((), device="cuda"),
    )

    if rank == 0:
        print("rank0_schedule_start", flush=True)
        schedule.step(
            torch.ones((n_microbatches, *hidden_shape[1:]), device="cuda"),
            return_outputs=False,
        )
    else:
        print("rank1_schedule_start", flush=True)
        schedule.step(
            target=torch.zeros(n_microbatches, device="cuda"),
            return_outputs=False,
        )
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
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    if production_cache and peak_gib >= 2.0:
        raise RuntimeError(f"production cache transport used {peak_gib:.2f} GiB")
    print(f"rank{rank}_peak_gib={peak_gib:.3f}", flush=True)

    if module.scale.grad is None or not bool(torch.isfinite(module.scale.grad)):
        raise RuntimeError(f"rank {rank} did not receive a finite parameter gradient")
    print(f"rank{rank}_gradient_done", flush=True)
    dist.barrier()
    if rank == 0:
        print("pipeline_sparse_cache=PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
