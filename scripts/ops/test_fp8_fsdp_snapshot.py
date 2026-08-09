"""Two-rank CUDA gate for native-FP8 backward across FSDP resharding.

Run in the trainer image:

    torchrun --standalone --nproc-per-node=2 \\
        scripts/ops/test_fp8_fsdp_snapshot.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

from prime_rl.trainer.models.layers.fp8_ada import FP8CheckpointLinear


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    torch.manual_seed(3)
    device = torch.device("cuda", local_rank)
    base = torch.nn.Linear(128, 256, bias=False, device=device)
    module = FP8CheckpointLinear.from_linear(base).to(device)
    with torch.no_grad():
        module.weight.copy_(
            torch.randn_like(module.weight.float())
            .mul_(0.02)
            .to(module.weight.dtype)
        )
        module.weight_scale_inv.fill_(0.01)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    fully_shard(module, mesh=mesh, reshard_after_forward=True)
    value = torch.randn(
        7, 128, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    for iteration in range(2):
        value.grad = None
        output = checkpoint(module, value, use_reentrant=False)
        output.float().square().mean().backward()
        if not bool(torch.isfinite(value.grad).all()):
            raise RuntimeError(
                f"rank {local_rank} iteration {iteration} produced "
                "a nonfinite native-FP8 input gradient"
            )
        if local_rank == 0:
            print(
                f"iteration {iteration}: finite, "
                f"absmax={float(value.grad.abs().max()):.6e}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
