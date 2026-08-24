"""Stress reentrant W8A16 blocks under FSDP resharding."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import prime_rl.trainer.models.layers.fp8_ada as fp8_ada
from prime_rl.trainer.models.layers.checkpointing import checkpoint_method
from prime_rl.trainer.models.layers.fp8_ada import (
    FP8CheckpointLinear,
    install_w8a16_reshard_wait,
    set_diagnose_linear_backward,
)


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--delay-cycles", type=int, default=0)
parser.add_argument("--diagnostics", action="store_true")
parser.add_argument("--kernel-delay-cycles", type=int, default=20_000_000)
parser.add_argument("--microbatches", type=int, default=4)
parser.add_argument("--rows", type=int, default=65)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--force-reuse-after-reshard", action="store_true")
parser.add_argument("--disable-fence", action="store_true")
args = parser.parse_args()

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
if torch.cuda.get_device_capability(device) != (8, 9):
    raise RuntimeError("this qualification requires SM89 GPUs")

rows = args.rows
width = 2048
rank = 32
layers = args.layers


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FP8CheckpointLinear(width, width, bias=False, device=device)
        self.base.weight = nn.Parameter(
            torch.randn(width, width, device=device).mul_(0.02).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.base.weight_scale_inv = nn.Parameter(
            torch.ones(width // 128, width // 128, device=device),
            requires_grad=False,
        )
        self.base.w8a16 = True
        self.lora_a = nn.Linear(width, rank, bias=False, device=device, dtype=torch.bfloat16)
        self.lora_b = nn.Linear(rank, width, bias=False, device=device, dtype=torch.bfloat16)

    def forward(self, value):
        return self.base(value) + self.lora_b(self.lora_a(value))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(layers))

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()

    def forward(self, value):
        return value + self.attention(value)


def make_model():
    torch.manual_seed(9917)
    return Model().train()


torch.manual_seed(7712)
source_input = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
incoming = torch.randn(rows, width, device=device, dtype=torch.bfloat16).mul_(0.1)

reference = make_model()
reference_input = source_input.detach().clone().requires_grad_(True)
reference(reference_input).backward(incoming)
reference_grad = reference_input.grad.detach().clone()
torch.cuda.synchronize(device)
del reference, reference_input

model = make_model()
for layer in model.layers:
    checkpoint_method(
        layer.attention,
        "forward",
        use_reentrant=True,
        preserve_rng_state=True,
    )
    fully_shard(layer, mesh=init_device_mesh("cuda", (dist.get_world_size(),)))
fully_shard(model, mesh=init_device_mesh("cuda", (dist.get_world_size(),)))

if args.force_reuse_after_reshard:
    for layer in model.layers:
        group = layer._get_fsdp_state()._fsdp_param_group
        original_post_backward = group.post_backward

        def post_backward_with_reuse(
            *values,
            _group=group,
            _original=original_post_backward,
        ):
            reuse_stream = _group.comm_ctx.all_gather_stream
            with torch.cuda.stream(reuse_stream):
                for fsdp_param in _group.fsdp_params:
                    for tensor in fsdp_param.all_gather_outputs:
                        tensor.fill_(448)
                reuse_done = torch.cuda.Event()
                reuse_done.record(reuse_stream)
            torch.cuda.current_stream().wait_event(reuse_done)
            return _original(*values)

        group.post_backward = post_backward_with_reuse
if not args.disable_fence:
    for layer in model.layers:
        install_w8a16_reshard_wait(layer)
set_diagnose_linear_backward(args.diagnostics)
native_input_grad = fp8_ada._native_w8a16_input_grad


def delayed_input_grad(*values):
    torch.cuda._sleep(args.kernel_delay_cycles)
    return native_input_grad(*values)


fp8_ada._native_w8a16_input_grad = delayed_input_grad

results = []
for _ in range(args.iterations):
    pending = []
    for _ in range(args.microbatches):
        value = source_input.detach().clone().requires_grad_(True)
        pending.append((value, model(value)))
    for value, output in reversed(pending):
        torch.cuda._sleep(args.delay_cycles)
        output.backward(incoming)
        results.append(value.grad)
    del pending

torch.cuda.synchronize(device)
minimum_cosine = 1.0
maximum_relative_rmse = 0.0
nonfinite = 0
for result in results:
    if not bool(torch.isfinite(result).all()):
        nonfinite += 1
        continue
    difference = result.float() - reference_grad.float()
    minimum_cosine = min(
        minimum_cosine,
        float(
            F.cosine_similarity(
                result.float().flatten(), reference_grad.float().flatten(), dim=0
            )
        ),
    )
    maximum_relative_rmse = max(
        maximum_relative_rmse,
        float(
            difference.square().mean().sqrt()
            / reference_grad.float().abs().max().clamp_min(1e-12)
        ),
    )

report = {
    "diagnostics": args.diagnostics,
    "iterations": args.iterations,
    "layers": layers,
    "microbatches": args.microbatches,
    "maximum_relative_rmse": maximum_relative_rmse,
    "minimum_cosine": minimum_cosine,
    "nonfinite": nonfinite,
    "rank": dist.get_rank(),
    "rows": rows,
}
print(json.dumps(report, sort_keys=True), flush=True)
if nonfinite:
    raise AssertionError(f"{nonfinite} FSDP W8A16 gradients were nonfinite")
if minimum_cosine < 0.999 or maximum_relative_rmse > 0.005:
    raise AssertionError(f"FSDP W8A16 accuracy failed: {report}")

dist.destroy_process_group()
