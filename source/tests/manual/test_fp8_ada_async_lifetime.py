"""Stress W8A16 autograd across CUDA streams without diagnostic syncs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from prime_rl.trainer.models.layers.fp8_ada import (
    _W8A16Linear,
    _linear_input_grad_oracle,
    set_diagnose_linear_backward,
)


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--delay-cycles", type=int, default=20_000_000)
args = parser.parse_args()

if torch.cuda.get_device_capability() != (8, 9):
    raise RuntimeError("this qualification requires an RTX 4090 or another SM89 GPU")

model = Path("/model")
name = "model.layers.0.self_attn.indexer.wq_b.weight"
scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(model / index[name], framework="pt") as handle:
    source_weight = handle.get_tensor(name).cuda()
with safe_open(model / index[scale_name], framework="pt") as handle:
    source_scale = handle.get_tensor(scale_name).cuda()

torch.manual_seed(8721)
rows = 65
incoming = torch.randn(
    rows, source_weight.shape[0], device="cuda", dtype=torch.bfloat16
).mul_(0.1)
oracle = _linear_input_grad_oracle(incoming, source_weight, source_scale)
torch.cuda.synchronize()


class WeightOwner:
    weight: torch.Tensor
    scale: torch.Tensor


set_diagnose_linear_backward(False)
owner = WeightOwner()
producer = torch.cuda.Stream()
consumer = torch.cuda.current_stream()
results = []

for iteration in range(args.iterations):
    with torch.cuda.stream(producer):
        owner.weight = source_weight.clone()
        owner.scale = source_scale.clone()
    consumer.wait_stream(producer)
    torch.cuda._sleep(args.delay_cycles)
    activation = torch.randn(
        rows,
        source_weight.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = _W8A16Linear.apply(
        activation,
        owner.weight,
        owner.scale,
        f"async-lifetime-{iteration}",
        (owner, "weight", "scale"),
    )
    output.backward(incoming)
    results.append(activation.grad)
    del activation, output, owner.weight, owner.scale
    with torch.cuda.stream(producer):
        poison_weight = torch.empty_like(source_weight).fill_(448)
        poison_scale = torch.empty_like(source_scale).fill_(1000)
    del poison_weight, poison_scale

torch.cuda.synchronize()
minimum_cosine = 1.0
maximum_relative_rmse = 0.0
nonfinite = 0
for result in results:
    if not bool(torch.isfinite(result).all()):
        nonfinite += 1
        continue
    difference = result.float() - oracle.float()
    minimum_cosine = min(
        minimum_cosine,
        float(F.cosine_similarity(result.float().flatten(), oracle.float().flatten(), dim=0)),
    )
    maximum_relative_rmse = max(
        maximum_relative_rmse,
        float(
            difference.square().mean().sqrt()
            / oracle.float().abs().max().clamp_min(1e-12)
        ),
    )

report = {
    "cuda_capability": torch.cuda.get_device_capability(),
    "device": torch.cuda.get_device_name(),
    "iterations": args.iterations,
    "minimum_cosine": minimum_cosine,
    "maximum_relative_rmse": maximum_relative_rmse,
    "nonfinite": nonfinite,
}
print(json.dumps(report, sort_keys=True), flush=True)
if nonfinite:
    raise AssertionError(f"{nonfinite} asynchronous W8A16 gradients were nonfinite")
if minimum_cosine < 0.999 or maximum_relative_rmse > 0.005:
    raise AssertionError(f"asynchronous W8A16 accuracy failed: {report}")
