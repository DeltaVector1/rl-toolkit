"""Measure dense FP8 input-gradient kernels across token-row counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from prime_rl.trainer.models.layers.fp8_ada import (
    _linear_input_grad_oracle,
    _native_block_fp8_mm,
)


MODEL = Path("/model")
ROW_COUNTS = (1, 16, 63, 64, 65, 127, 128, 129, 512, 1024, 2048, 4096)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    default="model.layers.0.self_attn.o_proj.weight",
)
args = parser.parse_args()
NAME = args.name


def profiled(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
    "weight_map"
]
scale_name = NAME.removesuffix(".weight") + ".weight_scale_inv"
with safe_open(MODEL / index[NAME], framework="pt") as handle:
    weight = handle.get_tensor(NAME).cuda()
with safe_open(MODEL / index[scale_name], framework="pt") as handle:
    scale = handle.get_tensor(scale_name).cuda()

torch.manual_seed(1234)
for rows in ROW_COUNTS:
    grad = torch.randn(
        rows,
        weight.shape[0],
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.1)
    native = _native_block_fp8_mm(
        grad,
        weight,
        scale,
        weight_is_output_major=False,
    )
    oracle = _linear_input_grad_oracle(grad, weight, scale)
    if not bool(torch.isfinite(native).all() and torch.isfinite(oracle).all()):
        raise RuntimeError(f"nonfinite result at {rows=}")

    iterations = 20 if rows <= 256 else 5 if rows <= 1024 else 2
    native_ms = profiled(
        lambda: _native_block_fp8_mm(
            grad,
            weight,
            scale,
            weight_is_output_major=False,
        ),
        warmup=3,
        iterations=iterations,
    )
    oracle_ms = profiled(
        lambda: _linear_input_grad_oracle(grad, weight, scale),
        warmup=3,
        iterations=iterations,
    )
    activation = torch.randn(
        rows,
        weight.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.1)
    direct_forward = _native_block_fp8_mm(
        activation,
        weight,
        scale,
        weight_is_output_major=True,
    )
    direct_forward_ms = profiled(
        lambda: _native_block_fp8_mm(
            activation,
            weight,
            scale,
            weight_is_output_major=True,
        ),
        warmup=3,
        iterations=iterations,
    )
    for amplitude in (0.1, 1.0, 10.0, 100.0, 1000.0):
        stress_grad = grad * amplitude / 0.1
        stress_native = _native_block_fp8_mm(
            stress_grad,
            weight,
            scale,
            weight_is_output_major=False,
        )
        stress_oracle = _linear_input_grad_oracle(stress_grad, weight, scale)
        if not bool(
            torch.isfinite(stress_native).all()
            and torch.isfinite(stress_oracle).all()
        ):
            raise RuntimeError(f"nonfinite stress result at {rows=} {amplitude=}")

    diff = native.float() - oracle.float()
    oracle_absmax = oracle.float().abs().max()
    print(
        json.dumps(
            {
                "rows": rows,
                "native_ms": native_ms,
                "oracle_ms": oracle_ms,
                "native_over_oracle": native_ms / oracle_ms,
                "direct_forward_ms": direct_forward_ms,
                "native_vs_oracle_rel_rmse": float(
                    diff.square().mean().sqrt() / oracle_absmax
                ),
                "native_vs_oracle_cosine": float(
                    F.cosine_similarity(
                        native.float().flatten(),
                        oracle.float().flatten(),
                        dim=0,
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
