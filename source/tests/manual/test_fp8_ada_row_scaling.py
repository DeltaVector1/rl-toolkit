"""Measure dense FP8 input-gradient kernels across token-row counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from safetensors import safe_open

from prime_rl.trainer.models.layers.fp8_ada import (
    _linear_input_grad_oracle,
    _native_block_fp8_mm,
    _native_w8a16_input_grad,
)


MODEL = Path("/model")
ROW_COUNTS = (1, 16, 63, 64, 65, 128, 512, 2048)
STRESS_AMPLITUDES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MINIMUM_COSINE = 0.999
MAXIMUM_RELATIVE_RMSE = 0.005


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    default="model.layers.0.self_attn.o_proj.weight",
)
args = parser.parse_args()
NAME = args.name

if torch.cuda.get_device_capability() != (8, 9):
    raise RuntimeError("this qualification requires an RTX 4090 or another SM89 GPU")
print(
    json.dumps(
        {
            "cuda_capability": torch.cuda.get_device_capability(),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        sort_keys=True,
    ),
    flush=True,
)


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


def accuracy(native: torch.Tensor, oracle: torch.Tensor) -> tuple[float, float]:
    difference = native.float() - oracle.float()
    relative_rmse = float(
        difference.square().mean().sqrt()
        / oracle.float().abs().max().clamp_min(1e-12)
    )
    cosine = float(
        F.cosine_similarity(
            native.float().flatten(), oracle.float().flatten(), dim=0
        )
    )
    return cosine, relative_rmse


def assert_accurate(
    native: torch.Tensor,
    oracle: torch.Tensor,
    label: str,
) -> tuple[float, float]:
    if not bool(torch.isfinite(native).all() and torch.isfinite(oracle).all()):
        raise AssertionError(f"nonfinite result for {label}")
    cosine, relative_rmse = accuracy(native, oracle)
    if cosine < MINIMUM_COSINE:
        raise AssertionError(
            f"cosine {cosine:.8f} is below {MINIMUM_COSINE} for {label}"
        )
    if relative_rmse > MAXIMUM_RELATIVE_RMSE:
        raise AssertionError(
            f"relative RMSE {relative_rmse:.8f} exceeds "
            f"{MAXIMUM_RELATIVE_RMSE} for {label}"
        )
    return cosine, relative_rmse


torch.manual_seed(1234)
for rows, output_size, input_size in (
    (1, 128, 128),
    (7, 256, 384),
    (16, 130, 257),
    (65, 384, 256),
):
    synthetic_weight = (
        torch.randn(output_size, input_size, device="cuda")
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
    )
    synthetic_scale = torch.rand(
        (output_size + 127) // 128,
        (input_size + 127) // 128,
        device="cuda",
        dtype=torch.float32,
    ).mul_(0.5).add_(0.5)
    base_grad = torch.randn(
        rows, output_size, device="cuda", dtype=torch.bfloat16
    )
    for amplitude in STRESS_AMPLITUDES:
        synthetic_grad = base_grad * amplitude
        synthetic_native = _native_w8a16_input_grad(
            synthetic_grad, synthetic_weight, synthetic_scale
        )
        synthetic_oracle = _linear_input_grad_oracle(
            synthetic_grad, synthetic_weight, synthetic_scale
        )
        assert_accurate(
            synthetic_native,
            synthetic_oracle,
            f"synthetic rows={rows} weight={tuple(synthetic_weight.shape)} "
            f"amplitude={amplitude}",
        )


index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
    "weight_map"
]
scale_name = NAME.removesuffix(".weight") + ".weight_scale_inv"
with safe_open(MODEL / index[NAME], framework="pt") as handle:
    weight = handle.get_tensor(NAME).cuda()
with safe_open(MODEL / index[scale_name], framework="pt") as handle:
    scale = handle.get_tensor(scale_name).cuda()

for rows in ROW_COUNTS:
    grad = torch.randn(
        rows,
        weight.shape[0],
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.1)
    native = _native_w8a16_input_grad(grad, weight, scale)
    oracle = _linear_input_grad_oracle(grad, weight, scale)
    if not bool(torch.isfinite(native).all() and torch.isfinite(oracle).all()):
        raise RuntimeError(f"nonfinite result at {rows=}")

    iterations = 20 if rows <= 256 else 5 if rows <= 1024 else 2
    native_ms = profiled(
        lambda: _native_w8a16_input_grad(grad, weight, scale),
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
    direct_forward_ms = profiled(
        lambda: _native_block_fp8_mm(
            activation,
            weight,
            scale,
        ),
        warmup=3,
        iterations=iterations,
    )
    for amplitude in STRESS_AMPLITUDES:
        stress_grad = grad * amplitude / 0.1
        stress_native = _native_w8a16_input_grad(stress_grad, weight, scale)
        stress_oracle = _linear_input_grad_oracle(stress_grad, weight, scale)
        if not bool(
            torch.isfinite(stress_native).all()
            and torch.isfinite(stress_oracle).all()
        ):
            raise RuntimeError(f"nonfinite stress result at {rows=} {amplitude=}")
        assert_accurate(
            stress_native,
            stress_oracle,
            f"real rows={rows} amplitude={amplitude}",
        )

    cosine, relative_rmse = assert_accurate(native, oracle, f"real rows={rows}")
    print(
        json.dumps(
            {
                "rows": rows,
                "native_ms": native_ms,
                "oracle_ms": oracle_ms,
                "native_over_oracle": native_ms / oracle_ms,
                "direct_forward_ms": direct_forward_ms,
                "native_vs_oracle_rel_rmse": relative_rmse,
                "native_vs_oracle_cosine": cosine,
            },
            sort_keys=True,
        ),
        flush=True,
    )
