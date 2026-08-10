"""Measure dense FP8 input-gradient kernels across token-row counts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from prime_rl.trainer.models.layers.fp8_ada import (
    _grouped_mm,
    _linear_input_grad,
    _native_block_fp8_mm,
)


MODEL = Path("/model")
NAME = "model.layers.0.self_attn.o_proj.weight"
ROW_COUNTS = (16, 64, 256, 1024, 4096)


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
    offsets = torch.tensor([rows], device="cuda", dtype=torch.int32)

    native = _linear_input_grad(grad, weight, scale, fqn=NAME)
    fused = _grouped_mm(
        grad,
        weight.t().unsqueeze(0),
        scale.t().unsqueeze(0),
        offsets,
    )
    if not bool(torch.isfinite(native).all() and torch.isfinite(fused).all()):
        raise RuntimeError(f"nonfinite result at {rows=}")

    iterations = 20 if rows <= 256 else 5 if rows <= 1024 else 2
    native_ms = profiled(
        lambda: _linear_input_grad(grad, weight, scale, fqn=NAME),
        warmup=3,
        iterations=iterations,
    )
    fused_ms = profiled(
        lambda: _grouped_mm(
            grad,
            weight.t().unsqueeze(0),
            scale.t().unsqueeze(0),
            offsets,
        ),
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
    forward_offsets = torch.tensor(
        [rows], device="cuda", dtype=torch.int32
    )
    fused_forward = _grouped_mm(
        activation,
        weight.unsqueeze(0),
        scale.unsqueeze(0),
        forward_offsets,
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
    fused_forward_ms = profiled(
        lambda: _grouped_mm(
            activation,
            weight.unsqueeze(0),
            scale.unsqueeze(0),
            forward_offsets,
        ),
        warmup=3,
        iterations=iterations,
    )
    print(
        json.dumps(
            {
                "rows": rows,
                "native_ms": native_ms,
                "fused_transposed_ms": fused_ms,
                "native_over_fused": native_ms / fused_ms,
                "direct_forward_ms": direct_forward_ms,
                "fused_forward_ms": fused_forward_ms,
                "direct_forward_over_fused": (
                    direct_forward_ms / fused_forward_ms
                ),
                "forward_cross_cosine": float(
                    F.cosine_similarity(
                        direct_forward.float().flatten(),
                        fused_forward.float().flatten(),
                        dim=0,
                    )
                ),
                "cross_cosine": float(
                    F.cosine_similarity(
                        native.float().flatten(),
                        fused.float().flatten(),
                        dim=0,
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
