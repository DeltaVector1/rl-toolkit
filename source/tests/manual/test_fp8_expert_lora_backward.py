"""SM89 parity, memory, and timing test for native rank-32 expert LoRA."""

from __future__ import annotations

import argparse
import json

import torch

from prime_rl.trainer.models.layers.fp8_ada import (
    _expert_chunk_offsets,
    _grouped_lora_backward,
    _grouped_lora_projection,
)


def relative_rmse(value: torch.Tensor, reference: torch.Tensor) -> float:
    difference = value.float() - reference.float()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float(difference.square().mean().sqrt() / denominator)


def expert_offsets(rows: int, experts: int, device: torch.device) -> torch.Tensor:
    weights = torch.arange(1, experts + 1, dtype=torch.int64)
    counts = torch.div(weights * rows, weights.sum(), rounding_mode="floor")
    counts[-1] += rows - int(counts.sum())
    if int(counts.unique().numel()) == 1:
        raise RuntimeError("the benchmark requires uneven expert counts")
    return torch.cumsum(counts.to(device), 0, dtype=torch.int32)


def legacy_backward(x, a, b, offsets, grad, grad_a, grad_b):
    with torch.enable_grad():
        x_ref = x.detach().requires_grad_(True)
        a_ref = a.detach().requires_grad_(True)
        b_ref = b.detach().requires_grad_(True)
        output = _grouped_lora_projection(x_ref, a_ref, b_ref, offsets)
        input_grad, a_grad, b_grad = torch.autograd.grad(
            output,
            (x_ref, a_ref, b_ref),
            grad_outputs=grad,
        )
    grad_a.add_(a_grad)
    grad_b.add_(b_grad)
    return input_grad


def run_chunks(method, x, a, b, offsets, grad, chunk_rows):
    grad_a = torch.zeros_like(a)
    grad_b = torch.zeros_like(b)
    input_grad = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_rows):
        end = min(start + chunk_rows, x.shape[0])
        chunk_offsets = _expert_chunk_offsets(offsets, start, end)
        input_grad[start:end] = method(
            x[start:end],
            a,
            b,
            chunk_offsets,
            grad[start:end],
            grad_a,
            grad_b,
        )
    return input_grad, grad_a, grad_b


def measured_run(method, x, a, b, offsets, grad, chunk_rows):
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    values = run_chunks(method, x, a, b, offsets, grad, chunk_rows)
    end.record()
    end.synchronize()
    extra_mib = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
    return values, start.elapsed_time(end), extra_mib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this test requires an SM89 trainer GPU")
    if args.rows != 32768:
        raise ValueError("the qualification test requires exactly 32768 rows")

    torch.manual_seed(71)
    device = torch.device("cuda")
    offsets = expert_offsets(args.rows, 32, device)
    results = []
    for name, input_size, output_size in (
        ("w1", 6144, 2048),
        ("w3", 6144, 2048),
        ("w2", 2048, 6144),
    ):
        x = torch.randn(args.rows, input_size, device=device, dtype=torch.bfloat16)
        grad = torch.randn(
            args.rows,
            output_size,
            device=device,
            dtype=torch.bfloat16,
        )
        a = torch.randn(32, 32, input_size, device=device, dtype=torch.bfloat16)
        b = torch.randn(32, output_size, 32, device=device, dtype=torch.bfloat16)

        small_offsets = _expert_chunk_offsets(offsets, 0, args.chunk_rows)
        small_grad_a = torch.zeros_like(a)
        small_grad_b = torch.zeros_like(b)
        _grouped_lora_backward(
            x[: args.chunk_rows],
            a,
            b,
            small_offsets,
            grad[: args.chunk_rows],
            small_grad_a,
            small_grad_b,
        )
        torch.cuda.synchronize()
        del small_grad_a, small_grad_b

        reference, legacy_ms, legacy_peak = measured_run(
            legacy_backward,
            x,
            a,
            b,
            offsets,
            grad,
            args.chunk_rows,
        )
        optimized, optimized_ms, optimized_peak = measured_run(
            _grouped_lora_backward,
            x,
            a,
            b,
            offsets,
            grad,
            args.chunk_rows,
        )
        errors = {
            label: relative_rmse(value, expected)
            for label, value, expected in zip(
                ("input", "a", "b"),
                optimized,
                reference,
                strict=True,
            )
        }
        finite = all(bool(torch.isfinite(value).all()) for value in optimized)
        if not finite or max(errors.values()) > 0.02:
            raise RuntimeError(f"{name} parity failed: finite={finite}, errors={errors}")
        results.append(
            {
                "projection": name,
                "shape": [args.rows, input_size, output_size, 32],
                "relative_rmse": errors,
                "legacy_ms": legacy_ms,
                "optimized_ms": optimized_ms,
                "speedup": legacy_ms / optimized_ms,
                "legacy_peak_mib": legacy_peak,
                "optimized_peak_mib": optimized_peak,
            }
        )
        del x, grad, a, b, reference, optimized
        torch.cuda.empty_cache()

    print(json.dumps({"result": "PASS", "cases": results}, indent=2))


if __name__ == "__main__":
    main()
