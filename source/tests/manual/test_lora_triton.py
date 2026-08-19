"""SM89 parity, memory, and timing gate for the rank-32 LoRA GEMMs."""

from __future__ import annotations

import argparse
import json
import time

import torch

from prime_rl.trainer.models.layers.lora.multi_linear import _lora_mm


def relative_rmse(value: torch.Tensor, reference: torch.Tensor) -> float:
    difference = value.float() - reference.float()
    return float(
        difference.square().mean().sqrt()
        / reference.float().square().mean().sqrt().clamp_min(1e-12)
    )


def elapsed_ms(function, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this gate requires an SM89 trainer GPU")

    torch.manual_seed(47)
    device = torch.device("cuda")
    results = []
    for name, input_size, output_size in (
        ("q_a", 6144, 2048),
        ("q_b", 1536, 24576),
        ("o", 24576, 6144),
    ):
        rows = args.rows
        x = torch.randn(rows, input_size, device=device, dtype=torch.bfloat16)
        a = torch.randn(32, input_size, device=device, dtype=torch.bfloat16)
        b = torch.randn(output_size, 32, device=device, dtype=torch.bfloat16)
        grad = torch.randn(rows, output_size, device=device, dtype=torch.bfloat16)

        a_out = _lora_mm(x, a.T)
        output = _lora_mm(a_out, b.T)
        grad_a = _lora_mm(grad, b)
        grad_x = _lora_mm(grad_a, a)
        grad_a_weight = _lora_mm(grad_a.T, x)
        grad_b_weight = _lora_mm(grad.T, a_out)

        # Use a bounded prefix for an explicit FP32 oracle. The production
        # shapes are still used for the timing and workspace measurements.
        parity_rows = min(rows, 128)
        x_prefix = x[:parity_rows]
        grad_prefix = grad[:parity_rows]
        a_out_ref = (x_prefix.float() @ a.float().T).to(torch.bfloat16)
        output_ref = (a_out_ref.float() @ b.float().T).to(torch.bfloat16)
        grad_a_ref = (grad_prefix.float() @ b.float()).to(torch.bfloat16)
        grad_x_ref = (grad_a_ref.float() @ a.float()).to(torch.bfloat16)
        grad_a_weight_ref = (
            grad_a_ref.float().T @ x_prefix.float()
        ).to(torch.bfloat16)
        grad_b_weight_ref = (
            grad_prefix.float().T @ a_out_ref.float()
        ).to(torch.bfloat16)

        errors = {
            "forward": relative_rmse(output[:parity_rows], output_ref),
            "grad_x": relative_rmse(grad_x[:parity_rows], grad_x_ref),
            "grad_a": relative_rmse(
                _lora_mm(grad_a[:parity_rows].T, x_prefix),
                grad_a_weight_ref,
            ),
            "grad_b": relative_rmse(
                _lora_mm(grad_prefix.T, a_out[:parity_rows]),
                grad_b_weight_ref,
            ),
        }
        if max(errors.values()) > 0.015:
            raise RuntimeError(f"{name} LoRA parity failed: {errors}")

        torch.cuda.reset_peak_memory_stats()
        _ = (
            _lora_mm(grad, b),
            _lora_mm(grad_a, a),
            _lora_mm(grad_a.T, x),
            _lora_mm(grad.T, a_out),
        )
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        def run_backward_gemms(
            grad=grad,
            b=b,
            grad_a=grad_a,
            a=a,
            x=x,
            a_out=a_out,
        ):
            return (
                _lora_mm(grad, b),
                _lora_mm(grad_a, a),
                _lora_mm(grad_a.T, x),
                _lora_mm(grad.T, a_out),
            )

        timing = elapsed_ms(
            run_backward_gemms,
            repeats=5,
        )
        results.append(
            {
                "name": name,
                "shape": [rows, input_size, output_size, 32],
                "relative_rmse": errors,
                "backward_four_gemms_ms": timing,
                "peak_allocated_mib": peak_mib,
                "outputs_finite": all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        output,
                        grad_x,
                        grad_a_weight,
                        grad_b_weight,
                    )
                ),
            }
        )
        del x, a, b, grad, a_out, output, grad_a, grad_x
        del grad_a_weight, grad_b_weight

    print(json.dumps({"result": "PASS", "cases": results}, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
