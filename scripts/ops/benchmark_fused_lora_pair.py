from __future__ import annotations

import json
import statistics

import torch

from prime_rl.trainer.models.layers.fp8_ada import (
    _grouped_lora_projection,
    _grouped_lora_projection_pair,
)


ROWS = 32768
EXPERTS = 32
RANK = 32
WARMUP = 3
MEASURED = 7


def offsets_for_real_histogram(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(91)
    counts = torch.randint(768, 1281, (EXPERTS,), generator=generator)
    counts[-1] += ROWS - int(counts.sum())
    if int(counts.min()) <= 0 or int(counts.sum()) != ROWS:
        raise RuntimeError("invalid routed expert histogram")
    return torch.cumsum(counts.to(device), dim=0, dtype=torch.int32)


def pair_call(x, a1, b1, a3, b3, offsets, paired):
    if paired:
        return _grouped_lora_projection_pair(x, a1, b1, a3, b3, offsets)
    return (
        _grouped_lora_projection(x, a1, b1, offsets),
        _grouped_lora_projection(x, a3, b3, offsets),
    )


def timed_pair(x, a1, b1, a3, b3, offsets, paired):
    values = []
    for _ in range(WARMUP + MEASURED):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = pair_call(x, a1, b1, a3, b3, offsets, paired)
        end.record()
        end.synchronize()
        if len(values) >= WARMUP:
            values.append(start.elapsed_time(end))
        else:
            values.append(None)
        del output
    return [value for value in values if value is not None]


def measured_memory_pair(x, a1, b1, a3, b3, offsets, paired):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    outputs = pair_call(x, a1, b1, a3, b3, offsets, paired)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
    del outputs
    return peak


def backward_once(x, a1, b1, a3, b3, offsets, grad1, grad3, paired):
    x = x.detach().requires_grad_(True)
    a1 = a1.detach().requires_grad_(True)
    b1 = b1.detach().requires_grad_(True)
    a3 = a3.detach().requires_grad_(True)
    b3 = b3.detach().requires_grad_(True)
    outputs = pair_call(x, a1, b1, a3, b3, offsets, paired)
    torch.autograd.backward(outputs, (grad1, grad3))
    return x.grad, a1.grad, b1.grad, a3.grad, b3.grad


def timed_backward(x, a1, b1, a3, b3, offsets, grad1, grad3, paired):
    values = []
    for index in range(WARMUP + MEASURED):
        torch.cuda.synchronize()
        x_ref = x.detach().requires_grad_(True)
        a1_ref = a1.detach().requires_grad_(True)
        b1_ref = b1.detach().requires_grad_(True)
        a3_ref = a3.detach().requires_grad_(True)
        b3_ref = b3.detach().requires_grad_(True)
        outputs = pair_call(
            x_ref,
            a1_ref,
            b1_ref,
            a3_ref,
            b3_ref,
            offsets,
            paired,
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.backward(outputs, (grad1, grad3))
        end.record()
        end.synchronize()
        if index >= WARMUP:
            values.append(start.elapsed_time(end))
        del outputs, x_ref, a1_ref, b1_ref, a3_ref, b3_ref
    return values


def relative_rmse(value, reference):
    difference = value.float() - reference.float()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float(difference.square().mean().sqrt() / denominator)


def case(name, input_size, output_size, paired=True):
    device = torch.device("cuda")
    offsets = offsets_for_real_histogram(device)
    generator = torch.Generator(device="cpu").manual_seed(92 + input_size)
    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
    x = rand(ROWS, input_size)
    a1 = rand(EXPERTS, RANK, input_size)
    b1 = rand(EXPERTS, output_size, RANK)
    a3 = rand(EXPERTS, RANK, input_size)
    b3 = rand(EXPERTS, output_size, RANK)
    grad1 = rand(ROWS, output_size)
    grad3 = rand(ROWS, output_size)

    # Compile and warm both paths before collecting measurements.
    pair_call(x, a1, b1, a3, b3, offsets, paired)
    pair_call(x, a1, b1, a3, b3, offsets, False)
    torch.cuda.synchronize()

    expected = pair_call(x, a1, b1, a3, b3, offsets, False)
    actual = pair_call(x, a1, b1, a3, b3, offsets, paired)
    output_error = max(relative_rmse(got, want) for got, want in zip(actual, expected, strict=True))
    del expected, actual

    baseline_grads = backward_once(x, a1, b1, a3, b3, offsets, grad1, grad3, False)
    candidate_grads = backward_once(x, a1, b1, a3, b3, offsets, grad1, grad3, paired)
    gradient_error = max(
        relative_rmse(got, want)
        for got, want in zip(candidate_grads, baseline_grads, strict=True)
    )
    del baseline_grads, candidate_grads

    baseline_forward = timed_pair(x, a1, b1, a3, b3, offsets, False)
    candidate_forward = timed_pair(x, a1, b1, a3, b3, offsets, paired)
    baseline_backward = timed_backward(x, a1, b1, a3, b3, offsets, grad1, grad3, False)
    candidate_backward = timed_backward(x, a1, b1, a3, b3, offsets, grad1, grad3, paired)
    baseline_peak = measured_memory_pair(x, a1, b1, a3, b3, offsets, False)
    candidate_peak = measured_memory_pair(x, a1, b1, a3, b3, offsets, paired)

    result = {
        "name": name,
        "rows": ROWS,
        "experts": EXPERTS,
        "rank": RANK,
        "input_size": input_size,
        "output_size": output_size,
        "output_relative_rmse": output_error,
        "gradient_relative_rmse": gradient_error,
        "baseline_forward_ms_median": statistics.median(baseline_forward),
        "candidate_forward_ms_median": statistics.median(candidate_forward),
        "baseline_backward_ms_median": statistics.median(baseline_backward),
        "candidate_backward_ms_median": statistics.median(candidate_backward),
        "baseline_peak_mib": baseline_peak,
        "candidate_peak_mib": candidate_peak,
    }
    del x, a1, b1, a3, b3, grad1, grad3, offsets
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this benchmark requires an RTX 4090 SM89 GPU")
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cases": [
            case("w13_pair", 6144, 2048),
            case("w2_pair_shape", 2048, 6144),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
