"""Compare the GLM Q-A/KV-A LoRA pair on one SM89 GPU."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch


RANK = 32
HIDDEN = 6144
Q_OUTPUT = 2048
KV_OUTPUT = 576
TOKENS = 4096
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20
MAX_WARMUP = 10
MAX_ITERATIONS = 100
BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_NORM_RTOL = 5e-2
BF16_BYTES = 2
FUSED_COMBINED_A_TEMPORARY_BYTES = 2 * RANK * HIDDEN * BF16_BYTES


def reference_pair(
    x: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    kv_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from prime_rl.trainer.models.layers.lora.multi_linear import _SingleLoRA

    return (
        _SingleLoRA.apply(x, q_a, q_b, "sm89-smoke.q_a_proj"),
        _SingleLoRA.apply(x, kv_a, kv_b, "sm89-smoke.kv_a_proj_with_mqa"),
    )


def candidate_pair(
    x: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    kv_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from prime_rl.trainer.models.layers.lora.multi_linear import _FusedPairLoRA

    return _FusedPairLoRA.apply(x, q_a, q_b, kv_a, kv_b)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    return parser.parse_args()


def _check_arguments(args: argparse.Namespace) -> None:
    if not 0 <= args.warmup <= MAX_WARMUP:
        raise SystemExit(f"warmup must be between 0 and {MAX_WARMUP}")
    if not DEFAULT_ITERATIONS <= args.iterations <= MAX_ITERATIONS:
        raise SystemExit(
            f"iterations must be between {DEFAULT_ITERATIONS} and {MAX_ITERATIONS}"
        )
    if not 1 <= args.tokens <= 8192:
        raise SystemExit("tokens must be between 1 and 8192")


def _make_tensors(tokens: int, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def tensor(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
        return torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        ).mul_(scale)

    values = (
        tensor((tokens, HIDDEN)),
        tensor((RANK, HIDDEN)),
        tensor((Q_OUTPUT, RANK)),
        tensor((RANK, HIDDEN)),
        tensor((KV_OUTPUT, RANK)),
    )
    return tuple(value.requires_grad_() for value in values)


def _upstream(tokens: int, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            (tokens, Q_OUTPUT), device=device, dtype=torch.bfloat16, generator=generator
        ).mul_(1e-3),
        torch.randn(
            (tokens, KV_OUTPUT), device=device, dtype=torch.bfloat16, generator=generator
        ).mul_(1e-3),
    )


def _assert_finite(values: tuple[torch.Tensor, ...], label: str) -> None:
    if not all(bool(value.isfinite().all()) for value in values):
        raise RuntimeError(f"{label} contains a nonfinite value")


def _run_backward(
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
    upstream: tuple[torch.Tensor, ...],
) -> None:
    outputs = implementation(*tensors)
    torch.autograd.backward(outputs, upstream)


def _collect_gradients(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if any(value.grad is None for value in tensors):
        raise RuntimeError("backward did not produce every input or parameter gradient")
    gradients = tuple(value.grad.detach().clone() for value in tensors)
    _assert_finite(gradients, "gradient")
    return gradients


def _gradient_norms(
    gradients: tuple[torch.Tensor, ...], names: tuple[str, ...]
) -> dict[str, float]:
    norms = {}
    for name, gradient in zip(names, gradients, strict=True):
        norm = float(torch.linalg.vector_norm(gradient.float()).item())
        if not norm > 0.0:
            raise RuntimeError(f"reference {name} gradient norm is zero")
        norms[name] = norm
    return norms


def _clear_gradients(tensors: tuple[torch.Tensor, ...]) -> None:
    for value in tensors:
        value.grad = None


def _assert_gradients_finite(tensors: tuple[torch.Tensor, ...], label: str) -> None:
    if any(value.grad is None for value in tensors):
        raise RuntimeError(f"{label} is missing a gradient")
    _assert_finite(tuple(value.grad for value in tensors), label)


def _cuda_timed(callable: Callable[[], object]) -> tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p90_ms": sorted(values)[int(0.9 * (len(values) - 1))],
        "max_ms": max(values),
    }


def _measure(
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tokens: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    tensors = _make_tensors(tokens, device, 20260828)
    upstream = _upstream(tokens, device, 20260829)

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        return implementation(*tensors)

    def forward_backward() -> None:
        _run_backward(implementation, tensors, upstream)

    for _ in range(warmup):
        result = forward()
        _assert_finite(result, "warmup output")
        del result
        forward_backward()
        _assert_gradients_finite(tensors, "warmup gradient")
        _clear_gradients(tensors)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    for _ in range(iterations):
        result, elapsed = _cuda_timed(forward)
        _assert_finite(result, "forward output")
        forward_times.append(elapsed)
        del result
    forward_peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    forward_backward_times = []
    for _ in range(iterations):
        _, elapsed = _cuda_timed(forward_backward)
        _assert_gradients_finite(tensors, "gradient")
        forward_backward_times.append(elapsed)
        _clear_gradients(tensors)
    torch.cuda.synchronize()
    return {
        "forward": _summary(forward_times),
        "forward_backward": _summary(forward_backward_times),
        "peak_forward_allocated_bytes": forward_peak,
        "peak_backward_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
    }


def _error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    actual_norm = float(torch.linalg.vector_norm(actual_float).item())
    reference_scale = max(
        expected_norm / (expected_float.numel() ** 0.5),
        float(expected_float.abs().max().item()),
    )
    if not reference_scale > 0.0:
        raise RuntimeError("reference tensor scale is zero")
    allowed = BF16_ATOL * reference_scale + BF16_RTOL * expected_float.abs()
    relative = difference / reference_scale
    norm_relative = abs(actual_norm - expected_norm) / expected_norm
    return {
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float(relative.max().item()),
        "max_tolerance_ratio": float((difference / allowed).max().item()),
        "reference_scale": reference_scale,
        "reference_norm": expected_norm,
        "candidate_norm": actual_norm,
        "norm_relative_error": norm_relative,
    }


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, label: str
) -> dict[str, float]:
    metrics = _error_metrics(actual, expected)
    if not bool(
        torch.all(
            (actual.float() - expected.float()).abs()
            <= BF16_ATOL * metrics["reference_scale"]
            + BF16_RTOL * expected.float().abs()
        )
    ) or metrics["norm_relative_error"] > BF16_NORM_RTOL:
        raise RuntimeError(f"{label} exceeds BF16 tolerance: {metrics}")
    return metrics


def _expect_rejected(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    try:
        _assert_close(actual, expected, label)
    except RuntimeError as error:
        if label not in str(error):
            raise RuntimeError(f"{label} failure was not identified") from error
    else:
        raise RuntimeError(f"{label} invalid candidate was accepted")


def _perturbation_checks(values: dict[str, torch.Tensor]) -> None:
    for label, expected in values.items():
        expected_float = expected.float()
        scale = max(
            float(torch.linalg.vector_norm(expected_float).item())
            / (expected_float.numel() ** 0.5),
            float(expected_float.abs().max().item()),
        )
        if not scale > 0.0:
            raise RuntimeError(f"{label} reference scale is zero")
        actual = expected.detach().clone()
        actual.reshape(-1)[0] += 0.1 * scale
        _expect_rejected(actual, expected, f"{label} small perturbation")


def _zero_candidate_checks(values: dict[str, torch.Tensor]) -> None:
    for label, expected in values.items():
        _expect_rejected(
            torch.zeros_like(expected), expected, f"{label} zero candidate"
        )


def _compare(tokens: int, device: torch.device) -> dict[str, object]:
    reference_tensors = _make_tensors(tokens, device, 20260830)
    candidate_tensors = tuple(
        value.detach().clone().requires_grad_() for value in reference_tensors
    )
    upstream = _upstream(tokens, device, 20260831)
    reference_outputs = reference_pair(*reference_tensors)
    candidate_outputs = candidate_pair(*candidate_tensors)
    _assert_finite(reference_outputs, "reference output")
    _assert_finite(candidate_outputs, "candidate output")
    output_metrics = {
        name: _assert_close(actual, expected, f"{name} output")
        for name, actual, expected in zip(
            ("q_a", "kv_a"), candidate_outputs, reference_outputs, strict=True
        )
    }
    _run_backward(reference_pair, reference_tensors, upstream)
    reference_gradients = _collect_gradients(reference_tensors)
    reference_norms = _gradient_norms(
        reference_gradients, ("input", "q_a", "q_b", "kv_a", "kv_b")
    )
    validation_values = {
        "q_a output": reference_outputs[0],
        "kv_a output": reference_outputs[1],
        "input gradient": reference_gradients[0],
        "q_a gradient": reference_gradients[1],
        "q_b gradient": reference_gradients[2],
        "kv_a gradient": reference_gradients[3],
        "kv_b gradient": reference_gradients[4],
    }
    _zero_candidate_checks(validation_values)
    _perturbation_checks(validation_values)
    _clear_gradients(reference_tensors)
    _run_backward(candidate_pair, candidate_tensors, upstream)
    candidate_gradients = _collect_gradients(candidate_tensors)
    _clear_gradients(candidate_tensors)
    input_metrics = _assert_close(
        candidate_gradients[0], reference_gradients[0], "input gradient"
    )
    parameter_metrics = {
        name: _assert_close(actual, expected, f"{name} gradient")
        for name, actual, expected in zip(
            ("q_a", "q_b", "kv_a", "kv_b"),
            candidate_gradients[1:],
            reference_gradients[1:],
            strict=True,
        )
    }
    return {
        "forward": output_metrics,
        "input_gradient": input_metrics,
        "parameter_gradients": parameter_metrics,
        "reference_gradient_norms": reference_norms,
        "finite": "passed",
    }


def main() -> None:
    args = _arguments()
    _check_arguments(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise SystemExit(f"SM89 is required, found sm_{capability[0]}{capability[1]}")
    device = torch.device("cuda")
    comparison = _compare(args.tokens, device)
    baseline = _measure(reference_pair, args.tokens, device, args.warmup, args.iterations)
    fused = _measure(candidate_pair, args.tokens, device, args.warmup, args.iterations)
    fused["fused_combined_a_temporary_bytes"] = FUSED_COMBINED_A_TEMPORARY_BYTES
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(device),
                "capability": "sm_89",
                "shape": {
                    "tokens": args.tokens,
                    "hidden": HIDDEN,
                    "q_output": Q_OUTPUT,
                    "kv_output": KV_OUTPUT,
                    "rank": RANK,
                },
                "warmup": args.warmup,
                "iterations": args.iterations,
                "bf16_tolerance": {
                    "rtol": BF16_RTOL,
                    "atol_fraction_of_reference_scale": BF16_ATOL,
                    "norm_rtol": BF16_NORM_RTOL,
                },
                "comparison": comparison,
                "baseline": baseline,
                "fused": fused,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
