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


def reference_pair(
    x: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    kv_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_low = x @ q_a.T
    kv_low = x @ kv_a.T
    return q_low @ q_b.T, kv_low @ kv_b.T


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


def _backward(
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
    upstream: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    outputs = implementation(*tensors)
    _assert_finite(outputs, "forward output")
    torch.autograd.backward(outputs, upstream)
    gradients = tuple(
        value.grad.detach().clone() for value in tensors if value.grad is not None
    )
    if len(gradients) != len(tensors):
        raise RuntimeError("backward did not produce every input or parameter gradient")
    _assert_finite(gradients, "gradient")
    return gradients


def _clear_gradients(tensors: tuple[torch.Tensor, ...]) -> None:
    for value in tensors:
        value.grad = None


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

    def backward() -> None:
        _backward(implementation, tensors, upstream)
        _clear_gradients(tensors)

    for _ in range(warmup):
        result = forward()
        _assert_finite(result, "warmup output")
        del result
        backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    forward_times = []
    for _ in range(iterations):
        result, elapsed = _cuda_timed(forward)
        _assert_finite(result, "forward output")
        forward_times.append(elapsed)
        del result
    backward_times = []
    for _ in range(iterations):
        _, elapsed = _cuda_timed(backward)
        backward_times.append(elapsed)
    torch.cuda.synchronize()
    return {
        "forward": _summary(forward_times),
        "backward": _summary(backward_times),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _compare(tokens: int, device: torch.device) -> dict[str, object]:
    reference_tensors = _make_tensors(tokens, device, 20260830)
    candidate_tensors = tuple(
        value.detach().clone().requires_grad_() for value in reference_tensors
    )
    upstream = _upstream(tokens, device, 20260831)
    reference_outputs = reference_pair(*reference_tensors)
    candidate_outputs = candidate_pair(*candidate_tensors)
    for actual, expected in zip(candidate_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    reference_gradients = _backward(reference_pair, reference_tensors, upstream)
    candidate_gradients = _backward(candidate_pair, candidate_tensors, upstream)
    torch.testing.assert_close(
        candidate_gradients[0],
        reference_gradients[0],
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    for actual, expected in zip(
        candidate_gradients[1:], reference_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    return {
        "forward": "passed",
        "input_gradient": "passed",
        "parameter_gradients": "passed",
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
                "bf16_tolerance": {"rtol": BF16_RTOL, "atol": BF16_ATOL},
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
