"""Bounded SM89 comparison for sparse MLA backward head tiles."""

from __future__ import annotations

import importlib
import os
import statistics
import sys
from collections.abc import Callable

import torch


qualification_profiles = frozenset(
    {
        "glm52-native-small",
        "glm52-native-b2048",
        "glm52-native-b2048-memory",
        "glm52-native-b2048-nsys",
        "glm52-native-delta3",
        "glm52-native-stability20",
    }
)
sequence_lengths = (32, 128)
backward_head_tiles = (16, 32, 64)
batch_size = 1
attention_heads = 64
selected_keys = 2048
backward_key_tile = 16
backward_threads = 64
warmup_runs = 1
timed_runs = 2
relative_tolerance = 0.05
absolute_scale_fraction = 0.01

expected_optional_settings = {
    "PRIME_SPARSE_BWD_BLOCK_SIZE": "16",
    "PRIME_SPARSE_BWD_THREADS": "64",
    "PRIME_SPARSE_FWD_BLOCK_SIZE": "32",
    "PRIME_SPARSE_FWD_BLOCK_HEADS": "32",
    "PRIME_SPARSE_FWD_THREADS": "128",
}
forbidden_settings = (
    "PRIME_SPARSE_BWD_BLOCK_HEADS",
    "PRIME_SPARSE_BWD_USE_BLOCK_H32",
)


def _environment_integer(name: str, default: int) -> int:
    text = os.environ.get(name)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _validate_runtime() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    visible_parts = visible.split(",")
    if (
        len(visible_parts) != 1
        or not visible_parts[0].strip()
        or visible_parts[0].strip() == "-1"
    ):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one GPU")

    process_settings = {
        "WORLD_SIZE": (1, 1),
        "LOCAL_WORLD_SIZE": (1, 1),
        "RANK": (0, 0),
        "LOCAL_RANK": (0, 0),
    }
    for name, (default, expected) in process_settings.items():
        if _environment_integer(name, default) != expected:
            raise RuntimeError("the benchmark requires one process with rank zero")

    profile = os.environ.get("PRIME_QUALIFICATION_PROFILE", "")
    if profile not in qualification_profiles:
        raise RuntimeError("an allowlisted GLM qualification profile is required")

    for name, expected in expected_optional_settings.items():
        configured = os.environ.get(name)
        if configured is not None and configured != expected:
            raise RuntimeError(f"{name} must be {expected} when configured")
    for name in forbidden_settings:
        if name in os.environ:
            raise RuntimeError(
                f"{name} must be unset; the benchmark selects each head tile directly"
            )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the benchmark requires exactly one visible CUDA GPU")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("the benchmark requires an SM89 GPU")


def _apply_exact_settings() -> None:
    for name, value in expected_optional_settings.items():
        os.environ[name] = value


def _validate_gradient(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0:
        raise RuntimeError(f"{name} is empty")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} is nonfinite")
    if int(torch.count_nonzero(value).item()) == 0:
        raise RuntimeError(f"{name} is zero")


def _comparison(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | int]:
    if candidate.shape != reference.shape:
        raise RuntimeError("gradient shapes differ")
    candidate_float = candidate.float()
    reference_float = reference.float()
    difference = (candidate_float - reference_float).abs()
    reference_scale = float(reference_float.abs().max().item())
    absolute_tolerance = max(reference_scale * absolute_scale_fraction, 1e-12)
    allowed = absolute_tolerance + relative_tolerance * reference_float.abs()
    relative = difference / reference_float.abs().clamp_min(absolute_tolerance)
    return {
        "max_abs": float(difference.max().item()),
        "max_rel": float(relative.max().item()),
        "absolute_tolerance": absolute_tolerance,
        "violations": int((difference > allowed).sum().item()),
    }


def _make_backward(
    backward_module,
    query: torch.Tensor,
    key_value: torch.Tensor,
    output: torch.Tensor,
    output_gradient: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    block_heads: int,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    heads = query.shape[2]
    dimension = 512
    tail_dimension = query.shape[-1] - dimension
    topk = indices.shape[-1]
    key_value_groups = key_value.shape[2]
    preprocess_kernel = backward_module.preprocess(heads, dimension)
    backward_kernel = backward_module.bwd(
        heads,
        dimension,
        tail_dimension,
        topk,
        key_value_groups,
        scale,
        True,
        block_size=backward_key_tile,
        block_H=block_heads,
        threads=backward_threads,
        vectorized_atomic=False,
    )
    postprocess_kernel = backward_module.postprocess(
        dimension, tail_dimension, key_value_groups
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        delta = preprocess_kernel(output, output_gradient)
        key_value_gradient_fp32 = torch.zeros_like(key_value, dtype=torch.float32)
        query_gradient = backward_kernel(
            query,
            key_value,
            output_gradient,
            indices,
            lse,
            delta,
            key_value_gradient_fp32,
        )
        key_value_gradient = postprocess_kernel(key_value_gradient_fp32)
        return query_gradient, key_value_gradient

    return run


def _measure(
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...], int]:
    for _ in range(warmup_runs):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    elapsed: list[float] = []
    query_gradient = key_value_gradient = None
    for _ in range(timed_runs):
        if query_gradient is not None:
            del query_gradient, key_value_gradient
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        query_gradient, key_value_gradient = run()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))

    assert query_gradient is not None and key_value_gradient is not None
    return (
        query_gradient,
        key_value_gradient,
        tuple(elapsed),
        torch.cuda.max_memory_allocated(),
    )


def _inputs(seq_len: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260828 + seq_len)
    query = torch.randn(
        batch_size,
        seq_len,
        attention_heads,
        576,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value = torch.randn(
        batch_size,
        seq_len + 1,
        1,
        576,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value[:, -1].zero_()
    key_positions = torch.arange(selected_keys, device="cuda", dtype=torch.int32)
    query_positions = torch.arange(
        1, seq_len + 1, device="cuda", dtype=torch.int32
    ).unsqueeze(1)
    indices = torch.where(
        key_positions.unsqueeze(0) < query_positions,
        key_positions,
        seq_len,
    )
    indices = indices.view(batch_size, seq_len, 1, selected_keys).contiguous()
    return query, key_value, indices


def _run_shape(backward_module, forward_module, seq_len: int) -> None:
    query, key_value, indices = _inputs(seq_len)
    scale = 576**-0.5
    output, lse = forward_module.sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=scale
    )
    generator = torch.Generator(device="cuda").manual_seed(20260928 + seq_len)
    output_gradient = torch.randn(
        output.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)

    results = {}
    for block_heads in backward_head_tiles:
        run = _make_backward(
            backward_module,
            query,
            key_value,
            output,
            output_gradient,
            indices,
            lse,
            scale,
            block_heads,
        )
        result = _measure(run)
        _validate_gradient(f"block_H={block_heads} dQ", result[0])
        _validate_gradient(f"block_H={block_heads} dKV", result[1])
        results[block_heads] = result
        print(
            f"seq_len={seq_len} block_H={block_heads} "
            f"median_ms={statistics.median(result[2]):.3f} "
            f"times_ms={','.join(f'{value:.3f}' for value in result[2])} "
            f"peak_allocated={result[3]}",
            flush=True,
        )

    baseline = results[16]
    for block_heads in (32, 64):
        candidate = results[block_heads]
        query_check = _comparison(candidate[0], baseline[0])
        key_value_check = _comparison(candidate[1], baseline[1])
        print(
            f"seq_len={seq_len} block_H={block_heads} baseline_H=16 "
            f"dQ_max_abs={query_check['max_abs']:.6g} "
            f"dQ_max_rel={query_check['max_rel']:.6g} "
            f"dQ_atol={query_check['absolute_tolerance']:.6g} "
            f"dQ_violations={query_check['violations']} "
            f"dKV_max_abs={key_value_check['max_abs']:.6g} "
            f"dKV_max_rel={key_value_check['max_rel']:.6g} "
            f"dKV_atol={key_value_check['absolute_tolerance']:.6g} "
            f"dKV_violations={key_value_check['violations']}",
            flush=True,
        )
        if query_check["violations"] or key_value_check["violations"]:
            raise RuntimeError(
                f"seq_len={seq_len} block_H={block_heads} exceeded the scale-aware tolerance"
            )


def main() -> None:
    if len(sys.argv) != 1:
        raise ValueError("this benchmark has no variable command-line options")
    _validate_runtime()
    _apply_exact_settings()
    backward_module = importlib.import_module(
        "prime_rl.trainer.models.kernels.sparse_mla_bwd"
    )
    forward_module = importlib.import_module(
        "prime_rl.trainer.models.kernels.sparse_mla_fwd"
    )
    print(
        "fixed_shape "
        f"batch={batch_size} heads={attention_heads} topk={selected_keys} "
        f"seq_lens={sequence_lengths} block_H={backward_head_tiles} "
        f"block_size={backward_key_tile} threads={backward_threads} "
        f"warmup={warmup_runs} repeats={timed_runs}",
        flush=True,
    )
    for seq_len in sequence_lengths:
        _run_shape(backward_module, forward_module, seq_len)


if __name__ == "__main__":
    main()
