from __future__ import annotations

import gc
import json
import os
import statistics
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
from torchtitan.experiments.kernels.moe.indices import generate_permute_indices
from vllm import _custom_ops as ops

from prime_rl.trainer.distributed.expert_parallel import ReceivedFP8ExpertInput
import prime_rl.trainer.models.layers.fp8_ada as fp8_ada


SEED = 41517
SOURCE_ROWS = 8192
LOCAL_EXPERTS = 32
EP_SIZE = 8
WIDTH = 6144
INTERMEDIATE = 2048
RANK = 32
CHUNK_ROWS = 4096
WARMUP = 3
MEASURED = 7
STEP_MICRO_BATCHES = 64
MAX_LOCAL_RECOMPUTE_LAYERS_PER_STAGE = 11
MIN_MODELED_STEP_SAVING_SECONDS = 10.0
MAX_ADDED_GPU_MEMORY_MIB = 64.0
MAX_PINNED_CACHE_MIB = 80.0
MAX_CLEANUP_GPU_BYTES = 8 * 1024**2
LORA_SCALING = 2.0
FACTOR_NAMES = (
    "w1_lora_A",
    "w1_lora_B",
    "w2_lora_A",
    "w2_lora_B",
    "w3_lora_A",
    "w3_lora_B",
)


class BF16ForwardFP8Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        quantized, scales = ops.scaled_fp8_quant(
            grad,
            use_per_token_if_dynamic=True,
        )
        return quantized.to(grad.dtype) * scales.to(grad.dtype)


@dataclass
class Case:
    source: torch.Tensor
    scores: torch.Tensor
    upstream: torch.Tensor
    permutation: torch.Tensor
    offsets: torch.Tensor
    base_layer: SimpleNamespace
    factors: tuple[torch.Tensor, ...]


@dataclass
class PreparedRun:
    source: torch.Tensor
    scores: torch.Tensor
    upstream: torch.Tensor
    dispatched: torch.Tensor
    received: ReceivedFP8ExpertInput
    owner: SimpleNamespace
    factors: tuple[torch.Tensor, ...]


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def make_case(device: torch.device) -> Case:
    generator = torch.Generator(device=device).manual_seed(SEED)
    route_generator = torch.Generator(device="cpu").manual_seed(SEED + 1)

    def bf16_normal(*shape: int, scale: float = 1.0) -> torch.Tensor:
        value = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        return value.mul_(scale)

    def fp8_weight(*shape: int) -> torch.Tensor:
        return bf16_normal(*shape, scale=0.02).to(torch.float8_e4m3fn)

    def inverse_scales(*shape: int) -> torch.Tensor:
        return torch.empty(
            shape,
            device=device,
            dtype=torch.float32,
        ).uniform_(0.75, 1.25, generator=generator)

    w1 = fp8_weight(LOCAL_EXPERTS, INTERMEDIATE, WIDTH)
    w2 = fp8_weight(LOCAL_EXPERTS, WIDTH, INTERMEDIATE)
    w3 = fp8_weight(LOCAL_EXPERTS, INTERMEDIATE, WIDTH)
    s1 = inverse_scales(LOCAL_EXPERTS, INTERMEDIATE // 128, WIDTH // 128)
    s2 = inverse_scales(LOCAL_EXPERTS, WIDTH // 128, INTERMEDIATE // 128)
    s3 = inverse_scales(LOCAL_EXPERTS, INTERMEDIATE // 128, WIDTH // 128)
    base_layer = SimpleNamespace(
        w1=w1,
        w1_scale_inv=s1,
        w2=w2,
        w2_scale_inv=s2,
        w3=w3,
        w3_scale_inv=s3,
    )
    factors = (
        bf16_normal(LOCAL_EXPERTS, RANK, WIDTH, scale=0.01),
        bf16_normal(LOCAL_EXPERTS, INTERMEDIATE, RANK, scale=0.01),
        bf16_normal(LOCAL_EXPERTS, RANK, INTERMEDIATE, scale=0.01),
        bf16_normal(LOCAL_EXPERTS, WIDTH, RANK, scale=0.01),
        bf16_normal(LOCAL_EXPERTS, RANK, WIDTH, scale=0.01),
        bf16_normal(LOCAL_EXPERTS, INTERMEDIATE, RANK, scale=0.01),
    )
    assignments = torch.randint(
        EP_SIZE * LOCAL_EXPERTS,
        (SOURCE_ROWS,),
        generator=route_generator,
        dtype=torch.int64,
    )
    counts = torch.bincount(
        assignments,
        minlength=EP_SIZE * LOCAL_EXPERTS,
    ).to(device)
    with torch.no_grad():
        permutation, local_counts, _ = generate_permute_indices(
            counts,
            LOCAL_EXPERTS,
            EP_SIZE,
            SOURCE_ROWS + LOCAL_EXPERTS * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )
    permutation = permutation.to(torch.int64).contiguous()
    offsets = torch.cumsum(local_counts, dim=0, dtype=torch.int32)
    valid_routed_rows = int(offsets[-1])
    if (
        permutation.ndim != 1
        or valid_routed_rows < SOURCE_ROWS
        or valid_routed_rows > permutation.shape[0]
    ):
        raise RuntimeError("generated production routing layout is invalid")
    if offsets.shape != (LOCAL_EXPERTS,):
        raise RuntimeError("generated local expert offsets have the wrong shape")
    return Case(
        source=bf16_normal(SOURCE_ROWS, WIDTH).clamp_(-4, 4),
        scores=torch.empty(
            SOURCE_ROWS,
            device=device,
            dtype=torch.float32,
        ).uniform_(0.05, 0.95, generator=generator),
        upstream=bf16_normal(SOURCE_ROWS, WIDTH),
        permutation=permutation,
        offsets=offsets,
        base_layer=base_layer,
        factors=factors,
    )


def prepare_run(case: Case) -> PreparedRun:
    source = case.source.detach().clone().requires_grad_(True)
    scores = case.scores.detach().clone().requires_grad_(True)
    upstream = case.upstream.detach().clone()
    factors = tuple(
        value.detach().clone().requires_grad_(True) for value in case.factors
    )
    owner = SimpleNamespace(
        base_layer=case.base_layer,
        **{
            name: (value,)
            for name, value in zip(FACTOR_NAMES, factors, strict=True)
        },
    )
    received = ReceivedFP8ExpertInput(local_recompute=True)
    received.prepare_from_bf16(source.detach())
    dispatched = BF16ForwardFP8Backward.apply(source)
    torch.cuda.synchronize()
    return PreparedRun(
        source=source,
        scores=scores,
        upstream=upstream,
        dispatched=dispatched,
        received=received,
        owner=owner,
        factors=factors,
    )


def expert_forward(case: Case, run: PreparedRun, cache: bool) -> torch.Tensor:
    fp8_ada._FP8_EXPERT_ROUTE_OUTPUT_CACHE = cache
    return fp8_ada._InplacePermutedLoRAExperts.apply(
        run.dispatched,
        case.base_layer.w1,
        case.base_layer.w1_scale_inv,
        case.base_layer.w2,
        case.base_layer.w2_scale_inv,
        case.base_layer.w3,
        case.base_layer.w3_scale_inv,
        *run.factors,
        LORA_SCALING,
        case.permutation,
        case.offsets,
        run.owner,
        run.received,
        run.scores,
        CHUNK_ROWS,
    )


def collect_result(case: Case, cache: bool) -> dict[str, torch.Tensor]:
    run = prepare_run(case)
    output = expert_forward(case, run, cache)
    saved_output = output.detach().clone()
    output.backward(run.upstream)
    torch.cuda.synchronize()
    result = {
        "forward": saved_output.cpu(),
        "input_grad": run.source.grad.detach().cpu(),
        "route_score_grad": run.scores.grad.detach().cpu(),
    }
    result.update(
        {
            name: value.grad.detach().cpu()
            for name, value in zip(FACTOR_NAMES, run.factors, strict=True)
        }
    )
    return result


def compare_exact(
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool]]:
    report = {}
    for name in ("forward", "input_grad", "route_score_grad", *FACTOR_NAMES):
        got = candidate[name]
        expected = baseline[name]
        difference = got.float() - expected.float()
        denominator = expected.float().square().mean().sqrt().clamp_min(1e-12)
        report[name] = {
            "exact": bool(torch.equal(got, expected)),
            "max_abs": float(difference.abs().max()),
            "relative_rmse": float(
                difference.square().mean().sqrt() / denominator
            ),
        }
    return report


def timed_iteration(case: Case, cache: bool) -> tuple[float, float, float]:
    run = prepare_run(case)
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    start.record()
    output = expert_forward(case, run, cache)
    forward_done.record()
    output.backward(run.upstream)
    backward_done.record()
    backward_done.synchronize()
    forward_ms = start.elapsed_time(forward_done)
    backward_ms = forward_done.elapsed_time(backward_done)
    total_ms = start.elapsed_time(backward_done)
    return forward_ms, backward_ms, total_ms


def measure_times(case: Case) -> dict[str, dict[str, float | list[float]]]:
    for _ in range(WARMUP):
        timed_iteration(case, False)
        timed_iteration(case, True)
        gc.collect()
    samples = {False: [], True: []}
    for index in range(MEASURED):
        order = (False, True) if index % 2 == 0 else (True, False)
        for cache in order:
            samples[cache].append(timed_iteration(case, cache))
        gc.collect()

    result = {}
    for cache, name in ((False, "baseline"), (True, "cache")):
        forward = [item[0] for item in samples[cache]]
        backward = [item[1] for item in samples[cache]]
        total = [item[2] for item in samples[cache]]
        result[name] = {
            "forward_ms": forward,
            "backward_ms": backward,
            "total_ms": total,
            "forward_ms_median": statistics.median(forward),
            "backward_ms_median": statistics.median(backward),
            "total_ms_median": statistics.median(total),
        }
    return result


def measure_peak_memory(case: Case, cache: bool) -> dict[str, float]:
    torch.cuda.empty_cache()
    run = prepare_run(case)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = expert_forward(case, run, cache)
    output.backward(run.upstream)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "before_mib": before / 1024**2,
        "absolute_peak_mib": peak / 1024**2,
        "incremental_peak_mib": (peak - before) / 1024**2,
    }


def time_transfer(
    device_values: torch.Tensor,
    device_scales: torch.Tensor,
    host_values: torch.Tensor,
    host_scales: torch.Tensor,
    direction: str,
) -> list[float]:
    samples = []
    for index in range(WARMUP + MEASURED):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        copies = []
        for start in range(0, device_values.shape[0], CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, device_values.shape[0])
            if direction == "d2h":
                host_values[start:end].copy_(
                    device_values[start:end],
                    non_blocking=True,
                )
                host_scales[start:end].copy_(
                    device_scales[start:end],
                    non_blocking=True,
                )
            elif direction == "h2d":
                copies.append(host_values[start:end].to("cuda", non_blocking=True))
                copies.append(host_scales[start:end].to("cuda", non_blocking=True))
            else:
                raise ValueError(f"unknown transfer direction: {direction}")
        end_event.record()
        end_event.synchronize()
        if index >= WARMUP:
            samples.append(start_event.elapsed_time(end_event))
        del copies
    return samples


def measure_transfers(case: Case) -> dict[str, float | list[float]]:
    rows = case.permutation.shape[0]
    generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
    device_values = torch.randn(
        rows,
        WIDTH,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    device_scales = torch.empty(
        rows,
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.01, 0.1, generator=generator)
    host_values = torch.empty_like(
        device_values,
        device="cpu",
        pin_memory=True,
    )
    host_scales = torch.empty_like(
        device_scales,
        device="cpu",
        pin_memory=True,
    )
    pinned_bytes = tensor_bytes(host_values) + tensor_bytes(host_scales)
    d2h = time_transfer(
        device_values,
        device_scales,
        host_values,
        host_scales,
        "d2h",
    )
    h2d = time_transfer(
        device_values,
        device_scales,
        host_values,
        host_scales,
        "h2d",
    )
    return {
        "pinned_cache_bytes": pinned_bytes,
        "pinned_cache_mib": pinned_bytes / 1024**2,
        "d2h_ms": d2h,
        "d2h_ms_median": statistics.median(d2h),
        "h2d_ms": h2d,
        "h2d_ms_median": statistics.median(h2d),
    }


def verify_cache_cleanup(case: Case) -> bool:
    def run_probe() -> list[weakref.ReferenceType[torch.Tensor]]:
        run = prepare_run(case)
        output = expert_forward(case, run, True)
        saved = output.grad_fn.saved_tensors
        pinned = [
            value
            for value in saved
            if value.device.type == "cpu" and value.is_pinned()
        ]
        if len(pinned) != 2:
            raise RuntimeError(
                f"expected two pinned route-cache tensors, found {len(pinned)}"
            )
        references = [weakref.ref(value) for value in pinned]
        output.backward(run.upstream)
        torch.cuda.synchronize()
        return references

    references = run_probe()
    gc.collect()
    torch.cuda.empty_cache()
    return all(reference() is None for reference in references)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this benchmark requires one RTX 4090 SM89 GPU")
    if not hasattr(fp8_ada, "_FP8_EXPERT_ROUTE_OUTPUT_CACHE"):
        raise RuntimeError("route-output cache prototype is not installed")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    case = make_case(torch.device("cuda"))
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated()

    baseline = collect_result(case, False)
    candidate = collect_result(case, True)
    parity = compare_exact(candidate, baseline)
    del baseline, candidate
    gc.collect()

    times = measure_times(case)
    baseline_memory = measure_peak_memory(case, False)
    candidate_memory = measure_peak_memory(case, True)
    transfers = measure_transfers(case)
    cleanup_ok = verify_cache_cleanup(case)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup_gpu_bytes = max(0, torch.cuda.memory_allocated() - resident_bytes)

    call_saving_ms = (
        times["baseline"]["total_ms_median"]
        - times["cache"]["total_ms_median"]
    )
    modeled_step_saving_seconds = (
        call_saving_ms
        * STEP_MICRO_BATCHES
        * MAX_LOCAL_RECOMPUTE_LAYERS_PER_STAGE
        / 1000.0
    )
    added_gpu_memory_mib = (
        candidate_memory["incremental_peak_mib"]
        - baseline_memory["incremental_peak_mib"]
    )
    parity_ok = all(item["exact"] for item in parity.values())
    limits = {
        "forward_exact": True,
        "input_gradient_exact": True,
        "lora_a_b_gradients_exact": True,
        "route_score_gradient_exact": True,
        "min_modeled_step_saving_seconds": MIN_MODELED_STEP_SAVING_SECONDS,
        "max_added_gpu_memory_mib": MAX_ADDED_GPU_MEMORY_MIB,
        "max_pinned_cache_mib": MAX_PINNED_CACHE_MIB,
        "max_cleanup_gpu_bytes": MAX_CLEANUP_GPU_BYTES,
    }
    checks = {
        "parity": parity_ok,
        "modeled_step_saving": (
            modeled_step_saving_seconds >= MIN_MODELED_STEP_SAVING_SECONDS
        ),
        "gpu_memory": added_gpu_memory_mib <= MAX_ADDED_GPU_MEMORY_MIB,
        "pinned_memory": transfers["pinned_cache_mib"] <= MAX_PINNED_CACHE_MIB,
        "saved_tensor_cleanup": cleanup_ok,
        "gpu_cleanup": cleanup_gpu_bytes <= MAX_CLEANUP_GPU_BYTES,
    }
    report = {
        "profile": {
            "seed": SEED,
            "source_rows": SOURCE_ROWS,
            "padded_rows": case.permutation.shape[0],
            "local_experts": LOCAL_EXPERTS,
            "ep_size": EP_SIZE,
            "width": WIDTH,
            "intermediate": INTERMEDIATE,
            "rank": RANK,
            "chunk_rows": CHUNK_ROWS,
            "warmup": WARMUP,
            "measured": MEASURED,
            "step_micro_batches": STEP_MICRO_BATCHES,
            "max_local_recompute_layers_per_stage": (
                MAX_LOCAL_RECOMPUTE_LAYERS_PER_STAGE
            ),
            "native_fp8_base": True,
            "fp32_block_inverse_scale": [128, 128],
            "bf16_lora": True,
            "source_major_ep_routing": True,
        },
        "device": torch.cuda.get_device_name(),
        "parity": parity,
        "times": times,
        "transfers": transfers,
        "memory": {
            "baseline": baseline_memory,
            "cache": candidate_memory,
            "added_gpu_memory_mib": added_gpu_memory_mib,
            "cleanup_gpu_bytes": cleanup_gpu_bytes,
        },
        "call_saving_ms": call_saving_ms,
        "modeled_step_saving_seconds": modeled_step_saving_seconds,
        "limits": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"route-output cache rejected: {', '.join(failures)}")


if __name__ == "__main__":
    main()
