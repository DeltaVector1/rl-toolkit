"""Eight-rank native FP8 expert LoRA transport qualification.

Run this file with torchrun on one eight-GPU RTX 4090 host. It uses the
production expert module, expert-parallel ownership and permutation hooks,
route-score scaling, gradient clipping, and Muon optimizer.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path


os.environ.setdefault("PRIME_LORA_MM_PROFILE", "sm89-r32")
os.environ.setdefault("PRIME_INT4_COMBINE_ROUNDS", "8")
os.environ.setdefault("PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD", "0")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M

from prime_rl.configs.trainer import ConstantSchedulerConfig, MuonConfig
from prime_rl.trainer.distributed.expert_parallel import (
    NativeFP8LoRAAllToAllExpertParallel,
)
from prime_rl.trainer.models.layers.fp8_ada import scale_routed_output
from prime_rl.trainer.models.layers.lora.native_fp8_moe import (
    NativeFP8MultiLoRAGroupedExperts,
)
from prime_rl.trainer.models.layers.lora import (
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.scheduler import setup_scheduler


DIM = 128
HIDDEN_DIM = 256
NUM_EXPERTS = 256
RANK = 32
ALPHA = 64.0
PROJECTIONS = ("w1", "w2", "w3")
FACTORS = ("A", "B")
RUNTIME_FILES = (
    "configs/qualification/glm52-native-b2048-int4.toml",
    "configs/qualification/glm52-int4-fixed-token-trainer.toml",
    "scripts/china/cluster.sh",
    "source/src/prime_rl/trainer/lora.py",
    "source/src/prime_rl/trainer/model.py",
    "source/src/prime_rl/trainer/distributed/expert_parallel.py",
    "source/src/prime_rl/trainer/models/layers/fp8_ada.py",
    "source/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py",
    "source/src/prime_rl/trainer/models/layers/moe.py",
    "source/src/prime_rl/trainer/parallel_dims.py",
    "source/tests/manual/test_native_int4_expert_lora.py",
    "image-runtime/prime_rl/trainer/models/layers/lora/base.py",
    "image-runtime/prime_rl/trainer/optim.py",
    "image-runtime/prime_rl/trainer/scheduler.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--mounts-json", required=True)
    parser.add_argument("--mount-root", required=True)
    return parser.parse_args()


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_manifest(root: Path) -> dict[str, str]:
    missing = [relative for relative in RUNTIME_FILES if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"runtime manifest files are missing: {missing}")
    return {relative: sha256(root / relative) for relative in RUNTIME_FILES}


def all_rank_pair_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.detach().float()
    right = right.detach().float()
    if left.shape != right.shape:
        raise RuntimeError(f"metric shapes differ: {left.shape} and {right.shape}")
    difference = left - right
    square_sum = difference.square().sum(dtype=torch.float64)
    left_square_sum = left.square().sum(dtype=torch.float64)
    right_square_sum = right.square().sum(dtype=torch.float64)
    count = torch.tensor(left.numel(), device=left.device, dtype=torch.float64)
    maximum = difference.abs().max().to(torch.float64) if left.numel() else square_sum
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(left_square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(right_square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    divisor = max(count.item(), 1.0)
    rmse = math.sqrt(square_sum.item() / divisor)
    left_rms = math.sqrt(left_square_sum.item() / divisor)
    right_rms = math.sqrt(right_square_sum.item() / divisor)
    return {
        "rmse": rmse,
        "max_abs": maximum.item(),
        "elements": int(count.item()),
        "left_rms": left_rms,
        "right_rms": right_rms,
        "relative_rmse_to_left": rmse / max(left_rms, 1e-30),
    }


def all_rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def all_rank_sum(value: int, device: torch.device) -> int:
    result = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return int(result.item())


def factor_snapshot(module: torch.nn.Module, gradient: bool) -> dict[str, torch.Tensor]:
    result = {}
    for projection in PROJECTIONS:
        for factor in FACTORS:
            parameter = getattr(module, f"{projection}_lora_{factor}")[0]
            value = parameter.grad if gradient else parameter
            if value is None:
                raise RuntimeError(f"{projection} LoRA {factor} has no gradient")
            result[f"{projection}_{factor}"] = local_tensor(value).detach().clone()
    return result


def joined(values: dict[str, torch.Tensor], factor: str) -> torch.Tensor:
    return torch.cat(
        [values[f"{projection}_{factor}"].reshape(-1) for projection in PROJECTIONS]
    )


def clip_expert_gradients(module: torch.nn.Module, max_norm: float) -> torch.Tensor:
    square_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    gradients = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError("trainable expert parameter has no gradient")
        gradient = local_tensor(parameter.grad)
        gradients.append(gradient)
        square_sum.add_(gradient.float().square().sum(dtype=torch.float64))
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
    norm = square_sum.sqrt()
    coefficient = min(1.0, max_norm / (norm.item() + 1e-6))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return norm


def initialize_module(device: torch.device) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    torch.manual_seed(9137)
    torch.cuda.manual_seed(9137)
    base = GroupedExperts(
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        use_grouped_mm=True,
        fp8_checkpoint=True,
    ).to(device)
    module = NativeFP8MultiLoRAGroupedExperts(
        base,
        rank=RANK,
        n_adapters=1,
        alpha=ALPHA,
        dropout=0.0,
    )
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(417)
        for name in ("w1", "w2", "w3"):
            weight = getattr(base, name)
            value = torch.randn(
                weight.shape,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ).mul_(0.03)
            weight.copy_(value.to(weight.dtype))
            getattr(base, f"{name}_scale_inv").fill_(1.0)
        for projection in PROJECTIONS:
            for factor in FACTORS:
                value = getattr(module, f"{projection}_lora_{factor}")[0]
                value.normal_(mean=0.0, std=0.05, generator=generator)
    full_values = {
        f"{projection}_{factor}": getattr(module, f"{projection}_lora_{factor}")[
            0
        ].detach().clone()
        for projection in PROJECTIONS
        for factor in FACTORS
    }
    for projection in PROJECTIONS:
        full_values[f"base_{projection}"] = getattr(base, projection).detach().clone()
    return module, full_values


def deterministic_routes(
    rank: int, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
]:
    counts_cpu = torch.tensor(
        [0 if expert % 17 == 0 else 1 + (expert + rank) % 3 for expert in range(NUM_EXPERTS)],
        dtype=torch.int64,
    )
    expert_ids = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, dtype=torch.int64), counts_cpu
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    routed_input = torch.randn(
        (int(counts_cpu.sum()), DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.2)
    duplicate_rows = 0
    cursor = 0
    for count in counts_cpu.tolist():
        if count >= 2:
            routed_input[cursor + 1].copy_(routed_input[cursor])
            duplicate_rows += 1
        cursor += count
    scores = (
        0.35
        + 0.6
        * torch.rand(
            counts_cpu.sum().item(),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    )
    upstream = torch.randn(
        routed_input.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.125)
    cursor = 0
    for count in counts_cpu.tolist():
        if count >= 2:
            if scores[cursor] == scores[cursor + 1]:
                scores[cursor + 1].add_(0.125)
            upstream[cursor + 1].copy_(upstream[cursor])
        cursor += count
    return (
        routed_input,
        counts_cpu.to(device),
        expert_ids,
        scores,
        upstream,
        {"source_rows": routed_input.shape[0], "duplicate_rows": duplicate_rows},
    )


def expert_reference(
    routed_input: torch.Tensor,
    expert_ids: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> torch.Tensor:
    result = torch.empty_like(routed_input)
    scaling = ALPHA / RANK
    for expert in range(NUM_EXPERTS):
        rows = torch.nonzero(expert_ids == expert, as_tuple=False).flatten()
        if not rows.numel():
            continue
        x = routed_input.index_select(0, rows)
        z1 = torch.matmul(x, values["base_w1"][expert].to(x.dtype).transpose(0, 1))
        z1_lora = torch.matmul(x, values["w1_A"][expert].transpose(0, 1))
        z1.add_(torch.matmul(z1_lora, values["w1_B"][expert].transpose(0, 1)), alpha=scaling)
        z3 = torch.matmul(x, values["base_w3"][expert].to(x.dtype).transpose(0, 1))
        z3_lora = torch.matmul(x, values["w3_A"][expert].transpose(0, 1))
        z3.add_(torch.matmul(z3_lora, values["w3_B"][expert].transpose(0, 1)), alpha=scaling)
        hidden = F.silu(z1).mul_(z3)
        output = torch.matmul(hidden, values["base_w2"][expert].to(x.dtype).transpose(0, 1))
        output_lora = torch.matmul(hidden, values["w2_A"][expert].transpose(0, 1))
        output.add_(torch.matmul(output_lora, values["w2_B"][expert].transpose(0, 1)), alpha=scaling)
        result.index_copy_(0, rows, output)
    return result


def duplicate_stats(
    expert_output: torch.Tensor,
    scaled_output: torch.Tensor,
    input_gradient: torch.Tensor,
    score_gradient: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    counts_cpu = counts.cpu().tolist()
    expert_max = torch.zeros((), device=device, dtype=torch.float32)
    score_min = torch.full((), float("inf"), device=device, dtype=torch.float32)
    scaled_max = torch.zeros((), device=device, dtype=torch.float32)
    input_grad_max = torch.zeros((), device=device, dtype=torch.float32)
    score_grad_max = torch.zeros((), device=device, dtype=torch.float32)
    pair_counts = torch.zeros(3, device=device, dtype=torch.int64)
    cursor = 0
    for count in counts_cpu:
        if count >= 2:
            pair_counts[0] += 1
            expert_max = torch.maximum(
                expert_max,
                (expert_output[cursor] - expert_output[cursor + 1]).float().abs().max(),
            )
            score_min = torch.minimum(score_min, (scores[cursor] - scores[cursor + 1]).abs())
            scaled_difference = (
                scaled_output[cursor] - scaled_output[cursor + 1]
            ).float().abs().max()
            input_grad_difference = (
                input_gradient[cursor] - input_gradient[cursor + 1]
            ).float().abs().max()
            scaled_max = torch.maximum(scaled_max, scaled_difference)
            input_grad_max = torch.maximum(input_grad_max, input_grad_difference)
            pair_counts[1] += scaled_difference > 0
            pair_counts[2] += input_grad_difference > 0
            score_grad_max = torch.maximum(
                score_grad_max,
                (score_gradient[cursor] - score_gradient[cursor + 1]).abs(),
            )
        cursor += count
    dist.all_reduce(expert_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(score_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(scaled_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(input_grad_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(score_grad_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(pair_counts, op=dist.ReduceOp.SUM)
    return {
        "pairs": int(pair_counts[0]),
        "scaled_output_nonzero_pairs": int(pair_counts[1]),
        "input_gradient_nonzero_pairs": int(pair_counts[2]),
        "expert_output_max_abs": expert_max.item(),
        "score_min_abs": score_min.item(),
        "scaled_output_max_abs": scaled_max.item(),
        "input_gradient_max_abs": input_grad_max.item(),
        "route_score_gradient_max_abs": score_grad_max.item(),
    }


def gradient_coverage(
    gradients: dict[str, torch.Tensor], ep_rank: int
) -> dict[str, float]:
    local_experts = NUM_EXPERTS // 8
    global_ids = torch.arange(
        ep_rank * local_experts,
        (ep_rank + 1) * local_experts,
        device=next(iter(gradients.values())).device,
    )
    zero_mask = global_ids % 17 == 0
    active_mask = ~zero_mask
    zero_max = torch.zeros((), device=global_ids.device, dtype=torch.float32)
    active_min = torch.full((), float("inf"), device=global_ids.device, dtype=torch.float32)
    for gradient in gradients.values():
        zero_max = torch.maximum(
            zero_max,
            gradient[zero_mask].float().abs().max()
            if bool(zero_mask.any())
            else zero_max,
        )
        active_l1 = gradient[active_mask].float().abs().flatten(1).sum(1)
        active_min = torch.minimum(active_min, active_l1.min())
    dist.all_reduce(zero_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(active_min, op=dist.ReduceOp.MIN)
    return {
        "zero_expert_gradient_max_abs": zero_max.item(),
        "active_expert_gradient_min_l1": active_min.item(),
    }


def padding_stats(counts: torch.Tensor, ep_mesh) -> dict[str, int]:
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts, group=ep_mesh.get_group())
    local_experts = NUM_EXPERTS // dist.get_world_size(ep_mesh.get_group())
    local_counts = received.view(dist.get_world_size(ep_mesh.get_group()), local_experts).sum(0)
    padded = (
        torch.div(
            local_counts + TOKEN_GROUP_ALIGN_SIZE_M - 1,
            TOKEN_GROUP_ALIGN_SIZE_M,
            rounding_mode="floor",
        )
        * TOKEN_GROUP_ALIGN_SIZE_M
    )
    local = torch.tensor(
        [
            int((local_counts == 0).sum()),
            int(((local_counts > 0) & (local_counts % TOKEN_GROUP_ALIGN_SIZE_M != 0)).sum()),
            int((padded - local_counts).sum()),
        ],
        device=counts.device,
        dtype=torch.int64,
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return {
        "zero_local_experts": int(local[0]),
        "unaligned_local_experts": int(local[1]),
        "padding_rows": int(local[2]),
        "alignment": int(TOKEN_GROUP_ALIGN_SIZE_M),
    }


def run_sequence(
    transport: str,
    parallel_dims: ParallelDims,
    ep_mesh,
    sequence_index: int,
) -> dict:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    module, full_factors = initialize_module(device)
    style = NativeFP8LoRAAllToAllExpertParallel(transport)
    module = style._apply(module, ep_mesh)
    named_parameters = [
        (f"model.layers.0.mlp.experts.{name}", parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = setup_optimizer(
        MuonConfig(lr=8e-6, weight_decay=0.0, max_norm=1.0),
        named_parameters,
        parallel_dims,
        lora=False,
        cpu_offload=False,
    )
    scheduler = setup_scheduler(
        optimizer,
        ConstantSchedulerConfig(),
        max_steps=2,
        lr=8e-6,
    )
    routed_input, counts, expert_ids, scores, upstream, route_info = deterministic_routes(
        dist.get_rank(), device
    )
    original_input = routed_input.detach().clone()
    explicit_reference = expert_reference(original_input, expert_ids, full_factors)
    base_values = [full_factors[f"base_{projection}"].float() for projection in PROJECTIONS]
    base_count = sum(value.numel() for value in base_values)
    base_square_sum = sum(value.square().sum(dtype=torch.float64) for value in base_values)
    base_stats = {
        "nonzero_count": sum(int(torch.count_nonzero(value)) for value in base_values),
        "rms": math.sqrt(float(base_square_sum) / base_count),
        "max_abs": max(float(value.abs().max()) for value in base_values),
    }
    if base_stats["nonzero_count"] <= 0 or base_stats["rms"] <= 0.0:
        raise RuntimeError("native FP8 base expert weights are zero")
    del full_factors
    steps = []
    optimizer_state_entries = []
    for step in range(2):
        input_value = routed_input.detach().clone().requires_grad_()
        score_value = scores.detach().clone().requires_grad_()
        dist.barrier()
        forward_start = time.perf_counter()
        expert_output = module(input_value, counts)
        expert_output_copy = expert_output.detach().clone()
        scaled_output = scale_routed_output(expert_output, score_value)
        scaled_output_copy = scaled_output.detach().clone()
        loss = (scaled_output.float() * upstream.float()).sum()
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_start
        gradients = factor_snapshot(module, gradient=True)
        coverage = gradient_coverage(gradients, ep_mesh.get_local_rank())
        duplicate = duplicate_stats(
            expert_output_copy,
            scaled_output_copy,
            input_value.grad,
            score_value.grad,
            scores,
            counts,
            device,
        )
        before_update = factor_snapshot(module, gradient=False)
        grad_norm = clip_expert_gradients(module, max_norm=1.0)
        update_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        if any(
            parameter.grad is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("production Muon did not clear every expert gradient")
        scheduler.step()
        torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_start
        after_update = factor_snapshot(module, gradient=False)
        update_a = all_rank_pair_stats(joined(before_update, "A"), joined(after_update, "A"))
        update_b = all_rank_pair_stats(joined(before_update, "B"), joined(after_update, "B"))
        if update_a["max_abs"] == 0.0 and update_b["max_abs"] == 0.0:
            raise RuntimeError("production Muon did not update any expert LoRA parameter")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                expert_output_copy,
                scaled_output_copy,
                input_value.grad,
                score_value.grad,
                *gradients.values(),
                *after_update.values(),
            )
        ):
            raise RuntimeError("native expert LoRA run produced a nonfinite value")
        if (
            duplicate["expert_output_max_abs"] != 0.0
            or duplicate["score_min_abs"] <= 0.0
            or duplicate["scaled_output_nonzero_pairs"] <= 0
            or duplicate["input_gradient_nonzero_pairs"] <= 0
            or duplicate["route_score_gradient_max_abs"] != 0.0
        ):
            raise RuntimeError(
                f"nonuniform scored duplicate-route contract failed: {duplicate}"
            )
        if (
            coverage["zero_expert_gradient_max_abs"] != 0.0
            or coverage["active_expert_gradient_min_l1"] <= 0.0
        ):
            raise RuntimeError("zero or active expert gradient coverage failed")
        state_entries = len(optimizer.state_dict().get("state", {}))
        optimizer_state_entries.append(state_entries)
        if state_entries <= 0:
            raise RuntimeError("production Muon did not retain optimizer state")
        expert_reference_stats = (
            all_rank_pair_stats(expert_output_copy, explicit_reference)
            if step == 0
            else None
        )
        reference_limit = 0.25 if transport == "int4" else 0.05
        if (
            expert_reference_stats is not None
            and expert_reference_stats["relative_rmse_to_left"] > reference_limit
        ):
            raise RuntimeError(
                "native FP8 base plus LoRA expert ownership reference failed: "
                f"limit={reference_limit}, metrics={expert_reference_stats}"
            )
        metadata = {
            **route_info,
            "duplicate": duplicate,
            "gradient_coverage": coverage,
            "expert_id_reference": expert_reference_stats,
            "expert_id_reference_relative_rmse_limit": reference_limit,
            "grad_norm": all_rank_max(float(grad_norm), device),
            "forward_seconds": all_rank_max(forward_seconds, device),
            "backward_seconds": all_rank_max(backward_seconds, device),
            "muon_update_seconds": all_rank_max(update_seconds, device),
            "muon_parameter_update_A": update_a,
            "muon_parameter_update_B": update_b,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_state_entries": state_entries,
            "scheduler_last_epoch": int(scheduler.state_dict()["last_epoch"]),
            "step": step,
            "sequence_index": sequence_index,
        }
        steps.append(
            {
                "expert_output": expert_output_copy,
                "scaled_output": scaled_output_copy,
                "input_gradient": input_value.grad.detach().clone(),
                "route_score_gradient": score_value.grad.detach().clone(),
                "gradients": gradients,
                "updated_parameters": after_update,
                "metadata": metadata,
            }
        )
        del input_value, score_value, scaled_output, expert_output, loss
    peak_bytes = torch.cuda.max_memory_allocated(device)
    metadata = {
        **route_info,
        "base_weight_stats": base_stats,
        "peak_memory_bytes": int(all_rank_max(float(peak_bytes), device)),
        "optimizer_state_entries": optimizer_state_entries,
        "scheduler_last_epoch": int(scheduler.state_dict()["last_epoch"]),
        "sequence_index": sequence_index,
    }
    if optimizer_state_entries[1] != optimizer_state_entries[0]:
        raise RuntimeError("production Muon state inventory changed after initialization")
    del optimizer, scheduler, module
    gc.collect()
    torch.cuda.empty_cache()
    return {"steps": steps, "metadata": metadata}


def comparison(left: dict, right: dict) -> dict:
    metrics = {
        "expert_output": all_rank_pair_stats(left["expert_output"], right["expert_output"]),
        "scaled_output": all_rank_pair_stats(left["scaled_output"], right["scaled_output"]),
        "input_gradient": all_rank_pair_stats(left["input_gradient"], right["input_gradient"]),
        "route_score_gradient": all_rank_pair_stats(
            left["route_score_gradient"], right["route_score_gradient"]
        ),
        "lora_A_gradient": all_rank_pair_stats(
            joined(left["gradients"], "A"), joined(right["gradients"], "A")
        ),
        "lora_B_gradient": all_rank_pair_stats(
            joined(left["gradients"], "B"), joined(right["gradients"], "B")
        ),
        "post_muon_A_parameter": all_rank_pair_stats(
            joined(left["updated_parameters"], "A"),
            joined(right["updated_parameters"], "A"),
        ),
        "post_muon_B_parameter": all_rank_pair_stats(
            joined(left["updated_parameters"], "B"),
            joined(right["updated_parameters"], "B"),
        ),
    }
    for projection in PROJECTIONS:
        for factor in FACTORS:
            name = f"{projection}_{factor}"
            metrics[f"{name}_gradient"] = all_rank_pair_stats(
                left["gradients"][name], right["gradients"][name]
            )
            metrics[f"post_muon_{name}_parameter"] = all_rank_pair_stats(
                left["updated_parameters"][name], right["updated_parameters"][name]
            )
    return metrics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("qualification requires exactly eight visible CUDA GPUs")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise RuntimeError("qualification requires eight torchrun ranks")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=8,
        pp=1,
        ep=8,
        world_size=8,
        global_ep=False,
    )
    ep_mesh = parallel_dims.get_mesh("ep")
    set_lora_num_tokens(
        torch.zeros(1, device=device, dtype=torch.int32),
        reset_reference=True,
    )
    set_multilora_scaling(
        torch.ones(1, device=device, dtype=torch.bfloat16),
        reset_reference=True,
    )
    qualification_start = time.perf_counter()
    bf16_first = run_sequence("bf16", parallel_dims, ep_mesh, 0)
    bf16_second = run_sequence("bf16", parallel_dims, ep_mesh, 1)
    int4_first = run_sequence("int4", parallel_dims, ep_mesh, 0)
    int4_second = run_sequence("int4", parallel_dims, ep_mesh, 1)
    route_counts = deterministic_routes(dist.get_rank(), device)[1]
    padding = padding_stats(route_counts, ep_mesh)
    candidate_metrics = [
        comparison(bf16_first["steps"][step], int4_first["steps"][step])
        for step in range(2)
    ]
    relative_rmse_limits = {
        name: (
            5.0e-6
            if name.startswith("post_muon_")
            else 0.25
            if name in {"expert_output", "scaled_output", "route_score_gradient"}
            else 0.20
        )
        for name in candidate_metrics[0]
    }
    numeric_accepted = all(
        step_metrics[name]["relative_rmse_to_left"] <= limit
        for step_metrics in candidate_metrics
        for name, limit in relative_rmse_limits.items()
    )
    repeatability = {
        "bf16": [
            comparison(bf16_first["steps"][step], bf16_second["steps"][step])
            for step in range(2)
        ],
        "int4": [
            comparison(int4_first["steps"][step], int4_second["steps"][step])
            for step in range(2)
        ],
    }
    int4_update_exact = all(
        metric["max_abs"] == 0.0
        for step_metrics in repeatability["int4"]
        for name, metric in step_metrics.items()
        if name.startswith("post_muon_")
    )
    values_and_gradients_exact = all(
        metric["max_abs"] == 0.0
        for sequences in repeatability.values()
        for step_metrics in sequences
        for name, metric in step_metrics.items()
        if not name.startswith("post_muon_")
    )
    bf16_update_max_abs = max(
        metric["max_abs"]
        for step_metrics in repeatability["bf16"]
        for name, metric in step_metrics.items()
        if name.startswith("post_muon_")
    )
    deterministic = (
        values_and_gradients_exact
        and int4_update_exact
        and bf16_update_max_abs <= 2.0e-6
    )
    qualification_wall_seconds = all_rank_max(
        time.perf_counter() - qualification_start,
        device,
    )
    duplicate_exact = all(
        step["metadata"]["duplicate"]["expert_output_max_abs"] == 0.0
        for sequence in (bf16_first, bf16_second, int4_first, int4_second)
        for step in sequence["steps"]
    )
    if not duplicate_exact:
        raise RuntimeError("duplicate routes produced different expert outputs")
    if (
        padding["padding_rows"] <= 0
        or padding["unaligned_local_experts"] <= 0
        or padding["zero_local_experts"] <= 0
    ):
        raise RuntimeError("qualification did not exercise aligned expert padding")
    command = json.loads(args.command_json)
    mounts = json.loads(args.mounts_json)
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise RuntimeError("runtime command provenance is invalid")
    if not isinstance(mounts, dict) or set(mounts) != set(RUNTIME_FILES):
        raise RuntimeError("runtime mount provenance is incomplete")
    artifact = {
        "schema_version": 1,
        "kind": "glm52_native_int4_expert_lora_qualification",
        "verdict": "PASS" if deterministic and numeric_accepted else "FAIL",
        "source_commit": args.source_commit,
        "runtime_source_sha256": runtime_manifest(args.manifest_root),
        "runtime_command": command,
        "runtime_mounts": mounts,
        "runtime_mount_root": args.mount_root,
        "trainer_image_digest": args.image_digest,
        "system": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "world_size": dist.get_world_size(),
        },
        "shape": {
            "experts": NUM_EXPERTS,
            "local_experts": NUM_EXPERTS // 8,
            "dim": DIM,
            "hidden_dim": HIDDEN_DIM,
            "lora_rank": RANK,
            "lora_alpha": ALPHA,
            "ep": 8,
            "cp": 8,
        },
        "coverage": {
            "native_module": "NativeFP8MultiLoRAGroupedExperts",
            "expert_parallel_style": "NativeFP8LoRAAllToAllExpertParallel",
            "route_scaling": "scale_routed_output",
            "optimizer": "prime_rl.trainer.optim.setup_optimizer:muon",
            "gradient_clip": "global EP L2, max_norm=1.0, before each update",
            "scheduler": "prime_rl.trainer.scheduler.setup_scheduler:constant",
            "steps_per_persistent_optimizer": 2,
            "padding": padding,
            "routed_rows_total": all_rank_sum(
                bf16_first["metadata"]["source_rows"], device
            ),
            "duplicate_routes_total": all_rank_sum(
                bf16_first["metadata"]["duplicate_rows"], device
            ),
            "expert_id_reference": bf16_first["steps"][0]["metadata"]["expert_id_reference"],
        },
        "bf16": {
            "sequence": bf16_first["metadata"],
            "steps": [step["metadata"] for step in bf16_first["steps"]],
        },
        "int4": {
            "sequence": int4_first["metadata"],
            "steps": [step["metadata"] for step in int4_first["steps"]],
        },
        "bf16_vs_int4": candidate_metrics,
        "acceptance": {
            "relative_rmse_limits": relative_rmse_limits,
            "numeric_accepted": numeric_accepted,
            "qualification_wall_seconds": qualification_wall_seconds,
            "bf16_peak_memory_gib": bf16_first["metadata"]["peak_memory_bytes"]
            / 2**30,
            "int4_peak_memory_gib": int4_first["metadata"]["peak_memory_bytes"]
            / 2**30,
        },
        "repeatability": repeatability,
        "checks": {
            "deterministic": deterministic,
            "numeric_accepted": numeric_accepted,
            "values_and_gradients_bitwise_repeatable": values_and_gradients_exact,
            "int4_muon_update_bitwise_repeatable": int4_update_exact,
            "bf16_muon_update_repeatability_max_abs": bf16_update_max_abs,
            "bf16_muon_update_repeatability_limit": 2.0e-6,
            "duplicate_routes_exact": duplicate_exact,
            "padding_exercised": True,
            "zero_experts_exercised": True,
            "nonuniform_scored_duplicates_exercised": True,
            "all_lora_A_gradients_present": True,
            "all_lora_B_gradients_present": True,
            "route_score_gradient_present": True,
            "input_gradient_present": True,
            "real_muon_update": True,
            "persistent_two_step_muon": True,
            "optimizer_clear_exercised": True,
            "constant_scheduler_exercised": True,
            "nonzero_native_fp8_base": True,
        },
    }
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps(artifact, sort_keys=True), flush=True)
    dist.barrier()
    if artifact["verdict"] != "PASS":
        raise RuntimeError("native INT4 expert LoRA acceptance threshold failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
