"""Eight-rank CUDA check for native FP8 expert LoRA transport.

Run with ``torchrun --standalone --nproc_per_node=8`` in the qualification
image.  The script compares the existing BF16 and full per-token FP8
transports on the same production ownership layout and writes one JSON report
on rank zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.distributed.expert_parallel import (
    _BF16Dispatch,
    _FP8Dispatch,
    NativeFP8LoRAAllToAllExpertParallel,
    ReceivedBF16ExpertInput,
    ReceivedFP8ExpertInput,
)
import prime_rl.trainer.distributed.expert_parallel as expert_parallel_source
import prime_rl.trainer.models.layers.fp8_ada as fp8_ada_source
import prime_rl.trainer.models.layers.lora.native_fp8_moe as native_lora_source
import prime_rl.trainer.models.layers.lora.base as lora_base_source
import prime_rl.trainer.models.layers.moe as moe_source
from prime_rl.trainer.models.layers.fp8_ada import scale_routed_output
from prime_rl.trainer.models.layers.lora.native_fp8_moe import (
    NativeFP8MultiLoRAGroupedExperts,
)
from prime_rl.trainer.models.layers.lora.base import (
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts
from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M


WORLD = 8
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = GLOBAL_EXPERTS // WORLD
DIM = 6144
HIDDEN = 2048
RANK = 32
ROWS = 40
VALID_ROWS = 40
LR = 8e-6
RELATIVE_ERROR_LIMIT = 0.10
SOURCE_SHA256 = {
    "expert_parallel.py": "38d773541e6d2aee55ca4ce4b91ff2f1df6d96ce47681704d9e49af4dc306613",
    "fp8_ada.py": "fd05b0f216eaa3efca881ccc2154d5557c4998052690e6ee9c6902bb54fec45a",
    "native_fp8_moe.py": "5bfe894ec01412d47d597e9bd5276cd2186c146cae1aa3b2b81632d3dd90e974",
    "base.py": "5d6fac5b0ab0bc81e538a37ac46bb5751a74f9d8adf464d7445d102e078bcb24",
    "moe.py": "223555bf88a2c376825c92a9d2443e8b7845756bac4a8184c92a2a495d66b6ed",
}


class Mesh:
    shape = (WORLD,)

    @staticmethod
    def get_group():
        return dist.group.WORLD


def assert_runtime_sources() -> None:
    modules = (
        expert_parallel_source,
        fp8_ada_source,
        native_lora_source,
        lora_base_source,
        moe_source,
    )
    for module in modules:
        path = Path(module.__file__).resolve()
        expected = SOURCE_SHA256[path.name]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"runtime source checksum mismatch for {path}: {actual} != {expected}"
            )


def global_error(got: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = got.float() - reference.float()
    square_sum = difference.square().sum()
    reference_square_sum = reference.float().square().sum()
    count = torch.tensor(difference.numel(), device=got.device, dtype=torch.float64)
    maximum = difference.abs().max().double()
    values = torch.stack((square_sum.double(), reference_square_sum.double(), count, maximum))
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    # max was summed above; reduce it separately to avoid relying on MAX for a
    # mixed tensor dtype in older NCCL builds.
    maximum = torch.tensor(float(difference.abs().max()), device=got.device, dtype=torch.float64)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {
        "rmse": float(torch.sqrt(values[0] / values[2])),
        "relative_rmse": float(torch.sqrt(values[0] / values[1].clamp_min(1e-30))),
        "max_abs": float(maximum),
    }


def make_counts(rank: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(GLOBAL_EXPERTS, dtype=torch.int64, device=device)
    # Each route is repeated once.  Expert permutation adds alignment padding
    # after dispatch; input rows themselves must equal the split total.
    for token in range(0, VALID_ROWS, 2):
        expert = ((token // 2) * 17 + rank * 11) % GLOBAL_EXPERTS
        counts[expert] += 2
    return counts


def make_module(rank: int) -> NativeFP8MultiLoRAGroupedExperts:
    torch.manual_seed(7000 + rank)
    base = GroupedExperts(
        DIM,
        HIDDEN,
        LOCAL_EXPERTS,
        use_grouped_mm=True,
        fp8_checkpoint=True,
    ).cuda()
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            parameter = getattr(base, name)
            parameter.copy_(torch.randn_like(parameter, dtype=torch.bfloat16).to(parameter.dtype))
        for name in ("w1_scale_inv", "w2_scale_inv", "w3_scale_inv"):
            getattr(base, name).fill_(0.02)
    module = NativeFP8MultiLoRAGroupedExperts(
        base,
        RANK,
        1,
        alpha=64.0,
        dropout=0.0,
    ).cuda()
    with torch.no_grad():
        for projection, input_width, output_width in (
            ("w1", DIM, HIDDEN),
            ("w2", HIDDEN, DIM),
            ("w3", DIM, HIDDEN),
        ):
            for factor, shape in (
                ("A", (LOCAL_EXPERTS, RANK, input_width)),
                ("B", (LOCAL_EXPERTS, output_width, RANK)),
            ):
                values = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
                setattr(module, f"{projection}_lora_{factor}", nn.ParameterList([nn.Parameter(values)]))
    module.num_experts = LOCAL_EXPERTS
    module._ep_group = dist.group.WORLD
    return module


def muon_step(module: nn.Module) -> list[torch.Tensor]:
    """Apply the configured Muon update to each expert's 2-D local factor."""
    before: list[torch.Tensor] = []
    for name, parameter_list in (
        (f"{projection}_lora_{factor}", getattr(module, f"{projection}_lora_{factor}"))
        for projection in ("w1", "w2", "w3")
        for factor in ("A", "B")
    ):
        del name
        parameter = parameter_list[0]
        before.append(parameter.detach().clone())
        updated = parameter.detach().clone()
        for expert in range(LOCAL_EXPERTS):
            leaf = nn.Parameter(updated[expert].detach().clone())
            leaf.grad = parameter.grad[expert].detach().clone()
            optimizer = torch.optim.Muon(
                [leaf], lr=LR, weight_decay=0.0, momentum=0.95
            )
            optimizer.step()
            updated[expert].copy_(leaf.detach())
        parameter.data.copy_(updated)
    return before


def run_once(transport: str, rank: int) -> dict[str, object]:
    torch.manual_seed(9000 + rank)
    module = make_module(rank)
    torch.manual_seed(12000 + rank)
    values = torch.randn((ROWS, DIM), device="cuda", dtype=torch.bfloat16)
    values.requires_grad_()
    scores = torch.rand(ROWS, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn((ROWS, DIM), device="cuda", dtype=torch.bfloat16)
    counts = make_counts(rank, values.device)
    parallel = NativeFP8LoRAAllToAllExpertParallel(
        expert_activation_transport=transport
    )
    routed, grouped_counts, received = parallel._token_dispatch(
        module, [values, counts], Mesh()
    )
    routed_output = module(routed, grouped_counts, received)
    combined = parallel._token_combine(module, routed_output, Mesh())
    scaled = scale_routed_output(combined, scores)
    loss = (scaled * grad_output).float().sum()
    loss.backward()
    factors = [
        parameter_list[0].grad.detach().clone()
        for projection in ("w1", "w2", "w3")
        for factor in ("A", "B")
        for parameter_list in (getattr(module, f"{projection}_lora_{factor}"),)
    ]
    score_gradient = scores.grad.detach().clone()
    input_gradient = values.grad.detach().clone()
    parameter_before = muon_step(module)
    parameter_after = [
        parameter_list[0].detach().clone()
        for projection in ("w1", "w2", "w3")
        for factor in ("A", "B")
        for parameter_list in (getattr(module, f"{projection}_lora_{factor}"),)
    ]
    local_counts = grouped_counts.view(WORLD, LOCAL_EXPERTS).sum(dim=0)
    alignment_padding = int(
        sum(
            ((int(count) + TOKEN_GROUP_ALIGN_SIZE_M - 1) // TOKEN_GROUP_ALIGN_SIZE_M)
            * TOKEN_GROUP_ALIGN_SIZE_M
            - int(count)
            for count in local_counts
        )
    )
    gradient_coverage = [
        bool(torch.count_nonzero(gradient)) for gradient in factors
    ]
    return {
        "output": combined.detach(),
        "input_gradient": input_gradient,
        "factors": factors,
        "score_gradient": score_gradient,
        "parameter_before": parameter_before,
        "parameter_after": parameter_after,
        "alignment_padding": alignment_padding,
        "gradient_coverage": gradient_coverage,
        "score_gradient_nonzero": bool(torch.count_nonzero(score_gradient)),
        "input_gradient_nonzero": bool(torch.count_nonzero(input_gradient)),
    }


def compare_runs(candidate: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "output": global_error(candidate["output"], reference["output"]),
        "input_gradient": global_error(candidate["input_gradient"], reference["input_gradient"]),
        "score_gradient": global_error(candidate["score_gradient"], reference["score_gradient"]),
        "lora": {},
        "post_update": {},
    }
    for index, (got, expected) in enumerate(zip(candidate["factors"], reference["factors"], strict=True)):
        result["lora"][str(index)] = global_error(got, expected)
    for index, (got, expected) in enumerate(zip(candidate["parameter_after"], reference["parameter_after"], strict=True)):
        result["post_update"][str(index)] = global_error(got, expected)
    return result


def compare_repeat(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    result = {"output": global_error(first["output"], second["output"]), "input_gradient": global_error(first["input_gradient"], second["input_gradient"])}
    result["lora"] = {
        str(index): global_error(got, expected)
        for index, (got, expected) in enumerate(zip(first["factors"], second["factors"], strict=True))
    }
    result["post_update"] = {
        str(index): global_error(got, expected)
        for index, (got, expected) in enumerate(zip(first["parameter_after"], second["parameter_after"], strict=True))
    }
    return result


def production_transport_benchmark(rank: int) -> dict[str, dict[str, float]]:
    """Measure one production-shape dispatch/combine on every EP rank."""
    split_pattern = [0, 2048, 3072, 4096, 5120, 6144, 7168, 5120]
    input_splits = [split_pattern[(peer - rank) % WORLD] for peer in range(WORLD)]
    output_splits = [split_pattern[(rank - peer) % WORLD] for peer in range(WORLD)]
    rows = sum(input_splits)
    if rows != 32768 or sum(output_splits) != 32768:
        raise RuntimeError("production split pattern does not total 32768 rows")
    value = torch.randn((rows, 6144), device="cuda", dtype=torch.bfloat16)
    result: dict[str, dict[str, float]] = {}
    for name, dispatch, received_input in (
        ("bf16", _BF16Dispatch, ReceivedBF16ExpertInput()),
        ("fp8", _FP8Dispatch, ReceivedFP8ExpertInput()),
    ):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            routed = dispatch.apply(
                value,
                output_splits,
                input_splits,
                dist.group.WORLD.group_name,
                "dispatch",
                received_input,
            )
            combined = dispatch.apply(
                routed,
                input_splits,
                output_splits,
                dist.group.WORLD.group_name,
                "combine",
                None,
            )
        end.record()
        end.synchronize()
        elapsed_ms = torch.tensor(start.elapsed_time(end), device="cuda")
        peak_bytes = torch.tensor(
            torch.cuda.max_memory_allocated() - baseline,
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(elapsed_ms, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_bytes, op=dist.ReduceOp.MAX)
        result[name] = {
            "latency_ms_max_rank": float(elapsed_ms),
            "peak_allocated_bytes_max_rank": float(peak_bytes),
            "peak_allocated_mib_max_rank": float(peak_bytes / 2**20),
        }
        del routed, combined, received_input
    del value
    torch.cuda.empty_cache()
    return result


def validate(candidate: dict[str, object], comparison: dict[str, object], repeat: dict[str, object]) -> list[str]:
    failures = []
    for name in ("output", "input_gradient", "score_gradient"):
        metric = comparison[name]
        if metric["relative_rmse"] > RELATIVE_ERROR_LIMIT:
            failures.append(f"{name} relative RMSE {metric['relative_rmse']:.6f} > {RELATIVE_ERROR_LIMIT}")
    for index, metric in comparison["lora"].items():
        if metric["relative_rmse"] > RELATIVE_ERROR_LIMIT:
            failures.append(f"LoRA factor {index} relative RMSE {metric['relative_rmse']:.6f} > {RELATIVE_ERROR_LIMIT}")
    if candidate["alignment_padding"] <= 0:
        failures.append("expert permutation produced no alignment padding")
    if not all(candidate["gradient_coverage"]):
        failures.append("one or more LoRA A/B factor gradients are zero")
    if not candidate["score_gradient_nonzero"] or not candidate["input_gradient_nonzero"]:
        failures.append("route-score or input gradient is zero")
    for name, metrics in repeat.items():
        if isinstance(metrics, dict) and "relative_rmse" in metrics and metrics["max_abs"] != 0.0:
            failures.append(f"non-deterministic FP8 repeat for {name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        raise RuntimeError("this check requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    assert_runtime_sources()
    set_lora_num_tokens(
        torch.ones(1, dtype=torch.int32, device="cuda"), reset_reference=True
    )
    set_multilora_scaling(
        torch.ones(1, dtype=torch.float32, device="cuda"), reset_reference=True
    )
    dist.barrier()
    bf16_result = run_once("bf16", rank)
    dist.barrier()
    fp8_result = run_once("fp8", rank)
    dist.barrier()
    repeat_result = run_once("fp8", rank)
    comparison = compare_runs(fp8_result, bf16_result)
    repeat = compare_repeat(fp8_result, repeat_result)
    benchmark = production_transport_benchmark(rank)
    failures = validate(fp8_result, comparison, repeat)
    failed = torch.tensor(bool(failures), device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "result": "REJECT" if failed.item() else "PASS",
                    "world_size": WORLD,
                    "global_experts": GLOBAL_EXPERTS,
                    "local_experts": LOCAL_EXPERTS,
                    "rows": ROWS,
                    "valid_rows": VALID_ROWS,
                    "hidden": DIM,
                    "rank": RANK,
                    "optimizer": "torch.optim.Muon",
                    "learning_rate": LR,
                    "source_sha256": SOURCE_SHA256,
                    "bf16_vs_fp8": comparison,
                    "fp8_repeat": repeat,
                    "alignment_padding": fp8_result["alignment_padding"],
                    "gradient_coverage": fp8_result["gradient_coverage"],
                    "production_transport_benchmark": benchmark,
                    "relative_error_limit": RELATIVE_ERROR_LIMIT,
                    "failures": failures,
                },
                indent=2,
            )
        )
        print(json.dumps({"result": "REJECT" if failed.item() else "PASS", "output": str(args.output), "comparison": comparison, "repeat": repeat, "benchmark": benchmark, "failures": failures}))
    dist.barrier()
    dist.destroy_process_group()
    if failed.item():
        raise RuntimeError("native FP8 transport qualification rejected: " + "; ".join(failures))


if __name__ == "__main__":
    main()
