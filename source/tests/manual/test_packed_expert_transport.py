"""Exact packed value-and-scale transport benchmark for SM89 EP."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time

os.environ.setdefault("PRIME_INT4_COMBINE_ROUNDS", "8")

import torch
import torch.distributed as dist

import prime_rl.trainer.distributed.expert_parallel as expert_parallel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--expert-layers", type=int, default=75)
    return parser.parse_args()


def synchronized_call(callable):
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def median_chain(callable, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        synchronized_call(callable)
    return statistics.median(
        synchronized_call(callable)[1] for _ in range(iterations)
    )


def pinned_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmPin:"):
                return int(line.split()[1]) * 1024
    return 0


def run_chain(value, upstream, output_splits, input_splits, packed):
    expert_parallel._PACKED_EXPERT_TRANSPORT = packed
    input_value = value.detach().clone().requires_grad_()
    output = expert_parallel._INT4Dispatch.apply(
        input_value,
        output_splits,
        input_splits,
        dist.group.WORLD.group_name,
        "dispatch",
        None,
        None,
    )
    gradient = torch.autograd.grad(output, input_value, upstream)[0]
    return output.detach(), gradient.detach()


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main() -> None:
    args = arguments()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise RuntimeError(f"this benchmark requires SM89, found SM{capability[0]}{capability[1]}")
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"this benchmark requires eight ranks, found {world}")

    split_pattern = [0, 2048, 3072, 4096, 5120, 6144, 7168, 5120]
    input_splits = [split_pattern[(peer - rank) % world] for peer in range(world)]
    output_splits = [split_pattern[(rank - peer) % world] for peer in range(world)]
    rows = sum(input_splits)
    output_rows = sum(output_splits)
    generator = torch.Generator(device="cuda").manual_seed(20260828 + rank)
    value = torch.randn(
        rows, args.cols, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.02)
    upstream = torch.randn(
        output_rows, args.cols, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(1e-4)

    base_output, base_gradient = run_chain(
        value, upstream, output_splits, input_splits, packed=False
    )
    packed_output, packed_gradient = run_chain(
        value, upstream, output_splits, input_splits, packed=True
    )
    if not torch.equal(base_output, packed_output):
        raise RuntimeError("packed transport changed forward values")
    if not torch.equal(base_gradient, packed_gradient):
        raise RuntimeError("packed transport changed backward gradients")

    gc.collect()
    torch.cuda.empty_cache()
    expert_parallel._PACKED_EXPERT_TRANSPORT = False
    torch.cuda.reset_peak_memory_stats()
    base_pinned_before = pinned_bytes()
    base_seconds = median_chain(
        lambda: run_chain(value, upstream, output_splits, input_splits, False),
        args.warmup,
        args.iters,
    )
    base_peak = torch.cuda.max_memory_allocated()
    base_pinned_after = pinned_bytes()

    gc.collect()
    torch.cuda.empty_cache()
    expert_parallel._PACKED_EXPERT_TRANSPORT = True
    torch.cuda.reset_peak_memory_stats()
    packed_pinned_before = pinned_bytes()
    packed_seconds = median_chain(
        lambda: run_chain(value, upstream, output_splits, input_splits, True),
        args.warmup,
        args.iters,
    )
    packed_peak = torch.cuda.max_memory_allocated()
    packed_pinned_after = pinned_bytes()

    base_seconds = max_across_ranks(base_seconds, value.device)
    packed_seconds = max_across_ranks(packed_seconds, value.device)
    base_peak = max_across_ranks(float(base_peak), value.device)
    packed_peak = max_across_ranks(float(packed_peak), value.device)
    base_pinned = max_across_ranks(
        float(max(base_pinned_before, base_pinned_after)), value.device
    )
    packed_pinned = max_across_ranks(
        float(max(packed_pinned_before, packed_pinned_after)), value.device
    )

    speed_gain = (base_seconds - packed_seconds) / base_seconds
    gpu_increase_mib = (packed_peak - base_peak) / (1024 * 1024)
    projected_step_saving = max(0.0, base_seconds - packed_seconds) * args.expert_layers * 2
    passed = (
        gpu_increase_mib <= 64.0
        and packed_pinned <= base_pinned
        and (speed_gain >= 0.05 or projected_step_saving >= 10.0)
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "benchmark": "packed_value_scale_expert_transport",
                    "world": world,
                    "sm": f"{capability[0]}{capability[1]}",
                    "rows": rows,
                    "output_rows": output_rows,
                    "cols": args.cols,
                    "rounds": int(os.environ["PRIME_INT4_COMBINE_ROUNDS"]),
                    "base_chain_seconds": base_seconds,
                    "packed_chain_seconds": packed_seconds,
                    "speed_gain": speed_gain,
                    "projected_step_saving_seconds": projected_step_saving,
                    "gpu_peak_increase_mib": gpu_increase_mib,
                    "base_pinned_bytes": int(base_pinned),
                    "packed_pinned_bytes": int(packed_pinned),
                    "passed": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise SystemExit("packed transport did not meet a strict speed or memory guard")


if __name__ == "__main__":
    main()
