"""Eight-rank correctness and latency gate for compressed EP transport."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.expert_parallel import _INT4Dispatch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--cols", type=int, default=6144)
    parser.add_argument("--iters", type=int, default=3)
    return parser.parse_args()


def timed(callable):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def relative_rmse(got: torch.Tensor, expected: torch.Tensor) -> float:
    difference = got.float() - expected.float()
    return float(
        difference.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-30)
    )


def main() -> None:
    args = arguments()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError("the production EP transport gate requires eight ranks")
    if args.rows % world:
        raise ValueError("rows must divide evenly across ranks")

    generator = torch.Generator(device="cuda").manual_seed(20260810 + rank)
    value = torch.randn(
        args.rows,
        args.cols,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    splits = [args.rows // world] * world
    reference = torch.empty_like(value)

    def bf16_dispatch():
        dist.all_to_all_single(reference, value, splits, splits)
        return reference

    for _ in range(2):
        bf16_dispatch()
    _, bf16_seconds = timed(bf16_dispatch)

    def compressed_dispatch(input_value):
        return _INT4Dispatch.apply(
            input_value,
            splits,
            splits,
            dist.group.WORLD.group_name,
            "gate",
            None,
            None,
        )

    for _ in range(2):
        compressed_dispatch(value)
    forward_times = []
    compressed = None
    for _ in range(args.iters):
        compressed, seconds = timed(lambda: compressed_dispatch(value))
        forward_times.append(seconds)
    assert compressed is not None
    forward_rel_rmse = relative_rmse(compressed, reference)

    train_value = value.detach().clone().requires_grad_()
    upstream = torch.randn(
        reference.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    expected_gradient = torch.empty_like(train_value)
    dist.all_to_all_single(expected_gradient, upstream, splits, splits)
    output = compressed_dispatch(train_value)
    gradient, backward_seconds = timed(
        lambda: torch.autograd.grad(output, train_value, upstream)[0]
    )
    backward_rel_rmse = relative_rmse(gradient, expected_gradient)
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("compressed EP backward produced nonfinite gradients")
    if forward_rel_rmse > 0.20:
        raise RuntimeError(
            f"INT4 forward relative RMSE is too large: {forward_rel_rmse}"
        )
    if backward_rel_rmse > 0.08:
        raise RuntimeError(
            f"FP8 backward relative RMSE is too large: {backward_rel_rmse}"
        )

    metrics = torch.tensor(
        [
            bf16_seconds,
            statistics.median(forward_times),
            backward_seconds,
            forward_rel_rmse,
            backward_rel_rmse,
        ],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            "expert_parallel_transport=PASS "
            f"rows={args.rows} cols={args.cols} "
            f"bf16_forward_s={metrics[0]:.4f} "
            f"int4_forward_s={metrics[1]:.4f} "
            f"fp8_backward_s={metrics[2]:.4f} "
            f"forward_rel_rmse={metrics[3]:.6f} "
            f"backward_rel_rmse={metrics[4]:.6f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
