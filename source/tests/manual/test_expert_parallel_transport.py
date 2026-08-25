"""Multi-rank correctness and latency check for expert transport."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.expert_parallel import (
    BF16ExpertInputWorkspace,
    ReceivedBF16ExpertInput,
    ReceivedINT4ExpertInput,
    _BF16Dispatch,
    _INT4Dispatch,
    bf16_expert_transport_memory_bytes,
)
from prime_rl.trainer.models.layers.fp8_ada import scale_routed_output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rows-per-peer", type=int, default=0)
    parser.add_argument("--exact-production-rows", action="store_true")
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


def median_time(callable, warmup: int, iterations: int):
    for _ in range(warmup):
        timed(callable)
    times = [timed(callable)[1] for _ in range(iterations)]
    return statistics.median(times)


def main() -> None:
    args = arguments()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError("the expert transport check requires at least two ranks")

    if args.exact_production_rows:
        if world != 8:
            raise RuntimeError("exact production row splits require eight ranks")
        split_pattern = [0, 2048, 3072, 4096, 5120, 6144, 7168, 5120]
        input_splits = [split_pattern[(peer - rank) % world] for peer in range(world)]
        output_splits = [split_pattern[(rank - peer) % world] for peer in range(world)]
    else:
        input_splits = [
            args.rows_per_peer + rank * world + peer + 1 for peer in range(world)
        ]
        output_splits = [
            args.rows_per_peer + peer * world + rank + 1 for peer in range(world)
        ]
    rows = sum(input_splits)

    generator = torch.Generator(device="cuda").manual_seed(20260810 + rank)
    value = torch.randn(
        rows,
        args.cols,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    reference = torch.empty(
        (sum(output_splits), args.cols),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def reference_dispatch():
        dist.all_to_all_single(
            reference,
            value,
            output_splits,
            input_splits,
        )
        return reference

    reference_seconds = median_time(
        reference_dispatch,
        args.warmup,
        args.iters,
    )

    saved_input = ReceivedBF16ExpertInput()
    train_value = value.detach().clone().requires_grad_()
    exact, bf16_seconds = timed(
        lambda: _BF16Dispatch.apply(
            train_value,
            output_splits,
            input_splits,
            dist.group.WORLD.group_name,
            "dispatch",
            saved_input,
        )
    )
    if not torch.equal(exact, reference):
        raise RuntimeError("BF16 dispatch changed activation values")
    torch.cuda.synchronize()
    if not torch.equal(saved_input.values, reference):
        raise RuntimeError("BF16 saved expert input changed activation values")
    no_grad_input = ReceivedBF16ExpertInput(save_for_backward=False)
    no_grad_input.prepare(reference)
    if no_grad_input.values is not None:
        raise RuntimeError("BF16 no-grad transport retained a saved expert input")

    selected_rows = torch.tensor(
        [0, exact.shape[0] - 1, -1, exact.shape[0]],
        device="cuda",
        dtype=torch.int64,
    )
    reconstructed = torch.empty(
        (selected_rows.shape[0], args.cols),
        device="cuda",
        dtype=torch.bfloat16,
    )
    workspace = BF16ExpertInputWorkspace(
        selected_rows.shape[0],
        args.cols,
        reconstructed.device,
    )
    saved_input.reconstruct_rows(selected_rows, reconstructed, workspace)
    expected_rows = torch.stack(
        (reference[0], reference[-1], torch.zeros_like(reference[0]), torch.zeros_like(reference[0]))
    )
    if not torch.equal(reconstructed, expected_rows):
        raise RuntimeError("BF16 saved expert input reconstruction is not exact")

    int4_saved_input = ReceivedINT4ExpertInput()
    int4_received = _INT4Dispatch.apply(
        value,
        output_splits,
        input_splits,
        dist.group.WORLD.group_name,
        "dispatch",
        None,
        int4_saved_input,
    )
    int4_selected_rows = selected_rows.cpu()
    int4_reconstructed = torch.empty_like(reconstructed)
    int4_workspace = int4_saved_input.make_workspace(
        int4_selected_rows.shape[0],
        int4_reconstructed.device,
    )
    int4_saved_input.reconstruct_rows(
        int4_selected_rows,
        int4_reconstructed,
        int4_workspace,
    )
    expected_int4_rows = torch.stack(
        (
            int4_received[0],
            int4_received[-1],
            torch.zeros_like(int4_received[0]),
            torch.zeros_like(int4_received[0]),
        )
    )
    if not torch.equal(int4_reconstructed, expected_int4_rows):
        raise RuntimeError("INT4 saved expert input reconstruction changed values")
    del (
        int4_saved_input,
        int4_received,
        int4_selected_rows,
        int4_reconstructed,
        int4_workspace,
        expected_int4_rows,
    )

    combined = _BF16Dispatch.apply(
        exact,
        input_splits,
        output_splits,
        dist.group.WORLD.group_name,
        "combine",
        None,
    )
    if not torch.equal(combined, value):
        raise RuntimeError("BF16 combine changed activation values")

    raw_dispatch = torch.empty_like(reference)
    raw_combined = torch.empty_like(value)

    def raw_chain():
        dist.all_to_all_single(
            raw_dispatch,
            value,
            output_splits,
            input_splits,
        )
        dist.all_to_all_single(
            raw_combined,
            raw_dispatch,
            input_splits,
            output_splits,
        )
        return raw_combined

    raw_chain_seconds = median_time(raw_chain, args.warmup, args.iters)
    if not torch.equal(raw_chain(), value):
        raise RuntimeError("raw BF16 chain changed activation values")

    legacy_host = torch.empty_like(reference, device="cpu", pin_memory=True)
    legacy_restored = torch.empty_like(reference)

    def current_chain():
        dist.all_to_all_single(
            raw_dispatch,
            value,
            output_splits,
            input_splits,
        )
        legacy_host.copy_(raw_dispatch, non_blocking=True)
        dist.all_to_all_single(
            raw_combined,
            raw_dispatch,
            input_splits,
            output_splits,
        )
        legacy_restored.copy_(legacy_host, non_blocking=True)
        return raw_combined

    current_chain_seconds = median_time(
        current_chain,
        args.warmup,
        args.iters,
    )
    if not torch.equal(current_chain(), value):
        raise RuntimeError("current BF16 chain changed activation values")
    if not torch.equal(legacy_restored, reference):
        raise RuntimeError("current BF16 saved input roundtrip changed values")

    def optimized_chain():
        saved = ReceivedBF16ExpertInput()
        dispatched = _BF16Dispatch.apply(
            value,
            output_splits,
            input_splits,
            dist.group.WORLD.group_name,
            "dispatch",
            saved,
        )
        result = _BF16Dispatch.apply(
            dispatched,
            input_splits,
            output_splits,
            dist.group.WORLD.group_name,
            "combine",
            None,
        )
        return result, saved

    optimized_chain_seconds = median_time(
        optimized_chain,
        args.warmup,
        args.iters,
    )
    optimized_result, optimized_saved = optimized_chain()
    if not torch.equal(optimized_result, value):
        raise RuntimeError("optimized BF16 chain changed activation values")
    if not torch.equal(optimized_saved.values, reference):
        raise RuntimeError("optimized BF16 saved input changed activation values")

    chain_input = value.detach().clone().requires_grad_()
    chain_dispatch = _BF16Dispatch.apply(
        chain_input,
        output_splits,
        input_splits,
        dist.group.WORLD.group_name,
        "dispatch",
        None,
    )
    chain_output = _BF16Dispatch.apply(
        chain_dispatch,
        input_splits,
        output_splits,
        dist.group.WORLD.group_name,
        "combine",
        None,
    )
    route_scores = torch.linspace(
        0.1,
        0.9,
        chain_output.shape[0],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    scaled_chain_output = scale_routed_output(chain_output, route_scores)
    chain_upstream = torch.randn(
        scaled_chain_output.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    chain_gradient, score_gradient = torch.autograd.grad(
        scaled_chain_output,
        (chain_input, route_scores),
        chain_upstream,
    )
    expected_chain_gradient = (
        chain_upstream.float() * route_scores.detach()[:, None]
    ).to(torch.bfloat16)
    expected_score_gradient = (
        chain_upstream.float() * chain_input.detach().float()
    ).sum(dim=-1)
    if not torch.equal(chain_gradient, expected_chain_gradient):
        raise RuntimeError("BF16 dispatch, combine, and route scaling changed input gradients")
    if not torch.equal(score_gradient, expected_score_gradient):
        raise RuntimeError("BF16 dispatch, combine, and route scaling changed score gradients")

    gradient_input = value.detach().clone().requires_grad_()
    gradient_output = _BF16Dispatch.apply(
        gradient_input,
        output_splits,
        input_splits,
        dist.group.WORLD.group_name,
        "dispatch",
        None,
    )
    upstream = torch.randn(
        gradient_output.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    expected_gradient = torch.empty_like(gradient_input)
    dist.all_to_all_single(
        expected_gradient,
        upstream,
        input_splits,
        output_splits,
    )
    gradient, bf16_backward_seconds = timed(
        lambda: torch.autograd.grad(gradient_output, gradient_input, upstream)[0]
    )
    if not torch.equal(gradient, expected_gradient):
        raise RuntimeError("BF16 expert transport backward changed gradient values")

    def compressed_dispatch(input_value):
        return _INT4Dispatch.apply(
            input_value,
            output_splits,
            input_splits,
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

    int4_train_value = value.detach().clone().requires_grad_()
    int4_upstream = torch.randn(
        reference.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-4)
    int4_expected_gradient = torch.empty_like(int4_train_value)
    dist.all_to_all_single(
        int4_expected_gradient,
        int4_upstream,
        input_splits,
        output_splits,
    )
    output = compressed_dispatch(int4_train_value)
    int4_gradient, backward_seconds = timed(
        lambda: torch.autograd.grad(output, int4_train_value, int4_upstream)[0]
    )
    backward_rel_rmse = relative_rmse(int4_gradient, int4_expected_gradient)
    if not bool(torch.isfinite(int4_gradient).all()):
        raise RuntimeError("compressed EP backward produced nonfinite gradients")
    if forward_rel_rmse > 0.20:
        raise RuntimeError(
            f"INT4 forward relative RMSE is too large: {forward_rel_rmse}"
        )
    if backward_rel_rmse > 0.08:
        raise RuntimeError(
            f"FP8 backward relative RMSE is too large: {backward_rel_rmse}"
        )

    memory = bf16_expert_transport_memory_bytes(
        rows,
        sum(output_splits),
        args.cols,
        selected_rows.shape[0],
    )
    metrics = torch.tensor(
        [
            reference_seconds,
            bf16_seconds,
            bf16_backward_seconds,
            current_chain_seconds,
            raw_chain_seconds,
            optimized_chain_seconds,
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
            f"world={world} rows={rows} cols={args.cols} "
            f"reference_forward_s={metrics[0]:.4f} "
            f"bf16_forward_s={metrics[1]:.4f} "
            f"bf16_backward_s={metrics[2]:.4f} "
            f"current_chain_s={metrics[3]:.4f} "
            f"raw_chain_s={metrics[4]:.4f} "
            f"optimized_chain_s={metrics[5]:.4f} "
            f"int4_forward_s={metrics[6]:.4f} "
            f"fp8_backward_s={metrics[7]:.4f} "
            f"forward_rel_rmse={metrics[8]:.6f} "
            f"backward_rel_rmse={metrics[9]:.6f} "
            f"memory={memory}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
