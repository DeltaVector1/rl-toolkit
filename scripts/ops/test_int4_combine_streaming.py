"""Eight-GPU parity and constrained-memory smoke test for in-place INT4 combine."""

import os

import torch
import torch.distributed as dist
import triton

from prime_rl.trainer.distributed.expert_parallel import (
    INT4ExpertInputWorkspace,
    ReceivedINT4ExpertInput,
    _INT4Dispatch,
    _dequantize_int4_kernel,
    _streaming_int4_forward,
    _streaming_fp8_backward,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name

    torch.manual_seed(1000 + dist.get_rank())
    world = dist.get_world_size()
    rank = dist.get_rank()
    # A circulant matrix gives each rank equal total send/receive rows while
    # every peer segment has a different size. This covers the production
    # variable-split all-to-all, not only the equal-split memory case below.
    input_splits = [((peer - rank) % world) + 1 for peer in range(world)]
    output_splits = [((rank - peer) % world) + 1 for peer in range(world)]
    original = torch.randn((sum(input_splits), 64), device="cuda", dtype=torch.bfloat16)

    reference_input = original.detach().clone().requires_grad_(True)
    reference = _INT4Dispatch.apply(
        reference_input,
        output_splits,
        input_splits,
        group_name,
        "reference",
        None,
        None,
    )
    streamed_leaf = original.detach().clone().requires_grad_(True)
    streamed_input = streamed_leaf * 1
    received_input = ReceivedINT4ExpertInput()
    streamed = _INT4Dispatch.apply(
        streamed_input,
        output_splits,
        input_splits,
        group_name,
        "dispatch",
        None,
        received_input,
    )
    torch.testing.assert_close(streamed, reference, rtol=0, atol=0)
    received_input.copy_done.synchronize()
    captured_values = received_input.values.to("cuda", non_blocking=True)
    captured_scales = received_input.scales.to("cuda", non_blocking=True)
    captured = torch.empty_like(streamed)
    _dequantize_int4_kernel[(captured.shape[0],)](
        captured_values,
        captured_scales,
        captured,
        captured.shape[1],
        triton.next_power_of_2((captured.shape[1] + 1) // 2),
        num_warps=8,
    )
    torch.testing.assert_close(captured, streamed, rtol=0, atol=0)
    selected_rows = torch.tensor(
        [2, 0, -1, streamed.shape[0] - 1],
        device="cpu",
        dtype=torch.int64,
    )
    reconstructed = torch.empty(
        (selected_rows.shape[0], streamed.shape[1]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reconstruction_workspace = INT4ExpertInputWorkspace(
        selected_rows.shape[0],
        streamed.shape[1],
        reconstructed.device,
    )
    received_input.reconstruct_rows(
        selected_rows,
        reconstructed,
        reconstruction_workspace,
    )
    expected_rows = torch.stack(
        (streamed[2], streamed[0], torch.zeros_like(streamed[0]), streamed[-1])
    )
    torch.testing.assert_close(reconstructed, expected_rows, rtol=0, atol=0)
    assert streamed.data_ptr() != streamed_input.data_ptr()
    reference.float().square().mean().backward()
    streamed.float().square().mean().backward()
    assert reference_input.grad is not None
    assert streamed_leaf.grad is not None
    assert torch.isfinite(streamed_leaf.grad).all()
    torch.testing.assert_close(streamed_leaf.grad, reference_input.grad, rtol=0, atol=0)

    # Production routing has different total rows on each rank. Ranks 0 to 3
    # allocate a larger result; ranks 4 to 7 reuse and shorten their input.
    imbalanced_input_splits = [rank + 1] * world
    imbalanced_output_splits = [peer + 1 for peer in range(world)]
    imbalanced = torch.randn(
        (sum(imbalanced_input_splits), 64), device="cuda", dtype=torch.bfloat16
    )
    imbalanced_reference_input = imbalanced.detach().clone().requires_grad_(True)
    imbalanced_reference = _INT4Dispatch.apply(
        imbalanced_reference_input,
        imbalanced_output_splits,
        imbalanced_input_splits,
        group_name,
        "reference",
        None,
        None,
    )
    imbalanced_leaf = imbalanced.detach().clone().requires_grad_(True)
    imbalanced_streamed_input = imbalanced_leaf * 1
    imbalanced_streamed = _INT4Dispatch.apply(
        imbalanced_streamed_input,
        imbalanced_output_splits,
        imbalanced_input_splits,
        group_name,
        "dispatch",
        None,
        None,
    )
    torch.testing.assert_close(
        imbalanced_streamed, imbalanced_reference, rtol=0, atol=0
    )
    imbalanced_reference.float().square().mean().backward()
    imbalanced_streamed.float().square().mean().backward()
    assert imbalanced_reference_input.grad is not None
    assert imbalanced_leaf.grad is not None
    assert torch.isfinite(imbalanced_leaf.grad).all()
    torch.testing.assert_close(
        imbalanced_leaf.grad, imbalanced_reference_input.grad, rtol=0, atol=0
    )
    assert imbalanced_streamed.data_ptr() != imbalanced_streamed_input.data_ptr()

    del (
        original,
        reference_input,
        reference,
        streamed_leaf,
        streamed_input,
        streamed,
        received_input,
        captured_values,
        captured_scales,
        captured,
        selected_rows,
        reconstructed,
        reconstruction_workspace,
        expected_rows,
        imbalanced,
        imbalanced_reference_input,
        imbalanced_reference,
        imbalanced_leaf,
        imbalanced_streamed_input,
        imbalanced_streamed,
    )
    torch.cuda.empty_cache()
    dist.barrier()

    # Use production-size unequal routing. Low ranks must allocate a larger
    # dispatch result and high ranks must allocate a larger backward result.
    # Leave only 520 MiB free. x71p rank 21 had 41.5 MiB physically free after
    # allocating its unequal backward result, then failed on a 50 MiB four-
    # round receive. Eight rounds keep that receive near 25 MiB. The old
    # one-shot dispatch retained complete
    # packed send and receive buffers and could not allocate its BF16 result;
    # the round-streamed path allocates the result first and then retains only
    # one round of packed transport.
    cols = 6144
    per_peer = 4096 + (rank - 3) * 128
    large_input_splits = [per_peer] * world
    large_output_splits = [4096 + (peer - 3) * 128 for peer in range(world)]
    input_rows = sum(large_input_splits)
    output_rows = sum(large_output_splits)
    rows = input_rows
    large_input = torch.zeros((rows, cols), device="cuda", dtype=torch.bfloat16)
    free_bytes, _ = torch.cuda.mem_get_info()
    target_free = 560 * 1024 * 1024
    ballast = torch.empty(
        max(0, free_bytes - target_free), device="cuda", dtype=torch.uint8
    )
    before, _ = torch.cuda.mem_get_info()
    print(
        f"rank={rank} constrained-dispatch-start free={before / 2**20:.1f}MiB",
        flush=True,
    )
    large_output = _streaming_int4_forward(
        large_input,
        large_output_splits,
        large_input_splits,
        group_name,
        "dispatch-smoke",
        8,
        False,
    )
    torch.cuda.synchronize()
    print(f"rank={rank} constrained-dispatch-done", flush=True)
    assert large_output.shape == (output_rows, cols)
    forward_reused = large_output.data_ptr() == large_input.data_ptr()
    assert not forward_reused
    assert torch.count_nonzero(large_output[:2]).item() == 0
    after_forward, _ = torch.cuda.mem_get_info()
    # Dispatch reserves enough storage for its inverse combine even on ranks
    # that receive fewer rows than they must return.  The combine must reuse
    # that storage instead of allocating a full BF16 result at peak memory.
    dispatch_capacity_rows = (
        large_output.untyped_storage().nbytes() // large_output.element_size() // cols
    )
    assert dispatch_capacity_rows >= max(input_rows, output_rows)
    inverse_reuse = input_rows <= dispatch_capacity_rows
    inverse_output = _streaming_int4_forward(
        large_output,
        large_input_splits,
        large_output_splits,
        group_name,
        "combine-smoke",
        8,
        inverse_reuse,
    )
    torch.cuda.synchronize()
    print(f"rank={rank} constrained-combine-done", flush=True)
    assert inverse_output.shape == (input_rows, cols)
    assert inverse_output.data_ptr() == large_output.data_ptr()
    large_output = inverse_output
    # The autograd function is the sole consumer. Once its forward returns,
    # the caller replaces the routed-input name with the returned tensor. Do
    # the same here so a non-reused source does not create an artificial peak.
    del large_input
    large_output.zero_()
    large_grad = _streaming_fp8_backward(
        large_output,
        large_input_splits,
        large_output_splits,
        group_name,
        "combine-smoke",
        8,
    )
    torch.cuda.synchronize()
    print(f"rank={rank} constrained-backward-done", flush=True)
    assert large_grad.shape == (output_rows, cols)
    # Dispatch reserved max(input_rows, output_rows) rows before the memory
    # peak. Backward must use that physical capacity even when this tensor's
    # current logical row count is smaller than the returned gradient.
    assert large_grad.data_ptr() == large_output.data_ptr()
    assert torch.count_nonzero(large_grad[:2]).item() == 0
    after_backward, _ = torch.cuda.mem_get_info()

    # Pipeline/autograd can return a contiguous gradient view with a nonzero
    # storage offset. Verify that usable storage after that offset is counted,
    # so a small routing imbalance does not request another full BF16 result.
    del large_grad, large_output, ballast
    torch.cuda.empty_cache()
    offset_rows = 17
    offset_storage = torch.zeros(
        (offset_rows + max(input_rows, output_rows), cols),
        device="cuda",
        dtype=torch.bfloat16,
    )
    offset_grad = offset_storage[offset_rows : offset_rows + input_rows]
    assert offset_grad.is_contiguous() and offset_grad.storage_offset() > 0
    offset_result = _streaming_fp8_backward(
        offset_grad,
        large_input_splits,
        large_output_splits,
        group_name,
        "offset-capacity-smoke",
        8,
    )
    torch.cuda.synchronize()
    assert offset_result.shape == (output_rows, cols)
    assert offset_result.data_ptr() == offset_grad.data_ptr()

    if dist.get_rank() == 0:
        print(
            f"PASS exact forward/backward parity and constrained unequal forward/backward "
            f"(free_before={before / 2**20:.1f}MiB "
            f"after_forward={after_forward / 2**20:.1f}MiB "
            f"after_backward={after_backward / 2**20:.1f}MiB)",
            flush=True,
        )
    del offset_result, offset_grad, offset_storage
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
