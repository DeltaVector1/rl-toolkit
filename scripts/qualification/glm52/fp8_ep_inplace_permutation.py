"""SM89 exactness and memory check for in-place streamed FP8 EP experts."""

from __future__ import annotations

import json

import torch

from prime_rl.trainer.models.layers import fp8_ada


mib = 1024**2
target_free_mib = 1024
rows = 36_352
padding = 512
input_width = 6_144
intermediate_width = 2_048
experts = 2


def fp8_weight(shape, generator):
    return (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        .mul_(0.05)
        .to(torch.float8_e4m3fn)
    )


def scale_for(weight):
    return torch.full(
        (
            weight.shape[0],
            weight.shape[1] // 128,
            weight.shape[2] // 128,
        ),
        0.02,
        device="cuda",
        dtype=torch.float32,
    )


def reference(value, weights, scales, indices, offsets, upstream):
    real = (indices >= 0) & (indices < value.shape[0])
    safe_indices = indices.clamp(min=0, max=value.shape[0] - 1).long()
    permuted = value.index_select(0, safe_indices)
    permuted.masked_fill_((~real).unsqueeze(-1), 0)
    expert_output = fp8_ada._Experts.apply(
        permuted,
        weights[0],
        scales[0],
        weights[1],
        scales[1],
        weights[2],
        scales[2],
        offsets,
        None,
    )
    output = torch.zeros_like(value).index_copy(
        0,
        indices[real].long(),
        expert_output[real],
    )
    output.backward(upstream)
    return output.detach().cpu(), value.grad.detach().cpu()


def inplace(value_leaf, weights, scales, indices, offsets, upstream):
    value = value_leaf * 1
    input_pointer = value.data_ptr()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = fp8_ada._InplacePermutedExperts.apply(
            value,
            weights[0],
            scales[0],
            weights[1],
            scales[1],
            weights[2],
            scales[2],
            indices,
            offsets,
            None,
            fp8_ada._FP8_W13_CHUNK_ROWS,
        )
        if output.data_ptr() != input_pointer:
            raise RuntimeError("streamed EP experts did not reuse routed-input storage")
        output.backward(upstream)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    host_fp8 = [
        tensor
        for tensor in saved
        if tensor.device.type == "cpu" and tensor.dtype == torch.float8_e4m3fn
    ]
    return output.detach().cpu(), value_leaf.grad.detach().cpu(), peak, host_fp8


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("this check requires an SM89 RTX 4090")
    torch.cuda.set_device(0)
    fp8_ada._FP8_EXPERT_ACTIVATION_OFFLOAD = True
    generator = torch.Generator(device="cuda").manual_seed(97)
    weights = (
        fp8_weight((experts, intermediate_width, input_width), generator),
        fp8_weight((experts, input_width, intermediate_width), generator),
        fp8_weight((experts, intermediate_width, input_width), generator),
    )
    scales = tuple(scale_for(weight) for weight in weights)
    permutation = torch.randperm(rows, device="cuda", generator=generator)
    indices = torch.cat(
        (
            permutation[:20_000],
            permutation[20_000:],
            torch.full((padding // 2,), -1, device="cuda", dtype=torch.int64),
            torch.full(
                (padding // 2,),
                rows,
                device="cuda",
                dtype=torch.int64,
            ),
        )
    ).to(torch.int32)
    offsets = torch.tensor(
        [20_000, rows + padding],
        device="cuda",
        dtype=torch.int32,
    )
    template = torch.randn(
        rows,
        input_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    upstream = torch.randn(
        rows,
        input_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    reference_value = template.detach().clone().requires_grad_()
    expected_output, expected_grad = reference(
        reference_value,
        weights,
        scales,
        indices,
        offsets,
        upstream,
    )
    del reference_value
    torch.cuda.empty_cache()

    inplace_value = template.detach().clone().requires_grad_()
    # Account for the leaf-gradient destination before constraining free
    # memory. The production expert input is an intermediate activation, so
    # autograd forwards its returned gradient instead of allocating a new
    # 426 MiB leaf buffer after the expert backward has completed.
    inplace_value.grad = torch.zeros_like(inplace_value)
    free_bytes, _ = torch.cuda.mem_get_info()
    target_free = target_free_mib * mib
    ballast = torch.empty(
        max(0, free_bytes - target_free), device="cuda", dtype=torch.uint8
    )
    constrained_free_before, _ = torch.cuda.mem_get_info()
    got_output, got_grad, peak, host_fp8 = inplace(
        inplace_value,
        weights,
        scales,
        indices,
        offsets,
        upstream,
    )
    constrained_free_after, _ = torch.cuda.mem_get_info()
    torch.testing.assert_close(got_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(got_grad, expected_grad, rtol=0, atol=0)
    if len(host_fp8) != 2 or not all(tensor.is_pinned() for tensor in host_fp8):
        raise RuntimeError(
            "in-place streamed experts did not retain two pinned FP8 tensors"
        )
    host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_fp8)
    expected_host_bytes = 2 * (rows + padding) * intermediate_width
    if host_bytes != expected_host_bytes:
        raise RuntimeError(
            f"saved FP8 host bytes {host_bytes} do not match {expected_host_bytes}"
        )

    print(
        json.dumps(
            {
                "result": "PASS",
                "source_shape": [rows, input_width],
                "permuted_rows": rows + padding,
                "intermediate": intermediate_width,
                "chunk_rows": fp8_ada._FP8_W13_CHUNK_ROWS,
                "backward_chunk_rows": fp8_ada._FP8_EXPERT_BACKWARD_CHUNK_ROWS,
                "forward": "exact",
                "input_gradient": "exact",
                "routed_input_storage": "reused",
                "saved_fp8_location": "pinned CPU",
                "saved_fp8_mib": host_bytes / mib,
                "incremental_peak_mib": peak / mib,
                "constrained_free_before_mib": constrained_free_before / mib,
                "constrained_free_after_mib": constrained_free_after / mib,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    del ballast


if __name__ == "__main__":
    main()
