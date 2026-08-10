"""SM89 exactness and memory gate for in-place streamed FP8 EP experts."""

from __future__ import annotations

import json

import torch

from prime_rl.trainer.models.layers import fp8_ada


MIB = 1024**2
ROWS = 36_352
PADDING = 512
INPUT = 6_144
INTERMEDIATE = 2_048
EXPERTS = 2


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
        raise RuntimeError("this gate requires an SM89 RTX 4090")
    torch.cuda.set_device(0)
    fp8_ada._FP8_EXPERT_ACTIVATION_OFFLOAD = True
    generator = torch.Generator(device="cuda").manual_seed(97)
    weights = (
        fp8_weight((EXPERTS, INTERMEDIATE, INPUT), generator),
        fp8_weight((EXPERTS, INPUT, INTERMEDIATE), generator),
        fp8_weight((EXPERTS, INTERMEDIATE, INPUT), generator),
    )
    scales = tuple(scale_for(weight) for weight in weights)
    permutation = torch.randperm(ROWS, device="cuda", generator=generator)
    indices = torch.cat(
        (
            permutation[:20_000],
            permutation[20_000:],
            torch.full((PADDING // 2,), -1, device="cuda", dtype=torch.int64),
            torch.full(
                (PADDING // 2,),
                ROWS,
                device="cuda",
                dtype=torch.int64,
            ),
        )
    ).to(torch.int32)
    offsets = torch.tensor(
        [20_000, ROWS + PADDING],
        device="cuda",
        dtype=torch.int32,
    )
    template = torch.randn(
        ROWS,
        INPUT,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    upstream = torch.randn(
        ROWS,
        INPUT,
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
    got_output, got_grad, peak, host_fp8 = inplace(
        inplace_value,
        weights,
        scales,
        indices,
        offsets,
        upstream,
    )
    torch.testing.assert_close(got_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(got_grad, expected_grad, rtol=0, atol=0)
    if len(host_fp8) != 2 or not all(tensor.is_pinned() for tensor in host_fp8):
        raise RuntimeError(
            "in-place streamed experts did not retain two pinned FP8 tensors"
        )
    host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_fp8)
    expected_host_bytes = 2 * (ROWS + PADDING) * INTERMEDIATE
    if host_bytes != expected_host_bytes:
        raise RuntimeError(
            f"saved FP8 host bytes {host_bytes} do not match {expected_host_bytes}"
        )

    print(
        json.dumps(
            {
                "result": "PASS",
                "source_shape": [ROWS, INPUT],
                "permuted_rows": ROWS + PADDING,
                "intermediate": INTERMEDIATE,
                "chunk_rows": fp8_ada._FP8_W13_CHUNK_ROWS,
                "forward": "exact",
                "input_gradient": "exact",
                "routed_input_storage": "reused",
                "saved_fp8_location": "pinned CPU",
                "saved_fp8_mib": host_bytes / MIB,
                "incremental_peak_mib": peak / MIB,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
