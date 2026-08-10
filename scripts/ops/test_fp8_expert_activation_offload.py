"""SM89 correctness and memory gate for streamed FP8 expert activations.

Run this in the trainer image on one free RTX 4090. It compares the ordinary
full-row fused expert graph with the streamed W1/W3 graph that retains exact
FP8 activations in pinned host memory.
"""

from __future__ import annotations

import json

import torch

from prime_rl.trainer.models.layers import fp8_ada


MIB = 1024**2
ROWS = 36_352
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


def run(value, weights, scales, offsets, upstream, offload):
    fp8_ada._FP8_EXPERT_ACTIVATION_OFFLOAD = offload
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = fp8_ada._Experts.apply(
            value,
            weights[0],
            scales[0],
            weights[1],
            scales[1],
            weights[2],
            scales[2],
            offsets,
            None,
        )
        output.backward(upstream)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    host_fp8 = [
        tensor
        for tensor in saved
        if tensor.device.type == "cpu" and tensor.dtype == torch.float8_e4m3fn
    ]
    return output.detach().cpu(), value.grad.detach().cpu(), peak, host_fp8


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("this gate requires an SM89 RTX 4090")
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(83)
    weights = (
        fp8_weight((EXPERTS, INTERMEDIATE, INPUT), generator),
        fp8_weight((EXPERTS, INPUT, INTERMEDIATE), generator),
        fp8_weight((EXPERTS, INTERMEDIATE, INPUT), generator),
    )
    scales = tuple(scale_for(weight) for weight in weights)
    offsets = torch.tensor([20_000, ROWS], device="cuda", dtype=torch.int32)
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

    eager_value = template.detach().clone().requires_grad_()
    eager_output, eager_grad, eager_peak, eager_host = run(
        eager_value, weights, scales, offsets, upstream, False
    )
    if eager_host:
        raise RuntimeError(
            "the ordinary expert path unexpectedly offloaded FP8 tensors"
        )

    offload_value = template.detach().clone().requires_grad_()
    got_output, got_grad, offload_peak, host_fp8 = run(
        offload_value, weights, scales, offsets, upstream, True
    )
    torch.testing.assert_close(got_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(got_grad, eager_grad, rtol=0, atol=0)
    if len(host_fp8) != 2 or not all(tensor.is_pinned() for tensor in host_fp8):
        raise RuntimeError(
            "streamed experts did not retain exactly two pinned FP8 host tensors"
        )
    host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_fp8)
    expected_host_bytes = 2 * ROWS * INTERMEDIATE
    if host_bytes != expected_host_bytes:
        raise RuntimeError(
            f"saved FP8 host bytes {host_bytes} do not match {expected_host_bytes}"
        )

    print(
        json.dumps(
            {
                "result": "PASS",
                "shape": [ROWS, INPUT],
                "intermediate": INTERMEDIATE,
                "chunk_rows": fp8_ada._FP8_W2_CHUNK_ROWS,
                "forward": "exact",
                "input_gradient": "exact",
                "saved_fp8_location": "pinned CPU",
                "saved_fp8_mib": host_bytes / MIB,
                "eager_incremental_peak_mib": eager_peak / MIB,
                "offload_incremental_peak_mib": offload_peak / MIB,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
