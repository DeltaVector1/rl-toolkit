"""Production-shape CUDA gate for routed-expert inverse permutation memory.

Run on one free RTX 4090 in the trainer image. The row count reproduces the
426 MiB BF16 allocation that failed x60k at hidden size 6,144.
"""

from __future__ import annotations

import json

import torch


GIB = 1024**3
MIB = 1024**2
ROWS = 36_352
HIDDEN = 6_144
SENTINEL_REPEATS = 128


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(61)
    value = torch.randn(
        ROWS,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    padded = torch.vstack((value, value.new_zeros((1, HIDDEN))))
    permutation = torch.randperm(ROWS, device="cuda", generator=generator)
    indices = torch.cat(
        (
            permutation,
            torch.full(
                (SENTINEL_REPEATS,),
                ROWS,
                device="cuda",
                dtype=torch.int64,
            ),
        )
    )
    permuted = padded[indices]
    expert_output = permuted.mul(2)
    del permuted

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = padded
    out[indices] = expert_output
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - before
    if incremental_peak > 64 * MIB:
        raise RuntimeError(
            "inverse permutation allocated another routed-output-sized buffer: "
            f"incremental peak {incremental_peak / MIB:.2f} MiB"
        )

    torch.testing.assert_close(out[:-1], value.mul(2), rtol=0, atol=0)
    out[:-1].sum().backward()
    if value.grad is None or not bool(torch.isfinite(value.grad).all()):
        raise RuntimeError("inverse permutation produced a missing or nonfinite gradient")
    torch.testing.assert_close(
        value.grad,
        torch.full_like(value, 2),
        rtol=0,
        atol=0,
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "shape": [ROWS, HIDDEN],
                "single_buffer_gib": ROWS * HIDDEN * 2 / GIB,
                "incremental_unpermute_peak_mib": incremental_peak / MIB,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "gradient": "exact",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
