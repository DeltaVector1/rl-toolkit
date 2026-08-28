import os

import torch
import triton
import triton.language as tl


@triton.jit
def _write_sequence(output, size: tl.constexpr):
    offsets = tl.arange(0, size)
    tl.store(output + offsets, offsets)


def main() -> None:
    cache = os.environ.get("TRITON_CACHE_DIR")
    if cache != "/tmp/triton":
        raise RuntimeError(f"unexpected Triton cache path: {cache!r}")
    output = torch.empty(32, device="cuda", dtype=torch.int32)
    _write_sequence[(1,)](output, size=32)
    torch.cuda.synchronize()
    expected = torch.arange(32, device="cuda", dtype=torch.int32)
    if not torch.equal(output, expected):
        raise RuntimeError("Triton compile/load preflight returned wrong values")


if __name__ == "__main__":
    main()
