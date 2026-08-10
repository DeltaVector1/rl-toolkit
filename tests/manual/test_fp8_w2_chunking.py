"""CUDA equivalence and memory gate for routed-expert W2 row chunking."""

from __future__ import annotations

import json

import torch

from prime_rl.trainer.models.layers.fp8_ada import (
    _grouped_mm,
    _grouped_mm_chunked,
)


EXPERTS = 4
INPUT_SIZE = 1536
OUTPUT_SIZE = 6144
ROWS = 9000
COUNTS = (2200, 2200, 2200, 2200)


def measured(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        result = fn()
    end.record()
    end.synchronize()
    return (
        result,
        start.elapsed_time(end) / 10,
        (torch.cuda.max_memory_allocated() - baseline) / 2**20,
    )


if sum(COUNTS) > ROWS:
    raise RuntimeError("expert counts exceed allocated rows")
torch.manual_seed(20260810)
x = torch.randn(ROWS, INPUT_SIZE, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(
    EXPERTS,
    OUTPUT_SIZE,
    INPUT_SIZE,
    device="cuda",
    dtype=torch.bfloat16,
).mul_(0.02).to(torch.float8_e4m3fn)
scale = torch.full(
    (EXPERTS, OUTPUT_SIZE // 128, INPUT_SIZE // 128),
    0.01,
    device="cuda",
    dtype=torch.float32,
)
offsets = torch.tensor(COUNTS, device="cuda", dtype=torch.int32).cumsum(0)

full, full_ms, full_peak = measured(
    lambda: _grouped_mm(x, weight, scale, offsets)
)
chunked, chunked_ms, chunked_peak = measured(
    lambda: _grouped_mm_chunked(
        x,
        weight,
        scale,
        offsets,
        chunk_rows=4096,
    )
)
torch.testing.assert_close(chunked, full, rtol=0, atol=0)
if not bool(torch.isfinite(chunked).all()):
    raise RuntimeError("chunked W2 output is nonfinite")
if not bool((chunked[sum(COUNTS) :] == 0).all()):
    raise RuntimeError("chunked W2 padded rows are not zero")
print(
    json.dumps(
        {
            "result": "PASS",
            "rows": ROWS,
            "chunk_rows": 4096,
            "full_ms": full_ms,
            "chunked_ms": chunked_ms,
            "full_peak_mib": full_peak,
            "chunked_peak_mib": chunked_peak,
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
