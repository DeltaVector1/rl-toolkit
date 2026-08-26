# Sparse MLA backward redesign

This is a design and benchmark plan. It does not change the active trainer.

## Measurements

The reported production SM89 backward time is 36.4 CUDA seconds. The current
Ada configuration is `S=4096`, `H=64`, `topk=2048`, `D=512`,
`D_tail=64`, `block_size=16`, `block_H=16`, and `threads=64`.

The current kernel has 16,384 blocks and 128 key tiles in each block. The SM89
path uses scalar FP32 atomics. It therefore issues:

```text
4096 * 4 * 128 * 16 * 576 = 19,327,352,832 scalar dKV atomic updates
```

This is the main target. The dKV destination has only 4,096 * 576 FP32
elements, so many updates serialize on the same cache lines. A zero-sentinel
slot also receives updates in the current path. In the causal benchmark shape,
the index construction has 2,096,128 invalid entries out of 8,388,608 entries
(25 percent). Skipping these entries alone is not sufficient: a local pinned
container control run changed 3.51--3.68 ms to 3.49--3.63 ms on an RTX A4000.
The control change was discarded.

The current path performs eight logical GEMM operations for each key tile:
two score GEMMs, one `dO @ K` GEMM, two dQ GEMMs, and three dKV GEMMs. The
backward is split into `preprocess`, the main kernel, and `postprocess`.

The deployment pins are PyTorch `2.11.0+cu128`, CUDA `12.8`, TileLang `0.1.9`,
and Triton `3.6.0`. These values come from `docs/PINS.md`, the locked package
metadata used by the GLM image, and the GLM Docker build. The local control
compiled with TileLang 0.1.9 in the pinned image, but the available local GPU
was SM86. There is no SM89 performance result for a new kernel.

## Candidate with a real reduction

Use two CUDA kernels and an index-preparation pass:

1. Keep a q-major kernel for dQ. Remove its three dKV GEMMs and all dKV
   atomics. It keeps the existing Q/KV tile layout and its FP32 dQ reduction.
2. Add a key-major kernel for dKV. Give each program a key tile and a query
   chunk. Accumulate the dKV tile in FP32 registers or shared memory and write
   one FP32 partial tile to a unique location. A second reduction kernel sums
   query chunks into the FP32 dKV buffer and casts to BF16.
3. Build a reverse index once per backward. Flatten `(batch, query, group,
   slot)` entries, discard only indices outside `[0, S_kv - 1)`, and stable-sort
   by key tile. Keep every valid entry, including duplicates. The reverse index
   must contain the query and group for each entry; it must not deduplicate.

The key-major program can process a query chunk with a masked 16-key tile. It
reuses the loaded KV tile for all queries in that chunk and never uses a dKV
atomic. Query chunks are independent, so the partial buffer has a unique owner
and the final reduction has a fixed order. The first implementation should
sweep query chunks of 128, 256, and 512 rows. The partial sizes at the
production shape are:

| Query chunk | Partial FP32 dKV storage |
|---:|---:|
| 128 | 288 MiB |
| 256 | 144 MiB |
| 512 | 72 MiB |

These sizes are bounded workspace, not extra model state. A chunk of 256 gives
4,096 key-tile by 16 query-chunk programs and is the starting point. The
kernel must use a tile-level reverse index to avoid scanning all query rows
when routing is sparse. If the tile occupancy is near one, as it is for a
random half-length top-k list grouped into 16-key tiles, the benchmark must count
the masked work and may reject this variant.

The split recomputes score and dP work. It changes the logical GEMM count from
8 to 11, but removes about 19.3 billion scalar atomics and changes dKV writes
from contended read-modify-write operations to approximately 144 MiB of
unique partial writes plus 144 MiB of reduction reads at query chunk 256. It
is a valid candidate only if the larger query-tile GEMMs and the atomic removal
repay the extra score and dP work. This is a measurable tradeoff, not an
assumed speedup.

Do not save P or dP to avoid recomputation. At this shape, BF16 dP storage is
approximately 1 GiB and FP32 dP storage is approximately 2 GiB. This exceeds
the bounded workspace goal and would increase pipeline live memory.

## Numerical and shape contract

- Inputs Q, KV, and dO stay BF16.
- dQ and every dKV partial use FP32 accumulation. The final dKV is cast to
  BF16 only after the reduction.
- A valid duplicate index creates another contribution. It must not be
  deduplicated.
- The final KV row is the zero sentinel and is excluded from both kernels.
  Negative and out-of-range indices are also excluded after a guarded load.
- `D=512` and `D_tail=64` remain separate MMA tiles unless a pinned TileLang
  compile proves a single 576-column tile has the same output and gradient
  tolerance. The tail must never be dropped or padded into an uninitialized
  value.
- Stable sorting and a fixed chunk reduction are required for deterministic
  dKV output. Use the current candidate tolerance (`rtol=2e-2`,
  `atol=2e-3`) as the initial dKV gate, and record the tighter error from a
  deterministic reference. dQ must remain bitwise repeatable on SM89.
- The sentinel and malformed-index cases must return finite zero dQ and dKV
  contributions, including `LSE=+inf` for a fully masked query.

TileLang 0.1.9 does not provide a project-level ragged key-major primitive.
The implementation should therefore start as a standalone Triton 3.6.0
kernel or a small CUDA extension. CUTLASS is not a runtime dependency in the
pinned image. A TileLang port is acceptable only after the exact SM89 shape
compiles without relying on newer pass names or warp-specialization behavior.

## Microbenchmark plan

Run on one idle RTX 4090 in the pinned trainer image. Do not run a production
service. Report CUDA-event time and wall time separately.

1. Warm each exact shape before timing. Record baseline 36.4 s, the number of
   kernels, peak allocation, and Nsight Compute counters for FP32 atomics,
   global load/store bytes, tensor-core instructions, and achieved occupancy.
2. Compare the baseline against the discarded control with invalid stores
   skipped. This establishes the cost of the mask and confirms that the
   measured gain comes from the new reduction.
3. Time q-major dQ-only, key-major dKV-only, reverse-index construction, and
   final reduction separately. Sweep query chunks 128, 256, and 512; key tiles
   8, 16, and 32; and one versus two staging buffers.
4. Use four routing histograms: the production causal list, random unique
   indices, duplicate-heavy indices, and 100 percent sentinel indices. Include
   `S=4096`, `topk=2048`, `H=64`, `D_tail=64`, and a short exact-shape case
   with non-power-of-two sequence tails.
5. Compare output, dQ, and dKV with the current kernel and a small PyTorch
   reference. Check max absolute error, max relative error, nonfinite values,
   and repeated-call equality. Check all valid duplicate entries contribute
   exactly once.
6. Reject the candidate if the end-to-end CUDA time is not at least 10 percent
   below 36.4 s (32.76 s), if the partial dKV workspace exceeds 288 MiB (report
   reverse-index storage separately), if any shape does not compile in the
   pinned image, or if the numerical contract fails.

No implementation is committed from this worktree. The only tested change,
sentinel-store suppression, was not materially faster and was reverted.
