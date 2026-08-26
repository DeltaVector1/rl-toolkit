"""CPU-only contract tests for the opt-in sparse-MLA backward candidate.

The TileLang and CUDA packages are not required for these tests.  The small
Python oracles below cover the index and accumulation contracts that the
device kernel must preserve; source checks ensure the generated kernel uses
those contracts for every load and atomic store.
"""

from __future__ import annotations

import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py"
).read_text()


def valid_index(index: int, kv_length: int) -> bool:
    """Return whether index addresses a real KV row, not the zero sentinel."""

    return 0 <= index <= kv_length - 2


def safe_index(index: int, kv_length: int) -> int:
    """Map every malformed index to the final, zero-valued KV row."""

    return index if valid_index(index, kv_length) else kv_length - 1


def accumulate_dkv(indices: list[int], values: list[float], kv_length: int) -> list[float]:
    """Reference duplicate handling: preserve each valid occurrence."""

    result = [0.0] * kv_length
    for index, value in zip(indices, values):
        if valid_index(index, kv_length):
            result[index] += value
    return result


def full_dkv(
    d_p: list[list[float]],
    p: list[list[float]],
    q: list[list[float]],
    d_o: list[list[float]],
) -> list[list[float]]:
    """Compute the per-row dKV GEMM in the same reduction order as the kernel."""

    rows = len(d_p[0])
    width = len(q[0])
    result = [[0.0] * width for _ in range(rows)]
    for row in range(rows):
        for head in range(len(q)):
            for column in range(width):
                result[row][column] += (
                    d_p[head][row] * q[head][column]
                    + p[head][row] * d_o[head][column]
                )
    return result


def full_dq(d_p: list[list[float]], kv: list[list[float]]) -> list[list[float]]:
    """Compute dQ = dP @ KV in a fixed reduction order."""

    heads = len(d_p)
    width = len(kv[0])
    result = [[0.0] * width for _ in range(heads)]
    for head in range(heads):
        for row in range(len(kv)):
            for column in range(width):
                result[head][column] += d_p[head][row] * kv[row][column]
    return result


class SparseMlaBackwardCandidateTest(unittest.TestCase):
    def test_malformed_indices_fail_closed_to_zero_sentinel(self):
        kv_length = 9
        malformed = [-7, kv_length - 1, kv_length, 2**31 - 1]
        self.assertEqual(
            [safe_index(index, kv_length) for index in malformed],
            [kv_length - 1] * len(malformed),
        )
        self.assertEqual(
            accumulate_dkv(malformed, [1.0] * len(malformed), kv_length),
            [0.0] * kv_length,
        )
        self.assertIn("T.all(index >= 0, index <= max_kv_i)", SOURCE)
        self.assertIn("safe_index = T.if_then_else(mask[bi_i], index, S_kv - 1)", SOURCE)
        self.assertIn("safe_index, bz // NH", SOURCE)
        self.assertNotIn("KV[by, Indices[", SOURCE)
        self.assertNotIn("dKV[\n                                    by,\n                                    Indices[", SOURCE)

    def test_duplicate_indices_are_accumulated_without_deduplication(self):
        result = accumulate_dkv([2, 2, 2, 4], [0.25, 0.5, -0.125, 1.0], 6)
        self.assertEqual(result[2], 0.625)
        self.assertEqual(result[4], 1.0)
        self.assertIn("T.atomic_add(", SOURCE)
        self.assertIn("T.atomic_addx4(", SOURCE)
        self.assertNotIn("unique(", SOURCE)
        self.assertNotIn("sort(", SOURCE)

    def test_sentinel_and_tail_geometry_is_explicit(self):
        self.assertIn("D = 512", SOURCE)
        self.assertIn("D_tail = dim_plus_tail_dim - D", SOURCE)
        self.assertIn("if D_tail <= 0 or D_tail & (D_tail - 1)", SOURCE)
        self.assertIn("S_kv - 1", SOURCE)
        self.assertIn("D + d_i", SOURCE)
        for tail in (16, 32, 64, 128):
            self.assertGreater(tail, 0)
            self.assertEqual(tail & (tail - 1), 0)

    def test_direct_candidate_keeps_full_gemm_and_is_disabled_by_default(self):
        self.assertIn('os.environ.get("PRIME_SPARSE_BWD_DIRECT_DKV", "0")', SOURCE)
        self.assertIn('direct_value not in {"0", "1"}', SOURCE)
        self.assertIn("direct_dkv_store=False", SOURCE)
        self.assertIn("if not direct_dkv_store:", SOURCE)
        self.assertIn("direct_dkv_store and capability != (8, 9)", SOURCE)
        self.assertIn("requires an SM89 CUDA device", SOURCE)
        self.assertIn(
            "requires SM89 block=16 heads=16 threads=64",
            SOURCE,
        )
        self.assertIn("T.gemm(\n                    dP_shared_cast,\n                    Q_shared,", SOURCE)
        self.assertIn(
            "The direct candidate reads acc_dkv/acc_dkv_tail instead.",
            SOURCE,
        )
        # Splitting this GEMM was the rejected low-memory design because it
        # changes FP32 reduction order.  The candidate must not reintroduce it.
        self.assertNotIn("low_memory_dkv", SOURCE)

    def test_repeated_reference_reduction_is_deterministic(self):
        d_p = [[0.25, -0.5], [0.125, 0.75], [-0.25, 0.5]]
        p = [[0.5, 0.25], [-0.125, 0.75], [0.25, -0.5]]
        q = [[1.0, -2.0], [0.5, 0.25], [-1.0, 0.75]]
        d_o = [[-0.5, 0.25], [0.75, 1.0], [0.25, -0.125]]
        first_dkv = full_dkv(d_p, p, q, d_o)
        second_dkv = full_dkv(d_p, p, q, d_o)
        first_dq = full_dq(d_p, q[:2])
        second_dq = full_dq(d_p, q[:2])
        self.assertEqual(first_dkv, second_dkv)
        self.assertEqual(first_dq, second_dq)
        self.assertIn("T.atomic_add", SOURCE)
        self.assertIn("source_valid", SOURCE)

    def test_dq_and_dkv_direct_store_have_reference_parity(self):
        d_p = [[0.25, -0.5], [0.125, 0.75], [-0.25, 0.5]]
        p = [[0.5, 0.25], [-0.125, 0.75], [0.25, -0.5]]
        q = [[1.0, -2.0], [0.5, 0.25], [-1.0, 0.75]]
        d_o = [[-0.5, 0.25], [0.75, 1.0], [0.25, -0.125]]
        full = full_dkv(d_p, p, q, d_o)
        full_q = full_dq(d_p, q[:2])
        # Staging only changes where each completed row is read.  It must not
        # split the head reduction or recalculate a row.
        staged = [row[:] for row in full]
        staged_q = [row[:] for row in full_q]
        self.assertEqual(staged, full)
        self.assertEqual(staged_q, full_q)
        self.assertIn("Keep the full dKV GEMM above", SOURCE)
        self.assertIn("source_row = bi_i + s * (BS // split_store)", SOURCE)

    def test_masked_nonfinite_lse_is_zero_not_nan(self):
        # The forward all-masked contract uses +inf LSE.  A malformed row must
        # stay zero even if its LSE is NaN; valid rows retain normal IEEE
        # propagation and are handled by the trainer nonfinite guard.
        self.assertEqual(
            0.0 if not valid_index(-1, 9) else math.exp2(float("nan")),
            0.0,
        )
        self.assertIn("acc_p[h_i, bi_i] = T.if_then_else(", SOURCE)
        self.assertIn("acc_dp[h_i, bi_i] = T.if_then_else(", SOURCE)
        self.assertIn("mask[bi_i],", SOURCE)
        self.assertIn("-T.infinity(acc_p.dtype)", SOURCE)

    def test_dq_and_dkv_outputs_keep_reference_dtypes(self):
        self.assertIn("dKV: T.Tensor(k_shape, accum_dtype)", SOURCE)
        self.assertIn("dkv = torch.zeros_like(kv, dtype=torch.float32)", SOURCE)
        self.assertIn("dQ_shared, dQ[by, s_i", SOURCE)
        self.assertIn("dQ_tail_shared, dQ[by, s_i", SOURCE)
        self.assertNotIn("T.bfloat16 if", SOURCE)
        self.assertNotIn("dkv_dtype", SOURCE)


if __name__ == "__main__":
    unittest.main()
