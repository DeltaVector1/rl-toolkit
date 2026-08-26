"""CPU contract tests for the opt-in key-major sparse MLA dKV candidate."""

from __future__ import annotations

import unittest

import torch

from prime_rl.trainer.models.kernels.sparse_mla_csr import (
    build_reverse_csr,
    estimate_partial_bytes,
    sparse_mla_dkv_csr,
)


class SparseMlaCsrTest(unittest.TestCase):
    def test_reverse_index_filters_sentinel_and_malformed_entries(self) -> None:
        indices = torch.tensor([[[[2, 2, 4, -1]], [[0, 9, 1, 3]]]], dtype=torch.int32)
        reverse = build_reverse_csr(indices, sequence_kv=5)

        self.assertEqual(reverse.valid_entries, 5)
        self.assertEqual(reverse.keys.tolist(), [0, 1, 2, 2, 3])
        # Equal key entries preserve their original route order.
        self.assertEqual(reverse.positions.tolist(), [4, 6, 0, 1, 7])
        self.assertEqual(reverse.tile_ptr.tolist(), [0, 5])

    def test_duplicate_routes_contribute_once_per_entry(self) -> None:
        sequence, heads = 2, 2
        dim, tail = 4, 4
        q = torch.tensor(
            [[[[1.0, 0.5, -0.5, 0.25, 0.75, 0.5, -0.25, 1.0],
               [0.25, -0.5, 0.75, 0.5, -0.5, 0.25, 1.0, 0.75]],
              [[0.5, 1.0, 0.25, -0.75, 0.5, -0.5, 0.25, 0.75],
               [-0.25, 0.75, 0.5, 1.0, 0.25, 0.5, -0.75, -0.5]]]], dtype=torch.float32
        )
        kv = torch.tensor(
            [[[[0.5, -0.25, 0.75, 1.0, 0.25, 0.5, -0.5, 0.75],
               [0.25, 0.75, -0.5, 0.5, 1.0, -0.25, 0.5, 0.25],
               [-0.5, 0.5, 0.25, 0.75, 0.5, 1.0, -0.25, -0.5],
               [0.75, 0.25, 0.5, -0.5, -0.75, 0.5, 0.25, 1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32
        ).transpose(1, 2).contiguous()
        do = torch.tensor(
            [[[[0.5, -0.25, 0.75, -0.5], [0.25, 0.5, -0.75, 1.0]],
              [[-0.5, 0.25, 0.5, 0.75], [1.0, -0.5, 0.25, 0.5]]]], dtype=torch.float32
        )
        indices = torch.tensor([[[[1, 1, 3, 4]], [[0, -3, 9, 2]]]], dtype=torch.int32)
        lse = torch.tensor([[[1.5, 1.0], [0.75, 1.25]]], dtype=torch.float32)
        delta = torch.tensor([[[0.25, -0.5], [0.5, 0.75]]], dtype=torch.float32)
        result, reverse, partial_bytes = sparse_mla_dkv_csr(
            q,
            kv,
            do,
            indices,
            lse,
            delta,
            query_chunk=1,
            key_tile=2,
            entry_chunk=2,
        )

        expected = torch.zeros_like(kv)
        scale = (dim + tail) ** -0.5
        log2_e = 1.4426950408889634
        for s in range(sequence):
            for rank in range(indices.shape[-1]):
                key = int(indices[0, s, 0, rank])
                if not 0 <= key < kv.shape[1] - 1:
                    continue
                for h in range(heads):
                    score = torch.dot(q[0, s, h], kv[0, key, 0])
                    prob = torch.exp2(score * scale * log2_e - lse[0, s, h])
                    dp_score = torch.dot(do[0, s, h], kv[0, key, 0, :dim])
                    dp = prob * (dp_score - delta[0, s, h]) * scale
                    expected[0, key, 0, :dim] += dp * q[0, s, h, :dim] + prob * do[0, s, h]
                    expected[0, key, 0, dim:] += dp * q[0, s, h, dim:]

        self.assertEqual(reverse.valid_entries, 5)
        self.assertEqual(partial_bytes, estimate_partial_bytes(1, 2, 1, 5, 8, query_chunk=1, key_tile=2))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(result, sparse_mla_dkv_csr(
            q, kv, do, indices, lse, delta, query_chunk=1, key_tile=2, entry_chunk=2
        )[0]))
        self.assertTrue(torch.isfinite(result).all())

    def test_all_invalid_and_infinite_lse_are_exact_zero(self) -> None:
        q = torch.ones(1, 2, 2, 8)
        kv = torch.ones(1, 5, 1, 8)
        do = torch.ones(1, 2, 2, 4)
        indices = torch.tensor([[[[4, -1, 8, 4]], [[4, -2, 99, 4]]]], dtype=torch.int32)
        lse = torch.full((1, 2, 2), float("inf"))
        delta = torch.zeros_like(lse)
        result, reverse, _ = sparse_mla_dkv_csr(q, kv, do, indices, lse, delta, query_chunk=2)

        self.assertEqual(reverse.valid_entries, 0)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertTrue(torch.isfinite(result).all())

    def test_partial_memory_gate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "partial dKV"):
            sparse_mla_dkv_csr(
                torch.ones(1, 8, 2, 8),
                torch.ones(1, 9, 1, 8),
                torch.ones(1, 8, 2, 4),
                torch.zeros(1, 8, 1, 4, dtype=torch.int32),
                torch.zeros(1, 8, 2),
                torch.zeros(1, 8, 2),
                max_partial_bytes=1,
            )

    def test_supported_tail_widths_keep_the_tail_in_the_reduction(self) -> None:
        for tail in (16, 64, 128):
            width = 512 + tail
            q = torch.ones(1, 1, 2, width)
            kv = torch.ones(1, 3, 1, width)
            do = torch.ones(1, 1, 2, 512)
            indices = torch.tensor([[[[0, 1]]]], dtype=torch.int32)
            lse = torch.zeros(1, 1, 2)
            delta = torch.zeros_like(lse)
            result, _, _ = sparse_mla_dkv_csr(
                q, kv, do, indices, lse, delta, query_chunk=1, key_tile=2
            )
            self.assertEqual(tuple(result.shape), tuple(kv.shape))
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreater(float(result[0, 0, 0, 512:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
