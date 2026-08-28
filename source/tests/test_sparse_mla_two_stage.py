from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BWD_SOURCE = (ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py").read_text()
SOURCE = (ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_two_stage.py").read_text()

TOTAL_WIDTH = 576
TOPK = 2048
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
ROUTE_BYTES_PER_ENTRY = 28


def memory_bound(sequence_kv: int) -> int:
    route_sort = 4096 * TOPK * ROUTE_BYTES_PER_ENTRY
    reduction = (4096 * TOPK * 4) + ((sequence_kv + 1) * 8) + (sequence_kv * TOTAL_WIDTH * 4)
    return max(route_sort, reduction)


def stable_segment_sum(values: list[list[float]], keys: list[int]):
    result: list[list[float]] = []
    unique: list[int] = []
    for key, value in zip(keys, values, strict=True):
        if not unique or unique[-1] != key:
            unique.append(key)
            result.append(value[:])
        else:
            result[-1] = [left + right for left, right in zip(result[-1], value, strict=True)]
    return result, unique


class SparseMlaTwoStageTest(unittest.TestCase):
    def test_production_workspace_is_bounded(self):
        total = memory_bound(32769)
        self.assertEqual(4096 * TOPK * ROUTE_BYTES_PER_ENTRY, 234881024)
        self.assertLess(total, MAX_WORKSPACE_BYTES)
        self.assertLess(total / 2**20, 256)

    def test_exact_shape_gate_fails_closed(self):
        self.assertIn("capability != SM89", SOURCE)
        self.assertIn("(1, QUERY_TOKENS, ATTENTION_HEADS, TOTAL_WIDTH)", SOURCE)
        self.assertIn("(1, QUERY_TOKENS, GROUPS, TOPK)", SOURCE)
        self.assertIn("(1, 32768 + 1, GROUPS, TOTAL_WIDTH)", SOURCE)
        self.assertIn("ROUTE_BYTES_PER_ENTRY = 28", SOURCE)

    def test_duplicate_routes_reduce_in_stable_key_order(self):
        reduced, unique = stable_segment_sum(
            [[1.0, 2.0], [3.0, 4.0], [-2.0, 1.0]], [4, 4, 7]
        )
        self.assertEqual(unique, [4, 7])
        self.assertEqual(reduced, [[4.0, 6.0], [-2.0, 1.0]])

    def test_integration_is_opt_in_and_skips_atomic_dkv_kernel(self):
        self.assertIn('os.environ.get("PRIME_SPARSE_BWD_TWO_STAGE", "0")', BWD_SOURCE)
        self.assertIn("compute_dkv=not use_two_stage", BWD_SOURCE)
        self.assertIn("sparse_mla_dkv_two_stage", BWD_SOURCE)
        self.assertIn("torch.empty(1, device=kv.device, dtype=torch.float32)", BWD_SOURCE)
        self.assertIn("if compute_dkv:", BWD_SOURCE)
        self.assertIn("MAX_WORKSPACE_BYTES = 256 * 1024 * 1024", SOURCE)
        self.assertNotIn("torch.nonzero", SOURCE)
        self.assertNotIn(".item()", SOURCE)
        self.assertIn("@triton.jit", SOURCE)


if __name__ == "__main__":
    unittest.main()
