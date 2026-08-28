from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BWD_SOURCE = (ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py").read_text()
SOURCE = (ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_two_stage.py").read_text()
BENCHMARK = (ROOT.parent / "scripts/ops/benchmark_sparse_mla_two_stage_sm89.py").read_text()

TOTAL_WIDTH = 576
TOPK = 2048
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
ROUTE_BYTES_PER_ENTRY = 20
ROUTE_FILTER_BYTES_PER_ENTRY = 17


def memory_bound(sequence_kv: int) -> int:
    route_count = 4096 * TOPK
    route_filter = 4096 * TOPK * ROUTE_FILTER_BYTES_PER_ENTRY
    route_sort = route_count * ROUTE_BYTES_PER_ENTRY
    fp32_output = sequence_kv * TOTAL_WIDTH * 4
    reduction = (
        (route_count * (4 + 8))
        + ((sequence_kv + 1) * 8)
        + fp32_output
    )
    output_pair = sequence_kv * TOTAL_WIDTH * (4 + 2)
    return max(
        route_filter + fp32_output,
        route_sort + fp32_output,
        reduction,
        output_pair,
    )


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
        self.assertEqual(4096 * TOPK * ROUTE_BYTES_PER_ENTRY, 167772160)
        self.assertEqual(total, 243271936)
        self.assertLess(total, MAX_WORKSPACE_BYTES)
        self.assertLess(total / 2**20, 256)

    def test_exact_shape_gate_fails_closed(self):
        self.assertIn("capability != SM89", SOURCE)
        self.assertIn("(1, QUERY_TOKENS, ATTENTION_HEADS, TOTAL_WIDTH)", SOURCE)
        self.assertIn("(1, QUERY_TOKENS, GROUPS, TOPK)", SOURCE)
        self.assertIn("(1, 32768 + 1, GROUPS, TOTAL_WIDTH)", SOURCE)
        self.assertIn("PACKED_ROUTE_BYTES = 4", SOURCE)
        self.assertIn("ROUTE_ORDER_BYTES = 8", SOURCE)
        self.assertIn("ARGSORT_SCRATCH_BYTES = 8", SOURCE)
        self.assertIn("ROUTE_BYTES_PER_ENTRY = PACKED_ROUTE_BYTES", SOURCE)
        self.assertIn("ROUTE_FILTER_BYTES_PER_ENTRY = 4 + 1 + 4 + 4 + 4", SOURCE)

    def test_duplicate_routes_reduce_in_stable_key_order(self):
        reduced, unique = stable_segment_sum(
            [[1.0, 2.0], [3.0, 4.0], [-2.0, 1.0]], [4, 4, 7]
        )
        self.assertEqual(unique, [4, 7])
        self.assertEqual(reduced, [[4.0, 6.0], [-2.0, 1.0]])

    def test_benchmark_uses_production_shape_and_host_timeout(self):
        self.assertIn("SEQUENCE_KV = 32769", BENCHMARK)
        self.assertIn("route_skew=2048_keys_x_4096_routes", BENCHMARK)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL", BENCHMARK)
        self.assertIn("peak_reserved_delta", BENCHMARK)
        self.assertIn("peak_allocated_delta", BENCHMARK)
        self.assertIn("measured allocated-memory bound failed", BENCHMARK)
        self.assertIn('"PRIME_SPARSE_BWD_REUSE_DQ_SHARED"', BENCHMARK)
        self.assertIn('"PRIME_SPARSE_BWD_BLOCK_SIZE"', BENCHMARK)
        self.assertIn("free_hbm_before", BENCHMARK)

    def test_prototype_is_not_enabled_by_any_run_config(self):
        config_root = ROOT.parent / "configs"
        enabled = []
        for path in config_root.rglob("*.toml"):
            if "PRIME_SPARSE_BWD_TWO_STAGE" in path.read_text():
                enabled.append(str(path.relative_to(config_root)))
        self.assertEqual(enabled, [])

    def test_workspace_limit_validation_is_fail_closed(self):
        self.assertIn("max_workspace_bytes <= 0", SOURCE)
        self.assertIn("not isinstance(max_workspace_bytes, int)", SOURCE)
        self.assertIn("workspace limit must be a positive integer", SOURCE)

    def test_integration_is_opt_in_and_skips_atomic_dkv_kernel(self):
        self.assertIn('os.environ.get("PRIME_SPARSE_BWD_TWO_STAGE", "0")', BWD_SOURCE)
        self.assertIn("compute_dkv=not use_two_stage", BWD_SOURCE)
        self.assertIn("sparse_mla_dkv_two_stage", BWD_SOURCE)
        self.assertIn("torch.empty_like(kv, dtype=torch.float32)", BWD_SOURCE)
        self.assertIn("output=dkv", BWD_SOURCE)
        self.assertIn("if compute_dkv:", BWD_SOURCE)
        self.assertIn("MAX_WORKSPACE_BYTES = 256 * 1024 * 1024", SOURCE)
        self.assertNotIn("torch.nonzero", SOURCE)
        self.assertNotIn(".item()", SOURCE)
        self.assertNotIn("torch.isfinite(output).all()", SOURCE)
        self.assertNotIn("torch.nan_to_num", SOURCE)
        self.assertNotIn("probability == probability", SOURCE)
        self.assertNotIn("dp == dp", SOURCE)
        self.assertIn("BLOCK_WIDTH = 1024", SOURCE)
        self.assertIn("grid = (sequence_kv,)", SOURCE)
        self.assertIn("torch.argsort(route_ids, stable=True)", SOURCE)
        self.assertIn("QUERY_ID_BITS = 12", SOURCE)
        self.assertIn("route_ids = (keys << QUERY_ID_BITS)", SOURCE)
        self.assertNotIn("sorted_positions", SOURCE)
        self.assertIn("same-device tensors", SOURCE)
        self.assertIn("@triton.jit", SOURCE)


if __name__ == "__main__":
    unittest.main()
