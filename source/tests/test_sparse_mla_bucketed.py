from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_bucketed.py"
).read_text()
BENCHMARK = (
    ROOT.parent / "scripts/ops/benchmark_sparse_mla_bucketed_sm89.py"
).read_text()


def cpu_bucket(values: list[int], sequence_kv: int) -> tuple[list[int], list[int]]:
    buckets: list[list[int]] = [[] for _ in range(sequence_kv)]
    for source, key in enumerate(values):
        if 0 <= key < sequence_kv - 1:
            buckets[key].append(source)
    offsets = [0]
    sources: list[int] = []
    for bucket in buckets:
        sources.extend(bucket)
        offsets.append(len(sources))
    return sources, offsets


class SparseMlaBucketedTest(unittest.TestCase):
    def test_production_ledger_is_below_256_mib(self):
        route_count = 4096 * 2048
        sequence_kv = 32769
        total = (
            route_count * 4
            + sequence_kv * 4
            + (sequence_kv + 1) * 4
            + sequence_kv * 4
            + (sequence_kv + 1) * 4
            + sequence_kv * 576 * 4
            + sequence_kv * 576 * 2
        )
        self.assertEqual(total, 147328408)
        self.assertLess(total, 256 * 1024 * 1024)
        self.assertIn("route_sources_bytes", SOURCE)
        self.assertIn("prefix_temp_bytes", SOURCE)

    def test_cpu_route_contract_keeps_duplicates_and_drops_sentinel(self):
        sources, offsets = cpu_bucket([2, 4, 2, -1, 5, 7, 8], 9)
        self.assertEqual(sources, [0, 2, 1, 4, 5])
        self.assertEqual(offsets, [0, 0, 0, 2, 2, 3, 4, 4, 5, 5])

    def test_route_builder_has_no_full_route_copy_or_sort(self):
        self.assertNotIn("torch.argsort", SOURCE)
        self.assertNotIn("torch.sort", SOURCE)
        self.assertNotIn("torch.arange", SOURCE)
        self.assertNotIn("masked_select", SOURCE)
        self.assertNotIn("torch.bincount", SOURCE)
        self.assertIn("_count_route_keys", SOURCE)
        self.assertIn("_scatter_route_sources", SOURCE)
        self.assertIn("tl.atomic_add", SOURCE)
        self.assertIn("torch.cumsum(counts, dim=0, dtype=torch.int32)", SOURCE)

    def test_invalid_routes_use_a_safe_pointer_and_are_not_stored(self):
        self.assertIn("safe_key = tl.where(valid, key, 0)", SOURCE)
        self.assertIn("mask=valid", SOURCE)
        self.assertIn("route_sources + destination", SOURCE)
        self.assertIn("key < sequence_kv - 1", SOURCE)

    def test_kernel_uses_source_index_to_recover_query(self):
        self.assertIn("source_index = tl.load(route_sources + route_index)", SOURCE)
        self.assertIn("query_id = source_index // TOPK", SOURCE)
        self.assertNotIn("route_ids", SOURCE)
        self.assertNotIn("route_order", SOURCE)

    def test_candidate_is_exact_shape_and_explicitly_opt_in(self):
        self.assertIn("supports_glm_sm89_shape", SOURCE)
        self.assertIn("torch.cuda.get_device_capability(q.device)", SOURCE)
        self.assertIn("max_workspace_bytes > MAX_WORKSPACE_BYTES", SOURCE)
        config_root = ROOT.parent / "configs"
        for path in config_root.rglob("*.toml"):
            self.assertNotIn('PRIME_SPARSE_BWD_BUCKETED = "1"', path.read_text())

    def test_benchmark_checks_parity_and_both_cuda_memory_counters(self):
        self.assertIn("rtol=2e-2, atol=2e-3", BENCHMARK)
        self.assertIn("peak_allocated_delta", BENCHMARK)
        self.assertIn("peak_reserved_delta", BENCHMARK)
        self.assertIn("measured allocated-memory bound failed", BENCHMARK)
        self.assertIn("measured reserved-memory bound failed", BENCHMARK)
        self.assertIn("order=atomic", BENCHMARK)

    def test_benchmark_covers_wide_and_skewed_route_patterns(self):
        self.assertIn(
            'ROUTE_DISTRIBUTIONS = ("collision", "uniform-wide", "moderate-skew")',
            BENCHMARK,
        )
        self.assertIn('choices=(*ROUTE_DISTRIBUTIONS, "all")', BENCHMARK)
        self.assertIn('route_distribution == "uniform-wide"', BENCHMARK)
        self.assertIn('route_distribution == "moderate-skew"', BENCHMARK)


if __name__ == "__main__":
    unittest.main()
