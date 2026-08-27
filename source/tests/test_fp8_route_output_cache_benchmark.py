import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "ops"
    / "benchmark_fp8_route_output_cache.py"
).read_text()


def constants() -> dict[str, object]:
    tree = ast.parse(SCRIPT)
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


class FP8RouteOutputCacheBenchmarkTests(unittest.TestCase):
    def test_profile_is_production_shape_and_deterministic(self):
        values = constants()
        self.assertEqual(values["SOURCE_ROWS"], 8192)
        self.assertEqual(values["LOCAL_EXPERTS"], 32)
        self.assertEqual(values["EP_SIZE"], 8)
        self.assertEqual(values["WIDTH"], 6144)
        self.assertEqual(values["INTERMEDIATE"], 2048)
        self.assertEqual(values["RANK"], 32)
        self.assertEqual(values["CHUNK_ROWS"], 4096)
        self.assertEqual(values["WARMUP"], 3)
        self.assertEqual(values["MEASURED"], 7)
        self.assertIn("torch.use_deterministic_algorithms(True)", SCRIPT)
        self.assertIn("generate_permute_indices(", SCRIPT)
        self.assertIn("ReceivedFP8ExpertInput(local_recompute=True)", SCRIPT)

    def test_profile_checks_all_gradients_exactly(self):
        for name in (
            '"forward"',
            '"input_grad"',
            '"route_score_grad"',
            '"w1_lora_A"',
            '"w1_lora_B"',
            '"w2_lora_A"',
            '"w2_lora_B"',
            '"w3_lora_A"',
            '"w3_lora_B"',
        ):
            self.assertIn(name, SCRIPT)
        self.assertIn("torch.equal(got, expected)", SCRIPT)
        self.assertIn('"parity": parity_ok', SCRIPT)

    def test_profile_measures_transfer_memory_timing_and_cleanup(self):
        values = constants()
        self.assertEqual(values["MIN_MODELED_STEP_SAVING_SECONDS"], 10.0)
        self.assertIn('direction == "d2h"', SCRIPT)
        self.assertIn('direction == "h2d"', SCRIPT)
        self.assertIn("pinned_cache_bytes", SCRIPT)
        self.assertIn("torch.cuda.max_memory_allocated()", SCRIPT)
        self.assertIn("forward_ms_median", SCRIPT)
        self.assertIn("backward_ms_median", SCRIPT)
        self.assertIn("weakref.ref(value)", SCRIPT)
        self.assertIn('"saved_tensor_cleanup": cleanup_ok', SCRIPT)
        self.assertIn("CUDA_VISIBLE_DEVICES", SCRIPT)


if __name__ == "__main__":
    unittest.main()
