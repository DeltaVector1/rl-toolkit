import ast
import unittest
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "prime_rl"
    / "trainer"
    / "distributed"
    / "expert_parallel.py"
)
BENCHMARK_PATH = Path(__file__).parent / "manual" / "test_packed_expert_transport.py"
SOURCE = SOURCE_PATH.read_text()
BENCHMARK = BENCHMARK_PATH.read_text()


class PackedExpertTransportTests(unittest.TestCase):
    def test_source_has_one_opt_in_and_one_collective_payload(self):
        tree = ast.parse(SOURCE)
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.assertIn("_pack_value_scale_payload", names)
        self.assertIn("_unpack_value_scale_payload", names)
        self.assertIn("_dequantize_int4_strided_kernel", names)
        self.assertEqual(SOURCE.count("_PACKED_EXPERT_TRANSPORT ="), 1)
        self.assertIn("dist.all_to_all(receive, send, group=group)", SOURCE)

    def test_packed_payload_checks_are_fail_closed(self):
        self.assertIn("values must be a contiguous matrix", SOURCE)
        self.assertIn("scales must be a contiguous vector", SOURCE)
        self.assertIn("values and scales must have equal rows", SOURCE)
        self.assertIn("payload width does not match its scale", SOURCE)
        self.assertIn("payload rows do not match its scale", SOURCE)

    def test_reused_backward_storage_is_packed_before_write(self):
        backward = SOURCE.split("def _streaming_fp8_backward", 1)[1].split(
            "\n\ndef _validate_bf16_dispatch", 1
        )[0]
        self.assertIn("value_bytes = cols", backward)
        self.assertIn("packed_send_all = torch.empty", backward)
        self.assertLess(
            backward.index("_copy_value_scale_payload"),
            backward.index("grad_output.as_strided_((input_rows, cols)"),
        )
        self.assertIn("packed_send_all[offset : offset + count]", backward)

    def test_benchmark_requires_sm89_and_eight_ranks(self):
        self.assertIn("capability != (8, 9)", BENCHMARK)
        self.assertIn("world != 8", BENCHMARK)
        self.assertIn("gpu_increase_mib <= 64.0", BENCHMARK)
        self.assertIn("packed_pinned <= base_pinned", BENCHMARK)
        self.assertIn("speed_gain >= 0.05", BENCHMARK)
        self.assertIn("projected_step_saving >= 10.0", BENCHMARK)


if __name__ == "__main__":
    unittest.main()
