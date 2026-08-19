import ast
import types
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
SOURCE = SOURCE_PATH.read_text()


def load_round_splits():
    tree = ast.parse(SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_round_splits"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_round_splits"]


class Int4CombineMemoryTests(unittest.TestCase):
    def test_round_splits_cover_every_peer_segment_once(self):
        round_splits = load_round_splits()
        splits = [5, 0, 8, 2]
        covered = [[] for _ in splits]
        peer_bases = [0, 5, 5, 13]
        for round_idx in range(3):
            counts, offsets = round_splits(splits, round_idx, 3)
            for peer, (count, offset) in enumerate(zip(counts, offsets)):
                covered[peer].extend(range(offset - peer_bases[peer], offset - peer_bases[peer] + count))
        self.assertEqual(covered, [list(range(split)) for split in splits])

    def test_dispatch_and_combine_reuse_input_and_bound_transport_scratch(self):
        function = SOURCE.split("def _streaming_int4_forward", 1)[1].split(
            "\n\nclass _INT4Dispatch", 1
        )[0]
        quantize = function.index("_quantize_int4_kernel")
        loop = function.index("for round_idx in range(rounds)")
        receive = function.index("received = _empty_with_cache_retry")
        self.assertIn("dequantized = routed_input", function)
        self.assertIn("dequantized = _empty_with_cache_retry", function)
        self.assertIn("if reuse_input", function)
        self.assertIn('tag.startswith("dispatch")', function)
        self.assertIn("max(rows, output_rows)", function)
        self.assertIn("untyped_storage().nbytes()", SOURCE)
        self.assertIn("round_quantized = torch.empty", function)
        self.assertIn("output_rows <= storage_rows", SOURCE)
        self.assertIn('tag == "combine"', SOURCE)
        self.assertGreater(quantize, loop)
        self.assertGreater(receive, loop)
        self.assertIn("received_all = _empty_with_cache_retry", function)
        self.assertIn("round_packed_bytes <= tail_bytes", function)
        self.assertIn("routed_input.untyped_storage()", function)
        self.assertIn("if not reuse_input", function)
        self.assertIn("ctx.mark_dirty(routed_input)", SOURCE)
        self.assertIn('tag in {"dispatch", "combine"} and cache is None', SOURCE)

    def test_forward_allocations_have_one_cache_recovery_retry(self):
        function = SOURCE.split("def _streaming_int4_forward", 1)[1].split(
            "\n\ndef _streaming_fp8_backward", 1
        )[0]
        self.assertIn("dequantized = _empty_with_cache_retry(", function)
        self.assertIn("received_all = _empty_with_cache_retry(", function)
        self.assertIn("received = _empty_with_cache_retry(", function)

    def test_combine_backward_is_streamed_and_reuses_incoming_gradient(self):
        function = SOURCE.split("def _streaming_fp8_backward", 1)[1].split(
            "\n\nclass _INT4Dispatch", 1
        )[0]
        self.assertIn("for round_idx in range(rounds)", function)
        self.assertIn("dequantized = grad_output", function)
        self.assertIn("grad_output.untyped_storage().nbytes()", function)
        self.assertIn("storage_elements - grad_output.storage_offset()", function)
        self.assertIn("input_rows <= storage_rows", function)
        self.assertIn("for count, offset in zip(send_counts, send_offsets)", function)
        self.assertIn("grad_output[offset : offset + count]", function)
        self.assertIn("dist.all_to_all(receive, send", function)
        self.assertIn("ctx.rounds > 1", SOURCE)

    def test_combine_backward_receive_has_one_cache_recovery_retry(self):
        function = SOURCE.split("def _streaming_fp8_backward", 1)[1].split(
            "\n\nclass _INT4Dispatch", 1
        )[0]
        self.assertEqual(function.count("_empty_with_cache_retry("), 2)
        result_allocation = function.index("dequantized = _empty_with_cache_retry(")
        transport_loop = function.index("for round_idx in range(rounds)")
        receive_allocation = function.index(
            "received = _empty_with_cache_retry(", transport_loop
        )
        self.assertLess(result_allocation, transport_loop)
        self.assertGreater(receive_allocation, transport_loop)

    def test_cache_recovery_retries_exact_allocation_once(self):
        tree = ast.parse(SOURCE)
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_empty_with_cache_retry"
        )
        calls = []

        class FakeOOM(Exception):
            pass

        def empty(*size, **kwargs):
            calls.append((size, kwargs))
            if len(calls) == 1:
                raise FakeOOM
            return "allocated"

        cache_clears = []
        fake_torch = types.SimpleNamespace(
            empty=empty,
            OutOfMemoryError=FakeOOM,
            cuda=types.SimpleNamespace(empty_cache=lambda: cache_clears.append(True)),
        )
        namespace = {"torch": fake_torch}
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE_PATH), "exec"),
            namespace,
        )
        result = namespace["_empty_with_cache_retry"](
            (123, 456), device="cuda:3", dtype="fp8"
        )
        self.assertEqual(result, "allocated")
        self.assertEqual(cache_clears, [True])
        self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
