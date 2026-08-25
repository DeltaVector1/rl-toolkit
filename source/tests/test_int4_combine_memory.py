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
CONFIG_SOURCE = (
    Path(__file__).parents[1]
    / "packages"
    / "prime-rl-configs"
    / "src"
    / "prime_rl"
    / "configs"
    / "trainer.py"
).read_text()


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


def load_received_input_type():
    tree = ast.parse(SOURCE)
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ReceivedINT4ExpertInput"
    )
    module = ast.Module(body=[definition], type_ignores=[])
    fake_torch = types.SimpleNamespace(Tensor=object, device=object)
    namespace = {"torch": fake_torch}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["ReceivedINT4ExpertInput"]


class FakeTensor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows),)

    def __getitem__(self, row_slice):
        return FakeTensorView(self, row_slice)


class FakeTensorView:
    def __init__(self, tensor, row_slice):
        self.tensor = tensor
        self.row_slice = row_slice

    def copy_(self, source, non_blocking=False):
        del non_blocking
        if isinstance(source, FakeTensorView):
            values = source.tensor.rows[source.row_slice]
        else:
            values = source.rows
        self.tensor.rows[self.row_slice] = values


class Int4CombineMemoryTests(unittest.TestCase):
    def test_config_exposes_explicit_recompute_transport(self):
        self.assertIn('"bf16_forward_fp8_recompute_backward"', CONFIG_SOURCE)

    def test_received_input_copies_rounds_into_source_order(self):
        received_input = load_received_input_type()()
        received_input.columns = 4
        received_input.values = FakeTensor([[0, 0] for _ in range(7)])
        received_input.scales = FakeTensor([0 for _ in range(7)])
        round_values = FakeTensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        round_scales = FakeTensor([10, 11, 12, 13])

        received_input.copy_slices(
            round_values,
            round_scales,
            counts=[2, 0, 2],
            source_offsets=[0, 2, 2],
            destination_offsets=[4, 1, 0],
        )

        self.assertEqual(received_input.values.rows[4:6], round_values.rows[:2])
        self.assertEqual(received_input.values.rows[:2], round_values.rows[2:])
        self.assertEqual(received_input.scales.rows[4:6], round_scales.rows[:2])
        self.assertEqual(received_input.scales.rows[:2], round_scales.rows[2:])

        with self.assertRaisesRegex(ValueError, "metadata lengths"):
            received_input.copy_slices(
                round_values,
                round_scales,
                counts=[1],
                source_offsets=[],
                destination_offsets=[0],
            )
        with self.assertRaisesRegex(ValueError, "source slice"):
            received_input.copy_slices(
                round_values,
                round_scales,
                counts=[2],
                source_offsets=[3],
                destination_offsets=[0],
            )

    def test_received_input_is_owned_by_the_dispatch_call(self):
        dispatch = SOURCE.split("def _token_dispatch", 1)[1].split(
            "\n    def _token_combine", 1
        )[0]
        self.assertIn("ReceivedINT4ExpertInput()", dispatch)
        self.assertIn("requires_received_int4_input", dispatch)
        self.assertIn(
            "return routed_input, num_tokens_per_expert_group, received_input",
            dispatch,
        )
        self.assertNotIn("self._received_input", dispatch)

    def test_received_input_uses_pinned_memory_and_records_completion(self):
        definition = SOURCE.split("class ReceivedINT4ExpertInput", 1)[1].split(
            "\n\nclass INT4ExpertInputWorkspace", 1
        )[0]
        self.assertEqual(definition.count("pin_memory=True"), 2)
        self.assertIn("self.copy_done = torch.cuda.Event()", definition)
        self.assertIn(
            "self.copy_done.record(torch.cuda.current_stream(device))", definition
        )

    def test_reconstruction_uses_caller_output_and_bounded_workspace(self):
        received_input = SOURCE.split("class ReceivedINT4ExpertInput", 1)[1].split(
            "\n\nclass INT4ExpertInputWorkspace", 1
        )[0]
        reconstruction = received_input.split("def reconstruct_rows", 1)[1]
        workspace = SOURCE.split("class INT4ExpertInputWorkspace", 1)[1].split(
            "\n\nclass ReceivedBF16ExpertInput", 1
        )[0]
        self.assertIn("workspace.check(row_indices.shape[0]", reconstruction)
        self.assertIn("self.copy_done.synchronize()", reconstruction)
        self.assertIn("if not self.copy_ready", reconstruction)
        self.assertIn("torch.index_select(self.values", reconstruction)
        self.assertIn("_dequantize_int4_kernel", reconstruction)
        self.assertNotIn("torch.empty", reconstruction)
        self.assertEqual(workspace.count("pin_memory=True"), 2)
        self.assertIn("rows > self.row_capacity", workspace)
        self.assertIn("self.device_values", workspace)
        self.assertIn("self.device_scales", workspace)

    def test_round_splits_cover_every_peer_segment_once(self):
        round_splits = load_round_splits()
        splits = [5, 0, 8, 2]
        covered = [[] for _ in splits]
        peer_bases = [0, 5, 5, 13]
        for round_idx in range(3):
            counts, offsets = round_splits(splits, round_idx, 3)
            for peer, (count, offset) in enumerate(zip(counts, offsets)):
                covered[peer].extend(
                    range(offset - peer_bases[peer], offset - peer_bases[peer] + count)
                )
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
            "\n\ndef _validate_bf16_dispatch", 1
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
            "\n\ndef _validate_bf16_dispatch", 1
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
            compile(
                ast.Module(body=[helper], type_ignores=[]), str(SOURCE_PATH), "exec"
            ),
            namespace,
        )
        result = namespace["_empty_with_cache_retry"](
            (123, 456), device="cuda:3", dtype="fp8"
        )
        self.assertEqual(result, "allocated")
        self.assertEqual(cache_clears, [True])
        self.assertEqual(calls[0], calls[1])

    def test_native_expert_transport_uses_the_requested_codec(self):
        native = SOURCE.split(
            "class NativeFP8LoRAAllToAllExpertParallel", 1
        )[1].split("\n\nclass DeepEPExpertParallel", 1)[0]
        self.assertIn('"bf16_forward_fp8_recompute_backward"', native)
        self.assertIn("recompute_fp8", native)
        self.assertIn("_in_backward()", native)
        self.assertIn("ReceivedBF16ExpertInput(", native)
        self.assertIn("ReceivedFP8ExpertInput()", native)
        self.assertIn("_BF16Dispatch.apply", native)
        self.assertIn("_FP8Dispatch.apply", native)
        self.assertIn("expert_activation_transport not in", native)
        self.assertIn("transport is invalid", native)

    def test_bf16_transport_has_no_quantization(self):
        bf16 = SOURCE.split("class _BF16Dispatch", 1)[1].split(
            "\n\ndef _validate_fp8_dispatch", 1
        )[0]
        self.assertIn("dist.all_to_all_single", bf16)
        self.assertIn("dtype=torch.bfloat16", bf16)
        self.assertNotIn("quant", bf16.lower())
        self.assertNotIn("scale", bf16.lower())

    def test_bf16_combine_reuses_reserved_dispatch_storage_by_columns(self):
        function = SOURCE.split("def _streaming_bf16_combine", 1)[1].split(
            "\n\nclass _BF16Dispatch", 1
        )[0]
        dispatch = SOURCE.split("class _BF16Dispatch", 1)[1].split(
            "\n\nclass _INT4Dispatch", 1
        )[0]
        self.assertIn("result = routed_output", function)
        self.assertIn("result.as_strided_", function)
        self.assertIn("source[:, column_offset", function)
        self.assertIn("result[:, column_offset", function)
        self.assertIn("output_rows > storage_rows", function)
        self.assertEqual(function.count("_empty_with_cache_retry("), 2)
        self.assertIn("capacity_rows = max(routed_input.shape[0], output_rows)", dispatch)
        self.assertIn("ctx.mark_dirty(routed_input)", dispatch)
        self.assertIn("_BF16_COMBINE_ROUNDS", dispatch)

    def test_bf16_combine_rounds_fail_closed(self):
        prefix = SOURCE.split("def _in_backward", 1)[0]
        self.assertIn('os.environ.get("PRIME_BF16_COMBINE_ROUNDS", "8")', prefix)
        self.assertIn("not 1 <= _BF16_COMBINE_ROUNDS <= 64", prefix)
        self.assertIn("PRIME_BF16_COMBINE_ROUNDS must be between 1 and 64", prefix)

    def test_bf16_transport_memory_is_counted_in_bytes(self):
        function = SOURCE.split("def bf16_expert_transport_memory_bytes", 1)[1].split(
            "\n\ndef _streaming_int4_forward", 1
        )[0]
        self.assertIn('"dispatch_capacity_cuda"', function)
        self.assertIn('"saved_dispatch_input_cuda"', function)
        self.assertIn('"combine_scratch_cuda"', function)
        self.assertIn('"reconstruction_temporaries_cuda"', function)
        self.assertEqual(function.count("* element_bytes"), 3)
        self.assertIn("index_bytes + 2 * mask_bytes", function)

    def test_bf16_saved_input_is_only_kept_for_backward(self):
        saved_input = SOURCE.split("class ReceivedBF16ExpertInput", 1)[1].split(
            "\n\nclass BF16ExpertInputWorkspace", 1
        )[0]
        native = SOURCE.split(
            "class NativeFP8LoRAAllToAllExpertParallel", 1
        )[1].split("\n\nclass DeepEPExpertParallel", 1)[0]
        self.assertIn("if not self.save_for_backward", saved_input)
        self.assertIn("self.values = values.detach().clone()", saved_input)
        self.assertNotIn("pin_memory=True", saved_input)
        self.assertNotIn("synchronize()", saved_input)
        self.assertIn("row_indices.dtype != torch.int64", saved_input)
        self.assertIn("row_indices.device != self.values.device", saved_input)
        self.assertNotIn("device_indices", saved_input)
        self.assertIn("torch.index_select(self.values", saved_input)
        self.assertIn("save_for_backward = torch.is_grad_enabled()", native)
        self.assertIn("ReceivedBF16ExpertInput(save_for_backward)", native)

    def test_transport_types_declare_their_row_index_device(self):
        int4 = SOURCE.split("class ReceivedINT4ExpertInput", 1)[1].split(
            "\n\nclass INT4ExpertInputWorkspace", 1
        )[0]
        bf16 = SOURCE.split("class ReceivedBF16ExpertInput", 1)[1].split(
            "\n\nclass BF16ExpertInputWorkspace", 1
        )[0]
        self.assertIn('row_index_device = "cpu"', int4)
        self.assertIn('row_index_device = "cuda"', bf16)


if __name__ == "__main__":
    unittest.main()
