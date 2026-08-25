"""Regression checks for bounded native-FP8 expert backward storage."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/trainer/models/layers/fp8_ada.py"
).read_text()


class FP8ExpertBackwardMemoryTest(unittest.TestCase):
    def test_native_lora_discarded_forward_does_not_save_intermediates(self):
        forward = SOURCE.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "    @staticmethod\n    def backward", 1
        )[0]
        branch = forward.index("if save_intermediates:")
        quantize = forward.index("ops.scaled_fp8_quant", branch)
        skip = forward.index("if not save_backward_state:", quantize)
        self.assertLess(branch, quantize)
        self.assertLess(quantize, skip)
        self.assertNotIn('.to("cpu")', forward)
        self.assertIn('if row_index_device == "cuda":', forward)
        self.assertIn(
            "saved_permuted_indices = permuted_indices.to(torch.int64)", forward
        )
        self.assertIn('elif row_index_device == "cpu":', forward)
        self.assertIn("pin_memory=True", forward)
        self.assertNotIn("permuted_indices_host", forward)

    def test_native_lora_backward_only_waits_on_cpu_for_int4_indices(self):
        backward = SOURCE.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0].split("    @staticmethod\n    def backward", 1)[1]
        cpu_branch = backward.split('if ctx.row_index_device == "cpu":', 1)[1].split(
            'elif ctx.row_index_device != "cuda":', 1
        )[0]
        self.assertIn("ctx.offload_done.synchronize()", cpu_branch)
        self.assertEqual(backward.count("ctx.offload_done.synchronize()"), 1)
        self.assertNotIn("permuted_indices_host", backward)
        self.assertEqual(backward.count("ctx.received_input.reconstruct_rows("), 2)

    def test_native_lora_reacquires_current_parameter_storage(self):
        section = SOURCE.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0]
        forward, backward = section.split(
            "    @staticmethod\n    def backward", 1
        )
        saved = forward.split("ctx.save_for_backward(", 1)[1].split(
            "        )\n", 1
        )[0]
        self.assertNotIn("w1_lora_a", saved)
        self.assertNotIn("w3_lora_b", saved)
        self.assertIn("ctx.lora_parameter_metadata", forward)
        self.assertIn(
            'getattr(owner, f"{projection}_lora_{factor}")[0]', backward
        )
        self.assertIn("value.to_local() if isinstance(value, DTensor)", backward)
        self.assertIn("*parameter_grads,", backward)

    def test_native_lora_backward_has_no_nested_autograd(self):
        native_lora = SOURCE.split("def _grouped_lora_backward", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0]
        self.assertNotIn("torch.autograd.grad", native_lora)
        self.assertEqual(native_lora.count("_grouped_lora_backward("), 3)
        self.assertEqual(native_lora.count("_grouped_outer_accumulate("), 2)

    def test_grouped_output_allocation_has_one_cache_recovery_retry(self):
        dispatch = SOURCE.split("def _dispatch_grouped_mm", 1)[1].split(
            "\n\n@torch.compiler.disable", 1
        )[0]
        self.assertEqual(dispatch.count("except torch.OutOfMemoryError:"), 1)
        self.assertEqual(dispatch.count("torch.cuda.empty_cache()"), 1)
        self.assertEqual(dispatch.count("out = torch.empty("), 2)

    def test_inplace_ep_forward_releases_w13_temporaries_before_w2(self):
        forward = SOURCE.split("class _InplacePermutedExperts", 1)[1].split(
            "    @staticmethod\n    def backward", 1
        )[0]
        release_input = forward.index("del chunk_x, safe_indices")
        release_quantized = forward.index("del q1, q3", release_input)
        release_w3 = forward.index("del z3", release_quantized)
        w2 = forward.index("chunk_out = _grouped_mm_chunked", release_w3)
        release_chunk = forward.index(
            "del z1, chunk_out, indices, real, chunk_offsets", w2
        )
        self.assertLess(release_input, w2)
        self.assertLess(release_quantized, w2)
        self.assertLess(release_w3, w2)
        self.assertGreater(release_chunk, w2)

    def test_inplace_ep_scatter_uses_bounded_torch_path_with_one_retry(self):
        helper = SOURCE.split(
            "def _index_copy_real_rows_with_cache_retry", 1
        )[1].split("\n\ndef _grouped_mm_pair_chunks", 1)[0]
        self.assertEqual(helper.count("except torch.OutOfMemoryError:"), 1)
        self.assertEqual(helper.count("torch.cuda.empty_cache()"), 1)
        self.assertEqual(helper.count("destination.index_copy_("), 2)
        self.assertNotIn("_scatter_permuted_rows_kernel", SOURCE)

        inplace = SOURCE.split("class _InplacePermutedExperts", 1)[1].split(
            "\n\nclass _InplacePermutedLoRAExperts", 1
        )[0]
        self.assertEqual(
            inplace.count("_index_copy_real_rows_with_cache_retry("), 2
        )

    def test_forward_expert_workspace_is_bounded_to_4096_rows(self):
        self.assertIn("_FP8_W13_CHUNK_ROWS = 4096", SOURCE)
        self.assertIn("_FP8_W2_CHUNK_ROWS = 2048", SOURCE)

    def test_expert_row_gather_has_one_cache_recovery_retry(self):
        helper = SOURCE.split(
            "def _index_select_rows_with_cache_retry", 1
        )[1].split("\ndef _index_copy_real_rows_with_cache_retry", 1)[0]
        self.assertEqual(helper.count("except torch.OutOfMemoryError:"), 1)
        self.assertEqual(helper.count("torch.cuda.empty_cache()"), 1)
        self.assertEqual(helper.count("return x.index_select(0, indices)"), 2)

        inplace = SOURCE.split("class _InplacePermutedExperts", 1)[1].split(
            "\n\nclass _InplacePermutedLoRAExperts", 1
        )[0]
        self.assertEqual(
            inplace.count("_index_select_rows_with_cache_retry("), 2
        )

    def test_gate_gradient_reuses_backward_temporaries(self):
        helper = SOURCE.split("def _silu_glu_backward_inplace", 1)[1].split(
            "\n\nclass _Experts", 1
        )[0]
        self.assertIn("z3_grad = F.silu(z1)", helper)
        self.assertIn("hidden_grad.mul_(z3).mul_(sigmoid)", helper)
        self.assertIn("hidden_grad.mul_(z1)", helper)
        self.assertNotIn("hidden_grad * z3 * sigmoid", helper)

    def test_inplace_ep_backward_is_bounded_to_4096_rows(self):
        self.assertIn("_FP8_EXPERT_BACKWARD_CHUNK_ROWS = 4096", SOURCE)
        self.assertIn(
            "ctx.backward_chunk_rows = min(chunk_rows, _FP8_EXPERT_BACKWARD_CHUNK_ROWS)",
            SOURCE,
        )

    def test_each_backward_chunk_is_released_before_the_next(self):
        backward = SOURCE.split("class _InplacePermutedExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0]
        release = backward.index("del (\n                indices,")
        loop_end = backward.index("if _EXPERT_PROFILE:", release)
        self.assertLess(release, loop_end)
        self.assertIn("x_grad_w3,", backward[release:loop_end])

    def test_native_lora_releases_second_base_gradient_before_reconstruction(self):
        backward = SOURCE.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0].split("    @staticmethod\n    def backward", 1)[1]
        combine = backward.index("x_grad.add_(x_grad_w3)")
        release = backward.index("del x_grad_w3", combine)
        reconstruct = backward.index("ctx.received_input.reconstruct_rows", release)
        self.assertLess(combine, release)
        self.assertLess(release, reconstruct)
        self.assertNotIn("x_grad_w3,", backward[reconstruct:])


if __name__ == "__main__":
    unittest.main()
