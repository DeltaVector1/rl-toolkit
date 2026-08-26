"""Static gates for opt-in GLM-5.2 kernel candidates.

These tests do not import CUDA or vLLM.  The production-shaped benchmark must
run on an idle RTX 4090 after the candidate is compiled in isolation.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FP8_SOURCE = (ROOT / "src/prime_rl/trainer/models/layers/fp8_ada.py").read_text()
SPARSE_BWD_SOURCE = (
    ROOT / "src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py"
).read_text()


class GLM52KernelCandidateTest(unittest.TestCase):
    def test_candidates_are_disabled_by_default(self):
        self.assertIn('"PRIME_FP8_EXPERT_TRITON_SCATTER", "0"', FP8_SOURCE)
        self.assertIn('"PRIME_FP8_LORA_W2_LOW_RANK_REUSE", "0"', FP8_SOURCE)
        self.assertIn('"PRIME_FP8_SCATTERMOE_LORA", "0"', FP8_SOURCE)
        self.assertIn(
            "_SCATTERMOE_LORA_BWD_VALUE = os.environ.get(",
            FP8_SOURCE,
        )
        self.assertIn(
            'os.environ.get("PRIME_SPARSE_BWD_VARIANT", "baseline")',
            SPARSE_BWD_SOURCE,
        )

    def test_scatter_candidate_validates_permuted_rows(self):
        helper = FP8_SOURCE.split(
            "def _index_copy_real_rows_with_cache_retry", 1
        )[1].split("\n\ndef _grouped_mm_pair_chunks", 1)[0]
        self.assertIn("_expert_scatter_kernel", helper)
        self.assertIn("indices.dtype != torch.int64", helper)
        self.assertIn("indices.shape != (source.shape[0],)", helper)
        self.assertIn("destination.device.type != \"cuda\"", helper)
        self.assertIn("return destination", helper)

    def test_scattermoe_lora_candidate_keeps_native_base_path(self):
        self.assertIn("def _scattermoe_lora_forward_kernel", FP8_SOURCE)
        self.assertIn("INPUT_SIZE: tl.constexpr", FP8_SOURCE)
        self.assertIn("Accumulation is FP32", FP8_SOURCE)
        self.assertIn("low_rank_bf16 = low_rank.to(tl.bfloat16)", FP8_SOURCE)
        self.assertIn(
            "low_rank_bf16[:, None] * b_values.to(tl.float32), axis=0",
            FP8_SOURCE,
        )
        self.assertIn("_scattermoe_lora_projection", FP8_SOURCE)
        self.assertIn("_SCATTERMOE_LORA", FP8_SOURCE)
        self.assertIn(
            "ScatterMoE LoRA row indices must be INT32 or INT64",
            FP8_SOURCE,
        )
        self.assertIn(
            "ScatterMoE LoRA tensors must share one CUDA device",
            FP8_SOURCE,
        )
        self.assertIn("_grouped_mm_pair(", FP8_SOURCE)
        self.assertIn("ScatterMoE LoRA candidate requires rank 32", FP8_SOURCE)
        self.assertIn(
            "for n_start in tl.range(0, OUTPUT_SIZE, BLOCK_N)",
            FP8_SOURCE,
        )
        self.assertIn(
            "_scattermoe_lora_forward_kernel[(row_indices.shape[0],)]",
            FP8_SOURCE,
        )
        scatter_kernel = FP8_SOURCE.split(
            "def _scattermoe_lora_forward_kernel", 1
        )[1].split("def _expert_scatter_kernel", 1)[0]
        self.assertNotIn("tl.program_id(1)", scatter_kernel)
        self.assertIn("source_valid & valid_columns", scatter_kernel)
        self.assertIn("offsets + (MAX_EXPERTS - 1)", scatter_kernel)
        self.assertIn("mask=expert_index > 0", scatter_kernel)

    def test_scattermoe_lora_backward_is_fp32_and_bounded(self):
        self.assertIn("def _scattermoe_outer_accumulate_kernel", FP8_SOURCE)
        self.assertIn("def _scattermoe_outer_accumulate(", FP8_SOURCE)
        self.assertIn("_scattermoe_lora_backward", FP8_SOURCE)
        self.assertIn("dtype=torch.float32 if _SCATTERMOE_LORA_BWD", FP8_SOURCE)
        self.assertIn("value.to(parameter.dtype)", FP8_SOURCE)
        self.assertIn("BLOCK_ROWS=32", FP8_SOURCE)
        self.assertIn("output.dtype != torch.float32", FP8_SOURCE)
        self.assertIn(
            "ScatterMoE LoRA backward tensors must share one device",
            FP8_SOURCE,
        )
        backward = FP8_SOURCE.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("\n\nclass _ScaleRoutedOutput", 1)[0]
        self.assertIn("_scattermoe_lora_backward(", backward)
        self.assertIn("backward_row_indices = (", backward)
        self.assertIn("torch.arange(", backward)
        outer_kernel = FP8_SOURCE.split(
            "def _scattermoe_outer_accumulate_kernel", 1
        )[1].split("else:\n    _scattermoe_lora_forward_kernel", 1)[0]
        self.assertIn("row_mask[:, None]", outer_kernel)
        self.assertIn("source_valid[:, None]", outer_kernel)

    def test_low_rank_reuse_is_host_saved_and_backward_optional(self):
        forward = FP8_SOURCE.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("    @staticmethod\n    def backward", 1)[0]
        backward = FP8_SOURCE.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("    @staticmethod\n    def backward", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0]
        self.assertIn("reuse_w2_low_rank", forward)
        self.assertIn("w2_low_rank_host", forward)
        self.assertIn("pin_memory=True", forward)
        self.assertIn("low_rank=saved_w2_low_rank", backward)
        self.assertIn("if low_rank is None", FP8_SOURCE)

    def test_lora_backward_scales_intermediates_in_place(self):
        backward = FP8_SOURCE.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("\n\nclass _ScaleRoutedOutput", 1)[0]
        self.assertIn("scaling=1.0", FP8_SOURCE)
        self.assertIn("low_rank_grad.mul_(scaling)", FP8_SOURCE)
        self.assertIn("low_rank.mul_(scaling)", FP8_SOURCE)
        self.assertNotIn("grad_chunk.mul(ctx.scaling)", backward)
        self.assertNotIn("z1_grad.mul(ctx.scaling)", backward)
        self.assertNotIn("z3_grad.mul(ctx.scaling)", backward)
        self.assertIn("scaling=ctx.scaling", backward)
        self.assertIn("if _SCATTERMOE_LORA_BWD", backward)

    def test_lora_forward_releases_gathered_input_before_w2(self):
        forward = FP8_SOURCE.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("    @staticmethod\n    def backward", 1)[0]
        self.assertIn("del chunk_x, safe_indices", forward)
        self.assertLess(
            forward.index("del chunk_x, safe_indices"),
            forward.index("chunk_out = _grouped_mm_chunked"),
        )

    def test_routed_lora_phase_profiler_is_opt_in_and_aggregated(self):
        self.assertIn(
            'os.environ.get("PRIME_EXPERT_PHASE_PROFILE", "0")',
            FP8_SOURCE,
        )
        self.assertIn(
            'os.environ.get("PRIME_EXPERT_PHASE_PROFILE_EVERY", "180")',
            FP8_SOURCE,
        )
        self.assertIn("def _expert_phase_start", FP8_SOURCE)
        self.assertIn("def _expert_phase_end", FP8_SOURCE)
        self.assertIn('"forward.w13_base"', FP8_SOURCE)
        self.assertIn('"forward.w2_lora"', FP8_SOURCE)
        self.assertIn('"backward.w2_base"', FP8_SOURCE)
        self.assertIn('"backward.w1_lora"', FP8_SOURCE)
        self.assertIn('"backward.w3_lora"', FP8_SOURCE)
        self.assertIn('"backward.reconstruct"', FP8_SOURCE)
        self.assertIn('_profile_expert_phases_tick("forward")', FP8_SOURCE)
        self.assertIn('_profile_expert_phases_tick("backward")', FP8_SOURCE)
        self.assertIn("end.synchronize()", FP8_SOURCE)
        self.assertIn("[fp8-expert-phase-prof]", FP8_SOURCE)

    def test_sparse_bh32_candidate_does_not_change_forward(self):
        self.assertIn('variant not in {"baseline", "bh32"}', SPARSE_BWD_SOURCE)
        self.assertIn('variant == "bh32"', SPARSE_BWD_SOURCE)
        self.assertEqual(SPARSE_BWD_SOURCE.count("PRIME_SPARSE_BWD_VARIANT"), 2)

    def test_sparse_dkv_is_always_fp32_and_bf16_candidate_is_absent(self):
        self.assertNotIn("PRIME_SPARSE_BWD_DKV", SPARSE_BWD_SOURCE)
        self.assertIn("dKV: T.Tensor(k_shape, accum_dtype)", SPARSE_BWD_SOURCE)
        self.assertIn("dkv = torch.zeros_like(kv, dtype=torch.float32)", SPARSE_BWD_SOURCE)
        self.assertIn("dkv = postprocess(D, D_tail, kv_group)(dkv)", SPARSE_BWD_SOURCE)
        self.assertNotIn("dkv_dtype", SPARSE_BWD_SOURCE)
        self.assertNotIn("T.bfloat16 if", SPARSE_BWD_SOURCE)
        self.assertIn(
            "the dKV dtype",
            SPARSE_BWD_SOURCE,
        )
        self.assertNotIn("float8", SPARSE_BWD_SOURCE)

    def test_sparse_fp32_split_stage_is_opt_in(self):
        self.assertIn(
            'os.environ.get("PRIME_SPARSE_BWD_FP32_SPLIT_STORE", "2")',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn("split_store_value not in {\"2\", \"4\"}", SPARSE_BWD_SOURCE)
        self.assertIn("split_store=split_store", SPARSE_BWD_SOURCE)
        self.assertIn("low-memory path computes each split directly", SPARSE_BWD_SOURCE)

    def test_sparse_low_memory_dkv_is_opt_in_and_split_sized(self):
        self.assertIn(
            'os.environ.get(\n        "PRIME_SPARSE_BWD_LOW_MEMORY_DKV", "0"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            '"PRIME_SPARSE_BWD_LOW_MEMORY_DKV must be 0 or 1"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            '"PRIME_SPARSE_BWD_LOW_MEMORY_DKV requires split_store=4"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn("if low_memory_dkv:", SPARSE_BWD_SOURCE)
        self.assertIn(
            '"low-memory dKV requires block_size >= 32"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            '"low-memory dKV requires block_size % 16 == 0"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            '"low-memory dKV requires block_H % 16 == 0"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            '"low-memory dKV requires D and D_tail to be multiples of 16"',
            SPARSE_BWD_SOURCE,
        )
        self.assertIn(
            "((dkv_split_rows + 15) // 16) * 16",
            SPARSE_BWD_SOURCE,
        )
        self.assertIn("safe_source_column", SPARSE_BWD_SOURCE)
        self.assertIn("P_dkv_shared", SPARSE_BWD_SOURCE)
        self.assertIn("dP_dkv_shared", SPARSE_BWD_SOURCE)
        self.assertIn("[dkv_gemm_rows, D]", SPARSE_BWD_SOURCE)

    def test_sparse_low_memory_padding_covers_only_real_rows(self):
        for block_size in (32, 48, 64, 80):
            split_rows = block_size // 4
            gemm_rows = max(16, ((split_rows + 15) // 16) * 16)
            self.assertEqual(gemm_rows % 16, 0)
            for split in range(4):
                first = split * split_rows
                self.assertLess(first + split_rows - 1, block_size)
                self.assertGreaterEqual(gemm_rows, split_rows)

    def test_w2_chunk_candidate_has_bounded_values(self):
        self.assertIn(
            'os.environ.get("PRIME_FP8_W2_CHUNK_ROWS", "2048")',
            FP8_SOURCE,
        )
        self.assertIn(
            '_FP8_W2_CHUNK_ROWS_VALUE not in {"2048", "4096"}',
            FP8_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
