"""Regression checks for bounded native-FP8 expert backward storage."""

from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/trainer/models/layers/fp8_ada.py"
).read_text()


class FP8ExpertBackwardMemoryTest(unittest.TestCase):
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

    def test_each_backward_chunk_is_released_before_the_next(self):
        backward = SOURCE.split("class _InplacePermutedExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0]
        release = backward.index("del (\n                indices,")
        loop_end = backward.index("if _EXPERT_PROFILE:", release)
        self.assertLess(release, loop_end)
        self.assertIn("x_grad_w3,", backward[release:loop_end])


if __name__ == "__main__":
    unittest.main()
