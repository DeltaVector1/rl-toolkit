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
PP_RUNNER_SOURCE = (
    Path(__file__).parents[1]
    / "src"
    / "prime_rl"
    / "trainer"
    / "distributed"
    / "pp_runner.py"
).read_text()


def load_transport_selector():
    tree = ast.parse(SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_native_fp8_expert_transport"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace["_resolve_native_fp8_expert_transport"]


class FP8LocalRecomputeTests(unittest.TestCase):
    def test_local_recompute_requires_an_explicit_config(self):
        self.assertIn('] = "bf16"', CONFIG_SOURCE)
        self.assertIn(
            '"bf16_forward_fp8_local_recompute_backward"',
            CONFIG_SOURCE,
        )
        self.assertIn("def validate_local_expert_recompute", CONFIG_SOURCE)
        self.assertIn(
            "cutoff >= self.pp",
            CONFIG_SOURCE,
        )
        self.assertIn('self.ac.mode != "reentrant_attention_all_experts"', CONFIG_SOURCE)
        self.assertIn('self.ep_comm_backend != "torch"', CONFIG_SOURCE)
        self.assertIn('self.pp_schedule == "interleaved_1f1b"', CONFIG_SOURCE)

    def test_cutoff_five_selects_only_late_pipeline_stages(self):
        select = load_transport_selector()
        requested = "bf16_forward_fp8_local_recompute_backward"
        for stage in range(5):
            self.assertEqual(
                select(requested, stage, 8, 5),
                ("bf16_forward_fp8_recompute_backward", False),
            )
        for stage in range(5, 8):
            self.assertEqual(
                select(requested, stage, 8, 5),
                (requested, True),
            )

    def test_pipeline_stage_selection_fails_closed(self):
        select = load_transport_selector()
        local = "bf16_forward_fp8_local_recompute_backward"
        with self.assertRaisesRegex(TypeError, "must be integers"):
            select(local, None, 8, 5)
        with self.assertRaisesRegex(ValueError, "stage is invalid"):
            select(local, 8, 8, 5)
        with self.assertRaisesRegex(ValueError, "cutoff is invalid"):
            select(local, 7, 8, 8)
        with self.assertRaisesRegex(ValueError, "only valid"):
            select("bf16", 7, 8, 5)

    def test_early_stages_keep_checkpoint_recompute_transport(self):
        select = load_transport_selector()
        effective, active = select(
            "bf16_forward_fp8_local_recompute_backward",
            4,
            8,
            5,
        )
        self.assertFalse(active)
        native = SOURCE.split(
            "class NativeFP8LoRAAllToAllExpertParallel", 1
        )[1].split("\n\nclass DeepEPExpertParallel", 1)[0]
        recompute = native.split("recompute_fp8 =", 1)[1].split(
            "if recompute_fp8:", 1
        )[0]
        self.assertEqual(effective, "bf16_forward_fp8_recompute_backward")
        self.assertIn('== "bf16_forward_fp8_recompute_backward"', recompute)

    def test_interleaved_schedule_guards_private_api_and_ac_frequency(self):
        self.assertIn("def _require_pinned_interleaved_schedule_api", PP_RUNNER_SOURCE)
        self.assertIn('requires pinned PyTorch 2.11.0', PP_RUNNER_SOURCE)
        self.assertIn('self.pp_schedule == "interleaved_1f1b"', CONFIG_SOURCE)
        self.assertIn('self.ac.freq != 1', CONFIG_SOURCE)

    def test_model_passes_the_physical_stage_to_transport_selection(self):
        model = (
            Path(__file__).parents[1]
            / "src"
            / "prime_rl"
            / "trainer"
            / "model.py"
        ).read_text()
        apply_ep = model.split("def apply_ep", 1)[1].split(
            "def configure_trainable_parameters", 1
        )[0]
        self.assertIn('parallel_dims.get_mesh(\n                                "pp"', apply_ep)
        self.assertIn('"pp_size": parallel_dims.pp', apply_ep)
        self.assertIn('"local_recompute_cutoff": cutoff', apply_ep)

    def test_local_recompute_retains_only_compact_tensor_state(self):
        expert = (
            Path(__file__).parents[1]
            / "src"
            / "prime_rl"
            / "trainer"
            / "models"
            / "layers"
            / "fp8_ada.py"
        ).read_text()
        forward = expert.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "    @staticmethod\n    def backward", 1
        )[0]
        local_save = forward.rsplit("if local_recompute:", 1)[1].split(
            "else:", 1
        )[0]
        self.assertIn("saved_permuted_indices", local_save)
        self.assertIn("received_scores", local_save)
        self.assertNotIn("q1_host", local_save)
        self.assertNotIn("q3_host", local_save)

        backward = expert.split("class _InplacePermutedLoRAExperts", 1)[1].split(
            "\n\nclass _ScaleRoutedOutput", 1
        )[0].split("    @staticmethod\n    def backward", 1)[1]
        self.assertEqual(backward.count("ctx.received_input.reconstruct_rows("), 2)
        self.assertNotIn("dist.all_to_all", backward)

    def test_local_recompute_scales_before_expert_combine(self):
        moe = (
            Path(__file__).parents[1]
            / "src"
            / "prime_rl"
            / "trainer"
            / "models"
            / "layers"
            / "moe.py"
        ).read_text()
        routed = moe.split("def _run_routed_experts", 1)[1].split(
            "def _run_deepep_routed_experts", 1
        )[0]
        self.assertIn("top_scores_experts_sorted", routed)
        self.assertIn("if not self.score_before_experts and not local_recompute", routed)

        native = SOURCE.split(
            "class NativeFP8LoRAAllToAllExpertParallel", 1
        )[1].split("\n\nclass DeepEPExpertParallel", 1)[0]
        self.assertIn("_RouteScoreDispatch.apply", native)
        self.assertIn("route scores have the wrong shape", native)

    def test_local_recompute_matches_checkpoint_rounding_order(self):
        expert = (
            Path(__file__).parents[1]
            / "src"
            / "prime_rl"
            / "trainer"
            / "models"
            / "layers"
            / "fp8_ada.py"
        ).read_text()
        backward = expert.split(
            "class _InplacePermutedLoRAExperts", 1
        )[1].split("\n\nclass _ScaleRoutedOutput", 1)[0].split(
            "    @staticmethod\n    def backward", 1
        )[1]
        local = backward.split("if ctx.local_recompute:", 2)[2].split(
            "            else:\n                q1_device", 1
        )[0]
        score = local.index("grad_chunk.float() * chunk_scores[:, None]")
        grad_quant = local.index("ops.scaled_fp8_quant(\n                    grad_chunk", score)
        self.assertLess(score, grad_quant)
        self.assertIn("ops.scaled_fp8_quant(\n                    chunk_out", local)
        self.assertIn("ops.scaled_fp8_quant(\n                    z1", local)
        self.assertIn("ops.scaled_fp8_quant(\n                    z3", local)

        native = SOURCE.split(
            "class NativeFP8LoRAAllToAllExpertParallel", 1
        )[1].split("\n\nclass DeepEPExpertParallel", 1)[0]
        combine = native.split("def _token_combine", 1)[1].split(
            "    @staticmethod", 1
        )[0]
        codec = combine.split("backward_transport =", 1)[1].split(
            "return _BF16Dispatch.apply", 1
        )[0]
        self.assertNotIn(
            '"bf16_forward_fp8_local_recompute_backward"',
            codec,
        )

    def test_bf16_dispatch_quantizes_only_after_the_collective(self):
        dispatch = SOURCE.split("class _BF16Dispatch", 1)[1].split(
            "\n\ndef _validate_fp8_dispatch", 1
        )[0]
        collective = dispatch.index("dist.all_to_all_single")
        save = dispatch.index("received_input.prepare_from_bf16(output)")
        self.assertLess(collective, save)
        self.assertIn("return output", dispatch[save:])

    def test_saved_input_lifecycle_does_not_keep_discarded_forward_state(self):
        received = SOURCE.split("class ReceivedFP8ExpertInput", 1)[1].split(
            "\n\nclass FP8ExpertInputWorkspace", 1
        )[0]
        self.assertIn("save_for_backward", received)
        self.assertIn("if not self.save_for_backward:", received)
        self.assertIn("if self.prepared:", received)
        self.assertIn("self.values = values", received)
        self.assertNotIn("device=\"cpu\"", received)
        self.assertNotIn("pin_memory=True", received)

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "FP8 saved-input equivalence requires CUDA",
)
class FP8LocalRecomputeCudaTests(unittest.TestCase):
    def test_route_scaling_commutes_with_row_permutation(self):
        values = torch.randn(97, 6144, device="cuda", dtype=torch.bfloat16)
        scores = torch.rand(97, device="cuda", dtype=torch.float32)
        permutation = torch.randperm(values.shape[0], device=values.device)

        scale_then_permute = (
            values.float() * scores[:, None]
        ).to(values.dtype)[permutation]
        permute_then_scale = (
            values[permutation].float()
            * scores[permutation, None]
        ).to(values.dtype)

        self.assertTrue(torch.equal(scale_then_permute, permute_then_scale))

    def test_quantization_commutes_with_row_permutation(self):
        from prime_rl.trainer.distributed.expert_parallel import (
            ReceivedFP8ExpertInput,
        )
        from vllm import _custom_ops as ops

        values = torch.randn(
            97, 6144, device="cuda", dtype=torch.bfloat16
        ).clamp_(-8, 8)
        original = values.clone()
        permutation = torch.randperm(values.shape[0], device=values.device)
        quantized, scales = ops.scaled_fp8_quant(
            values, use_per_token_if_dynamic=True
        )

        received = ReceivedFP8ExpertInput()
        received.prepare_from_bf16(values[permutation].contiguous())

        self.assertTrue(torch.equal(values, original))
        self.assertTrue(torch.equal(received.values, quantized[permutation]))
        self.assertTrue(
            torch.equal(received.scales, scales.flatten()[permutation])
        )
        with self.assertRaisesRegex(RuntimeError, "prepared more than once"):
            received.prepare_from_bf16(values)


if __name__ == "__main__":
    unittest.main()
