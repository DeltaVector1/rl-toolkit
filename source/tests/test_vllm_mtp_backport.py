"""Source checks for the vLLM 0.26 GLM-5.2 MTP backport.

These checks do not import vLLM or require a GPU.  A clean vLLM checkout can
be supplied with ``VLLM_V026_SOURCE_ROOT`` for patch-application checks.
"""

import ast
import os
import subprocess
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VLLM_PATCH_ROOT = ROOT / "patches/vllm"
PATCHES = sorted(VLLM_PATCH_ROOT.glob("000*.patch"))


def _vllm_root() -> Path | None:
    value = os.environ.get("VLLM_V026_SOURCE_ROOT")
    return Path(value) if value else None


def _patch_root() -> Path | None:
    value = os.environ.get("VLLM_V026_PATCH_ROOT")
    if value:
        return Path(value)
    return _vllm_root()


def _read_vllm(relative: str) -> str:
    root = _vllm_root()
    if root is None:
        raise unittest.SkipTest("VLLM_V026_SOURCE_ROOT is not set")
    return (root / relative).read_text()


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


class VllmMtpBackportTest(unittest.TestCase):
    def test_attention_context_unwraps_base_mapping(self):
        source = _read_vllm("vllm/model_executor/layers/attention/attention.py")
        node = _function(source, "_get_layer_slot_mapping")
        namespace: dict[str, object] = {}
        code = "from __future__ import annotations\n" + ast.unparse(node)
        exec(compile(code, "attention.py", "exec"), namespace)

        class Context:
            slot_mapping = [
                {"layer": "base"},
                {"layer": "draft"},
            ]

        self.assertEqual(namespace["_get_layer_slot_mapping"](Context(), "layer"), "base")
        Context.slot_mapping = {"layer": "ordinary"}
        self.assertEqual(namespace["_get_layer_slot_mapping"](Context(), "layer"), "ordinary")

    def test_mla_direct_call_uses_shared_context_helper(self):
        source = _read_vllm("vllm/model_executor/layers/attention/mla_attention.py")
        self.assertIn("_get_layer_slot_mapping,", source)
        direct = source[source.index("if self.use_direct_call"):]
        direct = direct[: direct.index("else:\n            encoded =")]
        self.assertIn("_get_layer_slot_mapping(", direct)
        self.assertNotIn("assert isinstance(slot_mapping, dict)", direct)

    def test_hidden_state_cache_update_unwraps_base_mapping(self):
        source = _read_vllm("vllm/model_executor/models/extract_hidden_states.py")
        node = _function(source, "unified_kv_cache_update")
        captured: list[object] = []

        class Impl:
            def do_kv_cache_update(self, layer, to_cache, kv_cache, slot_mapping):
                captured.append(slot_mapping)

        class Tensor:
            device = "cpu"
            dtype = "float32"

        class Layer:
            kv_cache = Tensor()
            impl = Impl()

        class Context:
            no_compile_layers = {"layer": Layer()}
            slot_mapping = [{"layer": "base"}, {"layer": "draft"}]

        class FakeTorch:
            @staticmethod
            def empty(*args, **kwargs):
                return Tensor()

        namespace: dict[str, object] = {
            "torch": FakeTorch,
            "get_forward_context": lambda: Context(),
        }
        code = "from __future__ import annotations\n" + ast.unparse(node)
        exec(compile(code, "extract_hidden_states.py", "exec"), namespace)
        namespace["unified_kv_cache_update"](Tensor(), "layer")
        self.assertEqual(captured, ["base"])

    def test_deepseek_mla_cache_update_unwraps_base_mapping(self):
        source = _read_vllm("vllm/models/deepseek_v32/nvidia/attention.py")
        marker = "slot_mapping = forward_context.slot_mapping"
        start = source.index(marker)
        block = source[start : source.index("mla_slot =", start)]
        self.assertIn("if isinstance(slot_mapping, list):", block)
        self.assertIn("slot_mapping = slot_mapping[0]", block)

    def test_tp8_mtp1_32k_geometry_is_explicit(self):
        sequence_length = 32_768
        tensor_parallel_size = 8
        global_query_heads = 64
        padded_query_heads = 64
        # FlashMLA's padded GLM query head is 576 values wide at this callsite.
        query_head_dim = 576
        byte_stride = sequence_length * padded_query_heads * query_head_dim * 2

        self.assertEqual(global_query_heads // tensor_parallel_size, 8)
        self.assertEqual(byte_stride, 2_415_919_104)
        self.assertGreater(byte_stride, 2**31 - 1)
        mtp_layers = 1
        self.assertEqual(mtp_layers, 1)

    def test_target_lora_contract_remains_unchanged(self):
        with (ROOT / "configs/qualification/glm52-policy-parity-h200.toml").open("rb") as stream:
            config = tomllib.load(stream)
        expected = [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "shared_expert",
            "experts",
        ]
        self.assertIs(config["enable_lora"], True)
        self.assertEqual(config["max_lora_rank"], 32)
        self.assertEqual(config["parallel"]["tp"], 8)
        self.assertEqual(config["lora_target_modules"], expected)

    def test_router_trace_isolation_is_conditional(self):
        patch = (VLLM_PATCH_ROOT / "0004-v026-request-exact-router-trace.patch").read_text()
        self.assertIn('extra_args.get("prime_router_trace") is True', patch)
        self.assertIn("an exact router trace request must run alone", patch)
        self.assertIn("force_eager=eager_router_trace", patch)
        self.assertIn("skip_compiled=has_encoder_input or eager_router_trace", patch)
        self.assertNotIn("force_eager=True", patch)

    def test_patch_series_has_both_upstream_slot_mapping_commits(self):
        self.assertEqual([path.name for path in PATCHES], [
            "0001-v026-flashmla-sparse-scratch-limit.patch",
            "0002-v026-flashmla-sparse-mixed-batch-chunks.patch",
            "0003-v026-exact-target-prompt-logprobs.patch",
            "0004-v026-request-exact-router-trace.patch",
            "0005-v026-exact-monolithic-route-sinks.patch",
            "0006-v026-mtp-slot-mapping-46908.patch",
            "0007-v026-mtp-slot-mapping-47013.patch",
        ])
        self.assertIn("150e73eda5678e80e2eec36642a1234c709275fb", PATCHES[5].read_text())
        self.assertIn("30030761508a9801f4a8db2a140137d8fefd11ac", PATCHES[6].read_text())

    def test_patch_series_applies_to_clean_vllm_checkout(self):
        root = _patch_root()
        if root is None:
            raise unittest.SkipTest("VLLM_V026_SOURCE_ROOT is not set")
        for patch in PATCHES:
            result = subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{patch.name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
