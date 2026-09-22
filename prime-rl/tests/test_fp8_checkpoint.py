import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the trainer image provides Torch
    torch = None


MODEL_PATH = Path(__file__).parents[1] / "src/prime_rl/trainer/model.py"
READER_PATH = Path(__file__).parents[1] / "src/prime_rl/trainer/fp8_checkpoint.py"


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@unittest.skipIf(torch is None, "Torch is required")
class FP8CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {"torch": torch}
        _load_functions(
            MODEL_PATH,
            {"_config_mapping", "_config_value", "_is_compressed_tensors_fp8_block"},
            cls.namespace,
        )
        _load_functions(
            READER_PATH,
            {
                "_scale_source_for_weight",
                "_canonical_scale_key",
                "_validate_fp8_scale_metadata",
            },
            cls.namespace,
        )

    def test_compressed_tensors_schema_is_exact(self):
        config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "block",
                        "block_structure": [128, 128],
                        "dynamic": False,
                        "symmetric": True,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "group",
                        "dynamic": True,
                        "group_size": 128,
                        "symmetric": True,
                    },
                    "output_activations": None,
                }
            },
        }
        check = self.namespace["_is_compressed_tensors_fp8_block"]
        self.assertTrue(check(config))
        invalid = copy.deepcopy(config)
        invalid["config_groups"]["group_0"]["weights"]["block_structure"] = [64, 128]
        self.assertFalse(check(invalid))

    def test_scale_aliases_cover_dense_shared_and_expert_names(self):
        source_keys = {
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.down_proj.weight_scale",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            "model.layers.1.mlp.shared_experts.up_proj.weight_scale",
            "model.layers.2.mlp.experts.3.gate_proj.weight",
            "model.layers.2.mlp.experts.3.gate_proj.weight_scale",
        }
        get_scale = self.namespace["_scale_source_for_weight"]
        self.assertEqual(
            get_scale("model.layers.0.mlp.down_proj.weight", source_keys),
            "model.layers.0.mlp.down_proj.weight_scale",
        )
        self.assertEqual(
            get_scale("model.layers.1.mlp.shared_experts.up_proj.weight", source_keys),
            "model.layers.1.mlp.shared_experts.up_proj.weight_scale",
        )
        self.assertEqual(
            get_scale("model.layers.2.mlp.experts.3.gate_proj.weight", source_keys),
            "model.layers.2.mlp.experts.3.gate_proj.weight_scale",
        )
        self.assertEqual(
            self.namespace["_canonical_scale_key"]("x.weight_scale"),
            "x.weight_scale_inv",
        )

    def _metadata(self, entries):
        return SimpleNamespace(
            state_dict_metadata={
                key: SimpleNamespace(
                    properties=SimpleNamespace(dtype=dtype),
                    size=torch.Size(size),
                )
                for key, (dtype, size) in entries.items()
            }
        )

    def test_reader_rejects_invalid_block_scale_metadata(self):
        validate = self.namespace["_validate_fp8_scale_metadata"]
        valid = self._metadata(
            {
                "x.weight": (torch.float8_e4m3fn, (256, 384)),
                "x.weight_scale": (torch.bfloat16, (2, 3)),
            }
        )
        validate(valid)
        cases = [
            {
                "x.weight": (torch.float8_e4m3fn, (256, 384)),
            },
            {
                "x.weight": (torch.float8_e4m3fn, (256, 384)),
                "x.weight_scale": (torch.bfloat16, (1, 3)),
            },
            {
                "x.weight": (torch.float8_e4m3fn, (256, 384)),
                "x.weight_scale": (torch.float32, (2, 3)),
            },
            {
                "x.weight": (torch.float8_e4m3fn, (257, 384)),
                "x.weight_scale": (torch.bfloat16, (3, 3)),
            },
            {
                "x.weight": (torch.float8_e4m3fn, (256, 384)),
                "x.weight_scale": (torch.bfloat16, (2, 3)),
                "x.weight_scale_inv": (torch.float32, (2, 3)),
            },
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                with self.assertRaises((TypeError, ValueError)):
                    validate(self._metadata(entries))

        with self.assertRaises(ValueError):
            validate(self._metadata({"x.weight_scale": (torch.bfloat16, (2, 3))}))


if __name__ == "__main__":
    unittest.main()
