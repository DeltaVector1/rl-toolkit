import ast
import copy
import dataclasses
import unittest
from collections.abc import Mapping
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


def _load_class(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)


def _load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method = copy.deepcopy(method)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)


@unittest.skipIf(torch is None, "Torch is required")
class FP8CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {"torch": torch, "Mapping": Mapping}
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
        from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
        from torch.distributed.checkpoint.metadata import MetadataIndex

        cls.namespace.update(
            {
                "dataclasses": dataclasses,
                "_HFStorageInfo": _HFStorageInfo,
                "MetadataIndex": MetadataIndex,
            }
        )
        _load_class(READER_PATH, "_FP8StorageInfo", cls.namespace)
        _load_method(READER_PATH, "FP8CheckpointStorageReader", "_move_storage", cls.namespace)

    def test_compressed_tensors_schema_is_exact(self):
        config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "quantization_status": "compressed",
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
        missing_status = copy.deepcopy(config)
        missing_status.pop("quantization_status")
        self.assertFalse(check(missing_status))
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
        validate(
            self._metadata(
                {
                    "legacy.weight": (torch.float8_e5m2, (256, 384)),
                    "legacy.weight_scale_inv": (torch.float32, (2, 3)),
                }
            )
        )
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
                "x.weight": (torch.float8_e5m2, (256, 384)),
                "x.weight_scale": (torch.bfloat16, (2, 3)),
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

    def test_shared_scale_rename_keeps_physical_source_and_storage_mapping(self):
        source = "model.layers.1.mlp.shared_experts.gate_proj.weight_scale"
        intermediate = "model.layers.1.mlp.shared_experts.gate_proj.weight_scale_inv"
        target = "model.layers.1.mlp.shared_expert.w1_scale_inv"
        offset = torch.Size((0, 0))
        source_index = self.namespace["MetadataIndex"](source, offset)
        metadata = SimpleNamespace(
            state_dict_metadata={source: SimpleNamespace(size=torch.Size((2, 3)))},
            storage_data={
                source_index: SimpleNamespace(
                    relative_path="model.safetensors",
                    shape=torch.Size((2, 3)),
                    dtype=torch.bfloat16,
                )
            },
        )
        storage_by_fqn = {source: [source_index]}
        move = self.namespace["_move_storage"]

        metadata.state_dict_metadata[intermediate] = metadata.state_dict_metadata.pop(source)
        move(metadata, storage_by_fqn, source, intermediate, offset)
        metadata.state_dict_metadata[target] = metadata.state_dict_metadata.pop(intermediate)
        move(metadata, storage_by_fqn, intermediate, target, offset)

        target_index = self.namespace["MetadataIndex"](target, offset)
        self.assertEqual(set(metadata.state_dict_metadata), {target})
        self.assertEqual(storage_by_fqn, {target: [target_index]})
        self.assertEqual(set(metadata.storage_data), {target_index})
        info = metadata.storage_data[target_index]
        self.assertIsInstance(info, self.namespace["_FP8StorageInfo"])
        self.assertEqual(info.source_key, source)

    def test_move_storage_keeps_multi_chunk_offsets_in_mapping(self):
        target = "model.layers.2.mlp.experts.w1_scale_inv"
        metadata = SimpleNamespace(storage_data={})
        storage_by_fqn = {}
        move = self.namespace["_move_storage"]
        for expert, offset in enumerate((torch.Size((0, 0)), torch.Size((1, 0)))):
            source = f"model.layers.2.mlp.experts.{expert}.gate_proj.weight_scale"
            source_index = self.namespace["MetadataIndex"](source, offset)
            metadata.storage_data[source_index] = SimpleNamespace(
                relative_path="model.safetensors",
                shape=torch.Size((1, 3)),
                dtype=torch.bfloat16,
            )
            storage_by_fqn[source] = [source_index]
            move(metadata, storage_by_fqn, source, target, offset, stacked=True)

        target_indices = [
            self.namespace["MetadataIndex"](target, torch.Size((0, 0))),
            self.namespace["MetadataIndex"](target, torch.Size((1, 0))),
        ]
        self.assertEqual(storage_by_fqn[target], target_indices)
        self.assertEqual(set(metadata.storage_data), set(target_indices))
        self.assertEqual(
            {metadata.storage_data[index].source_key for index in target_indices},
            {
                "model.layers.2.mlp.experts.0.gate_proj.weight_scale",
                "model.layers.2.mlp.experts.1.gate_proj.weight_scale",
            },
        )


if __name__ == "__main__":
    unittest.main()
