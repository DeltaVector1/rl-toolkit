import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "src"))

from prime_rl.trainer.fp8_checkpoint import (  # noqa: E402
    FP8CheckpointStorageReader,
    _FP8StorageInfo,
)


class _TensorSlice:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, slices):
        return self.tensor[slices]


class _CheckpointFile:
    def __init__(self, weight, scale):
        self.weight = weight
        self.scale = scale
        self.scale_reads = 0

    def get_slice(self, key):
        if not key.endswith(".weight"):
            raise AssertionError(f"unexpected weight key: {key}")
        return _TensorSlice(self.weight)

    def get_tensor(self, key):
        if not key.endswith(".weight_scale_inv"):
            raise AssertionError(f"unexpected scale key: {key}")
        self.scale_reads += 1
        return self.scale


class _Planner:
    def __init__(self, target):
        self.target = target
        self.committed = None

    def resolve_tensor(self, request):
        return self.target

    def commit_tensor(self, request, tensor):
        self.committed = tensor


def _request(shape, offsets=None):
    return SimpleNamespace(
        storage_offsets=torch.Size(offsets or (0,) * len(shape)),
        lengths=torch.Size(shape),
        storage_index="test tensor",
    )


def _process(weight, scale, target_dtype):
    target = torch.empty(weight.shape, dtype=target_dtype)
    planner = _Planner(target)
    file = _CheckpointFile(weight, scale)
    info = _FP8StorageInfo(
        relative_path="model.safetensors",
        shape=weight.shape,
        dtype=weight.dtype,
        source_key="model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        quantized=True,
    )
    FP8CheckpointStorageReader._process_request(
        file,
        _request(weight.shape),
        planner,
        info,
    )
    if planner.committed is None:
        raise AssertionError("the checkpoint planner did not commit the resolved tensor")
    if planner.committed.data_ptr() != target.data_ptr():
        raise AssertionError("the checkpoint planner committed a different tensor")
    return planner.committed, file


def _tensor_metadata(shape, dtype):
    shape = torch.Size(shape)
    return TensorStorageMetadata(
        properties=TensorProperties(dtype=dtype),
        size=shape,
        chunks=[
            ChunkStorageMetadata(
                offsets=torch.Size((0,) * len(shape)),
                sizes=shape,
            )
        ],
    )


def _metadata_with_quantized_pairs():
    pairs = {
        # Direct storage. The partial output block makes this an ordinary BF16
        # Linear target on the native-checkpoint path.
        "model.layers.0.self_attn.kv_a_proj_with_mqa": ((576, 6144), (5, 48)),
        # Direct storage that can remain native FP8.
        "model.layers.0.self_attn.q_b_proj": ((24576, 1536), (192, 12)),
        # Renamed shared-expert storage.
        "model.layers.0.mlp.shared_experts.gate_proj": ((3072, 6144), (24, 48)),
        # Stacked routed-expert storage.
        "model.layers.0.mlp.experts.0.gate_proj": ((3072, 6144), (24, 48)),
    }
    state_dict_metadata = {}
    storage_data = {}
    for prefix, (weight_shape, scale_shape) in pairs.items():
        for suffix, shape, dtype in (
            (".weight", weight_shape, torch.float8_e4m3fn),
            (".weight_scale_inv", scale_shape, torch.float32),
        ):
            key = prefix + suffix
            state_dict_metadata[key] = _tensor_metadata(shape, dtype)
            storage_data[
                MetadataIndex(key, torch.Size((0,) * len(shape)))
            ] = _HFStorageInfo(
                relative_path="model.safetensors",
                shape=torch.Size(shape),
                dtype=dtype,
            )
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        storage_data=storage_data,
    )


class FP8CheckpointStorageReaderTest(unittest.TestCase):
    def test_every_fp8_scale_pair_is_marked_quantized_after_metadata_rewrite(self):
        metadata = _metadata_with_quantized_pairs()
        reader = object.__new__(FP8CheckpointStorageReader)
        reader.dequantize_all = False

        with patch.object(
            HuggingFaceStorageReader,
            "read_metadata",
            return_value=metadata,
        ):
            rewritten = reader.read_metadata()

        weight_targets = {
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.q_b_proj.weight",
            "model.layers.0.mlp.shared_expert.w1",
            "model.layers.0.mlp.experts.w1",
        }
        seen = set()
        for index, info in rewritten.storage_data.items():
            if index.fqn not in weight_targets:
                continue
            seen.add(index.fqn)
            self.assertIsInstance(info, _FP8StorageInfo)
            self.assertTrue(info.quantized)

        self.assertEqual(seen, weight_targets)

    def test_fp8_scale_pair_is_kept_fp8_or_dequantized_for_target_dtype(self):
        raw = (
            torch.arange(128 * 128, dtype=torch.float32)
            .remainder(15)
            .sub(7)
            .reshape(128, 128)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.tensor([[0.25]], dtype=torch.float32)

        native, native_file = _process(raw, scale, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(native, raw))
        self.assertEqual(native_file.scale_reads, 0)

        dequantized, dequantized_file = _process(raw, scale, torch.bfloat16)
        reference = raw.to(torch.bfloat16) * scale.to(torch.bfloat16)
        torch.testing.assert_close(dequantized, reference, rtol=0, atol=0)
        self.assertEqual(dequantized_file.scale_reads, 1)

    def test_native_loaded_partial_block_kv_a_matches_fp8_scale_reference(self):
        rows, columns = 576, 6144
        raw = (
            torch.arange(rows * columns, dtype=torch.int32)
            .remainder(31)
            .sub(15)
            .reshape(rows, columns)
            .to(torch.float8_e4m3fn)
        )
        scale = (
            torch.arange(5 * 48, dtype=torch.float32).reshape(5, 48) + 1
        ) / 256

        loaded, file = _process(raw, scale, torch.bfloat16)
        expanded_scale = (
            scale.repeat_interleave(128, dim=0)[:rows]
            .repeat_interleave(128, dim=1)[:, :columns]
            .to(torch.bfloat16)
        )
        reference = raw.to(torch.bfloat16) * expanded_scale

        self.assertEqual(scale.shape, (5, 48))
        self.assertEqual(expanded_scale.shape, (576, 6144))
        torch.testing.assert_close(loaded, reference, rtol=0, atol=0)
        self.assertEqual(file.scale_reads, 1)


if __name__ == "__main__":
    unittest.main()
