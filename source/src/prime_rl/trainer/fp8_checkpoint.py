import dataclasses
import re

import torch
from safetensors import safe_open
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem
from torch.futures import Future


_EXPERT = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|down|up)_proj\.(weight|weight_scale_inv)$"
)
_PROJECTIONS = {"gate": "w1", "down": "w2", "up": "w3"}


@dataclasses.dataclass
class _FP8StorageInfo(_HFStorageInfo):
    source_key: str = ""
    stacked: bool = False
    dequantize: bool = False
    bitcast: bool = False


class FP8CheckpointStorageReader(HuggingFaceStorageReader):
    def __init__(self, *args, dequantize_all: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dequantize_all = dequantize_all

    def read_metadata(self):
        metadata = super().read_metadata()
        storage_by_fqn = {}
        for index in metadata.storage_data:
            storage_by_fqn.setdefault(index.fqn, []).append(index)
        quantized_weights = {
            key
            for key in storage_by_fqn
            if key.endswith(".weight") and key.removesuffix(".weight") + ".weight_scale_inv" in storage_by_fqn
        }

        groups = {}
        for key in list(metadata.state_dict_metadata):
            if match := _EXPERT.match(key):
                prefix, expert, projection, kind = match.groups()
                groups.setdefault((prefix, projection, kind), []).append((int(expert), key))

        for (prefix, projection, kind), sources in groups.items():
            suffix = "_scale_inv" if kind == "weight_scale_inv" else ""
            target = f"{prefix}.experts.{_PROJECTIONS[projection]}{suffix}"
            chunks = []
            properties = None
            size = None
            for expert, source in sorted(sources):
                source_metadata = metadata.state_dict_metadata.pop(source)
                properties = source_metadata.properties
                source_shape = source_metadata.size
                size = torch.Size((len(sources), *source_shape))
                offset = torch.Size((expert, *([0] * len(source_shape))))
                chunks.append(
                    ChunkStorageMetadata(
                        offsets=offset,
                        sizes=torch.Size((1, *source_shape)),
                    )
                )
                self._move_storage(
                    metadata,
                    storage_by_fqn,
                    source,
                    target,
                    offset,
                    stacked=True,
                    dequantize=self.dequantize_all and source in quantized_weights,
                )
            metadata.state_dict_metadata[target] = TensorStorageMetadata(
                properties=properties,
                size=size,
                chunks=chunks,
            )

        renames = {
            ".mlp.gate.weight": ".mlp.router.gate.weight",
            ".mlp.gate.e_score_correction_bias": ".mlp.expert_bias",
            ".mlp.shared_experts.gate_proj.weight": ".mlp.shared_expert.w1",
            ".mlp.shared_experts.gate_proj.weight_scale_inv": ".mlp.shared_expert.w1_scale_inv",
            ".mlp.shared_experts.down_proj.weight": ".mlp.shared_expert.w2",
            ".mlp.shared_experts.down_proj.weight_scale_inv": ".mlp.shared_expert.w2_scale_inv",
            ".mlp.shared_experts.up_proj.weight": ".mlp.shared_expert.w3",
            ".mlp.shared_experts.up_proj.weight_scale_inv": ".mlp.shared_expert.w3_scale_inv",
        }
        for source in list(metadata.state_dict_metadata):
            for suffix, replacement in renames.items():
                if source.endswith(suffix):
                    target = source.removesuffix(suffix) + replacement
                    metadata.state_dict_metadata[target] = metadata.state_dict_metadata.pop(source)
                    self._move_storage(
                        metadata,
                        storage_by_fqn,
                        source,
                        target,
                        torch.Size([0] * len(metadata.state_dict_metadata[target].size)),
                        dequantize=self.dequantize_all and source in quantized_weights,
                    )
                    break

        for source in metadata.state_dict_metadata:
            dequantize = (
                self.dequantize_all and source in quantized_weights
            ) or source.endswith((".self_attn.kv_b_proj.weight", ".self_attn.indexer.wk.weight"))
            bitcast = False
            if dequantize or bitcast:
                for index in storage_by_fqn[source]:
                    info = metadata.storage_data[index]
                    metadata.storage_data[index] = _FP8StorageInfo(
                        relative_path=info.relative_path,
                        shape=info.shape,
                        dtype=info.dtype,
                        source_key=source,
                        dequantize=dequantize,
                        bitcast=bitcast,
                    )
        return metadata

    @staticmethod
    def _move_storage(
        metadata,
        storage_by_fqn,
        source,
        target,
        target_offset,
        stacked=False,
        dequantize=False,
    ):
        for index in storage_by_fqn.pop(source, ()):
            info = metadata.storage_data.pop(index)
            metadata.storage_data[MetadataIndex(target, target_offset)] = _FP8StorageInfo(
                relative_path=info.relative_path,
                shape=info.shape,
                dtype=info.dtype,
                source_key=source,
                stacked=stacked,
                dequantize=dequantize,
            )

    @staticmethod
    def _process_request(file, request: ReadItem, planner: LoadPlanner, info: _FP8StorageInfo):
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(request.storage_offsets, request.lengths)
        )
        tensor = file.get_slice(info.source_key)[slices[1:] if info.stacked else slices]
        target = planner.resolve_tensor(request).detach()
        if info.stacked:
            tensor = tensor.unsqueeze(0)
        if info.dequantize:
            scale = file.get_tensor(
                info.source_key.removesuffix(".weight") + ".weight_scale_inv"
            )
            row_slice = slices[1] if info.stacked else slices[0]
            col_slice = slices[2] if info.stacked else slices[1]
            rows = torch.arange(row_slice.start, row_slice.stop) // 128
            cols = torch.arange(col_slice.start, col_slice.stop) // 128
            block_scale = scale[rows][:, cols].to(target.dtype)
            if info.stacked:
                block_scale = block_scale.unsqueeze(0)
            tensor = tensor.to(target.dtype) * block_scale
        elif info.bitcast:
            tensor = tensor.view(torch.uint8)
        if target.size() != tensor.size():
            raise AssertionError(
                f"{request.storage_index} mismatch sizes {target.size()} vs {tensor.size()}"
            )
        target.copy_(tensor)
        planner.commit_tensor(request, target)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file = {}
        for item in plan.items:
            info = self.storage_data[item.storage_index]
            per_file.setdefault(info.relative_path, []).append((item, info))

        for file_name, requests in per_file.items():
            with safe_open(file_name, framework="pt") as file:
                for request, info in requests:
                    if isinstance(info, _FP8StorageInfo):
                        self._process_request(file, request, planner, info)
                    else:
                        self._process_read_request(file, request, planner)

        future = Future()
        future.set_result(None)
        return future
