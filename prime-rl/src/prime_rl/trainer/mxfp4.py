import dataclasses
import re

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem
from torch.futures import Future

_EXPERT = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|down|up)_proj\.(weight|weight_scale)$"
)
_PROJECTIONS = {"gate": "w1", "down": "w2", "up": "w3"}


@triton.jit
def _dequant_mxfp4_kernel(weight, scale, out, packed_k: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    logical_k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = logical_k < packed_k * 2
    packed = tl.load(weight + row * packed_k + logical_k // 2, mask=mask)
    value = (packed >> ((logical_k & 1) * 4)) & 0xF
    code = (value & 0x7).to(tl.int32)
    magnitude = tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.exp2((code // 2 - 1).to(tl.float32)) * (1.0 + (code & 1).to(tl.float32) * 0.5),
        ),
    )
    magnitude = tl.where((value & 0x8) != 0, -magnitude, magnitude)
    exponent = tl.load(scale + row * (packed_k // 16) + logical_k // 32, mask=mask)
    block_scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    tl.store(out + row * packed_k * 2 + logical_k, magnitude * block_scale, mask=mask)


@torch.library.custom_op("prime_rl::dequant_mxfp4", mutates_args=())
def dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    packed_k = weight.shape[-1]
    out = torch.empty((*weight.shape[:-1], packed_k * 2), dtype=torch.bfloat16, device=weight.device)
    _dequant_mxfp4_kernel[(weight.numel() // packed_k, triton.cdiv(packed_k * 2, 256))](
        weight,
        scale,
        out,
        packed_k,
        256,
    )
    return out


@dequant_mxfp4.register_fake
def _(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((*weight.shape[:-1], weight.shape[-1] * 2), dtype=torch.bfloat16, device=weight.device)


@dataclasses.dataclass
class _MXFP4StorageInfo(_HFStorageInfo):
    source_key: str = ""
    stacked: bool = False


class MXFP4HuggingFaceStorageReader(HuggingFaceStorageReader):
    def read_metadata(self):
        metadata = super().read_metadata()
        storage_by_fqn = {}
        for index in metadata.storage_data:
            storage_by_fqn.setdefault(index.fqn, []).append(index)
        groups = {}
        for key in list(metadata.state_dict_metadata):
            match = _EXPERT.match(key)
            if match:
                prefix, expert, projection, kind = match.groups()
                groups.setdefault((prefix, projection, kind), []).append((int(expert), key))

        for (prefix, projection, kind), sources in groups.items():
            target = f"{prefix}.experts.{_PROJECTIONS[projection]}{'_scale' if kind == 'weight_scale' else ''}"
            chunks = []
            properties = None
            size = None
            for expert, source in sorted(sources):
                source_metadata = metadata.state_dict_metadata.pop(source)
                properties = source_metadata.properties
                source_shape = source_metadata.size
                size = torch.Size((len(sources), *source_shape))
                offset = torch.Size((expert, *([0] * len(source_shape))))
                chunks.append(ChunkStorageMetadata(offsets=offset, sizes=torch.Size((1, *source_shape))))
                self._move_storage(metadata, storage_by_fqn, source, target, offset, stacked=True)
            metadata.state_dict_metadata[target] = TensorStorageMetadata(
                properties=properties,
                size=size,
                chunks=chunks,
            )

        renames = {
            ".mlp.gate.weight": ".mlp.router.gate.weight",
            ".mlp.gate.e_score_correction_bias": ".mlp.expert_bias",
            ".mlp.shared_experts.gate_proj.weight": ".mlp.shared_expert.w1",
            ".mlp.shared_experts.gate_proj.weight_scale": ".mlp.shared_expert.w1_scale",
            ".mlp.shared_experts.down_proj.weight": ".mlp.shared_expert.w2",
            ".mlp.shared_experts.down_proj.weight_scale": ".mlp.shared_expert.w2_scale",
            ".mlp.shared_experts.up_proj.weight": ".mlp.shared_expert.w3",
            ".mlp.shared_experts.up_proj.weight_scale": ".mlp.shared_expert.w3_scale",
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
                    )
                    break
        return metadata

    @staticmethod
    def _move_storage(metadata, storage_by_fqn, source, target, target_offset, stacked=False):
        for index in storage_by_fqn.pop(source, ()):
            info = metadata.storage_data.pop(index)
            metadata.storage_data[MetadataIndex(target, target_offset)] = _MXFP4StorageInfo(
                relative_path=info.relative_path,
                shape=info.shape,
                dtype=info.dtype,
                source_key=source,
                stacked=stacked,
            )

    @staticmethod
    def _process_mxfp4_request(f, req: ReadItem, planner: LoadPlanner, info: _MXFP4StorageInfo):
        slices = tuple(slice(offset, offset + length) for offset, length in zip(req.storage_offsets, req.lengths))
        if info.stacked:
            tensor = f.get_slice(info.source_key)[slices[1:]].unsqueeze(0)
        else:
            tensor = f.get_slice(info.source_key)[slices]
        target = planner.resolve_tensor(req).detach()
        if target.size() != tensor.size():
            raise AssertionError(f"req {req.storage_index} mismatch sizes {target.size()} vs {tensor.size()}")
        target.copy_(tensor)
        planner.commit_tensor(req, target)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file = {}
        for item in plan.items:
            info = self.storage_data[item.storage_index]
            per_file.setdefault(info.relative_path, []).append((item, info))

        for file_name, requests in per_file.items():
            with safe_open(file_name, framework="pt") as f:
                for request, info in requests:
                    if isinstance(info, _MXFP4StorageInfo):
                        self._process_mxfp4_request(f, request, planner, info)
                    else:
                        self._process_read_request(f, request, planner)

        future = Future()
        future.set_result(None)
        return future
