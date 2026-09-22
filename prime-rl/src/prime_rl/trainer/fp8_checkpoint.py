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
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|down|up)_proj\.(weight|weight_scale|weight_scale_inv)$"
)
_PROJECTIONS = {"gate": "w1", "down": "w2", "up": "w3"}
_GLM_QKV_A = re.compile(
    r"^(model\.layers\.\d+\.self_attn)\.(q_a_proj|kv_a_proj_with_mqa)\."
    r"(weight|weight_scale_inv)$"
)


@dataclasses.dataclass
class _FP8StorageInfo(_HFStorageInfo):
    source_key: str = ""
    scale_source_key: str = ""
    stacked: bool = False
    quantized: bool = False
    dequantize: bool = False
    bitcast: bool = False


def _scale_source_for_weight(weight_key: str, source_keys: set[str]) -> str | None:
    for suffix in (".weight_scale_inv", ".weight_scale"):
        scale_key = weight_key.removesuffix(".weight") + suffix
        if scale_key in source_keys:
            return scale_key
    return None


def _canonical_scale_key(source_key: str) -> str:
    return source_key.removesuffix(".weight_scale") + ".weight_scale_inv"


def _validate_fp8_scale_metadata(metadata) -> None:
    state = metadata.state_dict_metadata
    keys = set(state)
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dtype is not None
    }
    for weight in (key for key in keys if key.endswith(".weight")):
        scale_keys = [
            key
            for key in (
                weight.removesuffix(".weight") + ".weight_scale",
                weight.removesuffix(".weight") + ".weight_scale_inv",
            )
            if key in keys
        ]
        if len(scale_keys) > 1:
            raise ValueError(f"checkpoint has duplicate FP8 scales for {weight}")
        if state[weight].properties.dtype not in fp8_dtypes:
            continue
        if not scale_keys:
            raise ValueError(f"FP8 weight has no scale tensor: {weight}")
        scale = scale_keys[0]
        scale_meta = state[scale]
        if scale_meta.properties.dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError(f"FP8 scale has unsupported dtype: {scale}")
        if scale.endswith(".weight_scale"):
            if state[weight].properties.dtype != torch.float8_e4m3fn:
                raise TypeError(f"compressed-tensors FP8 weight must use float8_e4m3fn: {weight}")
            if scale_meta.properties.dtype != torch.bfloat16:
                raise TypeError(f"compressed-tensors FP8 scale must be bfloat16: {scale}")
        if len(weight_meta := state[weight].size) != 2 or len(scale_meta.size) != 2:
            raise ValueError(f"FP8 weight and scale must be rank 2: {weight}")
        rows, cols = map(int, weight_meta)
        expected = ((rows + 127) // 128, (cols + 127) // 128)
        if tuple(map(int, scale_meta.size)) != expected:
            raise ValueError(f"FP8 scale shape does not match {weight}: expected {expected}")
        if scale.endswith(".weight_scale") and (rows % 128 or cols % 128):
            raise ValueError(f"compressed-tensors FP8 weight has a partial block: {weight}")

    for scale in (key for key in keys if key.endswith((".weight_scale", ".weight_scale_inv"))):
        weight = scale.rsplit(".weight_scale", 1)[0] + ".weight"
        if weight not in keys:
            raise ValueError(f"FP8 scale has no weight tensor: {scale}")


class FP8CheckpointStorageReader(HuggingFaceStorageReader):
    def __init__(
        self,
        *args,
        dequantize_all: bool = False,
        fuse_glm_qkv_a: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dequantize_all = dequantize_all
        self.fuse_glm_qkv_a = fuse_glm_qkv_a

    def _fuse_glm_qkv_a_metadata(self, metadata, storage_by_fqn) -> None:
        groups: dict[tuple[str, str], dict[str, str]] = {}
        for key in list(metadata.state_dict_metadata):
            match = _GLM_QKV_A.match(key)
            if match is None:
                continue
            prefix, projection, kind = match.groups()
            groups.setdefault((prefix, kind), {})[projection] = key

        expected_groups = sum(
            1
            for key in metadata.state_dict_metadata
            if key.endswith(".self_attn.q_a_proj.weight")
        ) * 2
        if len(groups) != expected_groups:
            raise ValueError(
                f"GLM fused Q-A and KV-A found {len(groups)} weight groups; "
                f"expected {expected_groups}"
            )

        for (prefix, kind), sources in groups.items():
            if set(sources) != {"q_a_proj", "kv_a_proj_with_mqa"}:
                raise ValueError(
                    f"GLM fused Q-A and KV-A checkpoint is incomplete at {prefix}"
                )
            q_source = sources["q_a_proj"]
            kv_source = sources["kv_a_proj_with_mqa"]
            q_metadata = metadata.state_dict_metadata.pop(q_source)
            kv_metadata = metadata.state_dict_metadata.pop(kv_source)
            if len(q_metadata.size) != 2 or len(kv_metadata.size) != 2:
                raise ValueError(
                    f"GLM fused Q-A and KV-A tensors must be matrices at {prefix}"
                )
            if q_metadata.size[1] != kv_metadata.size[1]:
                raise ValueError(f"GLM fused Q-A and KV-A input sizes differ at {prefix}")
            if kind == "weight":
                expected_q, expected_kv, expected_input = 2048, 576, 6144
                expected_dtype = torch.float8_e4m3fn
            else:
                expected_q, expected_kv, expected_input = 16, 5, 48
                expected_dtype = torch.float32
            if (
                q_metadata.properties.dtype != expected_dtype
                or kv_metadata.properties.dtype != expected_dtype
            ):
                raise TypeError(
                    f"GLM fused Q-A and KV-A {kind} must use {expected_dtype} at "
                    f"{prefix}; got q={q_metadata.properties.dtype}, "
                    f"kv={kv_metadata.properties.dtype}"
                )
            if tuple(q_metadata.size) != (expected_q, expected_input) or tuple(
                kv_metadata.size
            ) != (expected_kv, expected_input):
                raise ValueError(
                    f"GLM fused Q-A and KV-A {kind} shapes differ at {prefix}: "
                    f"q={tuple(q_metadata.size)}, kv={tuple(kv_metadata.size)}"
                )

            target = f"{prefix}.q_a_proj.{kind}"
            chunks = [
                ChunkStorageMetadata(
                    offsets=torch.Size((0, 0)),
                    sizes=q_metadata.size,
                ),
                ChunkStorageMetadata(
                    offsets=torch.Size((expected_q, 0)),
                    sizes=kv_metadata.size,
                ),
            ]
            metadata.state_dict_metadata[target] = TensorStorageMetadata(
                properties=q_metadata.properties,
                size=torch.Size((expected_q + expected_kv, expected_input)),
                chunks=chunks,
            )
            self._move_storage(
                metadata,
                storage_by_fqn,
                q_source,
                target,
                torch.Size((0, 0)),
                quantized=kind == "weight",
            )
            self._move_storage(
                metadata,
                storage_by_fqn,
                kv_source,
                target,
                torch.Size((expected_q, 0)),
                quantized=kind == "weight",
            )

    def read_metadata(self):
        metadata = super().read_metadata()
        _validate_fp8_scale_metadata(metadata)
        storage_by_fqn = {}
        for index in metadata.storage_data:
            storage_by_fqn.setdefault(index.fqn, []).append(index)
        source_keys = set(storage_by_fqn)
        scale_source_by_weight = {
            key: scale_source
            for key in source_keys
            if key.endswith(".weight")
            if (scale_source := _scale_source_for_weight(key, source_keys)) is not None
        }
        quantized_weights = set(scale_source_by_weight)

        groups = {}
        for key in list(metadata.state_dict_metadata):
            if match := _EXPERT.match(key):
                prefix, expert, projection, kind = match.groups()
                groups.setdefault((prefix, projection, kind), []).append((int(expert), key))

        for (prefix, projection, kind), sources in groups.items():
            suffix = "_scale_inv" if kind in {"weight_scale", "weight_scale_inv"} else ""
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
                    quantized=source in quantized_weights,
                    dequantize=self.dequantize_all and source in quantized_weights,
                    scale_source_key=scale_source_by_weight.get(source),
                )
            metadata.state_dict_metadata[target] = TensorStorageMetadata(
                properties=properties,
                size=size,
                chunks=chunks,
            )

        for source in list(metadata.state_dict_metadata):
            if not source.endswith(".weight_scale"):
                continue
            target = _canonical_scale_key(source)
            if target in metadata.state_dict_metadata:
                metadata.state_dict_metadata.pop(source)
                for index in storage_by_fqn.pop(source, ()):
                    metadata.storage_data.pop(index, None)
                continue
            metadata.state_dict_metadata[target] = metadata.state_dict_metadata.pop(source)
            self._move_storage(
                metadata,
                storage_by_fqn,
                source,
                target,
                torch.Size([0] * len(metadata.state_dict_metadata[target].size)),
            )

        renames = {
            ".mlp.gate.weight": ".mlp.router.gate.weight",
            ".mlp.gate.e_score_correction_bias": ".mlp.expert_bias",
            ".mlp.shared_experts.gate_proj.weight": ".mlp.shared_expert.w1",
            ".mlp.shared_experts.gate_proj.weight_scale": ".mlp.shared_expert.w1_scale_inv",
            ".mlp.shared_experts.gate_proj.weight_scale_inv": ".mlp.shared_expert.w1_scale_inv",
            ".mlp.shared_experts.down_proj.weight": ".mlp.shared_expert.w2",
            ".mlp.shared_experts.down_proj.weight_scale": ".mlp.shared_expert.w2_scale_inv",
            ".mlp.shared_experts.down_proj.weight_scale_inv": ".mlp.shared_expert.w2_scale_inv",
            ".mlp.shared_experts.up_proj.weight": ".mlp.shared_expert.w3",
            ".mlp.shared_experts.up_proj.weight_scale": ".mlp.shared_expert.w3_scale_inv",
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
                        quantized=source in quantized_weights,
                        dequantize=self.dequantize_all and source in quantized_weights,
                        scale_source_key=scale_source_by_weight.get(source),
                    )
                    break

        for source in metadata.state_dict_metadata:
            quantized = source in quantized_weights
            dequantize = (
                self.dequantize_all and quantized
            ) or source.endswith((".self_attn.kv_b_proj.weight", ".self_attn.indexer.wk.weight"))
            bitcast = False
            if quantized or dequantize or bitcast:
                for index in storage_by_fqn[source]:
                    info = metadata.storage_data[index]
                    metadata.storage_data[index] = _FP8StorageInfo(
                        relative_path=info.relative_path,
                        shape=info.shape,
                        dtype=info.dtype,
                        source_key=source,
                        scale_source_key=scale_source_by_weight.get(source, ""),
                        quantized=quantized,
                        dequantize=dequantize,
                        bitcast=bitcast,
                    )
        if getattr(self, "fuse_glm_qkv_a", False):
            self._fuse_glm_qkv_a_metadata(metadata, storage_by_fqn)
        return metadata

    @staticmethod
    def _move_storage(
        metadata,
        storage_by_fqn,
        source,
        target,
        target_offset,
        stacked=False,
        quantized=False,
        dequantize=False,
        scale_source_key=None,
    ):
        for index in storage_by_fqn.pop(source, ()):
            info = metadata.storage_data.pop(index)
            metadata.storage_data[MetadataIndex(target, target_offset)] = _FP8StorageInfo(
                relative_path=info.relative_path,
                shape=info.shape,
                dtype=info.dtype,
                source_key=source,
                scale_source_key=scale_source_key or "",
                stacked=stacked,
                quantized=quantized,
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
        # Some serialized FP8 matrices cannot use the native 128x128 kernel
        # because one dimension has a partial final block. Those modules stay
        # as ordinary BF16 Linear layers. DCP otherwise casts their raw E4M3
        # values to BF16 without applying weight_scale_inv, corrupting the
        # model by orders of magnitude (GLM kv_a is [576, 6144]).
        dequantize = info.dequantize or (info.quantized and target.dtype != tensor.dtype)
        if dequantize:
            scale_key = info.scale_source_key or info.source_key.removesuffix(".weight") + ".weight_scale_inv"
            scale = file.get_tensor(scale_key)
            row_slice = slices[1] if info.stacked else slices[0]
            col_slice = slices[2] if info.stacked else slices[1]
            rows = torch.arange(row_slice.start, row_slice.stop) // 128
            cols = torch.arange(col_slice.start, col_slice.stop) // 128
            block_scale = scale[rows][:, cols]
            if info.stacked:
                block_scale = block_scale.unsqueeze(0)
            # Apply the checkpoint's FP32 inverse scales before the final
            # target-dtype rounding.  Casting the scales to BF16 first causes
            # a second, avoidable rounding and does not match vLLM's
            # scaled_dequantize path for fused MLA/indexer weights.
            tensor = (tensor.float() * block_scale.float()).to(target.dtype)
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
