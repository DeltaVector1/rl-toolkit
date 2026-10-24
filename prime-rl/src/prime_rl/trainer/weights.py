import json
import warnings
from pathlib import Path
from typing import Literal, cast

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from prime_rl.trainer.lora import (
    clean_lora_state_dict,
)
from prime_rl.utils.logger import get_logger


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
    if adapter:
        weights_name = ADAPTER_SAFE_WEIGHTS_NAME if save_format == "safetensors" else ADAPTER_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            save_index_file = save_dir / save_index_file
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


# ── pipeline parallelism context ─────────────────────────────────────────────
# Set once by train.py when pp > 1. Under PP each rank's model holds only its
# stage's layers, so the master's local state_dict is a fraction of the model;
# saving it as the adapter would ship a truncated LoRA to the inference engines
# with no error anywhere. The fix: each stage's first rank gathers its stage,
# then the stage dicts are merged on the global master over the pp group.
_PP_GROUP = None
_PP_CPU_GROUP = None
_PP_STAGE_MASTER_RANKS: tuple[int, ...] = ()
_IS_STAGE_MASTER = False


def set_pp_gather_context(
    pp_group,
    is_stage_master: bool,
    cpu_group=None,
    stage_master_ranks: list[int] | None = None,
) -> None:
    global _PP_GROUP, _PP_CPU_GROUP, _PP_STAGE_MASTER_RANKS, _IS_STAGE_MASTER
    _PP_GROUP = pp_group
    _PP_CPU_GROUP = cpu_group
    _PP_STAGE_MASTER_RANKS = tuple(stage_master_ranks or ())
    _IS_STAGE_MASTER = is_stage_master


def is_pp_stage_master() -> bool:
    return _IS_STAGE_MASTER


def merge_pp_state_dict(state_dict: dict, is_master: bool) -> dict:
    """Merge per-stage shards of an adapter into one dict on the global master.

    Under pipeline parallelism each rank's model owns only its stage's layers,
    and DTensor.full_tensor() reassembles across FSDP/EP but NOT across stages.
    The master would otherwise save only its own stage -- a truncated adapter
    that loads without complaint and serves a partly-unadapted model.

    Every stage master contributes its CPU tensors; layer names carry global
    indices, so the union is exactly the full adapter.
    """
    if _PP_GROUP is None:
        return state_dict
    if not _IS_STAGE_MASTER:
        return {}
    if not _PP_STAGE_MASTER_RANKS:
        raise RuntimeError("pipeline adapter export has no stage leaders")

    global _PP_CPU_GROUP
    if _PP_CPU_GROUP is None:
        _PP_CPU_GROUP = torch.distributed.new_group(
            ranks=list(_PP_STAGE_MASTER_RANKS),
            backend="gloo",
            use_local_synchronization=True,
        )

    rank = torch.distributed.get_rank()
    master = _PP_STAGE_MASTER_RANKS[0]
    metadata = [
        (key, tuple(value.shape), str(value.dtype), value.numel() * value.element_size())
        for key, value in state_dict.items()
    ]
    gathered_metadata = [None] * len(_PP_STAGE_MASTER_RANKS) if rank == master else None
    torch.distributed.gather_object(
        metadata,
        object_gather_list=gathered_metadata,
        dst=master,
        group=_PP_CPU_GROUP,
    )

    if rank == master:
        merged = dict(state_dict)
        duplicates: list[str] = []
        assert gathered_metadata is not None
        for source, part in zip(_PP_STAGE_MASTER_RANKS, gathered_metadata, strict=True):
            if source == master:
                continue
            assert part is not None
            for key, shape, dtype_name, byte_count in part:
                raw = torch.empty(byte_count, dtype=torch.uint8)
                torch.distributed.recv(raw, src=source, group=_PP_CPU_GROUP)
                dtype = getattr(torch, dtype_name.removeprefix("torch."))
                value = raw.view(dtype).reshape(shape)
                if key in merged:
                    duplicates.append(key)
                merged[key] = value
    else:
        for value in state_dict.values():
            raw = value.contiguous().view(torch.uint8).reshape(-1)
            torch.distributed.send(raw, dst=master, group=_PP_CPU_GROUP)
        merged = {}

    torch.distributed.barrier(group=_PP_CPU_GROUP)
    if rank == master and duplicates:
        raise RuntimeError(
            "pipeline adapter export found duplicate keys: "
            + ", ".join(duplicates[:10])
        )
    return merged


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Gather distributed weights on CPU on master rank."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        # Under PP every stage master materialises its stage; otherwise only
        # the global master does.
        build_local = is_master or (_PP_GROUP is not None and _IS_STAGE_MASTER)
        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if build_local:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)

        if _PP_GROUP is not None:
            # Gather over the WORLD group, not the pp subgroup. all_gather_object
            # sizes its receive buffers from the participating ranks, and mixing a
            # mesh-derived subgroup with global rank bookkeeping produced garbage
            # lengths -- a 3796 GiB allocation attempt inside recv_object_list.
            # World-group gather needs no rank translation: every rank sends its
            # stage dict (or {}), and layer names already carry global indices, so
            # the union is exactly the full model.
            world = torch.distributed.get_world_size()
            gathered: list[dict | None] = [None] * world
            torch.distributed.all_gather_object(gathered, cpu_state if _IS_STAGE_MASTER else {})
            if is_master:
                merged: dict[str, Tensor] = {}
                for part in gathered:
                    if part:
                        merged.update(part)
                cpu_state = merged
            else:
                cpu_state = {}
        torch.distributed.barrier()

    # Always clean up the state dict for HF compatibility
    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
        cpu_state = clean_lora_state_dict(cpu_state)

    return cpu_state
