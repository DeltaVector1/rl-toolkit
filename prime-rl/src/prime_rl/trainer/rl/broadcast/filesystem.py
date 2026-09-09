import json
import re
import shutil
import time
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.adapter_delta import publish_adapter_manifest
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _published_adapter_identity(step_path: Path) -> dict[str, object]:
    if (step_path / "STABLE").is_symlink() or not (step_path / "STABLE").is_file():
        raise RuntimeError(f"adapter step is not STABLE: {step_path}")
    manifest_path = step_path / "adapter_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"adapter manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"adapter manifest is invalid: {manifest_path}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        raise RuntimeError(f"adapter manifest has no files: {manifest_path}")
    if len(files) != 2 or any(not isinstance(item, dict) for item in files):
        raise RuntimeError(f"adapter manifest has an invalid file set: {manifest_path}")
    records = {}
    for item in files:
        name = item.get("name")
        if name in records or name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError(f"adapter manifest has an invalid file set: {manifest_path}")
        records[name] = item
    if set(records) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise RuntimeError(f"adapter manifest has an invalid file set: {manifest_path}")
    identity = {}
    for name, prefix in (("adapter_model.safetensors", "target"), ("adapter_config.json", "config")):
        record = records[name]
        size = record.get("size")
        digest = record.get("sha256")
        path = step_path / name
        if (
            type(size) is not int
            or size < 0
            or not _valid_digest(digest)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise RuntimeError(f"published adapter file identity is invalid: {path}")
        identity[f"{prefix}_size"] = size
        identity[f"{prefix}_sha256"] = digest
    sidecar = step_path / "adapter_model.safetensors.sha256"
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != identity["target_sha256"]:
        raise RuntimeError(f"published adapter checksum sidecar is invalid: {sidecar}")
    return identity


def _receiver_ids(config) -> list[str]:
    client = getattr(getattr(config, "model", None), "client", None)
    urls = getattr(client, "admin_base_url", None) or getattr(client, "base_url", None)
    if not isinstance(urls, (list, tuple)) or not urls:
        raise RuntimeError("receiver ACK has no configured inference endpoints")
    endpoint_ids = [str(url).rstrip("/").removesuffix("/v1") for url in urls]
    if any(not endpoint for endpoint in endpoint_ids) or len(set(endpoint_ids)) != len(endpoint_ids):
        raise RuntimeError("receiver ACK has duplicate or empty configured endpoints")
    return sorted(endpoint_ids)


def _receiver_ack_identity(config, step_path: Path, step: int, run_name: str) -> dict[str, object]:
    published = _published_adapter_identity(step_path)
    broadcast = config.weight_broadcast
    source_url = getattr(broadcast, "source_url", None)
    source_dir = getattr(broadcast, "source_dir", None)
    if not isinstance(source_url, str) or not source_url or source_dir is None:
        raise RuntimeError("receiver ACK requires configured adapter source URL and directory")
    try:
        relative = step_path.resolve().relative_to(Path(source_dir).resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("receiver ACK step is outside the configured adapter source directory") from exc
    lora = getattr(getattr(config, "model", None), "lora", None)
    base_name = getattr(lora, "name", None)
    if not isinstance(base_name, str) or not base_name:
        raise RuntimeError("receiver ACK requires a configured LoRA name")
    return {
        "version": 1,
        "run_name": run_name,
        "step": step,
        "source_url": f"{source_url.rstrip('/')}/{quote(relative)}",
        **published,
        "slot": base_name if step == 0 else f"{base_name}-slot{step % 2}",
        "receivers": None,
    }


def _validate_receiver_ack(ack_path: Path, config, step_path: Path, current_step: int, run_name: str) -> int:
    if ack_path.is_symlink() or not ack_path.is_file():
        raise RuntimeError(f"receiver ACK is not a regular file: {ack_path}")
    try:
        ack = json.loads(ack_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"receiver ACK is invalid: {ack_path}") from exc
    if not isinstance(ack, dict):
        raise RuntimeError(f"receiver ACK is not an object: {ack_path}")
    step_name = step_path.name.removeprefix("step_")
    step = int(step_name)
    expected = _receiver_ack_identity(config, step_path, step, run_name)
    if step > current_step:
        raise RuntimeError(f"receiver ACK is from a future step: {ack_path}")
    if set(ack) != set(expected):
        raise RuntimeError(f"receiver ACK fields are invalid: {ack_path}")
    for name, value in expected.items():
        if name != "receivers" and type(ack.get(name)) is not type(value):
            raise RuntimeError(f"receiver ACK {name} has an invalid type: {ack_path}")
        if name == "slot":
            base_name = value.removesuffix(f"-slot{step % 2}")
            if ack.get(name) not in {value, base_name}:
                raise RuntimeError(f"receiver ACK {name} differs from the published step: {ack_path}")
        elif name != "receivers" and ack.get(name) != value:
            raise RuntimeError(f"receiver ACK {name} differs from the published step: {ack_path}")
    receivers = ack.get("receivers")
    if not isinstance(receivers, list) or len(receivers) != len(_receiver_ids(config)):
        raise RuntimeError(f"receiver ACK endpoint set is incomplete: {ack_path}")
    expected_endpoints = _receiver_ids(config)
    actual_endpoints = []
    receipt_fields = {
        "source_url": "source_url",
        "target_sha256": "target_sha256",
        "target_size": "target_size",
        "config_sha256": "config_sha256",
        "config_size": "config_size",
    }
    for item in receivers:
        if not isinstance(item, dict) or set(item) != {"endpoint", "receipt"}:
            raise RuntimeError(f"receiver ACK record is invalid: {ack_path}")
        endpoint = item["endpoint"]
        receipt = item["receipt"]
        if not isinstance(endpoint, str) or endpoint in actual_endpoints or not isinstance(receipt, dict):
            raise RuntimeError(f"receiver ACK record is invalid: {ack_path}")
        if set(receipt) != {
            "source_url", "target_sha256", "target_size", "config_sha256", "config_size",
            "lora_name", "server_instance",
        }:
            raise RuntimeError(f"receiver ACK receipt is invalid: {ack_path}")
        actual_endpoints.append(endpoint)
        for receipt_name, ack_name in receipt_fields.items():
            if type(receipt.get(receipt_name)) is not type(ack[ack_name]) or receipt.get(receipt_name) != ack[ack_name]:
                raise RuntimeError(f"receiver ACK receipt differs in {receipt_name}: {ack_path}")
        if receipt.get("lora_name") != ack["slot"] or not isinstance(receipt.get("server_instance"), str) or not receipt["server_instance"]:
            raise RuntimeError(f"receiver ACK receipt slot or server is invalid: {ack_path}")
    if actual_endpoints != expected_endpoints:
        raise RuntimeError(f"receiver ACK endpoint set differs from configuration: {ack_path}")
    return step


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={self.save_sharded})"
        )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        adapter_only = self.lora_config is not None

        if not adapter_only:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_to_hf(state_dict)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                state_dict = revert_weight_conversion(model, state_dict)

        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.logger.debug(
                f"Broadcasting weights for run {idx} (ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            if adapter_only:
                # For adapter-only, MultiRunManager creates state dict directly for each run
                # All ranks must participate in DTensor gathering, but only master saves
                state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
                # Under pipeline parallelism each rank holds only its stage's
                # layers and full_tensor() does not reassemble across stages, so
                # every stage master must materialise its shard on CPU and the
                # global master merges them. Without this the adapter shipped to
                # the inference engines contains one stage's layers only.
                from prime_rl.trainer.weights import is_pp_stage_master, merge_pp_state_dict

                keep_local = self.world.is_master or is_pp_stage_master()
                for key, value in state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if keep_local:
                        state_dict[key] = value.to("cpu", non_blocking=False)
                state_dict = merge_pp_state_dict(state_dict, is_master=self.world.is_master)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    run_id = self.multi_run_manager.idx_2_id[idx]
                    # pack() already advanced progress to the next step, so the model we just
                    # trained — policy v(step-1) — broadcasts to broadcasts/step_{step-1}.
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step - 1,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    self.logger.debug(f"Saving weights for run {idx} to {save_dir}")
                    save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
                    if adapter_only:
                        if run_id == "run_trainer_only":
                            orch_lora = self.lora_config
                        else:
                            orch_lora = self.multi_run_manager.config[idx].model.lora
                        if orch_lora is None:
                            raise RuntimeError("adapter export has no LoRA configuration")
                        save_lora_config(
                            model,
                            save_dir,
                            rank=orch_lora.rank,
                            alpha=orch_lora.alpha,
                            dropout=self.lora_config.dropout,
                        )
                        broadcast_step = self.multi_run_manager.progress[idx].step - 1
                        previous_dir = get_step_path(
                            get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                            broadcast_step - 1,
                        )
                        if broadcast_step >= 2:
                            previous_previous_dir = get_step_path(
                                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                                broadcast_step - 2,
                            )
                            if not previous_dir.is_dir() or not previous_previous_dir.is_dir():
                                raise RuntimeError(
                                    f"cannot publish adapter v{broadcast_step}: "
                                    f"predecessors v{broadcast_step - 1} and v{broadcast_step - 2} "
                                    "are not available"
                                )
                            if not (previous_dir / "STABLE").is_file() or not (
                                previous_previous_dir / "STABLE"
                            ).is_file():
                                raise RuntimeError(
                                    f"cannot publish adapter v{broadcast_step}: both predecessors "
                                    "must be marked STABLE"
                                )
                            orchestrator_config = self.multi_run_manager.config[idx]
                            source_url = getattr(
                                orchestrator_config.weight_broadcast, "source_url", None
                            )
                            source_dir = getattr(
                                orchestrator_config.weight_broadcast, "source_dir", None
                            )
                            if not isinstance(source_url, str) or not source_url or source_dir is None:
                                raise RuntimeError(
                                    f"cannot publish adapter v{broadcast_step}: predictive transport "
                                    "requires weight_broadcast.source_url and source_dir"
                                )
                            try:
                                source_root = Path(source_dir)
                                target_relative = save_dir.relative_to(source_root).as_posix()
                            except ValueError as exc:
                                raise RuntimeError(
                                    f"cannot publish adapter v{broadcast_step}: target is outside "
                                    "weight_broadcast.source_dir"
                                ) from exc
                            target_base_url = source_url.rstrip("/")
                            target_url = f"{target_base_url}/{quote(target_relative)}"
                            base_url = f"{target_base_url}/{quote(previous_dir.relative_to(source_root).as_posix())}"
                            base2_url = f"{target_base_url}/{quote(previous_previous_dir.relative_to(source_root).as_posix())}"
                            manifest = publish_adapter_manifest(
                                save_dir,
                                previous_dir,
                                base2_dir=previous_previous_dir,
                                target_url=target_url,
                                base_url=base_url,
                                base2_url=base2_url,
                                target_step=broadcast_step,
                                base_step=broadcast_step - 1,
                                base2_step=broadcast_step - 2,
                            )
                        else:
                            manifest = publish_adapter_manifest(
                                save_dir,
                                previous_dir if previous_dir.is_dir() else None,
                                target_step=broadcast_step,
                                base_step=broadcast_step - 1 if previous_dir.is_dir() else None,
                            )
                        self.logger.info(
                            f"Published adapter {manifest['mode']} transport: "
                            f"{manifest['encoded_size']} encoded bytes for "
                            f"{manifest['full_size']} full bytes in "
                            f"{manifest['build_seconds']}s using "
                            f"{manifest['chunk_count']} chunks"
                        )

                    self._notify_orchestrator(save_dir)

                    # If the run is deleted, remove the run directory
                    # This is avoid the creation of zombie runs when the directory is deleted while we are broadcasting which recreates the directory
                    if (
                        run_id != "run_trainer_only"
                        and self.multi_run_manager.get_orchestrator_config(run_id) is None
                    ):
                        shutil.rmtree(self.multi_run_manager.get_run_dir(idx))

                except FileNotFoundError:
                    run_dir = self.multi_run_manager.get_run_dir(idx)
                    if (
                        run_id != "run_trainer_only"
                        and self.multi_run_manager.get_orchestrator_config(run_id) is None
                        and not run_dir.exists()
                    ):
                        self.logger.warning(f"Run {idx} is deleted, skipping")
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                    raise
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, interval_to_keep: int | None):
        if not self.world.is_master:
            return
        for idx in self.multi_run_manager.used_idxs:
            broadcast_dir = get_broadcast_dir(self.multi_run_manager.get_run_dir(idx))
            current_step = self.multi_run_manager.progress[idx].step - 1
            if self.lora_config is None:
                maybe_clean(broadcast_dir, current_step, interval_to_keep)
                continue
            run_id = self.multi_run_manager.idx_2_id[idx]
            if run_id == "run_trainer_only":
                self.logger.debug(f"Retaining LoRA broadcasts for {run_id}: no receiver ACK is possible")
                continue
            orchestrator_config = self.multi_run_manager.get_orchestrator_config(run_id)
            if orchestrator_config is None:
                self.logger.warning(f"Retaining LoRA broadcasts for {run_id}: orchestrator config is unavailable")
                continue
            highest_ack_step = None
            highest_ack_path = None
            for step_path in broadcast_dir.iterdir():
                if (
                    step_path.is_symlink()
                    or not step_path.is_dir()
                    or re.fullmatch(r"step_[0-9]+", step_path.name) is None
                ):
                    continue
                step = int(step_path.name.removeprefix("step_"))
                ack_path = step_path / "receivers-ready.json"
                if not ack_path.exists() and not ack_path.is_symlink():
                    continue
                if step > current_step:
                    raise RuntimeError(f"receiver ACK is from a future step: {ack_path}")
                if highest_ack_step is None or step > highest_ack_step:
                    highest_ack_step = step
                    highest_ack_path = ack_path
            if highest_ack_path is None:
                self.logger.debug(f"Retaining LoRA broadcasts for {run_id}: no receiver ACK yet")
                continue
            assert highest_ack_step is not None
            _validate_receiver_ack(
                highest_ack_path,
                orchestrator_config,
                get_step_path(broadcast_dir, highest_ack_step),
                current_step,
                run_id,
            )
            retention_floor = min(current_step - 2, highest_ack_step - 2)
            if retention_floor <= 0:
                continue
            for candidate_path in broadcast_dir.iterdir():
                if (
                    candidate_path.is_symlink()
                    or not candidate_path.is_dir()
                    or re.fullmatch(r"step_[0-9]+", candidate_path.name) is None
                ):
                    continue
                candidate_step = int(candidate_path.name.removeprefix("step_"))
                if candidate_step >= retention_floor:
                    continue
                self.logger.debug(f"Removing path {candidate_path}")
                shutil.rmtree(candidate_path)
