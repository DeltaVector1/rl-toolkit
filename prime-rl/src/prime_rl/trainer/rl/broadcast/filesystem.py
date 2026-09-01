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
                    self.logger.warning(f"Run {idx} is deleted, skipping")
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
            # The active policy may need both predecessors after a restart.
            # Keep the current adapter and its two bases; remove only an older
            # step after the new adapter has been published successfully.
            candidate_step = current_step - 3
            if candidate_step < 0:
                continue
            if interval_to_keep and candidate_step % interval_to_keep == 0:
                self.logger.debug(
                    f"Keeping path {get_step_path(broadcast_dir, candidate_step)} "
                    "(on ckpt interval)"
                )
                continue
            candidate_path = get_step_path(broadcast_dir, candidate_step)
            self.logger.debug(f"Removing path {candidate_path}")
            shutil.rmtree(candidate_path, ignore_errors=True)
