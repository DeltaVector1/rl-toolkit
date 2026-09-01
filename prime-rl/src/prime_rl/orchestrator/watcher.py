"""WeightWatcher: polls the broadcast dir, advances ``Policy``, notifies
observers (dispatcher → off-policy cancel). Standalone async task; the
orchestrator's barrier bounds the in-flight lead."""

from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import quote
from urllib.request import urlopen

from modelexpress import p2p_pb2

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import (
    InferencePool,
    _delta_stage_timeout_s,
    load_lora_adapter,
    model_is_visible,
    stage_full_checkpoint,
    stage_lora_adapter,
)
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step
from prime_rl.utils.checkpoint_delta import build_xor_delta_checkpoint


v1_adapter_transport = "chunked-shuffled-xor-zstd"
v2_adapter_transport = "chunked-bf16-linear-zigzag-sparse-zstd"


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _adapter_file_records(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("adapter manifest has no file records")
    files: dict[str, dict[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError("adapter manifest has an invalid file record")
        name = item["name"]
        if name in files or name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError("adapter manifest has an invalid file set")
        size = item.get("size")
        digest = item.get("sha256")
        if type(size) is not int or size < 0 or not _valid_digest(digest):
            raise RuntimeError("adapter manifest has an invalid file identity")
        files[name] = item
    if set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise RuntimeError("adapter manifest has an incomplete file set")
    return files


def validate_adapter_manifest(
    manifest: dict[str, object],
    *,
    target_step: int,
    target_url: str | None = None,
    base_url: str | None = None,
    base2_url: str | None = None,
) -> dict[str, object]:
    """Validate the exact adjacent transport selected by a policy update."""
    if manifest.get("kind") != "lora-adapter" or manifest.get("mode") != "delta":
        raise RuntimeError("adapter manifest is not a delta adapter")
    if type(manifest.get("target_step")) is not int or manifest["target_step"] != target_step:
        raise RuntimeError("adapter manifest target step is not the requested policy")
    version = manifest.get("version")
    if version == 1:
        if manifest.get("transport") != v1_adapter_transport:
            raise RuntimeError("adapter manifest selects an unsupported version-one transport")
        if manifest.get("base_step") != target_step - 1:
            raise RuntimeError("adapter manifest base step is not adjacent to its target")
    elif version == 2:
        if manifest.get("transport") != v2_adapter_transport:
            raise RuntimeError("adapter manifest selects an unsupported predictive transport")
        if target_step < 2:
            raise RuntimeError("predictive adapter transport requires two predecessor steps")
        if manifest.get("base_step") != target_step - 1 or manifest.get("base2_step") != target_step - 2:
            raise RuntimeError("predictive adapter manifest steps are not adjacent")
        for prefix in ("target", "base", "base2"):
            size = manifest.get(f"{prefix}_size")
            digest = manifest.get(f"{prefix}_sha256")
            if type(size) is not int or size <= 0 or not _valid_digest(digest):
                raise RuntimeError(f"predictive adapter manifest has an invalid {prefix} identity")
        for name in ("config_sha256", "base_config_sha256", "base2_config_sha256"):
            if not _valid_digest(manifest.get(name)):
                raise RuntimeError(f"predictive adapter manifest has an invalid {name}")
        for name in ("config_size", "base_config_size", "base2_config_size"):
            value = manifest.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"predictive adapter manifest has an invalid {name}")
        data_offset = manifest.get("data_offset")
        if type(data_offset) is not int or data_offset < 0 or data_offset % 2:
            raise RuntimeError("predictive adapter manifest has an invalid data offset")
    else:
        raise RuntimeError("adapter manifest version is unsupported")

    if version == 2:
        if not isinstance(target_url, str) or manifest.get("target_url") != target_url:
            raise RuntimeError("predictive adapter manifest target URL differs from the requested policy")
        if not isinstance(base_url, str) or manifest.get("base_url") != base_url:
            raise RuntimeError("predictive adapter manifest base URL differs from the adjacent policy")
        if not isinstance(base2_url, str) or manifest.get("base2_url") != base2_url:
            raise RuntimeError("predictive adapter manifest second base URL differs from the adjacent policy")
    if target_url is not None and manifest.get("target_url") not in (None, target_url):
        raise RuntimeError("adapter manifest target URL differs from the requested policy")
    if base_url is not None and manifest.get("base_url") not in (None, base_url):
        raise RuntimeError("adapter manifest base URL differs from the adjacent policy")
    if version == 2:
        files = _adapter_file_records(manifest)
        if manifest["target_size"] != files["adapter_model.safetensors"]["size"]:
            raise RuntimeError("predictive adapter target size differs from its file record")
        if manifest["target_sha256"] != files["adapter_model.safetensors"]["sha256"]:
            raise RuntimeError("predictive adapter target digest differs from its file record")
        if manifest["config_size"] != files["adapter_config.json"]["size"]:
            raise RuntimeError("predictive adapter config size differs from its file record")
        if manifest["config_sha256"] != files["adapter_config.json"]["sha256"]:
            raise RuntimeError("predictive adapter config digest differs from its file record")
        if manifest["base_size"] != manifest["target_size"] or manifest["base2_size"] != manifest["target_size"]:
            raise RuntimeError("predictive adapter predecessor sizes differ from the target")
    return manifest


def inference_weight_path(config: OrchestratorConfig, weights_path, lora_name: str | None):
    broadcast = config.weight_broadcast
    if not (broadcast.type == "filesystem" and broadcast.source_url):
        return weights_path
    if broadcast.source_dir is None:
        raise ValueError("weight_broadcast.source_dir is required with source_url")
    return (
        f"{broadcast.source_url.rstrip('/')}/"
        f"{quote(weights_path.relative_to(broadcast.source_dir).as_posix())}"
    )


def _read_adapter_manifest(path, source_url: str | None, timeout: float | int | None) -> tuple[dict[str, object], str]:
    """Read a published adapter manifest from local storage or its source URL."""
    manifest_path = path / "adapter_manifest.json"
    if source_url is None or manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as source:
                manifest = json.load(source)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError("published adapter manifest is unavailable") from exc
        location = manifest_path.as_posix()
    else:
        try:
            with urlopen(f"{source_url.rstrip('/')}/adapter_manifest.json", timeout=timeout) as source:
                manifest = json.load(source)
        except (OSError, ValueError) as exc:
            raise RuntimeError("published adapter manifest is unavailable from the adapter source") from exc
        location = f"{source_url.rstrip('/')}/adapter_manifest.json"
    if not isinstance(manifest, dict):
        raise RuntimeError("published adapter manifest is not an object")
    return manifest, location


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        policy: Policy,
        inference: InferencePool,
        observers: list[VersionObserver],
        lora_name: str | None,
        ckpt_step: int = 0,
        poll_interval: float = 1.0,
        reconcile_interval: float = 30.0,
        model_express: ModelExpressSession | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.inference = inference
        self.observers = observers
        self.lora_name = lora_name
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval
        self.reconcile_interval = reconcile_interval
        self.model_express = model_express

        self.last_update_weights_time: float = 0.0
        self.last_stage_weights_time: float = 0.0
        self.last_adapter_delta_build_time: float = -1.0
        self.last_adapter_wire_stage_time: float = 0.0
        self.last_adapter_reconstruct_time: float = -1.0
        self.last_adapter_load_time: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0
        self.reconcile_count: int = 0
        self.last_reconcile_time: float = 0.0
        self.last_reconcile_check_at: float = 0.0
        self.current_weights_path = None

        # A delta is safe only when the inference pool has confirmed the
        # immediately previous adapter. A restarted watcher may trust the
        # startup load, but it must never skip an adapter version.
        self.verified_adapter_step: int | None = ckpt_step if lora_name is not None else None
        self.verified_adapter_url: str | None = None
        self.bootstrap_staged = lora_name is None or ckpt_step != 0 or model_express is not None

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                if not self.bootstrap_staged:
                    try:
                        await self.stage_bootstrap_adapter()
                    except Exception:
                        get_logger().exception("Failed to stage the initial adapter; retrying")
                    if not self.bootstrap_staged:
                        await asyncio.sleep(self.poll_interval)
                        continue
                next_step = self.compute_next_ckpt_step()
                if next_step > self.ckpt_step:
                    try:
                        await self.apply_policy_update(next_step)
                    except Exception:
                        get_logger().exception(f"Failed to apply policy update {next_step}; retrying")
                elif time.monotonic() - self.last_reconcile_check_at >= self.reconcile_interval:
                    self.last_reconcile_check_at = time.monotonic()
                    try:
                        await self.reconcile_current_policy()
                    except Exception:
                        get_logger().exception("Failed to reconcile the current LoRA; retrying")
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stage_bootstrap_adapter(self) -> None:
        """Stage policy zero without loading it or advancing policy state."""
        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        local_path = get_step_path(broadcast_dir, 0)
        if not (local_path / "STABLE").is_file():
            return
        source_path = inference_weight_path(self.config, local_path, self.lora_name)
        if isinstance(source_path, str):
            started = time.perf_counter()
            result = await stage_lora_adapter(
                self.inference.admin_clients,
                source_path,
                require_delta=False,
                origin_base_urls=self.inference.adapter_origin_base_urls,
            )
            self.last_adapter_wire_stage_time = float(result.get("wire_stage_seconds", 0.0))
            origin_stage = format_time(float(result.get("origin_stage_seconds", 0.0)))
            peer_stage = format_time(float(result.get("peer_stage_seconds", 0.0)))
            reconstruct_seconds = result.get("reconstruct_seconds")
            self.last_adapter_reconstruct_time = (
                float(reconstruct_seconds) if reconstruct_seconds is not None else -1.0
            )
            self.last_stage_weights_time = time.perf_counter() - started
            get_logger().info(
                f"Staged initial adapter policy v0 on every inference host in "
                f"{format_time(self.last_stage_weights_time)} "
                f"(origin {origin_stage}, peer {peer_stage})"
            )
        self.current_weights_path = source_path
        self.verified_adapter_step = 0
        self.verified_adapter_url = str(source_path)
        self.bootstrap_staged = True

    async def stop(self) -> None:
        self.stopped.set()
        # Let an in-flight apply finish before dying: the trainer blocks inside
        # its in-memory broadcast until the apply completes, so cancelling
        # mid-apply would strand it. The orchestrator's global teardown budget
        # bounds this wait.
        async with self.update_lock:
            pass
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    def compute_next_ckpt_step(self) -> int:
        """Return the next policy version exposed by the configured transport."""
        if self.model_express is not None:
            if self.config.max_steps is not None and self.ckpt_step >= self.config.max_steps - 2:
                return self.ckpt_step
            # ModelExpress status changes are unversioned, so its rendezvous advances
            # exactly one policy version per READY/INITIALIZING cycle.
            return self.ckpt_step + 1

        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0
        if self.lora_name is not None:
            # Delta manifests are adjacent-step objects. Do not jump to the
            # newest checkpoint: that would apply a delta against the wrong
            # base after a delayed or restarted watcher.
            if latest_ckpt_step > self.ckpt_step:
                return self.ckpt_step + 1
            return self.ckpt_step
        return max(self.policy.version, latest_ckpt_step)

    def _adapter_delta_base(
        self,
        next_step: int,
        local_weights_path,
        target_url: str | None = None,
    ) -> str:
        """Return the verified adjacent base URL and published delta manifest."""
        if self.lora_name is None or next_step <= 0:
            raise RuntimeError("an adapter delta requires a positive LoRA policy step")
        if self.verified_adapter_step != next_step - 1:
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: verified base is "
                f"v{self.verified_adapter_step}, expected v{next_step - 1}"
            )

        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        previous_path = get_step_path(broadcast_dir, next_step - 1)
        if not (previous_path / "STABLE").is_file():
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: previous policy v{next_step - 1} "
                "is not marked STABLE"
            )
        previous_url = inference_weight_path(self.config, previous_path, self.lora_name)
        if not isinstance(previous_url, str):
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: previous policy has no source URL"
            )
        if self.verified_adapter_url is not None and previous_url != self.verified_adapter_url:
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: verified base URL changed"
            )

        previous_previous_url = None
        if next_step >= 2:
            previous_previous_path = get_step_path(broadcast_dir, next_step - 2)
            if not (previous_previous_path / "STABLE").is_file():
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: previous policy v{next_step - 2} "
                    "is not marked STABLE"
                )
            previous_previous_url = inference_weight_path(
                self.config, previous_previous_path, self.lora_name
            )
            if not isinstance(previous_previous_url, str):
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: second base has no source URL"
                )

        if target_url is None:
            target_url = inference_weight_path(self.config, local_weights_path, self.lora_name)
        try:
            manifest, manifest_location = _read_adapter_manifest(
                local_weights_path,
                target_url if isinstance(target_url, str) else None,
                _delta_stage_timeout_s(),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: {exc}"
            ) from exc
        if manifest.get("version") != 1:
            try:
                validate_adapter_manifest(
                    manifest,
                    target_step=next_step,
                    target_url=target_url if isinstance(target_url, str) else None,
                    base_url=previous_url,
                    base2_url=previous_previous_url,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: {exc}"
                ) from exc
        else:
            if (
                manifest.get("kind") != "lora-adapter"
                or manifest.get("mode") != "delta"
                or manifest.get("transport") != "chunked-shuffled-xor-zstd"
                or manifest.get("target_step") != next_step
                or manifest.get("base_step") != next_step - 1
            ):
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: manifest is not the adjacent delta"
                )
        build_seconds = manifest.get("build_seconds")
        self.last_adapter_delta_build_time = (
            float(build_seconds) if isinstance(build_seconds, (int, float)) and build_seconds >= 0 else -1.0
        )
        return manifest_location

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return

            t0 = time.perf_counter()
            weights_path = None
            local_weights_path = None
            require_delta = False
            if self.model_express is not None:
                await self.wait_for_model_express_status(p2p_pb2.SOURCE_STATUS_READY)
            else:
                broadcast_dir = get_broadcast_dir(self.config.output_dir)
                weights_path = get_step_path(broadcast_dir, next_step)
                stable_marker = weights_path / "STABLE"
                if not stable_marker.exists():
                    get_logger().info(
                        f"Orchestrator paused: waiting for trainer to broadcast checkpoint {next_step}. "
                        "Training is progressing normally."
                    )
                    await wait_for_path(stable_marker)
                local_weights_path = weights_path
                weights_path = inference_weight_path(self.config, weights_path, self.lora_name)
                if self.lora_name is None and next_step > 1 and isinstance(weights_path, str):
                    previous_path = get_step_path(broadcast_dir, next_step - 1)
                    previous_url = inference_weight_path(self.config, previous_path, self.lora_name)
                    if isinstance(previous_url, str):
                        delta_path = await asyncio.to_thread(
                            build_xor_delta_checkpoint,
                            previous_path,
                            local_weights_path,
                            base_url=previous_url,
                            target_url=weights_path,
                        )
                        if delta_path is not None:
                            get_logger().info(
                                f"Built lossless checkpoint delta for policy v{next_step}: {delta_path}"
                            )
                elif self.lora_name is not None and next_step > 0 and isinstance(weights_path, str):
                    if local_weights_path is None:
                        raise RuntimeError("adapter delta target path is not local")
                    manifest_path = self._adapter_delta_base(next_step, local_weights_path, weights_path)
                    require_delta = True
                    build_time = (
                        format_time(self.last_adapter_delta_build_time)
                        if self.last_adapter_delta_build_time >= 0
                        else "not reported by trainer"
                    )
                    get_logger().info(
                        f"Using lossless adapter delta for policy v{next_step}: {manifest_path} "
                        f"(build {build_time})"
                    )
            self.last_wait_for_ckpt_time = time.perf_counter() - t0

            # Remote LoRA files cross the high-loss China link. Stage them on
            # both inference hosts before stopping old-policy generation. This
            # keeps useful complete groups flowing during the transfer and
            # reduces the engine pause to the local in-place load.
            if isinstance(weights_path, str):
                stage_started = time.perf_counter()
                if self.lora_name is not None:
                    stage_result = await stage_lora_adapter(
                        self.inference.admin_clients,
                        weights_path,
                        require_delta=require_delta,
                        origin_base_urls=self.inference.adapter_origin_base_urls,
                    )
                    self.last_adapter_wire_stage_time = float(
                        stage_result.get("wire_stage_seconds", time.perf_counter() - stage_started)
                    )
                    reconstruct_seconds = stage_result.get("reconstruct_seconds")
                    if reconstruct_seconds is not None:
                        self.last_adapter_reconstruct_time = float(reconstruct_seconds)
                    else:
                        self.last_adapter_reconstruct_time = -1.0
                    origin_stage = format_time(float(stage_result.get("origin_stage_seconds", 0.0)))
                    peer_stage = format_time(float(stage_result.get("peer_stage_seconds", 0.0)))
                else:
                    await stage_full_checkpoint(self.inference.admin_clients, weights_path)
                self.last_stage_weights_time = time.perf_counter() - stage_started
                get_logger().info(
                    f"Staged policy v{next_step} on every inference host in "
                    f"{format_time(self.last_stage_weights_time)}"
                )
                if self.lora_name is not None:
                    reconstruct = (
                        format_time(self.last_adapter_reconstruct_time)
                        if self.last_adapter_reconstruct_time >= 0
                        else "not reported by inference host"
                    )
                    get_logger().info(
                        f"Adapter v{next_step} transfer: wire-stage "
                        f"{format_time(self.last_adapter_wire_stage_time)}, "
                        f"origin {origin_stage}, peer {peer_stage}, reconstruct {reconstruct}"
                    )

            # An observer can request a bounded safe boundary before we stop
            # old-policy work. The production orchestrator lets the current
            # valid batch ship instead of regenerating its incomplete groups.
            for observer in self.observers:
                wait_for_boundary = getattr(observer, "wait_for_policy_update_boundary", None)
                if wait_for_boundary is None:
                    continue
                try:
                    await wait_for_boundary(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.wait_for_policy_update_boundary"
                        f"({next_step}) raised: {exc!r}; applying the update now"
                    )

            # Stop old-policy rollouts before changing the adapter.
            for observer in self.observers:
                try:
                    await observer.on_version_pending(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.on_version_pending({next_step}) raised: {exc!r}"
                    )

            get_logger().debug(f"Updating weights to step {next_step}")
            t1 = time.perf_counter()
            lora_name = f"{self.lora_name}-slot{next_step % 2}" if self.lora_name is not None else None
            await self.inference.update_weights(
                weights_path,
                lora_name=lora_name,
                step=next_step,
                require_delta=require_delta,
                prefetched=True,
            )
            self.last_update_weights_time = time.perf_counter() - t1
            if self.lora_name is not None:
                self.last_adapter_load_time = self.last_update_weights_time
            self.update_count += 1
            get_logger().debug(f"Updated weights to step {next_step} in {format_time(self.last_update_weights_time)}")
            self.ckpt_step = next_step
            self.policy.version = next_step
            self.current_weights_path = weights_path

            if self.lora_name is not None:
                self.verified_adapter_step = next_step
                self.verified_adapter_url = str(weights_path)

            if lora_name is not None:
                self.inference.update_model_name(lora_name)
                self.policy.model_name = lora_name

            # The staging call and weight update cover all endpoints, but a
            # rollout must not use the new policy until both inference hosts
            # report the adapter as visible.
            await self.inference.wait_for_ready(self.policy.model_name)

            for observer in self.observers:
                try:
                    await observer.on_new_version(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.on_new_version({next_step}) raised: {exc!r}"
                    )

            if self.model_express is not None:
                await self.wait_for_model_express_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

    async def reconcile_current_policy(self) -> None:
        """Restore the current LoRA on endpoints that restarted after an update.

        A healthy base engine is not sufficient: every endpoint must expose the
        exact policy model before it receives more rollout requests.
        """
        if self.lora_name is None or self.policy.model_name == self.config.model.name:
            return
        async with self.update_lock:
            weights_path = self.current_weights_path
            if weights_path is None and self.model_express is None and self.ckpt_step > 0:
                broadcast_dir = get_broadcast_dir(self.config.output_dir)
                candidate = get_step_path(broadcast_dir, self.ckpt_step)
                if (candidate / "STABLE").exists():
                    weights_path = inference_weight_path(self.config, candidate, self.lora_name)
                    self.current_weights_path = weights_path
            if weights_path is None:
                return

            require_delta = self.ckpt_step > 0
            if require_delta and self.verified_adapter_step != self.ckpt_step:
                raise RuntimeError(
                    f"cannot reconcile policy v{self.ckpt_step}: exact loaded policy is not verified"
                )
            if require_delta and self.verified_adapter_url is not None:
                if str(weights_path) != self.verified_adapter_url:
                    raise RuntimeError(
                        f"cannot reconcile policy v{self.ckpt_step}: verified adapter URL changed"
                    )

            checks = await asyncio.gather(
                *(model_is_visible(client, self.policy.model_name) for client in self.inference.admin_clients)
            )
            missing = [client for client, visible in zip(self.inference.admin_clients, checks) if not visible]
            if not missing:
                return

            get_logger().warning(
                f"Policy model {self.policy.model_name} is absent on {len(missing)}/"
                f"{len(self.inference.admin_clients)} inference endpoint(s); restoring it"
            )
            t0 = time.perf_counter()
            await load_lora_adapter(
                missing,
                self.policy.model_name,
                weights_path,
                require_delta=require_delta,
                prefetched=False,
            )
            visible = await asyncio.gather(
                *(model_is_visible(client, self.policy.model_name) for client in self.inference.admin_clients)
            )
            if not all(visible):
                raise RuntimeError(
                    f"Policy model {self.policy.model_name} is still missing on "
                    f"{visible.count(False)} inference endpoint(s) after reload"
                )
            self.last_reconcile_time = time.perf_counter() - t0
            self.reconcile_count += 1
            get_logger().info(
                f"Restored policy model {self.policy.model_name} on every inference endpoint "
                f"in {format_time(self.last_reconcile_time)}"
            )

    async def wait_for_model_express_status(self, status: int) -> None:
        while not self.stopped.is_set():
            found = await asyncio.to_thread(
                self.model_express.exists_role_with_status,
                "trainer",
                status,
            )
            if found:
                return
            await asyncio.sleep(self.poll_interval)
        raise asyncio.CancelledError

    def gauges(self) -> dict[str, float]:
        return {
            "watcher/policy_version": float(self.policy.version),
            "watcher/update_count": float(self.update_count),
            "watcher/reconcile_count": float(self.reconcile_count),
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_stage_weights_time": self.last_stage_weights_time,
            "watcher/last_adapter_delta_build_time": self.last_adapter_delta_build_time,
            "watcher/last_adapter_wire_stage_time": self.last_adapter_wire_stage_time,
            "watcher/last_adapter_reconstruct_time": self.last_adapter_reconstruct_time,
            "watcher/last_adapter_load_time": self.last_adapter_load_time,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
            "watcher/last_reconcile_time": self.last_reconcile_time,
        }
