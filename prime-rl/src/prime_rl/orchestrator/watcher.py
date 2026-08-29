"""WeightWatcher: polls the broadcast dir, advances ``Policy``, notifies
observers (dispatcher → off-policy cancel). Standalone async task; the
orchestrator's barrier bounds the in-flight lead."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import quote

from modelexpress import p2p_pb2

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import (
    InferencePool,
    load_lora_adapter,
    model_is_visible,
    stage_full_checkpoint,
    stage_lora_adapter,
)
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step
from prime_rl.utils.checkpoint_delta import build_xor_delta_checkpoint


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
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0
        self.reconcile_count: int = 0
        self.last_reconcile_time: float = 0.0
        self.last_reconcile_check_at: float = 0.0
        self.current_weights_path = None

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
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
        return max(self.policy.version, latest_ckpt_step)

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return

            t0 = time.perf_counter()
            weights_path = None
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
            self.last_wait_for_ckpt_time = time.perf_counter() - t0

            # Remote LoRA files cross the high-loss China link. Stage them on
            # both inference hosts before stopping old-policy generation. This
            # keeps useful complete groups flowing during the transfer and
            # reduces the engine pause to the local in-place load.
            if isinstance(weights_path, str):
                stage_started = time.perf_counter()
                if self.lora_name is not None:
                    await stage_lora_adapter(self.inference.admin_clients, weights_path)
                else:
                    await stage_full_checkpoint(self.inference.admin_clients, weights_path)
                self.last_stage_weights_time = time.perf_counter() - stage_started
                get_logger().info(
                    f"Staged policy v{next_step} on every inference host in "
                    f"{format_time(self.last_stage_weights_time)}"
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
            await self.inference.update_weights(weights_path, lora_name=lora_name, step=next_step)
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            get_logger().debug(f"Updated weights to step {next_step} in {format_time(self.last_update_weights_time)}")
            self.ckpt_step = next_step
            self.policy.version = next_step
            self.current_weights_path = weights_path

            if lora_name is not None:
                self.inference.update_model_name(lora_name)
                self.policy.model_name = lora_name

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
            await load_lora_adapter(missing, self.policy.model_name, weights_path)
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
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
            "watcher/last_reconcile_time": self.last_reconcile_time,
        }
