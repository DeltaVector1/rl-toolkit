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
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step


def inference_weight_path(config: OrchestratorConfig, weights_path, lora_name: str | None):
    broadcast = config.weight_broadcast
    if not (lora_name and broadcast.type == "filesystem" and broadcast.source_url):
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
        model_express: ModelExpressSession | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.inference = inference
        self.observers = observers
        self.lora_name = lora_name
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval
        self.model_express = model_express

        self.last_update_weights_time: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0

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
                weights_path = inference_weight_path(self.config, weights_path, self.lora_name)
            self.last_wait_for_ckpt_time = time.perf_counter() - t0

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
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
        }
