"""RolloutDispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight_rollouts``) is shared across train + eval.
  A group-scoring task that runs N rollouts in one call reserves N permits.
- Optional rate limiting via ``AsyncLimiter(tasks_per_minute, 60)``.
- Emit-everything invariant: every dispatched env-rollout eventually reaches
  ``out_q`` exactly once, as one episode (a ``list[Rollout]``). Failures
  (env error, empty trajectory, task exception, off-policy cancel) carry
  ``trace.error`` set; sinks decide drop / partial-train policy.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight rollouts of the opposite kind drain
  naturally on either side of an eval boundary.
- ``on_version_pending`` (called by the watcher before the engines pause for
  the weight update) bumps ``off_policy_steps`` on in-flight train rollouts and
  drops groups past ``max_off_policy_steps``.
  Eval rollouts are measurements for the policy version they started with,
  so they are allowed to finish even if training advances. Train rollouts
  sampled from a frozen model never age — their sampler doesn't change
  with policy updates.
  Cancellations surface as synthetic ``Cancelled`` markers so the sink's
  count-to-``group_size`` finalization still fires.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import Literal

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    GroupState,
    InflightRollout,
    Policy,
    Rollout,
    RolloutKind,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import get_logger


# Let active requests finish their GRPO group before a policy change.
POLICY_GROUP_DRAIN_GRACE_S = 60
TRANSIENT_RETRY_WINDOW_S = 180
TRANSIENT_RETRIES_PER_ROLLOUT = 3


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class DispatcherMetrics:
    """Per-tick drain counters for the orchestrator's periodic log.
    ``drained()`` returns the current values and clears them; point-in-time
    gauges live on ``RolloutDispatcher.gauges`` instead."""

    cancelled_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errored_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_cancellation(self, *, kind: Literal["train", "eval"], env_name: str, n: int = 1) -> None:
        self.cancelled_by_kind_env[(kind, env_name)] += n

    def record_error(self, *, kind: Literal["train", "eval"], env_name: str) -> None:
        self.errored_by_kind_env[(kind, env_name)] += 1

    def drained(self, *, train_envs: set[str], eval_envs: set[str]) -> dict[str, float]:
        """Return per-tick counters and clear them. Emits the full pre-
        registered key set every tick (zero when no activity) so the wandb
        time axis stays dense and ``define_metric`` lines up."""
        out: dict[str, float] = {}
        for kind in ("train", "eval"):
            envs = train_envs if kind == "train" else eval_envs
            cancelled_total = sum(self.cancelled_by_kind_env.get((kind, e), 0) for e in envs)
            errored_total = sum(self.errored_by_kind_env.get((kind, e), 0) for e in envs)
            out[f"dispatcher/cancelled/{kind}"] = float(cancelled_total)
            out[f"dispatcher/errored/{kind}"] = float(errored_total)
        for env in train_envs | eval_envs:
            out[f"dispatcher/cancelled/{env}"] = float(
                self.cancelled_by_kind_env.get(("train", env), 0) + self.cancelled_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/errored/{env}"] = float(
                self.errored_by_kind_env.get(("train", env), 0) + self.errored_by_kind_env.get(("eval", env), 0)
            )
        self.cancelled_by_kind_env.clear()
        self.errored_by_kind_env.clear()
        return out

    @staticmethod
    def drain_keys(*, train_envs: set[str], eval_envs: set[str]) -> list[str]:
        """Full set of keys ``drained`` may emit; used by the periodic
        logger for ``wandb.define_metric``."""
        keys = [
            "dispatcher/cancelled/train",
            "dispatcher/cancelled/eval",
            "dispatcher/errored/train",
            "dispatcher/errored/eval",
        ]
        for env in train_envs | eval_envs:
            keys.append(f"dispatcher/cancelled/{env}")
            keys.append(f"dispatcher/errored/{env}")
        return keys


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules
    rollouts under shared capacity, and emits ``Rollout``\\ s to
    ``out_q``. The watcher drives ``on_version_pending`` for off-policy
    cancellation; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource | None,
        policy_pool: InferencePool,
        policy: Policy,
        max_inflight_rollouts: int,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
    ) -> None:
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env sampler's pool; eval always
        # evaluates the policy.
        self.policy_pool = policy_pool
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps

        self.max_inflight = max_inflight_rollouts
        self.inflight_permits = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        self.inflight: dict[asyncio.Task, InflightRollout] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow sink. One entry per
        # episode — the sinks count episodes, never loose traces.
        self.out_q: asyncio.Queue[list[Rollout]] = asyncio.Queue(maxsize=max(8, self.max_inflight))
        self.queued_train_episodes = 0

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set by the orchestrator after the final train step; pipeline then
        # winds down without scheduling new train rollouts
        self.train_scheduling_disabled: bool = False
        self.fresh_train_groups_allowed = asyncio.Event()
        self.fresh_train_groups_allowed.set()
        self.fresh_train_rollout_budget = self.max_inflight
        self.metrics = DispatcherMetrics()
        self.unhealthy_clients: dict[tuple[str, str | None], float] = {}
        self.transient_retries: dict[uuid.UUID, tuple[float, int]] = {}

        # Orchestrator-owned permission. When clear, ``fill_inflight`` returns
        # without scheduling new groups. The dispatcher itself doesn't know
        # *why* — the orchestrator toggles this based on step / policy lead.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()
        self.weight_update_pending = False
        self.policy_group_drain_pending = False
        # Policy versions whose active train batch already shipped while the
        # watcher waited. Only overflow groups remain, so another drain grace
        # would delay the update and then discard the same overflow anyway.
        self.batch_boundary_complete_steps: set[int] = set()
        # Old group ids whose partial results were invalidated at a policy
        # boundary. Queued results with these ids must not reach a sink.
        self.abandoned_group_ids: set[uuid.UUID] = set()
        self.policy_restart_groups: dict[int, set[uuid.UUID]] = {}

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    def _train_pool_for(self, env_name: str) -> tuple[InferencePool, str, bool]:
        """``(pool, model_name, is_live)`` for *train* rollouts of this env —
        the env sampler's pool. (Eval always uses the policy.)"""
        sampler = self.train_envs.get(env_name).sampler
        if sampler.samples_from_live_policy:
            return sampler.pool, self.policy.model_name, True
        return sampler.pool, sampler.pool.model_name, False

    @property
    def inflight_train_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "train")


    @property
    def queued_train_count(self) -> int:
        return self.queued_train_episodes

    def mark_episode_consumed(self, kind: RolloutKind) -> None:
        if kind == "train":
            self.queued_train_episodes -= 1

    @property
    def pending_train_schedule_count(self) -> int:
        return sum(g.rollouts_to_schedule for g in self.groups.values() if g.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "eval")

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.inflight_permits

    @property
    def inflight_by_env(self) -> dict[tuple[RolloutKind, str], int]:
        counts: dict[tuple[RolloutKind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += meta.rollout_count
        return dict(counts)

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source) if self.eval_source is not None else 0

    @property
    def eval_has_work(self) -> bool:
        """Eval has work while its source queue is non-empty OR any opened eval group still has
        rollouts to schedule. An example leaves ``eval_source`` when its group opens
        (``next_fresh_group``), but its ``group_size`` rollouts dispatch one at a time across
        ``fill_inflight`` passes — so the queue can be empty while a group is still mid-schedule."""
        return bool(self.eval_source) or any(
            g.kind == "eval" and g.rollouts_to_schedule > 0 for g in self.groups.values()
        )

    @property
    def is_idle(self) -> bool:
        """True once nothing is in flight, no eval work remains (queued *or* a partly-scheduled eval
        group), and ``out_q`` is empty — the pipeline has fully drained."""
        return not self.inflight and not self.eval_has_work and self.out_q.empty()

    def disable_train_scheduling(self) -> None:
        """Stop scheduling new train rollouts; in-flight train + any
        triggered eval drain naturally."""
        self.train_scheduling_disabled = True

    def set_fresh_train_rollout_budget(self, budget: int) -> None:
        budget = max(0, budget)
        allowed = budget > 0
        changed = allowed != self.fresh_train_groups_allowed.is_set()
        self.fresh_train_rollout_budget = budget
        if not changed:
            return
        if allowed:
            self.fresh_train_groups_allowed.set()
            get_logger().info("Opening fresh train groups")
        else:
            self.fresh_train_groups_allowed.clear()
            get_logger().info("Pausing fresh train groups while the current batch drains")

    @property
    def max_off_policy_level(self) -> int:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return max(steps) if steps else 0

    @property
    def mean_off_policy_level(self) -> float:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return sum(steps) / len(steps) if steps else 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly. Eval triggers from the
                    # orchestrator wake the next iteration via a mode flip
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (mode flips)
                )
                for task in done:
                    await self.handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_rollouts()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def on_version_pending(self, step: int) -> None:
        """Stop admission and restart incomplete live-policy groups at ``step``.

        A LoRA load needs an empty vLLM scheduler. Regenerate a complete group
        after cancellation so the GRPO baseline stays valid.
        """
        # Set this before any cancellation so fill_inflight cannot schedule a
        # task between the snapshot and the group reset.
        self.weight_update_pending = True
        old_group_ids = {
            gid
            for gid, group in self.groups.items()
            if group.kind == "train"
            and self.train_envs.get(group.env_name).sampler.samples_from_live_policy
        }
        self.abandoned_group_ids.update(old_group_ids)
        self.policy_restart_groups[step] = set(old_group_ids)

        tasks_to_cancel: list[asyncio.Task] = []
        cancelled_rollouts = 0
        for task, meta in list(self.inflight.items()):
            if meta.group_id not in old_group_ids:
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name, n=meta.rollout_count)
            cancelled_rollouts += meta.rollout_count
            tasks_to_cancel.append(task)

        restarted = 0
        discarded_episodes = 0
        for old_gid in old_group_ids:
            group = self.groups.pop(old_gid, None)
            self.transient_retries.pop(old_gid, None)
            if group is None:
                continue
            discarded_episodes += group.emitted
            group.emitted = 0
            group.rollouts_to_schedule = group.target_rollouts
            group.pinned_client = None
            group.policy_version_at_start = step
            self.groups[uuid.uuid4()] = group
            restarted += 1

        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        if restarted:
            get_logger().info(
                f"Policy v{step}: restarting {restarted} incomplete GRPO group(s) from their original prompts "
                f"({cancelled_rollouts} live and {discarded_episodes} completed rollout(s) discarded)"
            )

        # Eval is a version-pinned measurement. Do not silently move it to a
        # different policy.
        while any(meta.kind == "eval" for meta in self.inflight.values()) and not self.stopped.is_set():
            await asyncio.sleep(0.1)

    async def wait_for_policy_update_boundary(self, step: int) -> None:
        """Bounded grace for incomplete groups that already have useful rows.

        Stop opening fresh prompts, but continue scheduling the remaining
        members of groups already in progress. Fully completed groups leave
        ``self.groups`` and remain valid old-policy input. The normal strict
        boundary then restarts only the groups that remain incomplete.
        """
        if step in self.batch_boundary_complete_steps:
            self.batch_boundary_complete_steps.discard(step)
            get_logger().info(
                f"Policy v{step}: active rollout batch already shipped; "
                "skipping overflow-group grace"
            )
            return

        live_groups = {
            gid: group
            for gid, group in self.groups.items()
            if group.kind == "train"
            and self.train_envs.get(group.env_name).sampler.samples_from_live_policy
        }
        completed = sum(group.emitted for group in live_groups.values())
        smallest_group = min((group.target_rollouts for group in live_groups.values()), default=1)
        if completed < smallest_group:
            return

        self.policy_group_drain_pending = True
        get_logger().info(
            f"Policy v{step}: giving {len(live_groups)} incomplete GRPO group(s) up to "
            f"{POLICY_GROUP_DRAIN_GRACE_S}s to finish ({completed} completed member(s) retained)"
        )
        deadline = monotonic() + POLICY_GROUP_DRAIN_GRACE_S
        while monotonic() < deadline and not self.stopped.is_set():
            if not any(
                group.kind == "train"
                and self.train_envs.get(group.env_name).sampler.samples_from_live_policy
                for group in self.groups.values()
            ):
                break
            await asyncio.sleep(0.1)

    def mark_batch_boundary_complete(self, step: int) -> None:
        """Skip grace once when the orchestrator already shipped the batch."""
        self.batch_boundary_complete_steps.add(step)

    def is_abandoned_group(self, group_id: uuid.UUID) -> bool:
        """Return whether a queued episode belongs to an invalidated group."""
        return group_id in self.abandoned_group_ids

    def restarted_group_ids(self, step: int) -> set[uuid.UUID]:
        """Old group ids invalidated while preparing policy ``step``."""
        return set(self.policy_restart_groups.get(step, set()))

    async def on_new_version(self, step: int) -> None:
        """Allow scheduling after the pre-pause request drain and update."""
        self.weight_update_pending = False
        self.policy_group_drain_pending = False

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. Eval scheduling ignores the orchestrator's dispatch
        permission (evals are version-pinned measurements); only train scheduling
        respects it. When ``PREFER_EVAL``'s source exhausts we flip back to
        ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.weight_update_pending:
                return
            if self.available_permits <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL is only entered when the orchestrator triggers
                # eval, which requires ``eval_source`` to be configured
                assert self.eval_source is not None
                if not self.eval_has_work:
                    # Eval source + all eval groups fully dispatched. Flip
                    # to PREFER_TRAIN so any remaining permits go to train
                    # while the in-flight eval tail completes naturally
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN — respects the orchestrator's dispatch permission
                if not self.dispatch_allowed.is_set():
                    return
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} rollouts because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: RolloutKind) -> bool:
        """Schedule one rollout of ``kind``: prefer continuing an existing
        group (keeps prefix-cache hits); otherwise open a fresh group from
        the corresponding source. Returns False if nothing could be
        scheduled."""
        if kind == "train" and self.train_scheduling_disabled:
            return False
        envs = self.train_envs if kind == "train" else self.eval_envs
        if envs is None:
            return False

        for gid, group in list(self.groups.items()):
            if group.kind != kind or group.rollouts_to_schedule <= 0:
                continue
            env = envs.get(group.env_name)
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= self.available_permits:
                return await self.schedule_group_rollout(gid, group)

        if kind == "train" and (
            not self.fresh_train_groups_allowed.is_set() or self.policy_group_drain_pending
        ):
            return False

        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh)

    def next_fresh_group(self, kind: RolloutKind, envs) -> GroupState | None:
        """Pop the next example from the corresponding source and wrap it in
        a ``GroupState``. Returns ``None`` if the source is empty or the
        picked env's permit cost doesn't fit."""
        if kind == "train":
            source = self.train_source
        else:
            assert self.eval_source is not None
            source = self.eval_source
        available_permits = self.available_permits
        if kind == "train":
            available_permits = min(available_permits, self.fresh_train_rollout_budget)
        example = source.next_example(available_permits)
        if example is None:
            return None

        env_name = example["env_name"]
        group_size = envs.get(env_name).config.group_size
        eval_step: int | None = example.get("eval_step") if kind == "eval" else None

        fresh = GroupState(
            kind=kind,
            env_name=env_name,
            task_idx=example["task_idx"],
            task=example.get("task"),
            rollouts_to_schedule=group_size,
            target_rollouts=group_size,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
        )
        if kind == "train":
            self.fresh_train_rollout_budget -= group_size
        return fresh

    async def schedule_group_rollout(self, group_id: uuid.UUID, group: GroupState) -> bool:
        """Dispatch one ``run`` / ``run_group`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Train rollouts use the env sampler's pool via the
        # renderer/token train client. Eval always evaluates the policy and
        # goes through the eval client (chat-completions) — the same path the
        # legacy orchestrator used, so eval scores stay comparable.
        if group.kind == "eval":
            pool, model_name = self.policy_pool, self.policy.model_name
            live_sourced = True
        else:
            pool, model_name, live_sourced = self._train_pool_for(group.env_name)

        if group.pinned_client is not None and self.unhealthy_clients.get(
            client_identity(group.pinned_client), 0
        ) > monotonic():
            group.pinned_client = None

        # Pin a single client per group to keep prefix-cache hits
        if group.pinned_client is None:
            now = monotonic()
            self.unhealthy_clients = {key: until for key, until in self.unhealthy_clients.items() if until > now}
            if group.kind == "eval":
                for _ in range(max(1, len(pool.train_clients))):
                    client = await pool.get_eval_client()
                    if client_identity(client) not in self.unhealthy_clients:
                        break
                else:
                    return False
            else:
                load = Counter(
                    client_identity(m.client_config) for m in self.inflight.values() if m.client_config is not None
                )
                if pool.train_clients and all(
                    client_identity(candidate) in self.unhealthy_clients for candidate in pool.train_clients
                ):
                    return False
                for identity in self.unhealthy_clients:
                    load[identity] = self.max_inflight + 1
                client = await pool.select_train_client(load)
            if group_id not in self.groups:
                return False
            group.pinned_client = client
        else:
            client = group.pinned_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        # Frozen-sourced train rollouts hit a frozen pool; salting per policy
        # version would invalidate its prefix cache every weight update for
        # no reason.
        if live_sourced:
            cache_salt = str(group.policy_version_at_start)
        else:
            cache_salt = None

        if env.requires_group_scoring:
            # Legacy-only route (a v1 env never group-scores) — addressed by row.
            permits = group.rollouts_to_schedule
            group.rollouts_to_schedule = 0
            await self.acquire(permits)
            task: asyncio.Task = asyncio.create_task(
                env.run_group(
                    client=client,
                    task_idx=group.task_idx,
                    model_name=model_name,
                    group_size=permits,
                    cache_salt=cache_salt,
                )
            )
        else:
            # A v1 env takes the task itself; the legacy bridge its dataset row.
            if group.task is not None:
                addressing = {"task_data": group.task.data.model_dump(mode="json")}
            else:
                addressing = {"task_idx": group.task_idx}
            permits = 1
            group.rollouts_to_schedule -= 1
            await self.acquire(permits)
            task = asyncio.create_task(
                env.run(
                    client=client,
                    model_name=model_name,
                    cache_salt=cache_salt,
                    **addressing,
                )
            )

        self.inflight[task] = InflightRollout(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            rollout_count=permits,
            client_config=client,
            eval_step=group.eval_step,
        )
        return True

    async def acquire(self, n: int) -> None:
        """Reserve ``n`` permits + rate-limit each one. Caller must precheck
        ``available_permits >= n``; this is not a blocking acquire."""
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            self.inflight_permits += 1

    def release(self, n: int) -> None:
        self.inflight_permits -= n

    async def handle_completed_rollout(self, task: asyncio.Task) -> None:
        """Emit every dispatched env-rollout exactly once to ``out_q``: a ``run``
        result as one episode, a legacy ``run_group`` result as ``rollout_count``
        single-trace episodes. Task exceptions synthesize ``rollout_count``
        error-marker episodes so the sink's count-to-``group_size`` finalization
        still triggers. Cancelled tasks (popped by ``drop_group``) raise
        ``CancelledError`` and are discarded — ``drop_group`` already emitted
        their markers.
        """
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_inflight_rollouts
        self.release(meta.rollout_count)
        group = self.groups.get(meta.group_id)

        is_synth_exception = False
        try:
            result = task.result()
            rollouts: list[Rollout] = result if isinstance(result, list) else [result]
            if not rollouts:
                raise RuntimeError("env run returned an empty episode (no traces)")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            task_idx = group.task_idx if group is not None else -1
            rollouts = [
                Rollout(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)))
                for _ in range(meta.rollout_count)
            ]
            for r in rollouts:
                r.capture_error(exc)
            is_synth_exception = True

        transient_policy_failure = (
            len(rollouts) == 1 and self._is_transient_policy_failure(rollouts[0])
        )
        transient_user_simulator_failure = (
            len(rollouts) == 1 and self._is_transient_user_simulator_failure(rollouts[0])
        )
        if (
            meta.rollout_count == 1
            and group is not None
            and meta.client_config is not None
            and len(rollouts) == 1
            and (
                transient_policy_failure
                or transient_user_simulator_failure
            )
        ):
            started, retries = self.transient_retries.get(meta.group_id, (monotonic(), 0))
            retry_limit = group.target_rollouts * TRANSIENT_RETRIES_PER_ROLLOUT
            if monotonic() - started < TRANSIENT_RETRY_WINDOW_S and retries < retry_limit:
                self.transient_retries[meta.group_id] = (started, retries + 1)
                identity = client_identity(meta.client_config)
                if not transient_user_simulator_failure:
                    # Policy transport failures are endpoint-local. Unpin and cool
                    # that endpoint before retrying. The user simulator is shared;
                    # moving the policy request cannot repair it, so retain the
                    # pinned client and retry only the missing group member.
                    self.unhealthy_clients[identity] = monotonic() + 60
                    group.pinned_client = None
                group.rollouts_to_schedule += 1
                if group.kind == "eval":
                    self.switch_mode(DispatcherMode.PREFER_EVAL, reason="an eval rollout is being retried")
                detail = "; ".join(
                    f"{error.type}: {error.message}" for error in rollouts[0].errors[-2:]
                )
                get_logger().warning(
                    f"Retrying transient rollout failure in group {meta.group_id} "
                    f"({meta.env_name})"
                    f"{' away from ' + identity[0] if not transient_user_simulator_failure else ''}: "
                    f"{detail}"
                )
                return
            self.transient_retries[meta.group_id] = (started, retry_limit)
            get_logger().warning(
                f"Transient retry limit reached in group {meta.group_id} ({meta.env_name})"
            )

        for r in rollouts:
            # A wall-clock harness timeout is a terminal rollout result. Do not
            # reschedule it: an earlier run retried one zero-turn timeout
            # indefinitely and held a batch open for more than 90 minutes. The empty
            # trajectory guard below turns it into an explicit failed member;
            # TrainSink then rejects the complete GRPO group and replenishes it.
            if not r.has_error and r.num_turns == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure (``has_error`` reads ``ok``)
                r.errors.append(vf.Error(type="EmptyTrajectory", message="Rollout returned with no trajectory steps"))
                r.ok = False
                get_logger().warning(
                    f"Empty trajectory in group {meta.group_id} ({meta.env_name}); "
                    f"task={r.task.data.idx}, stop={repr(r.stop_condition)}, "
                    f"user_done={getattr(r.state, 'user_done', None)}, "
                    f"target_turns={getattr(r.state, 'target_turns', None)}, "
                    f"client={repr(getattr(meta.client_config, 'base_url', None))}, "
                    f"calls={[(c.finish_reason, c.error.type if c.error else None) for c in r.calls]}"
                )
            if r.has_error:
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception and r.error is not None:
                    get_logger().warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — {r.error.type}: {r.error.message}"
                    )
        if meta.rollout_count == 1:
            # A ``run`` task: the whole result is one episode.
            await self.emit_episode(meta, group, rollouts)
        else:
            # A legacy ``run_group`` task: one single-trace episode per trace.
            for r in rollouts:
                await self.emit_episode(meta, group, [r])

    def _is_transient_policy_failure(self, rollout: Rollout) -> bool:
        error = rollout.error
        if error is None or error.type not in ("ProviderError", "HarnessError"):
            return False
        message = error.message.lower()
        failed_policy_call = any(
            getattr(call, "model", None) == self.policy.model_name
            and getattr(call, "error", None) is not None
            for call in getattr(rollout, "calls", [])
        )
        # A restarted vLLM process is healthy before it has the live LoRA.
        # Requests then fail with a policy-model 404. Do not emit those as
        # completed rollouts: unpin the endpoint and keep the GRPO group open
        # while the weight watcher restores the adapter.
        missing_policy_model = (
            ("404" in message or "not found" in message or "does not exist" in message)
            and "model" in message
        )
        if missing_policy_model:
            return True
        # vLLM Router returns this 503 when every worker is unhealthy or
        # circuit-open. Keep the group open while workers recover.
        if (
            error.type == "ProviderError"
            and "503" in message
            and ("no available workers" in message or "no_available_workers" in message)
            and failed_policy_call
        ):
            return True
        # vLLM can occasionally produce a NaN in response metadata while the
        # generated token sequence itself remains bounded. Starlette refuses
        # to serialize that response and vLLM returns this exact HTTP 400.
        # Retry the missing group member on the other policy endpoint instead
        # of discarding all 16 members of an otherwise valid GRPO group.
        response_metadata_nan = (
            error.type == "ProviderError"
            and "out of range float values are not json compliant" in message
            and "nan" in message
        )
        if response_metadata_nan:
            return True
        if not any(
            marker in message
            for marker in ("connection", "timed out", "timeout", "readtimeout")
        ):
            return False
        # A timeout without a failed policy call can be an outer episode
        # deadline, not a policy transport failure.
        return failed_policy_call

    @staticmethod
    def _is_transient_user_simulator_failure(rollout: Rollout) -> bool:
        """Retry a user-simulator network failure without counting a bad row."""
        error = rollout.error
        if error is None or error.type != "UserError":
            return False
        message = error.message.lower()
        return "user simulator failed" in message and any(
            marker in message
            for marker in ("connection", "timed out", "timeout", "readtimeout")
        )

    async def emit_episode(self, meta: InflightRollout, group: GroupState | None, rollouts: list[Rollout]) -> None:
        """Stamp prime-rl metadata onto one completed episode and put it on
        ``out_q``. Pops the group from ``self.groups`` once every owed episode
        has been emitted."""
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        if group is not None:
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start
            group.emitted += 1
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)
                self.transient_retries.pop(meta.group_id, None)

        for rollout in rollouts:
            rollout.kind = meta.kind
            rollout.env_name = meta.env_name
            rollout.group_id = meta.group_id
            rollout.policy_version = policy_version
            rollout.off_policy_steps = meta.off_policy_steps
            if meta.kind == "eval":
                assert eval_step is not None, "eval rollout missing eval_step"
                rollout.eval_step = eval_step
        if meta.kind == "train":
            self.queued_train_episodes += 1
        await self.out_q.put(rollouts)

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel remaining in-flight tasks for this group and emit a
        ``Cancelled`` marker for every rollout it still owes the sink
        (both in-flight and not-yet-scheduled). Returns the count for
        off-policy metrics."""
        group = self.groups.pop(group_id, None)
        self.transient_retries.pop(group_id, None)
        task_idx = group.task_idx if group is not None else -1

        # Sync claim phase: pop matching tasks from ``self.inflight`` and
        # release their permits in one non-yielding sweep. After this loop
        # the dropped tasks are no longer reachable from ``self.inflight``,
        # so ``handle_completed_rollout``'s existing None-guard makes the
        # subsequent async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release(meta.rollout_count)
            claimed.append((task, meta))

        tasks_to_cancel = [task for task, _ in claimed]
        inflight_cancelled = sum(meta.rollout_count for _, meta in claimed)
        last_meta: InflightRollout | None = claimed[-1][1] if claimed else None
        for _, meta in claimed:
            for _ in range(meta.rollout_count):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    ok=False,
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_episode(meta, group, [trace])

        # For non-group-scoring envs, the group may have rollouts that
        # were never dispatched (``rollouts_to_schedule > 0``). Emit
        # markers for those too so the sink hits ``target_rollouts``
        #
        # ``last_meta`` can be ``None`` if the only inflight task for this
        # group completed naturally between ``on_version_pending``'s snapshot
        # and us reaching it — synthesize a stand-in from the group state
        unscheduled_cancelled = 0
        if group is not None and group.rollouts_to_schedule > 0:
            fallback_meta = last_meta or InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                rollout_count=1,
                eval_step=group.eval_step,
            )
            unscheduled_cancelled = group.rollouts_to_schedule
            for _ in range(unscheduled_cancelled):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    ok=False,
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_episode(fallback_meta, group, [trace])

        cancelled = inflight_cancelled + unscheduled_cancelled
        if cancelled > 0:
            meta_for_log = last_meta or (
                InflightRollout(
                    kind=group.kind,
                    env_name=group.env_name,
                    group_id=group_id,
                    policy_version=group.policy_version_at_start if group else 0,
                    rollout_count=1,
                    eval_step=group.eval_step,
                )
                if group is not None
                else None
            )
            if meta_for_log is not None:
                self.metrics.record_cancellation(kind=meta_for_log.kind, env_name=meta_for_log.env_name, n=cancelled)
                get_logger().debug(
                    f"drain {meta_for_log.kind} | group={str(group_id)[:8]} env={meta_for_log.env_name} | "
                    f"cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled_cancelled})"
                )

        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts. Used on shutdown — doesn't emit
        markers since the sinks are being torn down anyway."""
        for meta in self.inflight.values():
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name, n=meta.rollout_count)
            self.release(meta.rollout_count)
        tasks = list(self.inflight.keys())
        self.inflight.clear()
        self.groups.clear()
        self.transient_retries.clear()
        if tasks:
            await safe_cancel_all(tasks)

    async def cancel_inflight_train_rollouts(self) -> int:
        """Cancel in-flight train rollouts, leaving eval alone. Used by the
        orchestrator at ``max_steps`` so triggered eval can still complete
        through the pipeline while wasted train inference is short-circuited."""
        train_tasks: list[asyncio.Task] = []
        train_group_ids: set[uuid.UUID] = set()
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.kind != "train":
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name, n=meta.rollout_count)
            cancelled += meta.rollout_count
            train_tasks.append(task)
            train_group_ids.add(meta.group_id)
        for gid in train_group_ids:
            self.groups.pop(gid, None)
            self.transient_retries.pop(gid, None)
        if train_tasks:
            await safe_cancel_all(train_tasks)
        return cancelled

    # ── metrics ────────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        """Instantaneous, read-only gauges sampled by the periodic logger."""
        return {
            "dispatcher/inflight_train": float(self.inflight_train_count),
            "dispatcher/inflight_eval": float(self.inflight_eval_count),
            "dispatcher/queued/eval": float(self.queued_eval_examples),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
        }
