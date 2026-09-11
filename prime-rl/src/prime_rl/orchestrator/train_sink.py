"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   and untrainable rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands the trainable
   survivors to the env algorithm's ``finalize_group`` (advantages +
   per-sample wire stamping), runs the pre-batch filter pass.
3. ``process_batch`` — applies post-batch filter annotations and assembles
   the trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` takes one episode (``list[Rollout]``) and returns
``TrainBatch | None``; group accounting counts episodes, never loose traces.
I/O concerns (ship to trainer, save_rollouts, monitor.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, TrainBatch
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


# Transient policy-endpoint failures are retried in the dispatcher and do not
# reach the sink. Repeated failures that do reach this point usually mean an
# environment or judge fault. Stop before the orchestrator can spend a full
# batch on unusable groups. The arrival traces are already persisted by the
# orchestrator, so the failure remains diagnosable after this trips.
MAX_CONSECUTIVE_ERRORED_GROUPS = 8


def payload_tokens(rollout: Rollout) -> int:
    """Token cost of the rollout's trainer-bound payload — the samples built by
    ``process_rollout``. This is what actually ships: forked traces can drop
    branches with no trainable tokens, so ``Trace.num_total_tokens`` (which sums
    over all branches) may overcount. For linear traces the two agree.

    Zero-payload rollouts (no trainable samples at all) fall back to the trace
    total so they still advance token batching — a degenerate all-zero-payload
    stream then ships empty batches and trips the orchestrator's
    consecutive-empty-batch abort instead of stalling the readiness check."""
    return sum(len(sample.token_ids) for sample in rollout.samples) or rollout.num_total_tokens


def _refusal_score(rollout: Rollout) -> int | float | None:
    """Read the decensor refusal score without changing the rollout record."""
    info = getattr(rollout, "info", None)
    if not isinstance(info, dict):
        return None
    decensor = info.get("decensor")
    if not isinstance(decensor, dict):
        return None
    score = decensor.get("refusal_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return score


def should_skip_all_refusal_group(group: list[Rollout], enabled: bool) -> bool:
    """Return True only for an enabled, fully scored, all-refusal group."""
    return (
        enabled
        and bool(group)
        and not any(getattr(rollout, "has_error", False) for rollout in group)
        and all(_refusal_score(rollout) == 9 for rollout in group)
    )


def _skip_refusal_group(group: list[Rollout]) -> None:
    """Remove training samples while retaining rollout observations."""
    for rollout in group:
        rollout.samples = []
        rollout.is_filtered = True
        rollout.filter_results = {"all_refusal_group": True}


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        training_policy_version: int,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        # Batch ``step`` trains policy v{step-1}. This value advances only
        # after that batch is shipped.
        self.training_policy_version = training_policy_version
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + filtered + survivors).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Episodes arrived per group — the finalization count (an episode may
        # add several traces to ``pending_groups`` but counts once here).
        self.pending_group_episodes: dict[uuid.UUID, int] = defaultdict(int)
        self.abandoned_groups: set[uuid.UUID] = set()
        self.pending_batch: list[Rollout] = []
        # Running payload-token total of ``pending_batch`` (token-batched
        # runs), kept in sync on append/pop so the readiness check never
        # re-sums per arrival.
        self.pending_tokens: int = 0
        self.consecutive_errored_groups: int = 0

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}
        self.all_refusal_groups_skipped = 0

    def _is_stale(self, rollout: Rollout) -> bool:
        sampler = self.train_envs.get(rollout.env_name).sampler
        return (
            sampler.samples_from_live_policy
            and self.training_policy_version - rollout.policy_version > self.config.max_off_policy_steps
        )

    @staticmethod
    def _mark_off_policy(rollouts: list[Rollout]) -> None:
        for rollout in rollouts:
            rollout.samples = []
            rollout.is_filtered = True
            rollout.filter_results = {"max_off_policy_steps": True}

    def _drop_stale_pending(self) -> int:
        """Remove complete stale groups already queued for a later batch.

        The group is the GRPO baseline unit, so if any member is stale the
        whole group is removed. The normal rollout-budget update replaces the
        removed rollouts before the next trainer batch can become ready.
        """
        stale_group_ids = {r.group_id for r in self.pending_batch if self._is_stale(r)}
        if not stale_group_ids:
            return 0
        stale = [r for r in self.pending_batch if r.group_id in stale_group_ids]
        self.pending_batch = [r for r in self.pending_batch if r.group_id not in stale_group_ids]
        self._mark_off_policy(stale)
        if self.token_batch_size is not None:
            self.pending_tokens = sum(payload_tokens(r) for r in self.pending_batch)
        get_logger().warning(
            f"Dropped {len(stale)} completed rollout(s) from {len(stale_group_ids)} group(s) "
            f"past max_off_policy_steps={self.config.max_off_policy_steps}; replenishing the batch"
        )
        return len(stale)

    def advance_training_policy_version(self, version: int) -> int:
        """Advance the policy that the next trainer batch will update.

        This also rejects completed overflow left in ``pending_batch``. Partial
        groups are checked when they finish, before they can enter the batch.
        """
        if version < self.training_policy_version:
            raise ValueError(
                f"training policy version moved backwards: {self.training_policy_version} -> {version}"
            )
        self.training_policy_version = version
        return self._drop_stale_pending()

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the train batch — counts only
        ``pending_batch`` (survivors of finalized groups, queued for the
        trainer), so it's an honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        """Episodes that have arrived but sit in not-yet-complete groups
        (non-group-scoring envs) — buffered in the sink ahead of the batch."""
        return sum(
            self.pending_group_episodes.get(group_id, 0)
            for group_id, rollouts in self.pending_groups.items()
            if rollouts and not self.train_envs.get(rollouts[0].env_name).requires_group_scoring
        )

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, episode: list[Rollout]) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode[0].group_id
        if group_id in self.abandoned_groups:
            return None
        env_name = episode[0].env_name
        for rollout in episode:
            await self.process_rollout(rollout)
        # A policy boundary can invalidate this group while tokenization runs
        # in a worker thread. Check again before mutating group state.
        if group_id in self.abandoned_groups:
            return None
        self.pending_groups[group_id].extend(episode)
        self.pending_group_episodes[group_id] += 1
        if self.pending_group_episodes[group_id] < self.group_size_for(env_name):
            return None
        await self.process_group(group_id)
        # ``pending_batch`` only grows on group finalization, so readiness is
        # only re-checked here — the window of a shipped batch then always
        # contains at least the group that finalized it.
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        if ready:
            return self.process_batch()
        return None

    def abandon_groups(self, group_ids: set[uuid.UUID]) -> int:
        """Discard partial groups that will restart under a new policy.

        These are scheduling cancellations, not rollout failures. They must not
        enter the errored-group path or contribute to batch failure metrics.
        """
        self.abandoned_groups.update(group_ids)
        discarded = 0
        for group_id in group_ids:
            discarded += self.pending_group_episodes.pop(group_id, 0)
            self.pending_groups.pop(group_id, None)
        return discarded

    async def process_rollout(self, rollout: Rollout) -> None:
        """Build training samples from the rollout's Trace (one per branch), walking the
        message graph. Training is renderer-only across all modes (RL/OPD student, SFT teacher),
        so every node already carries its tokens. Errored rollouts are dropped at the group
        level, so skip them here; untrainable traces never become training data."""
        if rollout.has_error or not rollout.trainable:
            return
        samples = await asyncio.to_thread(
            trace_to_samples,
            rollout,
            env_name=rollout.env_name,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
        )
        rollout.samples = samples or []
        # Arrival phase: rollout-local scoring (raw reward, echo observation
        # weighting, opd/opsd reference logprobs) runs as soon as the rollout is
        # tokenized — before its group is complete.
        await self.train_envs.get(rollout.env_name).algorithm.finalize_rollout(rollout)

    async def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one complete GRPO group: skip the complete group when
        any member failed, is oversized, or is untrainable; assign advantages;
        run pre-batch filters; append clean survivors to ``pending_batch``."""
        group = self.pending_groups.pop(group_id, [])
        self.pending_group_episodes.pop(group_id, None)
        if not group:
            return
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        for r in group:
            self.pending_rollouts.append(r)
        env_name = group[0].env_name
        task_idx = group[0].task.data.idx
        if any(self._is_stale(r) for r in group):
            self._mark_off_policy(group)
            get_logger().warning(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"dropped {len(group)} rollout(s) past max_off_policy_steps="
                f"{self.config.max_off_policy_steps}; replenishing the batch"
            )
            return
        survivors = [r for r in group if not r.has_error]
        num_errored = len(group) - len(survivors)
        env = self.train_envs.get(env_name)
        if num_errored > 0:
            # A GRPO group is the advantage baseline even when the environment
            # scores each rollout independently. Training on its survivors
            # changes that baseline and lets infrastructure failures alter the
            # optimization signal. Reject and replenish the complete group.
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"errored_group": True}
            self.consecutive_errored_groups += 1
            get_logger().warning(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped complete group; "
                f"consecutive errored groups={self.consecutive_errored_groups}/"
                f"{MAX_CONSECUTIVE_ERRORED_GROUPS}"
            )
            if self.consecutive_errored_groups >= MAX_CONSECUTIVE_ERRORED_GROUPS:
                raise RuntimeError(
                    f"{self.consecutive_errored_groups} consecutive rollout groups contained errors; "
                    "stopping collection before an unusable train batch can be built"
                )
            return
        self.consecutive_errored_groups = 0

        # Never assign a full-trajectory reward to a prefix that the trainer
        # silently truncates. One oversized member makes the complete GRPO
        # baseline unsafe.
        oversized = [
            r
            for r in survivors
            if any(len(sample.token_ids) > self.config.seq_len for sample in r.samples)
        ]
        if oversized:
            longest = max(
                len(sample.token_ids)
                for r in oversized
                for sample in r.samples
            )
            get_logger().warning(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"dropped training payload: {len(oversized)} rollout(s) exceed "
                f"trainer seq_len {self.config.seq_len} (longest={longest})"
            )
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"trainer_seq_len_group": True}
            return

        # Untrainable traces carry no samples and must not skew the group baseline.
        if any(not r.trainable for r in survivors):
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"untrainable_group": True}
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | dropped complete group: untrainable member"
            )
            return

        # Keep the group in pending_rollouts, but send no optimizer samples.
        if should_skip_all_refusal_group(group, self.config.skip_all_refusal_groups):
            self.all_refusal_groups_skipped += 1
            _skip_refusal_group(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | skipped all-refusal group"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(survivors)

        # ``Rollout.is_trainable`` becomes meaningful only after advantages
        # are assigned. A uniform-reward GRPO group has all-zero advantages:
        # shipping it consumes one of the 64 batch groups but produces no
        # gradient. Reject and replenish the complete group.
        if not any(r.is_trainable for r in survivors):
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"zero_advantage_group": True}
            get_logger().warning(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | dropped complete zero-advantage group"
            )
            return

        # The env has a single sampling temperature; fan it out per token
        # (context tokens are masked out, so their temperature is don't-care).
        temperature = env.sampling_args["temperature"]
        for r in survivors:
            for sample in r.samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        if self.pre_filters:
            apply_filters(self.pre_filters, survivors)
        filtered_by_name: dict[str, int] = {}
        num_filtered = 0
        for r in survivors:
            self.pre_filter_seen += 1
            if r.is_filtered:
                self.pre_filter_dropped += 1
                num_filtered += 1
                for name, hit in r.filter_results.items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                        filtered_by_name[name] = filtered_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean
            r.filter_results = {}
            r.is_filtered = False
            self.pending_batch.append(r)
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)
        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"rollouts={len(group)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    def process_batch(self) -> TrainBatch | None:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), apply post-batch filter annotations, and assemble the
        trainer-bound ``TrainingSample`` list. Overflow stays for the next
        batch."""
        # Delivery invariant: even completed overflow that crossed a batch
        # boundary cannot become trainer samples.
        self._drop_stale_pending()
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        if not ready:
            return None
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += payload_tokens(r)
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_tokens -= running

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample. Filtered rollouts don't ship.
        samples: list[TrainingSample] = [sample for r in cohort if not r.is_filtered for sample in r.samples]

        # ``rollouts`` is the observation window — every rollout of every group finalized since the
        # last ship (errored + filtered + survivors) — while ``samples`` is the shipped cohort's
        # trainable payload. ``rollouts.effective`` / ``rollouts.metrics`` derive the clean subset +
        # metric views on demand. Reset the window only when the batch actually ships (non-empty
        # samples) — an empty batch is dropped unlogged by the orchestrator, so keep accumulating its
        # finalized groups (and any overflow) into the next shipped batch's window.
        rollouts = self.pending_rollouts
        if samples:
            self.pending_rollouts = TrainRollouts()
        return TrainBatch(rollouts=rollouts, samples=samples)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
        self.all_refusal_groups_skipped = 0
