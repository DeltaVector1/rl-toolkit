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
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.dispatch_results import DispatchFailure, GroupCancellation
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


_PRE_BATCH_REJECTION_REASONS = (
    "stale_policy",
    "errored_group",
    "dispatch_failure_group",
    "trainer_seq_len_group",
    "untrainable_group",
    "zero_advantage_group",
)


@dataclass
class PreBatchGroupRecord:
    """Numeric outcome for one finalized group, without retaining traces."""

    clean: bool
    rejected_rollouts: int
    rejection_reason: str | None = None
    policy_invalid_rollouts: int = 0


@dataclass
class PreBatchProgress:
    """Bounded, prompt-free collection counters for the next train batch."""

    complete_groups: int = 0
    clean_groups: int = 0
    rejected_groups: int = 0
    rejected_rollouts: int = 0
    policy_invalid_rollouts: int = 0
    rejection_reasons: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _PRE_BATCH_REJECTION_REASONS}
    )

    def _add_record(self, record: PreBatchGroupRecord) -> None:
        self.complete_groups += 1
        self.policy_invalid_rollouts += record.policy_invalid_rollouts
        if record.clean:
            self.clean_groups += 1
            return
        self.rejected_groups += 1
        self.rejected_rollouts += record.rejected_rollouts
        reason = record.rejection_reason or "unknown"
        if reason not in self.rejection_reasons:
            # Keep the schema bounded if a future caller adds a reason.
            reason = "unknown"
            self.rejection_reasons.setdefault(reason, 0)
        self.rejection_reasons[reason] += 1

    def _remove_record(self, record: PreBatchGroupRecord) -> None:
        self.complete_groups -= 1
        self.policy_invalid_rollouts -= record.policy_invalid_rollouts
        if record.clean:
            self.clean_groups -= 1
            return
        self.rejected_groups -= 1
        self.rejected_rollouts -= record.rejected_rollouts
        reason = record.rejection_reason or "unknown"
        if reason not in self.rejection_reasons:
            reason = "unknown"
            self.rejection_reasons.setdefault(reason, 0)
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) - 1

    def record_group(
        self,
        group: list[Rollout],
        *,
        rejection_reason: str | None = None,
        policy_invalid_rollouts: int = 0,
        rejected_rollouts: int | None = None,
    ) -> PreBatchGroupRecord:
        record = PreBatchGroupRecord(
            clean=rejection_reason is None,
            rejected_rollouts=(
                0
                if rejection_reason is None
                else len(group) if rejected_rollouts is None else max(0, int(rejected_rollouts))
            ),
            rejection_reason=rejection_reason,
            policy_invalid_rollouts=policy_invalid_rollouts,
        )
        self._add_record(record)
        return record

    @staticmethod
    def stale_record(rejected_rollouts: int) -> PreBatchGroupRecord:
        return PreBatchGroupRecord(
            clean=False,
            rejected_rollouts=max(0, int(rejected_rollouts)),
            rejection_reason="stale_policy",
        )

    def reclassify_group(
        self,
        previous: PreBatchGroupRecord,
        replacement: PreBatchGroupRecord,
    ) -> None:
        self._remove_record(previous)
        self._add_record(replacement)

    def reset_for_batch(
        self,
        overflow: tuple[PreBatchGroupRecord, ...] | list[PreBatchGroupRecord] = (),
    ) -> None:
        """Reset counters at shipment and carry accepted overflow forward."""
        self.complete_groups = 0
        self.clean_groups = 0
        self.rejected_groups = 0
        self.rejected_rollouts = 0
        self.policy_invalid_rollouts = 0
        self.rejection_reasons = {
            name: 0 for name in _PRE_BATCH_REJECTION_REASONS
        }
        for record in overflow:
            self._add_record(record)

    def log_fields(
        self,
        *,
        accepted_buffer: int,
        accepted_target: int,
        accepted_unit: str,
        accepted_buffer_tokens: int,
        partial_groups: int,
        partial_rollouts: int,
        oldest_partial_age_seconds: float | None,
    ) -> str:
        fields = [
            f"complete_groups={self.complete_groups}",
            f"clean_groups={self.clean_groups}",
            f"rejected_groups={self.rejected_groups}",
            f"rejected_rollouts={self.rejected_rollouts}",
            f"policy_invalid_rollouts={self.policy_invalid_rollouts}",
            f"accepted_buffer={accepted_buffer}",
            f"accepted_target={accepted_target}",
            f"accepted_unit={accepted_unit}",
            f"accepted_buffer_tokens={accepted_buffer_tokens}",
            f"partial_groups={partial_groups}",
            f"partial_rollouts={partial_rollouts}",
        ]
        if oldest_partial_age_seconds is not None:
            fields.append(f"oldest_partial_age_seconds={oldest_partial_age_seconds:.3f}")
        fields.extend(
            f"reject_{reason}={self.rejection_reasons.get(reason, 0)}"
            for reason in _PRE_BATCH_REJECTION_REASONS
        )
        return " ".join(fields)


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
        # Monotonic first-arrival times make the oldest partial-group age
        # authoritative without scanning trace files or retaining content.
        self.pending_group_started_at: dict[uuid.UUID, float] = {}
        self.pending_group_failures: dict[uuid.UUID, list[DispatchFailure]] = defaultdict(list)
        self.pending_group_cancellations: dict[uuid.UUID, GroupCancellation] = {}
        self.abandoned_groups: set[uuid.UUID] = set()
        self.pending_batch: list[Rollout] = []
        # Running payload-token total of ``pending_batch`` (token-batched
        # runs), kept in sync on append/pop so the readiness check never
        # re-sums per arrival.
        self.pending_tokens: int = 0
        self.consecutive_errored_groups: int = 0
        self.pre_batch_progress = PreBatchProgress()
        # Numeric outcomes for finalized groups still represented in the
        # pending buffer. These records let shipment reset carry overflow
        # without retaining prompts, responses, or rollout objects.
        self.pending_group_progress: dict[uuid.UUID, PreBatchGroupRecord] = {}

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
        stale_counts: dict[uuid.UUID, int] = defaultdict(int)
        for rollout in stale:
            stale_counts[rollout.group_id] += 1
        telemetry_changed = False
        for group_id, count in stale_counts.items():
            previous = self.pending_group_progress.pop(group_id, None)
            if previous is None:
                continue
            replacement = self.pre_batch_progress.stale_record(count)
            replacement = PreBatchGroupRecord(
                clean=replacement.clean,
                rejected_rollouts=replacement.rejected_rollouts,
                rejection_reason=replacement.rejection_reason,
                policy_invalid_rollouts=previous.policy_invalid_rollouts,
            )
            self.pre_batch_progress.reclassify_group(previous, replacement)
            telemetry_changed = True
        if self.token_batch_size is not None:
            self.pending_tokens = sum(payload_tokens(r) for r in self.pending_batch)
        if telemetry_changed:
            self._pre_batch_log()
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
        count = 0
        group_ids = (
            set(self.pending_groups)
            | set(self.pending_group_failures)
            | set(self.pending_group_cancellations)
        )
        for group_id in group_ids:
            rollouts = self.pending_groups.get(group_id, [])
            if rollouts:
                env_name = rollouts[0].env_name
            elif self.pending_group_failures.get(group_id):
                env_name = self.pending_group_failures[group_id][0].env_name
            else:
                env_name = self.pending_group_cancellations[group_id].env_name
            if not self.train_envs.get(env_name).requires_group_scoring:
                count += self.pending_group_episodes.get(group_id, 0)
        return count

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    def _partial_progress(self) -> tuple[int, int, float | None]:
        group_ids = (
            set(self.pending_groups)
            | set(self.pending_group_failures)
            | set(self.pending_group_cancellations)
        )
        partial_groups = sum(self.pending_group_episodes.get(group_id, 0) > 0 for group_id in group_ids)
        partial_rollouts = sum(self.pending_group_episodes.get(group_id, 0) for group_id in group_ids)
        started = [
            self.pending_group_started_at[group_id]
            for group_id in group_ids
            if self.pending_group_episodes.get(group_id, 0) and group_id in self.pending_group_started_at
        ]
        oldest_age = max(0.0, time.monotonic() - min(started)) if started else None
        return partial_groups, partial_rollouts, oldest_age

    def pre_batch_telemetry_fields(self, *, inflight_train: int | None = None) -> str:
        """Return bounded numeric fields for group and batch accumulation."""
        if self.batch_size is not None:
            accepted_buffer = len(self.pending_batch)
            accepted_target = self.batch_size
            accepted_unit = "rollouts"
        else:
            accepted_buffer = self.pending_tokens
            accepted_target = self.token_batch_size or 0
            accepted_unit = "tokens"
        partial_groups, partial_rollouts, oldest_age = self._partial_progress()
        fields = self.pre_batch_progress.log_fields(
            accepted_buffer=accepted_buffer,
            accepted_target=accepted_target,
            accepted_unit=accepted_unit,
            accepted_buffer_tokens=self.pending_tokens,
            partial_groups=partial_groups,
            partial_rollouts=partial_rollouts,
            oldest_partial_age_seconds=oldest_age,
        )
        if inflight_train is not None:
            fields += f" inflight_train={max(0, int(inflight_train))}"
        return fields

    def _pre_batch_log(self) -> None:
        """Emit one numeric-only INFO record for one finalized group."""
        fields = self.pre_batch_telemetry_fields()
        get_logger().info(f"PIPELINE_TELEMETRY {fields}")

    @staticmethod
    def _policy_invalid_rollouts(group: list[Rollout]) -> int:
        """Count only fixed error-type labels; never log provider messages."""
        count = 0
        for rollout in group:
            errors = getattr(rollout, "errors", ()) or ()
            for error in errors:
                error_type = str(getattr(error, "type", "")).casefold().replace("-", "_")
                if error_type in {
                    "policy_invalid",
                    "invalid_policy",
                    "policyinvalid",
                    "invalidpolicy",
                    "policy_validation",
                    "policyvalidation",
                }:
                    count += 1
                    break
        return count

    def _record_group(
        self,
        group_id: uuid.UUID,
        group: list[Rollout],
        rejection_reason: str | None = None,
        *,
        rejected_rollouts: int | None = None,
    ) -> None:
        policy_invalid_rollouts = self._policy_invalid_rollouts(group)
        record = self.pre_batch_progress.record_group(
            group,
            rejection_reason=rejection_reason,
            policy_invalid_rollouts=policy_invalid_rollouts,
            rejected_rollouts=rejected_rollouts,
        )
        self.pending_group_progress[group_id] = record
        self._pre_batch_log()

    async def add(self, episode: list[Rollout]) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode[0].group_id
        if group_id in self.abandoned_groups:
            return None
        env_name = episode[0].env_name
        self.pending_group_started_at.setdefault(group_id, time.monotonic())
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

    async def fail(self, failure: DispatchFailure) -> TrainBatch | None:
        """Count an attempt that returned no rollout without making a fake trace."""
        if failure.kind != "train":
            raise ValueError(f"TrainSink cannot process an {failure.kind} dispatch failure")
        group_id = failure.group_id
        self.pending_group_started_at.setdefault(group_id, time.monotonic())
        self.pending_group_failures[group_id].append(failure)
        self.pending_group_episodes[group_id] += failure.count
        if self.pending_group_episodes[group_id] >= self.group_size_for(failure.env_name):
            await self.process_group(group_id)
        return self._ready_batch()

    async def cancel(self, cancellation: GroupCancellation) -> TrainBatch | None:
        """Count cancelled group members without making fake error traces."""
        if cancellation.kind != "train":
            raise ValueError(f"TrainSink cannot process an {cancellation.kind} group cancellation")
        group_id = cancellation.group_id
        self.pending_group_started_at.setdefault(group_id, time.monotonic())
        if group_id in self.pending_group_cancellations:
            raise ValueError(f"duplicate cancellation for group {group_id}")
        self.pending_group_cancellations[group_id] = cancellation
        self.pending_group_episodes[group_id] += cancellation.count
        if self.pending_group_episodes[group_id] >= self.group_size_for(cancellation.env_name):
            await self.process_group(group_id)
        return self._ready_batch()

    def _ready_batch(self) -> TrainBatch | None:
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        return self.process_batch() if ready else None

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
            self.pending_group_started_at.pop(group_id, None)
            self.pending_group_failures.pop(group_id, None)
            self.pending_group_cancellations.pop(group_id, None)
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
        failures = self.pending_group_failures.pop(group_id, [])
        cancellation = self.pending_group_cancellations.pop(group_id, None)
        cancelled = cancellation.count if cancellation is not None else 0
        self.pending_group_episodes.pop(group_id, None)
        self.pending_group_started_at.pop(group_id, None)
        if not group and not failures and not cancelled:
            return
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        for r in group:
            self.pending_rollouts.append(r)
        if group:
            env_name = group[0].env_name
            task_idx = group[0].task.data.idx
        elif failures:
            env_name = failures[0].env_name
            task_idx = failures[0].task_idx
        else:
            env_name = cancellation.env_name
            task_idx = -1
        if failures or cancelled:
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"dispatch_failure_group": True}
            detail = ", ".join(f"{f.error.type}={f.count}" for f in failures)
            if cancelled:
                detail = f"{detail}, cancelled={cancelled}" if detail else f"cancelled={cancelled}"
            get_logger().warning(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"dispatch result: {detail}; dropped complete group"
            )
            if failures:
                self.consecutive_errored_groups += 1
            self._record_group(
                group_id,
                group,
                "dispatch_failure_group",
                rejected_rollouts=len(group) + sum(f.count for f in failures) + cancelled,
            )
            if failures:
                if self.consecutive_errored_groups >= MAX_CONSECUTIVE_ERRORED_GROUPS:
                    raise RuntimeError(
                        f"{self.consecutive_errored_groups} consecutive rollout groups contained dispatch failures"
                    )
            return
        if any(self._is_stale(r) for r in group):
            self._mark_off_policy(group)
            self._record_group(group_id, group, "stale_policy")
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
            self._record_group(group_id, group, "errored_group")
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
            self._record_group(group_id, group, "trainer_seq_len_group")
            return

        # Untrainable traces carry no samples and must not skew the group baseline.
        if any(not r.trainable for r in survivors):
            for r in group:
                r.samples = []
                r.is_filtered = True
                r.filter_results = {"untrainable_group": True}
            self._record_group(group_id, group, "untrainable_group")
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
            self._record_group(group_id, group, "zero_advantage_group")
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
        self._record_group(group_id, group)

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
            overflow_group_ids = {rollout.group_id for rollout in self.pending_batch}
            overflow_records = tuple(
                self.pending_group_progress[group_id]
                for group_id in overflow_group_ids
                if group_id in self.pending_group_progress
            )
            self.pending_group_progress = {
                group_id: record
                for group_id, record in self.pending_group_progress.items()
                if group_id in overflow_group_ids
            }
            self.pre_batch_progress.reset_for_batch(overflow_records)
            self.pending_rollouts = TrainRollouts()
            self._pre_batch_log()
        return TrainBatch(rollouts=rollouts, samples=samples)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
        self.all_refusal_groups_skipped = 0
