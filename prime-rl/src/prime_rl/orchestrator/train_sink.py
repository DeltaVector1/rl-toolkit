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
import math
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


def train_batch_ratio_quotas(
    batch_size: int, env_specs: list[tuple[str, float, int]]
) -> tuple[dict[str, int], int]:
    if not env_specs:
        raise ValueError("keep_train_batch_ratios needs at least one training environment")
    names = [name for name, _, _ in env_specs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("keep_train_batch_ratios needs unique non-empty environment names")
    group_size = env_specs[0][2]
    if any(size != group_size for _, _, size in env_specs):
        raise ValueError("keep_train_batch_ratios requires equal group sizes")
    if group_size < 1 or batch_size % group_size:
        raise ValueError("keep_train_batch_ratios requires batch_size divisible by group_size")
    ratios = [ratio for _, ratio, _ in env_specs]
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("keep_train_batch_ratios requires finite positive ratios")
    group_slots = batch_size // group_size
    raw_groups = [group_slots * ratio / math.fsum(ratios) for ratio in ratios]
    group_quotas = [math.floor(value) for value in raw_groups]
    order = sorted(
        range(len(env_specs)),
        key=lambda index: (-(raw_groups[index] - group_quotas[index]), index),
    )
    for index in order[: group_slots - sum(group_quotas)]:
        group_quotas[index] += 1
    if any(quota == 0 for quota in group_quotas):
        raise ValueError("keep_train_batch_ratios batch is too small for every environment")
    return (
        {
            name: group_quotas[index] * group_size
            for index, (name, _, _) in enumerate(env_specs)
        },
        group_size,
    )


def partition_train_batch(
    pending_batch: list[Rollout],
    batch_size: int,
    rollout_quotas: dict[str, int] | None = None,
    group_size: int | None = None,
) -> tuple[list[Rollout], list[Rollout], dict[str, dict[str, int]]]:
    if rollout_quotas is None:
        return pending_batch[:batch_size], pending_batch[batch_size:], {}
    if group_size is None:
        raise RuntimeError("keep_train_batch_ratios is missing group_size")
    counts = {name: 0 for name in rollout_quotas}
    cohort: list[Rollout] = []
    carry: list[Rollout] = []
    for start in range(0, len(pending_batch), group_size):
        group = pending_batch[start : start + group_size]
        first = group[0]
        env_name = getattr(first, "env_name", None)
        group_id = getattr(first, "group_id", None)
        if (
            len(group) != group_size
            or env_name not in rollout_quotas
            or group_id is None
            or any(
                getattr(rollout, "env_name", None) != env_name
                or getattr(rollout, "group_id", None) != group_id
                for rollout in group
            )
        ):
            raise RuntimeError("keep_train_batch_ratios requires exactly one rollout trace per group member and valid contiguous group identity")
        target = cohort if counts[env_name] < rollout_quotas[env_name] else carry
        target.extend(group)
        counts[env_name] += group_size
    state = {
        name: {
            "accepted": min(count, rollout_quotas[name]),
            "carry": max(0, count - rollout_quotas[name]),
            "wait": max(0, rollout_quotas[name] - count),
        }
        for name, count in counts.items()
    }
    return cohort, carry, state


def reject_ratio_group_after_post_filters(post_filters: list[RolloutFilter], group: list[Rollout]) -> bool:
    apply_filters(post_filters, group)
    if not any(rollout.is_filtered for rollout in group):
        return False
    for rollout in group:
        rollout.is_filtered = True
        rollout.filter_results.setdefault("ratio_group_member_filtered", True)
    return True


def _valid_refusal_score(score: int | float | None) -> bool:
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    if isinstance(score, int):
        return 1 <= score <= 9
    return math.isfinite(score) and 1 <= score <= 9


def _refusal_score(rollout: Rollout) -> int | float | None:
    """Read the decensor refusal score without changing the rollout record."""
    info = getattr(rollout, "info", None)
    if not isinstance(info, dict):
        return None
    decensor = info.get("decensor")
    if not isinstance(decensor, dict):
        return None
    score = decensor.get("refusal_score")
    return score if _valid_refusal_score(score) else None


_refusal_bands: tuple[tuple[str, int, int], ...] = (
    ("1-2", 1, 2),
    ("3-6", 3, 6),
    ("7-9", 7, 9),
)
_advantage_signs = ("positive", "negative", "zero")


def should_skip_strong_refusal_group(
    group: list[Rollout], skip_threshold: int | float | None
) -> bool:
    """Return True for a fully scored group at or above the threshold."""
    return (
        skip_threshold is not None
        and not isinstance(skip_threshold, bool)
        and isinstance(skip_threshold, (int, float))
        and (isinstance(skip_threshold, int) or math.isfinite(skip_threshold))
        and bool(group)
        and not any(getattr(rollout, "has_error", False) for rollout in group)
        and all(
            (score := _refusal_score(rollout)) is not None
            and score >= skip_threshold
            for rollout in group
        )
    )


def _skip_strong_refusal_group(group: list[Rollout]) -> None:
    """Remove training samples while retaining rollout observations."""
    for rollout in group:
        rollout.samples = []
        rollout.is_filtered = True
        rollout.filter_results = {"strong_refusal_group": True}


_usable_response_metrics = {
    "visible_answer_is_clear": 1,
    "reasoning_trace_is_clear": 1,
    "reasoning_has_safety_or_policy_discussion": 0,
    "meta_commentary": 0,
}


def _decensor_details(rollout: Rollout) -> dict:
    info = getattr(rollout, "info", None)
    if not isinstance(info, dict) or not isinstance(info.get("decensor"), dict):
        raise RuntimeError(
            "require_usable_response_in_group needs rollout.info.decensor scoring details"
        )
    return info["decensor"]


def is_usable_response(rollout: Rollout) -> bool:
    details = _decensor_details(rollout)
    if "response_rejected_because" not in details:
        raise RuntimeError(
            "require_usable_response_in_group needs decensor metric "
            "'response_rejected_because'"
        )
    if details["response_rejected_because"] is not None:
        return False
    refusal_score = details.get("refusal_score")
    if not _valid_refusal_score(refusal_score):
        raise RuntimeError(
            "require_usable_response_in_group needs refusal_score in [1, 9]; "
            f"got {refusal_score!r}"
        )
    for name in _usable_response_metrics:
        value = details.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
            raise RuntimeError(
                f"require_usable_response_in_group needs binary decensor metric {name!r}; "
                f"got {value!r}"
            )
    return refusal_score <= 4 and all(
        details[name] == expected for name, expected in _usable_response_metrics.items()
    )


def should_skip_group_without_usable_response(
    group: list[Rollout], enabled: bool
) -> bool:
    """Return True when an enabled, fully scored group has no usable response."""
    if not enabled or not group or any(
        getattr(rollout, "has_error", False) for rollout in group
    ):
        return False
    usable = [is_usable_response(rollout) for rollout in group]
    return not any(usable)


def _filter_group(group: list[Rollout], reason: str) -> None:
    for rollout in group:
        rollout.samples = []
        rollout.is_filtered = True
        rollout.filter_results = {reason: True}


def _empty_refusal_advantage_counts() -> dict[str, dict[str, int]]:
    return {band: {sign: 0 for sign in _advantage_signs} for band, _, _ in _refusal_bands}


def _refusal_band(score: int | float | None) -> str | None:
    if not _valid_refusal_score(score):
        return None
    for band, lower, upper in _refusal_bands:
        if lower <= score <= upper:
            return band
    return None


def _scalar_advantage(rollout: Rollout) -> int | float | None:
    getter = getattr(rollout, "scalar_advantage", None)
    if not callable(getter):
        return None
    advantage = getter()
    if isinstance(advantage, bool) or not isinstance(advantage, (int, float)):
        return None
    if isinstance(advantage, int):
        return advantage
    if not math.isfinite(advantage):
        return None
    return advantage


def should_skip_positive_strong_refusal_advantage_group(
    group: list[Rollout], enabled: bool
) -> bool:
    """Return True when an enabled group rewards a strong refusal."""
    if not enabled or not group or any(
        getattr(rollout, "has_error", False) for rollout in group
    ):
        return False
    skip = False
    for rollout in group:
        details = _decensor_details(rollout)
        if details["response_rejected_because"] is not None:
            continue
        score = _refusal_score(rollout)
        if score is None:
            raise RuntimeError(
                "require_usable_response_in_group needs refusal_score in [1, 9] "
                "before checking strong-refusal advantages"
            )
        advantage = _scalar_advantage(rollout)
        if advantage is None:
            raise RuntimeError(
                "require_usable_response_in_group needs a finite scalar advantage "
                "after group scoring"
            )
        if score >= 7 and advantage > 0:
            skip = True
    return skip


def refusal_advantage_counts(rollouts: list[Rollout]) -> dict[str, dict[str, int]]:
    """Count assigned advantage signs in refusal-score bands.

    Rollouts without a valid refusal score or scalar advantage are omitted.
    This is called after ``finalize_group`` so it observes the algorithm's
    assigned advantages without recomputing or changing them.
    """
    counts = _empty_refusal_advantage_counts()
    for rollout in rollouts:
        band = _refusal_band(_refusal_score(rollout))
        advantage = _scalar_advantage(rollout)
        if band is None or advantage is None:
            continue
        sign = "positive" if advantage > 0 else "negative" if advantage < 0 else "zero"
        counts[band][sign] += 1
    return counts


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
        self.train_batch_rollout_quotas: dict[str, int] | None = None
        self.train_batch_group_size: int | None = None
        if config.keep_train_batch_ratios:
            if batch_size is None or config.train.ratio_stages is not None:
                raise ValueError(
                    "keep_train_batch_ratios requires rollout batch_size without ratio_stages"
                )
            (
                self.train_batch_rollout_quotas,
                self.train_batch_group_size,
            ) = train_batch_ratio_quotas(
                batch_size,
                [(env.resolved_name, float(env.ratio), env.group_size) for env in config.train.env],
            )
            get_logger().info(f"Final train batch quotas: {self.train_batch_rollout_quotas}")
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
        self.strong_refusal_groups_skipped = 0
        self.no_usable_response_groups_skipped = 0
        self.positive_strong_refusal_advantage_groups_skipped = 0
        self.refusal_advantage_counts = _empty_refusal_advantage_counts()

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
            if self.train_batch_rollout_quotas is not None:
                return (
                    sum(values["accepted"] for values in self.ratio_batch_state().values()),
                    self.batch_size,
                    "rollouts",
                )
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

    def ratio_batch_state(self) -> dict[str, dict[str, int]]:
        if self.train_batch_rollout_quotas is None:
            return {}
        _, _, state = partition_train_batch(
            self.pending_batch,
            self.batch_size or 0,
            self.train_batch_rollout_quotas,
            self.train_batch_group_size,
        )
        return state

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
        if self.train_batch_rollout_quotas is not None:
            return self.process_batch()
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
        if self.train_batch_group_size is not None and len(group) != self.train_batch_group_size:
            raise RuntimeError("keep_train_batch_ratios requires exactly one rollout trace per group member")
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

        if should_skip_group_without_usable_response(
            group, self.config.require_usable_response_in_group
        ):
            self.no_usable_response_groups_skipped += 1
            _filter_group(group, "no_usable_response_group")
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | skipped group without usable response"
            )
            return

        # Keep the group in pending_rollouts, but send no optimizer samples.
        if should_skip_strong_refusal_group(
            group, self.config.skip_if_every_refusal_score_is_at_least
        ):
            self.strong_refusal_groups_skipped += 1
            _skip_strong_refusal_group(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | skipped strong-refusal group"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(survivors)

        if should_skip_positive_strong_refusal_advantage_group(
            group, self.config.require_usable_response_in_group
        ):
            self.positive_strong_refusal_advantage_groups_skipped += 1
            _filter_group(group, "positive_strong_refusal_advantage_group")
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} | skipped group with positive "
                "strong-refusal advantage"
            )
            return

        for band, sign_counts in refusal_advantage_counts(survivors).items():
            for sign, count in sign_counts.items():
                self.refusal_advantage_counts[band][sign] += count

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
        clean: list[Rollout] = []
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
            clean.append(r)
        if (
            self.train_batch_rollout_quotas is not None
            and clean
            and len(clean) != len(survivors)
        ):
            reason = "ratio_group_member_filtered"
            _filter_group(clean, reason)
            self.pre_filter_dropped += len(clean)
            self.pre_filter_dropped_by_name[reason] = (
                self.pre_filter_dropped_by_name.get(reason, 0) + len(clean)
            )
            num_filtered += len(clean)
            filtered_by_name[reason] = len(clean)
            clean = []
        if self.train_batch_rollout_quotas is not None and clean and self.post_filters and reject_ratio_group_after_post_filters(self.post_filters, clean):
            num_filtered += len(clean)
            filtered_by_name["post_batch_group"] = len(clean)
            clean = []
        for r in clean:
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
        if self.batch_size is not None:
            cohort, carry, state = partition_train_batch(
                self.pending_batch,
                self.batch_size,
                self.train_batch_rollout_quotas,
                self.train_batch_group_size,
            )
            if len(cohort) < self.batch_size or any(values["wait"] for values in state.values()):
                return None
            self.pending_batch = carry
        else:
            assert self.token_batch_size is not None
            if self.pending_tokens < self.token_batch_size:
                return None
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

        if self.post_filters and self.train_batch_rollout_quotas is None:
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
        self.strong_refusal_groups_skipped = 0
        self.no_usable_response_groups_skipped = 0
        self.positive_strong_refusal_advantage_groups_skipped = 0
        self.refusal_advantage_counts = _empty_refusal_advantage_counts()
