"""Terminal dispatcher results for attempts without an episode."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import verifiers.v1 as vf


WorkKind = Literal["train", "eval"]
CancelReason = Literal["stale", "overload", "superseded"]


@dataclass(frozen=True)
class DispatchFailure:
    """An environment request that failed before producing an episode."""

    kind: WorkKind
    env_name: str
    group_id: uuid.UUID
    step: int
    policy_version: int
    task_idx: int
    task_type: str
    task_key: str
    task_hash: str
    count: int
    error: vf.Error


@dataclass(frozen=True)
class GroupCancellation:
    """A terminal count for group members cancelled by the dispatcher."""

    kind: WorkKind
    env_name: str
    group_id: uuid.UUID
    step: int
    count: int
    reason: CancelReason


if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

    DispatchResult = list[Rollout] | DispatchFailure | GroupCancellation
else:
    DispatchResult = object
