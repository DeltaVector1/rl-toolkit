from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


EvaluationStatus = Literal[
    "scored",
    "rejected_invalid_task",
    "unscored_incomplete_group",
    "unscored_service_failure",
    "unscored_policy_mismatch",
    "unscored_verifier_mismatch",
]
SolutionStatus = Literal["success", "failure", "service_failure"]


@dataclass(frozen=True)
class PolicyIdentity:
    version: str
    adapter_hash: str


@dataclass(frozen=True)
class TaskFamilySpecification:
    family_id: str
    description: str
    categories: tuple[str, ...]
    minimum_task_characters: int = 40
    maximum_task_characters: int = 4_000
    minimum_plan_steps: int = 1
    maximum_plan_steps: int = 8


@dataclass(frozen=True)
class PolicyHistorySummary:
    policy: PolicyIdentity
    completed_task_count: int
    success_count: int
    category_success_rates: dict[str, float] = field(default_factory=dict)
    recent_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateTask:
    task_id: str
    task_text: str
    family_id: str
    category: str
    solver_plan: tuple[str, ...]
    proposer_policy: PolicyIdentity
    history_summary_hash: str


@dataclass(frozen=True)
class ExternalVerifierAuthority:
    verifier_id: str
    revision: str
    rubric_hash: str


@dataclass(frozen=True)
class SolutionResult:
    rollout_id: str
    task_id: str
    solver_policy: PolicyIdentity
    verifier: ExternalVerifierAuthority
    status: SolutionStatus
    verifier_details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SolutionGroup:
    task_id: str
    expected_size: int
    results: tuple[SolutionResult, ...]


@dataclass(frozen=True)
class DeterministicTaskReview:
    accepted: bool
    reasons: tuple[str, ...]
    decisions: dict[str, bool]


@dataclass(frozen=True)
class CurriculumEvaluation:
    status: EvaluationStatus
    reward: float | None
    task_id: str
    decisions: dict[str, object]
    metrics: dict[str, float | int | str | None]

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalCallAttempt:
    attempt: int
    status: Literal["completed", "request_failed", "invalid_response"]
    elapsed_seconds: float
    error_type: str | None = None
    error: str | None = None
    response_text: str | None = None


@dataclass(frozen=True)
class ExternalCallResult:
    status: Literal[
        "completed",
        "invalid_model_response",
        "unscored_service_failure",
    ]
    value: object | None
    attempts: tuple[ExternalCallAttempt, ...]
