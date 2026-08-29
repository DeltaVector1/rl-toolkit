from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ornith_curriculum_v1.archive import (
    ArchiveRecord,
    TaskArchive,
    task_content_id,
)
from ornith_curriculum_v1.models import (
    CandidateTask,
    CurriculumEvaluation,
    EvaluationStatus,
    ExternalVerifierAuthority,
    PolicyHistorySummary,
    PolicyIdentity,
    SolutionGroup,
    TaskFamilySpecification,
)
from ornith_curriculum_v1.validation import review_candidate_task


@dataclass(frozen=True)
class CurriculumSettings:
    solution_group_size: int = 16
    target_success_rate: float = 0.20
    difficulty_width: float = 0.15
    minimum_archive_novelty: float = 0.05
    minimum_static_evaluation_novelty: float = 0.20

    def __post_init__(self) -> None:
        if self.solution_group_size <= 0:
            raise ValueError("solution_group_size must be positive")
        if not 0.0 <= self.target_success_rate <= 1.0:
            raise ValueError("target_success_rate must be from zero through one")
        if self.difficulty_width <= 0:
            raise ValueError("difficulty_width must be positive")
        for name, value in (
            ("minimum_archive_novelty", self.minimum_archive_novelty),
            (
                "minimum_static_evaluation_novelty",
                self.minimum_static_evaluation_novelty,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be from zero through one")


def policy_history_hash(history: PolicyHistorySummary) -> str:
    encoded = json.dumps(asdict(history), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def proposal_prompt(
    specification: TaskFamilySpecification,
    history: PolicyHistorySummary,
) -> str:
    categories = ", ".join(specification.categories)
    category_rates = json.dumps(history.category_success_rates, sort_keys=True)
    return (
        "Create one candidate task for an offline curriculum review.\n"
        f"Task family: {specification.family_id}\n"
        f"Family purpose: {specification.description}\n"
        f"Allowed categories: {categories}\n"
        f"Completed tasks: {history.completed_task_count}\n"
        f"Successful tasks: {history.success_count}\n"
        f"Success rates by category: {category_rates}\n"
        "Return one JSON object with exactly these keys: task_text, family_id, "
        "category, solver_plan. solver_plan must be a list of plain-language "
        "steps. It must not contain code or verifier instructions. Do not define, "
        "change, or replace the external verifier."
    )


def parse_candidate_task(
    text: str,
    *,
    proposer_policy: PolicyIdentity,
    history: PolicyHistorySummary,
) -> CandidateTask:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"candidate is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("candidate must be one JSON object")
    expected_keys = {"task_text", "family_id", "category", "solver_plan"}
    if set(value) != expected_keys:
        raise ValueError("candidate fields must exactly match the required fields")
    task_text = value["task_text"]
    family_id = value["family_id"]
    category = value["category"]
    solver_plan = value["solver_plan"]
    if not all(isinstance(item, str) for item in (task_text, family_id, category)):
        raise ValueError("task_text, family_id, and category must be strings")
    if not isinstance(solver_plan, list) or not all(
        isinstance(step, str) for step in solver_plan
    ):
        raise ValueError("solver_plan must be a list of strings")
    return CandidateTask(
        task_id=task_content_id(family_id, category, task_text),
        task_text=task_text,
        family_id=family_id,
        category=category,
        solver_plan=tuple(solver_plan),
        proposer_policy=proposer_policy,
        history_summary_hash=policy_history_hash(history),
    )


def frontier_difficulty(
    success_rate: float,
    *,
    target_success_rate: float,
    difficulty_width: float,
) -> float:
    if difficulty_width <= 0:
        raise ValueError("difficulty_width must be positive")
    return math.exp(
        -((success_rate - target_success_rate) ** 2)
        / (2 * difficulty_width**2)
    )


def _unscored_evaluation(
    status: EvaluationStatus,
    candidate: CandidateTask,
    decisions: dict[str, object],
    metrics: dict[str, float | int | str | None],
) -> CurriculumEvaluation:
    metrics["reward"] = None
    metrics["evaluation_status"] = status
    return CurriculumEvaluation(
        status=status,
        reward=None,
        task_id=candidate.task_id,
        decisions=decisions,
        metrics=metrics,
    )


def evaluate_candidate_task(
    candidate: CandidateTask,
    specification: TaskFamilySpecification,
    solution_group: SolutionGroup,
    *,
    solver_policy: PolicyIdentity,
    verifier: ExternalVerifierAuthority,
    archive: TaskArchive,
    static_evaluation_archive: TaskArchive,
    settings: CurriculumSettings = CurriculumSettings(),
) -> CurriculumEvaluation:
    review = review_candidate_task(candidate, specification)
    novelty, nearest_task_id, nearest_similarity = archive.novelty(
        candidate.task_text
    )
    _, static_similarity = static_evaluation_archive.nearest(candidate.task_text)
    archive_novelty_ok = novelty >= settings.minimum_archive_novelty
    static_evaluation_separate = (
        1.0 - static_similarity >= settings.minimum_static_evaluation_novelty
    )
    decisions: dict[str, object] = {
        **review.decisions,
        "deterministic_task_checks_passed": review.accepted,
        "deterministic_rejection_reasons": list(review.reasons),
        "archive_novelty_accepted": archive_novelty_ok,
        "static_evaluation_separate": static_evaluation_separate,
        "solution_group_complete": False,
        "solution_group_policy_consistent": False,
        "fixed_external_verifier_matched": False,
        "service_results_complete": False,
        "candidate_accepted": None,
    }
    metrics: dict[str, float | int | str | None] = {
        "task_valid": int(review.accepted),
        "archive_novelty": novelty,
        "nearest_archive_task_id": nearest_task_id,
        "nearest_archive_similarity": nearest_similarity,
        "static_evaluation_similarity": static_similarity,
        "solution_group_expected_size": settings.solution_group_size,
        "solution_group_received_size": len(solution_group.results),
        "proposer_policy_version": candidate.proposer_policy.version,
        "proposer_adapter_hash": candidate.proposer_policy.adapter_hash,
        "solver_policy_version": solver_policy.version,
        "solver_adapter_hash": solver_policy.adapter_hash,
        "verifier_id": verifier.verifier_id,
        "verifier_revision": verifier.revision,
        "verifier_rubric_hash": verifier.rubric_hash,
        "frontier_target_success_rate": settings.target_success_rate,
        "valid_solution_count": None,
        "success_count": None,
        "success_rate": None,
        "frontier_difficulty": None,
        "reward": None,
    }

    if not review.accepted or not archive_novelty_ok or not static_evaluation_separate:
        decisions["candidate_accepted"] = False
        metrics["reward"] = 0.0
        metrics["evaluation_status"] = "rejected_invalid_task"
        return CurriculumEvaluation(
            status="rejected_invalid_task",
            reward=0.0,
            task_id=candidate.task_id,
            decisions=decisions,
            metrics=metrics,
        )

    group_size_matches = (
        solution_group.expected_size == settings.solution_group_size
        and len(solution_group.results) == settings.solution_group_size
    )
    unique_rollout_ids = len(
        {result.rollout_id for result in solution_group.results}
    ) == len(solution_group.results)
    task_ids_match = solution_group.task_id == candidate.task_id and all(
        result.task_id == candidate.task_id for result in solution_group.results
    )
    group_complete = group_size_matches and unique_rollout_ids and task_ids_match
    decisions["solution_group_complete"] = group_complete
    if not group_complete:
        return _unscored_evaluation(
            "unscored_incomplete_group", candidate, decisions, metrics
        )

    policy_consistent = all(
        result.solver_policy == solver_policy for result in solution_group.results
    )
    decisions["solution_group_policy_consistent"] = policy_consistent
    if not policy_consistent:
        return _unscored_evaluation(
            "unscored_policy_mismatch", candidate, decisions, metrics
        )

    verifier_matches = all(
        result.verifier == verifier for result in solution_group.results
    )
    decisions["fixed_external_verifier_matched"] = verifier_matches
    if not verifier_matches:
        return _unscored_evaluation(
            "unscored_verifier_mismatch", candidate, decisions, metrics
        )

    service_results_complete = all(
        result.status != "service_failure" for result in solution_group.results
    )
    metrics["valid_solution_count"] = sum(
        result.status != "service_failure" for result in solution_group.results
    )
    decisions["service_results_complete"] = service_results_complete
    if not service_results_complete:
        return _unscored_evaluation(
            "unscored_service_failure", candidate, decisions, metrics
        )

    success_count = sum(
        result.status == "success" for result in solution_group.results
    )
    success_rate = success_count / settings.solution_group_size
    difficulty = frontier_difficulty(
        success_rate,
        target_success_rate=settings.target_success_rate,
        difficulty_width=settings.difficulty_width,
    )
    reward = difficulty * novelty
    decisions["candidate_accepted"] = True
    metrics.update(
        {
            "evaluation_status": "scored",
            "valid_solution_count": settings.solution_group_size,
            "success_count": success_count,
            "success_rate": success_rate,
            "frontier_difficulty": difficulty,
            "reward": reward,
        }
    )
    return CurriculumEvaluation(
        status="scored",
        reward=reward,
        task_id=candidate.task_id,
        decisions=decisions,
        metrics=metrics,
    )


def archive_record_from_evaluation(
    candidate: CandidateTask,
    solver_policy: PolicyIdentity,
    verifier: ExternalVerifierAuthority,
    evaluation: CurriculumEvaluation,
) -> ArchiveRecord:
    if evaluation.status != "scored":
        raise ValueError("only scored tasks can enter the archive")
    return ArchiveRecord(
        task_id=candidate.task_id,
        family_id=candidate.family_id,
        category=candidate.category,
        task_text=candidate.task_text,
        proposer_policy_version=candidate.proposer_policy.version,
        proposer_adapter_hash=candidate.proposer_policy.adapter_hash,
        solver_policy_version=solver_policy.version,
        solver_adapter_hash=solver_policy.adapter_hash,
        verifier_id=verifier.verifier_id,
        verifier_revision=verifier.revision,
        verifier_rubric_hash=verifier.rubric_hash,
        success_count=int(evaluation.metrics["success_count"]),
        valid_solution_count=int(evaluation.metrics["valid_solution_count"]),
        success_rate=float(evaluation.metrics["success_rate"]),
        frontier_score=float(evaluation.metrics["frontier_difficulty"]),
        novelty_score=float(evaluation.metrics["archive_novelty"]),
    )
