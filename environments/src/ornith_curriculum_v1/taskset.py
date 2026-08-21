from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

from ornith_curriculum_v1.archive import TaskArchive
from ornith_curriculum_v1.curriculum import (
    parse_candidate_task,
    proposal_prompt,
)
from ornith_curriculum_v1.models import (
    PolicyHistorySummary,
    PolicyIdentity,
    TaskFamilySpecification,
)
from ornith_curriculum_v1.validation import review_candidate_task


class OrnithCurriculumData(vf.TaskData):
    family_id: str
    family_description: str
    allowed_categories: list[str]
    completed_task_count: int = 0
    success_count: int = 0
    category_success_rates: dict[str, float] = {}
    recent_task_ids: list[str] = []
    proposer_policy_version: str
    proposer_adapter_hash: str


class OrnithCurriculumTaskConfig(vf.TaskConfig):
    verifier_id: str = "fixed-external-verifier"
    verifier_revision: str = "unconfigured"
    verifier_rubric_hash: str = "unconfigured"
    minimum_task_characters: int = 40
    maximum_task_characters: int = 4_000
    minimum_plan_steps: int = 1
    maximum_plan_steps: int = 8
    minimum_archive_novelty: float = 0.05
    minimum_static_evaluation_novelty: float = 0.20
    archive_path: str | None = None
    static_evaluation_path: str | None = None


class OrnithCurriculumTasksetConfig(vf.TasksetConfig):
    family_id: str = "decensor-conversation"
    family_description: str = (
        "Create diverse conversation tasks within the fixed Decensor task family."
    )
    allowed_categories: list[str] = ["general"]
    proposer_policy_version: str = "offline"
    proposer_adapter_hash: str = "not-trained"
    proposal_contexts_path: str | None = None
    task: OrnithCurriculumTaskConfig = OrnithCurriculumTaskConfig()


def _history(data: OrnithCurriculumData) -> PolicyHistorySummary:
    return PolicyHistorySummary(
        policy=PolicyIdentity(
            version=data.proposer_policy_version,
            adapter_hash=data.proposer_adapter_hash,
        ),
        completed_task_count=data.completed_task_count,
        success_count=data.success_count,
        category_success_rates=dict(data.category_success_rates),
        recent_task_ids=tuple(data.recent_task_ids),
    )


def _specification(
    data: OrnithCurriculumData,
    config: OrnithCurriculumTaskConfig,
) -> TaskFamilySpecification:
    return TaskFamilySpecification(
        family_id=data.family_id,
        description=data.family_description,
        categories=tuple(data.allowed_categories),
        minimum_task_characters=config.minimum_task_characters,
        maximum_task_characters=config.maximum_task_characters,
        minimum_plan_steps=config.minimum_plan_steps,
        maximum_plan_steps=config.maximum_plan_steps,
    )


class OrnithCurriculumTask(
    vf.Task[OrnithCurriculumData, vf.State, OrnithCurriculumTaskConfig]
):
    @vf.metric
    async def offline_curriculum_review(self, trace: vf.Trace) -> dict[str, float]:
        specification = _specification(self.data, self.config)
        history = _history(self.data)
        record: dict[str, object] = {
            "mode": "offline_shadow",
            "training_reward_emitted": False,
            "verifier_authority": {
                "verifier_id": self.config.verifier_id,
                "revision": self.config.verifier_revision,
                "rubric_hash": self.config.verifier_rubric_hash,
            },
            "proposer_policy_version": self.data.proposer_policy_version,
            "proposer_adapter_hash": self.data.proposer_adapter_hash,
        }
        try:
            candidate = parse_candidate_task(
                trace.last_reply,
                proposer_policy=history.policy,
                history=history,
            )
        except ValueError as error:
            record.update(
                {
                    "status": "rejected_invalid_task",
                    "reasons": ["invalid_candidate_schema"],
                    "error": str(error),
                }
            )
            trace.info["ornith_curriculum"] = record
            return {
                "ornith_candidate_schema_valid": 0.0,
                "ornith_candidate_deterministic_checks_valid": 0.0,
            }

        review = review_candidate_task(candidate, specification)
        archive = (
            TaskArchive.load(Path(self.config.archive_path))
            if self.config.archive_path
            else TaskArchive()
        )
        novelty, nearest_id, nearest_similarity = archive.novelty(
            candidate.task_text
        )
        archive_novelty_accepted = (
            novelty >= self.config.minimum_archive_novelty
        )
        static_archive = (
            TaskArchive.load(Path(self.config.static_evaluation_path))
            if self.config.static_evaluation_path
            else TaskArchive()
        )
        _, static_similarity = static_archive.nearest(candidate.task_text)
        static_evaluation_novelty = 1.0 - static_similarity
        static_evaluation_separate = (
            static_evaluation_novelty
            >= self.config.minimum_static_evaluation_novelty
        )
        reasons = list(review.reasons)
        if not archive_novelty_accepted:
            reasons.append("archive_novelty_below_minimum")
        if not static_evaluation_separate:
            reasons.append("static_evaluation_overlap")
        candidate_accepted = (
            review.accepted
            and archive_novelty_accepted
            and static_evaluation_separate
        )
        record.update(
            {
                "status": (
                    "awaiting_complete_solution_group"
                    if candidate_accepted
                    else "rejected_invalid_task"
                ),
                "candidate": {
                    "task_id": candidate.task_id,
                    "task_text": candidate.task_text,
                    "family_id": candidate.family_id,
                    "category": candidate.category,
                    "solver_plan": list(candidate.solver_plan),
                    "history_summary_hash": candidate.history_summary_hash,
                },
                "deterministic_decisions": review.decisions,
                "reasons": reasons,
                "archive_novelty": novelty,
                "minimum_archive_novelty": (
                    self.config.minimum_archive_novelty
                ),
                "archive_novelty_accepted": archive_novelty_accepted,
                "nearest_archive_task_id": nearest_id,
                "nearest_archive_similarity": nearest_similarity,
                "static_evaluation_similarity": static_similarity,
                "static_evaluation_novelty": static_evaluation_novelty,
                "minimum_static_evaluation_novelty": (
                    self.config.minimum_static_evaluation_novelty
                ),
                "static_evaluation_separate": static_evaluation_separate,
            }
        )
        trace.info["ornith_curriculum"] = record
        return {
            "ornith_candidate_schema_valid": 1.0,
            "ornith_candidate_deterministic_checks_valid": float(review.accepted),
            "ornith_candidate_archive_novelty": novelty,
            "ornith_candidate_archive_novelty_accepted": float(
                archive_novelty_accepted
            ),
            "ornith_candidate_static_evaluation_similarity": static_similarity,
            "ornith_candidate_static_evaluation_separate": float(
                static_evaluation_separate
            ),
            "ornith_candidate_accepted": float(candidate_accepted),
        }


class OrnithCurriculumTaskset(
    vf.Taskset[OrnithCurriculumTask, OrnithCurriculumTasksetConfig]
):
    def _contexts(self) -> list[dict[str, object]]:
        if not self.config.proposal_contexts_path:
            return [{}]
        rows: list[dict[str, object]] = []
        path = Path(self.config.proposal_contexts_path)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"proposal context at line {line_number} is not an object"
                    )
                rows.append(value)
        return rows

    def load(self) -> list[OrnithCurriculumTask]:
        tasks: list[OrnithCurriculumTask] = []
        for index, row in enumerate(self._contexts()):
            policy = PolicyIdentity(
                version=str(
                    row.get(
                        "proposer_policy_version",
                        self.config.proposer_policy_version,
                    )
                ),
                adapter_hash=str(
                    row.get(
                        "proposer_adapter_hash",
                        self.config.proposer_adapter_hash,
                    )
                ),
            )
            history = PolicyHistorySummary(
                policy=policy,
                completed_task_count=int(row.get("completed_task_count", 0)),
                success_count=int(row.get("success_count", 0)),
                category_success_rates=dict(
                    row.get("category_success_rates", {})  # type: ignore[arg-type]
                ),
                recent_task_ids=tuple(
                    row.get("recent_task_ids", ())  # type: ignore[arg-type]
                ),
            )
            specification = TaskFamilySpecification(
                family_id=self.config.family_id,
                description=self.config.family_description,
                categories=tuple(self.config.allowed_categories),
                minimum_task_characters=self.config.task.minimum_task_characters,
                maximum_task_characters=self.config.task.maximum_task_characters,
                minimum_plan_steps=self.config.task.minimum_plan_steps,
                maximum_plan_steps=self.config.task.maximum_plan_steps,
            )
            tasks.append(
                OrnithCurriculumTask(
                    OrnithCurriculumData(
                        idx=index,
                        prompt=proposal_prompt(specification, history),
                        family_id=specification.family_id,
                        family_description=specification.description,
                        allowed_categories=list(specification.categories),
                        completed_task_count=history.completed_task_count,
                        success_count=history.success_count,
                        category_success_rates=history.category_success_rates,
                        recent_task_ids=list(history.recent_task_ids),
                        proposer_policy_version=policy.version,
                        proposer_adapter_hash=policy.adapter_hash,
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = ["OrnithCurriculumTaskset"]
