from __future__ import annotations

import re

from ornith_curriculum_v1.models import (
    CandidateTask,
    DeterministicTaskReview,
    TaskFamilySpecification,
)


executable_plan_patterns = (
    re.compile(r"```"),
    re.compile(r"(?im)^\s*(?:python|bash|sh|powershell)\b"),
    re.compile(r"(?im)^\s*import\s+[a-zA-Z_][\w.]*\s*$"),
    re.compile(r"(?im)^\s*from\s+[a-zA-Z_][\w.]*\s+import\s+"),
    re.compile(r"(?im)^\s*(?:async\s+def|def|class)\s+[a-zA-Z_]"),
    re.compile(r"(?m)^\s*#!"),
    re.compile(r"\b(?:eval|exec|subprocess|os\.system)\s*\("),
    re.compile(r"(?:^|\s)/(?:etc|proc|sys|data|home|root)(?:/|\s|$)"),
)
verifier_change_patterns = (
    re.compile(r"(?i)\b(?:change|edit|replace|override|disable|bypass)\b.{0,40}\b(?:judge|verifier|rubric|reward|hidden test)\b"),
    re.compile(r"(?i)\b(?:judge|verifier|rubric|reward|hidden test)\b.{0,40}\b(?:change|edit|replace|override|disable|bypass)\b"),
)


def review_candidate_task(
    candidate: CandidateTask,
    specification: TaskFamilySpecification,
) -> DeterministicTaskReview:
    reasons: list[str] = []
    task_text_length_ok = (
        specification.minimum_task_characters
        <= len(candidate.task_text.strip())
        <= specification.maximum_task_characters
    )
    if not task_text_length_ok:
        reasons.append("task_text_length_out_of_range")

    family_matches = candidate.family_id == specification.family_id
    if not family_matches:
        reasons.append("task_family_mismatch")

    category_allowed = candidate.category in specification.categories
    if not category_allowed:
        reasons.append("category_not_allowed")

    plan_step_count_ok = (
        specification.minimum_plan_steps
        <= len(candidate.solver_plan)
        <= specification.maximum_plan_steps
    )
    if not plan_step_count_ok:
        reasons.append("solver_plan_step_count_out_of_range")

    plan_steps_have_text = all(step.strip() for step in candidate.solver_plan)
    if not plan_steps_have_text:
        reasons.append("solver_plan_contains_empty_step")

    plan_text = "\n".join(candidate.solver_plan)
    plan_is_non_executable = not any(
        pattern.search(plan_text) for pattern in executable_plan_patterns
    )
    if not plan_is_non_executable:
        reasons.append("solver_plan_contains_executable_content")

    plan_has_no_verifier_instructions = re.search(
        r"(?i)\b(?:judge|verifier|rubric|reward function|hidden test)\b",
        plan_text,
    ) is None
    if not plan_has_no_verifier_instructions:
        reasons.append("solver_plan_contains_verifier_instructions")

    verifier_authority_unchanged = not any(
        pattern.search(candidate.task_text) or pattern.search(plan_text)
        for pattern in verifier_change_patterns
    )
    if not verifier_authority_unchanged:
        reasons.append("candidate_attempts_to_change_verification")

    decisions = {
        "task_text_length_ok": task_text_length_ok,
        "task_family_matches": family_matches,
        "category_allowed": category_allowed,
        "solver_plan_step_count_ok": plan_step_count_ok,
        "solver_plan_steps_have_text": plan_steps_have_text,
        "solver_plan_is_non_executable": plan_is_non_executable,
        "solver_plan_has_no_verifier_instructions": (
            plan_has_no_verifier_instructions
        ),
        "external_verifier_unchanged": verifier_authority_unchanged,
    }
    return DeterministicTaskReview(
        accepted=not reasons,
        reasons=tuple(reasons),
        decisions=decisions,
    )
