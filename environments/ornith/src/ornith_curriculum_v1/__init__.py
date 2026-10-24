from ornith_curriculum_v1.archive import (
    ArchiveRecord,
    StaticEvaluationSet,
    TaskArchive,
)
from ornith_curriculum_v1.curriculum import (
    CurriculumSettings,
    archive_record_from_evaluation,
    evaluate_candidate_task,
    frontier_difficulty,
    parse_candidate_task,
    proposal_prompt,
)
from ornith_curriculum_v1.models import (
    CandidateTask,
    CurriculumEvaluation,
    ExternalVerifierAuthority,
    PolicyHistorySummary,
    PolicyIdentity,
    SolutionGroup,
    SolutionResult,
    TaskFamilySpecification,
)

__all__ = [
    "ArchiveRecord",
    "CandidateTask",
    "CurriculumEvaluation",
    "CurriculumSettings",
    "ExternalVerifierAuthority",
    "PolicyHistorySummary",
    "PolicyIdentity",
    "SolutionGroup",
    "SolutionResult",
    "StaticEvaluationSet",
    "TaskArchive",
    "TaskFamilySpecification",
    "archive_record_from_evaluation",
    "evaluate_candidate_task",
    "frontier_difficulty",
    "parse_candidate_task",
    "proposal_prompt",
]

try:
    import verifiers.v1  # noqa: F401
except ModuleNotFoundError as error:
    if error.name not in {"verifiers", "verifiers.v1"}:
        raise
else:
    from ornith_curriculum_v1.taskset import (
        OrnithCurriculumTaskset as OrnithCurriculumTaskset,
    )

    __all__.append("OrnithCurriculumTaskset")
