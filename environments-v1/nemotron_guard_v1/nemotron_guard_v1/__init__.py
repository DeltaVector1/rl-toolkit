"""nemotron-guard-v1: the shared nemotron anti-hacking guardrail as v1 helpers.

The reusable component behind the four `nemotron-*` category envs. Given a `vf.Trace` it
computes the v0 `anti_hacking_multiplier` (hard-zero gates plus judge/meta soft penalties)
and folds it into a task reward as `final = anti_hacking_multiplier * task_reward`.
"""

from nemotron_guard_v1.guard import (
    AntiHackingConfig,
    GuardedTask,
    anti_hacking_multiplier,
    apply_guard,
)
from nemotron_guard_v1.judges import (
    CoherencyJudge,
    GuardJudgeConfig,
    MetaCommentaryJudge,
    ReasoningCoherencyJudge,
    ReasoningEthicsJudge,
    ScoreTagJudge,
)

__all__ = [
    "AntiHackingConfig",
    "GuardedTask",
    "GuardJudgeConfig",
    "anti_hacking_multiplier",
    "apply_guard",
    "CoherencyJudge",
    "ReasoningEthicsJudge",
    "ReasoningCoherencyJudge",
    "MetaCommentaryJudge",
    "ScoreTagJudge",
]
