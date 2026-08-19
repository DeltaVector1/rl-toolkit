from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field


class DecensorState(vf.State):
    target_turns: int = 1
    user_done: bool = False
    user_sim_failed: bool = False
    user_sim_error: str = ""
    follow_up_strategies: list[str] = Field(default_factory=list)
    follow_up_attempts: list[dict] = Field(default_factory=list)
