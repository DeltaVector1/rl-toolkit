from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field


class TestingPerfState(vf.State):
    target_turns: int = 2
    follow_ups: list[str] = Field(default_factory=list)
    next_follow_up: int = 0


__all__ = ["TestingPerfState"]
