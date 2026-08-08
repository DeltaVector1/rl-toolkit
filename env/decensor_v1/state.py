"""Shared per-rollout state for the decensor env.

Written by the user simulator, read by the task's stop condition.
"""

from __future__ import annotations

import verifiers.v1 as vf


class DecensorState(vf.State):
    target_turns: int = 1
    """How many assistant turns the model should produce before the user simulator stops."""
    user_done: bool = False
    """Set once the simulator has driven the conversation to `target_turns`."""
    user_sim_failed: bool = False
    """The simulator returned no parseable follow-up before `target_turns`."""
