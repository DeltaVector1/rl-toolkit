"""Prime-RL v0.8 multi-agent control flow for the Decensor taskset.

Verifiers d30a3f48 removed the task-attached ``vf.User`` server.  The environment
now owns the turn exchange.  The policy remains the only agent and the existing
in-process simulator produces deterministic follow-up user messages.
"""

from __future__ import annotations

import verifiers.v1 as vf

from decensor_v1.taskset import DecensorTasksetConfig
from decensor_v1.user import DecensorUser


class DecensorEnvConfig(vf.EnvConfig):
    """One trainable assistant driven by the Decensor in-process user simulator."""

    # Bind the seed taskset at the type boundary.  EnvConfig's base default is
    # an empty, unspecialized TasksetConfig, which is not sufficient when a
    # caller constructs this config directly instead of going through the CLI
    # narrowing validator.
    taskset: DecensorTasksetConfig = DecensorTasksetConfig(id="decensor-v1")
    assistant: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        max_turns=6,
    )


class DecensorEnv(vf.Env[DecensorEnvConfig]):
    """Run the opening prompt and deterministic follow-up turns as one policy trace."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.assistant.interaction(task) as assistant:
            simulator = DecensorUser()
            simulator.config = task.config.user
            simulator.state = assistant.trace.state
            # Task.setup normally initializes this before the first harness
            # segment. Set it here too so the Env owns a complete invariant.
            simulator.state.target_turns = max(1, int(task.data.target_turns))
            await simulator.setup_task(task.data)

            segment = await assistant.turn()
            while not segment.terminated:
                completed = assistant.trace.num_turns
                if completed >= simulator.state.target_turns:
                    simulator.state.user_done = True
                    break

                follow_up = await simulator.respond(segment.last_reply)
                if not follow_up:
                    break
                segment = await assistant.turn(follow_up)


__all__ = ["DecensorEnv"]
