from __future__ import annotations

import verifiers.v1 as vf

from decensor_experimental_v1.taskset import DecensorTasksetConfig
from decensor_experimental_v1.user import DecensorUser


class DecensorEnvConfig(vf.EnvConfig):
    taskset: DecensorTasksetConfig = DecensorTasksetConfig(id="decensor-experimental-v1")
    assistant: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        max_turns=6,
    )


class DecensorEnv(vf.Env[DecensorEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.assistant.interaction(task) as assistant:
            simulator = DecensorUser()
            simulator.config = task.config.user
            simulator.state = assistant.trace.state

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
