from __future__ import annotations

import verifiers.v1 as vf

from testing_perf_env.taskset import TestingPerfTasksetConfig


class TestingPerfEnvConfig(vf.EnvConfig):
    taskset: TestingPerfTasksetConfig = TestingPerfTasksetConfig(id="testing-perf-env")
    assistant: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        max_turns=6,
    )


class TestingPerfEnv(vf.Env[TestingPerfEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.assistant.interaction(task) as assistant:
            target_turns = max(2, min(6, int(task.data.target_turns)))
            segment = await assistant.turn()
            for follow_up in task.data.follow_ups[: target_turns - 1]:
                if segment.terminated:
                    break
                segment = await assistant.turn(follow_up)


__all__ = ["TestingPerfEnv"]
