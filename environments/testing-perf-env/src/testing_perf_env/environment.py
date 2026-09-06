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
        await agents.assistant.run(task)


__all__ = ["TestingPerfEnv"]
