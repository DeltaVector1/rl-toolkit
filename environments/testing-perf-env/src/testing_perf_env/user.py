from __future__ import annotations

import verifiers.v1 as vf

from testing_perf_env.state import TestingPerfState


class TestingPerfUserConfig(vf.UserConfig):
    pass


class TestingPerfUser(vf.User[TestingPerfUserConfig, TestingPerfState]):
    async def setup_task(self, task: vf.TaskData) -> None:
        self.state.next_follow_up = 0

    async def respond(self, message: str) -> vf.Messages:
        if self.state.next_follow_up >= max(0, self.state.target_turns - 1):
            return []
        if self.state.next_follow_up >= len(self.state.follow_ups):
            raise RuntimeError("testing-perf user follow-ups exhausted")
        follow_up = self.state.follow_ups[self.state.next_follow_up]
        self.state.next_follow_up += 1
        return [vf.UserMessage(content=follow_up)]


__all__ = ["TestingPerfUser", "TestingPerfUserConfig"]


if __name__ == "__main__":
    TestingPerfUser.run()
