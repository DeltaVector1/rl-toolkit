"""Focused mocked-backend check for the installed verifiers.v1 integration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
from types import SimpleNamespace

check_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(check_dir, "src"))

import verifiers.v1 as vf  # noqa: E402
from testing_perf_env.environment import TestingPerfEnv  # noqa: E402
from testing_perf_env.judge import TestingPerfJudge, TurnReview  # noqa: E402
from testing_perf_env.taskset import (  # noqa: E402
    TestingPerfData,
    TestingPerfTask,
    TestingPerfTaskConfig,
    TestingPerfTasksetConfig,
)


class FakeInteraction:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list = []
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def turn(self, message=None):
        self.calls.append(message)
        reply = self.replies[self.index]
        self.index += 1
        return SimpleNamespace(
            terminated=self.index >= len(self.replies),
            last_reply=reply,
        )


class FakeAssistant:
    def __init__(self, replies: list[str]):
        self.interaction_instance = FakeInteraction(replies)

    def interaction(self, task):
        return self.interaction_instance


class FakeAgents:
    def __init__(self, replies: list[str]):
        self.assistant = FakeAssistant(replies)


class FakeServiceError(ConnectionError):
    status_code = 503


def make_task(target_turns: int = 2, max_attempts: int = 3) -> TestingPerfTask:
    return TestingPerfTask(
        TestingPerfData(
            idx=0,
            prompt="Make a short plan for sorting a desk. Use plain text.",
            task_id="desk-plan",
            requirements=["name two groups", "include one review step"],
            follow_ups=["Add one way to review the plan."],
            target_turns=target_turns,
            response_format="plain",
        ),
        TestingPerfTaskConfig(
            judge={
                "model": "mock",
                "max_attempts": max_attempts,
                "retry_delay_seconds": 0,
            }
        ),
    )


def make_trace(task: TestingPerfTask, answers: list[str]) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig(harness={"id": "null"})),
    )
    parent: int | None = None
    for index, answer in enumerate(answers):
        user_text = (
            task.data.prompt_text if index == 0 else task.data.follow_ups[index - 1]
        )
        trace.nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.UserMessage(content=user_text),
            )
        )
        parent = len(trace.nodes) - 1
        trace.nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.AssistantMessage(content=answer),
                sampled=True,
            )
        )
        parent = len(trace.nodes) - 1
    return trace


async def run() -> None:
    with open(
        os.path.join(
            check_dir, "..", "..", "configs", "testing-perf", "orchestrator.toml"
        ),
        "rb",
    ) as stream:
        orchestrator_config = tomllib.load(stream)
    profile_env = vf.resolve_env_config(orchestrator_config["train"]["env"][0]["env"])
    assert isinstance(vf.load_environment(profile_env), TestingPerfEnv)
    assert profile_env.id == profile_env.taskset.id == "testing-perf-env"
    assert profile_env.assistant.max_turns == 6
    assert profile_env.assistant.max_output_tokens == 16384
    assert profile_env.taskset.num_examples >= 64
    assert profile_env.taskset.max_turns == 6
    assert profile_env.taskset.task.judge.base_url.startswith("http://202.214.223.66:")
    assert profile_env.taskset.task.judge.model.startswith("RedHatAI/gemma-")
    assert profile_env.taskset.task.judge.sampling.extra_body[
        "chat_template_kwargs"
    ] == {"enable_thinking": False}

    taskset_config = TestingPerfTasksetConfig(
        id="testing-perf-env", num_examples=5, dataset_seed=1, max_turns=6
    )
    taskset = vf.load_taskset(taskset_config)
    loaded_tasks = list(taskset)
    assert [task.data.target_turns for task in loaded_tasks] == [2, 3, 4, 5, 6]
    for max_turns in range(2, 7):
        bounded = TestingPerfTasksetConfig(
            id="testing-perf-env", num_examples=1, max_turns=max_turns
        )
        assert 2 <= next(iter(vf.load_taskset(bounded))).data.target_turns <= max_turns
    env_config = vf.resolve_env_config(
        {
            "id": "testing-perf-env",
            "taskset": {"id": "testing-perf-env", "num_examples": 1},
        }
    )
    env = vf.load_environment(env_config)
    assert isinstance(env, TestingPerfEnv)

    agents = FakeAgents(["first", "second"])
    await env.run(make_task(), agents)
    assert agents.assistant.interaction_instance.calls == [
        None,
        "Add one way to review the plan.",
    ]
    six_turn_agents = FakeAgents(["reply"] * 6)
    await env.run(loaded_tasks[-1], six_turn_agents)
    assert len(six_turn_agents.assistant.interaction_instance.calls) == 6

    original_complete = TestingPerfJudge.complete
    responses = [
        json.dumps(
            {
                "correctness": 0.9,
                "task_adherence": 0.8,
                "coherence": 0.7,
                "evidence": ["covers the requested groups"],
            }
        )
    ] * 2

    async def mocked_complete(self, messages, *, trace=None, schema=None, **kwargs):
        assert schema is TurnReview
        text = responses.pop(0)
        response = vf.JudgeResponse(
            text=text,
            parsed=TurnReview.model_validate_json(text),
            usage=vf.Usage(prompt_tokens=2, completion_tokens=5),
        )
        if trace is not None:
            trace.record_judge(response)
        return response

    TestingPerfJudge.complete = mocked_complete
    try:
        task = make_task()
        trace = make_trace(task, ["sort by use", "review the groups"])
        await task.score(trace)
        high_score = trace.rewards["testing_perf_reward"].score
        assert high_score > 0.7
        assert trace.info["testing_perf"]["evaluation_status"] == "scored"
        assert trace.info["testing_perf"]["judge_calls"] == 2
        assert trace.info["testing_perf"]["judge_input_tokens"] == 4
        assert trace.info["testing_perf"]["judge_output_tokens"] == 10
        assert trace.info["testing_perf_judges"][0]["prompt"]

        responses.extend(
            [
                json.dumps(
                    {
                        "correctness": 0.1,
                        "task_adherence": 0.2,
                        "coherence": 0.3,
                        "evidence": ["misses requested details"],
                    }
                )
            ]
            * 2
        )
        low_trace = make_trace(task, ["sort by use", "review the groups"])
        await task.score(low_trace)
        assert low_trace.rewards["testing_perf_reward"].score < high_score

        empty_trace = make_trace(task, ["", "review the groups"])
        await task.score(empty_trace)
        assert empty_trace.rewards["testing_perf_reward"].score == 0
        assert empty_trace.info["testing_perf"]["evaluation_status"] == (
            "invalid_model_response"
        )

        bad_responses = iter(["not-json", "still-not-json", "{}"])

        async def malformed_complete(
            self, messages, *, trace=None, schema=None, **kwargs
        ):
            assert schema is TurnReview
            response = vf.JudgeResponse(text=next(bad_responses))
            if trace is not None:
                trace.record_judge(response)
            return response

        TestingPerfJudge.complete = malformed_complete
        failed = make_trace(make_task(), ["sort by use", "review the groups"])
        try:
            await make_task().score(failed)
        except vf.TaskError:
            pass
        else:
            raise AssertionError("exhausted malformed judges must fail scoring")
        assert failed.rewards["testing_perf_reward"] is None
        assert failed.info["testing_perf"]["evaluation_status"] == (
            "unscored_invalid_judge_response"
        )

        stale_calls = 0

        async def stale_record_complete(
            self, messages, *, trace=None, schema=None, **kwargs
        ):
            assert schema is TurnReview
            nonlocal stale_calls
            stale_calls += 1
            if stale_calls == 1:
                response = vf.JudgeResponse(text="not-json")
                if trace is not None:
                    trace.record_judge(response)
                return response
            raise FakeServiceError(
                "authorization=demo-token token=demo-token connection failed"
            )

        TestingPerfJudge.complete = stale_record_complete
        stale_failed = make_trace(
            make_task(max_attempts=2), ["sort by use", "review the groups"]
        )
        try:
            await make_task(max_attempts=2).score(stale_failed)
        except vf.TaskError:
            pass
        else:
            raise AssertionError("mixed judge failures must fail scoring")
        stale_attempts = stale_failed.info["testing_perf_judges"][0]["attempts"]
        assert stale_attempts[0]["response"] == "not-json"
        assert stale_attempts[1]["response"] is None
        assert stale_attempts[1]["error"]["status_code"] == 503
        assert "demo-token" not in str(stale_attempts[1]["error"])

        async def service_failed_complete(
            self, messages, *, trace=None, schema=None, **kwargs
        ):
            assert schema is TurnReview
            raise ConnectionError("mock judge is offline")

        TestingPerfJudge.complete = service_failed_complete
        service_failed = make_trace(make_task(), ["sort by use", "review the groups"])
        try:
            await make_task().score(service_failed)
        except vf.TaskError:
            pass
        else:
            raise AssertionError("exhausted service attempts must fail scoring")
        assert service_failed.rewards["testing_perf_reward"] is None
        assert service_failed.info["testing_perf"]["evaluation_status"] == (
            "unscored_service_failure"
        )
        assert len(service_failed.info["testing_perf_judges"][0]["attempts"]) == 3
    finally:
        TestingPerfJudge.complete = original_complete


if __name__ == "__main__":
    asyncio.run(run())
    print("testing-perf-env mocked framework check: ok")
