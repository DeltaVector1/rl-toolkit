"""Focused compatibility check for the original verifiers.v1 integration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib

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
    _word_band,
    _sample_turn_counts,
)
from testing_perf_env.state import TestingPerfState  # noqa: E402
from testing_perf_env.user import TestingPerfUser, TestingPerfUserConfig  # noqa: E402


class FakeAssistant:
    def __init__(self):
        self.calls = 0

    async def run(self, task):
        self.calls += 1


class FakeAgents:
    def __init__(self):
        self.assistant = FakeAssistant()


class FakeServiceError(ConnectionError):
    status_code = 503


def make_task(target_turns: int = 2, max_attempts: int = 3) -> TestingPerfTask:
    return TestingPerfTask(
        TestingPerfData(
            idx=0,
            prompt="Make a short plan for sorting a desk. Use plain text.",
            task_id="desk-plan",
            requirements=["name two groups", "include one review step"],
            follow_ups=[
                "Add one way to review the plan.",
                "Group the plan into two sections.",
                "Add one practical reminder.",
                "Summarise the plan in one line.",
                "Give the plan a short title.",
            ],
            target_turns=target_turns,
            target_words=300,
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
        state=TestingPerfState(),
        agent=vf.AgentInfo(model="mock", harness=vf.HarnessConfig(id="null")),
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
        os.path.join(check_dir, "..", "..", "configs", "operations.toml"), "rb"
    ) as stream:
        operations_config = tomllib.load(stream)
    profiles = operations_config["profiles"]
    assert profiles["current"]["orchestrator_image"] == "prime-rl:glm52-decensor"
    assert profiles["testing-perf"]["orchestrator_image"] == (
        "prime-rl:glm52-decensor"
    )
    with open(
        os.path.join(
            check_dir, "..", "..", "configs", "testing-perf", "orchestrator.toml"
        ),
        "rb",
    ) as stream:
        orchestrator_config = tomllib.load(stream)
    assert "runtime" not in orchestrator_config["train"]["env"][0]["env"]["assistant"]
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
    loaded_tasks = taskset.select()
    assert [task.data.target_turns for task in loaded_tasks] == [1, 2, 3, 1, 1]
    assert [task.data.target_words for task in loaded_tasks] == [
        163,
        300,
        1400,
        163,
        300,
    ]
    assert "Respond tersely" in loaded_tasks[0].data.prompt_text
    assert "about 163 words in each answer (acceptable range 75-250)" in loaded_tasks[0].data.prompt_text
    assert "near 163 words (75-250)" in loaded_tasks[0].data.follow_ups[0]
    sampled_turns = _sample_turn_counts(2000, 6, 42)
    assert [
        sampled_turns.count(turn)
        for turn in range(1, 7)
    ] == [1016, 508, 254, 127, 63, 32]
    for total in range(1, 201):
        for max_turns in range(1, 7):
            sampled = _sample_turn_counts(total, max_turns, 42)
            assert len(sampled) == total
            assert all(1 <= turn <= max_turns for turn in sampled)
    try:
        _word_band(400)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported word target accepted")
    for max_turns in range(1, 7):
        bounded = TestingPerfTasksetConfig(
            id="testing-perf-env", num_examples=1, max_turns=max_turns
        )
        assert 1 <= vf.load_taskset(bounded).select(1)[0].data.target_turns <= max_turns
    env_config = vf.resolve_env_config(
        {
            "id": "testing-perf-env",
            "taskset": {"id": "testing-perf-env", "num_examples": 1},
        }
    )
    env = vf.load_environment(env_config)
    assert isinstance(env, TestingPerfEnv)

    agents = FakeAgents()
    await env.run(make_task(), agents)
    assert agents.assistant.calls == 1

    from verifiers.v1.mcp.server import _call_state

    for target_turns in (1, 2, 6):
        task = make_task(target_turns=target_turns)
        trace = make_trace(task, [])
        await task.setup(trace, None)
        user = TestingPerfUser(TestingPerfUserConfig())
        token = _call_state.set(trace.state)
        try:
            await user.setup_task(task.data)
            follow_ups = []
            for index in range(target_turns - 1):
                messages = await user.respond(f"answer-{index}")
                assert len(messages) == 1
                follow_ups.append(messages[0].content)
            assert await user.respond("final-answer") == []
        finally:
            _call_state.reset(token)
        assert follow_ups == task.data.follow_ups[: target_turns - 1]
        assert trace.state.next_follow_up == target_turns - 1
        assert not await task.done(make_trace(task, ["reply"] * (target_turns - 1)))
        assert await task.done(make_trace(task, ["reply"] * target_turns))

    original_complete = TestingPerfJudge.complete
    responses = [
        json.dumps(
            {
                "correctness": 1.0,
                "task_adherence": 1.0,
                "coherence": 1.0,
                "evidence": ["covers the requested groups"],
            }
        )
    ] * 2

    async def mocked_complete(self, messages, *, trace=None, schema=None, **kwargs):
        assert schema is TurnReview
        assert "Word count is scored separately" in messages
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
        exact_answer = "word " * task.data.target_words
        trace = make_trace(task, [exact_answer, exact_answer])
        await task.score(trace)
        exact_score = trace.rewards["testing_perf_reward"]
        assert exact_score == 1.0
        assert trace.info["testing_perf"]["evaluation_status"] == "scored"
        assert trace.info["testing_perf"]["target_words"] == 300
        assert trace.info["testing_perf"]["word_counts"] == [300, 300]
        assert trace.info["testing_perf"]["length_scores"] == [1.0, 1.0]
        assert trace.info["testing_perf"]["judge_calls"] == 2
        assert trace.info["testing_perf"]["judge_input_tokens"] == 4
        assert trace.info["testing_perf"]["judge_output_tokens"] == 10
        assert trace.info["testing_perf_judges"][0]["prompt"]

        responses.extend(
            [
                json.dumps(
                    {
                        "correctness": 1.0,
                        "task_adherence": 1.0,
                        "coherence": 1.0,
                        "evidence": ["covers the requested groups"],
                    }
                )
            ]
            * 2
        )
        short_trace = make_trace(task, ["word " * 100] * 2)
        await task.score(short_trace)
        short_score = short_trace.rewards["testing_perf_reward"]
        assert 0.0 < short_score < exact_score
        assert short_trace.info["testing_perf"]["word_counts"] == [100, 100]
        assert short_trace.info["testing_perf"]["length_scores"] == [0.5, 0.5]

        responses.extend(
            [
                json.dumps(
                    {
                        "correctness": 1.0,
                        "task_adherence": 1.0,
                        "coherence": 1.0,
                        "evidence": ["covers the requested groups"],
                    }
                )
            ]
            * 2
        )
        overlong_trace = make_trace(task, ["word " * 800] * 2)
        await task.score(overlong_trace)
        assert overlong_trace.rewards["testing_perf_reward"] == short_score
        assert overlong_trace.info["testing_perf"]["word_counts"] == [800, 800]
        assert overlong_trace.info["testing_perf"]["length_scores"] == [0.5, 0.5]

        responses.extend(
            [
                json.dumps(
                    {
                        "correctness": 0.0,
                        "task_adherence": 0.0,
                        "coherence": 0.0,
                        "evidence": ["no quality evidence"],
                    }
                )
            ]
            * 2
        )
        zero_trace = make_trace(task, [exact_answer, exact_answer])
        await task.score(zero_trace)
        assert zero_trace.rewards["testing_perf_reward"] == 0.25

        empty_trace = make_trace(task, ["", "review the groups"])
        await task.score(empty_trace)
        assert empty_trace.rewards["testing_perf_reward"] == 0
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
        assert failed.rewards.get("testing_perf_reward") is None
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
        assert service_failed.rewards.get("testing_perf_reward") is None
        assert service_failed.info["testing_perf"]["evaluation_status"] == (
            "unscored_service_failure"
        )
        assert len(service_failed.info["testing_perf_judges"][0]["attempts"]) == 3
    finally:
        TestingPerfJudge.complete = original_complete


if __name__ == "__main__":
    asyncio.run(run())
    print("testing-perf-env original-image compatibility check: ok")
