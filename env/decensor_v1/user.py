from __future__ import annotations

import asyncio
import importlib
import json
import random
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import verifiers.v1 as vf
from openai import APIError, AsyncOpenAI
from pydantic import Field

try:
    from verifiers.v1.clients.config import resolve_api_key
except ImportError:
    from verifiers.v1.configs.client import resolve_api_key
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.null.harness import NullHarness

try:
    from verifiers.v1.mcp.user import UserConfig
except ImportError:
    from pydantic_config import BaseConfig

    class UserConfig(BaseConfig):
        in_process: bool = True


try:
    from verifiers.v1.runtimes import ProgramResult
except ImportError:
    from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.types import UserMessage

from decensor_v1.state import DecensorState
from decensor_v1.follow_up import (
    FollowUpReview,
    choose_follow_up_strategy,
    follow_up_generation_instructions,
    follow_up_instructions_by_strategy,
    follow_up_review_instructions,
    follow_up_review_tool,
    review_result_name,
    user_turn_seed,
)
from decensor_v1.text import (
    has_visible_assistant_answer,
    message_content,
    message_role,
    strip_think_tags,
    validate_follow_up_message,
)

_http = httpx.AsyncClient(
    verify=False,
    trust_env=False,
    timeout=httpx.Timeout(180.0, connect=30.0),
    limits=httpx.Limits(max_connections=None, max_keepalive_connections=1024),
)
_rollout_http = httpx.AsyncClient(
    verify=False,
    trust_env=False,
    timeout=httpx.Timeout(1200.0, connect=30.0),
    limits=httpx.Limits(max_connections=None, max_keepalive_connections=1024),
)
_openai_clients = {}


@asynccontextmanager
async def _shared_openai(config):
    key = (
        config.base_url,
        config.api_key_var,
        tuple(sorted(config.headers.items())),
    )
    if key not in _openai_clients:
        _openai_clients[key] = AsyncOpenAI(
            base_url=config.base_url,
            api_key=resolve_api_key(config),
            default_headers=config.headers or None,
            http_client=_http,
            max_retries=0,
        )
    yield _openai_clients[key]


def parse_user_turn_output(raw: str) -> str:
    text, _, _ = validate_follow_up_message(raw)
    return text


async def review_follow_up_message(
    client: AsyncOpenAI,
    config: Any,
    *,
    original_request: str,
    latest_assistant_answer: str,
    proposed_user_message: str,
    strategy: str,
    seed: int,
    extra_body: dict[str, Any],
) -> tuple[FollowUpReview, str]:
    tool = follow_up_review_tool()
    completion = await asyncio.wait_for(
        client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": follow_up_review_instructions},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_request": original_request,
                            "latest_assistant_answer": latest_assistant_answer,
                            "proposed_user_message": proposed_user_message,
                            "assigned_strategy": strategy,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": review_result_name},
            },
            parallel_tool_calls=False,
            temperature=0.0,
            max_tokens=768,
            seed=seed,
            **({"extra_body": extra_body} if extra_body else {}),
        ),
        timeout=90.0,
    )
    calls = list(completion.choices[0].message.tool_calls or [])
    if len(calls) != 1 or calls[0].function.name != review_result_name:
        raise ValueError("follow-up reviewer returned no valid review tool call")
    arguments = str(calls[0].function.arguments)
    return FollowUpReview.model_validate_json(arguments), arguments


class DecensorUserConfig(UserConfig):
    in_process: bool = True
    model: str = "google/gemma-4-26B-A4B-it"
    base_url: str = "http://198.13.252.41:40002/v1"
    api_key_var: str = "VLLM_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 1.0
    min_p: float = 0.0
    max_tokens: int = 256
    seed: int = 0


class _RemovedUser:
    def __class_getitem__(cls, _args):
        return cls

    @classmethod
    def run(cls) -> None:
        raise RuntimeError("the v0.8 environment drives DecensorUser in process")


_UserBase = getattr(vf, "User", _RemovedUser)


class DecensorUser(_UserBase[DecensorUserConfig, DecensorState]):
    async def setup_task(self, task: Any) -> None:
        self._transcript: list[dict[str, str]] = []
        self._initial_user_request = ""
        self._accepted_follow_ups: list[str] = []
        self._rollout_nonce = secrets.randbits(63)
        system_prompt = getattr(task, "system_prompt", None)
        if system_prompt:
            self._transcript.append({"role": "system", "content": str(system_prompt)})
        prompt = getattr(task, "prompt", None)
        if isinstance(prompt, str):
            self._transcript.append({"role": "user", "content": prompt})
            self._initial_user_request = prompt
        elif isinstance(prompt, list):
            for m in prompt:
                role = message_role(m)
                content = message_content(m)
                self._transcript.append({"role": role, "content": content})
                if role == "user" and not self._initial_user_request:
                    self._initial_user_request = content

        task_idx = int(getattr(task, "idx", 0) or 0)
        self._task_seed = self.config.seed + task_idx * 1_000_003

    async def respond(self, message: str) -> vf.Messages:
        self._transcript.append({"role": "assistant", "content": message})
        produced_turns = sum(
            1 for m in self._transcript if message_role(m) == "assistant"
        )
        if produced_turns >= max(1, self.state.target_turns):
            self.state.user_done = True
            return []

        last_asst = strip_think_tags(message)
        if not has_visible_assistant_answer(message):
            self.state.user_done = True
            return []
        first_attempt_seed = user_turn_seed(
            self._task_seed,
            self._rollout_nonce,
            produced_turns,
            last_asst,
            0,
        )
        strategy_name = choose_follow_up_strategy(
            last_asst,
            random.Random(first_attempt_seed),
        )
        self.state.follow_up_strategies.append(strategy_name)
        strategy_instruction = follow_up_instructions_by_strategy[strategy_name]
        prior_user_messages = [
            message_content(item)
            for item in self._transcript
            if message_role(item) == "user"
        ]

        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        if self.config.min_p:
            extra_body["min_p"] = self.config.min_p
        text = ""
        final_error = "unparseable_reply"
        retry_instruction = ""
        try:
            async with asyncio.timeout(550.0):
                async with _shared_openai(self.config) as client:
                    for attempt in range(3):
                        attempt_started_at = time.monotonic()
                        generation_seed = user_turn_seed(
                            self._task_seed,
                            self._rollout_nonce,
                            produced_turns,
                            last_asst,
                            attempt,
                        )
                        request_data = {
                            "original_request": self._initial_user_request,
                            "prior_user_messages": prior_user_messages,
                            "latest_assistant_answer": last_asst,
                            "assigned_strategy": strategy_name,
                            "strategy_instruction": strategy_instruction,
                        }
                        if retry_instruction:
                            request_data["correction_for_this_attempt"] = (
                                retry_instruction
                            )
                        try:
                            completion = await asyncio.wait_for(
                                client.chat.completions.create(
                                    model=self.config.model,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": follow_up_generation_instructions,
                                        },
                                        {
                                            "role": "user",
                                            "content": json.dumps(
                                                request_data,
                                                ensure_ascii=False,
                                            ),
                                        },
                                    ],
                                    temperature=self.config.temperature,
                                    max_tokens=self.config.max_tokens,
                                    seed=generation_seed,
                                    extra_body=extra_body,
                                ),
                                timeout=180.0,
                            )
                            raw_text = str(
                                completion.choices[0].message.content or ""
                            )
                            candidate, rejection, validation = (
                                validate_follow_up_message(
                                    raw_text,
                                    tuple(self._accepted_follow_ups),
                                )
                            )
                            attempt_record = {
                                "turn": produced_turns + 1,
                                "attempt": attempt + 1,
                                "strategy": strategy_name,
                                "generation_seed": generation_seed,
                                "text": raw_text,
                                "parsed_message": candidate,
                                "local_validation": validation,
                            }
                            if rejection is not None:
                                final_error = rejection
                                retry_instruction = (
                                    f"The previous output failed local validation: "
                                    f"{rejection}. Produce a different valid message."
                                )
                                attempt_record["status"] = "rejected_local"
                                attempt_record["rejection_reason"] = rejection
                                attempt_record["elapsed_seconds"] = (
                                    time.monotonic() - attempt_started_at
                                )
                                self.state.follow_up_attempts.append(attempt_record)
                                continue

                            review_seed = user_turn_seed(
                                self._task_seed,
                                self._rollout_nonce,
                                produced_turns,
                                candidate,
                                attempt + 101,
                            )
                            review, review_raw = await review_follow_up_message(
                                client,
                                self.config,
                                original_request=self._initial_user_request,
                                latest_assistant_answer=last_asst,
                                proposed_user_message=candidate,
                                strategy=strategy_name,
                                seed=review_seed,
                                extra_body=extra_body,
                            )
                            attempt_record["review_seed"] = review_seed
                            attempt_record["review_text"] = review_raw
                            attempt_record["review"] = review.model_dump()
                            if not review.accepted():
                                final_error = "semantic_review_rejected"
                                retry_instruction = (
                                    f"{review.correction()} The review found: "
                                    f"{review.reason}"
                                )
                                attempt_record["retry_instruction"] = (
                                    retry_instruction
                                )
                                attempt_record["status"] = "rejected_semantic"
                                attempt_record["elapsed_seconds"] = (
                                    time.monotonic() - attempt_started_at
                                )
                                self.state.follow_up_attempts.append(attempt_record)
                                continue
                            text = candidate
                            attempt_record["status"] = "accepted"
                            attempt_record["elapsed_seconds"] = (
                                time.monotonic() - attempt_started_at
                            )
                            self.state.follow_up_attempts.append(attempt_record)
                        except (
                            APIError,
                            httpx.HTTPError,
                            TimeoutError,
                            IndexError,
                            AttributeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            final_error = type(exc).__name__
                            text = ""
                            self.state.follow_up_attempts.append(
                                {
                                    "turn": produced_turns + 1,
                                    "attempt": attempt + 1,
                                    "status": "request_failed",
                                    "strategy": strategy_name,
                                    "generation_seed": generation_seed,
                                    "error_type": final_error,
                                    "error": str(exc),
                                    "elapsed_seconds": time.monotonic()
                                    - attempt_started_at,
                                }
                            )
                        if text:
                            break
        except TimeoutError:
            final_error = "total_timeout"
            text = ""
            self.state.follow_up_attempts.append(
                {
                    "turn": produced_turns + 1,
                    "attempt": None,
                    "status": "total_timeout",
                }
            )
        if not text:
            self.state.user_sim_failed = True
            self.state.user_done = True
            self.state.user_sim_error = final_error
            return []
        self._accepted_follow_ups.append(text)
        self._transcript.append({"role": "user", "content": text})
        return [UserMessage(content=text)]


_null_launch = NullHarness.launch


async def _shared_null_launch(
    self,
    ctx,
    trace,
    runtime,
    endpoint,
    secret,
    mcp_urls,
):
    if mcp_urls:
        return await _null_launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls)

    system_prompt, prompt = self.resolve_prompt(trace.task.data)
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif prompt is not None:
        messages.extend(message_to_wire(message) for message in prompt)
    response = await asyncio.wait_for(
        _rollout_http.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
            json={"model": ctx.model, "messages": messages},
        ),
        timeout=1200.0,
    )
    if response.is_error and trace.stop_condition is None:
        response.raise_for_status()
    return ProgramResult(exit_code=0, stdout="", stderr="")


def _enable_shared_clients() -> None:
    judge = importlib.import_module("verifiers.v1.judge")
    judge.build_async_openai = _shared_openai
    if hasattr(vf, "User"):
        NullHarness.launch = _shared_null_launch


def _enable_in_process_user() -> None:
    rollout = importlib.import_module("verifiers.v1.rollout")

    @asynccontextmanager
    async def serve_user(
        user,
        harness_runtime=None,
        *,
        state_secret="",
        state_base=None,
    ):
        if user is None:
            yield None
            return

        state_url = f"{state_base.rstrip('/')}/state"
        user._state_channel = lambda: (state_url, state_secret)
        await user.setup()
        await user.setup_task(user._task_data)
        synced = user._with_state(user.respond)
        lock = asyncio.Lock()
        last_turn = None
        last_payload = []

        async def respond(message, seq=-1):
            nonlocal last_turn, last_payload
            async with lock:
                if seq >= 0 and (seq, message) == last_turn:
                    return last_payload
                last_payload = await synced(message)
                last_turn = (seq, message)
                return last_payload

        try:
            yield respond
        finally:
            await user._exit_stack.aclose()

    rollout.serve_user = serve_user


_enable_shared_clients()
if hasattr(vf, "User"):
    _enable_in_process_user()


if __name__ == "__main__" and hasattr(vf, "User"):
    DecensorUser.run()
