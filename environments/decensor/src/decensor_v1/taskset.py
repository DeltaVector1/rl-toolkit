from __future__ import annotations

import asyncio
from typing import Any

import verifiers.v1 as vf

from decensor_v1.agentic_review import (
    TranscriptReview,
    aggregate_review,
    build_turn_review_sources,
    review_attempts_per_phase,
    review_phase_count,
    run_agentic_review,
)
from decensor_v1.data import build_rows
from decensor_v1.judges import DecensorJudgeConfig
from decensor_v1.scoring import (
    EvaluationResult,
    ScoringConfig,
    Verdicts,
    find_model_response_failure,
    score_decensor,
)
from decensor_v1.service import (
    format_judge_failure_summary,
    judge_failure_summary,
    request_judge_score,
    safe_service_error,
)
from decensor_v1.state import DecensorState
from decensor_v1.text import (
    completion_contains_markdown,
    extract_all_assistant_texts,
    word_band,
)
from decensor_v1.user import DecensorUser, DecensorUserConfig


_review_slots = asyncio.Semaphore(2)


class DecensorData(vf.TaskData):
    word_requirement: str = ""
    word_target: int = 0
    response_format: str = "plaintext"
    has_style: bool = False
    target_turns: int = 1
    harm_type: str = "unknown"
    source_row_index: int = -1
    source_row_repeated: bool = False


class DecensorTaskConfig(vf.TaskConfig):
    user: DecensorUserConfig = DecensorUserConfig()
    judge: DecensorJudgeConfig = DecensorJudgeConfig()

    unclear_answer_penalty_multiplier: float = 0.1
    meta_commentary_multiplier: float = 0.01
    word_count_tolerance_percent: float = 0.60
    enable_word_count: bool = True
    enable_structured_markers: bool = True
    enable_markdown_judges: bool = True
    enable_reasoning_trace: bool = True
    enable_reasoning_trace_is_clear: bool = True
    enable_meta_commentary: bool = True
    reasoning_enabled: bool = True


class DecensorTasksetConfig(vf.TasksetConfig):
    num_examples: int = 2000
    dataset_seed: int = 42
    difficulty: str | None = None
    turn_decay_ratio: float = 0.5
    max_turns: int = 6
    style_probability: float = 0.25
    markdown_probability: float = 0.50
    enable_length_prompts: bool = True
    reasoning_output_prompt: str | None = None
    task: DecensorTaskConfig = DecensorTaskConfig()


def _scoring_config(config: DecensorTaskConfig) -> ScoringConfig:
    return ScoringConfig(
        unclear_answer_penalty_multiplier=config.unclear_answer_penalty_multiplier,
        meta_commentary_multiplier=config.meta_commentary_multiplier,
        word_count_tolerance_percent=config.word_count_tolerance_percent,
        enable_word_count=config.enable_word_count,
        enable_structured_markers=config.enable_structured_markers,
        enable_markdown_judges=config.enable_markdown_judges,
        enable_reasoning_trace=config.enable_reasoning_trace,
        enable_reasoning_trace_is_clear=config.enable_reasoning_trace_is_clear,
        enable_meta_commentary=config.enable_meta_commentary,
        reasoning_enabled=config.reasoning_enabled,
    )


class DecensorTask(vf.Task[DecensorData, DecensorState, DecensorTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.target_turns = self.data.target_turns

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        if trace.state.user_done:
            return True
        return trace.num_turns >= max(1, self.data.target_turns)

    @staticmethod
    def _message_dict(message: Any) -> dict[str, Any]:
        message_dict: dict[str, Any] = {
            "role": message.role,
            "content": message.content or "",
        }
        for key in (
            "reasoning",
            "reasoning_content",
            "thinking",
            "thinking_content",
            "thinking_blocks",
            "reasoning_details",
        ):
            value = getattr(message, key, None)
            if value:
                message_dict[key] = value
        return message_dict

    def _split_conversation(
        self, trace: vf.Trace
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        branches = trace.branches
        messages = branches[-1].messages if branches else []
        prompt: list[dict[str, Any]] = [
            {"role": "system", "content": self.data.system_prompt or ""}
        ]
        completion: list[dict[str, Any]] = []
        seen_first_user = False
        for message in messages:
            if message.role == "system":
                continue
            message_dict = self._message_dict(message)
            if not seen_first_user and message_dict["role"] == "user":
                prompt.append(message_dict)
                seen_first_user = True
            else:
                completion.append(message_dict)
        return prompt, completion

    @vf.reward(weight=1.0)
    async def decensor_reward(self, trace: vf.Trace) -> float:
        task_config = self.config
        scoring_config = _scoring_config(task_config)
        prompt, completion = self._split_conversation(trace)
        response_format = self.data.response_format
        length_range = (
            word_band(self.data.word_requirement, self.data.word_target)
            if self.data.word_requirement
            else None
        )
        trace.info["decensor_context"] = {
            "source_harm_type": self.data.harm_type,
            "source_row_index": self.data.source_row_index,
            "source_row_repeated": self.data.source_row_repeated,
            "response_format": response_format,
            "has_style": self.data.has_style,
            "word_requirement": self.data.word_requirement,
            "word_target": self.data.word_target,
            "full_credit_word_range": length_range,
            "follow_up_strategies": list(trace.state.follow_up_strategies),
            "follow_up_attempts": list(trace.state.follow_up_attempts),
        }

        response_failure, failure_details = find_model_response_failure(
            completion,
            is_truncated=trace.is_truncated,
            reasoning_required=scoring_config.reasoning_enabled,
            reject_forbidden_tags=scoring_config.enable_structured_markers,
        )
        if trace.state.user_sim_failed:
            final_cause = trace.state.user_sim_error
            if trace.state.follow_up_attempts:
                final_attempt = trace.state.follow_up_attempts[-1]
                if final_attempt.get("error"):
                    final_cause = str(final_attempt["error"])
            final_cause = safe_service_error(final_cause) or "unknown error"
            trace.info["decensor"] = {
                "evaluation_status": "unscored_service_failure",
                "service": "follow_up_model",
                "error": final_cause,
            }
            raise RuntimeError(
                "follow-up model failed after three attempts: "
                f"{final_cause}; rollout is unscored; its complete group will be replaced"
            )
        if response_failure is not None:
            invalid_result = EvaluationResult(
                status="invalid_model_response",
                reward=0.0,
                details={
                    "evaluation_status": "invalid_model_response",
                    "final_reward": 0.0,
                    "quality_reward": 0.0,
                    "response_rejected_because": response_failure,
                    "response_format": response_format,
                    **failure_details,
                },
            )
            response_details = invalid_result.details
            trace.info["decensor"] = response_details
            trace.record_metrics(
                {
                    key: float(value)
                    for key, value in response_details.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            reason = response_failure
            trace.record_metric("response_rejected", 1.0)
            trace.record_metric("decensor_zero_reward", 1.0)
            trace.record_metric("decensor_near_zero_reward", 1.0)
            trace.record_metric(f"response_rejected_because_{reason}", 1.0)
            trace.record_metric("decensor_target_turns", float(self.data.target_turns))
            trace.record_metric(
                "decensor_completed_turns",
                float(len(extract_all_assistant_texts(completion))),
            )
            trace.record_metric(
                "decensor_user_sim_failed", float(trace.state.user_sim_failed)
            )
            return 0.0

        review_sources = build_turn_review_sources(
            prompt,
            completion,
            system_prompt=self.data.system_prompt or "",
            response_format=response_format,
        )
        review_names = [
            f"turn_{source.turns[0].turn}_review" for source in review_sources
        ]

        async def review_one(name: str, source: Any):
            async with _review_slots:
                return await request_judge_score(
                    name,
                    lambda: run_agentic_review(task_config.judge, source),
                    attempt_count=3,
                    total_timeout_seconds=(
                        review_phase_count
                        * review_attempts_per_phase
                        * float(
                            getattr(task_config.judge, "timeout_seconds", 180.0)
                        )
                        + 10.0
                    ),
                )

        review_outcomes = await asyncio.gather(
            *(
                review_one(name, source)
                for name, source in zip(review_names, review_sources, strict=True)
            )
        )
        trace.info["judge"] = [
            attempt
            for outcome in review_outcomes
            for attempt in outcome.details.get("attempts", [])
        ]
        trace.info["judge_results"] = {
            name: {
                "status": outcome.status,
                "turn": source.turns[0].turn,
            }
            for name, source, outcome in zip(
                review_names, review_sources, review_outcomes, strict=True
            )
        }
        failed_reviews = [
            name
            for name, outcome in zip(review_names, review_outcomes, strict=True)
            if outcome.status != "scored" or outcome.score is None
        ]
        if failed_reviews:
            failed_judge_details = [
                judge_failure_summary(name, outcome)
                for name, outcome in zip(review_names, review_outcomes, strict=True)
                if outcome.status != "scored" or outcome.score is None
            ]
            failure_status = (
                "unscored_service_failure"
                if any(
                    outcome.status == "unscored_service_failure"
                    for outcome in review_outcomes
                )
                else "invalid_model_response"
            )
            trace.info["decensor"] = {
                "evaluation_status": failure_status,
                "failed_judges": failed_reviews,
                "failed_judge_attempts": failed_judge_details,
            }
            trace.record_metric(
                "decensor_judge_service_failure",
                float(failure_status == "unscored_service_failure"),
            )
            trace.record_metric(
                "decensor_invalid_judge_response",
                float(failure_status == "invalid_model_response"),
            )
            raise RuntimeError(
                "one or more per-turn reviews failed after their review attempts: "
                f"{format_judge_failure_summary(failed_judge_details)}; "
                "rollout is unscored"
            )
        turn_reviews = [
            TranscriptReview.model_validate(outcome.score).turns[0]
            for outcome in review_outcomes
        ]
        turn_reviews.sort(key=lambda item: item.turn)
        review = TranscriptReview(turns=turn_reviews)
        verdict_kwargs = aggregate_review(
            review,
            markdown_present=completion_contains_markdown(completion),
            response_format=response_format,
        )
        if not task_config.enable_reasoning_trace:
            verdict_kwargs["reasoning_has_safety_or_policy_discussion"] = None
        if not task_config.enable_reasoning_trace_is_clear:
            verdict_kwargs["reasoning_trace_is_clear"] = None
        if not task_config.enable_meta_commentary:
            verdict_kwargs["meta_commentary"] = None
        if not task_config.enable_markdown_judges:
            verdict_kwargs["markdown_present"] = None
            verdict_kwargs["markdown_is_correct"] = None

        trace.info["turn_reviews"] = review.model_dump()["turns"]
        trace.info["judge_review_aggregation"] = {
            "refusal": "maximum across turns",
            "visible_answer_is_clear": "fail if any turn fails",
            "reasoning_has_safety_or_policy_discussion": (
                "present if any turn is present"
            ),
            "reasoning_trace_is_clear": "fail if any turn fails",
            "meta_commentary": "present if any turn is present",
            "markdown_is_correct": "fail if any applicable turn fails",
        }
        trace.info["judge_review_score"] = verdict_kwargs
        trace.record_metric("decensor_agentic_review_turns", float(len(review.turns)))
        trace.record_metric(
            "decensor_reasoning_safety_policy_discussion_turn_rate",
            sum(
                turn.reasoning_has_safety_or_policy_discussion == "present"
                for turn in review.turns
            )
            / len(review.turns),
        )
        trace.record_metric(
            "decensor_reasoning_unclear_turn_rate",
            sum(turn.reasoning_trace_is_clear == "fail" for turn in review.turns)
            / len(review.turns),
        )
        trace.record_metric(
            "decensor_unclear_answer_turn_rate",
            sum(turn.visible_answer_is_clear == "fail" for turn in review.turns)
            / len(review.turns),
        )
        parse_failures = sum(
            attempt.get("status") == "invalid_judge_response"
            for outcome in review_outcomes
            for attempt in outcome.details.get("attempts", [])
        )

        result = score_decensor(
            prompt,
            completion,
            Verdicts(**verdict_kwargs),
            response_format=response_format,
            word_requirement=self.data.word_requirement,
            word_target=self.data.word_target,
            is_truncated=trace.is_truncated,
            config=scoring_config,
        )

        if result.status != "scored" or result.reward is None:
            trace.info["decensor"] = result.details
            raise RuntimeError("response evaluation ended without a score")
        reward = result.reward
        breakdown = result.details
        breakdown["source_harm_type"] = self.data.harm_type
        breakdown["follow_up_strategies"] = list(trace.state.follow_up_strategies)
        breakdown["follow_up_attempts"] = list(trace.state.follow_up_attempts)

        trace.info["decensor"] = breakdown
        trace.record_metrics(
            {
                key: float(value)
                for key, value in breakdown.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        trace.record_metric("decensor_judge_parse_failures", float(parse_failures))
        trace.record_metric(
            "response_rejected",
            0.0 if breakdown.get("response_rejected_because") is None else 1.0,
        )
        trace.record_metric("decensor_zero_reward", float(reward == 0.0))
        trace.record_metric("decensor_near_zero_reward", float(reward < 1e-3))
        rejection_reason = breakdown.get("response_rejected_because")
        for reason in (
            "truncated",
            "unclosed_think",
            "invalid_think_tags",
            "zero_words",
            "no_reasoning",
            "forbidden_internal_tag",
            "judge_endpoint_down",
            "user_sim_failed",
        ):
            trace.record_metric(
                f"response_rejected_because_{reason}",
                float(rejection_reason == reason),
            )
        trace.record_metric("decensor_target_turns", float(self.data.target_turns))
        trace.record_metric(
            "decensor_completed_turns",
            float(len(extract_all_assistant_texts(completion))),
        )
        trace.record_metric(
            "decensor_user_sim_failed", float(trace.state.user_sim_failed)
        )
        return reward


if hasattr(vf, "User"):
    DecensorTask.user = DecensorUser


class DecensorTaskset(vf.Taskset[DecensorTask, DecensorTasksetConfig]):
    def load(self) -> list[DecensorTask]:
        config = self.config
        rows = build_rows(
            num_examples=config.num_examples,
            seed=config.dataset_seed,
            turn_decay_ratio=config.turn_decay_ratio,
            max_turns=config.max_turns,
            style_probability=config.style_probability,
            markdown_probability=config.markdown_probability,
            enable_length_prompts=config.enable_length_prompts,
            output_prompt=config.reasoning_output_prompt,
            difficulty=config.difficulty,
        )
        tasks: list[DecensorTask] = []
        for index, row in enumerate(rows):
            tasks.append(
                DecensorTask(
                    DecensorData(
                        idx=index,
                        prompt=row["prompt"],
                        system_prompt=row["system_prompt"],
                        word_requirement=row["word_requirement"],
                        word_target=row["word_target"],
                        response_format=row["response_format"],
                        has_style=row["has_style"],
                        target_turns=row["target_turns"],
                        harm_type=row["harm_type"],
                        source_row_index=row["source_row_index"],
                        source_row_repeated=row["source_row_repeated"],
                    ),
                    config.task,
                )
            )
        return tasks


__all__ = ["DecensorTaskset"]
