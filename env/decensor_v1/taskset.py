from __future__ import annotations

import asyncio
import random
from typing import Any

import verifiers.v1 as vf

from decensor_v1.agentic_review import (
    TranscriptReview,
    aggregate_review,
    build_turn_review_sources,
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
from decensor_v1.service import request_judge_score
from decensor_v1.state import DecensorState
from decensor_v1.text import (
    completion_contains_markdown,
    extract_all_assistant_texts,
    word_band,
)
from decensor_v1.user import DecensorUser, DecensorUserConfig


class DecensorData(vf.TaskData):
    word_requirement: str = ""
    word_target: int = 0
    prompt_bucket: str = "styles"
    target_turns: int = 1
    harm_type: str = "unknown"
    source_row_index: int = -1
    source_row_repeated: bool = False


class DecensorTaskConfig(vf.TaskConfig):
    user: DecensorUserConfig = DecensorUserConfig()
    judge: DecensorJudgeConfig = DecensorJudgeConfig()

    incoherent_penalty_multiplier: float = 0.1
    meta_commentary_multiplier: float = 0.01
    word_count_tolerance_percent: float = 0.60
    markdown_exempt_probability: float = 0.25

    enable_word_count: bool = True
    enable_structured_markers: bool = True
    enable_markdown_judges: bool = True
    enable_reasoning_trace: bool = True
    enable_reasoning_coherency: bool = True
    enable_meta_commentary: bool = True
    reasoning_enabled: bool = True


class DecensorTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "NewEden/RL-seed-Decensor"
    num_examples: int = 2000
    dataset_seed: int = 42
    turn_decay_ratio: float = 0.5
    max_turns: int = 6
    bucket_styles_ratio: float = 0.25
    bucket_plaintext_ratio: float = 0.25
    bucket_markdown_ratio: float = 0.50
    enable_length_prompts: bool = True
    reasoning_output_prompt: str | None = None
    task: DecensorTaskConfig = DecensorTaskConfig()


def _scoring_config(cfg: DecensorTaskConfig) -> ScoringConfig:
    return ScoringConfig(
        incoherent_penalty_multiplier=cfg.incoherent_penalty_multiplier,
        meta_commentary_multiplier=cfg.meta_commentary_multiplier,
        word_count_tolerance_percent=cfg.word_count_tolerance_percent,
        enable_word_count=cfg.enable_word_count,
        enable_structured_markers=cfg.enable_structured_markers,
        enable_markdown_judges=cfg.enable_markdown_judges,
        enable_reasoning_trace=cfg.enable_reasoning_trace,
        enable_reasoning_coherency=cfg.enable_reasoning_coherency,
        enable_meta_commentary=cfg.enable_meta_commentary,
        reasoning_enabled=cfg.reasoning_enabled,
    )


class DecensorTask(vf.Task[DecensorData, DecensorState, DecensorTaskConfig]):
    if hasattr(vf, "User"):
        user = DecensorUser

        def user_server(self):
            user = super().user_server()
            user._task_data = self.data
            return user

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.target_turns = self.data.target_turns

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        if trace.state.user_done:
            return True
        return trace.num_turns >= max(1, self.data.target_turns)

    @staticmethod
    def _message_dict(message: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role, "content": message.content or ""}
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = getattr(message, key, None)
            if value:
                out[key] = value
        return out

    def _split_conversation(
        self, trace: vf.Trace
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        branches = trace.branches
        messages = branches[-1].messages if branches else []
        convo = [self._message_dict(m) for m in messages if m.role != "system"]
        prompt: list[dict[str, Any]] = [
            {"role": "system", "content": self.data.system_prompt or ""}
        ]
        completion: list[dict[str, Any]] = []
        seen_first_user = False
        for m in convo:
            if not seen_first_user and m["role"] == "user":
                prompt.append(m)
                seen_first_user = True
            else:
                completion.append(m)
        return prompt, completion

    @vf.reward(weight=1.0)
    async def decensor_reward(self, trace: vf.Trace) -> float:
        cfg = self.config
        scfg = _scoring_config(cfg)
        prompt, completion = self._split_conversation(trace)
        bucket = self.data.prompt_bucket
        md_exempt = (bucket == "styles") and (
            random.Random(self.data.idx).random() < cfg.markdown_exempt_probability
        )
        length_range = (
            word_band(self.data.word_requirement, self.data.word_target)
            if self.data.word_requirement
            else None
        )
        trace.info["decensor_context"] = {
            "source_harm_type": self.data.harm_type,
            "source_row_index": self.data.source_row_index,
            "source_row_repeated": self.data.source_row_repeated,
            "response_format": bucket,
            "word_requirement": self.data.word_requirement,
            "word_target": self.data.word_target,
            "full_credit_word_range": length_range,
            "markdown_exception_applied": md_exempt,
            "follow_up_strategies": list(trace.state.follow_up_strategies),
            "follow_up_attempts": list(trace.state.follow_up_attempts),
        }

        response_failure, failure_details = find_model_response_failure(
            completion,
            is_truncated=trace.is_truncated,
            reasoning_required=scfg.reasoning_enabled,
            reject_forbidden_tags=scfg.enable_structured_markers,
        )
        if trace.state.user_sim_failed:
            trace.info["decensor"] = {
                "evaluation_status": "unscored_service_failure",
                "service": "follow_up_model",
                "error": trace.state.user_sim_error,
            }
            raise RuntimeError(
                "follow-up model failed after three attempts; rollout is unscored"
            )
        if response_failure is not None:
            invalid_result = EvaluationResult(
                status="invalid_model_response",
                reward=0.0,
                details={
                    "evaluation_status": "invalid_model_response",
                    "final_reward": 0.0,
                    "quality_reward": 0.0,
                    "reasoning_requirement_pass": 0,
                    "advantage_scheme": "separate_reasoning_and_quality",
                    "early_termination": response_failure,
                    "prompt_bucket": bucket,
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
            trace.record_metric("decensor_gated", 1.0)
            trace.record_metric("decensor_zero_reward", 1.0)
            trace.record_metric("decensor_near_zero_reward", 1.0)
            trace.record_metric(f"decensor_gate_{reason}", 1.0)
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
            response_format=bucket,
        )
        review_names = [
            f"turn_{source.turns[0].turn}_review" for source in review_sources
        ]
        review_outcomes = await asyncio.gather(
            *(
                request_judge_score(
                    name,
                    lambda source=source: run_agentic_review(cfg.judge, source),
                )
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
            trace.info["decensor"] = {
                "evaluation_status": "unscored_service_failure",
                "failed_judges": failed_reviews,
            }
            trace.record_metric("decensor_judge_service_failure", 1.0)
            raise RuntimeError(
                "one or more per-turn reviews failed after three attempts; "
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
            response_format=bucket,
        )
        if not cfg.enable_reasoning_trace:
            verdict_kwargs["reasoning_ethics"] = None
        if not cfg.enable_reasoning_coherency:
            verdict_kwargs["reasoning_coherency"] = None
        if not cfg.enable_meta_commentary:
            verdict_kwargs["meta_commentary"] = None
        if not cfg.enable_markdown_judges:
            verdict_kwargs["md_presence"] = None
            verdict_kwargs["md_correctness"] = None

        trace.info["turn_reviews"] = review.model_dump()["turns"]
        trace.info["judge_review_aggregation"] = {
            "refusal": "maximum across turns",
            "coherency": "fail if any turn fails",
            "reasoning_ethics": "present if any turn is present",
            "reasoning_coherency": "fail if any turn fails",
            "meta_commentary": "present if any turn is present",
            "markdown_correctness": "fail if any applicable turn fails",
        }
        trace.info["judge_review_score"] = verdict_kwargs
        trace.record_metric("decensor_agentic_review_turns", float(len(review.turns)))
        trace.record_metric(
            "decensor_reasoning_restriction_turn_rate",
            sum(turn.reasoning_restriction == "present" for turn in review.turns)
            / len(review.turns),
        )
        trace.record_metric(
            "decensor_reasoning_incoherent_turn_rate",
            sum(turn.reasoning_coherence == "fail" for turn in review.turns)
            / len(review.turns),
        )
        trace.record_metric(
            "decensor_visible_incoherent_turn_rate",
            sum(turn.visible_coherence == "fail" for turn in review.turns)
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
            bucket=bucket,
            word_requirement=self.data.word_requirement,
            word_target=self.data.word_target,
            is_truncated=trace.is_truncated,
            md_exempt=md_exempt,
            config=scfg,
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
            "decensor_gated", 0.0 if breakdown.get("early_termination") is None else 1.0
        )
        trace.record_metric("decensor_zero_reward", float(reward == 0.0))
        trace.record_metric("decensor_near_zero_reward", float(reward < 1e-3))
        gate_reason = breakdown.get("early_termination")
        for reason in (
            "truncated",
            "unclosed_think",
            "zero_words",
            "no_reasoning",
            "structured_markers",
            "judge_endpoint_down",
            "user_sim_failed",
        ):
            trace.record_metric(f"decensor_gate_{reason}", float(gate_reason == reason))
        trace.record_metric("decensor_target_turns", float(self.data.target_turns))
        trace.record_metric(
            "decensor_completed_turns",
            float(len(extract_all_assistant_texts(completion))),
        )
        trace.record_metric(
            "decensor_user_sim_failed", float(trace.state.user_sim_failed)
        )
        return reward


class DecensorTaskset(vf.Taskset[DecensorTask, DecensorTasksetConfig]):
    def load(self) -> list[DecensorTask]:
        c = self.config
        rows = build_rows(
            ds_name=c.dataset_name,
            num_examples=c.num_examples,
            seed=c.dataset_seed,
            turn_decay_ratio=c.turn_decay_ratio,
            max_turns=c.max_turns,
            bucket_styles_ratio=c.bucket_styles_ratio,
            bucket_plaintext_ratio=c.bucket_plaintext_ratio,
            bucket_markdown_ratio=c.bucket_markdown_ratio,
            enable_length_prompts=c.enable_length_prompts,
            output_prompt=c.reasoning_output_prompt,
        )
        tasks: list[DecensorTask] = []
        for idx, row in enumerate(rows):
            tasks.append(
                DecensorTask(
                    DecensorData(
                        idx=idx,
                        prompt=row["prompt"],
                        system_prompt=row["system_prompt"],
                        word_requirement=row["word_requirement"],
                        word_target=row["word_target"],
                        prompt_bucket=row["prompt_bucket"],
                        target_turns=row["target_turns"],
                        harm_type=row["harm_type"],
                        source_row_index=row["source_row_index"],
                        source_row_repeated=row["source_row_repeated"],
                    ),
                    c.task,
                )
            )
        return tasks


__all__ = ["DecensorTaskset"]
