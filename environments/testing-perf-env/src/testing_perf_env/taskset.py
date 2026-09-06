from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from testing_perf_env.judge import TestingPerfJudge, TestingPerfJudgeConfig, TurnReview
from testing_perf_env.state import TestingPerfState
from testing_perf_env.user import TestingPerfUser, TestingPerfUserConfig


def _safe_error(error: Exception) -> dict[str, Any]:
    message = " ".join(str(error).split())
    message = re.sub(
        r"(?i)(authorization|api[-_ ]?key|token)(\s*[:=]\s*)([^,; ]+)",
        r"\1\2[removed]",
        message,
    )
    message = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [removed]", message)
    message = re.sub(r"(?i)\b(?:sk|pk)-[A-Za-z0-9_-]+", "[removed]", message)
    return {
        "type": type(error).__name__,
        "message": message[:400] or type(error).__name__,
        "status_code": getattr(error, "status_code", None),
    }


class TestingPerfData(vf.TaskData):
    task_id: str
    requirements: list[str]
    follow_ups: list[str]
    target_turns: int = Field(ge=2, le=6)
    target_words: int = Field(ge=400, le=700)
    response_format: Literal["plain", "markdown"]


class TestingPerfTaskConfig(vf.TaskConfig):
    user: TestingPerfUserConfig = TestingPerfUserConfig()
    judge: TestingPerfJudgeConfig = TestingPerfJudgeConfig()


class TestingPerfTasksetConfig(vf.TasksetConfig):
    num_examples: int = Field(default=8, ge=1)
    dataset_seed: int = 42
    max_turns: int = Field(default=4, ge=2, le=6)
    markdown_ratio: float = Field(default=0.5, ge=0, le=1)
    task: TestingPerfTaskConfig = TestingPerfTaskConfig()


_task_templates: tuple[dict[str, Any], ...] = (
    {
        "task_id": "trip-checklist",
        "topic": "a two-day rainy city trip",
        "target_words": 400,
        "requirements": [
            "include a waterproof jacket",
            "include comfortable shoes",
            "include a phone charger",
        ],
        "follow_ups": [
            "Add a short morning and evening routine to the checklist.",
            "Group the checklist into items to pack and items to do before leaving.",
            "End with one sentence about keeping the plan easy to scan.",
            "Add one practical reminder for checking the weather.",
            "Give the final checklist a short, descriptive title.",
        ],
    },
    {
        "task_id": "file-plan",
        "topic": "organising digital project files",
        "target_words": 500,
        "requirements": [
            "use folders for source, notes, and exports",
            "give files a consistent date format",
            "include one backup step",
        ],
        "follow_ups": [
            "Add a small example folder tree.",
            "Explain how to handle an old draft without deleting it.",
            "Summarise the naming rule in one line.",
            "Add one step for checking that the backup is current.",
            "Give the final plan a short, descriptive title.",
        ],
    },
    {
        "task_id": "study-plan",
        "topic": "a one-week study plan for a history exam",
        "target_words": 600,
        "requirements": [
            "divide work across five weekdays",
            "include review and practice questions",
            "include one rest period",
        ],
        "follow_ups": [
            "Add a simple way to track completed sessions.",
            "Make the final day focus on recall instead of new material.",
            "Add one short tip for adjusting the plan when a session is missed.",
            "Add a brief reminder to check progress at the end of the week.",
            "Give the final plan a short, descriptive title.",
        ],
    },
    {
        "task_id": "garden-plan",
        "topic": "a small balcony herb garden",
        "target_words": 700,
        "requirements": [
            "name three herbs",
            "mention sunlight and drainage",
            "include a weekly care reminder",
        ],
        "follow_ups": [
            "Put the care reminders into a compact list.",
            "Add a note about labelling each pot.",
            "Close with a two-sentence summary for a beginner.",
            "Add one reminder to check the soil before watering.",
            "Give the final plan a short, descriptive title.",
        ],
    },
)


def _prompt(row: dict[str, Any], response_format: str) -> str:
    format_request = (
        "Use Markdown headings and bullets."
        if response_format == "markdown"
        else "Use plain text with short paragraphs."
    )
    requirements = "; ".join(row["requirements"])
    return (
        f"Create a practical plan for {row['topic']}. {format_request} "
        f"Aim for about {row['target_words']} words in each answer. "
        f"The first answer must address these points: {requirements}."
    )


class TestingPerfTask(vf.Task[TestingPerfData, TestingPerfState, TestingPerfTaskConfig]):
    user = TestingPerfUser

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.target_turns = self.data.target_turns
        trace.state.follow_ups = list(self.data.follow_ups)

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.data.target_turns

    @staticmethod
    def _visible_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content or "").strip()

    def _prompt_for_turn(self, turn: int, answers: list[str]) -> str:
        user_requests = [self.data.prompt_text, *self.data.follow_ups]
        conversation: list[dict[str, str]] = []
        for index, answer in enumerate(answers):
            conversation.append({"role": "user", "content": user_requests[index]})
            conversation.append({"role": "assistant", "content": answer})
        payload = {
            "task": self.data.prompt_text,
            "requirements": self.data.requirements,
            "response_format": self.data.response_format,
            "target_words": self.data.target_words,
            "turn": turn,
            "current_user_request": user_requests[turn - 1],
            "visible_response": answers[-1],
            "conversation_so_far": conversation,
        }
        return (
            "Review only the visible assistant response. Do not assess hidden "
            "reasoning. Return one JSON object with numeric fields correctness, "
            "task_adherence, coherence in [0, 1], and an evidence array with "
            "one to four short strings. Use 1.0 for complete work, 0.5 for "
            "partial work, and 0.0 for absent or contradictory work. Higher "
            "numbers mean better work. Word count is scored separately; do not "
            "use it when assigning these three scores.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    async def _review_turn(
        self,
        trace: vf.Trace,
        judge: TestingPerfJudge,
        turn: int,
        answers: list[str],
    ) -> TurnReview:
        prompt = self._prompt_for_turn(turn, answers)
        config = self.config.judge
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, config.max_attempts + 1):
            raw: str | None = None
            try:
                response = await asyncio.wait_for(
                    judge.complete(prompt, trace=trace, schema=TurnReview),
                    timeout=config.timeout_seconds,
                )
                raw = response.text
                if response.parsed is None:
                    raise ValueError("judge returned no structured review")
                review = TurnReview.model_validate(response.parsed)
            # Judge service and strict-parser failures retry alike.
            except Exception as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": (
                            "invalid_judge_response"
                            if isinstance(error, ValueError)
                            else "service_error"
                        ),
                        "response": raw,
                        "error": _safe_error(error),
                    }
                )
                if attempt < config.max_attempts:
                    await asyncio.sleep(config.retry_delay_seconds)
                continue
            attempts.append({"attempt": attempt, "status": "scored", "response": raw})
            trace.info.setdefault("testing_perf_judges", []).append(
                {"turn": turn, "prompt": prompt, "attempts": attempts}
            )
            return review

        trace.info.setdefault("testing_perf_judges", []).append(
            {"turn": turn, "prompt": prompt, "attempts": attempts}
        )
        service_failure = any(item["status"] == "service_error" for item in attempts)
        status = (
            "unscored_service_failure"
            if service_failure
            else "unscored_invalid_judge_response"
        )
        attempts_seen = [
            attempt
            for item in trace.info.get("testing_perf_judges", [])
            for attempt in item["attempts"]
        ]
        trace.info["testing_perf"].update(
            {
                "evaluation_status": status,
                "judge_turn": turn,
                "judge_calls": len(attempts_seen),
                "judge_reviews": turn - 1,
                "judge_parse_failures": sum(
                    item["status"] == "invalid_judge_response" for item in attempts_seen
                ),
                "judge_input_tokens": sum(
                    getattr(usage, "input_tokens", 0) for usage in trace.extra_usage
                ),
                "judge_output_tokens": sum(
                    getattr(usage, "completion_tokens", 0)
                    for usage in trace.extra_usage
                ),
            }
        )
        trace.record_metric("testing_perf_judge_calls", float(len(attempts_seen)))
        trace.record_metric("testing_perf_judge_reviews", float(turn - 1))
        trace.record_metric(
            "testing_perf_judge_input_tokens",
            float(trace.info["testing_perf"]["judge_input_tokens"]),
        )
        trace.record_metric(
            "testing_perf_judge_output_tokens",
            float(trace.info["testing_perf"]["judge_output_tokens"]),
        )
        raise RuntimeError(f"{status} after {len(attempts)} judge attempts")

    @vf.reward
    async def testing_perf_reward(self, trace: vf.Trace) -> float:
        answers = [self._visible_text(message) for message in trace.assistant_messages]
        target_turns = self.data.target_turns
        target_words = self.data.target_words
        word_counts = [len(answer.split()) for answer in answers]
        length_scores = [
            0.0
            if word_count == 0
            else min(word_count, target_words) / max(word_count, target_words)
            for word_count in word_counts
        ]
        trace.info["testing_perf"] = {
            "target_turns": target_turns,
            "completed_turns": len(answers),
            "response_format": self.data.response_format,
            "visible_output_only": True,
            "target_words": target_words,
            "word_counts": word_counts,
            "length_scores": length_scores,
        }
        trace.record_metric("testing_perf_target_turns", float(target_turns))
        trace.record_metric("testing_perf_completed_turns", float(len(answers)))

        if len(answers) != target_turns:
            trace.info["testing_perf"].update(
                {
                    "evaluation_status": "invalid_model_response",
                    "response_rejected_because": "wrong_turn_count",
                }
            )
            return 0.0
        if any(not answer for answer in answers):
            trace.info["testing_perf"].update(
                {
                    "evaluation_status": "invalid_model_response",
                    "response_rejected_because": "empty_visible_answer",
                }
            )
            return 0.0

        judge = TestingPerfJudge(self.config.judge)
        reviews = [
            await self._review_turn(trace, judge, turn, answers[:turn])
            for turn in range(1, target_turns + 1)
        ]
        scores = [
            (
                review.correctness
                + review.task_adherence
                + review.coherence
                + length_score
            )
            / 4
            for review, length_score in zip(reviews, length_scores, strict=True)
        ]
        score = sum(scores) / len(scores)
        attempts = [
            attempt
            for item in trace.info.get("testing_perf_judges", [])
            for attempt in item["attempts"]
        ]
        judge_input_tokens = sum(
            getattr(usage, "input_tokens", 0) for usage in trace.extra_usage
        )
        judge_output_tokens = sum(
            getattr(usage, "completion_tokens", 0) for usage in trace.extra_usage
        )
        trace.info["testing_perf"].update(
            {
                "evaluation_status": "scored",
                "score": score,
                "target_words": target_words,
                "word_counts": word_counts,
                "length_scores": length_scores,
                "turn_reviews": [review.model_dump() for review in reviews],
                "judge_calls": len(attempts),
                "judge_reviews": len(reviews),
                "judge_parse_failures": sum(
                    item["status"] == "invalid_judge_response" for item in attempts
                ),
                "judge_input_tokens": judge_input_tokens,
                "judge_output_tokens": judge_output_tokens,
                "judge_total_tokens": judge_input_tokens + judge_output_tokens,
            }
        )
        trace.record_metric("testing_perf_judge_calls", float(len(attempts)))
        trace.record_metric("testing_perf_judge_reviews", float(len(reviews)))
        trace.record_metric(
            "testing_perf_judge_input_tokens", float(judge_input_tokens)
        )
        trace.record_metric(
            "testing_perf_judge_output_tokens", float(judge_output_tokens)
        )
        return score


class TestingPerfTaskset(vf.Taskset[TestingPerfTask, TestingPerfTasksetConfig]):
    def load(self) -> list[TestingPerfTask]:
        config = self.config
        tasks: list[TestingPerfTask] = []
        for index in range(config.num_examples):
            row = _task_templates[(index + config.dataset_seed) % len(_task_templates)]
            response_format = (
                "markdown"
                if (index * 997 + config.dataset_seed) % 100
                < config.markdown_ratio * 100
                else "plain"
            )
            target_turns = 2 + (index % (config.max_turns - 1))
            tasks.append(
                TestingPerfTask(
                    TestingPerfData(
                        idx=index,
                        name=row["task_id"],
                        prompt=_prompt(row, response_format),
                        task_id=row["task_id"],
                        requirements=list(row["requirements"]),
                        follow_ups=[
                            f"{follow_up} Keep each answer near {row['target_words']} words."
                            for follow_up in row["follow_ups"]
                        ],
                        target_turns=target_turns,
                        target_words=row["target_words"],
                        response_format=response_format,
                    ),
                    config.task,
                )
            )
        return tasks


__all__ = ["TestingPerfTaskset"]
