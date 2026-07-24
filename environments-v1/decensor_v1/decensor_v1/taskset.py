"""decensor_v1 — the v0 decensor env ported to the task-centric v1 API.

A multi-turn rollout under the `null` harness: an adversarial red-team `vf.User` drives
follow-up turns while the model produces every assistant turn, then ONE `@vf.reward`
returns the scalar product of the refusal-derived base reward and the coherency / markdown /
reasoning-ethics / meta / length multipliers (v1 sums rewards, so the multiplication happens
inside the single method). Every factor is logged with `trace.record_metrics`.

Audit fixes baked in (see README): length multiplier floored (#2), `reasoning_coherency`
re-added as a real working gate fed the true `reasoning_content` trace (undoes the old #4
misdiagnosis), and judge failures split (#6): an endpoint outage tanks the reward + flags a
metric, while a received-but-unparseable reply stays fail-open.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, NamedTuple

import verifiers.v1 as vf

from decensor_v1.data import build_rows
from decensor_v1.judges import (
    CoherencyJudge,
    DecensorJudgeConfig,
    MarkdownCorrectnessJudge,
    MarkdownPresenceJudge,
    MetaCommentaryJudge,
    ReasoningCoherencyJudge,
    ReasoningEthicsJudge,
    RefusalJudge,
)
from decensor_v1.scoring import ScoringConfig, Verdicts, score_decensor
from decensor_v1.state import DecensorState
from decensor_v1.text import (
    extract_completion_text,
    extract_last_user_across,
    extract_message_reasoning_trace,
    extract_think_trace,
    strip_think_tags,
    _format_full_conv_for_coherency,
)
from decensor_v1.user import DecensorUser, DecensorUserConfig


class DecensorData(vf.TaskData):
    word_requirement: str = ""
    word_target: int = 0
    prompt_bucket: str = "styles"
    target_turns: int = 1


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


class _JudgeOutcome(NamedTuple):
    """One judge result (FIX #6). `score` is the parsed verdict, or `None` when gemma replied
    but the reply was unparseable (fail-open). `endpoint_down` is True when the judge CALL
    itself errored (gemma unreachable) — that outage tanks the reward, it is not fail-open."""

    score: int | None
    endpoint_down: bool


class DecensorTask(vf.Task[DecensorData, DecensorState, DecensorTaskConfig]):
    user = DecensorUser

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.target_turns = self.data.target_turns

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        if trace.state.user_done:
            return True
        return trace.num_turns >= max(1, self.data.target_turns)

    # --- Trace → message lists ---

    @staticmethod
    def _message_dict(message: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role, "content": message.content or ""}
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = getattr(message, key, None)
            if value:
                out[key] = value
        return out

    def _split_conversation(self, trace: vf.Trace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Reconstruct (prompt, completion) as v0-shaped message lists.

        prompt = [system, opening user]; completion = every turn after the opening user.
        """
        branches = trace.branches
        messages = branches[-1].messages if branches else []
        convo = [self._message_dict(m) for m in messages if m.role != "system"]
        prompt: list[dict[str, Any]] = [{"role": "system", "content": self.data.system_prompt or ""}]
        completion: list[dict[str, Any]] = []
        seen_first_user = False
        for m in convo:
            if not seen_first_user and m["role"] == "user":
                prompt.append(m)
                seen_first_user = True
            else:
                completion.append(m)
        return prompt, completion

    async def _verdict(self, judge: Any, trace: vf.Trace, **fields: Any) -> "_JudgeOutcome":
        """Run one judge, distinguishing the two failure modes (FIX #6).

        A raised `evaluate` call means gemma is unreachable (endpoint down) — surfaced so it
        tanks the reward. A received-but-unparseable reply (verdict None) stays fail-open.
        """
        try:
            result = await judge.evaluate(trace=trace, **fields)
        except Exception:
            return _JudgeOutcome(score=None, endpoint_down=True)
        return _JudgeOutcome(score=result.parsed, endpoint_down=False)

    @vf.reward(weight=1.0)
    async def decensor_reward(self, trace: vf.Trace) -> float:
        cfg = self.config
        scfg = _scoring_config(cfg)
        prompt, completion = self._split_conversation(trace)
        bucket = self.data.prompt_bucket

        response_text_raw = extract_completion_text(completion)
        response_text = strip_think_tags(response_text_raw)
        # The LAST assistant turn, to match `response_text`. Taking the first paired turn 1's
        # reasoning with turn N's answer on every multi-turn rollout — and half of them are
        # multi-turn — so the reasoning judges were grading a mismatched pair.
        reasoning_trace = extract_think_trace(response_text_raw) or ""
        for turn in reversed(completion):
            if turn.get("role") == "assistant":
                reasoning_trace = extract_message_reasoning_trace(turn) or reasoning_trace
                break
        question = extract_last_user_across(prompt, completion)
        system_prompt_text = self.data.system_prompt or ""

        judge_cfg = cfg.judge
        verdict_kwargs = {"refusal": None, "coherency": None, "reasoning_ethics": None,
                          "reasoning_coherency": None, "meta_commentary": None,
                          "md_presence": None, "md_correctness": None}

        jobs: list[tuple[str, Any, dict[str, Any]]] = [
            ("refusal", RefusalJudge(judge_cfg), {"question": question, "response": response_text}),
            ("coherency", CoherencyJudge(judge_cfg),
             {"prompt": _format_full_conv_for_coherency(prompt, completion), "response": response_text}),
        ]
        if cfg.enable_reasoning_trace:
            jobs.append(("reasoning_ethics", ReasoningEthicsJudge(judge_cfg),
                         {"reasoning_trace": reasoning_trace, "response": response_text}))
        if cfg.enable_reasoning_coherency:
            # Fed the SAME real reasoning_content trace the ethics judge gets (not v0's empty
            # content-only trace) — so this is a genuine coherency gate, not a rubber stamp.
            jobs.append(("reasoning_coherency", ReasoningCoherencyJudge(judge_cfg),
                         {"reasoning_trace": reasoning_trace, "question": question}))
        if cfg.enable_meta_commentary:
            jobs.append(("meta_commentary", MetaCommentaryJudge(judge_cfg),
                         {"system_prompt": system_prompt_text, "response": response_text}))
        if cfg.enable_markdown_judges:
            jobs.append(("md_presence", MarkdownPresenceJudge(judge_cfg), {"response": response_text}))
            if bucket in ("styles", "markdown"):
                jobs.append(("md_correctness", MarkdownCorrectnessJudge(judge_cfg), {"response": response_text}))

        outcomes: list[_JudgeOutcome] = await asyncio.gather(
            *(self._verdict(judge, trace, **fields) for _name, judge, fields in jobs)
        )
        for (name, _judge, _fields), outcome in zip(jobs, outcomes):
            verdict_kwargs[name] = outcome.score

        # FIX #6: an endpoint outage (gemma unreachable) tanks the reward; a received-but-
        # unparseable reply is a parse failure that stays fail-open (verdict None).
        endpoint_down = any(o.endpoint_down for o in outcomes)
        parse_failures = sum(o.score is None and not o.endpoint_down for o in outcomes)

        # Seeded by the row, not drawn fresh per rollout: an unseeded draw gave two identical
        # answers in the same GRPO group a 10x reward difference, which is pure noise in the
        # advantage. Per-row it shifts the whole group's baseline instead, which GRPO cancels.
        md_exempt = (bucket == "styles") and (
            random.Random(self.data.idx).random() < cfg.markdown_exempt_probability
        )

        reward, breakdown = score_decensor(
            prompt,
            completion,
            Verdicts(**verdict_kwargs),
            bucket=bucket,
            word_requirement=self.data.word_requirement,
            word_target=self.data.word_target,
            is_truncated=trace.is_truncated,
            md_exempt=md_exempt,
            judge_endpoint_down=endpoint_down,
            config=scfg,
        )

        trace.info["decensor"] = breakdown
        trace.record_metrics(
            {
                key: float(value)
                for key, value in breakdown.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        trace.record_metric("decensor_judge_parse_failures", float(parse_failures))
        trace.record_metric("decensor_gated", 0.0 if breakdown.get("early_termination") is None else 1.0)
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
                    ),
                    c.task,
                )
            )
        return tasks


__all__ = ["DecensorTaskset"]
