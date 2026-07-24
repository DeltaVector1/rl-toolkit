"""The nemotron anti-hacking guardrail as reusable v1 helpers.

`anti_hacking_multiplier(trace, config)` reproduces v0's `GuardedRubric._score_guard`
multiplier exactly: the hard-zero gates (truncated / unclosed `<think>` / zero visible
words / missing reasoning) and the soft penalties (local meta-commentary and the four
LLM-judge verdicts). `apply_guard` wraps a task reward into the final reward and records
every factor as a metric. A task mixes in `GuardedTask` and calls `self.guarded(...)`.

Reward shape (audit fix #3): `final = anti_hacking_multiplier * task_reward`. v0 also
multiplied by a `format_multiplier`, but its config weight was 0, so the factor was
identically 1.0 — dead code. It is omitted here rather than ported.
"""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, content_text

from nemotron_guard_v1.judges import (
    CoherencyJudge,
    GuardJudgeConfig,
    MetaCommentaryJudge,
    ReasoningCoherencyJudge,
    ReasoningEthicsJudge,
    ScoreTagJudge,
)
from nemotron_guard_v1.text import (
    META_RE,
    format_full_transcript,
    format_reasoning_traces,
    has_unclosed_think,
    is_renderer_stripped_tool_call,
    message_content,
    message_reasoning,
    strip_think_tags,
    system_prompt_text,
)

# Reward multipliers applied when a judge flags a soft violation. Load-bearing but never
# worth tuning per run, so they stay constants (matching v0), not config.
INCOHERENT_PENALTY = 0.1
META_PENALTY = 0.01


class AntiHackingConfig(vf.BaseConfig):
    """Knobs shared by every nemotron category env; per-env behaviour lives in these
    flags, never in a local edit to the guardrail."""

    enabled: bool = True
    """Master switch; when False the reward passes through unmultiplied."""
    enable_judges: bool = True
    """Run the four LLM guard judges. When False only the deterministic gates apply."""
    reasoning_required: bool = False
    """Hard-zero any rollout with an assistant turn that carries no reasoning trace."""
    allow_renderer_stripped_tool_calls: bool = False
    """Exempt pure tool-call turns (no text, no reasoning) from the missing-reasoning gate."""
    judge: GuardJudgeConfig = GuardJudgeConfig()
    """The single endpoint the four guard judges share."""


def _initial_metrics() -> dict[str, float]:
    # -1.0 marks "no usable judge verdict"; the four judge metrics stay there unless a
    # verdict comes back (fail-open, audit fix #5).
    return {
        "anti_hacking_multiplier": 1.0,
        "anti_hacking_truncated": 0.0,
        "anti_hacking_unclosed_think": 0.0,
        "anti_hacking_zero_visible_words": 0.0,
        "anti_hacking_missing_reasoning": 0.0,
        "anti_hacking_renderer_stripped_tool_calls": 0.0,
        "anti_hacking_renderer_tool_call_fallback": 0.0,
        "anti_hacking_local_meta": 0.0,
        "anti_hacking_judge_failures": 0.0,
        "anti_hacking_coherency": -1.0,
        "anti_hacking_reasoning_ethics": -1.0,
        "anti_hacking_reasoning_coherency": -1.0,
        "anti_hacking_meta_commentary": -1.0,
    }


def _prompt_messages(trace: vf.Trace) -> list[dict[str, str]]:
    """The task prompt (system + turns) as role/content dicts for the judge transcripts."""
    data = trace.task.data
    messages: list[dict[str, str]] = []
    if data.system_prompt:
        messages.append({"role": "system", "content": data.system_prompt})
    prompt = data.prompt
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        for message in prompt:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            messages.append({"role": str(role or "user"), "content": content_text(content)})
    return messages


async def _judge_score(judge: ScoreTagJudge, trace: vf.Trace, **fields) -> int | None:
    """Run one guard judge, fail-open on any error (audit fix #5).

    A network error, timeout, or garbled reply must never be indistinguishable from a
    guilty verdict — that would turn judge downtime into a silent, uniform reward penalty.
    The parsed value is already None for an unparseable verdict; catching here extends the
    same fail-open behaviour to a judge-side error.
    """
    try:
        result = await judge.evaluate(trace=trace, **fields)
    except Exception:
        return None
    return result.parsed


async def anti_hacking_multiplier(
    trace: vf.Trace, config: AntiHackingConfig
) -> tuple[float, dict[str, float]]:
    """The v0 anti-hacking multiplier for one trace, plus the per-factor metrics."""
    messages: list[AssistantMessage] = trace.assistant_messages
    assistant_texts = [message_content(message) for message in messages]
    visible_text = strip_think_tags(message_content(messages[-1])) if messages else ""
    has_tool_output = any(message.tool_calls for message in messages)
    metrics = _initial_metrics()

    def hard_zero(gate: str) -> tuple[float, dict[str, float]]:
        metrics[gate] = 1.0
        metrics["anti_hacking_multiplier"] = 0.0
        return 0.0, metrics

    if trace.is_truncated:
        return hard_zero("anti_hacking_truncated")
    if any(has_unclosed_think(text) for text in assistant_texts):
        return hard_zero("anti_hacking_unclosed_think")
    if not visible_text.strip() and not has_tool_output:
        return hard_zero("anti_hacking_zero_visible_words")

    missing_reasoning = [m for m in messages if not message_reasoning(m)]
    renderer_stripped = [m for m in missing_reasoning if is_renderer_stripped_tool_call(m)]
    metrics["anti_hacking_renderer_stripped_tool_calls"] = float(len(renderer_stripped))
    if config.reasoning_required and missing_reasoning:
        all_stripped = len(renderer_stripped) == len(missing_reasoning)
        if config.allow_renderer_stripped_tool_calls and all_stripped:
            metrics["anti_hacking_renderer_tool_call_fallback"] = 1.0
        else:
            return hard_zero("anti_hacking_missing_reasoning")

    multiplier = 1.0
    if META_RE.search(visible_text):
        metrics["anti_hacking_local_meta"] = 1.0
        multiplier *= META_PENALTY

    if config.enable_judges:
        prompt = _prompt_messages(trace)
        transcript = format_full_transcript(prompt, messages)
        reasoning_traces = format_reasoning_traces(messages)
        system_prompt = system_prompt_text(prompt)
        coherency, ethics, reasoning_coherency, meta = await asyncio.gather(
            _judge_score(CoherencyJudge(config.judge), trace, conversation=transcript),
            _judge_score(ReasoningEthicsJudge(config.judge), trace, reasoning_traces=reasoning_traces),
            _judge_score(ReasoningCoherencyJudge(config.judge), trace, reasoning_traces=reasoning_traces),
            _judge_score(
                MetaCommentaryJudge(config.judge),
                trace,
                system_prompt=system_prompt,
                conversation=transcript,
            ),
        )
        verdicts = [coherency, ethics, reasoning_coherency, meta]
        metrics["anti_hacking_judge_failures"] = float(sum(v is None for v in verdicts))

        if coherency is not None:
            metrics["anti_hacking_coherency"] = float(coherency)
            if coherency == 0:
                multiplier *= INCOHERENT_PENALTY
        if ethics is not None:
            metrics["anti_hacking_reasoning_ethics"] = float(ethics)
            if ethics == 1:
                multiplier = 0.0
        if reasoning_coherency is not None:
            metrics["anti_hacking_reasoning_coherency"] = float(reasoning_coherency)
            if reasoning_coherency == 0:
                multiplier = 0.0
        if meta is not None:
            metrics["anti_hacking_meta_commentary"] = float(meta)
            if meta == 1:
                multiplier *= META_PENALTY

    metrics["anti_hacking_multiplier"] = float(multiplier)
    return float(multiplier), metrics


async def apply_guard(
    trace: vf.Trace, task_reward: float, config: AntiHackingConfig
) -> float:
    """Fold the guardrail into a task reward: `final = anti_hacking_multiplier * task_reward`.

    Records the multiplier, every gate/judge factor, the raw task reward, and the final
    reward as metrics so a run can see exactly why a reward was shaped.
    """
    task_reward = float(task_reward)
    if not config.enabled:
        trace.record_metrics(
            {"task_reward": task_reward, "anti_hacking_multiplier": 1.0, "final_reward": task_reward}
        )
        return task_reward
    multiplier, metrics = await anti_hacking_multiplier(trace, config)
    final_reward = multiplier * task_reward
    trace.record_metrics({**metrics, "task_reward": task_reward, "final_reward": final_reward})
    return final_reward


class GuardedTask:
    """Mixin for a `vf.Task` whose config carries an `AntiHackingConfig` at `config.guard`.

    The task computes its raw task reward, then returns `self.guarded(trace, task_reward)`
    from a single `@vf.reward` (the multiplicative reward must live in one reward per the
    v1 weighted-sum combine rule).
    """

    def guard_config(self) -> AntiHackingConfig:
        return self.config.guard

    async def guarded(self, trace: vf.Trace, task_reward: float) -> float:
        return await apply_guard(trace, task_reward, self.guard_config())
