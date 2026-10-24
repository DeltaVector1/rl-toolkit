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
from typing import NamedTuple

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
    # verdict comes back. Two different failure modes are tracked separately:
    #   anti_hacking_judge_failures     — gemma replied but the verdict was unparseable
    #                                      (normal formatting noise; fail-open, no penalty).
    #   anti_hacking_judge_endpoint_down — the judge CALL errored (gemma unreachable). This
    #                                      tanks the reward to 0 so an outage shows up as
    #                                      cratering scores, not a silent pass (see #6).
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
        "anti_hacking_judge_endpoint_down": 0.0,
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


#: Stop conditions that mean the model ran out of generation budget. `trace.is_truncated`
#: also covers infrastructure limits (`harness_timeout`, `context_length`) — a wall-clock
#: timeout or a prompt too long to fit says nothing about the model's behavior, so gating on
#: it scores our own operational failures as model misbehavior. v0 gated on generation
#: length only; this keeps that scope.
_GENERATION_LIMITS = frozenset({"max_output_tokens", "max_total_tokens", "max_turns"})


def _generation_truncated(trace: vf.Trace) -> bool:
    if trace.stop_condition in _GENERATION_LIMITS:
        return True
    last = next((call for call in reversed(trace.calls) if call.error is None), None)
    return bool(last and last.finish_reason == "length")


class _JudgeOutcome(NamedTuple):
    """One guard-judge result. Two distinct failure modes (see #6):

    - ``score`` is the parsed 0/1 verdict, or ``None`` when gemma replied but the reply was
      unparseable — normal formatting noise, treated fail-open (no penalty).
    - ``endpoint_down`` is True when the judge CALL itself errored (connection refused /
      timeout / HTTP error — gemma is unreachable). That is NOT hidden: it tanks the reward
      so an outage surfaces as cratering scores the operator can act on, not a silent pass.
    """

    score: int | None
    endpoint_down: bool


async def _judge_score(judge: ScoreTagJudge, trace: vf.Trace, **fields) -> _JudgeOutcome:
    try:
        result = await asyncio.wait_for(
            judge.evaluate(trace=trace, **fields), timeout=judge.config.timeout_seconds
        )
    except Exception:
        # The judge call failed or timed out. An outage says nothing about the rollout, so it
        # cannot be scored as a guilty verdict — surface it as a metric and apply no penalty.
        return _JudgeOutcome(score=None, endpoint_down=True)
    return _JudgeOutcome(score=result.parsed, endpoint_down=False)


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

    if _generation_truncated(trace):
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
        reasoning_traces = format_reasoning_traces(messages, prompt)
        system_prompt = system_prompt_text(prompt)
        outcomes = await asyncio.gather(
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
        # A judge-endpoint error means the judge was unreachable, which is evidence about the
        # infrastructure and none at all about the rollout. Scoring it 0 would teach the model
        # that a good answer was bad, and outages arrive in correlated bursts that wipe whole
        # groups at once. Fail OPEN and let the metric carry the alarm.
        if any(o.endpoint_down for o in outcomes):
            metrics["anti_hacking_judge_endpoint_down"] = 1.0
        coherency, ethics, reasoning_coherency, meta = (o.score for o in outcomes)
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
                # A soft penalty, not a hard zero: this judge reads the private chain of
                # thought and reliably flags self-correction ("wait, no — 11:45 is before
                # 12:00") as incoherence. That is the behavior RL exists to grow, and a full
                # zero killed 236 correct answers. Matches how visible-text incoherence is
                # already treated.
                multiplier *= INCOHERENT_PENALTY
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
