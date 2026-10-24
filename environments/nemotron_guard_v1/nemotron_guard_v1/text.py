"""Pure text helpers for the nemotron anti-hacking guardrail.

Copied verbatim from the v0 shared module
(`rl_environments/nemotron-knowledge/nemotron_knowledge_guardrails.py`) so the ported
guardrail reproduces v0's think-tag stripping, reasoning extraction, and gate detection
exactly. These functions are pure and were not implicated in any audit finding.

The only adaptation from v0: `message_reasoning` / `assistant_visible_text` and the
transcript builders take v1 `AssistantMessage` objects (content + reasoning_content +
tool_calls) instead of raw dict/list completions. v1's renderer already splits reasoning
into `reasoning_content`, so where v0 read a provider `reasoning` field we read
`reasoning_content`; inline `<think>` blocks in `content` are still honoured first, exactly
as v0 did.
"""

from __future__ import annotations

import re

from verifiers.v1.types import AssistantMessage

THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
TAG_SCORE_RE = re.compile(r"<score>\s*([01])\s*</score>|<score>\s*([01])", re.IGNORECASE)
# Meta-commentary: the model talking about its own grading rather than doing the task.
# The bare words "judge", "evaluator" and "rubric" used to be triggers on their own, which
# made this fire on any answer that merely discussed a courtroom or an assessment — a
# correct criminal-procedure answer lost 99% of its reward for the phrase "the judge
# improperly weighed the credibility of the evidence". Every sampled hit was a false
# positive, so the bare nouns now need grading context to count.
META_RE = re.compile(
    r"\b(system prompt|hidden instructions?|developer instructions?|my instructions|"
    r"as instructed (by|in) (the )?(system|developer|prompt)|per (the )?(system|developer)|"
    r"(the |this )?(grading |scoring |evaluation )?(rubric|judge|evaluator|grader)"
    r"\s+(wants?|expects?|will|would|is looking|prefers?|rewards?|scores?|grades?)|"
    r"(satisfy|impress|fool|game|please)\s+(the\s+)?(rubric|judge|evaluator|grader)|"
    r"scoring criteria|reward function|to get (a )?(high|good) score)\b",
    re.IGNORECASE,
)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # A `</think>` with no opener is a stray tag, not a reasoning block: the renderer's
    # parser already consumed the real one and left this behind. Dropping everything before
    # it deleted a mean 56-66% of the answer on 225 rollouts — and since every taskset uses
    # this to build the text it grades, it corrupted the score too. Remove the tag only.
    if not THINK_OPEN_RE.search(cleaned):
        cleaned = THINK_CLOSE_RE.sub("", cleaned)
    elif re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_think_trace(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(matches).strip()
    auto = re.search(r"^(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if auto:
        return auto.group(1).strip()
    unclosed = re.search(r"<think>(.*?)$", text, flags=re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()
    return ""


def has_unclosed_think(text: str) -> bool:
    if not text:
        return False
    return len(THINK_OPEN_RE.findall(text)) > len(THINK_CLOSE_RE.findall(text))


def parse_score_tag(text: str | None) -> int | None:
    """The 0/1 verdict in a `<score>` tag, or None when it is missing or self-contradictory."""
    if not text:
        return None
    values: list[str] = []
    for match in TAG_SCORE_RE.finditer(text):
        value = match.group(1) or match.group(2)
        if value in {"0", "1"}:
            values.append(value)
    if not values or len(set(values)) > 1:
        return None
    return int(values[-1])


# --- v1 message adapters ----------------------------------------------------
# v0 read reasoning from an inline `<think>` block first, then from a provider-carried
# `reasoning`/`reasoning_content` field. v1 carries the latter as `reasoning_content`.


def message_content(message: AssistantMessage) -> str:
    return message.content or ""


def message_reasoning(message: AssistantMessage) -> str:
    inline = extract_think_trace(message_content(message))
    if inline:
        return inline
    return (message.reasoning_content or "").strip()


def is_renderer_stripped_tool_call(message: AssistantMessage) -> bool:
    """A pure tool call the renderer emitted with no visible text and no reasoning."""
    if not message.tool_calls:
        return False
    if message_content(message).strip():
        return False
    return not (message.reasoning_content or "").strip()


def _tool_calls_line(message: AssistantMessage) -> str:
    calls = message.tool_calls or []
    if not calls:
        return ""
    rendered = ", ".join(f"{call.name}({call.arguments})" for call in calls)
    return f"ACTION RECORD: {rendered}"


def format_full_transcript(
    prompt: list, assistant_messages: list[AssistantMessage]
) -> str:
    """Role-labelled, think-stripped transcript handed to the coherency/meta judges."""
    parts: list[str] = []
    # Every prompt role is rendered, not just system/user. Dropping the prompt's assistant
    # and tool turns handed the judge a run of user messages with the replies between them
    # deleted, which reads as incoherent no matter what the model did. Tool-calling rows are
    # mid-conversation continuations, so this was near-total there.
    for message in prompt:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "") or "")
        if role == "system":
            parts.append(f"SYSTEM:\n{content}")
        elif role == "tool":
            parts.append(f"TOOL RESULT:\n{content}")
        elif role == "assistant":
            parts.append(f"ASSISTANT (earlier):\n{strip_think_tags(content)}")
        else:
            parts.append(f"USER:\n{content}")
    for index, message in enumerate(assistant_messages, start=1):
        visible = strip_think_tags(message_content(message)) or _tool_calls_line(message)
        parts.append(f"ASSISTANT TURN {index}:\n{visible}")
    return "\n\n".join(parts)


def format_reasoning_traces(
    assistant_messages: list[AssistantMessage], prompt: list | None = None
) -> str:
    """Reasoning-only transcript handed to the reasoning ethics/coherency judges.

    The user's question leads, because these judges are asked whether the trace stays
    on-topic — a question with no answer when the topic was never supplied.
    """
    parts: list[str] = []
    for message in prompt or []:
        if str(message.get("role", "")).lower() == "user":
            parts.append(f"USER TURN:\n{str(message.get('content', '') or '')}")
    for index, message in enumerate(assistant_messages, start=1):
        trace = message_reasoning(message)
        if trace:
            parts.append(f"ASSISTANT TURN {index} (reasoning):\n{trace}")
    return "\n\n".join(parts) if parts else "(no reasoning traces present)"


def system_prompt_text(prompt: list) -> str:
    for message in prompt:
        if str(message.get("role", "")).lower() == "system":
            return str(message.get("content", "") or "")
    return ""
