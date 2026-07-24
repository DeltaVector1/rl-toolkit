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
META_RE = re.compile(
    r"\b(system prompt|hidden instructions?|developer instructions?|my instructions|"
    r"as instructed|per (the )?(system|developer)|rubric|judge|evaluator|"
    r"scoring criteria|reward function|to get (a )?(high|good) score)\b",
    re.IGNORECASE,
)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
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
    for message in prompt:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "") or "")
        if role == "system":
            parts.append(f"SYSTEM:\n{content}")
        elif role == "user":
            parts.append(f"USER:\n{content}")
    for index, message in enumerate(assistant_messages, start=1):
        visible = strip_think_tags(message_content(message)) or _tool_calls_line(message)
        parts.append(f"ASSISTANT TURN {index}:\n{visible}")
    return "\n\n".join(parts)


def format_reasoning_traces(assistant_messages: list[AssistantMessage]) -> str:
    """Reasoning-only transcript handed to the reasoning ethics/coherency judges."""
    parts: list[str] = []
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
