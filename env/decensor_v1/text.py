"""Pure text helpers for decensor scoring.

These are copied verbatim from the v0 env (`rl_environments/decensor-env/decensor_env.py`)
so the ported reward reproduces v0's parsing/word-counting/gate detection exactly,
including `calculate_word_count_multiplier` (verbose targets 2000 words, so the model is
pushed to write long — a short verbose answer zeros the length term, matching v0).
"""

from __future__ import annotations

import random
import re
from typing import Any

WORD_REQUIREMENTS: tuple[tuple[str, int, int, int], ...] = (
    ("Be verbose", 2000, 1900, 2100),
    ("Respond tersely", 163, 75, 250),
    ("Respond in a medium-length response", 300, 200, 400),
)

# Below the floor a wrong-length answer zeros the whole multiplicative reward, so the
# rollout carries no gradient no matter how good the answer was.
WORD_MULTIPLIER_FLOOR = 0.1

# "Be verbose" is the only requirement whose target the model cannot infer from the
# wording, so the prompt states a sampled target instead of hiding a fixed 2000.
VERBOSE_REQUIREMENT = "Be verbose"
VERBOSE_TARGET_RANGE = (800, 2000)
STATED_BAND_FRACTION = 0.25
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "i", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "with", "will", "would",
}


def strip_think_tags(text: str) -> str:
    """Remove reasoning content across all three think-tag variants."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_think_trace(text: str) -> str:
    """Return the reasoning content across all three think-tag variants."""
    if not text:
        return ""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(matches).strip()
    m_auto = re.search(r"^(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if m_auto:
        return m_auto.group(1).strip()
    match = re.search(r"<think>(.*?)$", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_value_to_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content", "thinking", "summary"):
            text = _value_to_text(value.get(key))
            if text:
                return text
        return ""
    return str(value).strip()


def _message_get(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        if key in message:
            return message.get(key)
        for container_key in ("additional_kwargs", "metadata", "extra"):
            container = message.get(container_key)
            if isinstance(container, dict) and key in container:
                return container.get(key)
        return None
    value = getattr(message, key, None)
    if value is not None:
        return value
    for container_key in ("additional_kwargs", "metadata", "extra"):
        container = getattr(message, container_key, None)
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return None


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def extract_provider_reasoning_trace(message: Any) -> str:
    """Return reasoning carried outside assistant content by some providers."""
    for key in ("reasoning", "reasoning_content", "thinking", "thinking_content", "thinking_blocks"):
        text = _value_to_text(_message_get(message, key))
        if text:
            return text
    details = _message_get(message, "reasoning_details")
    text = _value_to_text(details)
    if text and not text.startswith("{"):
        return text
    return ""


def extract_message_reasoning_trace(message: Any) -> str:
    content_trace = extract_think_trace(message_content(message))
    if content_trace:
        return content_trace
    return extract_provider_reasoning_trace(message)


def has_unclosed_think(text: str) -> bool:
    """True if <think> opens but never closes — hard malformed-trace gate."""
    if not text:
        return False
    opens = len(re.findall(r"<think>", text, flags=re.IGNORECASE))
    closes = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
    return opens > closes


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    for msg in reversed(completion):
        if message_role(msg) == "assistant":
            return message_content(msg)
    return ""


def extract_last_user_message(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    for msg in reversed(prompt):
        if message_role(msg) == "user":
            return message_content(msg)
    return ""


def extract_last_user_across(prompt: Any, completion: Any) -> str:
    """Last user message searching completion first, then prompt."""
    if isinstance(completion, list):
        for msg in reversed(completion):
            if message_role(msg) == "user":
                return message_content(msg)
    return extract_last_user_message(prompt)


def _format_full_conv_for_coherency(prompt: Any, completion: Any) -> str:
    """Prompt + completion minus the final assistant message, for the coherency judge."""
    msgs: list[Any] = []
    if isinstance(prompt, list):
        msgs.extend(prompt)
    if isinstance(completion, list):
        comp = list(completion)
        while comp and message_role(comp[-1]) != "assistant":
            comp.pop()
        if comp:
            comp.pop()
        msgs.extend(comp)
    return "\n\n".join(f"{message_role(m).upper()}: {message_content(m)}" for m in msgs)


def extract_all_assistant_texts(completion: Any) -> list[str]:
    if isinstance(completion, str):
        return [completion]
    out: list[str] = []
    for m in completion:
        if message_role(m) == "assistant":
            out.append(message_content(m))
    return out


def extract_all_assistant_messages(completion: Any) -> list[Any]:
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    if not isinstance(completion, list):
        return []
    return [m for m in completion if message_role(m) == "assistant"]


def assistant_turn_early_termination(message: Any, *, reasoning_required: bool) -> str | None:
    content = message_content(message)
    if has_unclosed_think(content):
        return "unclosed_think"
    visible_text = strip_think_tags(content)
    # Fenced code and fenced Markdown are visible answers. count_words()
    # deliberately excludes fences from prose-length shaping, but it must not
    # make a complete fenced answer look empty.
    if not visible_text.strip():
        return "zero_words"
    if reasoning_required and not extract_message_reasoning_trace(message):
        return "no_reasoning"
    return None


def extract_system_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return ""
    for msg in prompt:
        if message_role(msg) == "system":
            return message_content(msg)
    return ""


def count_words(text: str) -> int:
    if not text:
        return 0
    cleaned_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return len([t for t in re.split(r"\s+", cleaned_text.strip()) if t])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in word_tokens(text) if len(token) > 2 and token not in STOPWORDS}


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(
        message_content(m) for m in prompt if message_role(m) in {"system", "user"}
    )


def reasoning_trace_quality(prompt: Any, completion: Any) -> tuple[float, int]:
    traces = [
        trace for turn in extract_all_assistant_messages(completion) if (trace := extract_message_reasoning_trace(turn))
    ]
    text = "\n".join(trace.strip() for trace in traces if trace.strip())
    tokens = word_tokens(text)
    word_count = len(tokens)
    if word_count == 0:
        return 0.0, 0

    length_score = clamp01(word_count / 96.0)
    if word_count > 640:
        length_score *= max(0.25, 1.0 - ((word_count - 640) / 1200.0))

    unique_ratio = len(set(tokens)) / max(1, word_count)
    repetition_score = clamp01(unique_ratio / 0.38)

    prompt_terms = meaningful_tokens(prompt_text(prompt))
    trace_terms = meaningful_tokens(text)
    if prompt_terms:
        overlap_score = clamp01(len(prompt_terms & trace_terms) / min(16, len(prompt_terms)))
    else:
        overlap_score = 0.5

    structure_score = 1.0 if re.search(r"[.!?]\s|[\n;:]", text) else 0.4
    quality = 0.40 * length_score + 0.25 * repetition_score + 0.25 * overlap_score + 0.10 * structure_score
    return clamp01(quality), word_count


def sample_word_target(requirement: str, seed: int) -> int:
    """Per-prompt word target, or 0 when the requirement states no number.

    Seeded by the row so every rollout in a GRPO group shares one target — a target that
    varied within a group would land in the advantage as pure noise.
    """
    if requirement != VERBOSE_REQUIREMENT:
        return 0
    low, high = VERBOSE_TARGET_RANGE
    return random.Random(seed).randrange(low, high + 1, 50)


def length_instruction(requirement: str, target: int) -> str:
    """The `<Response-Length>` body shown to the model."""
    if target > 0:
        return f"{requirement} — aim for about {target} words."
    return requirement


def word_band(requirement: str, target: int = 0) -> tuple[int, int, int]:
    """(target, minimum, maximum) words for full length credit.

    A stated `target` sets the band; otherwise the requirement's static band applies.
    """
    if target > 0:
        half = round(target * STATED_BAND_FRACTION)
        return target, target - half, target + half
    for name, static_target, minimum, maximum in WORD_REQUIREMENTS:
        if name == requirement:
            return static_target, minimum, maximum
    return 300, 200, 400


def calculate_word_count_multiplier(
    word_count: int,
    requirement: str,
    tolerance_percent: float,
    target: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Length multiplier: full credit inside the band, linear falloff to
    `WORD_MULTIPLIER_FLOOR` outside it."""
    target, minimum, maximum = word_band(requirement, target)
    tolerance = max(1, int(target * tolerance_percent))

    if word_count == 0:
        return 0.0, {"multiplier": 0.0, "zero_words": True}
    if minimum <= word_count <= maximum:
        return 1.0, {"multiplier": 1.0, "zero_words": False}

    distance = min(abs(word_count - minimum) if word_count < minimum else abs(word_count - maximum), tolerance)
    multiplier = max(WORD_MULTIPLIER_FLOOR, 1.0 - (distance / tolerance))
    return multiplier, {"multiplier": multiplier, "zero_words": False}


def detect_structured_markers(text: str) -> bool:
    """Detect machine-readable scaffolding the response was asked not to emit.

    Fenced code is removed first, the same way `count_words` does: Python inside a fence
    trips the YAML rule on two `try:` / `except:` lines, which was most of what this gate
    actually fired on.
    """
    if not text:
        return False
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    xml = bool(re.search(r"<[A-Za-z_][A-Za-z0-9_\-]*>.*?</[A-Za-z_][A-Za-z0-9_\-]*>", text, re.DOTALL))
    bracket = bool(re.search(r"\[[A-Z_]{2,}\]", text))
    yaml = len(re.findall(r"^\s*[A-Za-z_][A-Za-z0-9_\-]*:\s*$", text, flags=re.MULTILINE)) >= 2
    return xml or bracket or yaml


def parse_tag_int(text: str, tag: str, min_value: int, max_value: int) -> int | None:
    for pattern in [rf"<{tag}>\s*([0-9]+)\s*</{tag}>", rf"<{tag}>\s*([0-9]+)"]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            value = int(m.group(1))
            if min_value <= value <= max_value:
                return value
    return None
