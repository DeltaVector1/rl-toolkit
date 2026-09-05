from __future__ import annotations

import difflib
import re
from typing import Any

word_requirements: tuple[tuple[str, int, int, int], ...] = (
    ("Be verbose", 1400, 800, 2000),
    ("Respond tersely", 163, 75, 250),
    ("Respond in a medium-length response", 300, 200, 400),
)


minimum_length_value = 0.1


stated_range_fraction = 0.25
word_pattern = re.compile(r"[A-Za-z0-9_']+")
ignored_words = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "will",
    "would",
}

markdown_patterns = (
    re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE),
    re.compile(r"^\s{0,3}>\s+", re.MULTILINE),
    re.compile(r"^\s*[-+*]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),
    re.compile(r"^\s{0,3}(```|~~~)", re.MULTILINE),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"!?\[[^\]\n]+\]\([^\)\n]+\)"),
    re.compile(r"(\*\*|__)[^\n]+?\1"),
    re.compile(r"(?<![\w*])\*[^*\n]+\*(?!\*)"),
    re.compile(r"(?<!\w)_[^_\n]+_(?!\w)"),
    re.compile(r"~~[^\n]+?~~"),
    re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE),
    re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE),
)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(
            r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
        )
    return cleaned.strip()


def has_visible_assistant_answer(text: str) -> bool:
    return bool(strip_think_tags(text))


def extract_think_trace(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(matches).strip()
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
        for key in (
            "text",
            "content",
            "reasoning",
            "reasoning_content",
            "thinking",
            "summary",
        ):
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
    for key in (
        "reasoning",
        "reasoning_content",
        "thinking",
        "thinking_content",
        "thinking_blocks",
    ):
        text = _value_to_text(_message_get(message, key))
        if text:
            return text
    details = _message_get(message, "reasoning_details")
    text = _value_to_text(details)
    if text and not text.startswith("{"):
        return text
    return ""


def extract_message_reasoning_trace(message: Any) -> str:
    provider_trace = extract_provider_reasoning_trace(message)
    if provider_trace:
        return provider_trace
    return extract_think_trace(message_content(message))


def think_tag_rejection_reason(text: str) -> str | None:
    tags = [
        match.group(0).lower()
        for match in re.finditer(r"</?think>", text or "", flags=re.IGNORECASE)
    ]
    if tags in ([], ["<think>", "</think>"]):
        return None
    if tags == ["<think>"]:
        return "unclosed_think"
    return "invalid_think_tags"


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    for msg in reversed(completion):
        if message_role(msg) == "assistant":
            return message_content(msg)
    return ""


def extract_all_assistant_texts(completion: Any) -> list[str]:
    if isinstance(completion, str):
        return [completion]
    return [message_content(m) for m in completion if message_role(m) == "assistant"]


def extract_all_assistant_messages(completion: Any) -> list[Any]:
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    if not isinstance(completion, list):
        return []
    return [message for message in completion if message_role(message) == "assistant"]


def contains_markdown(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in markdown_patterns)


def completion_contains_markdown(completion: Any) -> bool:
    return any(
        contains_markdown(strip_think_tags(message_content(turn)))
        for turn in extract_all_assistant_messages(completion)
    )


def assistant_turn_response_rejection(
    message: Any, *, reasoning_required: bool
) -> str | None:
    content = message_content(message)
    tag_rejection = think_tag_rejection_reason(content)
    if tag_rejection:
        return tag_rejection
    visible_text = strip_think_tags(content)

    if not visible_text.strip():
        return "zero_words"
    if reasoning_required and not extract_message_reasoning_trace(message):
        return "no_reasoning"
    return None


def count_words(text: str) -> int:
    if not text:
        return 0
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in word_pattern.findall(text or "")]


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in word_tokens(text)
        if len(token) > 2 and token not in ignored_words
    }


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(
        message_content(m) for m in prompt if message_role(m) in {"system", "user"}
    )


def reasoning_trace_quality(prompt: Any, completion: Any) -> tuple[float, int]:
    traces = [
        trace
        for turn in extract_all_assistant_messages(completion)
        if (trace := extract_message_reasoning_trace(turn))
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
        overlap_score = clamp01(
            len(prompt_terms & trace_terms) / min(16, len(prompt_terms))
        )
    else:
        overlap_score = 0.5

    structure_score = 1.0 if re.search(r"[.!?]\s|[\n;:]", text) else 0.4
    quality = (
        0.40 * length_score
        + 0.25 * repetition_score
        + 0.25 * overlap_score
        + 0.10 * structure_score
    )
    return clamp01(quality), word_count


def word_band(requirement: str, target: int = 0) -> tuple[int, int, int]:
    if target > 0:
        half = round(target * stated_range_fraction)
        return target, target - half, target + half
    for name, static_target, minimum, maximum in word_requirements:
        if name == requirement:
            return static_target, minimum, maximum
    return 300, 200, 400


def calculate_word_count_multiplier(
    word_count: int,
    requirement: str,
    tolerance_percent: float,
    target: int = 0,
) -> tuple[float, dict[str, Any]]:
    target, minimum, maximum = word_band(requirement, target)
    tolerance = max(1, int(target * tolerance_percent))

    details = {
        "minimum": minimum,
        "maximum": maximum,
        "target": target,
    }
    if word_count == 0:
        return 0.0, {"multiplier": 0.0, "zero_words": True, **details}
    if minimum <= word_count <= maximum:
        return 1.0, {"multiplier": 1.0, "zero_words": False, **details}

    distance = min(
        abs(word_count - minimum)
        if word_count < minimum
        else abs(word_count - maximum),
        tolerance,
    )
    multiplier = max(minimum_length_value, 1.0 - (distance / tolerance))
    return multiplier, {"multiplier": multiplier, "zero_words": False, **details}


def detect_structured_markers(text: str) -> str | None:
    if not text:
        return None
    for tag in (
        "Response-Style",
        "Response-Length",
        "user-turn-output",
        "conversation",
        "thinking",
        "thought",
    ):
        if re.search(rf"</?{re.escape(tag)}(?:\s[^>]*)?>", text, flags=re.IGNORECASE):
            return tag
    return None


def validate_follow_up_message(
    raw: str,
    prior_follow_ups: tuple[str, ...] = (),
) -> tuple[str, str | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    if not raw:
        return "", "empty_response", details
    match = re.fullmatch(
        r"\s*<user-turn-output>(.*?)</user-turn-output>\s*",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return "", "invalid_wrapper", details
    text = match.group(1).strip()
    if not text:
        return "", "empty_message", details
    if detect_structured_markers(text) is not None:
        return "", "unexpected_tag", details
    if re.search(r"^\s*(?:SYSTEM|USER|ASSISTANT)\s*:", text, flags=re.MULTILINE):
        return "", "role_label", details
    if re.search(r"^\s*(?:[-*•]|\d+[.)])\s+", text, flags=re.MULTILINE):
        return "", "list_format", details
    sentence_text = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc)\.",
        lambda match: match.group(0)[:-1],
        text,
        flags=re.IGNORECASE,
    )
    sentence_text = re.sub(
        r"\b(?:e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", ""),
        sentence_text,
        flags=re.IGNORECASE,
    )
    sentence_count = max(1, len(re.findall(r"[.!?]+(?:\s|$)", sentence_text)))
    details["sentence_count"] = sentence_count
    if sentence_count > 3:
        return "", "too_many_sentences", details
    word_count = len(re.findall(r"\b\w+(?:['’-]\w+)?\b", text))
    details["word_count"] = word_count
    if word_count < 6:
        return "", "too_few_words", details
    if word_count > 80:
        return "", "too_many_words", details
    if re.search(
        r"\b(synthetic (?:data|conversation)|red[- ]team data|training data|"
        r"system prompt|evaluation rubric|as an ai)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "", "generator_commentary", details

    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    maximum_similarity = 0.0
    for previous in prior_follow_ups:
        previous_normalized = re.sub(r"\s+", " ", previous.lower()).strip()
        maximum_similarity = max(
            maximum_similarity,
            difflib.SequenceMatcher(None, normalized, previous_normalized).ratio(),
        )
    details["maximum_prior_similarity"] = maximum_similarity
    if maximum_similarity >= 0.85:
        return "", "repeats_prior_turn", details
    return text, None, details
