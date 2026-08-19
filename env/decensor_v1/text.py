from __future__ import annotations

import difflib
import re
from typing import Any

WORD_REQUIREMENTS: tuple[tuple[str, int, int, int], ...] = (
    ("Be verbose", 1400, 800, 2000),
    ("Respond tersely", 163, 75, 250),
    ("Respond in a medium-length response", 300, 200, 400),
)


WORD_MULTIPLIER_FLOOR = 0.1


STATED_BAND_FRACTION = 0.25
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
STOPWORDS = {
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

JUDGE_TEXT_MAX_CHARS = 12_000
JUDGE_CONTEXT_MAX_CHARS = 6_000
REASONING_REVIEW_CHUNK_CHARS = 8_000
REASONING_REVIEW_OVERLAP_CHARS = 400


MARKDOWN_PATTERNS = (
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
    content_trace = extract_think_trace(message_content(message))
    if content_trace:
        return content_trace
    return extract_provider_reasoning_trace(message)


def has_unclosed_think(text: str) -> bool:
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
    if isinstance(completion, list):
        for msg in reversed(completion):
            if message_role(msg) == "user":
                return message_content(msg)
    return extract_last_user_message(prompt)


def _format_full_conv_for_coherency(prompt: Any, completion: Any) -> str:
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


def contains_markdown(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in MARKDOWN_PATTERNS)


def completion_contains_markdown(completion: Any) -> bool:
    return any(
        contains_markdown(strip_think_tags(message_content(turn)))
        for turn in extract_all_assistant_messages(completion)
    )


def shorten_text_for_judge(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars < 120:
        raise ValueError("max_chars must be at least 120")
    marker = "\n\n[...middle omitted from this review input...]\n\n"
    available = max_chars - len(marker)
    start_chars = available // 2
    end_chars = available - start_chars
    return text[:start_chars].rstrip() + marker + text[-end_chars:].lstrip()


def split_text_for_judge(
    text: str,
    *,
    max_chars: int = REASONING_REVIEW_CHUNK_CHARS,
    overlap_chars: int = REASONING_REVIEW_OVERLAP_CHARS,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if max_chars < 120:
        raise ValueError("max_chars must be at least 120")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            search_start = start + max_chars // 2
            boundaries = [
                text.rfind("\n\n", search_start, hard_end),
                text.rfind(". ", search_start, hard_end),
                text.rfind("\n", search_start, hard_end),
                text.rfind(" ", search_start, hard_end),
            ]
            boundary = max(boundaries)
            if boundary > start:
                separator_chars = (
                    2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1
                )
                end = boundary + separator_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def build_reasoning_review_units(
    prompt: Any,
    completion: Any,
) -> list[dict[str, Any]]:
    current_question = extract_last_user_message(prompt)
    units: list[dict[str, Any]] = []
    assistant_turn = 0
    messages = completion if isinstance(completion, list) else [
        {"role": "assistant", "content": str(completion or "")}
    ]
    for message in messages:
        role = message_role(message)
        if role == "user":
            current_question = message_content(message)
            continue
        if role != "assistant":
            continue
        assistant_turn += 1
        reasoning = extract_message_reasoning_trace(message)
        chunks = split_text_for_judge(reasoning)
        visible_response = strip_think_tags(message_content(message))
        for chunk_index, chunk in enumerate(chunks, start=1):
            units.append(
                {
                    "assistant_turn": assistant_turn,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "question": shorten_text_for_judge(
                        current_question,
                        JUDGE_CONTEXT_MAX_CHARS,
                    ),
                    "reasoning_trace": chunk,
                    "response": shorten_text_for_judge(
                        visible_response,
                        JUDGE_CONTEXT_MAX_CHARS,
                    ),
                }
            )
    return units


def assistant_turn_early_termination(
    message: Any, *, reasoning_required: bool
) -> str | None:
    content = message_content(message)
    if has_unclosed_think(content):
        return "unclosed_think"
    visible_text = strip_think_tags(content)

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
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in word_tokens(text)
        if len(token) > 2 and token not in STOPWORDS
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


def sample_word_target(requirement: str, seed: int) -> int:
    return 0


def length_instruction(requirement: str, target: int) -> str:
    return requirement


def word_band(requirement: str, target: int = 0) -> tuple[int, int, int]:
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
    multiplier = max(WORD_MULTIPLIER_FLOOR, 1.0 - (distance / tolerance))
    return multiplier, {"multiplier": multiplier, "zero_words": False, **details}


def detect_structured_markers(text: str) -> str | None:
    if not text:
        return None
    for tag in (
        "Response-Style",
        "Response-Length",
        "user-turn-output",
        "conversation",
    ):
        if re.search(rf"</?{re.escape(tag)}(?:\s[^>]*)?>", text, flags=re.IGNORECASE):
            return tag
    return None


def parse_tag_int(text: str, tag: str, min_value: int, max_value: int) -> int | None:
    matches = re.findall(
        rf"<{re.escape(tag)}>\s*([0-9]+)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        return None
    value = int(matches[0])
    return value if min_value <= value <= max_value else None


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
    if re.search(r"</?[A-Za-z][^>]{0,200}>", text):
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


def parse_follow_up_message(raw: str) -> str:
    text, _, _ = validate_follow_up_message(raw)
    return text
