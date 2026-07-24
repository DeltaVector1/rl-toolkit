"""Deterministic IFEval/IFBench constraint checkers."""

from __future__ import annotations

import json
import re
import string
from typing import Any, Callable

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

# === IFEval / IFBench constraint checkers (48 IDs) ==========================


WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]+", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


try:
    from verifiable_instructions.instructions_registry import INSTRUCTION_DICT as _UPSTREAM_CHECKERS
except ImportError:  # pragma: no cover - the package is a declared dependency
    _UPSTREAM_CHECKERS = {}


def _strip_code_blocks(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text)


def _split_paragraphs(text: str, separator: str = "\n\n") -> list[str]:
    if separator == "***" or "* * *" in text:
        parts = re.split(r"\n\s*\*\s*\*\s*\*\s*\n|\n\*{3,}\n", text.strip())
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    if separator in ("\n\n", "\n \n"):
        return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return [p.strip() for p in text.strip().split(separator) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    matches = SENTENCE_RE.findall(text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return [s.strip() for s in text.splitlines() if s.strip()]


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _normalize_word(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha())


def _check_relation(value: int, target: int, relation: str | None) -> bool:
    if relation is None:
        return value == target
    rel = str(relation).strip().lower()
    if rel in {"at least", ">="}:
        return value >= target
    if rel in {"at most", "<="}:
        return value <= target
    if rel in {"more than", ">"}:
        return value > target
    if rel in {"less than", "<"}:
        return value < target
    return value == target


def _ifeval_length_num_words(text, kw):
    n = kw.get("num_words") or kw.get("N")
    return n is not None and _check_relation(_count_words(text), int(n), kw.get("relation"))


def _ifeval_length_num_sentences(text, kw):
    n = kw.get("num_sentences") or kw.get("N")
    return n is not None and _check_relation(len(_split_sentences(text)), int(n), kw.get("relation"))


def _ifeval_length_num_paragraphs(text, kw):
    """Exactly N paragraphs split by ``***``.

    The old `\\n\\n` fallback gave the model two independent chances to satisfy one
    constraint, which is reward leakage — ordinary prose is already `\\n\\n`-separated.
    """
    n = kw.get("num_paragraphs") or kw.get("N")
    if n is None:
        return False
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    return count == int(n)


def _ifeval_nth_paragraph_first_word(text, kw):
    """The Nth `\\n\\n`-separated paragraph must start with a given word."""
    paragraphs = re.split(r"\n\n", text)
    count = sum(1 for paragraph in paragraphs if paragraph.strip())
    nth = int(kw.get("nth_paragraph", 1))
    if not 1 <= nth <= count or count != int(kw.get("num_paragraphs", count)):
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    # Truncate at the first punctuation rather than stripping all non-letters, so a
    # hyphenated "Wrap-around" reads as "wrap" and not "wraparound".
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first_word = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        first_word += letter.lower()
    return first_word == str(kw.get("first_word", "")).strip().lower()


def _ifeval_last_word_answer(text, kw):
    target = str(kw.get("last_word") or kw.get("end_phrase") or "").strip().lower()
    if not target:
        return False
    words = WORD_RE.findall(text)
    return bool(words) and _normalize_word(words[-1]) == _normalize_word(target)


def _ifeval_first_word_answer(text, kw):
    target = str(kw.get("first_word") or "").strip().lower()
    if not target:
        return False
    words = WORD_RE.findall(text)
    return bool(words) and _normalize_word(words[0]) == _normalize_word(target)


def _ifeval_first_word_sent(text, kw):
    target = str(kw.get("first_word") or "").strip().lower()
    if not target:
        return False
    sentences = _split_sentences(text)
    if not sentences:
        return False
    for s in sentences:
        words = WORD_RE.findall(s)
        if not words or _normalize_word(words[0]) != _normalize_word(target):
            return False
    return True


def _ifeval_last_word_sent(text, kw):
    target = str(kw.get("last_word") or "").strip().lower()
    if not target:
        return False
    sentences = _split_sentences(text)
    if not sentences:
        return False
    for s in sentences:
        words = WORD_RE.findall(s)
        if not words or _normalize_word(words[-1]) != _normalize_word(target):
            return False
    return True


def _ifeval_end_phrase(text, kw):
    phrase = str(kw.get("end_phrase") or "").strip()
    if not phrase:
        return False
    # Only surrounding double quotes are ignored. Stripping punctuation from both sides let
    # "Any other questions" satisfy a constraint that demanded "Any other questions?".
    return text.strip().strip('"').lower().endswith(phrase.lower())


def _ifeval_start_phrase(text, kw):
    phrase = str(kw.get("start_phrase") or "").strip()
    if not phrase:
        return False
    return text.strip().lower().startswith(phrase.lower())


def _ifeval_forbidden_words(text, kw):
    words = kw.get("forbidden_words") or []
    if isinstance(words, str):
        words = [words]
    # Word boundaries, not substrings: forbidding "name" must not fail on "Names".
    return not any(re.search(rf"\b{w}\b", text, flags=re.IGNORECASE) for w in words if w)


def _ifeval_keyword_existence(text, kw):
    keywords = kw.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    text_lower = text.lower()
    return all(k.lower() in text_lower for k in keywords)


def _ifeval_keyword_frequency(text, kw):
    keyword = str(kw.get("keyword") or "").lower()
    n = kw.get("frequency") if kw.get("frequency") is not None else kw.get("N")
    if not keyword or n is None:
        return False
    occurrences = len(re.findall(rf"\b{re.escape(keyword)}\b", text.lower()))
    return _check_relation(occurrences, int(n), kw.get("relation"))


def _ifeval_word_once(text, kw):
    keyword = str(kw.get("keyword") or "").lower()
    if not keyword:
        return False
    return len(re.findall(rf"\b{re.escape(keyword)}\b", text.lower())) == 1


def _ifeval_letter_frequency(text, kw):
    letter = str(kw.get("letter") or "").lower()
    n = kw.get("let_frequency") if kw.get("let_frequency") is not None else kw.get("N")
    if not letter or n is None:
        return False
    return _check_relation(text.lower().count(letter), int(n), kw.get("let_relation") or kw.get("relation"))


def _ifeval_letter_counting(text, kw):
    n = kw.get("N") if kw.get("N") is not None else kw.get("num_letters")
    if n is None:
        return False
    letter_count = sum(1 for c in text if c.isalpha())
    return _check_relation(letter_count, int(n), kw.get("relation"))


def _ifeval_no_comma(text, _kw):
    return "," not in text


def _ifeval_no_dot(text, _kw):
    return "." not in text


def _ifeval_no_exclamation(text, _kw):
    return "!" not in text


def _ifeval_capital_word_frequency(text, kw):
    n = kw.get("capital_frequency") if kw.get("capital_frequency") is not None else kw.get("N")
    if n is None:
        return False
    caps = sum(1 for w in WORD_RE.findall(text) if len(w) > 1 and w.isupper())
    return _check_relation(caps, int(n), kw.get("capital_relation") or kw.get("relation"))


def _ifeval_english_capital(text, _kw):
    s = text.strip()
    return bool(s) and s == s.upper() and any(c.isalpha() for c in s)


def _ifeval_english_lowercase(text, _kw):
    s = text.strip()
    return bool(s) and s == s.lower() and any(c.isalpha() for c in s)


def _ifeval_json_format(text, _kw):
    candidate = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if m:
        candidate = m.group(1)
    try:
        json.loads(candidate)
        return True
    except Exception:
        return False


def _ifeval_number_bullets(text, kw):
    n = kw.get("num_bullets") if kw.get("num_bullets") is not None else kw.get("N")
    if n is None:
        return False
    bullets = re.findall(r"^\s*[\-*•]\s+", text, re.MULTILINE)
    return _check_relation(len(bullets), int(n), kw.get("relation"))


def _ifeval_number_placeholders(text, kw):
    # "at least N", hardcoded: the prompt says "must contain at least N placeholders" and the
    # rows carry no `relation`, so defaulting to `==` failed the model for exceeding the ask.
    n = kw.get("num_placeholders") if kw.get("num_placeholders") is not None else kw.get("N")
    if n is None:
        return False
    return len(re.findall(r"\[.*?\]", text)) >= int(n)


def _ifeval_number_highlighted(text, kw):
    """"Highlight at least N sections" — *italic* and **bold** both count."""
    n = kw.get("num_highlights") if kw.get("num_highlights") is not None else kw.get("N")
    if n is None:
        return False
    count = sum(1 for h in re.findall(r"\*[^\n\*]*\*", text) if h.strip("*").strip())
    count += sum(
        1
        for h in re.findall(r"\*\*[^\n\*]*\*\*", text)
        if h.removeprefix("**").removesuffix("**").strip()
    )
    return count >= int(n)


def _ifeval_two_responses(text, _kw):
    """Exactly two non-empty responses split by ``******``, and they must differ.

    A bare ``"******" in text`` passed on any six asterisks anywhere — including inside a
    Python code block — which is free reward for ignoring the instruction.

    Fenced code is removed before splitting. That deviates from upstream, which splits the
    raw text, but a separator inside a code fence is never the response separator the
    constraint asks for, and it accounted for 8 of 48 observed passes.
    """
    parts = _strip_code_blocks(text).split("******")
    non_empty = []
    for index, part in enumerate(parts):
        if part.strip():
            non_empty.append(part)
        elif index not in (0, len(parts) - 1):
            return False
    return len(non_empty) == 2 and non_empty[0].strip() != non_empty[1].strip()


def _ifeval_repeat_prompt(text, kw):
    prompt = str(kw.get("prompt_to_repeat") or kw.get("prompt") or "").strip()
    return bool(prompt) and prompt.lower() in text.lower()


def _ifeval_quotation(text, _kw):
    """Double quotes only — the constraint asks for them specifically."""
    s = text.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


def _ifeval_title(text, _kw):
    return bool(re.search(r"<<[^<>\n]+>>", text))


def _ifeval_square_brackets(text, _kw):
    s = text.strip()
    return s.startswith("[") and s.endswith("]")


def _ifeval_bigram_wrapping(text, _kw):
    bigrams = re.findall(r"<<([^<>\n]+)>>", text)
    if not bigrams:
        return False
    inside = sum(_count_words(b) for b in bigrams)
    total = _count_words(text)
    return total > 0 and inside / total >= 0.5


def _ifeval_constrained_response(text, _kw):
    candidates = ("my answer is yes.", "my answer is no.", "my answer is maybe.")
    return text.strip().lower() in candidates


def _ifeval_multiple_sections(text, kw):
    splitter = str(kw.get("section_spliter") or kw.get("section_splitter") or "SECTION").strip()
    n = kw.get("num_sections") if kw.get("num_sections") is not None else kw.get("N")
    if not splitter or n is None:
        return False
    # Split on the marker and count the gaps, case-sensitively, "at least N".
    sections = re.split(r"\s?" + re.escape(splitter) + r"\s?\d+\s?", text)
    return len(sections) - 1 >= int(n)


def _ifeval_table(text, kw):
    rows = [line for line in text.splitlines() if "|" in line]
    min_rows = int(kw.get("min_rows") or kw.get("N") or 1)
    min_cols = int(kw.get("min_cols") or 2)
    if len(rows) < min_rows:
        return False
    return any(len([cell for cell in row.split("|") if cell.strip()]) >= min_cols for row in rows)


def _ifeval_sentence_count(text, kw):
    n = kw.get("num_sentences") or kw.get("N")
    if n is None:
        return False
    return _check_relation(len(_split_sentences(text)), int(n), kw.get("relation"))


def _ifeval_unique_words(text, kw):
    n = kw.get("num_unique") or kw.get("N")
    if n is None:
        return False
    words = {_normalize_word(w) for w in WORD_RE.findall(text) if _normalize_word(w)}
    return _check_relation(len(words), int(n), kw.get("relation"))


def _ifeval_all_caps_target(text, kw):
    target = str(kw.get("target_string") or "").strip()
    if not target:
        return False
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    return bool(matches) and all(m.group(0).isupper() for m in matches)


def _ifeval_sentence_hyphens(text, _kw):
    body = _strip_code_blocks(text).strip()
    if not body:
        return False
    has_hyphen_joiners = bool(re.search(r"\w-\w", body))
    period_space = len(re.findall(r"[.!?]\s+[A-Za-z]", body))
    return has_hyphen_joiners and period_space == 0


def _ifeval_postscript(text, kw):
    marker = str(kw.get("postscript_marker") or "P.S.").strip()
    return marker.lower() in text.lower()


def _ifeval_repeat_phrase(text, kw):
    phrase = str(kw.get("phrase") or "").strip()
    n = kw.get("small_n") if kw.get("small_n") is not None else kw.get("N")
    if not phrase or n is None:
        return False
    tokens = phrase.split()
    if len(tokens) < 3:
        return text.lower().count(phrase.lower()) >= int(n)
    head, tail = tokens[0], tokens[-1]
    pattern = re.escape(head) + r"\s+\S+(?:\s+\S+){0," + str(max(0, len(tokens) - 2)) + r"}\s+" + re.escape(tail)
    return len(re.findall(pattern, text, re.IGNORECASE)) >= int(n)


def _ifeval_lowercase_counting(text, kw):
    n = kw.get("N") if kw.get("N") is not None else kw.get("max_count")
    if n is None:
        return False
    counts: dict[str, int] = {}
    for w in WORD_RE.findall(text):
        if w.islower():
            counts[w] = counts.get(w, 0) + 1
    if not counts:
        return True
    return _check_relation(max(counts.values()), int(n), kw.get("relation") or "at most")


def _ifeval_count_increment_word(text, kw):
    def _list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v]
        return []

    keyword1 = [k.lower() for k in _list(kw.get("keyword1"))]
    keyword2 = [k.lower() for k in _list(kw.get("keyword2"))]
    text_lower = text.lower()
    ok1 = all(len(re.findall(rf"\b{re.escape(k)}\b", text_lower)) == 1 for k in keyword1) if keyword1 else True
    ok2 = all(len(re.findall(rf"\b{re.escape(k)}\b", text_lower)) == 2 for k in keyword2) if keyword2 else True
    return ok1 and ok2


def _ifeval_count_unique(text, _kw):
    words = [_normalize_word(w) for w in WORD_RE.findall(text) if _normalize_word(w)]
    return len(words) == len(set(words))


def _ifeval_counting_composition(text, kw):
    n_sent = kw.get("n_sent")
    n_words = kw.get("n_words")
    if n_sent is None or n_words is None:
        return False
    paragraphs = _split_paragraphs(text, "***")
    if len(paragraphs) <= 1:
        paragraphs = _split_paragraphs(text, "\n\n")
    if not paragraphs:
        return False
    for p in paragraphs:
        sentences = _split_sentences(p)
        if len(sentences) != int(n_sent):
            return False
        for s in sentences:
            if _count_words(s) != int(n_words):
                return False
    return True


def _ifeval_keyword_specific_position(text, kw):
    keyword = str(kw.get("keyword") or "").lower()
    sent_idx = kw.get("n")
    word_idx = kw.get("m")
    if not keyword or sent_idx is None or word_idx is None:
        return False
    sentences = _split_sentences(text)
    if int(sent_idx) < 1 or int(sent_idx) > len(sentences):
        return False
    words = WORD_RE.findall(sentences[int(sent_idx) - 1])
    if int(word_idx) < 1 or int(word_idx) > len(words):
        return False
    return _normalize_word(words[int(word_idx) - 1]) == _normalize_word(keyword)


def _ifeval_no_adjacent_consecutive(text, _kw):
    words = WORD_RE.findall(text)
    for a, b in zip(words, words[1:]):
        if _normalize_word(a) and _normalize_word(a) == _normalize_word(b):
            return False
    return True


def _ifeval_palindrome(text, _kw):
    for w in WORD_RE.findall(text):
        norm = _normalize_word(w)
        if len(norm) >= 4 and norm == norm[::-1]:
            return True
    return False


def _ifeval_start_end(text, _kw):
    words = WORD_RE.findall(text)
    return len(words) >= 2 and _normalize_word(words[0]) == _normalize_word(words[-1])


def _ifeval_response_language(text, kw):
    lang = str(kw.get("language") or "").strip().lower()
    if not lang:
        return False
    try:
        detected = detect(text).lower()
    except Exception:
        return False
    return detected == lang or detected.startswith(lang + "-")


def _ifeval_paragraphs(text, _kw):
    return len(_split_paragraphs(text, "***")) >= 2


def _ifeval_paragraphs2(text, _kw):
    return len(_split_paragraphs(text, "\n\n")) >= 2


CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "change_case:capital_word_frequency": _ifeval_capital_word_frequency,
    "change_case:all_caps_target": _ifeval_all_caps_target,
    "change_case:english_capital": _ifeval_english_capital,
    "change_case:english_lowercase": _ifeval_english_lowercase,
    "combination:repeat_prompt": _ifeval_repeat_prompt,
    "combination:two_responses": _ifeval_two_responses,
    "copy:repeat_phrase": _ifeval_repeat_phrase,
    "count:count_increment_word": _ifeval_count_increment_word,
    "count:count_unique": _ifeval_count_unique,
    "count:counting_composition": _ifeval_counting_composition,
    "count:lowercase_counting": _ifeval_lowercase_counting,
    "detectable_content:number_placeholders": _ifeval_number_placeholders,
    "detectable_content:postscript": _ifeval_postscript,
    "detectable_format:bigram_wrapping": _ifeval_bigram_wrapping,
    "detectable_format:constrained_response": _ifeval_constrained_response,
    "detectable_format:json_format": _ifeval_json_format,
    "detectable_format:multiple_sections": _ifeval_multiple_sections,
    "detectable_format:number_bullet_lists": _ifeval_number_bullets,
    "detectable_format:number_paragraphs": _ifeval_length_num_paragraphs,
    "detectable_format:number_highlighted_sections": _ifeval_number_highlighted,
    "detectable_format:sentence_count": _ifeval_sentence_count,
    "detectable_format:sentence_hyphens": _ifeval_sentence_hyphens,
    "detectable_format:square_brackets": _ifeval_square_brackets,
    "detectable_format:table": _ifeval_table,
    "detectable_format:title": _ifeval_title,
    "first_word:first_word_answer": _ifeval_first_word_answer,
    "first_word:first_word_sent": _ifeval_first_word_sent,
    "keywords:existence": _ifeval_keyword_existence,
    "keywords:forbidden_words": _ifeval_forbidden_words,
    "keywords:frequency": _ifeval_keyword_frequency,
    "keywords:keyword_specific_position": _ifeval_keyword_specific_position,
    "keywords:letter_frequency": _ifeval_letter_frequency,
    "keywords:no_adjacent_consecutive": _ifeval_no_adjacent_consecutive,
    "keywords:palindrome": _ifeval_palindrome,
    "keywords:start_end": _ifeval_start_end,
    "keywords:word_count_different_numbers": _ifeval_keyword_frequency,
    "keywords:word_once": _ifeval_word_once,
    "language:response_language": _ifeval_response_language,
    "last_word:last_word_answer": _ifeval_last_word_answer,
    "last_word:last_word_sent": _ifeval_last_word_sent,
    "length_constraints:nth_paragraph_first_word": _ifeval_nth_paragraph_first_word,
    "length_constraints:number_paragraphs": _ifeval_length_num_paragraphs,
    "length_constraints:number_sentences": _ifeval_length_num_sentences,
    "length_constraints:number_words": _ifeval_length_num_words,
    "length_constraints:unique_words": _ifeval_unique_words,
    "letters:letter_counting": _ifeval_letter_counting,
    "letters:letter_counting2": _ifeval_letter_frequency,
    "paragraphs:paragraphs": _ifeval_paragraphs,
    "paragraphs:paragraphs2": _ifeval_paragraphs2,
    "punctuation:no_comma": _ifeval_no_comma,
    "punctuation:punctuation_dot": _ifeval_no_dot,
    "punctuation:punctuation_exclamation": _ifeval_no_exclamation,
    "startend:end_checker": _ifeval_end_phrase,
    "startend:quotation": _ifeval_quotation,
    "startend:start_checker": _ifeval_start_phrase,
}


def checker_supported(instruction_id: str) -> bool:
    """Can this constraint be graded at all? Rows with an ungradeable constraint are dropped
    at load — an unchecked constraint always counts as failed, making the row unwinnable."""
    return instruction_id in _UPSTREAM_CHECKERS or instruction_id in CHECKERS


def check_instruction(instruction_id: str, text: str, kwargs: dict | None) -> bool | None:
    """Does `text` satisfy one constraint? None when no checker covers the id.

    `verifiable-instructions` is authoritative — it is the package NVIDIA's own
    instruction-following server uses, and each class freezes its own comparison semantics
    (that is what stops "at least N" drifting back to "exactly N"). The hand-written
    `CHECKERS` remain as a fallback for the handful of ids the package doesn't define.
    """
    instruction_cls = _UPSTREAM_CHECKERS.get(instruction_id)
    if instruction_cls is not None:
        instruction = instruction_cls(instruction_id)
        # None means "unset" upstream: passing it through triggers each checker's random
        # re-sampling path, so the keys are dropped and the class default applies instead.
        instruction.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        return bool(instruction.check_following(text))

    checker = CHECKERS.get(instruction_id)
    return None if checker is None else bool(checker(text, kwargs or {}))


def _evaluate_constraints(text: str, ids: list[str], kwargs_list: list[dict]) -> dict[str, float]:
    total = max(1, len(ids))
    passed = 0
    unsupported = 0
    for cid, kw in zip(ids, kwargs_list):
        try:
            result = check_instruction(cid, text, kw)
        except Exception:
            continue
        if result is None:
            unsupported += 1
        elif result:
            passed += 1
    return {
        "pass_fraction": passed / total,
        "fully_passed": 1.0 if passed == total else 0.0,
        "unsupported_fraction": unsupported / total,
    }

