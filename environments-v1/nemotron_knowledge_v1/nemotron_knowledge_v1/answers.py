"""Answer extraction and normalization for the knowledge tasks.

Copied verbatim from the v0 env (`nemotron_knowledge.py` and its shared guardrails module)
so the ported rewards parse MCQA letters and QA answers exactly as v0 did. These are pure
functions with no audit findings against them.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

MCQA_ANSWER_RE = re.compile(r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])", re.IGNORECASE)
MCQA_FALLBACK_RE = re.compile(r"\b([A-P])\b\s*$", re.MULTILINE)

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
SCORE_TAG_RE = re.compile(r"<score>\s*([01])\s*</score>|<score>\s*([01])", re.IGNORECASE)

ABSTENTION_ANSWERS = {"[idk]", "idk", "i don't know", "i do not know"}


def coerce(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
    return value


def last_boxed_content(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}``, honouring nested braces."""
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text or "")]
    for start in reversed(starts):
        open_idx = text.find("{", start)
        if open_idx < 0:
            continue
        depth = 0
        for idx in range(open_idx, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[open_idx + 1 : idx].strip()
    return None


def mcqa_extract_letter(text: str) -> str:
    if not text:
        return ""
    boxed = last_boxed_content(text)
    if boxed is not None:
        boxed = boxed.strip().upper()
        if re.fullmatch(r"[A-P0-9]", boxed):
            return boxed
    matches = MCQA_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().upper()
    matches = MCQA_FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].strip().upper()
    return text.strip().upper()


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    answer_matches = list(ANSWER_TAG_RE.finditer(text))
    if answer_matches:
        return answer_matches[-1].group(1).strip()
    boxed = last_boxed_content(text)
    if boxed is not None:
        return boxed
    for line in reversed(text.splitlines()):
        answer_line = ANSWER_LINE_RE.match(line)
        if answer_line:
            return answer_line.group(1).strip()
    return text.strip()


def normalize_answer(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\\\[(.*?)\\\]", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value, flags=re.DOTALL)
    value = value.replace("$", "")
    value = re.sub(r"\\boxed\{(.*?)\}", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value.strip(" .,:;`'\"")


def parse_equivalence_score(verdict: str) -> float | None:
    """Return 1.0/0.0, or None when the verdict is missing or self-contradictory."""
    values = [
        value
        for match in SCORE_TAG_RE.finditer(verdict or "")
        if (value := match.group(1) or match.group(2)) in {"0", "1"}
    ]
    if not values or len(set(values)) > 1:
        return None
    return 1.0 if values[-1] == "1" else 0.0
