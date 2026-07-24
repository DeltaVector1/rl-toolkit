"""Answer extraction and the deterministic graders for the reasoning tasks.

Ported verbatim from the v0 env (`nemotron_reasoning.py` and its shared guardrails module)
so the ported rewards parse and grade answers exactly as v0 did:

* `extract_final_answer` / `normalize_answer` — the shared answer parsing.
* `math_answers_equivalent` — symbolic equivalence via `math-verify` (the v0 `math` grader).
* `reasoning_gym_reward` — reasoning_gym's own per-dataset scorer, with a normalize-equality
  fallback (the v0 `reasoning_gym` grader).
* `parse_grid` / `grid_reward` — ARC-AGI output-grid parsing and exact-match (the v0 `arc_agi`
  grader).
* `parse_equivalence_score` — the `<score>` tag parser the science answer-equivalence judge uses.

These are pure/deterministic (the judge-graded science subtask lives in `taskset.py`) and were
not implicated in any audit finding.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

# math_verify's signal-based timeout only fires on the main thread, so v0 called it with
# parsing_timeout=None (inside a worker thread) and bounded wall-time from the reward. Silence
# its per-call "timeout disabled" warning, exactly as v0 did.
logging.getLogger("math_verify").setLevel(logging.ERROR)

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
SCORE_TAG_RE = re.compile(r"<score>\s*([01])\s*</score>|<score>\s*([01])", re.IGNORECASE)
JSON_GRID_RE = re.compile(r"(\[\s*\[.*?\]\s*\])", re.DOTALL)


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


# --- math (math-verify symbolic equivalence) --------------------------------


_MATH_DELIMITERS = (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$"))


def strip_math_delimiters(text: str) -> str:
    """Peel LaTeX display/inline delimiters off an answer.

    Nearly half the math golds ship wrapped in ``\\(...\\)`` or ``\\[...\\]``. Wrapping those
    in ``\\boxed{}`` as-is produces ``\\boxed{\\(x\\)}``, which the parser reads differently
    from the model's bare ``x`` — a correct answer scored 0 about 10% of the time.
    """
    text = (text or "").strip()
    for _ in range(3):  # handles nesting like \[ \( x \) \]
        for opening, closing in _MATH_DELIMITERS:
            if text.startswith(opening) and text.endswith(closing) and len(text) > len(opening) + len(closing) - 1:
                text = text[len(opening) : -len(closing)].strip()
                break
        else:
            return text
    return text


def math_answers_equivalent(candidate: str, expected: str, raw_response: str = "") -> float:
    """Symbolic answer equivalence via math-verify's sympy grader (no LLM judge).

    `raw_response` is the full reply; math-verify extracts from it directly, which recovers
    answers our own extraction mangled. Upstream passes the whole generation for this reason.
    """
    if not candidate or not expected:
        return 0.0
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify

    gold = mv_parse(f"\\boxed{{{strip_math_delimiters(expected)}}}", parsing_timeout=None)
    pred = mv_parse(f"\\boxed{{{strip_math_delimiters(candidate)}}}", parsing_timeout=None)
    if mv_verify(gold, pred, timeout_seconds=None):
        return 1.0
    if raw_response:
        return 1.0 if mv_verify(gold, mv_parse(raw_response, parsing_timeout=None), timeout_seconds=None) else 0.0
    return 0.0


# --- reasoning_gym ----------------------------------------------------------

_RG_SCORE_FN_CACHE: dict[str, Any] = {}


def resolve_rg_score_fn(source_dataset: str):
    if not source_dataset:
        return None
    if source_dataset in _RG_SCORE_FN_CACHE:
        return _RG_SCORE_FN_CACHE[source_dataset]
    try:
        import reasoning_gym

        fn = reasoning_gym.get_score_answer_fn(source_dataset)
    except Exception:
        fn = None
    _RG_SCORE_FN_CACHE[source_dataset] = fn
    return fn


def reasoning_gym_reward(
    model_answer: str, answer: str, source_dataset: str, entry_metadata_json: str, question: str = ""
) -> float:
    """Grade with reasoning_gym's own per-dataset scorer, else normalize-equality fallback."""
    score_fn = resolve_rg_score_fn(source_dataset)
    if score_fn is not None:
        try:
            entry_metadata = json.loads(entry_metadata_json or "{}")
        except Exception:
            entry_metadata = {}
        entry = {"question": question, "answer": answer, "metadata": entry_metadata}
        try:
            return float(score_fn(answer=model_answer, entry=entry))
        except Exception as error:
            # Falling through to string equality silently turns a broken scorer into noisy
            # rewards, so say so loudly even though we still degrade gracefully.
            print(f"reasoning_gym scorer failed for {source_dataset}: {error!r}", flush=True)
    return 1.0 if normalize_answer(model_answer) == normalize_answer(answer) else 0.0


# --- ARC-AGI transductive grids ---------------------------------------------


def is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row and all(isinstance(cell, int) for cell in row) for row in value)
    )


def parse_grid(text: str) -> list[list[int]] | None:
    candidate = extract_final_answer(text)
    # Some replies carry literal "\n" escapes rather than real newlines; without this the
    # whole grid reads as one unparseable line and a correct answer scores 0.
    if "\\n" in candidate and "\n" not in candidate:
        candidate = candidate.replace("\\n", "\n")
    match = JSON_GRID_RE.search(candidate)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if is_grid(parsed):
                return parsed
        except Exception:
            pass
    try:
        parsed = ast.literal_eval(candidate)
        if is_grid(parsed):
            return parsed
    except Exception:
        pass
    rows: list[list[int]] = []
    for raw_line in candidate.splitlines():
        line = raw_line.strip().strip("|")
        if not line:
            continue
        if re.fullmatch(r"[0-9 ]+", line):
            parts = line.split() if " " in line else list(line)
            rows.append([int(part) for part in parts])
    return rows if is_grid(rows) else None


def grid_reward(visible_text: str, answer: str) -> float:
    predicted = parse_grid(visible_text)
    try:
        expected = json.loads(answer)
    except Exception:
        expected = None
    return 1.0 if predicted is not None and predicted == expected else 0.0
