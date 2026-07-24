"""The instruction-following task judges, ported to `vf.Judge`.

Three v0 subtask families grade with an LLM judge:

* the ultra-judge subtasks (identity / sysbench / cfbench / multichallenge / inverse_ifeval)
  and adversarial ask a per-criterion YES/NO question — `YesNoJudge`;
* calendar asks for a single 0-3 integer score — `CalendarJudge`.

AUDIT FIX #1 (the whole reason this port is careful): in v0 `ultra_score` referenced
`JUDGE_MAX_TOKENS` — a name defined only in the guardrails module and never imported into the
env module — so every ultra_score call raised `NameError` *before* the judge ran and the reward
was silently recorded as a constant 0.0, killing all five ultra-judge subtasks. Here the judge
output budget is a real config field (`IFJudgeConfig.sampling.max_tokens`, default 2048 to match
v0's intended value) that `vf.Judge.complete` actually applies to the request. There is no
undefined name: the judge runs and produces a real reward.

The caller formats the full judge prompt (per-row `judge_prompt_template` or a default) and
hands it in as `judge_input`; these judges only carry the model/endpoint config and the verdict
parsing. Verdict parsing is fail-*closed* for these correctness judges (an unparseable or absent
verdict → `None` → the task scores it 0.0), matching v0; the anti-hacking guard judges in
`nemotron_guard_v1` are the ones that fail *open*.
"""

from __future__ import annotations

import re

import verifiers.v1 as vf
from nemotron_guard_v1.text import strip_think_tags

# --- Judge prompt templates (verbatim from v0) -------------------------------

CALENDAR_PROMPT = """You evaluate whether an assistant correctly scheduled a set of calendar events.

Expected calendar state:
{exp_cal_state}

Assistant response:
{response}

Answer with a single integer score 0-3 only.
"""

ADVERSARIAL_DEFAULT_TEMPLATE = """## Question
{prompt}

---
## Student Response
{model_response}

---
## Standard Response
{standard_response}

---
## Evaluation Criteria
{criteria}

---

Answer with just YES or NO.
"""

ULTRA_CRITERION_PROMPT = """You evaluate whether an assistant response satisfies a criterion.

Conversation / task prompt:
{prompt}

Reference response, if any:
{reference}

Assistant response:
{response}

Criterion:
{criterion}

Answer with just YES or NO.
"""


# --- Verdict parsing (verbatim from v0) --------------------------------------

YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
TAGGED_SCORE_RE = re.compile(r"<score>\s*([0-3])\s*</score>|<score>\s*([0-3])", re.IGNORECASE)
SCORE_RE = re.compile(r"^\s*([0-3])\s*[\.\!]*\s*$")


def parse_yes_no(verdict: str) -> str | None:
    """Take the last standalone YES/NO, ignoring any hidden reasoning.

    Judges are not always terse: a reasoning model answers after a <think> block, and requiring
    the whole reply to *be* the word YES silently scored every such verdict 0.
    """
    text = strip_think_tags(verdict or "").strip()
    matches = YES_NO_RE.findall(text)
    return matches[-1].upper() if matches else None


def parse_0_3_score(verdict: str) -> int | None:
    text = strip_think_tags(verdict or "").strip()
    tagged_values = [
        value for match in TAGGED_SCORE_RE.finditer(text) if (value := match.group(1) or match.group(2)) is not None
    ]
    if tagged_values:
        return int(tagged_values[-1]) if len(set(tagged_values)) == 1 else None
    for line in reversed(text.splitlines()):
        match = SCORE_RE.fullmatch(line)
        if match:
            return int(match.group(1))
    return None


# --- Prompt formatting helpers (verbatim from v0) ----------------------------


def format_with(template: str, fallback: str, **fields) -> str:
    """Adversarial's forgiving formatter: use the row template, fall back on a bad one."""
    base = template.strip() or fallback
    try:
        return base.format(**fields)
    except (KeyError, IndexError):
        return fallback.format(**fields)


# --- Judges ------------------------------------------------------------------


class IFJudgeConfig(vf.JudgeConfig):
    """The single vLLM endpoint the instruction-following judges grade against.

    Defaults mirror the v0 env. `sampling.max_tokens` is the audit-fix-#1 budget: a real
    field that `vf.Judge.complete` applies to every request, so a reasoning judge has room to
    think before its YES/NO or 0-3 verdict lands.
    """

    model: str = "google/gemma-4-26B-A4B-it"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_var: str = "VLLM_API_KEY"
    sampling: vf.JudgeSamplingConfig = vf.JudgeSamplingConfig(temperature=0.0, max_tokens=2048)


class YesNoJudge(vf.Judge[str, IFJudgeConfig]):
    """Grades a single YES/NO criterion; the caller supplies the fully-formatted prompt."""

    def build_messages(self, *, judge_input: str) -> str:
        return judge_input

    def parse(self, response: vf.JudgeResponse[str]) -> str | None:
        return parse_yes_no(response.text)


class CalendarJudge(vf.Judge[int, IFJudgeConfig]):
    """Grades a calendar schedule with a single 0-3 integer score."""

    def build_messages(self, *, judge_input: str) -> str:
        return judge_input

    def parse(self, response: vf.JudgeResponse[int]) -> int | None:
        return parse_0_3_score(response.text)
