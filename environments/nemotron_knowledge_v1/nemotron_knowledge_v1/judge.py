"""The answer-equivalence judge for the QA-abstention task.

Ported from v0's `judge_equivalence` / `EQUIVALENCE_PROMPT`. Unlike the anti-hacking guard
judges (which fail *open*), this is a correctness grader and faithfully stays fail-*closed*:
a missing or self-contradictory verdict parses to `None`, and the task treats that as "not
equivalent" (0.0) rather than crediting an unverified answer. `EquivalenceJudge.parse`
returns `None` on no verdict; the task reward maps `None` to 0.0.
"""

from __future__ import annotations

import verifiers.v1 as vf

from nemotron_knowledge_v1.answers import parse_equivalence_score

EQUIVALENCE_PROMPT = """You are grading final-answer equivalence.

Task:
{question}

Reference answer:
{expected}

Candidate answer:
{candidate}

Score 1 if the candidate final answer is mathematically, scientifically, or factually equivalent to the reference answer. Score 0 otherwise. Ignore formatting differences, but do not give credit for unsupported guesses or answers that only partially overlap.

Output exactly:
<score>0</score> or <score>1</score>
"""


class EquivalenceJudgeConfig(vf.JudgeConfig):
    """Same local vLLM endpoint the v0 knowledge env graded against."""

    model: str = "google/gemma-4-26B-A4B-it"
    base_url: str = "http://198.13.252.41:40002/v1"
    api_key_var: str = "VLLM_API_KEY"
    sampling: vf.JudgeSamplingConfig = vf.JudgeSamplingConfig(temperature=0.0, max_tokens=2048)


class EquivalenceJudge(vf.Judge[float, EquivalenceJudgeConfig]):
    prompt = EQUIVALENCE_PROMPT

    def parse(self, response: vf.JudgeResponse[float]) -> float | None:
        return parse_equivalence_score(response.text)
