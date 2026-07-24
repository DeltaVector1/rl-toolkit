"""Parity test: v0 decensor reward vs. the v1 port, on canned rollouts.

Both sides run the SAME (system_prompt, conversation, model_response) cases with judges
stubbed to fixed verdicts, so only the deterministic multiplicative math is compared.

  * v0: the real `DecensorRubric._run_decensor_rollout`, imported from
    /root/prime/rl_environments/decensor-env, with `_judge_text` monkeypatched.
  * v1: `decensor_v1.scoring.score_decensor` fed the same verdicts.

Most cases assert exact equality. The audit-fix cases assert the *intended* behavior:
  - FIX #2: "Be verbose" at an unreachable word count is reverted to v0 exactly (plain parity).
  - reasoning_coherency: re-added as a real working gate fed the true reasoning trace. Both
    sides enable it, so a stubbed verdict of 0 zeroes the product on both and 1 is a no-op
    (exact parity either way).
  - FIX #6: the two judge failure modes are split. A judge ENDPOINT outage tanks the v1
    reward to 0 (matching v0's zero-on-judge-error) and sets `decensor_judge_endpoint_down`.
    A received-but-unparseable reply (parse failure) stays fail-open on the v1 side.

Run from the prime-modern venv:
    uv run --project /root/prime-modern pytest \
        /root/prime-modern/custom_environments/decensor_v1/tests/test_parity.py -v
or directly:
    uv run --project /root/prime-modern python \
        /root/prime-modern/custom_environments/decensor_v1/tests/test_parity.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "/root/prime/rl_environments/decensor-env")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import decensor_env as v0  # noqa: E402
from decensor_v1.scoring import ScoringConfig, Verdicts, score_decensor  # noqa: E402
from decensor_v1.text import length_instruction  # noqa: E402

PENALTY = 0.1
META_MULT = 0.01
TOL = 0.60


@dataclass
class Case:
    name: str
    bucket: str
    word_requirement: str
    words: int
    verdicts: dict  # refusal(1-9), coherency, reasoning_ethics, meta_commentary, md_presence, md_correctness
    word_target: int = 0  # stated target; 0 => the requirement's static band (v0 shape)
    include_reasoning: bool = True
    unclosed_think: bool = False
    truncated: bool = False
    judge_error: str | None = None  # name of a judge whose ENDPOINT errors (outage, FIX #6)
    parse_failure: str | None = None  # name of a judge that replies with an unparseable tag
    reasoning_enabled: bool = True
    # None => assert exact parity; a float => assert v1 equals this (fixed behavior)
    expect_v1: float | None = None
    expect_v0: float | None = None
    expect_endpoint_down: bool = False  # assert v1 sets decensor_judge_endpoint_down


def build_assistant(case: Case) -> str:
    if case.unclosed_think:
        return "<think> planning the answer but ran out of tokens"
    visible = " ".join(["lorem"] * case.words) if case.words > 0 else ""
    if case.include_reasoning:
        return f"<think> internal reasoning about the topic here </think>{visible}"
    return visible


def build_conversation(case: Case):
    body = length_instruction(case.word_requirement, case.word_target)
    system = f"<Response-Length>\n{body}\n</Response-Length>" if case.word_requirement else "sys"
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": "walk me through it"}]
    completion = [{"role": "assistant", "content": build_assistant(case)}]
    return prompt, completion


ALL_JUDGE_NAMES = (
    "refusal", "coherency", "reasoning_ethics", "reasoning_coherency",
    "meta_commentary", "md_presence", "md_correctness",
)


def verdict_strings(case: Case) -> dict:
    v = case.verdicts
    out = {
        "refusal": f"<Refusal_Score>{v.get('refusal', 1)}</Refusal_Score>",
        "coherency": f"<score>{v.get('coherency', 1)}</score>",
        "reasoning_ethics": f"<score>{v.get('reasoning_ethics', 0)}</score>",
        "reasoning_coherency": f"<score>{v.get('reasoning_coherency', 1)}</score>",
        "meta_commentary": f"<score>{v.get('meta_commentary', 0)}</score>",
        "md_presence": f"<score>{v.get('md_presence', 1)}</score>",
        "md_correctness": f"<score>{v.get('md_correctness', 1)}</score>",
    }
    return out


async def v0_reward(case: Case) -> float:
    rubric = v0.DecensorRubric(
        judge_client=None, judge_model="m", judge_temperature=0.8, judge_min_p=0.05,
        judge_timeout=1.0, judge_max_retries=0, judge_retry_backoff=0.0,
        word_count_tolerance_percent=TOL, incoherent_penalty_multiplier=PENALTY,
        meta_commentary_multiplier=META_MULT, enable_word_count=True,
        enable_structured_markers=True, enable_markdown_judges=True,
        enable_reasoning_trace=True, enable_reasoning_coherency=True,
        enable_meta_commentary=True, reasoning_enabled=case.reasoning_enabled,
        markdown_exempt_probability=0.0,  # deterministic: never exempt
    )
    strings = verdict_strings(case)

    async def fake_judge_text(name, prompt):
        if case.judge_error == name:
            # Endpoint outage: gemma unreachable → v0 gates the rollout to 0 on judge error.
            return {"response": None, "error": "simulated judge outage", "attempt_count": 1, "failed_attempts": []}
        if case.parse_failure == name:
            # Gemma replied, but the verdict is unparseable (no error) → v0 parses to None.
            return {"response": "no score tag here", "error": None, "attempt_count": 1, "failed_attempts": []}
        return {"response": strings[name], "error": None, "attempt_count": 1, "failed_attempts": []}

    rubric._judge_text = fake_judge_text
    prompt, completion = build_conversation(case)
    state = {"info": {"word_requirement": case.word_requirement, "prompt_bucket": case.bucket}}
    if case.truncated:
        state["is_truncated"] = True
    reward, _ = await rubric._run_decensor_rollout(prompt, completion, state)
    return reward


def v1_reward(case: Case) -> tuple[float, dict]:
    prompt, completion = build_conversation(case)
    v = case.verdicts

    def maybe(name, default):
        # Both an endpoint outage and a parse failure resolve to a None verdict on the v1
        # side; the outage additionally sets judge_endpoint_down below (FIX #6).
        if case.judge_error == name or case.parse_failure == name:
            return None
        return v.get(name, default)

    verdicts = Verdicts(
        refusal=maybe("refusal", 1),
        coherency=maybe("coherency", 1),
        reasoning_ethics=maybe("reasoning_ethics", 0),
        reasoning_coherency=maybe("reasoning_coherency", 1),
        meta_commentary=maybe("meta_commentary", 0),
        md_presence=maybe("md_presence", 1),
        # md_correctness only exists for styles/markdown; plaintext leaves it None (unused)
        md_correctness=(maybe("md_correctness", 1) if case.bucket in ("styles", "markdown") else None),
    )
    cfg = ScoringConfig(
        incoherent_penalty_multiplier=PENALTY, meta_commentary_multiplier=META_MULT,
        word_count_tolerance_percent=TOL, reasoning_enabled=case.reasoning_enabled,
    )
    reward, breakdown = score_decensor(
        prompt, completion, verdicts, bucket=case.bucket,
        word_requirement=case.word_requirement, word_target=case.word_target,
        is_truncated=case.truncated,
        md_exempt=False, judge_endpoint_down=case.judge_error is not None, config=cfg,
    )
    return reward, breakdown


CASES: list[Case] = [
    # --- Compliant baselines, each length bucket in-band ---
    Case("compliant_terse_inband", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}),
    Case("compliant_medium_inband", "markdown", "Respond in a medium-length response", 300,
         {"refusal": 1, "md_presence": 1}),
    Case("compliant_styles_inband", "styles", "Respond tersely", 100,
         {"refusal": 2, "md_presence": 0, "md_correctness": 1}),
    # --- Refusal severity → base reward ---
    Case("hard_refusal", "markdown", "Respond tersely", 100,
         {"refusal": 9, "md_presence": 1}),
    Case("mid_refusal", "plaintext", "Respond tersely", 100,
         {"refusal": 5, "md_presence": 0}),
    # --- Length falloff (achievable buckets => exact parity) ---
    Case("terse_over_band", "markdown", "Respond tersely", 300,
         {"refusal": 1, "md_presence": 1}),
    Case("medium_over_band", "markdown", "Respond in a medium-length response", 450,
         {"refusal": 1, "md_presence": 1}),
    Case("terse_under_band", "markdown", "Respond tersely", 40,
         {"refusal": 1, "md_presence": 1}),
    # --- Markdown handling per bucket ---
    Case("markdown_bucket_missing_md", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 0}),  # markdown expected, absent => penalty
    Case("plaintext_bucket_has_md", "plaintext", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}),  # plaintext, md present => penalty
    Case("styles_bad_correctness", "styles", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 0, "md_correctness": 0}),  # correctness penalty
    # --- Coherency / reasoning-ethics / meta multipliers ---
    Case("incoherent", "markdown", "Respond tersely", 100,
         {"refusal": 1, "coherency": 0, "md_presence": 1}),
    Case("reasoning_ethics_refuses", "markdown", "Respond tersely", 100,
         {"refusal": 1, "reasoning_ethics": 1, "md_presence": 1}),  # => reasoning_mult 0
    # reasoning_coherency is a real working gate again (both sides enable it).
    Case("reasoning_incoherent", "markdown", "Respond tersely", 100,
         {"refusal": 1, "reasoning_coherency": 0, "md_presence": 1}),  # => reasoning_mult 0, product 0
    Case("reasoning_coherent_noop", "markdown", "Respond tersely", 100,
         {"refusal": 1, "reasoning_coherency": 1, "md_presence": 1}),  # verdict 1 => no effect
    Case("meta_commentary", "markdown", "Respond tersely", 100,
         {"refusal": 1, "meta_commentary": 1, "md_presence": 1}),
    Case("stacked_penalties", "styles", "Respond in a medium-length response", 300,
         {"refusal": 3, "coherency": 0, "meta_commentary": 1, "md_presence": 0, "md_correctness": 0}),
    # --- Hard gates (both sides zero) ---
    Case("gate_truncated", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}, truncated=True, expect_v1=0.0, expect_v0=0.0),
    Case("gate_unclosed_think", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}, unclosed_think=True, expect_v1=0.0, expect_v0=0.0),
    Case("gate_missing_reasoning", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}, include_reasoning=False, expect_v1=0.0, expect_v0=0.0),
    Case("gate_zero_words", "markdown", "Respond tersely", 0,
         {"refusal": 1, "md_presence": 1}, expect_v1=0.0, expect_v0=0.0),
    # --- FIX #2: an off-target length costs credit but no longer zeros the product, so the
    #     rollout still carries gradient. v0 falls off to 0.0; v1 floors at 0.1. ---
    Case("verbose_short_falloff", "markdown", "Be verbose", 500,
         {"refusal": 1, "md_presence": 1}, expect_v0=0.0, expect_v1=0.9 * 0.1),
    # --- FIX #2: a stated target defines the band, so hitting the number the prompt asked
    #     for earns full length credit (v0 scores it against the hidden 1900-2100 band). ---
    Case("verbose_stated_target_inband", "markdown", "Be verbose", 1000,
         {"refusal": 1, "md_presence": 1}, word_target=1000,
         expect_v0=0.9 * 0.25, expect_v1=0.9),
    # --- FIX #6: judge ENDPOINT outage tanks the reward + sets the metric (both sides 0) ---
    Case("fix6_refusal_endpoint_down", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}, judge_error="refusal",
         expect_v0=0.0, expect_v1=0.0, expect_endpoint_down=True),
    # --- FIX #6: a parse failure (gemma replied, tag unparseable) stays fail-open on v1.
    #     reasoning_coherency parse-fail defaults to 1 (no gate) on both sides => parity. ---
    Case("fix6_reasoning_coherency_parse_fail", "markdown", "Respond tersely", 100,
         {"refusal": 1, "md_presence": 1}, parse_failure="reasoning_coherency"),
]


def run() -> int:
    failures = 0
    for case in CASES:
        r0 = asyncio.run(v0_reward(case))
        r1, breakdown = v1_reward(case)

        if case.expect_v0 is not None:
            ok0 = abs(r0 - case.expect_v0) < 1e-9
        else:
            ok0 = True
        if case.expect_v1 is not None:
            ok1 = abs(r1 - case.expect_v1) < 1e-9
            parity = ok1 and ok0
            kind = "FIXED"
        else:
            parity = abs(r0 - r1) < 1e-9 and ok0
            ok1 = True
            kind = "PARITY"

        metric_ok = breakdown.get("decensor_judge_endpoint_down", 0.0) == (1.0 if case.expect_endpoint_down else 0.0)
        parity = parity and metric_ok

        status = "OK " if parity else "FAIL"
        if not parity:
            failures += 1
        metric_note = "" if not case.expect_endpoint_down else " [endpoint_down set]"
        print(f"[{status}] {kind:6} {case.name:34} v0={r0:.6f} v1={r1:.6f}"
              + ("" if case.expect_v1 is None else f" (expect v1={case.expect_v1:.6f})") + metric_note)

    print()
    if failures:
        print(f"{failures} case(s) FAILED")
    else:
        print(f"All {len(CASES)} cases passed.")
    return failures


def test_parity():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
