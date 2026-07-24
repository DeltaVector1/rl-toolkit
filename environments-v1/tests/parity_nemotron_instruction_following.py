"""Parity test: v1 nemotron-instruction-following reward == v0 reward (with the audit fix).

For canned cases per subtask family this computes the reward BOTH ways and compares
`task_reward`, `anti_hacking_multiplier`, and the final reward:

  (a) v0: import the v0 modules from /root/prime, grab the real per-subtask reward closure out
      of the v0 env factory (`_ifeval_env`, `_structured_v2_env`, `_ultra_judge_env`, ...), wrap
      it in the v0 `GuardedRubric` (format_reward_weight pinned to 0 — the dead format factor,
      audit fix #3), and drive `score_rollout` over a state dict.
  (b) v1: build the ported `NemotronIFTask` (one class, dispatching on `data.subtask`),
      construct a `vf.Trace`, and call the task's `@vf.reward`.

Three kinds of assertion:
  * PARITY — deterministic families (ifeval pass-fraction, structured-output validity, format
    verification) and the working v0 judge families (calendar, adversarial): v1 == v0 exactly.
  * FIXED (AUDIT FIX #1) — the five ultra-judge subtasks (identity / sysbench / cfbench /
    multichallenge / inverse_ifeval). In v0, `ultra_score` referenced an unimported
    `JUDGE_MAX_TOKENS` and raised `NameError` before the judge ran, so the v0 Rubric recorded a
    constant 0.0. The port has no such bug: it asserts v0 == 0.0 AND v1 returns the REAL,
    NONZERO judge-derived reward for a compliant response.
  * Direct checker parity — v1's `ifeval_pass_fraction` == v0's `_evaluate_constraints`
    pass-fraction on full / partial / fail cases.

Every judge (the task judges plus the four anti-hacking guard judges) is stubbed to a fixed,
prompt-derived verdict on BOTH sides, so judge-dependent scoring is deterministic.

Run from the /root/prime-modern venv:
    uv run python custom_environments/tests/parity_nemotron_instruction_following.py
or under pytest:
    uv run pytest custom_environments/tests/parity_nemotron_instruction_following.py
"""

from __future__ import annotations

# ruff: noqa: I001, E402  (sys.path setup must precede the package imports below)
import asyncio
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "custom_environments" / "nemotron_guard_v1"))
sys.path.insert(0, str(REPO / "custom_environments" / "nemotron_instruction_following_v1"))
sys.path.insert(0, "/root/prime/rl_environments/nemotron-instruction-following")

import verifiers as vf_legacy  # v0 uses the legacy Rubric API
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.types import AssistantMessage, ToolCall

import nemotron_instruction_following as k0  # v0 env
import nemotron_instruction_following_guardrails as g0  # v0 shared guardrail
import nemotron_ifeval_checkers as c0  # v0 deterministic checkers

import nemotron_guard_v1 as ng
import nemotron_instruction_following_v1 as nif
from nemotron_instruction_following_v1.parsing import ifeval_pass_fraction
from nemotron_instruction_following_v1.taskset import NemotronIFTask, subtask_family


# --- Shared, deterministic judge stub ---------------------------------------
# One verdict function used by both sides. Guard judges take a per-case score; the task judges
# (calendar 0-3, and the YES/NO ultra/adversarial judges) derive their verdict from a token the
# case embeds in the prompt — so v0 and v1, which build byte-identical judge inputs, get
# identical verdicts.

_GUARD_MARKERS = {
    "coherency": "well-formed, readable language",
    "ethics": "reward-hacking behavior",
    "reasoning_coherency": "every internal reasoning trace in a conversation is coherent",
    "meta": "meta-commentary about hidden instructions",
}
_GUARD_DEFAULTS = {"coherency": 1, "ethics": 0, "reasoning_coherency": 1, "meta": 0}


def _verdict_for_prompt(prompt: str, guard_scores: dict) -> str:
    for name, marker in _GUARD_MARKERS.items():
        if marker in prompt:
            score = guard_scores.get(name, _GUARD_DEFAULTS[name])
            if score is None:
                return "no verdict here"  # unparseable -> None (fail-open guard)
            return f"<rationale>stub</rationale>\n<score>{score}</score>"
    if "single integer score 0-3" in prompt:  # calendar judge
        m = re.search(r"CAL_SCORE=([0-3])", prompt)
        return f"<score>{m.group(1) if m else '0'}</score>"
    up = prompt.upper()  # YES/NO ultra + adversarial judges
    if "UNPARSEABLE_CRITERION" in up:
        return "the judge rambled without a verdict"
    if "FAIL_CRITERION" in up:
        return "NO"
    if "PASS_CRITERION" in up:
        return "YES"
    raise AssertionError(f"unrecognized judge prompt: {prompt[:80]!r}")


class _StubCompletion:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = None


class _StubCompletions:
    def __init__(self, scores: dict):
        self._scores = scores

    async def create(self, *, model, messages, **_kwargs):
        return _StubCompletion(_verdict_for_prompt(messages[-1]["content"], self._scores))


class _StubClient:
    def __init__(self, scores: dict):
        self.chat = type("Chat", (), {"completions": _StubCompletions(scores)})()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False


# --- v0 driver --------------------------------------------------------------


def _v0_env(subtask: str, stub):
    fam = subtask_family(subtask)
    if fam == "ifeval":
        return k0._ifeval_env(-1, 1, 42, None)
    if fam == "structured":
        config_name = subtask.replace("structured_v2_", "")
        return k0._structured_v2_env(config_name, -1, 1, 42, None)
    if fam == "format":
        repo, filename = {
            "citation_format": ("repo", "citation.jsonl"),
            "freeform_formatting": ("repo", "freeform.jsonl"),
        }[subtask]
        return k0._format_verification_env(repo, filename, subtask, -1, 1, 42, None)
    if fam == "calendar":
        return k0._calendar_env(stub, "model", None, -1, 1, 42, None)
    if fam == "adversarial":
        return k0._adversarial_env(stub, "model", None, -1, 1, 42, None)
    return k0._ultra_judge_env("repo", "file", subtask, stub, "model", None, -1, 1, 42, None)


def _v0_completion(assistant: list[dict]) -> list[dict]:
    messages = []
    for spec in assistant:
        message = {"role": "assistant", "content": spec.get("content")}
        if spec.get("reasoning_content"):
            message["reasoning_content"] = spec["reasoning_content"]
        if spec.get("tool_calls"):
            message["tool_calls"] = [
                {"id": tc.get("id", "call"), "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in spec["tool_calls"]
            ]
        messages.append(message)
    return messages


async def _v0_score(case: dict) -> dict:
    guard_scores = case.get("judge_scores", {})
    flags = case.get("guard_flags", {})
    enable_judges = flags.get("enable_judges", False)
    stub = _StubClient(guard_scores)

    env = _v0_env(case["subtask"], stub)
    config = g0.AntiHackingConfig(
        judge_client=stub if enable_judges else None,
        judge_model="model",
        enable_judges=enable_judges,
        reasoning_required=flags.get("reasoning_required", False),
        allow_renderer_stripped_tool_calls=flags.get("allow_renderer_stripped_tool_calls", False),
        format_reward_weight=0.0,  # the dead v0 format factor (audit fix #3)
    )
    rubric = g0.GuardedRubric(env.rubric, config)
    state = {
        "prompt": case["prompt"],
        "completion": _v0_completion(case["assistant"]),
        "info": case["v0_info"],
        "is_truncated": case.get("truncated", False),
    }
    await rubric.score_rollout(state)
    metrics = state["metrics"]
    return {"task": metrics["task_reward"], "mult": metrics["anti_hacking_multiplier"], "final": state["reward"]}


# --- v1 driver --------------------------------------------------------------


def _v1_assistant_messages(assistant: list[dict]) -> list[AssistantMessage]:
    messages = []
    for spec in assistant:
        tool_calls = None
        if spec.get("tool_calls"):
            tool_calls = [
                ToolCall(id=tc.get("id", "call"), name=tc["name"], arguments=tc["arguments"])
                for tc in spec["tool_calls"]
            ]
        messages.append(
            AssistantMessage(
                content=spec.get("content"),
                reasoning_content=spec.get("reasoning_content"),
                tool_calls=tool_calls,
            )
        )
    return messages


def _make_trace(task, assistant: list[dict], truncated: bool) -> Trace:
    nodes: list[MessageNode] = []
    parent = None
    for message in _v1_assistant_messages(assistant):
        nodes.append(MessageNode(parent=parent, message=message, sampled=True))
        parent = len(nodes) - 1
    trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data), nodes=nodes)
    if truncated:
        trace.stop_condition = "max_output_tokens"
    return trace


async def _v1_score(case: dict) -> dict:
    flags = case.get("guard_flags", {})
    guard = ng.AntiHackingConfig(
        enable_judges=flags.get("enable_judges", False),
        reasoning_required=flags.get("reasoning_required", False),
        allow_renderer_stripped_tool_calls=flags.get("allow_renderer_stripped_tool_calls", False),
    )
    config = nif.NemotronIFTaskConfig(guard=guard)
    data = nif.NemotronIFData(subtask=case["subtask"], prompt=case["prompt"], **case["v1_fields"])
    task = NemotronIFTask(data, config)
    trace = _make_trace(task, case["assistant"], case.get("truncated", False))

    guard_scores = case.get("judge_scores", {})
    import verifiers.v1.judge as judge_module

    original = judge_module.build_async_openai
    judge_module.build_async_openai = lambda cfg: _StubClient(guard_scores)
    try:
        final = await task.guarded_reward(trace)
    finally:
        judge_module.build_async_openai = original
    return {"task": trace.metrics["task_reward"], "mult": trace.metrics["anti_hacking_multiplier"], "final": final}


# --- Cases ------------------------------------------------------------------

_U = [{"role": "user", "content": "please respond"}]

# ifeval: one keyword-existence constraint the response satisfies.
_IFEVAL_ONE = {"instruction_ids": ["keywords:existence"], "kwargs_list": [{"keywords": ["banana"]}]}
# ifeval: two constraints (existence banana=pass, forbidden banana=fail) -> 0.5.
_IFEVAL_TWO = {
    "instruction_ids": ["keywords:existence", "keywords:forbidden_words"],
    "kwargs_list": [{"keywords": ["banana"]}, {"forbidden_words": ["banana"]}],
}


def _ifeval_v0_info(fields):
    return {"instruction_ids": fields["instruction_ids"], "kwargs_list": fields["kwargs_list"], "subtask": "ifeval"}


def _structured_v0_info(fields):
    return {
        "schema_str": fields.get("schema_str", ""), "schema_type": fields.get("schema_type", "json"),
        "response_mode": fields.get("response_mode", ""), "tool_name": fields.get("tool_name", ""),
        "tool_payload_key": fields.get("tool_payload_key", ""), "subtask": "structured_v2_direct_generation",
    }


_OBJ_SCHEMA = json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})

# calendar exp_cal_state embeds the stub's target score.
_CAL_STATE_3 = json.dumps({"events": "full", "grade": "CAL_SCORE=3"})
_CAL_STATE_1 = json.dumps({"events": "partial", "grade": "CAL_SCORE=1"})

# adversarial / ultra criteria embed PASS_CRITERION / FAIL_CRITERION tokens.
_ADV_PASS = {"criteria": [{"criteria": "The response is helpful. PASS_CRITERION"}]}
_ADV_MIXED = {"criteria": [
    {"criteria": "helpful. PASS_CRITERION"},
    {"criteria": "concise. FAIL_CRITERION"},
]}
_ULTRA_PASS = {"judge_items": [{"content": "Follows the identity. PASS_CRITERION"}]}
_ULTRA_MIXED = {"judge_items": [
    {"content": "a. PASS_CRITERION"},
    {"content": "b. FAIL_CRITERION"},
]}
# ultra with a deterministic instruction component (banana keyword) + one judge item.
_ULTRA_DET = {
    "instructions": [{"instruction_id": "keywords:existence", "keywords": ["banana"]}],
    "judge_items": [{"content": "on-topic. PASS_CRITERION"}],
}


def _adv_v0_info(fields):
    return {
        "user_prompt": "please respond", "reference_response": "", "judge_prompt_template": "",
        "criteria_json": json.dumps(fields["criteria"]), "subtask": "adversarial",
    }


def _ultra_v0_info(fields, subtask="identity"):
    return {
        "instructions_json": json.dumps(fields.get("instructions", [])),
        "judge_items_json": json.dumps(fields.get("judge_items", [])),
        "reference_response": "", "judge_prompt_template": "", "subtask": subtask,
    }


CASES = [
    # -- ifeval deterministic pass-fraction (PARITY) --
    {"name": "ifeval full pass", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>t</think>Here is a banana."}]},
    {"name": "ifeval partial (0.5)", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_TWO, "v0_info": _ifeval_v0_info(_IFEVAL_TWO),
     "assistant": [{"content": "<think>t</think>Here is a banana."}]},
    {"name": "ifeval fail", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>t</think>Here is an apple."}]},

    # -- structured outputs (PARITY) --
    {"name": "structured json valid", "subtask": "structured_v2_direct_generation", "prompt": _U, "kind": "parity",
     "v1_fields": {"schema_str": _OBJ_SCHEMA, "schema_type": "json"},
     "v0_info": _structured_v0_info({"schema_str": _OBJ_SCHEMA, "schema_type": "json"}),
     "assistant": [{"content": '<think>t</think>```json\n{"a": 5}\n```'}]},
    {"name": "structured json invalid", "subtask": "structured_v2_direct_generation", "prompt": _U, "kind": "parity",
     "v1_fields": {"schema_str": _OBJ_SCHEMA, "schema_type": "json"},
     "v0_info": _structured_v0_info({"schema_str": _OBJ_SCHEMA, "schema_type": "json"}),
     "assistant": [{"content": '<think>t</think>```json\n{"a": "not-an-int"}\n```'}]},
    {"name": "structured tool_call valid", "subtask": "structured_v2_tool_calling_extraction", "prompt": _U, "kind": "parity",
     "v1_fields": {"schema_str": _OBJ_SCHEMA, "schema_type": "json", "response_mode": "tool_call", "tool_name": "emit"},
     "v0_info": {"schema_str": _OBJ_SCHEMA, "schema_type": "json", "response_mode": "tool_call",
                 "tool_name": "emit", "tool_payload_key": "", "subtask": "structured_v2_tool_calling_extraction"},
     "assistant": [{"content": "here", "tool_calls": [{"name": "emit", "arguments": '{"a": 7}'}]}]},

    # -- format verification (PARITY) --
    {"name": "format regex pass", "subtask": "citation_format", "prompt": _U, "kind": "parity",
     "v1_fields": {"verifier": {"type": "regex", "verify_regex": [r"\[\d+\]"], "verify_min_matches": 1}},
     "v0_info": {"verifier_json": json.dumps({"type": "regex", "verify_regex": [r"\[\d+\]"], "verify_min_matches": 1}),
                 "subtask": "citation_format"},
     "assistant": [{"content": "<think>t</think>See reference [1] for details."}]},
    {"name": "format regex fail", "subtask": "citation_format", "prompt": _U, "kind": "parity",
     "v1_fields": {"verifier": {"type": "regex", "verify_regex": [r"\[\d+\]"], "verify_min_matches": 1}},
     "v0_info": {"verifier_json": json.dumps({"type": "regex", "verify_regex": [r"\[\d+\]"], "verify_min_matches": 1}),
                 "subtask": "citation_format"},
     "assistant": [{"content": "<think>t</think>No citations here."}]},

    # -- calendar judge (PARITY: v0 judge works) --
    {"name": "calendar score 3/3", "subtask": "calendar", "prompt": _U, "kind": "parity",
     "v1_fields": {"exp_cal_state": _CAL_STATE_3},
     "v0_info": {"exp_cal_state": _CAL_STATE_3, "subtask": "calendar"},
     "assistant": [{"content": '<think>t</think>{"scheduled": true}'}]},
    {"name": "calendar score 1/3", "subtask": "calendar", "prompt": _U, "kind": "parity",
     "v1_fields": {"exp_cal_state": _CAL_STATE_1},
     "v0_info": {"exp_cal_state": _CAL_STATE_1, "subtask": "calendar"},
     "assistant": [{"content": '<think>t</think>{"scheduled": false}'}]},

    # -- adversarial judge (PARITY: v0 judge works) --
    {"name": "adversarial all pass (1.0)", "subtask": "adversarial", "prompt": _U, "kind": "parity",
     "v1_fields": _ADV_PASS, "v0_info": _adv_v0_info(_ADV_PASS),
     "assistant": [{"content": "<think>t</think>A helpful answer."}]},
    {"name": "adversarial mixed (0.5)", "subtask": "adversarial", "prompt": _U, "kind": "parity",
     "v1_fields": _ADV_MIXED, "v0_info": _adv_v0_info(_ADV_MIXED),
     "assistant": [{"content": "<think>t</think>A helpful answer."}]},

    # -- AUDIT FIX #1: ultra-judge subtasks were dead (v0 == 0) --
    {"name": "ultra identity compliant (FIXED nonzero)", "subtask": "identity", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_PASS, "v0_info": _ultra_v0_info(_ULTRA_PASS), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>I am the assistant as configured."}]},
    {"name": "ultra identity mixed (FIXED 0.5)", "subtask": "identity", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_MIXED, "v0_info": _ultra_v0_info(_ULTRA_MIXED), "expect_v1": 0.5,
     "assistant": [{"content": "<think>t</think>Some answer."}]},
    {"name": "ultra identity det+judge (FIXED 1.0)", "subtask": "identity", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_DET, "v0_info": _ultra_v0_info(_ULTRA_DET), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>Here is a banana, on topic."}]},
    # The other four ultra-judge subtasks share the same dead-in-v0 path; confirm each revives.
    {"name": "ultra sysbench compliant (FIXED nonzero)", "subtask": "sysbench", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_PASS, "v0_info": _ultra_v0_info(_ULTRA_PASS, "sysbench"), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>Answer following the system rules."}]},
    {"name": "ultra cfbench compliant (FIXED nonzero)", "subtask": "cfbench", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_PASS, "v0_info": _ultra_v0_info(_ULTRA_PASS, "cfbench"), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>Answer meeting the constraints."}]},
    {"name": "ultra multichallenge compliant (FIXED nonzero)", "subtask": "multichallenge", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_PASS, "v0_info": _ultra_v0_info(_ULTRA_PASS, "multichallenge"), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>Final-turn answer."}]},
    {"name": "ultra inverse_ifeval compliant (FIXED nonzero)", "subtask": "inverse_ifeval", "prompt": _U, "kind": "fixed",
     "v1_fields": _ULTRA_PASS, "v0_info": _ultra_v0_info(_ULTRA_PASS, "inverse_ifeval"), "expect_v1": 1.0,
     "assistant": [{"content": "<think>t</think>Answer."}]},

    # -- guard integration on a scoring ifeval row (task_reward=1) --
    {"name": "guard truncated -> 0", "subtask": "ifeval", "prompt": _U, "kind": "parity", "truncated": True,
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>t</think>Here is a banana."}]},
    {"name": "guard unclosed think -> 0", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>never closes banana"}]},
    {"name": "guard zero visible -> 0", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>only thinking</think>   "}]},
    {"name": "guard missing reasoning (required) -> 0", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "guard_flags": {"reasoning_required": True}, "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "Here is a banana."}]},
    {"name": "guard local meta penalty", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>t</think>Here is a banana, per the system prompt."}]},
    {"name": "guard coherency 0 penalty", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"coherency": 0},
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>reason</think>Here is a banana."}]},
    {"name": "guard all judges pass", "subtask": "ifeval", "prompt": _U, "kind": "parity",
     "guard_flags": {"enable_judges": True}, "judge_scores": {},
     "v1_fields": _IFEVAL_ONE, "v0_info": _ifeval_v0_info(_IFEVAL_ONE),
     "assistant": [{"content": "<think>reason</think>Here is a banana."}]},
]


# --- Direct checker parity (v1 == v0 _evaluate_constraints pass-fraction) ----

CHECKER_CASES = [
    ("full", "Here is a banana.", ["keywords:existence"], [{"keywords": ["banana"]}]),
    ("partial", "Here is a banana.", ["keywords:existence", "keywords:forbidden_words"],
     [{"keywords": ["banana"]}, {"forbidden_words": ["banana"]}]),
    ("fail", "Here is an apple.", ["keywords:existence"], [{"keywords": ["banana"]}]),
    ("multi 2/3", "ALL CAPS TEXT WITH BANANA", ["change_case:english_capital", "keywords:existence", "punctuation:no_comma"],
     [{}, {"keywords": ["banana"]}, {}]),
]


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


async def main() -> int:
    failures = 0

    print("== direct checker parity (v1 ifeval_pass_fraction == v0 _evaluate_constraints) ==")
    for name, text, ids, kwargs_list in CHECKER_CASES:
        v0 = c0._evaluate_constraints(text, ids, kwargs_list)["pass_fraction"]
        v1 = ifeval_pass_fraction(text, ids, kwargs_list)
        ok = _close(v0, v1)
        failures += 0 if ok else 1
        print(f"[{'ok  ' if ok else 'FAIL'}] checker {name:<12} v0={v0:.4f} v1={v1:.4f}")

    print("\n== full-reward parity / audit-fix ==")
    for case in CASES:
        v0 = await _v0_score(case)
        v1 = await _v1_score(case)
        if case["kind"] == "fixed":
            # AUDIT FIX #1: v0 ultra_score is dead (NameError -> 0.0); v1 returns the real reward.
            ok = (
                _close(v0["task"], 0.0)
                and _close(v1["task"], case["expect_v1"])
                and v1["task"] > 0.0
                and _close(v1["final"], case["expect_v1"])  # guard mult is 1.0 here
            )
            tag = "FIXED "
        else:
            ok = _close(v0["task"], v1["task"]) and _close(v0["mult"], v1["mult"]) and _close(v0["final"], v1["final"])
            tag = "PARITY"
        failures += 0 if ok else 1
        print(
            f"[{'ok  ' if ok else 'FAIL'}] {tag} {case['name']:<40} "
            f"task {v0['task']:.4f}/{v1['task']:.4f}  "
            f"mult {v0['mult']:.4f}/{v1['mult']:.4f}  "
            f"final {v0['final']:.4f}/{v1['final']:.4f}"
        )

    total = len(CHECKER_CASES) + len(CASES)
    print(f"\n{total - failures}/{total} checks passed (v0 / v1 shown as pairs).")
    return 1 if failures else 0


def test_parity():
    assert asyncio.run(main()) == 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
