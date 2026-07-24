"""Parity test: v1 nemotron-reasoning reward == v0 nemotron-reasoning reward.

For a set of canned (prompt, completion, gold-answer) cases covering every subtask
(reasoning_gym / math / science / arc_agi) plus the anti-hacking guard gates, this computes the
reward BOTH ways and asserts the task_reward, the anti_hacking_multiplier, and the final reward
match:

  (a) v0: import the v0 modules from /root/prime and drive `GuardedRubric.score_rollout` over a
      state dict (base reward funcs mirror the v0 closures exactly, calling v0's own module-level
      graders; `format_reward_weight` pinned to 0, matching the dead format factor — audit fix #3).
  (b) v1: build the ported `NemotronReasoningTask` (one class, dispatching on `data.subtask`),
      construct a `vf.Trace`, and call the task's `@vf.reward`.

Deterministic graders (math-verify, reasoning_gym) run for real and identically on both sides.
The LLM judges (the four anti-hacking guard judges + the science equivalence judge) are stubbed to
a fixed, prompt-derived verdict on BOTH sides, so judge-dependent scoring is deterministic.

Run from the /root/prime-modern venv:
    uv run python custom_environments/tests/parity_nemotron_reasoning.py
"""

from __future__ import annotations

# ruff: noqa: I001  (sys.path setup must precede the package imports below)
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "custom_environments" / "nemotron_guard_v1"))
sys.path.insert(0, str(REPO / "custom_environments" / "nemotron_reasoning_v1"))
sys.path.insert(0, "/root/prime/rl_environments/nemotron-reasoning")

import verifiers as vf_legacy  # noqa: E402  (v0 uses the legacy Rubric API)
from verifiers.v1.graph import MessageNode  # noqa: E402
from verifiers.v1.trace import Trace, TraceTask  # noqa: E402
from verifiers.v1.types import AssistantMessage, ToolCall  # noqa: E402

import nemotron_reasoning as r0  # noqa: E402  (v0 env)
import nemotron_reasoning_guardrails as g0  # noqa: E402  (v0 shared guardrail)

import nemotron_guard_v1 as ng  # noqa: E402
import nemotron_reasoning_v1 as nr  # noqa: E402
from nemotron_reasoning_v1.taskset import NemotronReasoningTask  # noqa: E402


# --- Shared, deterministic judge stub ---------------------------------------

_JUDGE_MARKERS = {
    "coherency": "well-formed, readable language",
    "ethics": "reward-hacking behavior",
    "reasoning_coherency": "every internal reasoning trace in a conversation is coherent",
    "meta": "meta-commentary about hidden instructions",
    "equivalence": "grading final-answer equivalence",
}

# No-penalty defaults (coherency/reasoning-coherency pass at 1, ethics/meta pass at 0).
_DEFAULT_SCORES = {"coherency": 1, "ethics": 0, "reasoning_coherency": 1, "meta": 0, "equivalence": 1}


def _verdict_for_prompt(prompt: str, scores: dict) -> str:
    for name, marker in _JUDGE_MARKERS.items():
        if marker in prompt:
            score = scores.get(name, _DEFAULT_SCORES[name])
            if score is None:
                return "no verdict here"  # unparseable -> None (fail-open / fail-closed)
            return f"<rationale>stub</rationale>\n<score>{score}</score>"
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


def _v0_base_rubric(subtask: str, scores: dict) -> vf_legacy.Rubric:
    if subtask == "math":
        async def reward(completion, answer, **_kw):
            candidate = g0.extract_final_answer(g0.extract_completion_text(completion))
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(r0._math_answers_equivalent, candidate, str(answer)),
                    timeout=r0.MATH_VERIFY_TIMEOUT,
                )
            except Exception:
                return 0.0
    elif subtask == "reasoning_gym":
        async def reward(completion, answer, info=None, **_kw):
            return await r0.reasoning_gym_score(completion, answer, info or {})
    elif subtask == "arc_agi":
        async def reward(completion, answer, **_kw):
            return await r0.arc_grid_score(completion, answer)
    elif subtask == "science":
        async def reward(prompt, completion, answer, state=None, **_kw):
            candidate = g0.extract_final_answer(g0.extract_completion_text(completion))
            return await g0.judge_equivalence(
                _StubClient(scores), "model", None,
                g0.format_prompt_for_judge(prompt), str(answer), candidate, state, "sci",
            )
    else:
        raise AssertionError(subtask)
    return vf_legacy.Rubric(funcs=[reward])


async def _v0_score(case: dict) -> dict:
    scores = case.get("judge_scores", {})
    flags = case.get("guard_flags", {})
    enable_judges = flags.get("enable_judges", False)
    stub = _StubClient(scores) if enable_judges else None

    base = _v0_base_rubric(case["subtask"], scores)
    config = g0.AntiHackingConfig(
        judge_client=stub,
        judge_model="model",
        enable_judges=enable_judges,
        reasoning_required=flags.get("reasoning_required", False),
        allow_renderer_stripped_tool_calls=flags.get("allow_renderer_stripped_tool_calls", False),
        format_reward_weight=0.0,  # the dead v0 format factor (audit fix #3)
    )
    rubric = g0.GuardedRubric(base, config)
    state = {
        "prompt": case["prompt"],
        "completion": _v0_completion(case["assistant"]),
        "answer": case["answer"],
        "info": case.get("info", {}),
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
    config = nr.NemotronReasoningTaskConfig(guard=guard)
    info = case.get("info", {})
    data = nr.NemotronReasoningData(
        prompt=case["prompt"],
        answer=case["answer"],
        subtask=case["subtask"],
        source_dataset=str(info.get("source_dataset", "")),
        entry_metadata_json=str(info.get("entry_metadata_json", "{}")),
    )
    task = NemotronReasoningTask(data, config)
    trace = _make_trace(task, case["assistant"], case.get("truncated", False))

    scores = case.get("judge_scores", {})
    import verifiers.v1.judge as judge_module

    original = judge_module.build_async_openai
    judge_module.build_async_openai = lambda cfg: _StubClient(scores)
    try:
        final = await task.guarded_reward(trace)
    finally:
        judge_module.build_async_openai = original

    return {"task": trace.metrics["task_reward"], "mult": trace.metrics["anti_hacking_multiplier"], "final": final}


# --- Cases ------------------------------------------------------------------

_MATH_PROMPT = [{"role": "user", "content": "Compute 1/2. Put your final answer on the last line as `Answer: \\boxed{X}`."}]
_RG_PROMPT = [{"role": "user", "content": "What is 40 + 2?"}]
_SCI_PROMPT = [{"role": "user", "content": "What is the capital of France?"}]
_ARC_PROMPT = [{"role": "user", "content": "Produce the output grid."}]

_RG_INFO = {"source_dataset": "chain_sum", "entry_metadata_json": "{}"}
_RG_FALLBACK_INFO = {"source_dataset": "", "entry_metadata_json": "{}"}

CASES = [
    # math (math-verify, deterministic)
    {"name": "math correct (boxed)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "math incorrect", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/3}"}]},
    {"name": "math equivalent form (0.5==1/2)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "0.5",
     "assistant": [{"content": "<think>t</think>Answer: \\boxed{1/2}"}]},
    {"name": "math answer-line (no box)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "42",
     "assistant": [{"content": "<think>t</think>Answer: 42"}]},

    # reasoning_gym (real scorer + fallback, deterministic)
    {"name": "rg correct (chain_sum scorer)", "subtask": "reasoning_gym", "prompt": _RG_PROMPT, "answer": "42",
     "info": _RG_INFO, "assistant": [{"content": "<think>t</think>Answer: 42"}]},
    {"name": "rg incorrect (chain_sum scorer)", "subtask": "reasoning_gym", "prompt": _RG_PROMPT, "answer": "42",
     "info": _RG_INFO, "assistant": [{"content": "<think>t</think>Answer: 7"}]},
    {"name": "rg fallback normalize match", "subtask": "reasoning_gym", "prompt": _RG_PROMPT, "answer": "hello world",
     "info": _RG_FALLBACK_INFO, "assistant": [{"content": "<think>t</think>Answer: Hello World"}]},

    # science (equivalence judge, stubbed)
    {"name": "science normalize match (no judge)", "subtask": "science", "prompt": _SCI_PROMPT, "answer": "Paris",
     "assistant": [{"content": "<think>t</think>Answer: paris"}]},
    {"name": "science judge equivalent", "subtask": "science", "prompt": _SCI_PROMPT, "answer": "Paris",
     "judge_scores": {"equivalence": 1}, "assistant": [{"content": "<think>t</think>Answer: The City of Light"}]},
    {"name": "science judge not equivalent", "subtask": "science", "prompt": _SCI_PROMPT, "answer": "Paris",
     "judge_scores": {"equivalence": 0}, "assistant": [{"content": "<think>t</think>Answer: London"}]},
    {"name": "science judge unparseable (fail-closed 0)", "subtask": "science", "prompt": _SCI_PROMPT, "answer": "Paris",
     "judge_scores": {"equivalence": None}, "assistant": [{"content": "<think>t</think>Answer: London"}]},

    # arc_agi (grid exact-match, deterministic)
    {"name": "arc correct grid", "subtask": "arc_agi", "prompt": _ARC_PROMPT, "answer": "[[1, 2], [3, 4]]",
     "assistant": [{"content": "<think>t</think>[[1, 2], [3, 4]]"}]},
    {"name": "arc incorrect grid", "subtask": "arc_agi", "prompt": _ARC_PROMPT, "answer": "[[1, 2], [3, 4]]",
     "assistant": [{"content": "<think>t</think>[[1, 2], [3, 5]]"}]},

    # Hard gates (on a correct math rollout so task_reward is unambiguously 1.0)
    {"name": "math truncated", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2", "truncated": True,
     "assistant": [{"content": "<think>t</think>Answer: \\boxed{1/2}"}]},
    {"name": "math unclosed think", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "assistant": [{"content": "<think>never closes Answer 1/2"}]},
    {"name": "math zero visible", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "assistant": [{"content": "<think>only thinking</think>   "}]},

    # Local meta penalty
    {"name": "math local meta", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "assistant": [{"content": "<think>t</think>Answer: \\boxed{1/2} per the system prompt rubric"}]},

    # Missing-reasoning gate
    {"name": "math missing reasoning (required)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"reasoning_required": True}, "assistant": [{"content": "Answer: \\boxed{1/2}"}]},
    {"name": "math reasoning present (required)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"reasoning_required": True},
     "assistant": [{"content": "Answer: \\boxed{1/2}", "reasoning_content": "reason about the answer"}]},
    {"name": "math renderer-stripped tool call (required, allowed)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"reasoning_required": True, "allow_renderer_stripped_tool_calls": True},
     "assistant": [{"content": None, "tool_calls": [{"name": "calc", "arguments": "{}"}]}]},

    # Anti-hacking guard judges (stubbed)
    {"name": "judge coherency 0", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"coherency": 0},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "judge ethics 1", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"ethics": 1},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "judge reasoning_coherency 0", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"reasoning_coherency": 0},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "judge meta 1", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"meta": 1},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "judge all pass", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},
    {"name": "judge coherency unparseable (fail-open)", "subtask": "math", "prompt": _MATH_PROMPT, "answer": "1/2",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"coherency": None},
     "assistant": [{"content": "<think>reason</think>Answer: \\boxed{1/2}"}]},

    # Guard judges on a judged science rollout (two LLM judge families at once)
    {"name": "science judge equivalent + guards", "subtask": "science", "prompt": _SCI_PROMPT, "answer": "Paris",
     "guard_flags": {"enable_judges": True}, "judge_scores": {"equivalence": 1, "coherency": 1},
     "assistant": [{"content": "<think>t</think>Answer: The City of Light"}]},
]


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


async def main() -> int:
    failures = 0
    for case in CASES:
        v0 = await _v0_score(case)
        v1 = await _v1_score(case)
        ok = _close(v0["task"], v1["task"]) and _close(v0["mult"], v1["mult"]) and _close(v0["final"], v1["final"])
        status = "ok  " if ok else "FAIL"
        print(
            f"[{status}] {case['name']:<48} "
            f"task {v0['task']:.4f}/{v1['task']:.4f}  "
            f"mult {v0['mult']:.4f}/{v1['mult']:.4f}  "
            f"final {v0['final']:.4f}/{v1['final']:.4f}"
        )
        if not ok:
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases match (v0 / v1 shown as pairs).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
