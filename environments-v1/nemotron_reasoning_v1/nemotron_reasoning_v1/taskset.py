"""nemotron-reasoning-v1: the reasoning category taskset.

The v1 port of the v0 `nemotron-reasoning` env. Four subtasks, each wrapped by the shared
nemotron anti-hacking guardrail (`nemotron_guard_v1`):

* ``reasoning_gym`` — nvidia/Nemotron-RL-ReasoningGym-v1, graded by reasoning_gym's own
  per-dataset scorer (normalize-equality fallback). Deterministic.
* ``math`` — nvidia/Nemotron-RL-Math-v2, graded by `math-verify` symbolic equivalence of the
  final answer. Deterministic.
* ``science`` — nvidia/Nemotron-RL-Science-v1, graded by an answer-equivalence LLM judge
  (normalize-equality short-circuit first).
* ``arc_agi`` — ARC-AGI transductive output-grid tasks, graded by exact grid match. Deterministic.

All four subtasks are served by ONE concrete `NemotronReasoningTask` whose single guarded
`@vf.reward` dispatches on `data.subtask` — so a trace rebuilt from the wire (replay uses the
taskset's one `task_type()`) still scores correctly, and the pattern matches the reference
`nemotron_knowledge_v1`. The reward is `final = anti_hacking_multiplier * task_reward`,
multiplied inside a single `@vf.reward` per the v1 weighted-sum combine rule.

Divergence from v0 (intended): v0 optionally exposed a native-Python-tool multi-turn env and a
`python_tool_call_emitted` *metric*. That is a runtime/harness concern (tool execution) and never
entered `task_reward`, so — like the reference — only the graders are ported here; tool execution
is left to the v1 harness config. v0's dead `format_multiplier` (config weight 0) is omitted too.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import verifiers.v1 as vf
from nemotron_guard_v1 import AntiHackingConfig, GuardedTask, SubtaskMix, apply_mix, mix_summary
from nemotron_guard_v1.text import strip_think_tags
from verifiers.v1.types import content_text

from nemotron_reasoning_v1.answers import (
    coerce,
    extract_final_answer,
    grid_reward,
    is_grid,
    math_answers_equivalent,
    normalize_answer,
    reasoning_gym_reward,
)
from nemotron_reasoning_v1.judge import EquivalenceJudge, EquivalenceJudgeConfig

# --- Task data / config ------------------------------------------------------


class NemotronReasoningData(vf.TaskData):
    answer: str
    """Gold answer: the expected final answer (math/science/reasoning_gym) or the ARC output grid
    as a JSON string."""
    subtask: str = "math"
    """Which subtask produced the row (`reasoning_gym` | `math` | `science` | `arc_agi`)."""
    uuid: str = ""
    answer_format: str = ""
    """math/science only: `boxed` or `answer_line` — the output format the prompt requested."""
    source_dataset: str = ""
    """reasoning_gym only: the underlying reasoning_gym dataset whose scorer grades the answer."""
    entry_metadata_json: str = "{}"
    """reasoning_gym only: the row metadata (JSON) passed to the reasoning_gym scorer's `entry`."""


class NemotronReasoningTaskConfig(vf.TaskConfig):
    guard: AntiHackingConfig = AntiHackingConfig()
    equivalence_judge: EquivalenceJudgeConfig = EquivalenceJudgeConfig()
    """science answer-equivalence grader."""
    math_verify_timeout: float = 8.0
    """Per-answer wall-clock budget for the math-verify comparison, in seconds (v0 default)."""
    judge_full_reply_gold_chars: int = 120
    """Golds longer than this are prose, not a one-liner, so the judge sees the whole reply
    rather than the extracted answer line."""
    partial_credit_on_full_reply: float = 0.5
    """Credit when the extracted line failed but the full reply passed — the answer was
    there, just not on the line the format asked for."""


# --- Tasks -------------------------------------------------------------------


def prompt_transcript(data: NemotronReasoningData) -> str:
    """Render the task prompt as a role-labelled transcript (v0's `format_prompt_for_judge`)."""
    parts: list[str] = []
    if data.system_prompt:
        parts.append(f"SYSTEM: {data.system_prompt}")
    prompt = data.prompt
    if isinstance(prompt, str):
        parts.append(f"USER: {prompt}")
    elif isinstance(prompt, list):
        for message in prompt:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            parts.append(f"{str(role or 'user').upper()}: {content_text(content)}")
    return "\n\n".join(parts)


class NemotronReasoningTask(
    GuardedTask, vf.Task[NemotronReasoningData, vf.State, NemotronReasoningTaskConfig]
):
    """One concrete task for all four reasoning subtasks.

    A single class (rather than per-subtask subclasses) so a trace reconstructed from the wire —
    which rebuilds rows as the taskset's one `task_type()` — still scores correctly: the subtask
    lives in `data.subtask`, not in the Python type. This mirrors the reference
    `nemotron_knowledge_v1`.

    The reward is a single `@vf.reward` (the multiplicative guard must live in one reward per the
    v1 weighted-sum combine rule): compute the subtask's raw task reward, then fold in the
    anti-hacking multiplier.
    """

    @vf.reward(weight=1.0)
    async def guarded_reward(self, trace: vf.Trace) -> float:
        task_reward = await self.task_reward(trace)
        return await self.guarded(trace, task_reward)

    async def task_reward(self, trace: vf.Trace) -> float:
        subtask = self.data.subtask
        if subtask == "reasoning_gym":
            return self._reasoning_gym_reward(trace)
        if subtask == "science":
            return await self._science_reward(trace)
        if subtask == "arc_agi":
            return self._arc_agi_reward(trace)
        if subtask != "math":
            raise ValueError(f"unknown reasoning subtask {subtask!r}")
        return await self._math_reward(trace)

    async def _math_reward(self, trace: vf.Trace) -> float:
        reply = strip_think_tags(trace.last_reply)
        candidate = extract_final_answer(reply)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(math_answers_equivalent, candidate, str(self.data.answer), reply),
                timeout=self.config.math_verify_timeout,
            )
        except Exception:
            return 0.0

    def _reasoning_gym_reward(self, trace: vf.Trace) -> float:
        model_answer = extract_final_answer(strip_think_tags(trace.last_reply))
        return reasoning_gym_reward(
            model_answer,
            str(self.data.answer),
            self.data.source_dataset,
            self.data.entry_metadata_json,
            prompt_transcript(self.data),
        )

    def _arc_agi_reward(self, trace: vf.Trace) -> float:
        return grid_reward(strip_think_tags(trace.last_reply), str(self.data.answer))

    async def _science_reward(self, trace: vf.Trace) -> float:
        """Grade a free-form science answer against a prose gold.

        Science golds are paragraphs (median ~200 chars), but the prompt asks for a one-line
        `Answer:` and we used to send only that line to a judge told not to credit partial
        answers — so 12.3% of "wrong" answers were literally contained in the gold. When the
        gold is long the whole reply goes to the judge instead, and a reply that only passes
        on the second look earns partial credit rather than zero.
        """
        reply = strip_think_tags(trace.last_reply)
        candidate = extract_final_answer(reply)
        if normalize_answer(candidate) == normalize_answer(self.data.answer):
            return 1.0

        expected = str(self.data.answer)
        question = prompt_transcript(self.data)[-4000:]
        long_gold = len(expected) > self.config.judge_full_reply_gold_chars
        first = reply if long_gold else candidate

        score = await self._judge_equivalence(trace, question, expected, first)
        if score == 0.0 and not long_gold and first != reply:
            if await self._judge_equivalence(trace, question, expected, reply) == 1.0:
                return self.config.partial_credit_on_full_reply
        return score

    async def _judge_equivalence(
        self, trace: vf.Trace, question: str, expected: str, candidate: str
    ) -> float:
        """Fail-closed on an unparseable verdict; a judge ERROR is recorded so an outage is
        visible instead of looking like a genuine reward collapse."""
        if not candidate:
            return 0.0
        try:
            result = await EquivalenceJudge(self.config.equivalence_judge).evaluate(
                trace=trace, question=question, expected=expected, candidate=candidate
            )
        except Exception:
            trace.record_metric("equivalence_judge_error", 1.0)
            return 0.0
        return float(result.parsed) if result.parsed is not None else 0.0


# --- Dataset loading ---------------------------------------------------------


def _messages_from_input(payload: Any) -> list[dict[str, str]]:
    """Extract chat messages from a dataset row's input/`responses_create_params`."""
    payload = coerce(payload)
    items = payload.get("input") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    messages: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower() or "user"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = item.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if str(content).strip():
            messages.append({"role": role, "content": str(content)})
    return messages


_FINAL_ANSWER_STRIP_PATTERNS = [
    r"Make sure your answer is inside\s*\\boxed\s*\{\s*\}\s*\.?",
    r"The final answer must be placed at the end of your response and enclosed within\s*\\boxed\s*\{\s*\}\s*\.?\s*It is essential to adhere to this format\.?",
    r"Put your final answer in\s*\\boxed\s*\{\s*\}\s*\.?",
    r"Express your answer using\s*\\boxed\s*\{\s*\}\s*\.?",
    r"Provide just the answer inside\s*\\boxed\s*\{\s*\}\s*\.?",
    r"Give the answer in\s*\\boxed\s*\{\s*\}\s*format\.?",
    r"Place your final answer in\s*\\boxed\s*\{\s*\}\s*\.?",
    r"Make sure to use\s*\\boxed\s*\{\s*\}\s*for your answer\.?",
    r"Conclude with\s*\(Answer:\s*X\),\s*where X is the final answer\.?",
    r"The last line of your response should be in the following format:\s*'Answer:\s*\\boxed\s*\{.*?\}'\s*\(e\.g\.\s*'.*?'\)\.?",
    r"[^.\n]*\\boxed\s*\{\s*\}[^.\n]*(?:\.|$)",
]


def _strip_final_answer_format_instruction(content: str) -> str:
    stripped = content
    for pattern in _FINAL_ANSWER_STRIP_PATTERNS:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _rewrite_final_answer_prompt(messages: list[dict[str, str]], boxed: bool) -> list[dict[str, str]]:
    """Normalize the first user turn's final-answer instruction (v0 behaviour)."""
    messages = [dict(message) for message in messages]
    user_idx = next(
        (idx for idx, message in enumerate(messages) if str(message.get("role", "")).lower() == "user"), None
    )
    if user_idx is None:
        return messages
    content = _strip_final_answer_format_instruction(str(messages[user_idx].get("content", "")))
    if boxed:
        instruction = "Put your final answer on the last line as `Answer: \\boxed{X}`."
    else:
        instruction = "Put your final answer on the last line as `Answer: X`."
    messages[user_idx] = {**messages[user_idx], "content": f"{content}\n\n{instruction}".strip()}
    return messages


class NemotronReasoningConfig(vf.TasksetConfig):
    task: NemotronReasoningTaskConfig = NemotronReasoningTaskConfig()
    dataset: str = "all"
    """Which subtasks to load: `reasoning_gym`, `math`, `science`, `arc_agi`, a comma-separated
    subset, or `all`."""
    reasoning_gym_repo: str = "nvidia/Nemotron-RL-ReasoningGym-v1"
    reasoning_gym_file: str = "data/train.jsonl"
    math_repo: str = "nvidia/Nemotron-RL-Math-v2"
    math_file: str = "data/train.jsonl"
    science_repo: str = "nvidia/Nemotron-RL-Science-v1"
    science_file: str = "so_openq.jsonl"
    arc_repo: str = "nvidia/Nemotron-RL-ARC-AGI-v1"
    arc_file: str = "data/transductive/train.jsonl"
    seed: int = 42
    system_prompt: str | None = None
    mix: dict[str, SubtaskMix] = {}
    """Per-subtask blend, keyed by subtask name. Empty means concatenate every row of every
    subtask, which lets raw source size dictate the mix (science outnumbers math 40:1)."""
    mix_total: int | None = None
    """Pool size to split across weighted subtasks; defaults to the rows available."""


_ALIASES = {
    "reasoning-gym": "reasoning_gym",
    "rg": "reasoning_gym",
    "math_v2": "math",
    "math-v2": "math",
    "science_v1": "science",
    "arc": "arc_agi",
    "arc-agi": "arc_agi",
    "arc_agi_transductive": "arc_agi",
}
_VALID_SUBTASKS = ("reasoning_gym", "math", "science", "arc_agi")


class NemotronReasoningTaskset(
    vf.Taskset[NemotronReasoningTask, NemotronReasoningConfig]
):
    def _subtasks(self) -> list[str]:
        choice = (self.config.dataset or "all").strip().lower()
        if choice in ("all", ""):
            return list(_VALID_SUBTASKS)
        out: list[str] = []
        for name in choice.split(","):
            key = _ALIASES.get(name.strip(), name.strip())
            if key not in _VALID_SUBTASKS:
                # Silently loading everything on a typo means `dataset = "sciene"` trains on
                # all four subtasks and nobody finds out.
                raise ValueError(f"unknown reasoning subtask {name.strip()!r}; valid: {_VALID_SUBTASKS}")
            if key not in out:
                out.append(key)
        return out or list(_VALID_SUBTASKS)

    def load(self) -> list[NemotronReasoningTask]:
        by_subtask: dict[str, list[NemotronReasoningTask]] = {}
        for subtask in self._subtasks():
            if subtask == "reasoning_gym":
                rows = self._load_reasoning_gym()
            elif subtask == "math":
                rows = self._load_expected_answer(self.config.math_repo, self.config.math_file, "math")
            elif subtask == "science":
                rows = self._load_expected_answer(self.config.science_repo, self.config.science_file, "science")
            else:
                rows = self._load_arc_agi()
            by_subtask[subtask] = list(rows)

        tasks = apply_mix(by_subtask, self.config.mix, self.config.seed, self.config.mix_total)
        # idx is verifiers' example identity (dedup, grouping, dashboards). Each loader
        # enumerates from 0, so it has to be reassigned across the blend or the subtasks
        # collide on it.
        tasks = [
            NemotronReasoningTask(task.data.model_copy(update={"idx": idx}), self.config.task)
            for idx, task in enumerate(tasks)
        ]
        realized = mix_summary(by_subtask, tasks, lambda task: task.data.subtask)
        print(f"nemotron-reasoning blend: {realized} (total {len(tasks)})", flush=True)
        return tasks

    def _jsonl_rows(self, repo: str, filename: str) -> list[dict[str, Any]]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename, repo_type="dataset")
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        import random

        random.Random(self.config.seed).shuffle(rows)
        return rows

    def _make_task(self, idx: int, prompt: list[dict[str, str]], answer: str, subtask: str, **info: Any) -> NemotronReasoningTask:
        return NemotronReasoningTask(
            NemotronReasoningData(
                idx=idx,
                prompt=prompt,
                system_prompt=self.config.system_prompt,
                answer=answer,
                subtask=subtask,
                uuid=str(info.get("uuid", "")),
                answer_format=str(info.get("answer_format", "")),
                source_dataset=str(info.get("source_dataset", "")),
                entry_metadata_json=str(info.get("entry_metadata_json", "{}")),
            ),
            self.config.task,
        )

    def _load_reasoning_gym(self) -> list[NemotronReasoningTask]:
        tasks: list[NemotronReasoningTask] = []
        for i, row in enumerate(self._jsonl_rows(self.config.reasoning_gym_repo, self.config.reasoning_gym_file)):
            messages = _messages_from_input(row.get("input"))
            if not messages:
                question = row.get("question")
                if question:
                    messages = [{"role": "user", "content": str(question)}]
            answer = str(row.get("answer", "")).strip()
            metadata = coerce(row.get("metadata"))
            if not isinstance(metadata, dict):
                metadata = {}
            if not messages or not answer:
                continue
            tasks.append(
                self._make_task(
                    i, messages, answer, "reasoning_gym",
                    uuid=row.get("uuid", ""),
                    source_dataset=metadata.get("source_dataset", ""),
                    entry_metadata_json=json.dumps(metadata, default=str),
                )
            )
        return tasks

    def _load_expected_answer(self, repo: str, filename: str, subtask: str) -> list[NemotronReasoningTask]:
        tasks: list[NemotronReasoningTask] = []
        for i, row in enumerate(self._jsonl_rows(repo, filename)):
            messages = _messages_from_input(row.get("responses_create_params"))
            if not messages:
                question = row.get("question") or row.get("problem")
                if question:
                    messages = [{"role": "user", "content": str(question)}]
            expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
            if not messages or not expected:
                continue
            boxed = (len(tasks) + self.config.seed) % 2 == 0
            messages = _rewrite_final_answer_prompt(messages, boxed)
            tasks.append(
                self._make_task(
                    i, messages, expected, subtask,
                    uuid=row.get("uuid") or row.get("id") or "",
                    answer_format="boxed" if boxed else "answer_line",
                )
            )
        return tasks

    def _load_arc_agi(self) -> list[NemotronReasoningTask]:
        tasks: list[NemotronReasoningTask] = []
        for i, row in enumerate(self._jsonl_rows(self.config.arc_repo, self.config.arc_file)):
            messages = _messages_from_input(row.get("responses_create_params"))
            expected = row.get("expected_output")
            if not messages or not is_grid(expected):
                continue
            tasks.append(
                self._make_task(
                    i, messages, json.dumps(expected), "arc_agi",
                    uuid=row.get("problem_id", ""),
                )
            )
        return tasks
