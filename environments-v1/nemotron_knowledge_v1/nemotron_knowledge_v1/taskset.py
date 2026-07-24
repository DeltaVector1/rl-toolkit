"""nemotron-knowledge-v1: the knowledge category taskset.

The v1 port of the v0 `nemotron-knowledge` env. Two subtasks, each wrapped by the shared
nemotron anti-hacking guardrail (`nemotron_guard_v1`):

* ``mcqa`` — nvidia/Nemotron-RL-knowledge-mcqa, scored by exact answer-letter match.
* ``qa_abstention`` — nvidia/Nemotron-RL-QA-Abstention-v1, scored by abstention-aware
  answer equivalence (an LLM judge). Yields ~0 usable rows in practice but is ported
  faithfully.

Both subtasks are served by ONE concrete `NemotronKnowledgeTask` whose single guarded
`@vf.reward` dispatches on `data.subtask` — so a trace rebuilt from the wire (replay uses
the taskset's one `task_type()`) still scores correctly, and the pattern scales to the
other multi-subtask nemotron envs. The reward is `final = anti_hacking_multiplier *
task_reward`, multiplied inside a single `@vf.reward` per the v1 weighted-sum combine rule.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers.v1 as vf
from nemotron_guard_v1 import AntiHackingConfig, GuardedTask
from nemotron_guard_v1.text import strip_think_tags
from verifiers.v1.types import content_text

from nemotron_knowledge_v1.answers import (
    ABSTENTION_ANSWERS,
    coerce,
    extract_final_answer,
    mcqa_extract_letter,
    normalize_answer,
)
from nemotron_knowledge_v1.judge import EquivalenceJudge, EquivalenceJudgeConfig

# --- Task data / config ------------------------------------------------------


class NemotronKnowledgeData(vf.TaskData):
    answer: str
    """Gold answer: the expected MCQA letter, or the QA reference answer."""
    subtask: str = "mcqa"
    """Which subtask produced the row (`mcqa` | `qa_abstention`)."""
    uuid: str = ""
    answer_format: str = ""
    """MCQA only: `boxed` or `answer_line` — the output format the prompt requested."""


class NemotronKnowledgeTaskConfig(vf.TaskConfig):
    guard: AntiHackingConfig = AntiHackingConfig()
    equivalence_judge: EquivalenceJudgeConfig = EquivalenceJudgeConfig()
    """QA-abstention answer-equivalence grader."""


# --- Tasks -------------------------------------------------------------------


def prompt_transcript(data: NemotronKnowledgeData) -> str:
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


class NemotronKnowledgeTask(
    GuardedTask, vf.Task[NemotronKnowledgeData, vf.State, NemotronKnowledgeTaskConfig]
):
    """One concrete task for both knowledge subtasks.

    A single class (rather than per-subtask subclasses) so a trace reconstructed from the
    wire — which rebuilds rows as the taskset's one `task_type()` — still scores correctly:
    the subtask lives in `data.subtask`, not in the Python type. This is the reference
    pattern for the other multi-subtask nemotron envs (reasoning, IF, tool-calling).

    The reward is a single `@vf.reward` (the multiplicative guard must live in one reward
    per the v1 weighted-sum combine rule): compute the subtask's raw task reward, then fold
    in the anti-hacking multiplier.
    """

    @vf.reward(weight=1.0)
    async def guarded_reward(self, trace: vf.Trace) -> float:
        task_reward = await self.task_reward(trace)
        return await self.guarded(trace, task_reward)

    async def task_reward(self, trace: vf.Trace) -> float:
        if self.data.subtask == "qa_abstention":
            return await self._qa_abstention_reward(trace)
        return self._mcqa_reward(trace)

    def _mcqa_reward(self, trace: vf.Trace) -> float:
        predicted = mcqa_extract_letter(strip_think_tags(trace.last_reply))
        expected = str(self.data.answer).strip().upper()
        return 1.0 if predicted and predicted == expected else 0.0

    async def _qa_abstention_reward(self, trace: vf.Trace) -> float:
        candidate = extract_final_answer(strip_think_tags(trace.last_reply))
        if normalize_answer(candidate) in ABSTENTION_ANSWERS:
            return 1.0 if normalize_answer(self.data.answer) in ABSTENTION_ANSWERS else 0.0
        if normalize_answer(candidate) == normalize_answer(self.data.answer):
            return 1.0
        # Fail-closed: an unparseable verdict or a judge error scores 0.0 (not equivalent),
        # never credits an unverified answer. This is the correctness grader, distinct from
        # the fail-open anti-hacking guard judges.
        question = prompt_transcript(self.data)[-4000:]
        try:
            result = await EquivalenceJudge(self.config.equivalence_judge).evaluate(
                trace=trace,
                question=question,
                expected=str(self.data.answer),
                candidate=candidate,
            )
        except Exception:
            return 0.0
        return float(result.parsed) if result.parsed is not None else 0.0


# --- Dataset loading ---------------------------------------------------------

_MCQA_INSTRUCTION_RE = re.compile(
    r"\AAnswer the following multiple choice question\..*?(?:\n\s*\n)", re.DOTALL
)


def _messages_from_input(payload: Any) -> list[dict[str, str]]:
    """Extract chat messages from a dataset row's `responses_create_params`."""
    payload = coerce(payload)
    items = payload.get("input") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    messages: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower() or "user"
        content = item.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if str(content).strip():
            messages.append({"role": role, "content": str(content)})
    return messages


def _rewrite_mcqa_prompt(messages: list[dict[str, str]], boxed: bool) -> list[dict[str, str]]:
    """Inject the answer-format instruction into the first user turn (v0 behaviour)."""
    messages = [dict(message) for message in messages]
    user_idx = next(
        (idx for idx, message in enumerate(messages) if message.get("role") == "user"), None
    )
    if user_idx is None:
        return messages
    content = str(messages[user_idx].get("content", ""))
    labels = re.findall(r"(?m)^([A-P])\s*:", content)
    choice_spec = "/".join(dict.fromkeys(labels)) or "A/B/C/D"
    first_choice = labels[0] if labels else "A"
    if boxed:
        instruction = (
            "Answer the following multiple choice question. The last line of your response should be in the "
            f"following format: 'Answer: \\boxed{{{choice_spec}}}' (e.g. 'Answer: \\boxed{{{first_choice}}}')."
        )
    else:
        instruction = (
            "Answer the following multiple choice question. The last line of your response should be in the "
            f"following format: 'Answer: {choice_spec}' (e.g. 'Answer: {first_choice}')."
        )
    # A function replacement, not a string: a plain replacement would interpret the
    # backslash escapes in `instruction` and turn \boxed into a backspace.
    rewritten = _MCQA_INSTRUCTION_RE.sub(lambda _m: instruction + "\n\n", content, count=1)
    if rewritten == content:
        rewritten = f"{instruction}\n\n{content}"
    messages[user_idx] = {**messages[user_idx], "content": rewritten}
    return messages


class NemotronKnowledgeConfig(vf.TasksetConfig):
    task: NemotronKnowledgeTaskConfig = NemotronKnowledgeTaskConfig()
    dataset: str = "all"
    """Which subtasks to load: `mcqa`, `qa_abstention`, or `all`."""
    mcqa_dataset: str = "nvidia/Nemotron-RL-knowledge-mcqa"
    mcqa_split: str = "train"
    qa_repo: str = "nvidia/Nemotron-RL-QA-Abstention-v1"
    qa_file: str = "data/train.jsonl"
    seed: int = 42
    system_prompt: str | None = None


class NemotronKnowledgeTaskset(
    vf.Taskset[NemotronKnowledgeTask, NemotronKnowledgeConfig]
):
    def _subtasks(self) -> list[str]:
        choice = (self.config.dataset or "all").strip().lower()
        if choice in ("all", ""):
            return ["mcqa", "qa_abstention"]
        return [choice] if choice in ("mcqa", "qa_abstention") else ["mcqa", "qa_abstention"]

    def load(self) -> list[NemotronKnowledgeTask]:
        tasks: list[NemotronKnowledgeTask] = []
        for subtask in self._subtasks():
            if subtask == "mcqa":
                tasks.extend(self._load_mcqa())
            else:
                tasks.extend(self._load_qa_abstention())
        return tasks

    def _load_mcqa(self) -> list[NemotronKnowledgeTask]:
        from datasets import load_dataset

        raw = load_dataset(self.config.mcqa_dataset, split=self.config.mcqa_split)
        raw = raw.shuffle(seed=self.config.seed)
        tasks: list[NemotronKnowledgeTask] = []
        for i, row in enumerate(raw):
            messages = _messages_from_input(row.get("responses_create_params"))
            expected = str(row.get("expected_answer", "")).strip()
            if not messages or not expected:
                continue
            boxed = (len(tasks) + self.config.seed) % 2 == 0
            prompt = _rewrite_mcqa_prompt(messages, boxed)
            tasks.append(
                NemotronKnowledgeTask(
                    NemotronKnowledgeData(
                        idx=i,
                        prompt=prompt,
                        system_prompt=self.config.system_prompt,
                        answer=expected,
                        subtask="mcqa",
                        uuid=str(row.get("uuid", "")),
                        answer_format="boxed" if boxed else "answer_line",
                    ),
                    self.config.task,
                )
            )
        return tasks

    def _load_qa_abstention(self) -> list[NemotronKnowledgeTask]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.config.qa_repo, self.config.qa_file, repo_type="dataset")
        tasks: list[NemotronKnowledgeTask] = []
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = _messages_from_input(row.get("responses_create_params"))
                if not messages:
                    question = row.get("question") or row.get("problem")
                    if question:
                        messages = [{"role": "user", "content": str(question)}]
                expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
                if not messages or not expected:
                    continue
                tasks.append(
                    NemotronKnowledgeTask(
                        NemotronKnowledgeData(
                            idx=i,
                            prompt=messages,
                            system_prompt=self.config.system_prompt,
                            answer=expected,
                            subtask="qa_abstention",
                            uuid=str(row.get("uuid") or row.get("id") or ""),
                        ),
                        self.config.task,
                    )
                )
        return tasks
