"""nemotron-instruction-following-v1: the instruction-following category taskset.

The v1 port of the v0 `nemotron-instruction-following` env. It composes the NVIDIA
Nemotron-RL instruction-following sources into ~12 subtasks, each wrapped by the shared
anti-hacking guardrail (`nemotron_guard_v1`). The reward is `final =
anti_hacking_multiplier * task_reward`, folded inside a single `@vf.reward` per the v1
weighted-sum combine rule (the dead v0 format factor is omitted — audit fix #3).

Subtask families and how `task_reward` is computed (dispatched on `data.subtask`):

* `ifeval` — deterministic IFEval/IFBench constraint checkers; reward = pass-fraction.
* `structured_v2_*` (direct_generation / diversified_tasks / tool_calling_extraction) —
  deterministic: parse the emitted payload (json/yaml/toml/csv/xml, or a native tool-call
  argument) and validate it against the row schema; reward 0/1.
* `citation_format` / `freeform_formatting` — deterministic regex/string format verifier; 0/1.
* `calendar` — LLM judge, 0-3 integer score / 3.0.
* `adversarial` — LLM judge, mean of per-criterion YES/NO verdicts.
* `identity` / `sysbench` / `cfbench` / `multichallenge` / `inverse_ifeval` — the ultra-judge
  family: mean of an optional deterministic-instruction score and per-item YES/NO judge verdicts.

ONE concrete `NemotronIFTask` serves every subtask (dispatching on `data.subtask`), so a trace
rebuilt from the wire — which reconstructs rows as the taskset's single `task_type()` — still
scores correctly. This is the reference pattern shared with `nemotron_knowledge_v1`.

AUDIT FIX #1: the five ultra-judge subtasks were dead in v0 (a `NameError` on an unimported
`JUDGE_MAX_TOKENS` raised before the judge ran, silently scoring a constant 0.0). Here they
call the judge and produce a real reward; the judge output budget is a real, applied config
field (`IFJudgeConfig.sampling.max_tokens`). See `judging.py`.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from pydantic import model_validator

import verifiers.v1 as vf
from nemotron_guard_v1 import AntiHackingConfig, GuardedTask
from nemotron_guard_v1.text import strip_think_tags
from verifiers.v1.types import content_text

from nemotron_instruction_following_v1.checkers import CHECKERS
from nemotron_instruction_following_v1.judging import (
    ADVERSARIAL_DEFAULT_TEMPLATE,
    CALENDAR_PROMPT,
    ULTRA_CRITERION_PROMPT,
    CalendarJudge,
    IFJudgeConfig,
    YesNoJudge,
    format_with,
)
from nemotron_instruction_following_v1.parsing import (
    coerce,
    completion_tool_calls,
    deterministic_instruction_score,
    extract_json,
    format_verification_score,
    ifeval_fully_passed,
    ifeval_pass_fraction,
    parse_structured_payload,
    structured_payload_valid,
    tool_payload_candidates,
)

# --- Task data / config ------------------------------------------------------


class NemotronIFData(vf.TaskData):
    subtask: str = "ifeval"
    """Which subtask produced the row; the reward dispatches on this."""
    uuid: str = ""

    # ifeval
    instruction_ids: list[str] = []
    kwargs_list: list[dict[str, Any]] = []

    # structured_v2
    schema_str: str = ""
    schema_type: str = "json"
    response_mode: str = ""
    tool_name: str = ""
    tool_payload_key: str = ""

    # citation_format / freeform_formatting
    verifier: dict[str, Any] = {}

    # calendar
    exp_cal_state: str = ""

    # adversarial + ultra-judge (shared)
    reference_response: str = ""
    judge_prompt_template: str = ""

    # adversarial
    user_prompt: str = ""
    criteria: list[dict[str, Any]] = []

    # ultra-judge
    instructions: list[dict[str, Any]] = []
    judge_items: list[dict[str, Any]] = []

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_dict_entries(cls, data: Any) -> Any:
        """The real datasets carry ``null`` where an instruction has no kwargs (and can null a
        verifier), so the ``dict`` / ``list[dict]`` fields must tolerate None: a null list entry
        becomes ``{}`` (no-constraint), a null dict field becomes ``{}``. Keeps the row loadable
        instead of failing the whole taskset build on a single null."""
        if isinstance(data, dict):
            for key in ("kwargs_list", "criteria", "instructions", "judge_items"):
                value = data.get(key)
                if isinstance(value, list):
                    data[key] = [item if isinstance(item, dict) else {} for item in value]
            if "verifier" in data and not isinstance(data["verifier"], dict):
                data["verifier"] = {}
        return data


class NemotronIFTaskConfig(vf.TaskConfig):
    guard: AntiHackingConfig = AntiHackingConfig()
    judge: IFJudgeConfig = IFJudgeConfig()
    """The calendar / adversarial / ultra-judge grader endpoint."""


# --- Subtask families --------------------------------------------------------


def subtask_family(subtask: str) -> str:
    subtask = (subtask or "").strip()
    if subtask.startswith("structured_v2"):
        return "structured"
    if subtask in {"citation_format", "freeform_formatting"}:
        return "format"
    if subtask in {"ifeval", "calendar", "adversarial"}:
        return subtask
    return "ultra"


# --- Task --------------------------------------------------------------------


def _prompt_message_dicts(prompt: Any) -> list[dict[str, str]]:
    """Normalize `data.prompt` (dicts or pydantic messages) into role/content dicts."""
    if not isinstance(prompt, list):
        return []
    out: list[dict[str, str]] = []
    for message in prompt:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        out.append({"role": str(role or "user"), "content": content_text(content)})
    return out


class NemotronIFTask(GuardedTask, vf.Task[NemotronIFData, vf.State, NemotronIFTaskConfig]):
    """One concrete task for every instruction-following subtask, guarded by the shared
    anti-hacking multiplier. The subtask lives in `data.subtask`, so a wire-reconstructed
    trace scores correctly. The multiplicative guard is folded into a single `@vf.reward`.
    """

    @vf.reward(weight=1.0)
    async def guarded_reward(self, trace: vf.Trace) -> float:
        task_reward = await self.task_reward(trace)
        return await self.guarded(trace, task_reward)

    async def task_reward(self, trace: vf.Trace) -> float:
        family = subtask_family(self.data.subtask)
        if family == "ifeval":
            return self._ifeval_reward(trace)
        if family == "structured":
            return self._structured_reward(trace)
        if family == "format":
            return self._format_reward(trace)
        if family == "calendar":
            return await self._calendar_reward(trace)
        if family == "adversarial":
            return await self._adversarial_reward(trace)
        return await self._ultra_reward(trace)

    # -- deterministic families ------------------------------------------------

    def _ifeval_reward(self, trace: vf.Trace) -> float:
        text = strip_think_tags(trace.last_reply)
        ids = list(self.data.instruction_ids or [])
        kwargs_list = list(self.data.kwargs_list or [])
        trace.record_metric("ifeval_strict", ifeval_fully_passed(text, ids, kwargs_list))
        return ifeval_pass_fraction(text, ids, kwargs_list)

    def _structured_reward(self, trace: vf.Trace) -> float:
        schema_type = self.data.schema_type or "json"
        if str(self.data.response_mode or "").lower() == "tool_call":
            calls = completion_tool_calls(trace.assistant_messages)
            candidates = tool_payload_candidates(calls, self.data.tool_name, self.data.tool_payload_key)
            return max(
                [structured_payload_valid(c, self.data.schema_str, schema_type) for c in candidates] or [0.0]
            )
        text = strip_think_tags(trace.last_reply)
        payload = parse_structured_payload(text, schema_type)
        return structured_payload_valid(payload, self.data.schema_str, schema_type)

    def _format_reward(self, trace: vf.Trace) -> float:
        text = strip_think_tags(trace.last_reply)
        verifier = self.data.verifier if isinstance(self.data.verifier, dict) else {}
        return format_verification_score(text, verifier)

    # -- judge families --------------------------------------------------------

    def _judge_prompt_text(self) -> str:
        """The task prompt as a role-labelled transcript (v0's `format_prompt_for_judge`)."""
        parts: list[str] = []
        if self.data.system_prompt:
            parts.append(f"SYSTEM: {self.data.system_prompt}")
        for message in _prompt_message_dicts(self.data.prompt):
            role = str(message["role"]).upper()
            if role:
                parts.append(f"{role}: {message['content']}")
        return "\n\n".join(parts)

    async def _yes_no(self, trace: vf.Trace, judge_input: str) -> float:
        """Fail-closed YES/NO grade: an error or unparseable verdict scores 0.0 (matching v0)."""
        try:
            result = await YesNoJudge(self.config.judge).evaluate(trace=trace, judge_input=judge_input)
        except Exception:
            return 0.0
        return 1.0 if result.parsed == "YES" else 0.0

    async def _calendar_reward(self, trace: vf.Trace) -> float:
        response = strip_think_tags(trace.last_reply)
        trace.record_metric("calendar_is_json", 1.0 if extract_json(response) is not None else 0.0)
        judge_input = CALENDAR_PROMPT.format(exp_cal_state=self.data.exp_cal_state, response=response)
        try:
            result = await CalendarJudge(self.config.judge).evaluate(trace=trace, judge_input=judge_input)
        except Exception:
            return 0.0
        parsed = result.parsed
        return float(parsed / 3.0) if parsed is not None else 0.0

    async def _adversarial_reward(self, trace: vf.Trace) -> float:
        response = strip_think_tags(trace.last_reply)
        criteria = self.data.criteria or []
        if not criteria:
            return 0.0
        reference = self.data.reference_response
        template = self.data.judge_prompt_template

        async def _check(item: dict[str, Any]) -> float:
            criterion = str(item.get("criteria") or item.get("question") or "").strip()
            if not criterion:
                return 0.0
            judge_input = format_with(
                template,
                ADVERSARIAL_DEFAULT_TEMPLATE,
                prompt=self.data.user_prompt,
                model_response=response,
                standard_response=reference,
                criteria=criterion,
            )
            if "YES" not in judge_input.upper()[-200:]:
                judge_input = judge_input + "\n\nAnswer with just YES or NO."
            return await self._yes_no(trace, judge_input)

        scores = await asyncio.gather(*[_check(c) for c in criteria if isinstance(c, dict)])
        return float(sum(scores) / len(scores)) if scores else 0.0

    async def _ultra_reward(self, trace: vf.Trace) -> float:
        response = strip_think_tags(trace.last_reply)
        instructions = self.data.instructions if isinstance(self.data.instructions, list) else []
        judge_items = self.data.judge_items if isinstance(self.data.judge_items, list) else []
        scores: list[float] = []
        det = deterministic_instruction_score(response, instructions)
        if det is not None:
            scores.append(det)

        prompt_text = self._judge_prompt_text()
        reference = self.data.reference_response
        template = self.data.judge_prompt_template

        async def _check(item: dict[str, Any]) -> float:
            criterion = str(item.get("content") or item.get("criteria") or item.get("question") or "").strip()
            if not criterion:
                return 0.0
            if template:
                try:
                    judge_input = template.format(
                        prompt=prompt_text,
                        model_response=response,
                        standard_response=reference,
                        criteria=criterion,
                    )
                except Exception:
                    judge_input = ULTRA_CRITERION_PROMPT.format(
                        prompt=prompt_text, reference=reference, response=response, criterion=criterion
                    )
            else:
                judge_input = ULTRA_CRITERION_PROMPT.format(
                    prompt=prompt_text, reference=reference, response=response, criterion=criterion
                )
            return await self._yes_no(trace, judge_input)

        if judge_items:
            scores.extend(await asyncio.gather(*[_check(item) for item in judge_items if isinstance(item, dict)]))
        return float(sum(scores) / len(scores)) if scores else 0.0


# --- Dataset loading ---------------------------------------------------------

_ALIASES = {
    "adversarial-if": "adversarial",
    "structured": "structured_v2",
    "structured-outputs": "structured_v2",
    "structured-outputs-v2": "structured_v2",
    "structured-v2-direct": "structured_v2_direct",
    "structured-v2-diversified": "structured_v2_diversified",
    "structured-v2-tool-calling": "structured_v2_tool_calling",
    "citation-formatting": "citation_format",
    "freeform-formatting": "freeform_formatting",
    "identity-following": "identity",
    "multiturn": "multichallenge",
    "multiturn-chat": "multichallenge",
    "multi-challenge": "multichallenge",
    "inverse-if": "inverse_ifeval",
    "inverse-ifeval": "inverse_ifeval",
    "sys-bench": "sysbench",
    "cf-bench": "cfbench",
    "instruction-following": "ifeval",
}

_KEYS = {
    "ifeval", "adversarial", "structured_v2", "structured_v2_direct",
    "structured_v2_diversified", "structured_v2_tool_calling",
    "citation_format", "freeform_formatting",
    "calendar", "identity", "sysbench", "cfbench",
    "multichallenge", "inverse_ifeval",
}

_ALL = [
    "ifeval", "structured_v2_direct", "structured_v2_diversified",
    "citation_format", "freeform_formatting", "calendar",
    "multichallenge", "adversarial",
    "identity", "sysbench", "cfbench", "inverse_ifeval",
]


def _resolve_datasets(dataset: str) -> list[str]:
    if not dataset or dataset == "all":
        return list(_ALL)
    out: list[str] = []
    for d in dataset.split(","):
        key = _ALIASES.get(d.strip(), d.strip())
        if key == "structured_v2":
            key = "structured_v2_diversified"
        if key in _KEYS:
            out.append(key)
    return out or ["ifeval", "structured_v2_direct", "structured_v2_diversified", "calendar", "multichallenge", "adversarial"]


class NemotronIFConfig(vf.TasksetConfig):
    task: NemotronIFTaskConfig = NemotronIFTaskConfig()
    dataset: str = "all"
    """Which subtasks to load: `all`, one selector, or a comma-separated subset."""
    max_examples: int = -1
    """Per-subtask row cap (`-1` loads every row). prime-rl train/eval are separate blocks."""
    seed: int = 42
    system_prompt: str | None = None


def _messages_from_input(payload: Any) -> list[dict[str, str]]:
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


class NemotronIFTaskset(vf.Taskset[NemotronIFTask, NemotronIFConfig]):
    _STRUCTURED_FILES = {
        "structured_v2_direct": ("direct_generation", "direct_generation/train-00000-of-00001.parquet"),
        "structured_v2_diversified": ("diversified_tasks", "diversified_tasks/train-00000-of-00001.parquet"),
        "structured_v2_tool_calling": ("tool_calling_extraction", "tool_calling_extraction/train-00000-of-00001.parquet"),
    }
    _FORMAT_SOURCES = {
        "citation_format": ("nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1", "ds3_citation_train.jsonl"),
        "freeform_formatting": ("nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1", "ds2_freeform_train.jsonl"),
    }
    _ULTRA_SOURCES = {
        "identity": ("nvidia/Nemotron-RL-Identity-Following-v1", "train.jsonl"),
        "sysbench": ("nvidia/Nemotron-RL-SysBench-v1", "data/train.jsonl"),
        "cfbench": ("nvidia/Nemotron-RL-CFBench-v1", "data/train.jsonl"),
        "multichallenge": ("nvidia/Nemotron-RL-Multichallenge-v1", "data/advanced.jsonl"),
        "inverse_ifeval": ("nvidia/Nemotron-RL-InverseIFEval-v1", "data/train.jsonl"),
    }

    def load(self) -> list[NemotronIFTask]:
        tasks: list[NemotronIFTask] = []
        for key in _resolve_datasets(self.config.dataset):
            tasks.extend(self._load_subtask(key))
        return tasks

    # -- helpers ---------------------------------------------------------------

    def _make(self, idx: int, prompt: list[dict[str, str]], **fields: Any) -> NemotronIFTask:
        return NemotronIFTask(
            NemotronIFData(idx=idx, prompt=prompt, system_prompt=self.config.system_prompt, **fields),
            self.config.task,
        )

    def _load_jsonl(self, repo: str, filename: str) -> list[dict[str, Any]]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename, repo_type="dataset")
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rng = random.Random(self.config.seed)
        rng.shuffle(rows)
        if self.config.max_examples >= 0:
            rows = rows[: self.config.max_examples]
        return rows

    # -- subtask dispatch ------------------------------------------------------

    def _load_subtask(self, key: str) -> list[NemotronIFTask]:
        if key == "ifeval":
            return self._load_ifeval()
        if key in self._STRUCTURED_FILES:
            return self._load_structured(key)
        if key in self._FORMAT_SOURCES:
            return self._load_format(key)
        if key == "calendar":
            return self._load_calendar()
        if key == "adversarial":
            return self._load_adversarial()
        if key in self._ULTRA_SOURCES:
            return self._load_ultra(key)
        return []

    def _load_ifeval(self) -> list[NemotronIFTask]:
        raw = self._load_jsonl("nvidia/Nemotron-RL-instruction_following", "instruction_following.jsonl")
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(raw):
            prompt = _messages_from_input(row.get("responses_create_params"))
            if not prompt and row.get("prompt"):
                prompt = [{"role": "user", "content": str(row["prompt"])}]
            ids = coerce(row.get("instruction_id_list")) or []
            kwargs_list = coerce(row.get("kwargs")) or []
            if not prompt or not isinstance(ids, list) or not isinstance(kwargs_list, list):
                continue
            # A constraint with no checker always counts as failed, making the row unwinnable;
            # drop rows we cannot actually grade rather than train on constant-zero reward.
            if not ids or any(str(cid) not in CHECKERS for cid in ids):
                continue
            tasks.append(self._make(
                i, prompt, subtask="ifeval", uuid=str(row.get("id", "")),
                instruction_ids=[str(c) for c in ids], kwargs_list=list(kwargs_list),
            ))
        return tasks

    def _load_structured(self, key: str) -> list[NemotronIFTask]:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        config_name, filename = self._STRUCTURED_FILES[key]
        subtask = f"structured_v2_{config_name}"
        path = hf_hub_download(
            "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", filename, repo_type="dataset"
        )
        rows = pq.read_table(path).to_pylist()
        rng = random.Random(self.config.seed)
        rng.shuffle(rows)
        if self.config.max_examples >= 0:
            rows = rows[: self.config.max_examples]
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(rows):
            prompt = _messages_from_input(row.get("responses_create_params"))
            schema_str = str(row.get("schema_str", "")).strip()
            schema_type = str(row.get("schema_type", "json")).strip().lower()
            if not prompt:
                continue
            # Without a parseable schema there is nothing to grade against (XML is graded on
            # well-formedness alone and is exempt).
            from nemotron_instruction_following_v1.parsing import schema_from_str

            if schema_type != "xml" and not schema_from_str(schema_str):
                continue
            tasks.append(self._make(
                i, prompt, subtask=subtask, uuid=str(row.get("source_record_id", "")),
                schema_str=schema_str, schema_type=schema_type,
                response_mode=str(row.get("response_mode", "")),
                tool_name=str(row.get("tool_name", "")),
                tool_payload_key=str(row.get("tool_payload_key", "")),
            ))
        return tasks

    def _load_format(self, key: str) -> list[NemotronIFTask]:
        repo, filename = self._FORMAT_SOURCES[key]
        raw = self._load_jsonl(repo, filename)
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(raw):
            prompt = _messages_from_input(row.get("responses_create_params"))
            verifier = coerce(row.get("verifier"))
            if not prompt or not isinstance(verifier, dict):
                continue
            tasks.append(self._make(i, prompt, subtask=key, verifier=verifier))
        return tasks

    def _load_calendar(self) -> list[NemotronIFTask]:
        raw = self._load_jsonl("nvidia/Nemotron-RL-Instruction-Following-Calendar-v2", "train.jsonl")
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(raw):
            prompt = _messages_from_input(row.get("responses_create_params"))
            exp_state = row.get("exp_cal_state")
            if not prompt or exp_state is None:
                continue
            tasks.append(self._make(
                i, prompt, subtask="calendar", exp_cal_state=json.dumps(exp_state, default=str),
            ))
        return tasks

    def _load_adversarial(self) -> list[NemotronIFTask]:
        from datasets import load_dataset

        raw = load_dataset("nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1", split="train")
        raw = raw.shuffle(seed=self.config.seed)
        if self.config.max_examples >= 0:
            raw = raw.select(range(min(self.config.max_examples, len(raw))))
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(raw):
            prompt = _messages_from_input(row.get("responses_create_params"))
            if not prompt and row.get("prompt"):
                prompt = [{"role": "user", "content": str(row["prompt"])}]
            rubric_items = coerce(row.get("rubric")) or []
            if not prompt or not isinstance(rubric_items, list) or not rubric_items:
                continue
            tasks.append(self._make(
                i, prompt, subtask="adversarial", uuid=str(row.get("uuid", "")),
                user_prompt=str(row.get("prompt", "")),
                reference_response=str(row.get("reference_response", "")),
                judge_prompt_template=str(row.get("judge_prompt_template", "")),
                criteria=[x for x in rubric_items if isinstance(x, dict)],
            ))
        return tasks

    def _load_ultra(self, key: str) -> list[NemotronIFTask]:
        repo, filename = self._ULTRA_SOURCES[key]
        raw = self._load_jsonl(repo, filename)
        tasks: list[NemotronIFTask] = []
        for i, row in enumerate(raw):
            prompt = _messages_from_input(row.get("responses_create_params"))
            if not prompt:
                prompt = _messages_from_input(row.get("messages"))
            if not prompt:
                continue
            judge_items: list[dict[str, Any]] = []
            llm_judge = coerce(row.get("llm_judge")) or []
            rubric = coerce(row.get("rubric")) or []
            if isinstance(llm_judge, list):
                judge_items.extend([x for x in llm_judge if isinstance(x, dict)])
            if isinstance(rubric, list):
                judge_items.extend([x for x in rubric if isinstance(x, dict)])
            principle = str(row.get("principle", "")).strip()
            if principle:
                judge_items.append({"content": principle})
            instructions = coerce(row.get("instructions")) or []
            tasks.append(self._make(
                i, prompt, subtask=key, uuid=str(row.get("id") or row.get("uuid") or ""),
                instructions=[x for x in instructions if isinstance(x, dict)] if isinstance(instructions, list) else [],
                judge_items=judge_items,
                reference_response=str(row.get("reference_response", "")),
                judge_prompt_template=str(row.get("judge_prompt_template", "")),
            ))
        return tasks
