# Porting v0 environments to Verifiers v1

Reference for porting `/root/prime/rl_environments/*` to the v1 API in modern prime-rl.
The API reference is `/root/prime-modern/deps/verifiers/verifiers/v1/`; example environments are
in `/root/prime-modern/deps/research-environments/environments/`.

## Requirements
- Use plain code and clear function names. Do not hardcode values or add unnecessary all-caps constants.
- Test every port against the v0 environment: the same inputs must give the same reward.
- Fix the audited bugs in the port. See "Audit fixes" below.

## Core v1 model (verbatim from the API)
- `vf.TaskData` (frozen row: `idx`, `prompt`, + typed gold fields).
- `vf.Task[DataT, StateT, ConfigT]` — behavior: `@vf.reward(weight=...)` methods, `@vf.metric`,
  `@vf.stop`, `tools`/`user` ClassVars. Holds `self.data`, `self.config`.
- `vf.Taskset[TaskT, ConfigT]` — one hook `load() -> Iterable[Task]`.
- `vf.TaskConfig` / `vf.TasksetConfig` — knobs (every field needs a default). TasksetConfig has `id`.
- `vf.Judge[ParsedT, ConfigT]` — LLM judge: `prompt` template, `schema` (structured) or `parse()`.
  `JudgeConfig` holds model/base_url/api_key_var/sampling. Call `await judge.evaluate(trace=..., **fields)`.
- `vf.Trace` — `last_reply` (final assistant text), `transcript` (full branch), `rewards`, `metrics`,
  `info`, `assistant_messages[i].tool_calls`, `record_metrics({...})`, `record_reward(...)`.

## Reward combination
Rewards use a weighted sum: `trace.reward == sum(record_reward(name, value*weight))`.
There is no built-in product. For a multiplicative reward such as decensor, compute and return
the scalar product in one `@vf.reward` method. Log each factor via `trace.record_metrics`.

## Gates that give zero reward
A gate that must give zero reward must return 0.0 from Python. Do not `raise`: a judge/grader
exception fails the rollout for retry instead of scoring 0 (`judge.py:160-165`).

## Judges
`vf.Judge` subclass: set `prompt` (name-substitution, literal braces survive) + `schema=PydanticModel`
(→ `result.parsed`) or override `parse(response)->T`. Config: `class C(vf.JudgeConfig): model=...`.
Two-stage (deterministic then judge fallback): see `math_env_v1`.

## Datasets / split
`Taskset.load()` maps HF/JSONL rows → TaskData. No built-in disjoint split: expose a `dataset_split`
field or slice in `load()`; prime-rl train/eval are separate config blocks (eval capped by `num_examples`).

## Multi-turn / simulated user (decensor)
Subclass `vf.User[UserConfig, StateT].respond(message)->Messages`; share `vf.State`; terminate with a
`@vf.stop` reading `trace.state`. Run under the `null` harness (chat, no code exec). If the user sim
needs an LLM it calls the model itself in `respond()` (no built-in LLM-user; see tau2_bench_v1/harness.py).
Turn/token caps on `env.agent`: `max_turns`, `max_output_tokens`.

## Tool-calling
`vf.Toolset[Config,State]` with `@vf.tool` methods; wire on Task via `tools = (MyToolset,)` + a
type-matched config field. Grade by reading `trace.assistant_messages[...].tool_calls` (see bfcl_v3_v1).
`trace.state` is absent from the serialized trace. Save anything a reward needs in `trace.info`.

## Packaging + config
Hatchling package; `__init__.py` re-exports one Taskset in `__all__` (no entry points).
Config: `env.taskset = { id = "<pkg>", <TasksetConfig fields>, task = { <TaskConfig fields> } }`,
`env.agent.harness = { id = "null", runtime = { type = "subprocess" } }`, `ratio`, `group_size`.

## Example analogues (model each port on these)
- knowledge → `knowledge/hle_v1` (judge), `knowledge/simpleqa_v1` (text-parse judge); mcqa = letter match.
- reasoning → `math_env_v1` (math_verify two-stage), `reasoning/*` (reasoning_gym), science = judge.
- instruction-following → `if/*` (ifeval checkers) + judge subtasks.
- tool-calling → `tool_use/bfcl_v3_v1` (Toolset + @vf.stop + tool-call grading).
- decensor → multi-turn `vf.User` + judge + multiplicative reward (hardest).

## Audit fixes
1. IF: `JUDGE_MAX_TOKENS` was used but not imported → 5 subtasks scored constant 0.0. In the port,
   configure and apply the judge max-tokens value.
2. Decensor: "Be verbose" word band [1900,2100] is unreachable → word_multiplier=0 zeros the product on
   ~94% of those rollouts. Floor the length multiplier (e.g. >=0.1) or retarget to an achievable band.
3. nemotron format_reward had weight 0. Either omit it or give it a range that distinguishes answers.
4. decensor reasoning_coherency judge always returned 1.0; remove it.
5. Choose and document the judge failure policy: fail open or fail closed.

## v0 reward formulas (parity targets)
- decensor: `final = base(=1-refusal/10 ∈[.1,.9]) × word_mult × coherent × md_pres × md_corr × reasoning × meta`.
- nemotron: `final = anti_hacking_multiplier × task_reward × format_multiplier`,
  `format_multiplier = (1-w) + w*format_reward`; task_reward = mcqa 0/1 | math_verify 0/1 | judge 0/1 | ifeval frac.
