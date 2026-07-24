# Porting our v0 envs → modern verifiers v1

Reference for porting `/root/prime/rl_environments/*` to the v1 API in modern prime-rl.
All code grounded in `/root/prime-modern/deps/verifiers/verifiers/v1/` and the example envs
in `/root/prime-modern/deps/research-environments/environments/`.

## Constraints (non-negotiable)
- Humane, plain code. No hardcodes, no SCREAMING_CONSTANT spam, no cute/opaque function names.
- Every ported env must be **parity-tested** against the v0 env: same inputs → same reward.
- Fix the audited bugs *in the port* (don't reproduce them). See "Audit fixes" below.

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

## THE combine rule (critical)
Rewards are combined by **weighted SUM**: `trace.reward == sum(record_reward(name, value*weight))`.
There is NO built-in product. So a **multiplicative** reward (decensor) must be computed inside
**ONE** `@vf.reward` returning the scalar product; log each factor via `trace.record_metrics`.

## Gate semantics (critical)
A gate that should hard-zero the reward must **return 0.0** from Python. Do NOT `raise` — a raising
judge/grader *fails the rollout for retry*, it does not score 0 (`judge.py:160-165`).

## Judges
`vf.Judge` subclass: set `prompt` (name-substitution, literal braces survive) + `schema=PydanticModel`
(→ `result.parsed`) or override `parse(response)->T`. Config: `class C(vf.JudgeConfig): model=...`.
Two-stage (deterministic then judge fallback): see `math_env_v1`.

## Datasets / split
`Taskset.load()` maps HF/JSONL rows → TaskData. No built-in disjoint split: expose a `dataset_split`
field or slice in `load()`; prime-rl train/eval are separate config blocks (eval capped by `num_examples`).

## Multi-turn / simulated user (decensor)
Subclass `vf.User[UserConfig, StateT].respond(message)->Messages`; share `vf.State`; terminate with a
`@vf.stop` reading `trace.state`. Run under the **`null`** harness (chat, no code exec). If the user sim
needs an LLM it calls the model itself in `respond()` (no built-in LLM-user; see tau2_bench_v1/harness.py).
Turn/token caps on `env.agent`: `max_turns`, `max_output_tokens`.

## Tool-calling
`vf.Toolset[Config,State]` with `@vf.tool` methods; wire on Task via `tools = (MyToolset,)` + a
type-matched config field. Grade by reading `trace.assistant_messages[...].tool_calls` (see bfcl_v3_v1).
`trace.state` is NOT in the serialized trace — persist anything a reward needs into `trace.info`.

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

## Audit fixes to bake into the ports
1. IF: `JUDGE_MAX_TOKENS` was used but not imported → 5 subtasks scored constant 0.0. In the port,
   the judge max-tokens must be a real config value, actually applied.
2. Decensor: "Be verbose" word band [1900,2100] is unreachable → word_multiplier=0 zeros the product on
   ~94% of those rollouts. Floor the length multiplier (e.g. >=0.1) or retarget to an achievable band.
3. nemotron format_reward was dead (weight 0). Either omit it or give it a real, discriminating range.
4. decensor reasoning_coherency judge was dead (always 1.0) — drop it.
5. decide judge fail-open vs fail-closed deliberately and document it.

## v0 reward formulas (parity targets)
- decensor: `final = base(=1-refusal/10 ∈[.1,.9]) × word_mult × coherent × md_pres × md_corr × reasoning × meta`.
- nemotron: `final = anti_hacking_multiplier × task_reward × format_multiplier`,
  `format_multiplier = (1-w) + w*format_reward`; task_reward = mcqa 0/1 | math_verify 0/1 | judge 0/1 | ifeval frac.
