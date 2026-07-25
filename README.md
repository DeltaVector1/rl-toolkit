# Trinity RL Kit

Our RL environments, configs, patches, and skills for training Trinity-Mini on
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). This repo is **not** a
fork — it holds only our stuff, so we can pull upstream prime-rl updates freely and
drop this on top.

## Layout

| Folder | What's in it |
|---|---|
| `environments-v1/` | **Current.** The verifiers **v1** ports — `decensor_v1`, `nemotron_guard_v1` (shared guard + subtask blending), and the four `nemotron-*_v1` envs, plus `tests/` (reward parity suites). Each has its own `pyproject.toml`. |
| `environments/` | **Legacy v0.** `decensor-env`, `decensor-env-multiturn`, the four `nemotron-*` envs, and bench envs (`mmlu`, `arcee-ifeval`). Kept as the parity reference the v1 suites grade against. |
| `docs/` | `PORTING_GUIDE.md` (v0 → v1 API notes and the reward formulas) and `AUDIT_2026-07-24.md` (plain-English audit of every reward bug found against a real 50-step run, and what was changed). |
| `patches/submodules/` | Patches against prime-rl's `deps/` submodules — Trinity tokenizer (`renderers`) and multiturn error handling (`verifiers`). Both load-bearing for Trinity. |
| `patches/prime-rl/` | Our load-bearing edits to prime-rl `src/`, as a git patch. |
| `configs/` | Working RL configs. `trinity_mini_modern/` is the current one (v1 envs, `arcee-ai/Trinity-Mini`); the others are v0-era. |
| `skills/` | Our edited Claude Code skills (`configs`, `training/*`). |
| `other/` | Helper tooling — `rollout_viewer.py`, monitoring/GPU scripts, and the archived OPD distillation experiment. |

## Pinned base

These patches are cut against a specific prime-rl commit. Apply them onto that
commit (or expect to resolve conflicts on a newer one):

```
prime-rl            38494ccb   (v0.6.0)
deps/pydantic-config 896ade4e
deps/renderers       19332933
deps/research-environments 841f9a54
deps/verifiers       78c1b3c5
```

## Bootstrap

```bash
# 1. clone prime-rl at the pinned base
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl && git checkout 38494ccb && git submodule update --init --recursive

# 2. clone this kit next to it
git clone https://github.com/<you>/trinity-rl-kit.git ../trinity-rl-kit
KIT=../trinity-rl-kit

# 3. apply submodule patches (Trinity tokenizer + verifiers multiturn fix)
git -C deps/renderers apply "$KIT"/patches/submodules/renderers-trinity-tokenizer.patch
git -C deps/verifiers apply "$KIT"/patches/submodules/verifiers-trinity-cleanblend.patch

# 4. apply our prime-rl src patch
git apply "$KIT"/patches/prime-rl/*.patch

# 5. drop in envs, configs, skills
cp -r "$KIT"/environments/* rl_environments/
cp -r "$KIT"/configs/*      configs/
cp -r "$KIT"/skills/*       skills/

# 6. install
uv sync --all-extras
```

### Grabbing just one env

```bash
git clone --filter=blob:none --sparse https://github.com/<you>/trinity-rl-kit.git
cd trinity-rl-kit && git sparse-checkout set environments/decensor-env
```

## Trinity tool calling — read this before changing the renderer

Trinity's tokenizer has **no `<tool_call>` special token** (it encodes as four ordinary
tokens). Every tool parser shipped in `renderers` locates `<tool_call>` by token id, so all
of them return "no tool calls" for every completion, silently. Confirmed on both
`arcee-ai/Trinity-Mini` and the NewEden fine-tune.

`patches/submodules/renderers-hermes-tool-parser.patch` adds a text-level `"hermes"` parser
to the `renderers` package — matching the `--tool-call-parser hermes` on Trinity's own model
card — and the config selects it with `[orchestrator.renderer] tool_parser = "hermes"`.
`<think>` **is** a real token, so `reasoning_parser = "think"` is fine as-is.

It has to live in the `renderers` package itself, not in prime-rl: every env-server
subprocess builds its own renderer and imports `renderers` without importing `prime_rl`.
Registering from prime-rl leaves those processes with a stock registry, and every rollout
fails with `Unknown tool_parser 'hermes'`.

## Subtask blending

`ratio` balances envs against each other and does not reach inside an env. A taskset that
loads several subtasks concatenates them by default, so the blend follows raw source file
size — `nemotron-reasoning` ran 83% science / 2% math that way. Use `taskset.mix` (see
`nemotron_guard_v1/mixing.py` and the `configs/trinity_mini_modern/rl.toml` blocks):
`weight`, `max_examples`, `repeat`, `enabled` per subtask, with small sources cycled rather
than starved.

## Notes

- Envs pull their datasets from Hugging Face at runtime — no data is vendored here.
- The v1 envs depend on `verifiable-instructions` (pinned `f46a5ac`) for the IFEval checkers —
  the same package NVIDIA's own instruction-following server uses. Don't re-hand-write them.
- Reward parity suites run under pytest (`python_files = ["test_*.py", "parity_*.py"]`). They
  assert v1 == v0 *except* where a case declares `expect_v1`, which is how an intentional bug
  fix is recorded instead of showing up as a failure.
- `patches/prime-rl/` is the piece that rots when upstream moves; regenerate it
  against a newer base with `git diff <base> -- src/ > patches/prime-rl/prime-rl.patch`.
