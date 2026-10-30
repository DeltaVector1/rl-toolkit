# RL Toolkit

This repository holds RL environments, training configurations, evaluation code,
training tools, and related PRIME-RL changes.

## Layout

- `environments/` holds the active projects. It includes the experimental
  `decensor`, `decensor-experimental`, `general-agent`, and `ornith`
  projects; five Nemotron v1 projects; and the retained `arcee-ifeval` and
  `mmlu` projects.
- `environments/pyproject.toml` defines the workspace for the five Nemotron
  v1 projects. Run their workspace commands from `environments/`.
- `environments/xiaomi-envs/` holds the complete Xiaomi environment repository
  at the recorded source commit. Run its workspace commands from that directory.
- `environments/PORTING_GUIDE.md` preserves the v1 porting guide.
- `environments/tests/` keeps four Nemotron v1/v0 parity scripts. Run a script
  with `--legacy-root` pointing to the old Nemotron projects. Run the scripts
  with `uv run --project environments --all-packages python
  environments/tests/parity_nemotron_knowledge.py --legacy-root <old-projects>`.
  Pytest reports `PARITY NOT RUN` when that baseline is not available.
- `configs/` holds 42 TOML files, including `nodes.toml`. The three main branch `trinity_mini_*`
  configurations remain as historical examples. The two `*_rl` files refer to
  retired v0 IDs. `trinity_mini_modern` refers to the removed `decensor-v1`
  project. Update these IDs before you run those configurations.
- `evals/refusals-classification/` holds the refusal classifier evaluator.
- `training_ops/` holds the local training status and runtime tools.
- `agents/` holds repository instructions, training expectations, and the
  experiment notes. Codex skills are under `.agents/skills/`.
- `docs/`, `other/`, and the existing `patches/` files retain the main
  branch's guides, tools, and historical patches.

## PRIME-RL files

The root `prime-rl/` directory is a 79-file overlay from experimental commit
`67e53862aca11fd65de80b4694f44c398ee922b1`. The Dockerfile copies these files
into the `prime-rl:glm52-decensor` image. The source tree does not record the
upstream commit used to build that image.

The two files under `patches/prime-rl/` are separate historical patch sets:

- `prime-rl.patch` is the main branch patch, documented against PRIME-RL
  commit `38494ccb`.
- `glm52-decensor.patch` is the experimental branch patch. Its exact upstream
  base is not recorded.

Do not apply both patches together. Do not use a wildcard to select a patch.
Check the target tree before you apply either file.

Read `agents/AGENTS.md` before repository work. Read
`agents/TRAINING_EXPECTATIONS.md` before a live training test or run.

## Historical Trinity-Mini notes

These pins and renderer notes describe the older Trinity-Mini setup. They do
not identify the upstream base for the GLM-5.2 Docker image.

The legacy `prime-rl.patch` was cut against these revisions:

```
prime-rl             38494ccb   (v0.6.0)
deps/pydantic-config  896ade4e
deps/renderers        19332933
deps/research-environments 841f9a54
deps/verifiers        78c1b3c5
```

Trinity's tokenizer has no `<tool_call>` special token. It encodes this text as
four ordinary tokens. Renderers that search by token ID therefore miss every
tool call. The historical
`patches/submodules/renderers-hermes-tool-parser.patch` adds a text parser named
`hermes`, as specified by Trinity's model card. `<think>` is a real token, so
`reasoning_parser = "think"` works. The parser must live in `renderers`: each
environment server imports that package without importing `prime_rl`.

The legacy subtask blend also needs an explicit mix. The outer `ratio` setting
balances environments and does not set the blend inside one environment. A
taskset that loads several subtasks can concatenate them, which weights them
by source size. The historical `nemotron-reasoning` data had 83% science and
2% math under that default. Its `taskset.mix` used `weight`, `max_examples`,
`repeat`, and `enabled` so smaller sources could cycle instead of being starved.

The old bootstrap instructions no longer match this repository layout. When
using that legacy PRIME-RL patch, select `patches/prime-rl/prime-rl.patch`
explicitly and apply it to the pinned base above. Do not apply it together with
`patches/prime-rl/glm52-decensor.patch`; the GLM-5.2 image base for that overlay
is unknown.
