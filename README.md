# Trinity RL Kit

Our RL environments, configs, patches, and skills for training Trinity-Mini on
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). This repo is **not** a
fork — it holds only our stuff, so we can pull upstream prime-rl updates freely and
drop this on top.

## Layout

| Folder | What's in it |
|---|---|
| `environments/` | Installable verifiers env packages — `decensor-env` (single-turn, in-use), `decensor-env-multiturn` (old full-conversation version), the four `nemotron-*` envs, and bench envs (`mmlu`, `arcee-ifeval`). Each has its own `pyproject.toml`. |
| `patches/submodules/` | Patches against prime-rl's `deps/` submodules — Trinity tokenizer (`renderers`) and multiturn error handling (`verifiers`). Both load-bearing for Trinity. |
| `patches/prime-rl/` | Our load-bearing edits to prime-rl `src/`, as a git patch. |
| `configs/` | Working RL configs (`trinity_mini_ichthyo_rl`, `trinity_mini_rl`). |
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

## Notes

- Envs pull their datasets from Hugging Face at runtime — no data is vendored here.
- `patches/prime-rl/` is the piece that rots when upstream moves; regenerate it
  against a newer base with `git diff <base> -- src/ > patches/prime-rl/prime-rl.patch`.
