# Decensor RL

This repository operates the Muse-Glimmer-30B GRPO run. The active policy is
`muse13`. It uses BF16 model weights and a rank-128 LoRA adapter on supported
attention and MLP linear weights. The older GLM-5.2 run is paused.

Read these files first:

1. `AGENTS.md` gives the host topology and operating rules.
2. `TRAINING_EXPECTATIONS.md` gives the run acceptance requirements.
3. `docs/RUNBOOK.md` gives the operating commands.
4. `docs/HANDOFF.md` explains the design and known problems.

## Repository layout

```text
configs/   Current production, qualification, and parity configurations
docs/      Runbook, handoff, measurements, research, and design notes
env/       Decensor environment and tests
patches/   Maintained deployment patch sets
scripts/   Launch, repair, validation, and transport tools
source/    Patched Prime-RL source and tests
```

The current production files are:

- `configs/china/trainer.muse13.toml`
- `configs/h200/orchestrator.muse13.toml`
- `configs/h200/inference.toml`

Use the `glm52-ops` project skill for status, reward, repair, deployment, and
run control. Do not touch `n79`. Secrets must remain outside this repository
in `/data/trinity/training.env`.

SwanLab: `https://swanlab.cn/@MangyMangaur/prod-run`
