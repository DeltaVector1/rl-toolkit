# Decensor RL

This repository trains Muse-Glimmer-30B with GRPO. GLM-5.2 work is paused.

Read these files in order:

1. `AGENTS.md` for safety and host rules.
2. `configs/operations.toml` for hosts, ports, paths, and run profiles.
3. `TRAINING_EXPECTATIONS.md` before a qualification or production run.
4. `docs/RUNBOOK.md` for normal commands.

## Layout

```text
configs/current/        Active trainer and orchestrator settings
configs/paused/         Settings for paused work
configs/qualification/  Trainer-only checks
configs/runtime/        SSH, proxy, and inference settings
environments/src/       Verifiers environments
environments/tests/     Environment tests
evals/                  Held-out evaluations
scripts/china/          Trainer service internals
scripts/h200/           Inference service internals
scripts/qualification/  Manual GPU checks
scripts/setup/          Machine and model setup
source/                 Patched Prime-RL source and tests
patches/                Maintained source patch sets
```

Use the `training-ops` project skill for status, reward, logs, deployment,
repairs, and run control. Do not run service scripts by hand unless the
runbook says that the tool is unavailable.

Do not touch `n79`. Secrets stay in `/data/trinity/training.env`.

SwanLab: <https://swanlab.cn/@MangyMangaur/muse-runs>
