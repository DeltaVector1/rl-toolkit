---
name: training-ops
description: Inspect, monitor, deploy, and repair this repository's distributed RL runs with training_ops. Use for run starts, stops, status, timing, reward, and inference health.
---

# Training operations

Use the existing `training_ops` API or `uv run training-ops`. Do not make an
SSH helper. Use `uv run training-ops --help` for commands and arguments.

## Select the task

- **Inspect:** Start with `uv run training-ops status` when current run state
  is needed. Use `reward()`, `steps()`, `logs()`, `engines()`, or `parity()`
  only for the requested evidence. These functions are in `training_ops`.
- **Monitor or report measurements:** Read the measurement section in
  [rules.md](references/rules.md). Use `training_ops.watch()` in the foreground.
- **Deploy, start, stop, or repair:** Read [rules.md](references/rules.md).
  Use `deploy()`, `start()`, `stop()`, or `repair()` for the authorized action.
- **Diagnose a reported fault:** Read its section in
  [errors.md](references/errors.md). Diagnosis alone does not permit a repair.

Read `configs/nodes.toml` when selecting a host, path, port, container, or
mount. It is the only host inventory. Do not guess missing configuration.
Do not hard-code hosts or replica counts.
Read the selected run configuration and `TRAINING_EXPECTATIONS.md` before a
live training test or start. Search the relevant entry point and its imports
when source inspection is needed; a status check does not need a source audit.

Host launchers and helpers are under `training_ops/runtime/china/` and
`training_ops/runtime/h200/`. Keep runtime changes there. A repository edit
does not prove that a running container loaded it; use `mounted_files()` and
`package_versions()` when checking deployment.
