# Decensor RL

Use ASD-STE100 Simplified Technical English. State measurements before
conclusions. Keep reports short.

## Lead work

- Act as the manager: assign all implementation to subagents, run independent
  assignments concurrently, monitor progress, redirect off-task work, and
  review diffs and tests before accepting them.
- Give each task one goal and name the exact worktree, expected commit, files
  or services, allowed read/edit/commit/live actions, approach, and checks.
  Use `fork_turns="none"` unless recent context is required.
- Use one writer per branch and one operator per live test. Preserve
  existing changed and untracked files.

## Start and report

- Read `configs/nodes.toml` for hosts, paths, ports, containers, and topology.
  Read `TRAINING_EXPECTATIONS.md` before a training test or run.
- Use the `training-ops` skill. Do not write another SSH helper.
- Use `uv run training-ops status` for a read-only status check.
- First report `pwd`, repository root, `HEAD`, and `git status`; stop on a
  mismatch. Before a start, check all required source files, recursive imports,
  mounts, patches, and remote file modes together.
- Use `training_ops.watch()` in the foreground with an interval of 600 seconds
  or less. After ten minutes without completion, report the measured state and
  first blocker.
- Report trainer compute, rollout collection, waiting, judging, adapter
  preparation, transfer, loading, and full-weight transfer separately. Do not
  add overlapping times. Keep estimates separate from measurements.

## Safety and permissions

- Never touch `n79`. Do not delete `/data/ZiphuAI`, `/data/trinity`,
  `/data/nv-community_GLM-5.2-NVFP4`, or the only model at
  `/data/glm52-rl/model` on `n85`. Keep secrets in
  `/data/trinity/training.env`.
- Use native FP8 weights. Do not use QLoRA or INT4 model weights, cached
  rollout batches, or a full rollout batch to debug trainer execution. Do not
  change a measurement limit to make a run pass.
- Warn before copying the 755,632,050,320-byte GLM model (about 704 GiB).
  Keep full-model copy time separate from training, adapter, and rollout time.
- Repair only a fault reported by `training_ops.status()`. Needed repairs and
  restarts for current neutral GLM speed tests are approved; ask the owner for
  all other stops, restarts, run names, or setting changes.
- If `training_ops.reward()` says `inside the noise`, report that the measured
  change is too small to show improvement or decline.

## Repository

- Keep each project in its own `environments/<name>/` directory and put
  evaluation code under `evals/<name>/`. Load datasets from the Hub or cache.
- Put training-ops scripts under `training_ops/runtime/`.
  Remove temporary files when their task ends and do not add extra docs unless
  requested. Preserve files used by other active tasks.
- Use simple English and names. Do not add new all-caps identifiers. Do not
  silently replace a requested method, setting, or metric; report the code and
  metric source.
