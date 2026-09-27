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

## Scope and completion

- First report `pwd`, repository root, `HEAD`, and `git status`; stop on an
  unexplained mismatch. Announced commits from the assigned writer are
  expected during local review.
- Finish the authorized task and its checks. Fix faults within that scope;
  ask when permission or a required choice is missing. Read-only tasks permit
  no edits or live changes.
- Check the affected behavior. Broaden or repeat tests only for a change,
  failure, or unresolved risk. Local tests with disposable fixtures and the
  repository network guard need no further approval.
- After ten minutes without completion, report the measured state and first
  blocker. At handoff, report changed files, commit, checks, and remaining work.

## Context

- Use the `training-ops` skill for run status, monitoring, deployment, starts,
  stops, and repairs. Do not write another SSH helper.
- Read `configs/nodes.toml` when you need hosts, paths, ports, containers, or
  topology. Read `TRAINING_EXPECTATIONS.md` before a live training test or run.
- Read other files only when the task needs them.

## Safety and permissions

- Never touch `n79`. Do not delete `/data/ZiphuAI`, `/data/trinity`,
  `/data/nv-community_GLM-5.2-NVFP4`, or the only model at
  `/data/glm52-rl/model` on `n85`. Keep secrets in
  `/data/trinity/training.env`.
- Use native FP8 weights. Do not use QLoRA or INT4 model weights, cached
  rollout batches, or a full rollout batch to debug trainer execution. Do not
  change a measurement limit to make a run pass.
- Do not download or copy model weights through SSH when a direct Hugging Face
  or ModelScope download is available.
- Download model weights independently on each model or training node from
  Hugging Face, ModelScope, or another source appropriate to the node location.
  Never copy or stream weights between model or training nodes over SSH unless
  the user explicitly requires it, or a direct download is impossible and the
  user approves after receiving size and time measurements.
- Follow the skill's task-specific approval and measurement rules for live
  work. Do not silently replace a requested method, setting, or metric;
  report the code and metric source.

## Repository

- Keep each project in its own `environments/<name>/` directory and put
  evaluation code under `evals/<name>/`. Load datasets from the Hub or cache.
- Put training-ops scripts under `training_ops/runtime/`.
  Remove temporary files when their task ends and do not add extra docs unless
  requested. Preserve files used by other active tasks.
- Use simple English and names. Do not add new all-caps identifiers.
