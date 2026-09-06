# Decensor RL

Use ASD-STE100 Simplified Technical English. Keep reports short. State
measurements before conclusions. Do not use metaphors.

## Start here

Read `configs/operations.toml`. It is the only host, port, path, container,
and profile inventory. Read `TRAINING_EXPECTATIONS.md` before a qualification
or production run.

Use the `training-ops` skill. Do not write a new SSH helper for work that the
skill can do.

```python
import training_ops

training_ops.status()
training_ops.reward()
training_ops.steps(limit=20)
training_ops.watch(hours=2)
```

Prime Agent loads the same skill through `.prime/agent/skills/training-ops`.

## Sub-agent work

Use `fork_turns="none"` for a bounded task. Put all required context in the
task message. Use a positive turn count only when the task needs recent user
context. Do not use a full-history fork for normal implementation, audit, or
live operation work.

Each task message must state:

- the exact worktree and expected commit;
- any expected dirty or untracked paths that must be preserved;
- whether the agent may read, edit, commit, or operate a live service;
- the allowed files or service profile;
- the required checks and final evidence.

Use one writer for each branch. Other agents on that branch are read-only.
Use one operator for each live qualification. A repair agent edits code but
does not operate the live service. Reuse the same operator with
`followup_task` after the shared branch contains the repair commit.

An agent must first report `pwd`, the repository root, the current commit, and
the worktree state. It must stop on a mismatch. A commit on an isolated ref is
not ready for a live run until the target worktree contains that commit.

Before a live start, check the configured source list, recursive imports,
launcher mounts, patch targets, and remote file modes as one set. Do not fix
one missing file and start again without checking the full set.

After ten minutes without completion, report the measured state and the first
blocking item. Do not continue a broad search. A final handoff must contain the
outcome, measurements, changed files, commit, checks, and remaining blocker.

## Current system

The current profile trains GLM-5.2 with GRPO. It uses native FP8 base weights
and a rank-32 LoRA adapter on the supported attention, shared-expert, and
routed-expert linear weights. The router stays frozen.

- Eight trainer hosts use 64 RTX 4090 GPUs.
- Two inference hosts each use eight H200 GPUs as one TP8 server.
- One external judge host uses eight RTX 5090 GPUs.
- The orchestrator runs on the inference host.
- The adapter server runs on the trainer head.

The trainer time and rollout time overlap. Report them as separate values.
Also report adapter transfer, full-weight transfer, judging, and save time
separately.

## Full-weight warning

The GLM FP8 model on each H200 host is 755,632,050,320 bytes, approximately
704 GiB. Warn the owner before a full model copy. Do not include full model
copy time in trainer, adapter, or rollout time.

## Safety rules

- Never touch `n79`.
- Do not delete `/data/ZiphuAI`, `/data/trinity`, or
  `/data/nv-community_GLM-5.2-NVFP4`.
- The GLM model at `/data/glm52-rl/model` on `n85` is the only copy.
- Keep secrets in `/data/trinity/training.env`.
- Do not use QLoRA or INT4 model weights. INT4 expert-activation transport is
  separate and is allowed.
- Do not use a cached rollout batch.
- Do not use a full rollout batch to debug trainer execution.
- Do not change a measurement limit to make a failed run pass.
- Do not claim a reward trend when `training_ops.reward()` says it is inside
  the noise.

## Allowed work

You may read and measure the system. You may edit this repository and run
local tests. You may repair a known tunnel, inference, or supervisor fault
after `training_ops.status()` shows that fault.

For the current neutral GLM performance effort, the owner authorizes needed
repairs and test restarts. This does not authorize unrelated work.
For this effort, if combined forward and backward elapsed time, from the start
of forward to the end of backward, exceeds ten minutes, stop the test,
preserve relevant logs, and trace the cause before a fresh retry. GPU
utilization alone does not prove progress.

Get an owner decision before you, except for that scoped permission:

- stop or restart a run;
- start a new run name;
- change batch size, group size, inflight rollout count, parallelism, or
  learning rate;
- write outside this repository or the run directories.

Use `training_ops.watch()` in the foreground. Its interval must not exceed ten
minutes. Do not start a background monitor.

## Source of truth

- `configs/current/` contains the current run settings.
- `docs/RUNBOOK.md` contains commands.
- `docs/HANDOFF.md` contains design history and past faults.
- `docs/PERFORMANCE.md` contains measurements.
- `docs/PINS.md` contains version pins.
- `.agents/skills/training-ops/references/errors.md` maps known faults to
  repairs.

## Repository preferences

- Keep one project in each `environments/<name>/` directory. Do not mix
  Ornith, decensor, or another project in one environment.
- Put evaluation code under `evals/<name>/`. Keep evaluation datasets on the
  Hugging Face Hub. Do not keep prompt copies in the repository.
- Put internal operational launchers in
  `.agents/skills/training-ops/runtime/`, grouped by host role when needed.
  Do not leave launchers in a public `scripts/` directory.
- Treat scripts, temporary configs, tests, reports, and scratch files as
  disposable. Delete each item when its task is complete. Do not keep a
  one-time script for later reuse without a current owner-approved use.
- Load datasets from the Hugging Face Hub or the configured cache. Do not
  copy prompt datasets into the repository.
- Use simple English and simple file names. Do not add new all-caps
  identifiers. Do not hide fallback paths, fallback metrics, or metric
  changes. Report the selected path and metric source.
- Keep comments short and useful. Do not add verbose comments, README files,
  or extra Markdown unless the owner asks for them.
- Clean the worktree after each task. Remove unused files, old configs, and
  generated output before handoff.
- If a requested feature cannot be used as requested, stop and ask the owner
  before disabling it or choosing a fallback. Ask when a file, setting, or
  dataset purpose is unclear. Do not silently replace settings such as MTP.
