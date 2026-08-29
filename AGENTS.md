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

The current profile trains Muse-Glimmer-30B with GRPO. It uses BF16 base
weights and a rank-128 LoRA adapter on all supported attention and MLP linear
weights.

GLM-5.2 optimization work uses the sibling worktree
`/home/mango/Misc/data-gen/glm52-decensor-glm52-optims` and its qualification
profiles. Do not infer a GLM topology from the Muse profile below.

- Eight trainer hosts use 64 RTX 4090 GPUs.
- One inference host uses eight independent H200 replicas.
- One external judge host uses eight RTX 5090 GPUs.
- The orchestrator runs on the inference host.
- The adapter server runs on the trainer head.

The trainer time and rollout time overlap. Report them as separate values.
Also report adapter transfer, full-weight transfer, judging, and save time
separately.

## Full-weight warning

A Muse BF16 checkpoint is 55.5 GiB. A measured transfer to two H200 hosts took
about 37 minutes for each policy version. Warn the owner before any full-weight
run. Do not include this time in trainer or rollout time.

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

Get an owner decision before you:

- stop or restart a run;
- start a new run name;
- change batch size, group size, inflight rollout count, parallelism, or
  learning rate;
- write outside this repository or the run directories.

Use `training_ops.watch()` in the foreground. Its interval must not exceed ten
minutes. Do not start a background monitor.

## Source of truth

- `configs/current/` contains the current run settings.
- `configs/paused/` contains paused settings.
- `configs/qualification/` contains trainer-only settings.
- `docs/RUNBOOK.md` contains commands.
- `docs/HANDOFF.md` contains design history and past faults.
- `docs/PERFORMANCE.md` contains measurements.
- `docs/PINS.md` contains version pins.
- `.agents/skills/training-ops/references/errors.md` maps known faults to
  repairs.
