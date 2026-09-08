---
name: training-ops
description: Operate this distributed RL training project. Use it to read status, reward, step times, logs, inference health, and parity; run foreground monitoring; copy repository files; start configured run or qualification profiles; stop services; and repair known tunnel, inference, or supervisor failures. Host names, counts, ports, paths, containers, and profiles come from configs/operations.toml.
---

# Training operations

Import the tool as `training_ops` or use the `training-ops` command.

```python
training_ops.status()
training_ops.reward()
training_ops.steps()
training_ops.watch(hours=2)
```

The read functions print a short report and return structured data.

Read `references/rules.md` before a write. Read `references/errors.md` when
`status()` reports an error.

## Configuration

Read `configs/operations.toml`. It is the only inventory for hosts, ports,
paths, containers, profiles, trend settings, and performance requirements.
Do not add a host or replica count to Python code.

Read `TRAINING_EXPECTATIONS.md` before a qualification or production run.
Use `python3 -m training_ops --help` for the installed command surface.

`configs/current` contains the active run profile. Optional trainer-only
profiles must be registered in `configs/operations.toml`.

Missing configuration is an error. Do not guess a host, run, model, config,
or metric location.

Operational launchers are in `.agents/skills/training-ops/runtime/`, grouped
under `china/` and `h200/`. Use `training_ops.deploy()` for launcher changes;
do not create a separate scripts directory.

## Read commands

```python
training_ops.status()
training_ops.reward()
training_ops.steps(limit=20)
training_ops.acceptance()
training_ops.logs("trainer")
training_ops.logs("orchestrator")
training_ops.engines()
training_ops.parity()
training_ops.package_versions()
training_ops.mounted_files()
```

Use `reward().verdict` as written. Do not claim a change when the verdict is
`inside the noise`.

Keep trainer, rollout, waiting, judging, adapter preparation, transfer,
loading, and full-weight transfer measurements separate. A timed-out network
test failed; report its bytes, elapsed time, and errors.

`watch()` runs in the foreground. Do not run it in a thread or background
process.

## Write commands

```python
training_ops.deploy("prime-rl/path.py")
training_ops.start("glm-1", profile="current")
training_ops.repair("engine:h200a")
training_ops.repair("supervisors")
training_ops.stop()
```

`start`, `qualify`, and `stop` change live services. Get the required owner
decision first. `repair` is allowed only for an error that `status()` shows.

Do not use a full rollout batch to debug trainer execution or change a
measurement limit to make a run pass. A cached adapter does not qualify a new
adapter update, transfer, load, or parity path.

`deploy()` selects destinations from the operations config and launcher mount
lists. Pass `nodes=[...]` when the destination cannot be derived. It refuses
an unmounted source file.

`run_on_hosts()` runs an explicit command on the configured `trainer`,
`inference`, or `all` group. Treat it as a write unless the command is clearly
read-only.

## Start order

A new run starts the orchestrator first because it creates the trainer control
file. A restart starts the trainer first because the control file already
exists. Do not change this order.

The tool never forwards `PRIME_BATCH_CACHE`. It never touches a blocked host.

## Delegated qualification

Use one live operator for a qualification. The operator must use the exact
configured profile and target worktree. It must report the commit before the
start. It must not edit code during the live run.

When a run fails, report the first exact error and stop the service through the
configured operation. Give the repair to a separate writer. Before the next
start, the operator must verify that the shared target worktree contains the
repair commit.

Before the first host action, verify these as one dependency set:

- configured deployment files;
- recursive Python imports;
- launcher mounts;
- patch source and target files;
- file access modes;
- container identity and cleanup parsing.

Use focused searches and the named `training_ops` entry points. Do not scan a
whole source tree when the profile, launcher, or error already names the
relevant file.
