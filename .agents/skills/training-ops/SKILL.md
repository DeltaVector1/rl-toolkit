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

`configs/current` contains the active run profile. `configs/paused` contains
profiles that are not running. `configs/qualification` contains trainer-only
checks.

Missing configuration is an error. Do not guess a host, run, model, config,
or metric location.

## Read commands

```python
training_ops.status()
training_ops.reward()
training_ops.steps(limit=20)
training_ops.logs("trainer")
training_ops.logs("orchestrator")
training_ops.engines()
training_ops.tunnels()
training_ops.parity()
training_ops.package_versions()
training_ops.mounted_files()
```

Use `reward().verdict` as written. Do not claim a change when the verdict is
`inside the noise`.

`watch()` runs in the foreground. Do not run it in a thread or background
process.

## Write commands

```python
training_ops.deploy("source/path.py")
training_ops.start("muse19", profile="current")
training_ops.qualify("muse-qualification")
training_ops.qualify_glm52_pp_async_cuda()
training_ops.repair("tunnels")
training_ops.repair("engine:h200a")
training_ops.repair("supervisors")
training_ops.stop()
```

`start`, `qualify`, and `stop` change live services. Get the required owner
decision first. `repair` is allowed only for an error that `status()` shows.

`deploy()` selects destinations from the operations config and launcher mount
lists. Pass `nodes=[...]` when the destination cannot be derived. It refuses
an unmounted source file.

`run_on_hosts()` runs an explicit command on the configured `trainer_head`,
`trainer`, `inference`, or `all` group. The `trainer_head` group contains only
the inventory `trainer_head` host. Treat it as a write unless the command is
clearly read-only.

## Start order

A new run starts the orchestrator first because it creates the trainer control
file. A restart starts the trainer first because the control file already
exists. Do not change this order.

The tool never forwards `PRIME_BATCH_CACHE`. It never touches a blocked host.
