# Runbook

Use `training-ops` for normal work. It reads all host and path data from
`configs/operations.toml`.

From Python:

```python
import training_ops

training_ops.status()
training_ops.reward()
training_ops.steps(limit=20)
training_ops.logs("trainer")
training_ops.logs("orchestrator")
training_ops.engines()
training_ops.parity()
training_ops.mounted_files()
training_ops.package_versions()
```

From the repository shell:

```bash
python3 -m training_ops status
```

The shell profile exports the training-ops source path. If this import fails,
fix that export instead of adding a different command to each runbook entry.

Use this foreground monitor for an active run:

```python
training_ops.watch(hours=2, every=300)
```

Do not run it in the background. Do not use an interval above 600 seconds.
`watch()` reports only; it does not stop a test. The operator must enforce
the 600-second trainer and 720-second rollout limits and trace a slow test.

## Known repairs

Run `status()` first. Apply only the repair that matches its report.

```python
training_ops.repair("engine:h200a")
training_ops.repair("supervisors")
```

Read `.agents/skills/training-ops/references/errors.md` for the measured fault
conditions.

## Deploy one file

```python
training_ops.deploy("prime-rl/src/prime_rl/orchestrator/dispatcher.py")
```

The tool derives the hosts from the path and launcher mounts. It refuses an
unclear destination. Pass `nodes=[...]` only when you have checked the target.
A copied Python module does not replace the copy in a running process.

## Run control

Get an owner decision before these calls:

```python
training_ops.start("glm-1", profile="current")
training_ops.stop()
```

Use `training_ops.qualify()` with one of the configured trainer-only profiles:
`glm52-native-small` or `glm52-fixed-token`.

For the bounded synthetic diagnostic, use:

```python
training_ops.qualify("glm52-native-small", mode="cold")
```

Use `glm52-fixed-token` for the current neutral qualification. Use
`testing-perf` only for its configured testing profile. The `current` profile
is the Decensor RL run and is not a neutral qualification.

Before a qualification or run, check the mounted source list and image pins:

```python
training_ops.mounted_files()
training_ops.package_versions()
```

After a trainer-only qualification finishes, read its direct parity record:

```python
training_ops.parity("glm52-fixed-token")
```

This parity command is for the fixed-reference profile only; native-small is a
synthetic memory diagnostic and has no fixed-reference parity record.

For a full configured run, parity reads `parity-step*.json` from that
profile's output directory. An archived run name uses the legacy
`runs/<run>` search.

`start()` checks that each selected host has the named configuration. A cold
start creates the orchestrator control file before the trainer starts. A
restart starts the trainer first because that control file already exists.

Do not edit a run name into a copy of an old configuration. Update the named
profile in `configs/operations.toml` and its files.

The local `prime-rl/` overlay is copied to configured host paths at deploy;
launchers bind-mount files into `/app/src`. Dockerfile `COPY` is a separate
image build step, and deploy does not modify an existing image or hot-reload a
running process. The portable patch is not currently regenerated. Do not
restart an old cold run without an owner decision.

No Decensor environment or reward edits were made in these fixes.

## Direct host read

Use this only when no named read command covers the measurement:

```python
training_ops.run_on_hosts("nvidia-smi", group="trainer")
```

The valid groups are `trainer`, `inference`, and `all`. Any command that writes
data or changes a process needs the same decision as a direct SSH command.
