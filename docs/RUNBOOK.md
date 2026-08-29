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
```

From the repository shell:

```bash
PYTHONPATH=.agents/skills/training-ops/src python3 -m training_ops status
```

Use this foreground monitor for an active run:

```python
training_ops.watch(hours=2, every=300)
```

Do not run it in the background. Do not use an interval above 600 seconds.

## Known repairs

Run `status()` first. Apply only the repair that matches its report.

```python
training_ops.repair("tunnels")
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
training_ops.start("muse19", profile="current")
training_ops.stop()
```

Use `training_ops.qualify()` with a configured trainer-only qualification
profile when one is available.

`start()` checks that each selected host has the named configuration. A cold
start creates the orchestrator control file before the trainer starts. A
restart starts the trainer first because that control file already exists.

Do not edit a run name into a copy of an old configuration. Update the named
profile in `configs/operations.toml` and its files.

## Direct host read

Use this only when no named read command covers the measurement:

```python
training_ops.run_on_hosts("nvidia-smi", group="trainer")
```

The valid groups are `trainer`, `inference`, and `all`. Any command that writes
data or changes a process needs the same decision as a direct SSH command.
