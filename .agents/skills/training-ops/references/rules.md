# Operation rules

## Authority and safety

Repository safety limits in `AGENTS.md` apply. Also exclude every host in
`hosts.blocked` in `configs/nodes.toml`. Never forward `PRIME_BATCH_CACHE`.
Use one operator for a live test. Do not edit its source during the test.

Use only the approved run name, settings, and live actions. Needed repairs
and restarts for current neutral GLM speed tests are approved. Other stops,
restarts, run names, or setting changes need the owner's task-specific approval;
do not ask again for actions already approved.
Repair only a matching fault reported by `training_ops.status()`.

## Deploy and start

Before a deploy or start, verify the required source files and recursive
imports, mounts, patches, remote file modes, and exact container identities
together. Check cleanup targets before a restart. Use `deploy()` with the
configured destinations or an explicit node list. A `prime-rl/` file must be
mounted on each selected host, even with explicit nodes.

Use `start()` with the approved run, mode, and configuration. Cold start
creates orchestrator control before trainer launch;
adapter-only resume starts the trainer before the orchestrator. Resume must
agree on the next batch and exact stable policy, reject active cached rollout
binaries, and report any optimizer reset. Do not silently fall back to a cold
start. Preserve useful adapter partials and stable adapters unless corrupt.

Adapter-only resume needs the Python API; the CLI cannot pass its arguments:
`training_ops.start(run, mode="restart", config_dir=config,
resume_adapter=adapter, resume_step=step)`.

Apply `TRAINING_EXPECTATIONS.md` stop gates. Record the first failure and
measured state. A permitted isolated rollout replacement is not a run stop.
Do not hide terminal failures with more retries. After a source fix, review
its checks and verify the accepted commit before another live test.
A cached adapter does not qualify a new update, transfer, load, or parity path.
Treat `run_on_hosts()` as a write unless its command is clearly read-only.

## Measurements and monitoring

Use `training_ops.watch()` in the foreground for the requested duration. Set
`every` to 600 seconds or less; do not use a background thread or process.

Report trainer compute, rollout collection, waiting, judging, adapter
preparation, transfer, loading, and full-weight transfer separately. Separate
adapter gather, write, and compression when measured. Do not add overlapping
times. Label estimates. A network timeout is a failed test: report bytes,
elapsed time, and errors before giving a cause.

Warn before copying the 755,632,050,320-byte GLM model (about 704 GiB). Keep its
copy time separate from training, adapter, and rollout time. If `reward()`
says `inside the noise`, report that the measured change is too small to show
improvement or decline.
