# Known errors

Use the matching section for a reported fault and follow
[operation rules](rules.md) before a repair.

## Inference does not generate

A model-list response does not prove generation. Use `training_ops.engines()`.
For a failed endpoint, use `training_ops.repair("engine:<host>")` only if its
supervisor is not already restarting it.

## A supervisor stopped

Check `training_ops.alive(pattern, host=host, log=log)` with the exact process
pattern and log. If progress is stale, use `training_ops.repair("supervisors")`.

## A trainer worker stopped

Find the first worker error. Do not restart one worker. Adapter-only resume
resets the optimizer and must pass the saved-state checks in `start()`.
If they fail, report the blocker; do not select a fresh run silently.

## Rollouts stop at the end of a batch

A rollout wall timeout is terminal for that attempt, not a network retry.
Incomplete groups must not mix policy versions. Use only the existing group
replacement rules and stop gates in `agents/TRAINING_EXPECTATIONS.md`; do not raise
retry counts to hide a fault.

## Adapter update does not finish

Check each configured endpoint and its adapter identity. Record the failed
stage. Use the matching repair and its idle-request checks.

## GPU memory allocation fails

Record the first failed rank, requested bytes, free bytes, and reserved unused
bytes. A single retry after releasing unused allocator cache is allowed only
where the source implements it. A second failure is terminal.

## Host memory is low

Check `ac_offloading` against the approved configuration. Do not change it
without authority or call high host-memory use an accounting error.

## A source change has no effect

Check `training_ops.mounted_files()`, the loaded version, and any needed
restart. Do not copy container source into the repository.
