# Known errors

Use the repair only when `training_ops.status()` reports the matching error.

## Inference does not generate

The API process can answer model requests while its worker is dead. Use
`training_ops.engines()`. If a configured endpoint fails and its supervisor is
not already restarting it, use:

```python
training_ops.repair("engine:<host>")
```

## Inference tunnels do not pass data

Confirm that inference throughput is also low. A tunnel probe can time out
when it shares a busy link. If the endpoints are idle and the tunnel probes
fail, use:

```python
training_ops.repair("tunnels")
```

This repair does not touch the batch-transfer port.

## A supervisor stopped

Use `training_ops.alive()` to confirm the process and its log are stale. Then
use:

```python
training_ops.repair("supervisors")
```

## A trainer worker stopped

The trainer has no resumable optimizer state. Do not restart one worker. Find
the first worker error, fix it, and start a fresh run after the owner agrees.

## Rollouts stop at the end of a batch

A rollout wall timeout must be terminal. It must not enter the network retry
path. Incomplete groups must not mix policy versions. Do not increase retry
counts to hide this error.

## Adapter update does not finish

Check every configured inference endpoint. Confirm the adapter checksum on
each host. Restart only the orchestrator when it is at zero active requests
and the trainer is still healthy.

## GPU memory allocation fails

Record the first failed rank, requested bytes, free bytes, and reserved unused
bytes. A single retry after releasing unused allocator cache is allowed only
where the source already implements it. A second failure is terminal.

## Host memory is low

Confirm that `ac_offloading = "None"`. Do not call high host-memory use an
accounting error.

## A source change has no effect

Run `training_ops.mounted_files()`. A container ignores a repository file that
its launcher does not mount. Do not copy a file out of a container into the
repository.
