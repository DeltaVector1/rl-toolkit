# Durable rollout transport

The stopped x25 data path sent each rollout across the China link. This path
has 217 ms round-trip time and measured packet loss between 3 and 33 percent.
A failed HTTP response discarded the generation.

Test x26 used this edge topology:

```text
h200a orchestrator -> h200a/h200b engines -> local immutable batch
                                             |
                                      resumable rsync
                                             |
                                      n82 trainer
```

The orchestrator and judge traffic stay outside China. Configure the
orchestrator and trainer with filesystem rollout transport. Run
`scripts/h200/sync_rollout_batches.sh` beside the edge orchestrator. The script
copies `control/orch.toml` and each completed `train_rollouts.bin` to n82. It
uses a persistent partial file, resumes after disconnects, verifies SHA-256,
and publishes with one rename. The trainer cannot see a partial batch.

The filesystem on each trainer node is local. Rank 0 makes `rank_*.bin` under
the trainer `output_dir` on n82. `scripts/china/sync_microbatches.sh` copies
each file to the same path on the other seven nodes, verifies it, and renames
it atomically. `cluster.sh` starts this process on rank 0 for a filesystem
transport.

The weight watcher on h200a also needs a local policy marker. The H200 sync
process pulls each n82 `broadcasts/step_N/STABLE` marker to the matching local
path. Adapter files stay on n82 and the engines fetch them through the adapter
HTTP tunnel.

Example:

```bash
SOURCE_RUN_DIR=/data/glm52-rl/runs/xNN/run_xNN \
DEST_HOST=china-trainer \
scripts/h200/sync_rollout_batches.sh
```

Set this in both trainer and edge-orchestrator configs:

```toml
[rollout_transport]
type = "filesystem"
```

x26 tested this path and is stopped. Do not start the n82 orchestrator or the
n82 run supervisor with an edge run.

## Capacity

x26 step 1 collected 1,029 trainable attempts in 20 minutes 5 seconds. Step
2 collected 1,023 in 18 minutes 41 seconds. These rates are 51.2 and 54.8
trainable rollouts a minute.

The immutable files were 84.2 MB and 92.5 MB. Transfers finished in 28.8 and
29.4 seconds, and both destination hashes matched. Bulk transfer overlapped
the next collection and was not the collection-time limit.

Two H200 hosts did not reach the 68 rollouts a minute required for a 15-minute
collection. More inference capacity is necessary if 15 minutes remains the
target.

## Scale-out protocol

For more rollout sites, use the INTELLECT-2 pattern from arXiv:2505.07291:

- The trainer publishes the oldest underfilled policy step.
- Workers pull deterministic group leases and can join or leave.
- Workers persist completions locally before acknowledgment.
- Workers upload immutable, independently compressed shards with hashes.
- n82 accepts a record ID once and publishes a durable consumed acknowledgment.
- Adapter files are content-addressed, resumable, hash-checked, and activated
  only after all engines report the same version.
- Generation for step N+1 overlaps training step N. Start with one-step
  staleness and increase it only after mismatch and reward measurements.
- Choose transfer paths by an exponential moving average of
  `success_rate * bandwidth`, as SHARDCAST does.

The current provider exposes only SSH port 31331. Therefore the first version
uses resumable rsync through SSH. If the provider opens UDP, use a KCP/FEC
transport. Do not use WAN ZMQ for batches: it is fire-and-forget and does not
provide durable delivery or end-to-end integrity.

For 16K environments, keep complete samples intact. Store typed token IDs,
logprobs, masks, rewards, policy version, and adapter hash. Pack samples by
length only when the trainer builds micro-batches.
