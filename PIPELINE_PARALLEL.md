# Pipeline parallelism

Historical run p4 used `pp=7, cp=8, ep=8, global_ep=false`, batch 1024
over 7 nodes / 56 GPUs. Current x25 uses `pp=8, cp=8, ep=8` over 8 nodes /
64 GPUs. The measurements below are from p4 unless a run is named.

## What it bought

| | Before | With PP, cp=4 | With PP, cp=8 |
|---|---:|---:|---:|
| Forward+backward @ 1024 | ~60 min | 12.3 min | **7.9 min** |
| Peak memory (max rank) | 41-47 GiB | 47.3 (OOM) | **39.1 GiB** |
| Expert all-to-all group | 56 ranks, network | 8, node-local | 8, node-local |

`cp=8` was the difference between OOM and headroom, and it was also 1.6x faster
than `cp=4` on the identical batch: the extra sequence parallelism is node-local,
so it costs ring-attention traffic that never leaves the box. The EP group is
`dp_shard_in_ep * cp`, so raising cp keeps experts node-local.

The all-to-all was 88% of every micro batch and crossed one 10 GbE port per
node. Node-local it is 80x faster (971 ms/layer -> 12 ms/layer). `pp` is the
slowest-varying mesh dimension, so stage `s` is exactly one node and the
all-to-all stops touching the network. Expert memory per GPU is unchanged:
each rank holds `layers/pp` layers sharded `world/pp` ways instead of all
layers sharded `world` ways.

**The trainer is no longer the bottleneck.** Rollout generation on the two
H200 hosts now sets the pace; the trainer idles waiting for batches.

## Things that will bite again

1. **Pad before context-parallel sharding, never after.** Padding a CP shard
   puts pad tokens at the end of *each* shard, so after `gather_for_cp` they sit
   interleaved through the sequence while the full-length loss tensors carry
   theirs at the end. Every per-token tensor then lines up against the wrong
   logprobs. Trains happily, silently wrong. `pad_micro_batch_to_length` runs
   pre-shard; `pad_side_inputs_to_common_length` is only an invariant check.
2. **`temperatures` pads with 1.0, not 0.** The loss divides logits by
   temperature; a zero pad gives inf -> nan and poisons the whole micro batch.
3. **Stage boundaries must land on layers that compute their own sparse-MLA
   indices.** 57 of 78 layers are "shared" and reuse the previous layer's
   `cached_indices`, which cannot cross a stage boundary -- the stage receives
   `None` and dies inside the attention kernel. `balanced_boundaries` restricts
   splits to "full" layers and still achieves the optimal max stage size (12).
   The naive even split cuts at 12/23/34/45/56/67, every one a shared layer.
4. **Layers are a `ModuleDict` keyed by GLOBAL index.** A dense `ModuleList`
   slice renumbers layers and checkpoint keys stop resolving. Anything walking
   the decoder must go through `model.iter_layers`.
5. **Router replay indexes `routed_experts` by global layer index.** Getting
   this wrong replays the wrong layer's routing and corrupts training silently.
6. **`return_outputs=False`** on `schedule.step`, or it tries to concatenate the
   last stage's scalar losses.
7. **Gradient clipping needs `pp_mesh`**, or the norm is per-stage.
8. **Weight broadcast gathers over the WORLD group**, merging each stage's
   dict; layer names carry global indices so the union is the full model.
9. **Metric tensors are cast to float32 before gathering.** Ranks holding no
   loss contribute an empty float32 tensor while real values may be bfloat16,
   and `flexible_all_gather` sizes buffers from the local dtype.

10. **Do NOT use full activation checkpointing.** `mode = "full"` trips
   `CheckpointError: Recomputed values ... have different metadata` on this
   MoE+EP stack: recompute does not reproduce the per-expert token counts the
   original forward saw. Use `selective` and buy memory from `cp`.
11. **Peak memory must be all-reduced MAX across ranks.** Rank 0 is stage 0,
   which owns the fewest layers; reporting it hid 12-layer stages sitting at
   47.3 of 47.4 GiB until they OOM'd.
12. **`ppretry.sh` is only safe before the trainer's first completed step.**
   After that the orchestrator has advanced its policy and buffered batches a
   restarted trainer never receives; it then holds at its off-policy limit
   waiting for a policy that cannot arrive. Restart both, or start a new run.
13. **`max_off_policy_steps` needs slack (4, not 2).** ZMQ batches are
   fire-and-forget, so a trainer restart loses whatever was in flight.
14. **Verify the broadcast adapter after ANY change to the weight path**:
   `safe_open(...)` and count distinct layer indices. It must be 78. A truncated
   adapter loads without complaint and serves a partly-unadapted model.

## Debug tooling

- `PRIME_BATCH_CACHE=<dir>` -- save the first received batch, replay it on later
  starts. Turns a ~40 min debug cycle into a model load. 16 GB/node.
- `PRIME_PP_MAX_MICRO=N` -- truncate micro batches; a step takes ~1 min. Debug
  only, never in production.
- `scripts/ops/ppretry.sh <RUN>` -- restart only the trainer, reusing the
  orchestrator's buffered batch.

## Open

- **Entropy reads ~1.7-4.1 against 0.487 in the pre-PP run r35** on the same
  model and config. Mask alignment is verified correct (`ent_in < ent_out` on
  every micro batch, the same ordering as the baseline) and `compute_loss` never
  receives entropy, so it cannot affect training -- but the absolute level is
  unexplained. Watch reward across steps; that is the signal that settles it.
- **Grad norm varies widely by batch** (0.0005 to 0.0150 against a 0.013-0.047
  baseline band). Not yet understood.

## Node set

`n81` is down at the OS level. `cluster.sh` derives rank and `nnodes` from
`TRAINER_NODES`; to restore it add `81` to that list, re-add `n81` to the ops
scripts, and set `pp = 8`.

Rollback: `scripts/ops/cl.py rollback 20260802_174009`.
