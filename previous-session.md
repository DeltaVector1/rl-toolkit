# Previous session handoff

Source session: 2026-08-07 18:09 UTC to 2026-08-10 04:34 UTC.
The original `previous-session.html` export was deleted at the owner's request.

## Summary

The session audited the repository and live system, moved rollout orchestration
to the H200 side, added durable filesystem transport, and fixed the main native
GLM-5.2 FP8 correctness failures. It also created a portable eight-patch
Prime-RL v0.8.0 series.

The native-FP8 path has close forward parity, finite gradients, adapter export,
and correct H200 adapter loading. It does not meet the production gates. The
validated context is 2,048 tokens instead of 32,768. Trainer forward/backward
takes about 48 to 69 minutes instead of 10 to 15 minutes. Ordinary active
rollout collection takes about 18 to 22 minutes instead of approximately 10
minutes. Batch 2,048 is not validated.

The Prime-RL v0.8.0 Decensor migration that was incomplete at session end is
now complete in commits `14d22f8` and `f82bec8`. The exact v0.8 environment was
validated through both its canonical v1 evaluator and the environment-server
path used by training.

## Completed work

- Corrected documentation for topology, precision, tunnel count, image pins,
  throughput, watchdog behavior, and run operation.
- Added `TRAINING_EXPECTATIONS.md` as the mandatory acceptance specification.
- Moved orchestration to `h200a` and implemented resumable, checksummed,
  atomically published filesystem rollout transport with trainer-node fanout
  and reverse adapter `STABLE` marker transfer.
- Verified all 141 official GLM-5.2-FP8 checkpoint shards.
- Fixed partial-block dequantization for non-aligned `kv_a_proj_with_mqa`
  tensors.
- Fixed native routed-expert row ordering before and after the local fused
  expert call.
- Kept FP8 weights and scales as explicit FSDP/autograd inputs and kept owned
  backward snapshots outside checkpoint saved-tensor hooks.
- Fixed fully masked sparse-MLA rows so their backward result is finite.
- Removed silent gradient sanitization and added fail-fast nonfinite handling.
- Added FQN-based FP8 diagnostics.
- Fixed hung LoRA load responses with an absolute deadline, exact visibility
  checks, and full restart-window retries.
- Added queue-inclusive stale-rollout rejection.
- Restored final-assistant-turn-only Decensor judging.
- Built and validated the ordered Prime-RL v0.8.0 patch series in
  `patches/prime-rl/v0.8.0-glm52/`.

## Correctness evidence

- x46p after the expert-row fix: masked KL `0.356220`, unmasked KL
  `0.0372957`, and finite grad norm `0.0327431`.
- x55a zero-LR gate: 913 microbatches, grad norm `0.015471`, masked KL
  `0.357186`, `556 tokens/s`, peak `46.55 GiB`, and successful adapter export.
- x56p step 1: grad norm `0.01786058`, masked KL `0.347687`, unmasked KL
  `0.036602`, `564.39 tokens/s`, and peak `46.55 GiB`.
- x56p step-3 adapter SHA-256:
  `091298e90b00939de2f9b5edecc173d502230ecda941b6ed24f56e0428248119`.
  Both H200 engines loaded this file.

## Latest measured live state

At 2026-08-10 04:41 UTC, x56p had:

- trainer and orchestrator up;
- 48 of 48 tunnels up;
- judge and adapter services returning HTTP 200;
- trainer step 14 complete and the next step computing;
- rollout step 19 at 640/1024, plus 62 buffered and 114 in flight;
- finite grad norm `0.0182`;
- throughput `556 tokens/s`;
- both H200 engines generating at full utilization;
- no recent environment failures.

The reward verdict over 18 completed rollout steps was **inside the noise**.
Do not report a measured reward trend.

## Decensor v0.8 result

- Canonical command: `.work/prime-rl-upstream/.venv/bin/eval decensor-v1`.
- The global Prime CLI 0.5.60 `prime eval run` is a legacy v0 loader and is not
  the correct v1 environment runner.
- The retained Decensor tests pass: 11 tests.
- A CPU-only hosted rollout used
  `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` as policy and judge.
- All seven judge replies parsed with no endpoint failure.
- The refusal case produced refusal `9`, Markdown presence `0`, coherency `1`,
  and reward `0.01`.
- The same case passed through the v0.8 environment-server path.
- Temporary evaluation outputs and one-off fixtures were removed after the
  gate passed.

## Remaining work

1. Implement and benchmark a fast Ada/SM89-compatible block-FP8 forward and
   backward path. Keep `_linear_input_grad` as the numerical oracle.
2. Restore safe 32K attention checkpointing without the all-NaN sparse-Q
   failure from non-reentrant FSDP recomputation.
3. Add bounded native-FP8 routed-expert W2 chunking for delayed expert-load
   OOM protection.
4. Restore efficient packing and PP-aware balance so batch 1,024 does not need
   approximately 900 microbatches.
5. Add explicit acceptance alarms for configuration, timing, precision,
   nonfinite gradients, OOM, adapter, transport, environment, and step count.
6. Measure H200 active generation separately from judging, transfer, and gate
   waiting, then reduce it toward approximately 10 minutes.
7. Run gates in order: one matrix, one official layer, one node, one fresh
   full-topology zero-LR step, three post-warmup performance steps, and batch
   2,048 capacity.
8. Only after all earlier gates pass, start a fresh production identity and
   monitor at least 50 trainer steps with foreground `glm52_ops.watch()`.

Do not add `attention`, `attn_proj`, or `mla_up_proj` to current native-FP8
activation-checkpoint targets. These targets produced nonfinite sparse-Q
activations under FSDP resharding.

## Required constraints

- Read `TRAINING_EXPECTATIONS.md` before each trainer gate or production run.
- Use the `glm52-ops` skill for run status, repair, deployment, and run control.
- Never touch n79.
- Do not use QLoRA, INT4 weights, cached batches, background monitors, or the
  local RTX A4000. INT4 expert-activation transport remains allowed.
- Keep trainer compute, active rollout generation, judging, transfer, and gate
  waiting as separate measurements.
- Do not deploy the v0.8 series into x56p.
- Do not change batch size, group size, inflight rollout count, parallelism, or
  learning rate without an explicit experiment decision.
- Preserve x55a and x56p logs, metrics, adapters, manifests, and failed-gate
  evidence.

