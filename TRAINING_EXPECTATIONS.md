# Training expectations

Review this file before each trainer gate or production run. A run does not
pass if it changes these values or exceeds these limits without an explicit
owner decision.

## Required configuration

- Native FP8 trainer forward precision must match H200 FP8 inference as closely
  as measured parity permits.
- Batch size: 1,024, with a validated path to 2,048.
- Group size: 16.
- Context length: 32,768 tokens.
- Maximum output length: 16,384 tokens.
- Learning rate: `8e-6`.
- Maximum gradient norm: `1.0`.
- Weight decay: `0.0`.

## Performance gates after step 1

- Trainer forward and backward: 10 to 15 minutes per step.
- Active rollout generation: approximately 10 minutes per batch.
- Report trainer compute, rollout generation, transfer, and gate waiting as
  separate times.
- Trigger a regression alarm when a run exceeds these limits or reduces the
  required context, batch size, precision parity, or stability.

## Final stability validation

Do this only after the speed, memory, context, and precision gates pass.

Run the final full-topology configuration for at least 50 completed trainer
steps. Keep `glm52_ops.watch()` in the foreground with no interval longer than
10 minutes. Watch for random OOMs, NaN or Inf values, nonfinite gradients,
adapter-load failures, transport failures, environment errors, and reward
regressions. Continue beyond 50 steps when evidence is still incomplete.

Do not accept a short run, a successful step 1, or an early noisy reward rise
as final validation.
