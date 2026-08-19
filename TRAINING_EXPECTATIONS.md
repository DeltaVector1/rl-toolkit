# Training expectations

Review this file before each trainer qualification or production run. These
are acceptance requirements. Do not implement them as a sequence of full
rollout gates. A run does not pass if it changes these values or exceeds these
limits without an explicit owner decision.

Do not use a full rollout batch to debug trainer execution. First use a
trainer-only fake-data qualification on the exact distributed topology. Use a
direct deterministic Hugging Face reference to debug log-probability parity.
Start rollout production only after these direct tests pass.

## Muse-Glimmer-30B owner decision

These requirements replace the GLM-5.2 precision, batch, context, and
parallelism requirements for a run whose name starts with `muse`.

- BF16 base weights and BF16 LoRA weights. Do not use FP8, QLoRA, or INT4
  weights.
- Rank-128 LoRA with alpha 256 and dropout 0.
- Target all 416 linear modules in the 52 language-model layers: Q, K, V,
  attention output, attention gate, MLP gate, MLP up, and MLP down.
- Do not target the vision tower. Muse-Glimmer is dense and has no routed
  experts or router.
- The expected adapter has 838,336,512 trainable parameters.
- Batch size 2,048, group size 16, context length 65,536, maximum output length
  16,384, learning rate `8e-6`, maximum gradient norm 1.0, and weight decay 0.
- Use PP8, CP4 ring, and FSDP2 on all 64 RTX 4090 GPUs.
- H200 inference must enable BF16 rank-128 LoRA and the same seven projection
  suffixes.
- H200 inference must use the Transformers model implementation unless a
  separate gate proves that the native vLLM implementation has equal
  trainer-to-inference parity.
- Before a production run, complete 20 trainer-only steps on the exact PP8,
  CP4, FSDP2, BF16, rank-128 topology with changing packed-document
  boundaries and real backward work. This test does not use rollouts.
- Compare the exact distributed trainer log probabilities with an independent
  Hugging Face reference on deterministic tokens. Do not start rollout
  production while this direct comparison fails.
- Production uses fresh rollouts and learning rate `8e-6` from step 1. It does
  not use a separate full-batch zero-learning-rate rollout gate.
- The production trainer must reject a step before the optimizer update and
  adapter broadcast if unmasked mismatch KL exceeds 0.1, masked mismatch KL
  exceeds 0.5, or more than 6 percent of tokens are masked. Do not increase
  the DPPO mask thresholds to make the check pass.
- After step 1, alarm if trainer forward and backward exceeds 8 minutes or
  active rollout collection exceeds 20 minutes. A faster result is not a
  failure.
- Report trainer compute, rollout collection, adapter transfer, and gate wait
  as separate times.
- Complete at least 50 stable trainer steps before final acceptance.

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

## Performance requirements after step 1

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
