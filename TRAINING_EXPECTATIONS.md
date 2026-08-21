# Training requirements

Read this file before each trainer qualification or production run. These are
acceptance requirements. Do not change them without an owner decision.

Use trainer-only fake data to debug trainer execution. Use deterministic Hugging
Face output to check log probabilities. Start rollout production only after
these direct checks pass.

## Muse-Glimmer-30B

- Use BF16 base and LoRA weights. Do not use FP8, QLoRA, or INT4 weights.
- Use LoRA rank 128, alpha 256, and dropout 0.
- Train all 416 supported language-model linear modules. These are Q, K, V,
  attention output, attention control, MLP control, MLP up, and MLP down.
- Do not train the vision tower.
- Require 838,336,512 trainable parameters.
- Use batch 1,024, group 16, context 65,536, maximum output 16,384, learning
  rate `8e-6`, maximum gradient norm 1.0, and weight decay 0.
- Use PP8, CP4 ring, and FSDP2 on 64 RTX 4090 GPUs.
- H200 inference must load the same rank-128 adapter targets.
- Use the Transformers model implementation until the native vLLM model gives
  equal trainer and inference log probabilities.
- Complete 20 trainer-only steps on the production trainer topology. Use real
  backward work and changing packed-document boundaries.
- Compare distributed trainer log probabilities with an independent Hugging
  Face result on fixed tokens.
- Reject a step before the optimizer update if unmasked mismatch KL is above
  0.1, masked mismatch KL is above 0.5, or more than 6 percent of tokens are
  masked.
- After step 1, warn if forward and backward exceeds 8 minutes or active
  rollout collection exceeds 20 minutes.
- Complete at least 50 stable trainer steps for final acceptance.

## GLM-5.2

- Use the native FP8 trainer path and measure it against H200 FP8 inference.
- Use batch 1,024, group 16, context 32,768, maximum output 16,384, learning
  rate `8e-6`, maximum gradient norm 1.0, and weight decay 0.
- After step 1, forward and backward must take 10 to 15 minutes.
- Active rollout generation should take about 10 minutes per batch.
- Validate the path to batch 2,048 before using that batch in production.

## Final acceptance

Run the full topology for at least 50 completed trainer steps. Keep
`training_ops.watch()` in the foreground with an interval of ten minutes or
less. Stop acceptance if there is an OOM, a nonfinite value, a failed adapter
load, a transport failure, an environment failure, or a measured reward
regression.

A short run, step 1, or a reward change inside the noise does not pass.
