# Training requirements

Read this file before each trainer qualification or production run. These are
acceptance requirements. Do not change them without an owner decision.

Use trainer-only fake data to debug trainer execution. Use deterministic Hugging
Face output to check log probabilities. Start rollout production only after
these direct checks pass.

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
