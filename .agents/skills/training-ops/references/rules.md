# Rules

- Never touch a host listed in `hosts.blocked` in `configs/operations.toml`.
- Never delete `/data/ZiphuAI`, `/data/trinity`, or
  `/data/nv-community_GLM-5.2-NVFP4`.
- Never delete the only GLM model at `/data/glm52-rl/model` on n85.
- Keep secrets in `/data/trinity/training.env`.
- Do not use QLoRA or INT4 model weights.
- Do not use a cached rollout batch.
- Run monitoring in the foreground.
- Do not leave the run without a progress check for more than ten minutes.
- Report trainer compute, rollout collection, adapter transfer, and waiting as
  separate times.
- Report the reward verdict from `training_ops.reward()`.
- Measure before giving a cause.
- Starting, stopping, or changing an experiment needs an owner decision.
