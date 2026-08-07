# Patches

`glm52-fixes-all.patch` is the diff from the initial deployment commit for
`configs/`, `runtime/`, `scripts/`, and `source/`.
`glm52-source-only.patch` contains the corresponding `source/` diff.

## Prime-RL

- `prime-rl/glm52-fp8.patch` is the base GLM-5.2 FP8 change set against the
  pinned Prime-RL commit in `../PINS.md`.
- `prime-rl/build-deepep-fused-score.patch` is the small DeepEP slice applied
  by the Dockerfile when building from the existing base image. It is already
  included in the base patch.
- `prime-rl/fp8-expert-backward-memory.patch` releases FP8 expert-backward
  temporaries before allocating the second input-gradient contribution.
- `prime-rl/fp8-hybrid-cp-backward.patch` separates FP8 embedding and output-head
  selection, and keeps completion-free context slices in the backward graph so
  every context-parallel rank enters the same sparse-attention collectives.
- `prime-rl/ada-sparse-parity.patch` keeps GLM sparse attention enabled on RTX
  4090 trainers so their policy log-probs match the H200 rollout model instead
  of silently substituting dense causal attention.
- `prime-rl/ada-sparse-forward.patch` uses the actual Ada head-block width when
  splitting sparse MLA across thread blocks, preventing out-of-range head
  reads and NaN policy log-probs.
- `prime-rl/ada-sparse-backward.patch` keeps the valid 64-thread tensor-core
  layout on RTX 4090 and uses scalar gradient atomics there, avoiding TileLang's
  invalid vector layout while preserving the H200 vectorized path.
- `prime-rl/fp8-indexer-wk-parity.patch` dequantizes sparse-indexer key
  projections to BF16 in the trainer, matching vLLM's GLM path so discrete
  sparse token selection is based on the same numeric inputs.
- `prime-rl/reward-and-router-replay.patch` normalizes within-group advantages and keeps distributed padding rows out of inference route replay.

## Renderers

- `renderers/tokenizer-pool-init.patch` applies to the bundled renderers
  dependency. It keeps tokenizer loading lazy for the orchestrator worker pool.
- `renderers/disconnected-generate-retry.patch` retries policy response disconnects, including disconnects while reading the body.

## Verifiers

- `verifiers/disconnected-provider-retry.patch` retries simulated-user and judge disconnects, including disconnects while reading the body.
