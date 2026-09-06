# Version references

These values record the source used by the current files. They do not add a
new requirement to pin every package or dataset.

## Training code

- Prime-RL base: `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- Renderers base: `bdb96b0c84a307e2b71c6a366c9d718c3ac7fe78`
- Verifiers base: `295a0650585af8754eaf98c6fa9071314b5a531c`
- Measured trainer image: `prime-rl:glm52-fp8`
- Python in the measured trainer image: 3.12
- PyTorch in the measured trainer image: `2.11.0+cu128`
- NCCL in the measured trainer image: `2.28.9`
- Measured H200 image: `prime-rl:glm52-v080-glm52-h200-router`
- Measured H200 vLLM: `0.26`
- Orchestrator image: `prime-rl:glm52-decensor`
- Upstream `26b3131` was reviewed but is not upgraded.

The Dockerfile `COPY` is a separate image-build step. Deployment copies the
local `prime-rl/` overlay to configured host paths, and launchers bind-mount
files into `/app/src`; deployment does not modify an existing image. The local
directory is not a separate Git checkout. `prime-rl.patch` is the portable
copy of that overlay, refreshed against base
`f50ec5c4c8c45864ec535dead2c3b630de50d068` for four source files. Its other 47
blocks, including dependency blocks, are unchanged. Source/package hunks
passed apply-check against this base with `deps/*` excluded. Dependency hunks
were preserved, not reapplied to fresh dependency trees. It is not applied by
the Dockerfile. This patch does not apply unchanged to reviewed upstream
`26b3131`; that upgrade is not deployed.

## Models

- GLM-5.2 FP8: `zai-org/GLM-5.2-FP8`, revision
  `ba978f7d347eaf65d22f1a86833408afdb953541`
- GLM-5.2 China mirror: `ZhipuAI/GLM-5.2-FP8` on ModelScope
- Refusal classifier: `Dans-DiscountModels/Dans-Classifier-RP-Validity-V1.0.0-396m`,
  revision `2084304ece320e87820140adfc8845761fb357a0`

## Datasets

- Training prompts: `NewEden/RL-seed-Decensor-Difficulty`
- Markdown system prompts: `Delta-Vector/Tauri-RL-Markdown-System-V2`
- Plain-text system prompts: `Delta-Vector/Tauri-RL-Plaintext-System-V2`
- Style prompts: `Delta-Vector/Tauri-RL-Styles`
- Refusal evaluation: `NewEden/refusals-classifier-eval`
