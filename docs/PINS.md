# Version references

These values record the source used by the current files. They do not add a
new requirement to pin every package or dataset.

## Training code

- Prime-RL base: `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- Renderers base: `bdb96b0c84a307e2b71c6a366c9d718c3ac7fe78`
- Verifiers base: `295a0650585af8754eaf98c6fa9071314b5a531c`
- Trainer image: `prime-rl:glm52-decensor`
- Python in the measured trainer image: 3.12
- PyTorch in the measured trainer image: `2.11.0+cu128`

The Docker build applies the repository overlay and `prime-rl.patch` to that
base. The local `prime-rl/` directory is not a separate Git checkout.

## Models

- GLM-5.2 FP8: `zai-org/GLM-5.2-FP8`, revision
  `ba978f7d347eaf65d22f1a86833408afdb953541`
- GLM-5.2 China mirror: `ZhipuAI/GLM-5.2-FP8` on ModelScope
- GLM-5.3 judge: `zai-org/GLM-5.3-Flash`, revision
  `690b705278a3a58e538fcb37c2ca8b5f9511213c`
- GLM-5.3 SGLang image:
  `lmsysorg/sglang@sha256:d246b29aa6543ab28859439d4143bbf553b44cc74b29a98f4ee8768b401de252`
- Refusal classifier: `Dans-DiscountModels/Dans-Classifier-RP-Validity-V1.0.0-396m`,
  revision `2084304ece320e87820140adfc8845761fb357a0`

## Datasets

- Decensor environment: `NewEden/decensor-v1`, recorded revision
  `5ffbb76517e2415241536698a4ff55f73bbcb534`
- Seed prompts: `NewEden/RL-seed-Decensor`, revision
  `de87870ad0228f9253d5a5e8ac4cb968901a064e`
- Refusal evaluation: `NewEden/refusals-classifier-eval`

The active training file selects `NewEden/decensor-v1` through taskset ID
`decensor-v1` but does not state the revision. The seed-prompt operation does
state its revision in `configs/operations.toml`.
