# Version pins

## Current deployment source

- Prime-RL base commit: `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- Trainer base image: `prime-rl:glm52-decensor`
- Muse inference image:
  `vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d`
- Python in the trainer image: 3.12
- PyTorch in the measured trainer image: `2.11.0+cu128`

The Muse launcher uses the immutable image name above. It defaults to the
Transformers model implementation and full CUDA graphs for decode batches
only. Prompt and mixed batches do not use a full graph.

## GLM model

- `zai-org/GLM-5.2-FP8` revision:
  `ba978f7d347eaf65d22f1a86833408afdb953541`
- China mirror: `ZhipuAI/GLM-5.2-FP8` on ModelScope

## Environment data

- `NewEden/decensor-v1` revision:
  `5ffbb76517e2415241536698a4ff55f73bbcb534`
- `NewEden/RL-seed-Decensor` revision:
  `de87870ad0228f9253d5a5e8ac4cb968901a064e`
