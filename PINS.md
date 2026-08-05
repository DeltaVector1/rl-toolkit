# Pinned versions

## Source

- Prime-RL commit: `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- Base image: `ghcr.io/primeintellect-ai/prime-rl:v0.7.1.dev65-amd64`
- Trainer image on every 4090 node:
  `sha256:7d75ba7215c070a1c6d1e1a4ab93b62c48518ff8eb762b85fc36d0c2a4da863b`
- H200 rollout image:
  `sha256:0c8764d53791b651fbf9b028082939eafb80071219a2c2540dbed6480b679fc7`
- H200 rollout hosts: `152.236.142.243:40051` and `152.236.142.239:40050`
- Python: 3.12
- PyTorch: `2.11.0+cu128`
- Transformers: `5.6.2`
- vLLM: `0.24.0`
- Verifiers: `0.2.2.dev17`
- SwanLab: `0.9.0`

## Model and environment revisions

- `zai-org/GLM-5.2-FP8`: `ba978f7d347eaf65d22f1a86833408afdb953541`
- China mirror: `ZhipuAI/GLM-5.2-FP8` on ModelScope
- `NewEden/decensor-v1`: `5ffbb76517e2415241536698a4ff55f73bbcb534`
- `NewEden/RL-seed-Decensor`: `de87870ad0228f9253d5a5e8ac4cb968901a064e`

The patches under `patches/` are based on the Prime-RL commit above.

Runtime patch SHA-256 values:

- `reward-and-router-replay.patch`: `e4324d63f40e08939a1f99a8b15a49e038ffb46d0db06ca8deda123cb740b961`
- `disconnected-generate-retry.patch`: `513ca2bbccb97b9f5c57276b3f2827b147eb7d10d36ae239e7b6d25a832dff00`
- `disconnected-provider-retry.patch`: `bfc0c4234cb29efeae5f36a011f8901194aef4af35e7f2e029342d7042cfc2c5`
- `fp8-hybrid-cp-backward.patch`: `e5eac3c674485e132b6459daf9c891860dc69d7329bb4495d6759067d8919992`
- `ada-sparse-parity.patch`: `8e9d6d19698bcfb9669e64c302b182997981facf92179bef0ac52ccb5700db1a`
- `ada-sparse-forward.patch`: `35976caa86f320171c0f982b22bc9ec4c597bcc663eed5d79802c08785d21fcf`
- `ada-sparse-backward.patch`: `bb5cfa46e1a31a636c7d5c613fd8a9e5f334cf0161a35596bc65e7c45790b9a3`
- `fp8-indexer-wk-parity.patch`: `3f5c8ebe103e7276619af70596b73cf3f33830f2e8c43d81518d15c84126d68a`

## FP8 run

- `patches/prime-rl/glm52-fp8.patch` contains the base Prime-RL source change
  set used by the 64-GPU FP8 run; the additive patches above and the exact
  files under `source/` complete r33.
  - SHA-256: `a888463da0ec31397c01a7fe857544bc6a6da8f986b5b82b4d462c93fef287e0`.
- It applies cleanly to `f50ec5c4c8c45864ec535dead2c3b630de50d068`.
- Trainer: 64 RTX 4090 GPUs, block-FP8 frozen base, rank-32/alpha-64 LoRA,
  CP 4, global EP 8, eight data replicas, no CPU or optimizer offload.
- Rollouts: two 8-GPU H200 nodes, one TP8/EP8 vLLM engine per node, FP8 KV
  cache, and CUDA graphs.
- The token embedding and output head remain BF16 to match the H200 model.
- The trainer uses GLM sparse attention and the checkpoint's IndexShare
  schedule on Ada instead of substituting dense causal attention.
- User and judge: four TP2 Gemma 4 26B servers on n85.

## Exact r33 runtime source

- `dispatcher.py`: `98e6a0c797e69c467501b6775978d31cec322bcfe41039b8f3abdbad2e40a21b`
- `orchestrator.py`: `471b2eb940c674c45e16de1319dde73926d7b13f216c93e78e207921c1ce1014`
- `train_source.py`: `4d75d25fa10797d02243f61e2a87ce3042254379658d97d1e6cc4c55975fa2f9`
- `watcher.py`: `7435fd82970264b6be3bd3858da600a5bd872aca9944bddd751b8fcdb6688e95`
- `utils/client.py`: `a6bded395f7ddf15274fffbfefda899b33bafd6572443f340ae9e91e706a2af3`
- `renderers/base.py`: `f990822a59bd7c2868f3603416bd742150e7be4b1ba6c1d48a97ac8670bfdf7a`
- `renderers/client.py`: `b4f47f50ad100223bb33fa1729833f0464044a35b158de7a9832621ab0f9e323`
- `verifiers/v1/clients/eval.py`: `3e14fa33cf647f800bebff1c9e68b02e283eb88dba7a4d66645d2b3d55b21872`

The trainer image carries a two-line PyTorch 2.11 FSDP fix so frozen FP8
parameters preserve `requires_grad = false` in their sharded and post-forward
wrappers.

RTX 4090 expert and dense FP8 operations use the vLLM Triton block-FP8 kernels.
The H200 engines use the native H200 FP8 path.
