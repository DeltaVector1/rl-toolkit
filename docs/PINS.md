# Pinned versions

## Source

- Prime-RL commit: `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- Base image: `ghcr.io/primeintellect-ai/prime-rl:v0.7.1.dev65-amd64`
- Historical r33 trainer image:
  `sha256:7d75ba7215c070a1c6d1e1a4ab93b62c48518ff8eb762b85fc36d0c2a4da863b`
  Live x25 uses node-local image IDs and is not reproducibly pinned here.
- Live H200 rollout images measured on 2026-08-09:
  - h200a: `sha256:17423e0a392fdb74634aaf9c5f9b6b49d71f59d6b843c698021f4a835dd8d142`
  - h200b: `sha256:0c8764d53791b651fbf9b028082939eafb80071219a2c2540dbed6480b679fc7`
  The tag is `prime-rl:glm52-decensor` on both hosts, but the image IDs differ.
- H200 rollout hosts: `152.236.142.243:40051` and `152.236.142.239:40050`
- Python: 3.12
- PyTorch: `2.11.0+cu128`
- Transformers: `5.6.2`
- vLLM: `0.24.0`
- Verifiers: `0.2.2.dev17`
- SwanLab: `0.9.0`

MTP is disabled for the GLM-5.2 production profile. The isolated qualification
uses the vLLM slot-mapping fixes from PRs #46908 and #47013 and bounded
FlashMLA workspace chunking. Do not enable MTP in production until the
qualification passes one-token decode, trainer/H200 parity, and long-run
memory checks.

## Model and environment revisions

- `zai-org/GLM-5.2-FP8`: `ba978f7d347eaf65d22f1a86833408afdb953541`
- China mirror: `ZhipuAI/GLM-5.2-FP8` on ModelScope
- `NewEden/decensor-v1`: `5ffbb76517e2415241536698a4ff55f73bbcb534`
- `NewEden/RL-seed-Decensor`: `de87870ad0228f9253d5a5e8ac4cb968901a064e`

The patches under `patches/` are based on the Prime-RL commit above.

Runtime patch SHA-256 values:

- `reward-and-router-replay.patch`: `e4324d63f40e08939a1f99a8b15a49e038ffb46d0db06ca8deda123cb740b961`
- `disconnected-generate-retry.patch`: `5acc4fe4f649600275bb1fd03127d5bfb972bdd0d016387bb2d716eb18320683`
- `disconnected-provider-retry.patch`: `c8927a5310a6182ea776f46627386a2d3774c6f72c99354061b2ecfb456c7754`
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
- Historical r33 trainer: 64 RTX 4090 GPUs, block-FP8 frozen base,
  rank-32/alpha-64 LoRA, CP 4, global EP 8, and eight data replicas.
- Current x56p trainer: native block-FP8 frozen weights, BF16 activations
  and reductions, FP32 router, PP8, CP8, and node-local EP8.
- Rollouts: two 8-GPU H200 nodes, one TP8/EP8 vLLM engine per node,
  serialized block-FP8 weights, BF16 layer I/O, FP8 KV cache, and CUDA graphs.
- The token embedding and output head remain BF16 to match the H200 model.
- The trainer uses GLM sparse attention and the checkpoint's IndexShare
  schedule on Ada instead of substituting dense causal attention.
- Historical r33 user and judge: four TP2 Gemma 4 26B servers on n85.

## Exact r33 runtime source

- `dispatcher.py`: `98e6a0c797e69c467501b6775978d31cec322bcfe41039b8f3abdbad2e40a21b`
- `orchestrator.py`: `471b2eb940c674c45e16de1319dde73926d7b13f216c93e78e207921c1ce1014`
- `train_source.py`: `4d75d25fa10797d02243f61e2a87ce3042254379658d97d1e6cc4c55975fa2f9`
- `watcher.py`: `7435fd82970264b6be3bd3858da600a5bd872aca9944bddd751b8fcdb6688e95`
- `utils/client.py`: `a6bded395f7ddf15274fffbfefda899b33bafd6572443f340ae9e91e706a2af3`
- `renderers/base.py`: `5cfb4afd947cff82fb11b2a48c78e225f9342f58a3bc5b7d8336f5779e0719ea`
- `renderers/client.py`: `ca6093a9ccc1e0f3be7322352cc0cece67efcaa1ac1f0613877cf7ad3d8134bc`
- `verifiers/v1/clients/eval.py`: `95e03906aaffc41836fed976e761d3b3bf62cddfaf9082c4d0a07e24de4f27da`

The trainer image carries a two-line PyTorch 2.11 FSDP fix so frozen FP8
parameters preserve `requires_grad = false` in their sharded and post-forward
wrappers.

## Current external runtime overrides

- `source/src/prime_rl/inference/vllm/worker/filesystem.py`:
  `40a62de9e2c1f13fc17034d0a2f4d6292c6dd8607d72bb313a98fbdf9361df34`
- `source/src/prime_rl/inference/vllm/server.py`:
  `4d0bee6961144b6a9d1d1248d530dd4f8d2f8887846facad2bfb88b4d1e3f44a`
- `scripts/h200/sync_rollout_batches.sh`:
  `c6e3893f4425d9a322f712dc0f16fb1c5abcff3e24363c5d554bdd128979d634`
- `scripts/china/sync_microbatches.sh`:
  `2534a8fcece653899805d864068d63f1dc3904a16b036578d907ed023391ea7f`

The current RTX 4090 trainer uses the native block-FP8 checkpoint path with
custom Ada kernels. Both H200 engines report vLLM 0.24.0 and PyTorch
2.11.0+cu128. They resolve `GlmMoeDsaForCausalLM`, FP8 quantization,
`CutlassFp8BlockScaledMMKernel`, `FLASHINFER_CUTLASS` FP8 MoE,
`FLASHMLA_SPARSE`, and the `fp8_ds_mla` KV-cache format. Both use TP8 with EP8,
which gives MoE TP1/EP8. Router logits and the LM head are FP32.
