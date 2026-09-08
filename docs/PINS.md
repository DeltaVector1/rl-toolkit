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
- Latest upstream fetched: `04a61d3b75c3c99f263b2c133e822f998909adf7`
  (2026-09-07T11:20:40-07:00). Versus reviewed `26b3131`, it has exactly one
  change: a Muon NCCL all-to-all startup warm-up in the new
  `trainer/optim/__init__.py`. No other trainer, GRPO, transport, or verifier
  changes were found. The current base uses monolithic `trainer/optim.py`, so the full
  patch does not apply unchanged. The latest upstream is not deployed.

The Dockerfile `COPY` is a separate image-build step. Deployment copies the
local `prime-rl/` overlay to configured host paths, and launchers bind-mount
files into `/app/src`; deployment does not modify an existing image. The local
directory is not a separate Git checkout. `prime-rl.patch` is the portable
copy of that overlay, refreshed against base
`f50ec5c4c8c45864ec535dead2c3b630de50d068`. The latest local refresh is
`3759+sentinel`: two source blocks changed and the other 49 are unchanged. The
current apply-check covers those two blocks; the previous full source/package
apply-check is historical. Dependency blocks are preserved and were not
reapplied to fresh dependency trees. It is not applied by the Dockerfile. This
patch does not apply unchanged to reviewed upstream `26b3131`; that upgrade is
not deployed.

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

## General-agent

- Package version: `0.1.4`
- Official dataset repository revision: `e3f0ec777c02b734ad1a7d246cf06a709dc3e6e2`
- The draft rollout limit is 720 seconds. Completion-budget change `92f06f4c`
  passed five actual-image CPU tests. The owner-approved
  `/data/glm52-rl/runs/general-agent-check/budget-probe` repair is complete
  with owner `1000:1000` and mode `755`.
- A two-rollout `game_balance_t4` smoke finished at 11:01:56 UTC in
  174.521595 seconds. Generation took 173.79682 and 174.38272 seconds; rewards
  were 0 and 1, errors were null, and the 16,384-token outputs stopped at the
  cumulative budget after 4/5 turns. This is not a natural-completion or
  production result.

## Qualification evidence

- The 20-step LR0 synthetic check passed with warm forward/backward time
  177.66-179.92 seconds (median 178.39) and 42.9277 GiB peak memory for
  steps 3-20. The nonzero-LoRA two-step check reported 177.89248 seconds on
  step 2, 178.58255 seconds total, 42.6543 GiB peak memory, B norm 0.0922512
  after step 1, and A-gradient norm `4.75536e-5` on step 2.
- The one-round BF16 combine candidate finished at 05:09:24 UTC but was not
  selected: step 2 was 176.55202 seconds, only 0.75% faster, with estimated
  balanced scratch 672 MiB larger. Exact BF16 transport output/gradcheck
  passed in `8457f51` for balanced, uneven, and zero-peer cases.
- These are controlled checks. The 524,288-token synthetic run is not a
  three-minute production proof; the stopped real batch had 9,018,062 tokens.
  All trainers are stopped cleanly. Production rollout and adapter delivery
  remain unverified. Both H200 READY step-0 caches match. Worker `3759` is not
  deployed; deploy it and restart both workers before a fresh production run
  to avoid a WAN retransfer.
