# GLM-5.2 decensor RL

This repository contains the deployment files for a GLM-5.2 rank-32 LoRA GRPO
run. `AGENTS.md` gives the current operating constraints. `docs/RUNBOOK.md`
gives commands. `docs/PERFORMANCE.md` contains dated measurements.

## Current layout

- Trainer: n76, n77, n78, n80, n82, n83, n84, and n85. Each has eight RTX
  4090 GPUs.
- Rollouts: h200a and h200b. Each host runs one TP8/EP8 vLLM engine on eight
  H200 GPUs.
- Judge: one external host with eight RTX 5090 GPUs, reached through port
  26406.
- Latest test orchestrator: h200a. Adapter server: n82.
- n81 is not in the job because it has no authorized key. Do not touch n79.

x56p is the active native-FP8 production run. Its trainer data is in
`/data/glm52-rl/runs/x56p`. The edge orchestrator runs on h200a and uses the
durable filesystem transport.

## Current settings

- Source model: `zai-org/GLM-5.2-FP8`
- Environment: `NewEden/decensor-v1`
- Dataset: `NewEden/RL-seed-Decensor`
- Batch 1024, GRPO group 16, maximum 128 live rollouts
- Trainer sequence length 2,048 tokens
- LoRA rank 32, alpha 64, on `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, and `o_proj`
- Muon at 8e-6
- Trainer mesh: PP 8, CP 8, EP 8 node-local
- Full native-FP8 frozen weights, BF16 activations and reductions, FP32 router
- Adapter export after every completed step; no full frozen-base checkpoints
- Router replay off
- H200 serialized FP8 weights, BF16 layer I/O, and FP8 KV cache
- Selective activation checkpoints on routed experts and norms only

## Latest result

Zero-learning-rate gate x55a completed one full native-FP8 trainer step with
finite gradients. It processed 913 microbatches in 3,356.38 seconds. Grad norm
was 0.015471, masked mismatch KL was 0.357186, unmasked mismatch KL was
0.037908, and 3.171 percent of tokens were masked. Throughput was 556 tokens/s
and peak GPU memory was 46.55 GiB.

The x55a rollout batch took 19 minutes 13 seconds. It had zero environment
errors. Keep this rollout collection time separate from trainer compute.
Step 1 exported a 278,990,560-byte LoRA adapter and a `STABLE` marker.

x56p started from a new policy after this gate. It uses learning rate 8e-6.
No reward trend is available yet.

## Repository layout

```text
configs/       trainer, orchestrator, and inference settings
docs/          runbook, handoff, measurements, and transport design
env/           decensor environment
network/       SSH and proxy settings
patches/       patch artifacts
runtime/       runtime file overrides
scripts/       launch, repair, validation, and transport scripts
source/        exact patched source used by the images
```

Deploy the repository as `/data/glm52-rl/deployment`. Secrets stay in
`/data/trinity/training.env`.

## Validation

```bash
bash -n scripts/china/*.sh scripts/h200/*.sh scripts/network/*.sh
scripts/ops/watchdog_smoketest.sh
python3 -m py_compile runtime/vllm/*.py source/deps/renderers/renderers/*.py
```

Do not deploy a trainer source change unless `scripts/china/cluster.sh` mounts
that file. Use `glm52_ops.mounted_files()` to check.
