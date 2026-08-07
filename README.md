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

There is no active run. The latest test is x26 in
`/data/glm52-rl/runs/x26`. It used the edge orchestrator and durable filesystem
transport. It was stopped after two rollout batches because the trainer failed
twice during its first forward pass.

## Current settings

- Source model: `zai-org/GLM-5.2-FP8`
- Environment: `NewEden/decensor-v1`
- Dataset: `NewEden/RL-seed-Decensor`
- Batch 1024, GRPO group 16, maximum 128 live rollouts
- Context and trainer pack length 16,384 tokens
- LoRA rank 32, alpha 64, on `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, and `o_proj`
- Muon at 8e-6
- Trainer mesh: PP 8, CP 8, EP 8 node-local
- Checkpoints every 5 steps, with optimizer state
- Router replay off and engine routed-expert return off for x25
- H200 FP8 weights with BF16 activations (W8A16), FP8 KV cache
- Trainer checkpoint dequantized to BF16 weights, BF16 compute and reductions,
  FP32 MoE router

“FP8 model” identifies the source checkpoint. It does not mean that the current
trainer runs FP8 GEMMs. The native trainer W8A8 path produced 7 to 9 nats of
masked mismatch KL and broke the importance ratio, so x25 sets
`dequantize_checkpoint = true`.

## Latest result

x26 rollout step 1 took 20 minutes 5 seconds. Step 2 took 18 minutes 41
seconds. Keep these rollout collection times separate from trainer compute.
The two immutable batch files were 84.2 MB and 92.5 MB. Each transfer to n82
finished in about 29 seconds, and both SHA-256 hashes matched.

The trainer did not complete a step. On two attempts, n80 rank 35 exited about
44 seconds after the forward pass started. The other ranks then reported NCCL
remote-process errors. All trainer nodes read the same verified micro-batch
file on the second attempt. This result does not identify the reason that rank
35 exited.

x25 step 1 collected 1,072 attempts in 48 minutes 40 seconds, with 1,032
trainable attempts and a 3.7 percent error rate. The trainer forward and
backward pass took 17 minutes 11 seconds. Keep these as separate numbers.

The first no-router-replay step measured masked mismatch KL 0.362. The replay
baseline was 0.19 to 0.21. This is above the baseline but far below the failed
W8A8 range of 7 to 9 nats. One step is not a reward trend.

The stopped x25 path used 48 reverse SSH tunnels on n82 ports 18001 to
18048. The x26 test put the orchestrator on h200a and moved completed immutable
batches to n82 with resumable, verified transfer. `docs/ROLLOUT_TRANSPORT.md`
gives the design.

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
