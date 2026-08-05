# GLM-5.2 decensor RL

This folder contains the files used by the GLM-5.2 FP8 rank-32 LoRA run.

## Live layout

- Trainer: n76, n77, n78, n80, n81, n82, n83, and n84. Each node contributes
  eight RTX 4090 GPUs, for 64 trainer GPUs total.
- User and judge: four TP2 Gemma 4 26B servers on n85, behind port 40002.
- Rollouts: two eight-GPU H200 hosts. Each host runs one TP8/EP8 vLLM engine.
- Orchestrator and adapter file server: n82.
- n79 is not used.

The active output folder is:

```text
/data/glm52-rl/runs/r33/run_decensor_fp8_prod_run_r33
```

## Settings

- Model: `zai-org/GLM-5.2-FP8`
- Environment: `NewEden/decensor-v1`
- Dataset: `NewEden/RL-seed-Decensor`
- LoRA: rank 32, alpha 64, attention projections only
- Context and trainer pack length: 4,096 tokens
- Batch: 88 rollouts
- GRPO group: 8
- Advantages: group mean centered and divided by group standard deviation
- Maximum off-policy age: 1 optimizer step
- Router replay: enabled; distributed padding rows use trainer routing
- Attention: GLM sparse attention and IndexShare on both trainer and rollout models
- Token embedding: original BF16 weights
- Output head: original BF16 weights
- Maximum live rollouts: 88
- Minimum policy completion: 16 tokens
- Optimizer: Muon
- Learning rate: 8e-6
- Weight decay: 0
- Gradient limit: 1.0
- Trainer mesh: CP 4, global EP 8, eight data replicas
- CPU and optimizer offload: disabled
- Checkpoint interval: 25 optimizer steps
- Rollout KV cache: FP8
- CUDA graphs: enabled on both H200 rollout engines and the n85 Gemma servers

## Trainer speed settings

The trainer is bound by the MoE all-to-all on one 10 Gb/s link per node, not by
compute. See `AGENTS.md` for the measured breakdown. Two settings control this:

- `NCCL_MIN_NCHANNELS=8` and `NCCL_MAX_NCHANNELS=8`. The largest single knob:
  NCCL's automatic choice reaches only about 44% of line rate on this link,
  eight channels reach 87%. One channel hangs; do not set it.
- `fq` instead of `fq_codel` on the trainer NIC, applied by `cluster.sh` before
  the container starts. Worth 65% against 87% once channels are right. BBR was
  tested and is worse here; stay on `cubic`.
- `NCCL_SOCKET_NTHREADS=1` and `NCCL_NSOCKS_PERTHREAD=1`. The previous 4 x 4
  opened 128 TCP flows per node on one NIC and congested it.
- `PRIME_EP_AC_CACHE=1` reuses the INT4 payload received in the forward pass
  when activation checkpointing recomputes the routed-expert region, instead of
  repeating both all-to-alls. Numerically exact; costs about 3.8 GB per GPU.

`PRIME_EP_PROFILE=1` prints a per-micro-batch all-to-all breakdown from rank 0.
`PRIME_EP_PROFILE_EVERY` sets how many dispatch calls per report (150 = one
micro batch). Watch `fwd_dispatch` to confirm the cache is live: with the cache
off it covers two passes per layer, with it on, one.

`scripts/china/check_ep_ac_cache.py` verifies the cache on one node's eight
GPUs. It counts real `all_to_all_single` calls and compares outputs and
gradients against the uncached path, so it fails if the cache silently stops
engaging:

```bash
torchrun --nnodes=1 --nproc-per-node=8 check_ep_ac_cache.py
```

Launcher variables `RUN`, `RUN_NAME`, and `CONFIG_TAG` select the run directory
and the `configs/china/{trainer,orchestrator}${CONFIG_TAG}.toml` pair, so a new
run does not need edits to the committed r33 configs.


## Layout

```text
configs/
  china/       orchestrator and 4090 trainer settings
  h200/        H200 rollout settings
env/           decensor environment
network/       SSH and proxy settings
patches/       source backup patches
runtime/       runtime file overrides
scripts/
  china/       4090 cluster and Gemma launcher
  h200/        H200 rollout and tunnel launchers
  network/     rollout link helpers
source/        exact patched source used to build the image
```

Deploy this folder as `/data/glm52-rl/deployment`. Secrets stay outside the
repository in `/data/trinity/training.env` on the GPU nodes.

## Rebuilding

The base Prime-RL source change set is:

```text
patches/prime-rl/glm52-fp8.patch
```

It applies to the Prime-RL commit recorded in `PINS.md`. Apply the additive
kernel, FP8 parity, reward, renderer, and verifier patches listed in
`patches/README.md`; the live r33 deployment includes all three Ada
sparse-attention patches and the BF16 sparse-indexer key projection fix. The
Dockerfile also applies the PyTorch FSDP fix needed to preserve frozen FP8
parameters.

## Launchers

- `scripts/china/cluster.sh trainer` starts one eight-rank trainer worker on a
  4090 node. All eight trainer nodes must start it with the same files.
- `scripts/china/cluster.sh orchestrator` starts the orchestrator on n82.
- `scripts/china/cluster.sh gemma` starts the four Gemma servers and proxy on
  n85.
- `scripts/h200/start_inference.sh` starts one TP8/EP8 rollout engine on an H200
  host.
- `scripts/h200/start_tunnel.sh` creates one rollout data tunnel back to n82.
