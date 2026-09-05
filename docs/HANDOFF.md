# Current handoff

Updated 2026-09-05 UTC. Use `configs/operations.toml` for every host, path,
port, container, and profile value.

## Active run

- Run: `glm-1`.
- Profile: `current`.
- Dataset: `NewEden/RL-seed-Decensor-Difficulty`.
- Current stage: `easy` prompts, balanced across S1 through S14.
- Target: verify at least 20 complete steps without an OOM, nonfinite value,
  transport failure, service failure, or large trainer/inference mismatch.
- SwanLab: `https://swanlab.cn/@MangyMangaur/glm-runs/runs/ncugtehv`.

The profile has a 200-step limit, but its difficulty is static. It does not
change from easy to medium or hard by itself.

## First measurements

- Rollout step 1: 10m22s, reward 0.2198, 1,024 of 1,040 usable.
- Rollout step 2: 9m45s, reward 0.1838, 1,024 of 1,040 usable.
- Rollout step 3: 8m22s, reward 0.2270, 1,024 of 1,024 usable.
- Trainer step 1 forward/backward: 7m59s.
- Trainer step 1 throughput: 2,767 tokens/s.
- Trainer step 1 adapter save and broadcast: 3m48s.
- Trainer step 1 encoded adapter delta: 8,881,402,367 bytes from a
  30,767,549,784-byte BF16 adapter.
- Trainer step 1 peak GPU memory: 43.1 GiB.
- Trainer step 1 entropy: 0.4301.
- Trainer step 1 gradient norm: 0.0378.
- Trainer step 1 selected-token mismatch value: 0.0109.

Trainer work and rollout work overlap. Report rollout, forward/backward,
adapter save, network transfer, H200 activation, judging, and full model copy
as separate values.

## Training design

- Native FP8 base model.
- Rank-32 BF16 LoRA, alpha 64.
- Attention, shared-expert, and routed-expert linear weights are trained.
- The router stays frozen.
- PP8, CP8, and node-local EP8 use 64 RTX 4090 GPUs.
- Two TP8 H200 servers generate rollouts with CUDA graphs and expert
  parallelism.
- Batch 1,024, group 16, context 32,768, output limit 16,384, and learning
  rate `8e-6`.
- The orchestrator uses standard GRPO.

## Local work not loaded by the active run

- Removed completed seed-review commands and the old 4090 rollout-tunnel
  system.
- Fixed judge results so an invalid model reply and a service failure remain
  different unscored states.
- Removed broad exception handling that hid programming faults.
- Added bounded retries to the trainer dependency relay.
- Made cold starts wait for the orchestrator control file before starting the
  trainer.
- Removed unused environment functions and stale documentation.
- Restored the trainer-side LoRA layer check needed by the latest source.

Do not deploy these local environment changes into `glm-1` while it is healthy.
They should enter the next run after local checks and this 20-step test finish.

## Checks

- Training-ops tests: 61 passed.
- Python compilation: passed.
- Undefined-name check: passed.
- Shell syntax: passed.
- Diff whitespace check: passed.

## Next work

1. Monitor `glm-1` through trainer step 20.
2. Check reward only after at least 10 steps. Do not call movement inside the
   measured noise a trend.
3. Record forward/backward, adapter save, network transfer, H200 activation,
   rollout, reward, memory, and mismatch values.
4. Decide the medium and hard stage schedule before the easy stage ends.
