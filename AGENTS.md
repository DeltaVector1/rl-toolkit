# GLM-5.2 decensor RL

This file is loaded into every session. It is deliberately short. It gives the
topology, the rules, and pointers. It does not repeat the runbook or the fault
table.

Language: ASD-STE100 Simplified Technical English.

# What the system is

The system trains from the GLM-5.2-FP8 checkpoint, a 745 billion parameter
MoE, with GRPO and a rank-32 LoRA adapter. The H200 engines use FP8 weights
with BF16 activations. The trainer dequantizes the checkpoint to BF16 weights.
The goal is to reduce refusals.

There are four parts. Keep them separate when you report anything.

The trainer computes gradients on 8 nodes of 8 RTX 4090, which is 64 GPUs.

The inference engines make the rollouts on 2 hosts of 8 H200, which is 16 GPUs.

The judge scores the rollouts on one external host of 8 RTX 5090.

The orchestrator joins them. It runs on n82.

The trainer step time and the rollout collection time are different numbers
with different causes, and they overlap. Never give one combined number.

# The hosts

| Host | Role |
|---|---|
| n76, n77, n78, n80, n83, n84 | Trainer worker, 8x RTX 4090 |
| n82 | Trainer worker, orchestrator, adapter server on port 8100 |
| n85 | Trainer worker. Also holds the only copy of the model |
| n79 | Not ours. Do not touch it. There is no exception |
| n81 | Healthy, but it holds no authorized key. Not in the job |
| h200a, h200b | Rollout engine, 8x H200 each |
| judge | External, 8x RTX 5090, port 26406 |

The trainer nodes are in China. The engines and the judge are outside China.
Measurements from different hosts and times found 3 to 33 percent packet loss
and approximately 217 milliseconds round-trip time. The engines reach n82
through 48 reverse SSH tunnels on ports 18001 to 18048. The current
orchestrator runs on n82, so individual rollout responses cross this link.

# How to operate the run

Use the `glm52-ops` skill. Do not write one-off SSH commands for work the skill
already does.

```python
glm52_ops.status()       # the whole picture in one call
glm52_ops.reward()       # the reward series with a trend verdict
glm52_ops.watch(hours=2) # the same picture on a loop, in the foreground
```

`.agents/skills/glm52-ops/SKILL.md` describes it.

Prime Agent loads project skills from `<repo>/.prime/agent/skills/`, so that
path holds a symlink to the same directory. There is one copy, reachable by
both names. Verified: the loader finds `glm52-ops` and reads it as a
Python-backed skill with the import name `glm52_ops`.
`references/FAULTS.md` gives the symptom, the cause, and the one repair for
each fault that happened.
`references/RULES.md` gives the rules below with their reasons.

# What you may do without asking

Read anything. Probe anything. Run `status`, `reward`, `tunnels`, `engines`,
`alive`, and `watch` freely. They do not change the run.

Repair a known fault without asking. `repair("tunnels")`,
`repair("engine:<host>")`, and `repair("supervisors")` are safe and are the
correct response to their symptoms.

Edit files in this repository and run the smoke tests without asking.

# What needs a decision first

Stopping a run, starting a new run name, or restarting the trainer. These lose
work in progress.

Any change to `batch_size`, `group_size`, `max_inflight_rollouts`, the
parallelism, or the learning rate. These change what the experiment measures.

Anything that writes outside this repository and the run directories.

# The rules

Do not touch n79.

Do not delete the data of the owner. The protected paths are `/data/ZiphuAI`,
`/data/trinity`, and `/data/nv-community_GLM-5.2-NVFP4`. The model at
`/data/glm52-rl/model` on n85 is the only copy. Secrets stay in
`/data/trinity/training.env`, outside this repository.

Do not use QLoRA. Do not use INT4 weights. The INT4 expert-activation
transport is separate and is allowed.

Do not use a cached batch. Every attempt is a full production run.

Do not use an automated monitor tool. Use a foreground command with a sleep in
it, which prints the GPU use, the generation time, the step count, and the
rollout count together. `glm52_ops.watch()` is that command. Nothing may sit
still for more than 10 minutes.

Do not use metaphors. Do not give an opinion that the owner did not ask for.

Measure before you write a conclusion. Do not present a theory as a fact.

Do not report a trend that is inside the noise. `reward()` returns a verdict.
Report the verdict.

# The current run

x37 is active. The trainers started at approximately 09:45 UTC on 2026-08-08.
The H200 orchestrator started at 09:53 UTC.

x37 uses the BF16-dequantized trainer path, PP8, CP8, node-local EP8, batch
1024, group 16, sequence length 12,288, learning rate 8e-6, and 128 rollouts
in flight. The H200 engines use their automatic W8A8 backend. Router replay is
off. Full trainer checkpoints are off because the frozen base exceeds the
remaining disks. Each completed step still exports the LoRA adapter.

The shorter sequence ceiling is a memory repair. x31 reached 46.6 GiB at
14,336 tokens and later OOMed with only 179 MiB free. Do not increase the x37
sequence length while it is active.

Native FP8 training is rejected. x32 had entropy 4.3957, masked mismatch KL
3.8317, and 48.8 percent masked tokens. x35 route replay still had KL 3.1711.
x36 forced H200 Triton MoE on a fresh batch-1024 gate and still had entropy
4.4206, masked mismatch KL 6.5098, unmasked mismatch KL 3.1729, and 48.2
percent masked tokens. Do not start another native-FP8 trainer run without a
new measured correction.

x36 also proved the 64-GPU native step can run at 2,178 tokens/s with 46.6 GiB
peak memory. This is a compute and memory result only. It is not policy parity.

# Where the rest is

| Document | Content |
|---|---|
| `docs/RUNBOOK.md` | The commands, and what to do when the skill cannot |
| `docs/HANDOFF.md` | Why the system is built this way, and the mistakes made |
| `docs/CHANGES.md` | Every change, with the reason and the patch files |
| `docs/PERFORMANCE.md` | The transport and all-to-all measurements |
| `PIPELINE_PARALLEL.md` | The pipeline stage balance |
| `PINS.md` | The pinned versions |

SwanLab is at `https://swanlab.cn/@MangyMangaur/prod-run`. Do not compare x22
and later against x21 and earlier. The adapter never reached the engines before
x22, so those rollouts came from the base model and their reward data is not
valid.
