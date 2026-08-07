# GLM-5.2 decensor RL

This file is loaded into every session. It is deliberately short. It gives the
topology, the rules, and pointers. It does not repeat the runbook or the fault
table.

Language: ASD-STE100 Simplified Technical English.

# What the system is

The system trains GLM-5.2-FP8, a 745 billion parameter MoE, with GRPO
reinforcement learning and a rank-32 LoRA adapter. The goal is to reduce
refusals.

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
The link between them drops 10 to 33 percent of the packets and adds 217
milliseconds. The engines reach n82 through 24 reverse SSH tunnels on ports
18001 to 18024. The batch itself never crosses that link, because the
orchestrator runs on n82.

# How to operate the run

Use the `glm52-ops` skill. Do not write one-off SSH commands for work the skill
already does.

```python
glm52_ops.status()       # the whole picture in one call
glm52_ops.reward()       # the reward series with a trend verdict
glm52_ops.watch(hours=2) # the same picture on a loop, in the foreground
```

`.agents/skills/glm52-ops/SKILL.md` describes it.
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

Do not use QLoRA. Do not use int4 weights.

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

Run x24. Batch 1024, group size 16, 256 rollouts in flight, 16K context, bf16
trainer weights, `pp=8`, `cp=8`, `ep=8` node-local, LoRA rank 32, Muon at 8e-6.

Measured on step 3: 31 minutes for the trainer step, 29 minutes for the rollout
collection, 2645 tokens per second, 3.2 percent MFU, 46.5 GiB peak of 48.

The success criterion is that the reward rises, or that the reward distribution
rises. Nothing else counts. The target is a 15 minute step time.

The open question is whether batch 1024 gives a smaller step-to-step spread
than the 0.039 that run x22 gave at batch 256. That needs 10 to 15 clean steps.

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
