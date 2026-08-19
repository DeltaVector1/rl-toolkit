# Decensor RL

This file is loaded into every session. It is deliberately short. It gives the
topology, the rules, and pointers. It does not repeat the runbook or the fault
table.

Language: ASD-STE100 Simplified Technical English.

Read `TRAINING_EXPECTATIONS.md` before each trainer qualification or production run.
Treat its configuration, performance, precision, and final 50-step stability
requirements as acceptance requirements. Do not use full rollout batches to
debug trainer execution.

# What the system is

The current run trains Muse-Glimmer-30B with GRPO and a rank-128 LoRA adapter.
It targets all supported attention and MLP linear weights. The trainer and the
H200 rollout engine use BF16 model weights. The GLM-5.2 run is paused.

There are four parts. Keep them separate when you report anything.

The trainer computes gradients on 8 nodes of 8 RTX 4090, which is 64 GPUs.

The inference engine makes the rollouts on one host of 8 H200, which is 8 GPUs.

The judge scores the rollouts on one external host of 8 RTX 5090.

The orchestrator joins them. The current filesystem run uses the edge
orchestrator on h200a. The trainer-side adapter server runs on n82.

The trainer step time and the rollout collection time are different numbers
with different causes, and they overlap. Never give one combined number.

# Weight transport warning

Warn the owner before each full-weight policy run. A Muse-Glimmer BF16 policy
checkpoint is 55.5 GiB. The measured transfer from the China trainer to both
H200 hosts took approximately 37 minutes for each policy version on the
current link. Report this transfer time separately from trainer forward and
backward time and rollout collection time. Do not start a full-weight run
without making this cost clear.

# The hosts

| Host | Role |
|---|---|
| n76, n77, n78, n80, n83, n84 | Trainer worker, 8x RTX 4090 |
| n82 | Trainer worker, adapter server on port 8100 |
| n85 | Trainer worker. Also holds the only copy of the model |
| n79 | Not ours. Do not touch it. There is no exception |
| n81 | Healthy, but it holds no authorized key. Not in the job |
| h200a | Active rollout host, 8x H200. It also runs the edge orchestrator |
| h200b | Previous rollout host. It is not in the current job |
| judge | External, 8x RTX 5090, port 26406 |

The trainer nodes are in China. The engines and the judge are outside China.
Measurements from different hosts and times found 3 to 33 percent packet loss
and approximately 217 milliseconds round-trip time. The active engine reaches
n82 through 48 reverse SSH tunnels on ports 18001 to 18048. The orchestrator
runs on h200a, so individual rollout responses stay on h200a. Completed batches
and adapter data cross the link to the trainer nodes.

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

muse16 is active. It resumed the Muse15 step-60 LoRA policy. The checkpoint is
adapter-only, so it did not restore optimizer state. The trainer started a new
Muon optimizer at trainer step 61. It uses batch 1024, group 16, sequence
length 65,536, learning rate 8e-6, and 256 rollouts in flight. It uses a
rank-128 LoRA adapter on all supported attention and MLP linear weights.

h200a serves eight independent TP1 replicas. The orchestrator addresses all
eight replicas directly and pins each GRPO group to one replica. This is data
parallel serving across all eight H200 GPUs. CUDA graphs use
`FULL_AND_PIECEWISE` mode. Graph capture used 0.38 GiB on each GPU. A controlled
rank-128 LoRA benchmark with 2,048 generated tokens per request measured 5,077,
7,041, and 8,703 output tokens per second at 128, 192, and 256 concurrent
requests. Use 256 concurrent requests and TP1 data parallel serving for this
model and workload. The engine context limit is 65,536 tokens.

The environment rollout deadline is 1,200 seconds. A wall-clock harness timeout
is terminal and does not enter the transient network retry path. This prevents
the unlimited retry that held muse12 rollout batch 14 at 1008 of 1024 for more
than 90 minutes.

Muse16 rollout batch 61 completed in 15 minutes 36 seconds. It sent 1,024
trainable rows to the trainer, reported 0.3 percent rollout errors, and had no
mixed policy versions. At 992 of 1,024 rows, the scheduler still held 245
rollouts in flight and had 107 completed rows buffered. This verifies the tail
refill change under production load. Trainer forward and backward for step 61
started at 06:59:55 UTC on 2026-08-19. Rollout batch 62 started at the same
time. Do not report trainer step 61 as complete until forward, backward, and
the optimizer update finish.

# Previous GLM run

x72p stopped during trainer step 39. It started at approximately 05:59 UTC on
2026-08-12 and completed trainer steps 1 through 38. The fail-closed
supervisor stopped the orchestrator and all remaining trainer workers.

x72p uses the full native-FP8 trainer path, PP8, CP8, node-local EP8, batch
1024, group 16, sequence length 32,768, learning rate 8e-6, and 128 rollouts
in flight. The H200 engines use their automatic W8A8 backend. Router replay is
off. x72p started a new LoRA policy and a new Muon optimizer.

At a policy update, the dispatcher now invalidates each incomplete live-policy
GRPO group and regenerates the complete group from the same prompt under the
new policy. Partial members do not enter the error path and policy versions do
not mix inside a group. The H200 pause endpoint uses engine-native abort before
the fused MoE LoRA collective. The rollout budget permits thirty-two complete
overflow groups under the same 128 request cap. This is a logical scheduling
allowance, not additional concurrency. Completed overflow carries into the
next batch. New adapters are prefetched on both H200 hosts before the policy
marker becomes visible. Generation continues during this WAN transfer; the
later engine pause only does the local in-place load.

The control adapter tunnel uses a bounded HTTP progress probe and rebuilds
only its SSH child after two misses. Parallel rollout-batch transfer keeps 16
resumable stripes but spaces their SSH handshakes by 250 milliseconds. A
staged policy update gives useful incomplete groups a bounded 60-second grace
and can defer the last eight groups of a batch for at most 300 seconds. Fresh
prompts stop during the grace. The strict restart still applies to each group
that remains incomplete, so policy versions do not mix. Batch-transfer stripes
retry after 60 seconds without I/O; transfers that continue to make progress
have no total deadline.

x72p trainer steps 1 through 38 had finite gradients. Steps 31 through 38 took
10 minutes 54 seconds to 13 minutes 11 seconds for forward and backward.
Rollout steps 1 through 39 had valid trainable batches. The measured reward
verdict through rollout step 39 is inside the noise: mean 0.0814, fitted change
-0.0047, and threshold +/-0.0463. The run did not reach the final 50-step
stability gate.

Trainer step 39 failed on n77 local rank 2 in the native-FP8 expert forward.
The W2 grouped output requested 24 MiB with 1.5 MiB physically free and 1.16
GiB reserved but unallocated. The in-place expert path now releases its
gathered input, copied FP8 buffers, W3 output, gated activation, and completed
W2 output at their last use. The output allocation also releases unused CUDA
cache and retries once after an allocation OOM. The 67-test source suite, the
production-shape exact SM89 EP gate, and an injected allocation-failure gate
pass. The fix is deployed to all eight trainer-node deployment directories.
Policy v3 validated prefetch in production: rollout collection
continued from 784 to 960 while both adapters crossed the WAN, and both
engines then loaded and resumed in two seconds. The trainer and both H200
adapter files had SHA-256
`3d9a4717c454e102e151205c413f5751c83a79a14c9ff4e9c63d974466af46e2`.
The first live 60-second group grace at policy v8 let two of 14 incomplete
groups finish before the strict boundary. It prevented 32 regenerations.

A one-token native-MTP rollout trial did not pass production stability. At
11:16:03 UTC, h200b terminated all tensor-parallel workers in the sparse FP8
MLA path with `Stride exceeds int32 limit: 2415919104`. The stride is the
32,768-token BF16 query after the eight-head MTP draft query is padded to 64
heads. MTP is disabled and a regression test keeps it out of the production
launcher. Both engines run the stable non-MTP path and load the step-13
adapter with SHA-256
`3c0f7b589a04c59333f10b6f62e67a6b0260a52fc3bebe1e09ba6b762c7f7fc7`.

x71p completed trainer steps 1 through 7 with finite grad norms from 0.0190 to
0.0281. Step 8 failed on n77 local rank 5 after the unequal backward result
left 41.5 MiB free and a four-round FP8 receive requested 50 MiB. x72p uses
eight INT4/FP8 transport rounds. The production-size eight-GPU parity and
memory gate passed with 519.6 MiB free before the operation and 89.6 MiB after
both forward and backward.

The client verifies adapter visibility after an absolute timeout and retries
for the full 600-second engine-restart window. The weight watcher restores the
current LoRA after an engine restart before that endpoint receives rollouts.

Zero-learning-rate gate x55a validated this exact native-FP8 model path before
x56p started. It completed 913 microbatches with finite gradients. Grad norm
was 0.01547. Entropy was 0.47795, masked mismatch KL was 0.35719, unmasked
mismatch KL was 0.03791, and 3.171 percent of tokens were masked. Trainer
forward and backward took 3,356 seconds at 556 tokens/s. Peak GPU memory was
46.55 GiB. Rollout collection took 19 minutes 13 seconds. These times are
separate.

Do not add `attention`, `attn_proj`, or `mla_up_proj` to activation checkpoint
targets for the native-FP8 path. Under FSDP resharding, their non-reentrant
recompute used nonfinite sparse-Q activations. x55a checkpoints only routed
experts and norms.

The native routed-expert path must permute source-major EP rows before the
local fused expert call. Native checkpoint linears must keep their weight and
scale as explicit autograd inputs and keep owned backward snapshots outside
checkpoint saved-tensor hooks.

x38 was not a continuation of x37. Its intended adapter restore was erased by
the old run-slot reset order. The creation-hook loader fixes this for later
policy-only restarts.

# Where the rest is

| Document | Content |
|---|---|
| `docs/RUNBOOK.md` | The commands, and what to do when the skill cannot |
| `docs/HANDOFF.md` | Why the system is built this way, and the mistakes made |
| `docs/CHANGES.md` | Every change, with the reason and the patch files |
| `docs/PERFORMANCE.md` | The transport and all-to-all measurements |
| `docs/PIPELINE_PARALLEL.md` | The pipeline stage balance |
| `docs/PINS.md` | The pinned versions |

SwanLab is at `https://swanlab.cn/@MangyMangaur/prod-run`. Do not compare x22
and later against x21 and earlier. The adapter never reached the engines before
x22, so those rollouts came from the base model and their reward data is not
valid.
