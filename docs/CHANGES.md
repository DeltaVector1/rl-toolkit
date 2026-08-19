# GLM-5.2 DECENSOR RL. CHANGE LIST

This document lists each change. The patch files are in `patches/`.

Language: ASD-STE100 Simplified Technical English.

# 2026-08-19: START EXACT INPUT RETENTION FOR GLM EXPERT LORA

This change exists only on the `glm52-optims` branch. It does not change the
active Muse checkout or run.

Each requested INT4 expert dispatch now owns one pinned-host copy of the exact
compressed rows and scales that forward received. Streamed rounds write to
their correct source-order positions. A CUDA event records when the copy is
complete. The implementation does not use one mutable payload slot on a layer.

A bounded host and GPU workspace reconstructs selected rows into a BF16 output
that the caller owns. This prepares the native FP8 expert backward to reuse its
consumed 2,048-row gradient buffer instead of allocating another 24 MiB BF16
input chunk.

The capture remains off unless a future native routed-expert LoRA module asks
for it. Ten INT4 transport-memory tests and thirteen related FP8 memory tests
pass. The changed files pass Ruff, Python compilation, and `git diff --check`.
The eight-GPU CUDA parity test contains the new exact reconstruction checks but
has not run for this branch.

# 2026-08-18: REVIEW EACH ASSISTANT TURN INDEPENDENTLY

The first agentic-review candidate gave one judge access to the complete
conversation. A live test showed that the judge read every transcript document
in one request. It produced the expected result, but it did not limit its read
scope.

The new candidate starts one independent judge session for each assistant turn.
Each session receives the fixed rubrics, the system prompt, all user requests up
to its turn, and only the current assistant reasoning and answer. It cannot read
an earlier assistant answer. All turn sessions run concurrently. The scorer then
uses the highest refusal value and fails each binary review when any turn fails.

The read tool records complete document coverage. The submit tool requires a
strict schema, exact turn number, valid evidence document, exact character
offsets, and an exact quote. A service failure remains unscored after three
attempts.

A live two-turn Gemma test used two calls for each turn. It scored direct
compliance as refusal level 1 with no private-reasoning restriction. It scored a
complete refusal as level 9 with a private-reasoning restriction. The local test
suite has 29 passing tests. Ruff also passes. This candidate is not deployed to
Muse13.

# 2026-08-17: SEPARATE PRIVATE-REASONING AND RESPONSE-QUALITY ADVANTAGES

The old reward multiplied the response-quality value by the private-reasoning
value before GRPO normalized the result. In an all-failed group, this common
small multiplier disappeared during normalization. Better written failures
then received positive advantages.

The GRPO path now centers the private-reasoning pass result separately. It
calculates response-quality differences only among responses that passed that
review. A failed response cannot receive a positive advantage. An all-failed
group receives no update. An all-passed group still trains response quality.
The trace records both advantage parts and the final value.

The judge no longer receives one large reasoning trace. It reviews every
assistant turn in overlapping sections of at most 8,000 characters. One
request returns both the restriction result and the local coherence result.
Visible answers use overlapping sections of at most 12,000 characters. A
second small refusal review combines the section findings when an answer has
more than one section. Dynamic text is escaped before it enters a reviewer
instruction.

The saved Muse13 step-58 comparison contained 64 groups. The old estimator
gave 284 failed responses a positive advantage. The new estimator gave zero
failed responses a positive advantage and had zero pass-versus-fail ordering
violations. Forty groups were all failed and therefore had no new training
signal. On the same 1,024 rows, the section design would make 6,663 judge
requests instead of the recorded 6,713 requests.

# 2026-08-17: REMOVE THE MUSE ROLLOUT LONG-TAIL RETRY LOOP

Muse12 completed 13 trainer steps. Rollout batch 14 reached 1008 of 1024 rows,
then waited more than 90 minutes for one rollout. The one-hour harness deadline
expired, but the dispatcher classified the zero-turn wall-clock timeout as a
transient network fault and scheduled the same missing member again. This
retry had no final limit.

The eight independent H200 replicas were also hidden behind one proxy client.
The dispatcher therefore could not pin a multi-turn rollout group to a real
replica. Later turns could move to a different GPU and lose prefix-cache reuse.
When the orchestrator cancelled a request, the proxy could also keep its
upstream generation active until the full response completed.

Muse13 uses all eight replica URLs directly. The dispatcher can now pin each
GRPO group to one replica and balance new groups by the real per-replica load.
A complete environment rollout has a 1,200-second wall-clock deadline. A
wall-clock harness timeout is terminal: the sink rejects its complete GRPO
group and schedules a replacement. Transport read timeouts remain retryable.

All eight H200 GPUs had zero volatile uncorrected ECC errors. Every direct
replica passed a real completion probe in 0.77 to 0.85 seconds before restart.
After a clean engine restart, all eight replicas passed another real completion
probe. Muse13 rollout batch 1 completed in 12 minutes 20 seconds with 1,024
trainable rows out of 1,056, zero rollout errors, and mean reward 0.0652. Batch
2 started immediately with 122 requests active. Live generation measured
3,547 to 3,741 output tokens per second during the first batch.

# 2026-08-16: VERIFY THE FRESH MUSE LORA POLICY BEFORE STEP 1

Muse7 stopped before its first optimizer update. It measured unmasked
mismatch KL 20.551783, masked mismatch KL 15.61636, and 54.1441 percent
masked tokens. The trainer did not save or broadcast an adapter.

The stored rollout log probabilities reproduce on the H200 engine. On 466
sampled tokens, the mismatch-KL proxy was 0.000365. A separate deterministic
2,048-token H200 comparison against stock Hugging Face had a mismatch-KL
proxy of 0.001508.

The exact trainer path also passes. Tests used both data-parallel ranks, the
real tokens, log probabilities, masks, advantages, and sequence boundaries,
the full 65,536-token packed shape, PP8, CP4 ring, FSDP2, and all 20 rank-local
microbatches. Unmasked mismatch KL was 0.000575 on data rank 0 and 0.000668 on
data rank 1. Forward and backward took 236 to 239 seconds. The muse7 failure
does not reproduce from the saved input.

A fresh production run now explicitly zeros every local LoRA B matrix after
run discovery and optimizer creation. All ranks verify that every LoRA
parameter is finite, every B matrix is zero, and the scale is alpha divided
by rank. A failure stops startup before the trainer accepts a rollout batch.
The first production step also writes one token-level trainer-to-inference
parity record before the parity guard can reject the step.

Muse8 proved that this first check passed, but its forward still had unmasked
mismatch KL 20.719719. The token record had correct position IDs and a mean
absolute log-probability delta of 23.2091. The remaining production-only
operation is `MultiRunManager.synchronize_state()`. It runs after the rollout
file arrives and after the first zero check. Fake-data qualification does not
run it. The trainer now repeats the zero, finite-value, and scale checks after
this synchronization and immediately before it loads the batch. It reports
how many nonzero B shards it reset. This is before the first optimizer update,
so it does not remove learned state.

Muse9 measured that batch synchronization reset zero B shards, so LoRA state
was not the cause. Its first token record showed the actual alignment fault:
the H200 reference mean log probability was -0.626, but the trainer mean was
-23.83. The trainer data loader selected its rollout file with
`global_rank // 32`. PP8 makes each pipeline stage a contiguous group of eight
ranks. This formula made stages 0 through 3 read data rank 0 and stages 4
through 7 read data rank 1 in the same pipeline. The final stage therefore
scored labels from a different sample than the input on the first stage.

The loader now receives the data-parallel local rank from the distributed
mesh. It does not derive it from the global rank. Before each pipeline
forward, every stage also compares hashes of the input IDs, position IDs,
inference log probabilities, masks, LoRA token counts, and sequence lengths.
A difference stops the step before forward. A PP8, CP4, FSDP2 trainer-only
test used different deterministic data on DP ranks 0 and 1. The hash check and
forward/backward passed in 30.4 seconds with finite gradient norm 11.4375.

# 2026-08-15: REPLACE FULL ROLLOUT DEBUG GATES WITH DIRECT QUALIFICATION

Muse7p completed its complete trainer step. PP8, CP4, FSDP2 forward and
backward took 252 seconds. The gradient norm was finite at 2.640625, peak GPU
memory was 30.7 GiB, and the rank-128 adapter export took 7.38 seconds. The
process then failed while it gathered and wrote an unnecessary 55.5 GiB full
Hugging Face checkpoint. The trainer master had approximately 55 GiB free.
Muse trainer configurations now set `skip_gather_master_weights = true`.
The 20-step trainer-only qualification later completed all forward and
backward work, then its final distributed checkpoint filled n82. Muse trainer
configurations now also set `weights_only = true`. The trainer does not save
Muon optimizer, scheduler, dataloader, or progress state. It does not merge
the adapter into the BF16 base model. Production keeps the filesystem LoRA
adapter export that the H200 engines load. Fake-data qualification and direct
parity do not save an unchanged learning-rate-zero adapter.

The same step measured 58.97 percent masked tokens, unmasked mismatch KL
20.26927, and masked mismatch KL 35.00884. Production is not ready. Full
2,048-rollout batches are no longer used as trainer debuggers. A trainer-only
qualification now runs 20 steps on the exact 64-GPU topology with 65,536-token
packed inputs and changing document boundaries. A CE-only objective makes
every step run backward without DPPO hiding the work.

A separate direct parity job uses 2,048 deterministic tokens. Stock Hugging
Face writes the reference log probabilities on one H200. The exact PP8, CP4,
FSDP2 trainer writes its aligned log probabilities and token-by-token delta.
This job does not generate or replay rollouts.

Production now has a fail-closed parity guard. It rejects the step before the
optimizer update and adapter broadcast when unmasked mismatch KL exceeds 0.1,
masked mismatch KL exceeds 0.5, or the masked token fraction exceeds 0.06.

`glm52-ops` now detects trainer-only qualification mode. It does not report
the intentionally absent rollout components as failures. The new
`qualify(name)` operation starts a fake-data trainer without an orchestrator.
All 22 operations tests pass.

# 2026-08-15: PRESERVE RING BOUNDARIES DURING PIPELINE RECOMPUTE

Muse7o completed 2,080 rollouts with no rollout error. The accepted 2,048-row
batch reached the trainer. During the first pipeline backward, n84 stopped
with a checkpoint metadata error. Two saved ring-attention boundary tensors
had shape four in the forward and shape five in the recompute. n85 then
stopped while it waited for the failed pipeline peer. The hosts and GPUs did
not fail.

Ring attention keeps packed-sequence boundaries in one process-global
dictionary. Pipeline 1F1B starts a later microbatch before it recomputes an
earlier microbatch. The later forward replaced the earlier boundaries. Each
full activation checkpoint now captures the ring values at its forward and
restores them during its recompute. The previous values return when the
checkpoint context exits.

The trainer launcher now mounts `trainer/model.py`, which applies this context
to full activation checkpoints. A CP4 CUDA reproducer ran two interleaved Muse
microbatches with different packed-document counts. Backward completed with
finite gradients. The pipeline regression suite passed 13 tests. Fresh
zero-learning-rate parity gate muse7p started with all eight trainer nodes and
fresh rollouts.

# 2026-08-15: GATE MUSE TRAINER-TO-INFERENCE PARITY

Muse6 completed 22 trainer steps without a system failure, but its effective
reward change remained inside the measured noise. The adapters changed on all
832 tensors and the H200 engines loaded the policies. Approximately 51 to 62
percent of tokens were still masked by the DPPO trust region. Unmasked
mismatch KL was approximately 10 to 21 nats. The defect was present at step 1,
before an optimizer update could cause policy drift.

An H200 engine rescore reproduced the stored rollout log probabilities. Both
H200 engines also agreed. The mismatch is therefore between the native vLLM
Muse forward implementation and the Transformers Muse forward implementation
on the trainer. It is not a stored-log-probability alignment fault, a missing
adapter update, or an engine policy-load fault.

The Muse H200 launcher now selects the vLLM Transformers model implementation
by default. `MUSE_MODEL_IMPL` can select `auto`, `vllm`, or `transformers` for
a controlled comparison. This source change is not active in muse6. A fresh
zero-learning-rate parity gate must pass before a new production run starts.
The gate must not change the DPPO mask thresholds.

The Transformers implementation did not expose the inner language-model
tensor-parallel plan through the outer Muse model. The launcher site hook now
exposes that plan. It replicates the two key and value heads because TP8 cannot
shard two heads. It also keeps the Muse query, key, and value projections
separate. Other vLLM fusion rules do not change.

The compiled Transformers path with decode CUDA graphs returned nonfinite log
probabilities. The exact Prime-RL endpoint then returned HTTP 400. Separate
query, key, and value projections did not correct this result. Eager execution
corrected it, but produced only approximately 600 tokens/s on each H200 host.
vLLM compile mode 3 with CUDA graphs disabled also corrected it and produced
approximately 1,200 to 1,450 tokens/s on each host under the real gate load.
The launcher now uses this compiled, no-CUDA-graph path.

The no-CUDA-graph path completed 32 standard completion requests and 32 exact
Prime-RL requests on each engine with finite values. Under a fresh production
batch, it completed 121 exact requests without one HTTP 400 response. A
piecewise CUDA-graph test passed one-token prefill scoring but failed all 64
512-token decode requests with HTTP 400. Explicit graph input copying then
failed capture with a 65,536 versus 256 tensor-size mismatch. These graph
paths are not enabled.

Muse compilation and graph setup can exceed three minutes. The engine repair
helper now gives a Muse engine 600 seconds to start. The GLM engine grace stays
at 180 seconds. Fourteen operations regression tests pass for this change.

# 2026-08-14: MUSE RANK-128 FULL LANGUAGE-LAYER LORA

The Muse BF16 FFT run exported a 55.5 GiB policy at each step. The measured
transfer to both H200 hosts took approximately 37 minutes. `AGENTS.md` now
requires this warning before a full-weight policy run.

The new Muse configuration uses rank-128 BF16 LoRA with alpha 256 and dropout
0. Exact model inspection found 416 language-layer linear modules. The target
patterns include all 260 attention projections and gates and all 156 MLP gate,
up, and down projections. They exclude the vision tower. The adapter has
838,336,512 trainable parameters. The H200 launcher enables rank-128 BF16 LoRA,
fully sharded LoRA across TP8, and the same projection suffixes.

# 2026-08-14: CHECKPOINT THE SHARED-EXPERT BRANCH

x82p completed three finite trainer steps. Trainer step 4 had 115 packed
micro-batches. The first forward pass failed in the INT4 expert dispatch on
n76 and n77. The ranks requested 384 to 568 MiB for a BF16 dispatch result
with only 169.5 to 379.5 MiB physically free. The later NCCL failures were
consequences of these allocation failures.

The new activation-checkpoint mode uses the established safe reentrant path
for attention and also checkpoints the shared-expert branch. It does not
checkpoint or repeat the routed experts, expert dispatch, or all-to-all. The
first dense GLM-5.2 layers have no shared expert, so they checkpoint attention
only. A CUDA functional gate confirmed two attention calls, two shared-expert
calls, one routed-expert call, and finite gradients.

x83p found two startup configuration defects before model loading. The first
version rejected the dense layers. The rank-0 launcher also required an
explicit filesystem rollout-transport field in the trainer TOML. Both defects
now have regression guards. Fresh production run x84p started with the
required batch, sequence length, learning rate, precision, and parallelism.
Production trainer steps 1 through 5 completed with finite gradients. Step 4,
which was the failure position in x82p, completed without an OOM. Its forward
and backward took 747.5 seconds and its measured peak was 44.22 GiB. The run
continues because five steps are not the final stability gate.

# 2026-08-14: APPLY A STAGED POLICY AT ZERO BATCH PROGRESS

x84p rollout batch 4 started at zero accepted rows while policy v2 was ready.
The orchestrator waited for the full 1,200-second batch-tail timeout. It then
applied the policy and restarted 11 incomplete groups. This discarded 75 live
and 101 completed members. Rollout batch 4 took 36 minutes 34 seconds in total.

There is no accepted batch work to preserve when batch progress is zero. The
policy-boundary wait now returns immediately in this state. The normal bounded
wait remains active after the first accepted group. Thus, a policy update at
zero progress can drain or restart partial groups before they grow for 20
minutes. Eight policy-update regression tests pass. This change is local and
does not alter the active x84p process.

# 2026-08-13: BOUND THE FP8 EXPERT FORWARD LIVE SET

x78p completed eight trainer steps with finite gradients. Trainer step 9 used
a packed batch of 112. Five ranks failed at two forward expert sites. The
4,096-row FP8 expert-input quantization needed 24 to 48 MiB when some GPUs had
3.5 to 31.5 MiB physically free. One rank also needed 48 MiB because boolean
indexing in the in-place expert writeback made a hidden source-row copy.

The native-FP8 expert forward workspace now uses 2,048 rows. This bounds the
quantized input and grouped-matrix workspaces without changing the batch,
sequence length, model precision, expert ownership, or output order. A Triton
scatter writes each real permuted row directly into the reused destination.
It does not make the two boolean-index gather temporaries. The backward uses
the same direct scatter.

The SM89 gate passed exact forward output and exact input-gradient parity for
36,352 source rows and 36,864 permuted rows. It confirmed 2,048-row forward
and backward chunks, routed-input storage reuse, pinned-CPU saved FP8
activations, and a 426.3 MiB incremental peak.

# 2026-08-13: RETRY FRAGMENTED FP8 FORWARD ALLOCATIONS

x77p completed seven trainer steps with finite gradients. Trainer step 8 used
a larger packed batch and found two forward-path allocator failures. n77 local
ranks 0 and 4 could not allocate the 430 and 458 MiB BF16 INT4 dispatch
results. n76 local ranks 0, 3, and 5 could not allocate 24 to 48 MiB FP8
expert-input quantization results. In each case, the allocator held more
unused reserved memory than the failed allocation size.

The INT4 forward allocations and native-FP8 expert-input quantization now
release unused cached allocator blocks and retry the identical operation once.
The tensor shape, precision, transport rounds, and experimental configuration
do not change. A live-set OOM remains fatal on the second attempt.

The injection tests verify one cache release and one identical retry. The n77
eight-GPU gate passed exact forward and backward parity and production-size
unequal routing with 559.6 MiB initially free. It had 129.6 MiB free after
forward and 61.6 MiB free after backward.

The rollout dispatcher now retries the exact vLLM response-metadata NaN HTTP
400 on the other policy endpoint. It keeps the GRPO group open and replaces
only the missing member. Other provider, environment, and judge errors keep
their existing behavior.

# 2026-08-13: RETRY A FRAGMENTED FP8 BACKWARD RECEIVE

x76p completed four trainer steps with finite gradients. Trainer step 5 used a
packed batch of 104. n77 local rank 2 failed before the receive rounds while
it allocated the 488 MiB BF16 inverse-gradient result. The GPU had 389.5 MiB
physically free and PyTorch held 613.8 MiB reserved but unused. The full
result allocation now uses the same one-retry allocator recovery as the
bounded receive allocation. Its shape and lifetime do not change.

x75p completed seven trainer steps with finite gradients. Trainer step 8 used
a packed batch of 107. n77 local rank 3 failed in `_streaming_fp8_backward`
while it allocated one 26 MiB FP8 receive round. The GPU had 19.5 MiB
physically free and PyTorch held 284.7 MiB reserved but unused. The other NCCL
errors were consequences of this OOM.

The bounded eight-round FP8 backward path now retries this allocation once
after it releases unused cached allocator blocks. It does not change the
allocation shape or transport rounds. A live-memory OOM still fails on the
second identical allocation.

The unit test forces the first allocation to fail and verifies one cache
release followed by the identical allocation. The final n77 eight-GPU gate
passed exact forward and backward parity and production-size unequal routing
with 559.6 MiB initially free. It had 129.6 MiB free after forward and 61.6
MiB free after backward.

The acceptance probe now keeps each container log on a separate line. It also
excludes the known vLLM JSON serialization error from the synthetic completion
probe. Actual rollout errors still come from rollout metrics, and real trainer
NaN, Inf, nonfinite-gradient, and OOM records remain acceptance failures.

# 2026-08-13: RETRY A FRAGMENTED FP8 EXPERT ROW GATHER

x74p completed five trainer steps. Trainer step 6 used a packed batch of 110.
n77 local rank 1 failed when the native-FP8 expert forward tried to gather a
4,096 by 6,144 BF16 row chunk. The 48 MiB allocation had only 7.5 MiB
physically free, although PyTorch held 179.6 MiB reserved but unused. The
other NCCL errors were consequences of this OOM.

The expert row gather now keeps the 4,096-row fast path. If the gather fails,
it releases unused cached allocator blocks and retries once. A real live-set
OOM still fails on the second allocation. The same bounded retry protects the
smaller backward row gather. The n77 SM89 gate passed exact forward output and
input-gradient parity for 36,352 source rows and 36,864 permuted rows. Its
incremental peak was 426.3 MiB.

The failed 64-rank processes also left 73 to 108 GiB of Apport core files on
five trainer nodes. These crash files were removed. The trainer launcher now
sets its core-file limit to zero. This keeps a later worker failure from
filling the host root filesystems and blocking the NCCL diagnostic trace.

# 2026-08-13: RESERVE INVERSE-COMBINE STORAGE DURING INT4 DISPATCH

x73p completed seven trainer steps. Trainer step 8 used a packed batch of 123
and failed on n76 local rank 7 in `_streaming_int4_forward`. The combine hook
needed a 384 MiB BF16 result and had 335.5 MiB physically free. Other worker
exits were consequences of this OOM. The fail-closed supervisor stopped the
orchestrator and durable batch publisher.

The eight-round transport was active. The failed rank received fewer routed
rows during dispatch than it had to return during combine. Dispatch therefore
allocated less storage than the inverse result required, and combine could not
reuse its expert-output input.

Dispatch now reserves BF16 storage for the larger of its input and output row
counts while memory is available. It keeps the received logical shape. The
native-FP8 expert path preserves the allocation. Combine detects the physical
storage capacity, keeps the compact INT4 receive payload, quantizes only one
send round at a time in the unused reserved tail, and expands into BF16 only
after all sends finish.

The n77 eight-GPU test passed exact small-tensor forward and backward parity,
then production-size unequal dispatch, inverse combine, and backward with
559.6 MiB initially free. It had 129.6 MiB free after forward and 61.6 MiB free
after backward. Static memory guards and production configuration guards also
passed.

# HOW TO APPLY THE CHANGES

The checked-out Git tree is the current deployment source. Generated patches
of the complete working-tree diff were removed because they became stale and
duplicated the repository.

The old generated commit-by-commit patch directory was removed during the
repository cleanup. Git keeps those commits. Use the maintained portable
patch series instead:

```
git am patches/prime-rl/v0.8.0-glm52/*.patch
```

# PRIME-RL V0.8.0 SOURCE-MAINTENANCE PORT

`patches/prime-rl/v0.8.0-glm52/` contains an eight-patch port to the exact
Prime-RL v0.8.0 release at
`60bc29547a8824ad1de7b9af8d265e2b27b2a72d`. The directory README records the
inventory, excluded diagnostics and INT4 weight code, dependency revisions,
validation results, and the remaining semantic gates.

This patch series is not deployed. It does not change the current Prime-RL
pin or x56p. Prime-RL v0.8.0 changes packing, transport, LoRA, environment,
configuration, and vLLM interfaces. The current x56p orchestrator
configuration fails the v0.8.0 schema and must be rewritten before a runtime
gate.

The matching v0.8.0 dependency ports are:

```
patches/renderers/v0.8.0-disconnected-and-tokenizer.patch
patches/verifiers/v0.8.0-disconnected-provider-retry.patch
```

# NATIVE-FP8 SHARED-EXPERT LORA

x72p used rank-32 LoRA on four attention projections. It trained 139,456,512
adapter parameters and did not adapt an MLP. x73p keeps rank 32 and adds the
always-on shared expert in each MoE layer. This increases the complete adapter
to 198,503,424 BF16 parameters without adapting all 256 routed experts. Routed
expert LoRA would require approximately 15.7 billion adapter parameters and is
not suitable for the 48 GiB trainer GPUs.

`MultiLoRAFP8CheckpointFeedForward` keeps each frozen shared-expert weight and
its 128-by-128 inverse-scale grid as explicit inputs to the native FP8 linear.
The LoRA A and B matrices use BF16 and the existing tensor-core path with FP32
dot accumulation. The wrapper adapts gate, up, and down projections. Adapter
export renames the trainer's singular `shared_expert` path to the HF and vLLM
`shared_experts` path. The H200 server keeps routed fused-MoE LoRA disabled for
this target because the shared expert uses ordinary linear layers.

The SM89 eager and reentrant-checkpoint gate produced matching outputs, input
gradients, and all six LoRA gradients. The FSDP-wrapped SM89 gate also produced
finite gradients. Both H200 hosts loaded, generated with, and unloaded a
synthetic shared-expert adapter before x73p started.

x73p trainer step 1 completed with finite grad norm 0.0951. Forward and
backward took 695.7 seconds, peak GPU allocation was 46.11 GiB, and the
397,014,776-byte exported adapter loaded on both H200 hosts. Step 2 completed
with finite grad norm 0.0506, 669.4 seconds of forward and backward, and 46.35
GiB peak allocation. These are early memory and precision gates, not the final
50-step stability result.

x73p also found two configuration-copy errors before trainer compute. A stale
trainer `rollout_transport.output_dir` field was not valid in the current
schema. A missing orchestrator top-level `output_dir` sent control and rollout
files to `/app/outputs/run_default`. The corrected configurations now validate
and write to `/data/glm52-rl/runs/x73p/run_x73p`. The SwanLab tag list is also
limited to 30 items, as required by its API.

# BOUNDED POLICY UPDATES AND ROLLOUT TAIL RESERVE

x71p showed two rollout stalls with healthy engines and zero rollout errors.
At policy versions 6 and 7, the orchestrator stopped admission and waited for
all live requests before it called the LoRA loader. Each drain took several
minutes and one took approximately 10 minutes. The adapter apply then took
approximately two more minutes. Step 8 also stopped at 1,008 completed
rollouts with 15 members of one group buffered while the last member ran for
approximately 2.5 minutes.

A policy update now invalidates each incomplete live-policy GRPO group,
preserves its original prompt, and schedules a new complete group under the
new policy. Partial results are removed from the train sink without error
markers, so policy updates cannot create failed groups or mix policy versions
inside a baseline. The H200 pause endpoint now honors `mode=abort`. The
orchestrator uses that engine-native abort before the fused MoE LoRA load, so
cancelled HTTP clients cannot leave registered decode requests behind.
Version-pinned eval groups still drain before an update.

The rollout budget also keeps one complete group beyond the exact batch
target. This group is normal overflow for the next batch. It lets a batch ship
without waiting for the slowest member of the exact final group, and it does
not drop or partially train a rollout.

x71p trainer step 8 then failed independently on n77 local rank 5. The unequal
FP8 backward result allocation left 41.5 MiB physically free, and the first
four-round receive needed 50 MiB. Production now uses eight transport rounds,
which reduces that receive to approximately 25 MiB. The constrained eight-GPU
gate uses the same eight rounds and a 520 MiB initial-free-memory limit.

# PARALLEL AND RESUMABLE H200 ADAPTER STAGING

x72p validated the bounded policy update at policy version 1. The update
cancelled 31 live old-policy requests and removed 17 partial results from
three incomplete groups at 06:34:05 UTC. It did not wait for those requests.
Both engines resumed with the exact trainer adapter at 06:37:48 UTC. The
complete boundary took 3 minutes 43 seconds. The remaining delay was adapter
transfer, not rollout drain: h200b staged the 278,990,560-byte adapter in 99
seconds, while h200a needed 223 seconds. Both files matched the trainer
SHA-256 `5940e609a49dd5599737290e5fd7040dccbcad3a8ddf1766abac2615f5694152`.

Large adapter files now use eight independent HTTP byte ranges. Each range
has its own resumable part file. A reset retries only the incomplete range.
The parts are assembled to a temporary file, checked against the advertised
content length, and renamed atomically. Small files keep the previous
single-stream path. `PRIME_LORA_DOWNLOAD_STREAMS=1` disables the parallel
path for diagnosis.

The local range-server tests verify exact assembly and partial-range resume.
An H200 test over the real adapter tunnel transferred the same 278,990,560
bytes in 76.8 seconds and reproduced the trainer SHA-256. This was 22 percent
faster than the faster old host and 66 percent faster than the slower old
host. The engines were rebuilt one at a time. Transient requests to the
rebuilding endpoint were retried on the live endpoint and did not enter a
training batch as failures.

x72p policy version 2 confirmed that the local in-place LoRA operation is not
the delay. h200b wrote `READY` and reported the loaded adapter in the same
second. h200a did the same. The remote staging took 79 seconds on h200b and
179 seconds on h200a; the complete boundary took 3 minutes 7 seconds.

Remote adapters are now prefetched on both H200 hosts before the local
`STABLE` marker is exposed to the orchestrator. The API server downloads into
the host-shared cache without an engine collective and without pausing
generation. Only a successful prefetch on both hosts exposes the marker. The
normal worker-side stage remains as a fail-safe and returns immediately when
the cache has `READY`. The watcher also uses the same two-phase stage before
it notifies policy-version observers. This removes WAN transfer from the
engine pause and lets complete old-policy groups finish during the transfer.

x72p rollout step 3 then needed 37 minutes 29 seconds. It completed with 1,040
rollouts, of which 1,024 were trainable, zero rollout errors, 0.20 percent
truncation, and mean reward 0.08834. The last 64 scheduled episodes were spread across four
incomplete groups, so one spare group still left the batch waiting for the
slowest tail requests. The tail reserve is now four groups. It does not change
the 1,024-rollout trainer batch, group size 16, or the 128-request concurrency
cap. Completed overflow carries into the next batch.

The first live two-phase update was policy v3. Both hosts started staging at
07:39:34 UTC. h200b wrote `READY` after 87 seconds and h200a after 208 seconds.
During this transfer, rollout step 4 advanced from 784 to 960 completed
episodes. The boundary then restarted five incomplete groups instead of the
ten groups at policy v2. Both engines loaded locally and resumed in two
seconds. The trainer and both H200 files had SHA-256
`3d9a4717c454e102e151205c413f5751c83a79a14c9ff4e9c63d974466af46e2`.

The policy v4 prefetch found a separate control-path fault. The h200a reverse
SSH process and listener remained present, but the adapter download stopped
making byte progress after 134 MiB. Restarting only that SSH child let the
resumable download complete immediately. The control-tunnel supervisor now
uses a bounded HTTP `HEAD` probe every 30 seconds and rebuilds only the SSH
child after two consecutive failures. It does not restart an engine or the
orchestrator. The publisher still exposes `STABLE` only after both H200 hosts
have a complete adapter cache, so the watcher staged policy v4 locally in 15
milliseconds and both engines loaded it in two seconds.

The first rollout batch with the four-group tail reserve was step 5. It
completed 1,024 trainable rollouts with zero rollout errors in 20 minutes 37
seconds. This is one measured batch, not yet a stable performance trend.

Policy v5 exposed a remaining boundary cost. Its adapter was present on both
H200 hosts 81 to 101 seconds after trainer publication and matched SHA-256
`494984d193fe472fb9c8505a02de88de7f8d43a9e889465e0d35880d29c6256a`.
The watcher staged it locally in 9 milliseconds, but rollout step 6 was already
at 1,008/1,024. Applying the strict boundary immediately discarded 57
completed members from five incomplete groups and rollout step 6 then needed
23 minutes 52 seconds in total.

A staged policy update now waits for the current batch to ship when the sink
is inside its last four complete groups. The wait is limited to 300 seconds
and is allowed only when the current policy can ship that batch within the
existing off-policy limit. After the batch advances, the normal strict
full-group restart and adapter load still run before new-policy generation.
This removes avoidable tail regeneration without mixing policy versions in a
GRPO group or weakening the off-policy gate.

Policy v7 arrived in the middle of rollout step 8. Thirteen incomplete groups
contained 80 completed members, and immediate strict restart discarded those
members. A policy boundary now stops opening fresh prompts and gives existing
groups at least one complete group's worth of retained work up to 60 seconds
to finish. Completed groups remain valid old-policy training input. After the
bounded grace, the existing strict restart applies to every group that is
still incomplete. Existing group members can continue scheduling during the
grace; only fresh groups are blocked. This reduces repeated inference while
retaining a hard update-time bound.

The first live grace was policy v8. It began with 14 incomplete groups and
106 completed members. After the exact 60-second bound, two groups had
finished and remained valid; the strict boundary restarted the remaining 12
groups. This prevented 32 rollout regenerations. Both engines resumed under
policy v8 with 64 and 63 running requests. The remaining restarted groups
stayed policy-pure.

The batch-7 publisher also recorded three simultaneous SSH key-exchange resets
when it launched all 16 resumable stripes at once. The parts survived and the
retry published the exact file. Later waves still use 16 independent streams,
but their SSH handshakes are now spaced by 250 milliseconds. This keeps the
measured parallel data path and removes the connection burst at the jump host.

The edge activation initially stopped before rollout scheduling because the
x72p SwanLab tag list had grown to 31 entries and the service accepts at most
30. No trainer process or completed batch stopped. The redundant
`last-turn-judging` display tag was removed, the same run resumed at rollout
batch 8 from policy v6, and the active container was verified to contain the
tail-deferral code. This changes run metadata only.

# IN-PLACE INT4 COMBINE AND FAIL-CLOSED TRAINER SUPERVISION

x64p failed in the INT4 expert combine because it allocated a 384 MiB BF16
result while the packed source and receive buffers were still live. An
eight-round receive reduced the scratch space, but x65p step 8 had a larger
packed batch of 126 and failed with only 235 to 335 MiB free before the same
384 MiB result allocation.

The combine now quantizes the full source once, receives it in four bounded
all-to-all rounds, and writes the dequantized result into the consumed expert
output buffer. The extra peak is approximately 96 MiB for the packed source
and 24 MiB for one receive round. The eight-GPU smoke test passed with 159.6
MiB free before combine and 39.6 MiB free after it. The test also checks
variable peer split sizes, exact forward parity, finite backward gradients,
and storage aliasing.

The old run supervisor checked only the n82 container and automatically
restarted a failed distributed trainer. This production path cannot save a
resumable optimizer checkpoint because a full frozen-base distributed
checkpoint exceeds the available disks. An automatic restart therefore lost
the Muon state and could leave the orchestrator on a policy version that the
new trainer could not publish.

The supervisor now checks the container on every trainer node. If one worker
exits, it writes a failure marker, stops all remaining trainer workers, and
exits without a restart. A watchdog on h200a independently checks all eight
workers through n82. Two consecutive misses stop the edge orchestrator and
the durable batch publisher. `status()` also reports the trainer as up only
when all eight containers are up.

x68p then reached the same step-8 pressure point and exposed the remaining
backward allocation. The inverse all-to-all received a full FP8 gradient and
then converted it to a separate 384 MiB BF16 tensor. n77 rank 1 had 367.5 MiB
physically free and 911.12 MiB reserved but unused, so that conversion failed.

Combine backward now sends and receives FP8 gradients in four rounds and
dequantizes each round directly into the consumed incoming-gradient buffer
when its storage is large enough. Ranks with a larger returned row count use
a separate result but still have bounded receive scratch. The eight-GPU gate
compares both forward values and backward gradients exactly against the
original one-shot path for unequal routing. Its production-size constrained
case completed forward and backward with 395.6 MiB free initially, 275.6 MiB
after forward, and 155.6 MiB after backward.

The trainer allocator no longer sets `max_split_size_mb=256`. That option
prevented the failed 384 MiB request from using the 911 MiB reserved free
space. Expandable segments and the garbage-collection threshold remain.

x69p completed seven finite trainer steps. Its step-8 batch then exposed the
one remaining one-shot direction: expert dispatch. n77 local rank 5 received
42,445 routed rows and tried to allocate a 498 MiB BF16 tensor while the full
packed send and receive tensors were live. Only 413.5 MiB was physically free.
The fail-closed supervisors stopped the distributed trainer and the edge
orchestrator. No partial step was committed.

The four-round path now applies to both dispatch and combine. If the consumed
input has enough rows, it is quantized completely before its storage is reused.
If routing imbalance needs a larger output, the BF16 destination is allocated
first and only the current round is quantized and retained. FP8 backward uses
the same two branches. Per-row scaling makes round-local quantization exactly
equal to full-tensor quantization.

The revised eight-GPU gate uses unequal production-width routing so half of
the ranks allocate in forward and the other half allocate in backward. Exact
forward values and backward gradients match the one-shot path. The constrained
case completed with 559.6 MiB free before the operation and 95.6 MiB free after
both forward and backward.

x70p then failed closed at the first trainer forward before any optimizer
step. Dispatch had reused its input on ranks whose returned row count was
smaller. That input is an FSDP `RegisterPostBackwardFunction` view. PyTorch
correctly refused an in-place change that could replace FSDP's custom
backward. Dispatch now always uses the allocate-first, round-local branch.
Only combine reuses its private expert-output allocation. The 8-GPU parity
gate now invokes the public autograd function with the dispatch tag and
asserts that its result never aliases the input.

# ROLLOUT FAILURE CONTAINMENT AND LORA RECONCILIATION

x61p rollout batches 9 through 11 had 960 policy-inference failures in each
1,024-rollout window after h200b restarted without its dynamic LoRA. The
orchestrator sent each partial batch with only 64 trainable rollouts. Trainer
step 9 rejected the resulting five rank-local micro-batches because PP8 needs
at least eight, then waited for a replacement that the filesystem transport
cannot produce.

Policy-model 404 responses are now transient dispatcher failures. The
dispatcher keeps the GRPO group open and retries it away from the failed
endpoint. The weight watcher checks every inference endpoint at 30-second
intervals and reloads the current adapter on an endpoint that lost it after a
restart. It verifies that all endpoints expose the exact policy model after
the reload.

The train sink no longer uses errored or oversized rollouts as sample-free
placeholders in a fixed-size batch. One failed, oversized, or untrainable
member rejects the complete GRPO group, and the rollout budget opens a clean
replacement group. Eight consecutive errored groups stop collection. Thus,
an infrastructure or judge failure cannot become a large partial trainer
batch.

The deployed x61p Decensor environment also omitted the final assistant
answer from its refusal and coherency judge inputs. The repository already
contains the last-turn judging repair. A regression now checks that both
judges receive the final response. The next run deploys this repository
version and starts from a new policy because the x61p reward inputs are not
valid for continuation.

# BOUNDED BUSY-ENGINE WATCHDOG GRACE

x64p h200b stopped token progress during each of the first two LoRA policy
updates. The engine still held 1 to 23 registered requests and 133 to 143 GiB
on GPU 0, but the watchdog applied the same two-probe limit that it uses for
an empty dead EngineCore. It restarted h200b after approximately two minutes.

The watchdog now permits five failed probes when requests remain registered
or when the full engine remains resident above 64 GiB and the token counter
does not move. It still restarts an empty engine after two probes, and rising
production tokens still reset the failure count. The five-probe limit keeps a
real resident-engine wedge bounded below ten minutes. The smoke test runs both
sides of this rule: no restart at the ordinary limit, and a restart when each
bounded grace expires.

x64p policy v4 validated the deployed rule. h200b stopped token progress with
four requests and 130649 MiB resident. The watchdog recorded misses 1/5 and
2/5 instead of restarting at 2/2. The adapter load completed with HTTP 200 at
07:18:41 UTC, and the next generation probe recovered without changing the
container start time. Rollout batch 5 remained 1,024/1,024 trainable with zero
errors.

x64p rollout batch 6 then exposed a separate non-error case: one complete
16-rollout GRPO group had uniform rewards, so its assigned advantages were all
zero. The batch had no infrastructure errors but only 1,008 of 1,024 rollouts
carried a gradient. `TrainSink` now checks the post-finalization training
signal, rejects and replenishes a complete zero-advantage group, and stops
after eight consecutive zero-advantage groups instead of building a
no-gradient batch.

# BOUNDED NATIVE-FP8 EXPERT BACKWARD

x63p produced a clean 1,024-rollout batch with zero errors, but trainer step 1
reached 47.0 GiB and failed with a CUDA out-of-memory error in the routed-
expert SiLU backward. The backward expression created multiple full gate
temporaries. The in-place EP loop also kept the previous chunk referenced
until allocation of the next chunk completed.

The SiLU-GLU backward now reuses its backward-only input-gradient storage. It
keeps the original operation order, uses 2,048-row backward chunks, and
releases all per-chunk tensors before it starts the next chunk. The 4,096-row
forward chunks do not change.

The SM89 streamed-activation gate passes with exact forward output and exact
input gradient. It saves 142 MiB of pinned FP8 activations and uses a 1,851.17
MiB incremental peak, compared with 1,994.17 MiB for the ordinary path. The
in-place EP gate also passes with exact forward output and exact input
gradient. It reuses routed-input storage, saves 144 MiB of pinned FP8
activations, and has a 426.28 MiB incremental peak.

x64p also exposed a cross-step host-memory high-water problem. On n76, shared
memory rose from 380 to 391 GiB and available memory fell from 76 to 62 GiB by
trainer step 6 while streamed FP8 activation offload was active. The process
maps showed approximately 51 GiB of `/dev/zero (deleted)` shared pages per
worker, but that measurement does not identify the allocator that retained
them. After the optimizer step, the trainer now synchronizes pending H2D
copies and calls the runtime's host-cache release operation to prevent unused
pinned allocator blocks from carrying into the next variable-sized batch.
Missing runtime support is a hard error when FP8 expert activation offload is
enabled. The deployed PyTorch 2.11 runtime provides this operation. A future
trainer restart must validate the effect on the live host-memory series.

# PP8 UNDERSIZED-BATCH GUARD AND PRODUCTION X61P

x60r completed eight native-FP8 trainer steps with finite gradients. Step 8
used 94 micro-batches, took 797.05 seconds for pipeline forward and backward,
and peaked at 46.5 GiB. An h200a engine restart removed its loaded LoRA. The
resulting rollout batch for trainer step 9 packed to only two micro-batches.
PyTorch `Schedule1F1B` requires at least as many micro-batches as pipeline
stages and stopped PP8 during schedule construction.

The trainer now computes the minimum rank-local micro-batch count before later
step collectives. For 1F1B it discards a batch below the pipeline stage count
and waits for the next fresh batch without advancing the policy step. GPipe
keeps its minimum of one. The pipeline regression tests cover both schedules
and an unknown schedule.

x61p is the fresh production configuration. It keeps batch 1,024, group 16,
32,768-token context, 16,384-token maximum output, PP8, CP8, node-local EP8,
native FP8, Muon, maximum gradient norm 1.0, and weight decay 0.0. It changes
the zero-learning-rate gate to the required learning rate of 8e-6.

# HARD OFF-POLICY REJECTION

The dispatch gate limits new work, but completed groups can remain buffered
across a trainer-batch boundary. x56p rollout batch 9 therefore contained 128
policy-v3 rollouts at staleness 5 although `max_off_policy_steps = 4`.
`TrainSink` now rejects and replenishes complete stale groups before they can
become trainer samples, including buffered overflow. The limit is inclusive:
staleness equal to the configured value is accepted, and a zero limit accepts
only the current training policy. Three regressions cover the x56p overflow,
completed stale groups, and the zero-limit case.

This change is committed for later deployments. It is not active in the
already-running x56p containers.

# DECENSOR LAST-TURN JUDGING

Remote Decensor judges grade only the final assistant turn. The earlier
conversation remains context for the coherency judge, but refusal, reasoning,
meta-commentary, and Markdown verdicts use the final assistant response and
its reasoning trace. Full-conversation aggregation was reverted because prior
Decensor experiments produced flat reward.

# NATIVE FP8 REPAIR AND GATE X55A

The native-FP8 failures had three separate causes.

First, the routed-expert forward path did not apply TorchTitan's expert-
parallel row permutation after all-to-all. Source-major rows used global
expert IDs with local expert matrices. The repair uses
`generate_permute_indices()`, combines local counts, adds aligned sentinel
padding, calls the fused experts, and unpermutes the output. Gate x46p then
measured masked mismatch KL 0.3562 instead of x45r's 4.0768.

Second, FSDP released frozen FP8 parameter storage before the inner custom
linear backward. `_Linear` now keeps owned weight and scale snapshots directly
on the autograd context, outside checkpoint saved-tensor hooks. Weight and
scale remain explicit tensor inputs so FSDP can see the dependency. A two-rank
CUDA FSDP reshard and non-reentrant checkpoint regression passes. Backward
diagnostics carry the full module FQN and deduplicate by FQN, not tensor shape,
because several GLM projections have the same geometry.

Third, non-reentrant activation checkpoint recompute of native-FP8 attention
under FSDP produced an all-NaN sparse Q. This was measured directly in x53d
and x54e. Native-FP8 configs therefore do not checkpoint `attention`,
`attn_proj`, or `mla_up_proj`. They checkpoint only routed experts and norms.
The sparse MLA kernel also returns zero output and positive-infinity LSE for a
fully masked row, so its backward gradient is zero and finite.

Zero-learning-rate, batch-1024 gate x55a validated the combined full
native-FP8 path at sequence length 2,048:

- 913 microbatches completed in 3,356.38 seconds.
- Grad norm was finite at 0.015471.
- Entropy was 0.477953.
- Masked mismatch KL was 0.357186.
- Unmasked mismatch KL was 0.037908.
- The masked fraction was 3.171 percent.
- Throughput was 556 tokens/s and peak GPU memory was 46.55 GiB.
- Rollout collection took 19 minutes 13 seconds with zero environment errors.
- Step 1 exported a 278,990,560-byte LoRA adapter and its `STABLE` marker.

The trainer time and rollout time are separate. The low throughput is a
measured cost of the safe native linear backward and the reduced activation
checkpoint set.

# SPARSE-CACHE PIPELINE BACKWARD MEMORY

Gate x60h used exact FP16 bit transport for the sparse-MLA index cache across
balanced pipeline boundaries. PyTorch PipelineStage treated this discrete
cache as a differentiable output. The static pipeline shape also used the
configured 32 indexer heads, although `compute_sparse_indices()` emits one
shared index group. During schedule setup, x60h therefore allocated 512 MiB
cache-gradient buffers until it failed before the first forward. After the
unused cache gradients were removed, x60i reached its first forward and
reported the exact shape mismatch: the declared `[1, 4096, 32, 2048]` output
was actually `[1, 4096, 1, 2048]`.

`SparseCachePipelineStage` now sends the cache only in the forward direction.
Only the hidden-state output has a backward receive buffer. Cache inputs do
not require gradients, and stages do not send cache gradients. The declared
cache shape now has the one shared index group that the model emits. A real
cache buffer is 16 MiB for each production microbatch. The cache bits and the
hidden-state gradient are unchanged. The two-rank 1F1B transport test checks
the exact boundary values 0, 1, 2047, 2048, 32639, 32640, 32767, and 32768.
It also checks that the cache has no backward buffer. Four microbatches
completed with finite gradients on both ranks.

The custom pipeline stage also uses an event-protected receive ring for this
forward-only cache. It does not allocate one cache receive buffer for each
microbatch, and it removes sent and received cache tensors from the saved
forward state after the P2P operations own them. The ring has at most one slot
for each pipeline stage. This keeps cache transport memory bounded when the
batch-2,048 gate has approximately twice as many microbatches. The two-rank
test checks that four microbatches use no more than two cache buffers.
The production-shape test sends 95 caches of shape `[1, 4096, 1, 2048]`.
It completed with 0.117 GiB peak allocated memory on the receiving rank. The
old one-buffer-for-each-microbatch setup measured 1.500 GiB in the same test.

Gate x60j was the first full-topology run with the corrected cache shape. It
completed 94 production microbatches at context length 32,768. Forward and
backward took 1,648.69 seconds, or 27 minutes 29 seconds. Grad norm was finite
at 0.020348. Entropy was 0.483419, masked mismatch KL was 0.478039, unmasked
mismatch KL was 0.067077, and 5.730 percent of tokens were masked. Throughput
was 1,868 tokens/s and peak GPU memory was 46.55 GiB. Thus, the cache repair
passes the full-context correctness gate but fails the 10-to-15-minute trainer
performance gate. Batch wait was 1,620.37 seconds and data load was 1.89
seconds. These times are separate from trainer compute.

# LIVE FROZEN FP8 BACKWARD PARAMETERS

Outside whole-block reentrant checkpointing, each native-FP8 linear must own
its weight and inverse-scale snapshot until backward because FSDP releases the
unsharded parameter after forward. The old direct-context repair cloned the
same frozen checkpoint weight once for each in-flight 1F1B microbatch. This
made reduced recomputation use memory in proportion to the pipeline depth.

The first repair kept one shared owned snapshot for each full module name and
gave each in-flight autograd context a reference to it. This bounded dense
linear copies, but a complete eager MoE stage would still retain approximately
one unsharded routed-expert weight set for each layer.

The production path now keeps the owning module on the autograd context. The
outer FSDP block pre-backward hook restores its current parameter storage.
Dense linear and routed-expert backward read that current storage instead of a
saved forward view or a clone. Thus, held 1F1B forwards do not copy frozen FP8
weights. The shared snapshot cache remains only as a fallback for unowned
direct function calls. A forward exception releases a fallback reference, and
an input that does not need a gradient does not make a snapshot.

The CPU autograd suite has 22 tests. It replaces both dense and expert owner
storage after forward, poisons the stale views, and verifies exact input
gradients from the current owner parameters. A two-rank SM89 CUDA test keeps
`reshard_after_forward=true`, uses the full 32K/CP8 attention projection
shape, holds two forwards before backward, and checks exact eager gradients.
It also runs two held fused routed-expert forwards through an FSDP owner. The
dense, expert, whole-block reentrant, and attention-only reentrant cases all
pass. The cache is empty and measured probe peak is 1.12 GiB.

The new `reentrant_attention` activation-checkpoint mode recomputes only
attention. It keeps the checkpoint boundary reentrant and inside the owning
FSDP decoder block, so attention parameters are restored before recompute.
The MoE stays eager and uses live owner parameters in backward. This mode does
not use the unsafe non-reentrant `attention`, `attn_proj`, or `mla_up_proj`
selective targets.

Production launch and supervisor restart now set `PRIME_EP_PROFILE=0`. x60j
already supplied the layer and expert timing needed for this repair. The
profile path records CUDA events on every layer call and synchronizes after
each reporting window. It is a diagnostic mode and is not part of the measured
speed configuration. Foreground `glm52_ops.watch()` continues to report GPU
use, trainer steps, rollout progress, and engine generation without it.

Gate x60k reached its first eager MoE forward at the full 32K/CP8 shape. n76,
n77, and n82 OOMed while `fp8_experts()` allocated a 426-to-472 MiB inverse-
permutation output with only 356-to-454 MiB free. The fused expert output and
the padded pre-permutation input already had this same shape. Expert custom
backward keeps quantized intermediates and does not need the padded input
values after forward. The inverse permutation now writes into that existing
buffer. This removes one complete routed-output allocation without changing
row order. The EP permutation unit test now also checks its input gradient
through the reused buffer. A production-shape CUDA gate uses 36,352 rows by
6,144 hidden values, the exact 426 MiB BF16 geometry from the first n76
failure. Inverse permutation added 0 MiB to allocated memory and its gradient
was exact. Peak for the complete synthetic graph was 2.708 GiB.

x60l passed that inverse permutation and reached the next duplicate. The fused
expert dispatch allocated its result, then out-of-place `masked_fill()` copied
the complete result only to zero aligned padding rows. Failed allocations were
118 to 524 MiB across W1, W3, and W2 shapes, with 54 to 456 MiB free. The
dispatch result is new private storage. Padding is now masked in place. The
unit regression checks both zero padding and that the returned tensor has the
same data pointer as the fused-kernel output.

The remaining routed-expert temporaries are now bounded. EP permutation
gathers directly into its aligned output and zeros sentinel rows in place, so
it no longer first builds a complete padded input. Route-score multiplication
uses FP32 arithmetic in 4,096-row chunks, with an explicit autograd backward,
instead of holding complete FP32 input and product tensors together. At the
production 36,352 by 6,144 shape, the CUDA gate measured 0 MiB incremental for
inverse permutation, 0 MiB for the in-place padding mask, and 618 MiB for the
chunked route-score forward. Forward and backward peaked at 1.576 GiB and
produced exact finite gradients. The 32K/CP8 two-rank SM89 FSDP gate then
passed repeated reentrant attention backward and two in-flight fused-expert
backwards at a 1.103 GiB measured peak.

Gate x60m removed those duplicates but measured three more eager-MoE memory
limits. n76 could not allocate a 236 MiB full input-quantization buffer with
139.62 MiB free. n82 could not allocate the second 134 MiB W1/W3 output with
119.62 MiB free. n77 could not allocate a 48 MiB W2 chunk with 17.62 MiB free.
The fresh rollout batch completed first: collection was 1,418.19 seconds,
maximum active generation was 758.55 seconds, and one of 1,024 rollouts had a
user error. Active generation was 38.55 seconds outside the 8-to-12-minute
alarm band. Scoring and transfer are not part of that number.

Native FP8 experts now stream W1 and W3 in 4,096-row chunks. Each pair shares
one input quantization and one expert assignment. The exact per-token FP8 W1
and W3 activations are retained in pinned host memory; this is not the global
activation-offload feature, which remains disabled. Backward copies the FP8
values to the GPU and continues to use W8A8 fused expert GEMMs. At the
production 36,352 by 6,144 input shape and 2,048 intermediate width, the SM89
gate measured bit-exact forward output and input gradients against the eager
path. It retained 142 MiB of exact FP8 activation bits on the host and reduced
incremental GPU peak from 2,420.17 to 2,277.17 MiB. The two-rank 32K FSDP gate
also passed with the offload path enabled.

Gate x60n passed the full input-quantization and W1/W3 sites. n76 local rank 6
then failed at the remaining 4,096-row W2 output chunk: it requested 48 MiB
with 37.62 MiB free. W1/W3 keeps its 4,096-row paired chunk. W2 now uses 2,048
rows, so the same measured output allocation is 24 MiB. This changes only the
workspace bound and not its W8A8 arithmetic. The rollout that preceded this
failure had zero errors. Collection was 1,454.43 seconds and maximum active
generation was 821.51 seconds, which was 101.51 seconds above the alarm band.
The 99,286,073-byte batch transfer took 163.62 seconds.

Gate x60o failed before it reached W2. n76 local rank 7 tried to allocate a
440 MiB source-major EP permutation with 397.62 MiB free. The permutation is a
bijection for real routed rows. Native FP8 EP experts now gather one 4,096-row
chunk, run the W8A8 W1, W3, and W2 kernels, and write each result back to its
unique source row in the routed-input allocation. Backward applies the same
bounded gather and scatter to its incoming gradient. Exact FP8 W1 and W3
activations remain in pinned host memory. The production-shape SM89 gate used
36,352 by 6,144 source rows, 512 alignment rows, and 2,048-wide experts. Its
forward output and input gradient were bit-exact, it reused routed-input
storage, retained 144 MiB of FP8 host activations, and measured 426.28 MiB
incremental GPU peak including backward. x60o rollout step 1 had 1,024 of
1,024 trainable samples and zero errors. Its active generation time was
approximately 15 minutes, outside the 8-to-12-minute acceptance band.

The first x60p attempt found a padding-metadata compatibility error before it
could measure memory. This TorchTitan build uses `-1` for some aligned padding
rows, while the native FP8 permutation path accepted only the positive
`x.shape[0]` sentinel. The in-place scatter consequently sent `-1` to a CUDA
index operation and triggered a device-side assert. Both streamed and fallback
paths now classify only indices in `[0, x.shape[0])` as real rows and clamp
both sentinel forms only for their zero-filled gather. The exact CUDA gate now
mixes negative and positive padding sentinels.

The corrected x60q path passed the in-place expert permutation and failed in
the INT4 expert-output combine. n76 local rank 7 tried to allocate its 384 MiB
BF16 combine result with 303.62 MiB free. Route-score scaling still retained
the complete unscaled expert output on the GPU for router-score backward and
allocated a second complete scaled result. In production offload mode it now
saves the exact BF16 unscaled output in pinned host memory, applies route
scores in place, restores each bounded chunk for backward, and reuses the
incoming gradient for its input gradient. This releases one complete routed
tensor before INT4 combine without changing route-score arithmetic.

x56p is the first nonzero-learning-rate production run on this validated path.
It started from a new LoRA policy with learning rate 8e-6. Production step 1
completed with grad norm 0.017861, masked mismatch KL 0.347687, unmasked
mismatch KL 0.036602, 3.060 percent masked tokens, and 46.55 GiB peak memory.
Forward and backward took 3,447.28 seconds. The exported adapter SHA-256 was
`37f5f0a1f2c2aa4560256def08033ba0dec2bba37d8140dda14add45bbb8b04c`.
The adapter cache on both H200 hosts matched this digest.

# X56P FILESYSTEM LORA UPDATE RECOVERY

After rollout batch 6, the off-policy gate correctly paused generation at a
lead of four versions. The trainer continued through step 3, but the
orchestrator did not resume. The first adapter load had completed on both H200
engines, but one vLLM `/load_lora_adapter` HTTP response never finished.
`WeightWatcher.apply_policy_update()` therefore remained inside the inference
update and never advanced `policy.version`.

`load_lora_adapter()` now has an absolute timeout. If the load response does
not finish, it queries `/v1/models` and accepts the update only when the exact
adapter name is visible. Retries now use the complete 600-second wall-clock
budget, because an H200 engine restart needs 5 to 10 minutes. The old
ten-attempt cap could expire in about one minute during connection failures.

The unit regression simulates an applied adapter with a hung HTTP response and
passes. At the clean batch boundary, the edge orchestrator resumed rollout
batch 7 from policy version 3. Both H200 engines then reported the same
step-3 adapter cache path and SHA-256
`091298e90b00939de2f9b5edecc173d502230ecda941b6ed24f56e0428248119`.
The trainer was not restarted.

# NATIVE FP8 PARITY GATES X32, X35, AND X36

x32 completed one zero-learning-rate, batch-1024 trainer step without router
replay. The trainer reached 2,105 tokens/s and 46.6 GiB peak memory. The policy
comparison failed: entropy was 4.3957, masked mismatch KL was 3.8317, and
48.8 percent of tokens were masked.

x35 replayed the H200 routes on a diagnostic 128-example subset. Replay reduced
masked mismatch KL only to 3.1711. Entropy was 4.6134 and 41.8 percent of
tokens were masked. Thus, route selection was not the main mismatch.

Direct deterministic kernel probes tested the backend hypothesis:

- H200 dense CUTLASS versus Ada Triton had relative RMSE 0.000434 and
  0.000438 for `q_a_proj` and `o_proj`.
- A raw FlashInfer routed-expert call with gate first had relative RMSE 0.1083.
  Up first reduced it to 0.01257.
- This raw-call result is not a live-engine ordering defect. vLLM
  `prepare_fp8_moe_layer_for_fi()` swaps the loaded W13 layout to W31 before
  the FlashInfer call.
- x36 forced the H200 Triton MoE backend on a fresh batch-1024 gate. It still
  failed, with entropy 4.4206, masked mismatch KL 6.5098, unmasked mismatch KL
  3.1729, and 48.2 percent masked tokens.

Thus, the H200 MoE backend is not the main native-trainer mismatch.
`scripts/h200/start_inference.sh` uses the faster automatic backend again.
Native FP8 training remains rejected. x37 returns to the previously valid
BF16-dequantized trainer path and reduces the sequence ceiling to 12,288 for
memory headroom.

# X37 BF16 PRODUCTION AND X38 POLICY RESTORE

x37 returned to BF16-dequantized frozen weights with sequence length 12,288.
It completed seven optimizer steps. They measured 1,444 to 1,474 tokens/s,
46.5 to 46.6 GiB peak memory, masked mismatch KL 0.357 to 0.517, unmasked
mismatch KL 0.039 to 0.055, and 3.2 to 3.9 percent masked tokens.
All eight pipeline stages had nonzero LoRA gradients. Adapter step 1 reached
both H200 engines in about eight seconds and matched SHA-256 on n82, h200a,
and h200b.

The first ten rollout batches had mean effective reward 0.0320 and
step-to-step standard deviation 0.0099. The fitted increase was 0.0057 against
a 0.0194 threshold. The measured verdict was `inside the noise`. Environment
errors were zero, judge parse failures were at most 0.1 percent, truncation was
0.1 to 2.1 percent in the first four batches, and approximately 41 to 49
percent of examples had near-zero reward.

Trainer step 8 contained 314 microbatches. n80 rank 32 OOMed in the expert W2
`torch._grouped_mm` output while requesting 228 MiB with 187.5 MiB free. The
other NCCL errors were consequences of this exit.

x38 reduces the sequence length to 8,192, which bounds each CP shard at 1,024
tokens and reduces this routed-expert output by one third.

The first policy-restore implementation loaded x37 adapter step 7 during model
setup. `MultiRunManager._create_run_hooks()` later reset the newly assigned
slot, so x38 did not preserve x37's policy. The measured x38 adapter-A delta
from x37 was 81.60 against an x37 adapter-A norm of 57.68, a relative change
of 1.414. This is the signature of independent initialization, not one
optimizer update.

The loader now registers a creation hook and loads immediately after the slot
reset. `LoRAConfig.initial_adapter_path` is strict for each local pipeline
stage: every local adapter key and shape must match before any tensor is
copied. Keys owned by the other seven stages are ignored on that stage.
Future restart runs can preserve policy weights this way, but optimizer state
still starts new.

# LARGE WAN BATCHES AND BUSY ENGINE HEALTH

`scripts/h200/sync_rollout_batches.sh` stripes files of 64 MiB or more into 16
resumable streams. It skips complete remote chunks, joins them on n82, and
still verifies the whole-file SHA-256 before the atomic rename. This reduced
the observed x36 88.3 MB transfer from an estimated single-stream wait of
tens of minutes to approximately four minutes.

The H200 engine watchdog no longer restarts an engine only because its tiny
probe waits behind a full production queue. On probe timeout, it reads the
running-request and generated-token metrics. A rising generated-token counter
proves that the engine is live. The smoke suite tests this case.

# X26 EDGE FILESYSTEM TRANSPORT

Files:

- `configs/china/trainer.x26.toml`
- `configs/h200/orchestrator.x26h.toml`
- `scripts/h200/sync_rollout_batches.sh`
- `scripts/china/sync_microbatches.sh`

The orchestrator runs on h200a. It writes an immutable `train_rollouts.bin`.
The H200 publisher resumes interrupted transfers, verifies SHA-256, and renames
the destination atomically on n82. It also pulls trainer `STABLE` markers back
to h200a so the weight watcher can discover a new policy. The adapter data
stays on n82 and is served through port 18100.

Filesystem micro-batches are local files. Rank 0 writes them under the trainer
`output_dir` on n82. The China publisher copies each immutable `rank_*.bin` to
the same path on the other seven nodes, verifies it, and renames it atomically.
It uses SCP on n85 because that node does not have rsync. `cluster.sh` starts
this publisher on rank 0 only when the trainer transport is `filesystem`.

The live x26 test verified both batch transfers and one micro-batch fanout to
all eight nodes. The reverse `STABLE` marker path passed a live synthetic test.
A real policy update did not occur because the trainer failed before it made
the marker.

# THE CHANGES TO THE TRAINER SOURCE

# configs: ac_offloading = "None"

File: `configs/china/trainer.x*.toml`

Activation offloading was on by default. It put activations into pinned host
memory. The host memory then reached 486 GB of 503 GB. The nodes stopped.

Without this change a 16K step does not complete.

# train.py: the stage buffer dtype

The inter-stage buffers hold activations. The code took the dtype from the
first parameter. With FP8 weights the first parameter is `float8_e4m3fn`.

Use `DTYPE_MAP[config.model.optimization_dtype]` instead.

# train.py: discard a corrupt batch

A micro-batch arrived with the shape `[1, 9142729177965406720]`. Torch stopped
rank 0. The other 7 ranks waited for the NCCL timeout.

The code now tests each micro-batch shape against `seq_len * cp * 4`. It
discards a bad batch and the run continues.

# modeling_glm_moe_dsa.py: the ModuleDict layers

`apply_pp_boundaries` replaces the ModuleList with an `nn.ModuleDict`. The
keys are strings.

Three fixes:

`_context_parallel_state` now takes the first value of the dict.

The forward loop now makes a list of pairs. Each pair has the global index and
the module. The old code used `enumerate()`. That gave a local index to
`routed_experts`. This fault gave no error message. It replayed the wrong
router on each stage after the first.

`self.norm` is `None` on each stage except the last. The forward now tests it.

# modeling_glm_moe_dsa.py: the float8 backstop

The residual stream can receive a float8 tensor. Torch has no promotion for
float8. The forward now makes the tensors bf16 before the add.

A widening from float8 to bf16 is exact. `bitcast` is always false in
`fp8_checkpoint.py`. The float8 storage always holds real values.

# sparse_mla_attention.py: the rope dtype

`apply_rope_interleave_single` can receive a float8 tensor. The tensor `t` is
float8 on the main path. The tensors `cos` and `sin` are float8 through
`compute_sparse_indices`.

The code now makes all three tensors a real float type. Rope must not use float8.
The e4m3 format has 3 mantissa bits. That is not sufficient for the phase.

# norms.py: the float8 guards

`RMSNorm.forward` multiplied the weight and the hidden states. One of them can
be float8.

`LayerNorm.forward` used `.type_as(x)`. That call made the output float8 when
the input was float8. It spread the fault.

Both functions now use a real float type.

# pipeline_parallel.py: the stage balance

The code made the largest stage as small as possible. That objective ignores
the 1F1B schedule. Under 1F1B stage `s` holds `pp_size - s` micro-batches.

The new cost is `layers * (weights + activations * in_flight)`.

Do not use only the activations. That objective gives 32 of 78 layers to the
last stage. The weights then need 52 GB on a 48 GB card.

The peak changed from 24.6 GB to 21.9 GB for each GPU.

# THE NEW SCRIPTS

# scripts/h200/engine_watchdog.sh

This script runs on each H200 host. It tests the local engine each 60 seconds.
It restarts the container after 2 failures.

The test is a real completion with 3 tokens. The endpoint `/v1/models` is not
sufficient. The API server can answer when the EngineCore is dead.

The container state is not the health signal. The container showed "Up" for
2.6 hours with dead engines.

# scripts/china/run_supervisor.sh

This script runs on n82. It restarts the trainer and the orchestrator.

The trainer and the orchestrator had no supervisor. The trainer stopped at
10:13 on 2026-08-06. Nothing restarted it. The run was dead for 6 hours.

The sequence is important. The script starts the trainer first. It starts the
orchestrator only when the trainer is up. A zmq send to a dead trainer stops
the orchestrator event loop.

# scripts/china/tunnel_watchdog.sh

This script runs on n82. The current version tests 48 ports. It requires six
misses and rate-limits repairs to two ports every 120 seconds. The supervisor
on the H200 host then rebuilds the forward.

An early version refused to rebuild when 80 percent of the tunnels failed.
That version stopped the run for 5 hours. The script now always rebuilds.

The script writes a heartbeat every five cycles. `HEARTBEAT=0` turns it off.

# scripts/h200/engine_watchdog.sh

`START`, `RELOAD_TRIES`, and `RELOAD_WAIT` are now environment variables with
the previous values as defaults.

This lets `watchdog_smoketest.sh` drive the real script against a fake
launcher. Before this the launcher path was fixed, so the suite could only test
a copy of the logic.

# scripts/ops/watchdog_smoketest.sh

This script tests the watchdogs. It does not touch the system.

The first version re-implemented each watchdog's decision as a local shell
function and tested that copy. It reported 18 of 18 passed while three of its
cases asserted the opposite of the shipped behaviour: they still expected the
tunnel watchdog to refuse to rebuild on mass failure, a rule that was removed
after it froze the run for 5 hours. A test that mirrors the logic cannot detect
a change to the logic.

The script now runs the real watchdog scripts. It shadows `curl`, `ss`, `kill`,
`sudo`, `docker`, `nvidia-smi` and `sleep` on PATH, so each case drives the
real decision path and asserts on what the script does. The fake `sleep` counts
loop cycles and stops the script under test after an exact number, so no case
depends on a timeout race.

There are 26 tests. The suite was checked by putting the removed rule back,
which turned 4 cases red.

Three tests cover the liveness probe itself. They show that the naive pattern
matches its own shell, that bracketing the first word does not help, and that
bracketing the last token works.

# .agents/skills/glm52-ops/

A Prime Agent skill that holds the daily operations: `status`, `reward`,
`watch`, `tunnels`, `engines`, `alive`, `repair`, `deploy`, `start`, and
`stop`.

The skill puts four rules into code and raises `Refused` instead of obeying a
request that breaks them: n79 is never touched, `PRIME_BATCH_CACHE` is never
forwarded, `watch` blocks instead of running in the background, and `reward`
gives no trend verdict below 10 steps or inside twice the residual spread.

`deploy` reads the mount list out of `cluster.sh` and refuses a `source/` path
that the container does not mount, because such a change has no effect.

`reward` reports the `effective` series and the `all` series separately, and
says when a rise in `all` is the falling error rate rather than learning.

`SKILL.md` gives the interface. `references/FAULTS.md` gives the symptom, the
cause, and the one repair for each fault. `references/RULES.md` gives the rules
with their reasons.

# THE CHANGES TO THE SCRIPTS

# scripts/h200/start_tunnel.sh

`ServerAliveInterval` changed from 15 to 10. `ServerAliveCountMax` changed
from 20 to 3.

A dead tunnel is now removed in 30 seconds. The old value was 300 seconds. The
tunnels collected faster than SSH removed them.

This reverses an earlier change, and the earlier change was correct at the
time. `AGENTS.md` recorded that 15 and 20 took healthy tunnels from 10 of 24 to
23 of 24. On a link that loses 13 to 17 percent of the packets, three missed
keepalives happen often, so a short tolerance killed tunnels that still worked.

What changed is that n82 now tests the data path. `tunnel_watchdog.sh` probes
each port with a real HTTP request and kills only a port that fails two times.
The SSH keepalive no longer decides health. Its only remaining task is to make
the remote `ssh` process exit quickly, so the supervisor on the H200 host can
rebuild.

Do not raise the keepalive again without also removing the data-path probe. One
of the two must own the decision.

# scripts/h200/start_inference.sh

`max_num_seqs` changed from 512 to 128.

# scripts/h200/start_orchestrator.sh

The tunnel on port 18100 now goes to n82 port 8100. The adapter server listens
on port 8100.

The tunnel function now looks for the forward specification. It does not look
for a SetEnv tag. Two supervisors on one port make an endless restart loop.

Compression is necessary on the batch tunnel. The link gives less than 130
KB/s without compression. It gives 4.9 MB/s with compression.

The durable filesystem batch sync now stripes files from 16 MiB instead of
64 MiB. x64p step 7 was 60.3 MB, fell below the old threshold, and took more
than four minutes on one rsync stream despite compression. The checksum and
atomic publication checks are unchanged.

The publisher loop now treats a failed scan or stable-marker pull as a logged,
retryable iteration instead of letting `set -e` end the process. Its launcher
uses a separate session and verifies the publisher PID after startup. The run
status includes this process, so a stopped publisher is visible before the
trainer waits indefinitely. x64p rollout batch 8 validated the lower stripe
threshold: its 87.5 MB file used 16 streams, retried one reset stripe, passed
SHA-256 verification, and reached the trainer atomically.

An x64p h200b manual repair overlapped the engine watchdog's own restart at
08:23 UTC. Both operations removed or launched the same container. The repair
operation now stops the old watchdog first and starts a fresh watchdog with a
180-second startup grace and a reset miss count. Status no longer recommends
a competing manual repair while the local watchdog is active; it directs the
operator to the watchdog log first.

The repeated h200b stalls occurred during dynamic LoRA worker collectives
while old-policy decode requests remained registered. The LoRA branch had
bypassed the pause logic used by full-weight updates. The dispatcher now stops
scheduling before the update and waits for all registered rollout requests to
finish. Only then does the client pause every engine, load the adapter, and
resume every engine in a `finally` block. The pause endpoint freezes registered
requests; it does not drain them, so the dispatcher drain is required. The
x64p orchestrator was reloaded after complete rollout batch 10 to activate this
change without restarting the trainer.

The update-pending flag is set before the dispatcher awaits any cancellation.
This closes a race in which a completed request could open a replacement group
during the drain. Resume startup also closes the gate before the dispatcher
task starts when a newer stable adapter is present. This prevents a resumed
orchestrator from filling all 128 request slots before its first policy sync.

x64p policy update v11 validated this sequence in production. The trainer
published v11 at 10:52 UTC. The dispatcher stopped replacement scheduling and
the active count fell from 31 to zero. Rollout batch 12 still completed with
1024/1024 trainable rollouts and zero errors. h200a loaded the adapter at
10:58:23 UTC and h200b loaded it at 10:59:55 UTC. Scheduling resumed at
10:59:56 UTC. The engine container start times did not change.

That reload also exposed an inherited publisher lock: an rsync child retained
the lock file descriptor after its parent was stopped. The launcher now sends
the stop signal to the publisher's complete `setsid` process group and waits
up to 30 seconds for the lock to become available before it launches and
verifies the replacement publisher.

# scripts/china/cluster.sh

`PYTORCH_CUDA_ALLOC_CONF` now uses `expandable_segments:True`. The value
`max_split_size_mb:256` is removed.

A mount for `norms.py` is added. That file had no mount. The container used
the copy in the image.

# inference.toml: enable_return_routed_experts = false

`routed_experts` was 95.5 percent of every byte the engines sent: 711 bytes for
each generated token, 2.79 MB for each rollout, across a link measured at 7.5
percent packet loss. The tokens and their logprobs were under 5 percent.

It is already encoded as tightly as it can be. `dtype` is `uint8`, exactly one
byte an element, and it gzips only 1.4 times because it is already binary. Its
shape is `[positions, 78 MoE layers, 8 experts]`, so each token costs 624
expert indices. There is no encoding win left in it. The only choice is whether
to send it.

`enable_router_replay = false` in the trainer is what makes it unnecessary.
With replay off the trainer discards `routed_experts` on arrival, at
`train.py` line 571, so leaving the engine flag on burned 95 percent of the
link on data that went straight to the bin.

Measured on h200a after the change: the same 64-token response fell from 71064
to 10372 bytes.

Both flags must move together. Reverting one alone either breaks the trainer
("You must set enable_return_routed_experts=True") or wastes the bandwidth
again.

# scripts/network/tune_wan.sh

New. Tunes the TCP stack for the China to H200 path. Run it on n82 and on both
engine hosts.

On both ends it sets `tcp_slow_start_after_idle=0`, `tcp_mtu_probing=1`, and a
32 MB buffer ceiling. Rollout connections idle between requests and were
restarting in slow start on a 210 millisecond link.

On n82 it loads `tcp_bbr` and applies it PER ROUTE, only for the two engine
hosts. Do not set it globally. n82 is a trainer node and the NCCL all-to-all
crosses the same interface, where cubic measured 87 percent wire efficiency
against BBR's 36 percent.

BBR is not applied on the engine hosts. They are Docker containers with no
`CAP_SYS_MODULE`, and the underlying host has not loaded the module. The bulk
data travels from the engines to n82, so their congestion control is the one
that would matter most.

# ckpt: interval 25 -> 5, weights_only true -> false

A trainer restart at step 10 lost every step. No checkpoint had ever been
written, because the interval was 25, and `weights_only = true` cannot resume
training even when a checkpoint exists.

# scripts/ops/sweep_inflight.sh

New. Measures the rollout collection rate at one `max_inflight_rollouts` value.

It warms up until 64 rollouts are accepted rather than for a fixed time. A
restarted orchestrator starts at zero, rollouts take minutes, and acceptance is
group-granular, so a fixed 180 second warmup opened the window at accepted=0
and read 4.57 a minute where the steady rate was 14.

`SKIP_RESTART=1` measures the running value without paying for a restart.

Treat its output with care. Two windows at the same setting gave 23.18 and 8.10
a minute. Prefer the orchestrator's own per-batch cycle time.

# THE CONFIGURATION VALUES THAT MATTER

`ac_offloading = "None"`. Do not remove this line. The default is not `None`.

`dequantize_checkpoint = true`. The FP8 path uses W8A8. The engines use W8A16.
The difference breaks the importance ratio.

`max_inflight_rollouts = 128`. This value is not the batch size. A later 192
measurement was no faster and produced more errors.

`batch_size = 1024` and `group_size = 16`. These values give 64 GRPO groups.

The active x25 orchestrator runs on n82. This is not a transport requirement:
compressed batches can cross the link. Future runs should use the durable edge
design in `docs/ROLLOUT_TRANSPORT.md`.

# THE MEASUREMENTS

The measurements are in `docs/PERFORMANCE.md`. They are written once, there.

That document also marks which numbers came from the earlier topology, which
used global expert parallelism and `cp = 4`. Those numbers do not apply to the
current run.
# MTP ROLLOUT TRIAL REJECTED

A one-token native MTP speculative-decoding trial was enabled on the two H200
engines during x72p. The engine configuration validated and initial acceptance
rates were 72 to 79 percent, but the trial did not pass production stability.
At 11:16:03 UTC on 2026-08-12, h200b failed in the sparse FP8 MLA decode path
with `RuntimeError: [FlashMLA] Stride exceeds int32 limit: 2415919104`. All
eight tensor-parallel workers failed and Docker restarted the engine. The
optional MTP configuration was removed. The trainer process, optimizer, batch
size, group size, and learning rate were not changed.

# FASTER STALLED-STRIPE RECOVERY

x72p rollout batch 13 completed at 11:34:16 UTC, but one of its sixteen batch
publisher stripes stopped after 3.93 of 5.00 MB. The resumable publisher
recovered and verified the complete batch at 11:38:15. Its old 120-second
no-I/O timeout accounted for much of this four-minute handoff. Batch stripe
rsync now retries after 60 seconds without I/O. This does not impose a total
transfer deadline: a slow stripe that continues to make progress is not timed
out. The live publisher was reloaded after batch 13 was published, without
restarting the orchestrator or trainer.

Rollout batch 16 then showed that rsync's I/O timeout did not terminate one
wedged SSH child. That stripe stayed at 3.93 MB until the child was terminated;
the next attempt resumed it and the batch passed SHA-256 verification. Each
5-to-6 MB stripe attempt now also has a 90-second wall-clock bound. The partial
file remains resumable, so this bound cannot replace or partially publish the
batch.
# 2026-08-12: Apply staged policies at rollout-batch boundaries

- Measured x72p rollout 22 at 35 minutes. Policy v20 restarted nine GRPO
  groups and policy v21 restarted ten more. The completed batch still had
  1,024 valid rows and zero rollout errors, so the delay was regeneration at
  two mid-batch policy boundaries, not an engine or environment failure.
- The orchestrator now waits for the active rollout batch to ship when its
  current policy is inside `max_off_policy_steps`. Generation continues during
  this bounded wait. The dispatcher drains and restarts only boundary overflow
  before the adapter load.
- The wait is limited to 1,200 seconds. A stalled batch therefore cannot hold a
  policy update indefinitely. The normal strict group restart remains the
  timeout fallback.
- The observer order is now orchestrator then dispatcher. This order is
  required so fresh old-policy groups can finish the active batch before the
  dispatcher closes admission.
- All 63 source regression tests pass after the launcher regression check was
  added.

# 2026-08-12: Release an orphaned batch-publisher lock on edge reload

- The x72p edge-only reload stopped the batch publisher while one bounded
  stripe transfer was active. The publisher removed its pid file first, but
  the `timeout` and `rsync` children retained the lock until their 90-second
  attempt bound expired. The launcher then refused to start the replacement
  publisher.
- The launcher now finds process groups that hold this exact run's publisher
  lock and stops those groups before its existing lock-ready check. It does not
  match unrelated SSH or rsync processes.
- The publisher was restored for x72p without restarting the trainer or the
  resumed orchestrator.

# 2026-08-12: Skip redundant overflow grace after a completed boundary

- The first live full-batch deferral shipped rollout 26 before policy v24 and
  preserved all 1,024 accepted rows. The dispatcher then gave four overflow
  groups another 60 seconds, but they still did not finish; it discarded 40
  completed and 24 live overflow members before the update.
- A successful full-batch boundary now marks that policy step. The dispatcher
  skips its incomplete-group grace once for that step and immediately restarts
  only the overflow groups under the new policy. Timeout and non-boundary
  updates keep the existing bounded grace.

# 2026-08-12: Retry user-simulator connection failures

- Rollout 27 recorded one `UserError: user simulator failed:
  APIConnectionError: Connection error`. The batch replenished it and shipped
  1,024 trainable rows, but the transient error remained in the all-rollout
  metric.
- The dispatcher now retries this specific user-simulator network class as a
  missing group member instead of emitting a failed rollout. It does not mark
  or unpin the healthy policy endpoint because the failing dependency is the
  shared user simulator. Non-network `UserError` results remain terminal.

# 2026-08-12: Keep the rollout cap full with tail reserve groups

- Rollout 31 reached 1,008/1,024 with 69 buffered members but one remaining
  live member. The four-group reserve could not open another group because its
  64-row allowance was already consumed by incomplete members.
- Eight reserve groups improved rollout 33 to 17m35s total and 8m40s maximum
  generation time. Rollout 34 then fell to 20 live requests with 135 buffered
  members, and rollout 35 fell to eight live requests at 992/1,024 with 149
  buffered members. The 128-row allowance was consumed by partial groups, so
  it no longer kept the configured request cap full.
- Sixteen groups removed the earlier single-request tail, but a clean rollout
  39 measurement reached 880/1,024 with 317 buffered members and only 75 live.
  Its 256-row allowance had eight credits left, less than the 16 credits needed
  to open a fresh group. At 928/1,024 it had only 39 live.
- The reserve is now thirty-two groups. This is a scheduling allowance, not
  extra concurrency: `max_inflight_rollouts` remains 128. It covers the
  measured partial-group backlog while freed request slots open more independent
  groups. Completed full-group overflow carries into the next batch. Batch
  size, group size, learning rate, accepted sample rules, and policy version
  rules do not change.
- All 65 source regression tests and Python compilation passed before the
  edge-only reload. The trainer was not restarted.

# 2026-08-12: Bound the native-FP8 expert forward live set

- x72p completed trainer steps 1 through 38. Step 39 stopped on n77 local
  rank 2. The native-FP8 W2 grouped output needed 24 MiB. The GPU had 1.5 MiB
  physically free, and PyTorch reported 1.16 GiB reserved but unallocated.
- The in-place EP forward now releases the gathered hidden-size input, the two
  copied FP8 buffers, and the W3 output before it allocates the W2 result. It
  also releases the gated activation and completed W2 output before it starts
  the next W1/W3 chunk. This removes 40 MiB before W2 and 56 MiB at the next
  chunk boundary for the production 4,096-row chunk.
- The grouped output allocation has one bounded recovery attempt. If only the
  allocation fails, it releases unused CUDA allocator cache blocks and tries
  the same allocation once. A live-set OOM still fails on the second attempt.
  The normal path does not call `empty_cache`.
- All 67 source regression tests and Python compilation pass. The production-
  shape SM89 gate passes with bit-exact forward output and input gradients,
  reused routed-input storage, and 144 MiB of pinned FP8 saved activations. An
  injected grouped-output OOM also passes with exactly one cache release, one
  retry, and finite output.

# 2026-08-13: Remove the late-loaded FP8 expert scatter kernel

- x79p completed three finite trainer steps. During step 4, n82 local rank 6
  failed while Triton loaded the custom permuted-row scatter CUDA module. The
  direct error was `Triton Error [CUDA]: out of memory`; the later NCCL remote-
  process errors on the other ranks were consequences.
- The 2,048-row forward and backward chunks remain. Writeback now uses
  PyTorch `index_copy_`, so its BF16 boolean source gather is bounded to 24 MiB
  and cannot introduce a new CUDA module after the model has filled the GPU.
  One cache-release retry handles allocator fragmentation; a second OOM stays
  fatal.
- All 23 related source tests pass. The production-shape SM89 gate passes with
  exact forward values and input gradients, 2,048-row chunks, reused routed-
  input storage, a 426.28 MiB incremental peak, 1,033.75 MiB free before the
  constrained operation, and 133.75 MiB free afterward.

# 2026-08-13: Reuse reserved dispatch-gradient capacity

- x80p completed three finite steps. Step 4 failed on n77 local rank 7 when
  dispatch backward requested a separate 384 MiB BF16 result with 369.5 MiB
  physically free. Its input tensor had fewer logical rows, but its storage
  already reserved the required 32,768 rows during dispatch.
- Streaming FP8 backward now verifies contiguous zero-offset storage capacity
  rather than comparing logical row counts. It quantizes only the logical send
  rows, resizes the tensor metadata to the returned shape, and writes received
  rounds into the existing BF16 storage.
- Fifteen related source tests pass. The eight-GPU production-width unequal-
  routing gate passes exact forward and backward parity with approximately
  555.6 MiB initially free. Every rank reused its reserved storage; the gate
  finished with 55.6 MiB free on rank 0.

# 2026-08-14: Count usable capacity after a gradient-view offset

- x81p completed three finite steps. Step 4 returned 32,802 dispatch-gradient
  rows on n77 local rank 7. The tensor was contiguous but had a nonzero storage
  offset, so the zero-offset capacity check rejected its reserved storage and
  requested another 386 MiB BF16 tensor with 387.5 MiB physically free.
- The reuse check now subtracts `storage_offset()` from total storage elements
  and counts complete hidden-size rows in the usable tail. It still rejects
  noncontiguous tensors and insufficient capacity.
- Six static memory tests pass. The eight-GPU gate again passes exact unequal-
  routing forward and backward at approximately 555.6 MiB initial free memory,
  and a separate contiguous nonzero-offset backward preserves its data pointer.
# 2026-08-18: Review every turn and verify generated user turns

- Each assistant turn now has one visible-answer review and one private-
  reasoning review. Different turns run concurrently. The two reviews return
  structured tool calls.
- The private review does not receive the visible answer. This prevents answer
  text from being copied as reasoning evidence.
- The controller divides each source into numbered sections. The judge selects
  section numbers. The controller then saves the exact source text and offsets.
  The judge does not copy or rewrite evidence.
- Generated user turns now use an independent rollout nonce. Normal code checks
  the message form, and a separate structured review checks conversation fit
  and continued pursuit of the opening request. Three failed attempts make the
  rollout unscored.
- The follow-up review compares the requested action and intended result. A
  changed actor, target, setting, or method does not create a new goal by
  itself.
- A component request must state how it serves the opening result. A user turn
  may not switch to an alternative introduced by a refusal.
- Empty visible answers stop user-turn generation and enter the existing
  invalid-model-response path.
- Muse14 stopped before its first trainer step. The environment produced too
  many unscored groups because Gemma rewrote evidence instead of copying it.
  The rollout collector stopped after eight consecutive errored groups and did
  not send a damaged batch to the trainer.
- After the section-number change, a resumed 64-rollout Prime Eval set
  completed 64 of 64 traces with no error.
- A fresh shuffled 128-rollout Prime Eval stress test used the production
  16,384-token output limit. All 128 rollouts completed and scored. All 266
  per-turn reviews scored. All 1,184 saved evidence records matched the exact
  source characters and offsets. Five semantic rejections and two request
  faults recovered inside the bounded follow-up attempts. No rollout ended in
  a service failure or TaskError.
- The local environment suite has 49 passing tests. Ruff, Python compilation,
  and `git diff --check` pass.
- Muse15 started as a fresh production run. Trainer steps 1 through 3 completed
  with finite loss, entropy, and gradient norm. The sampled GPU memory maximum
  was 29.1 GiB.
- Rollout batches 1 through 3 each supplied 1,024 trainable rows. Their error
  rates were 0.4 percent, 0.2 percent, and 0.0 percent. The first two batches
  replaced a small number of failed follow-up generations. Batch 3 completed
  1,024 of 1,024 rows with no rollout error. Batch 4 started normally.
