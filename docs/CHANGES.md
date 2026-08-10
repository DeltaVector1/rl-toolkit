# GLM-5.2 DECENSOR RL. CHANGE LIST

This document lists each change. The patch files are in `patches/`.

Language: ASD-STE100 Simplified Technical English.

# HOW TO APPLY THE CHANGES

The combined patch has all changes to `source/` and `scripts/`:

```
git apply patches/glm52-fixes-all.patch
```

The source patch has only the changes to `source/`:

```
git apply patches/glm52-source-only.patch
```

The directory `patches/series/` has one file for each commit. Use these files
if you want to apply the changes one at a time:

```
git am patches/series/*.patch
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
