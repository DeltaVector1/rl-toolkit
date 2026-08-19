# Fault table

Each entry gives the symptom, the cause, and the one correct repair. Every
entry is a fault that happened. Language: ASD-STE100 Simplified Technical
English.

# Find the fault first

```python
s = glm52_ops.status()
s.warnings
```

`status` prints a warning for each condition below that it can detect. Match
the warning to an entry here. Do not repair a fault that `status` does not
show.

# Rank 0 exits before the distributed trainer starts

Symptom: the seven nonzero trainer nodes have containers, n82 has no trainer
container, and its launcher ends with `KeyError: 'rollout_transport'`.

Cause: the rank-0 launcher reads the rollout transport from the trainer TOML
before it starts Docker. The production trainer TOML did not contain that
explicit table.

Repair: add `[rollout_transport]` with `type = "filesystem"` to the trainer
TOML and run the production configuration guard. If no trainer rank loaded a
model and no optimizer step started, launch n82 against the existing clean
rendezvous. Otherwise, start a fresh run name.

# Shared-expert checkpoint setup rejects a dense decoder layer

Symptom: every trainer rank exits during model setup with
`GlmMoeDsaDecoderLayer has no shared expert for reentrant checkpointing`.

Cause: the first GLM-5.2 decoder layers are dense. They do not expose
`mlp.shared_expert`.

Repair: checkpoint attention on every decoder layer. Checkpoint the shared
expert only when `mlp.shared_expert` exists. Run the CUDA call-count and
finite-gradient gate, deploy the file to all trainer nodes, and start a fresh
run name.

# The engine answers /v1/models but makes no tokens

Symptom: `gen=000` from `engines()`, or an engine at 200 on `/v1/models` with
`Running: 64` and `generation throughput: 0.0`.

Cause: the vLLM API server is alive and the EngineCore is dead. The two are
separate processes. The API server keeps answering metadata requests.

Repair: if `/root/engine_watchdog.log` shows an active restart and load check,
let that check finish. Otherwise run `glm52_ops.repair("engine:h200a")`. The
repair resets the watchdog miss count and gives the new load a 180-second
probe grace. The model load needs 5 to 10 minutes. Test again with `engines()`,
never with `/v1/models`.

This happened on 2026-08-06. The container showed "Up" for 2.6 hours with dead
engines. The container state is not a health signal either.

# The tunnels are listening but pass no data

Symptom: fewer than 18 of 48 ports answer. The probe hangs for the whole
timeout instead of refusing at once.

Cause: the SSH forward stays established while the link drops the packets. The
port accepts the connection and never answers.

Repair: `glm52_ops.repair("tunnels")`. This kills the n82 side sshd. The
supervisor on the H200 host rebuilds the tunnel in about 5 seconds.

Always rebuild, even when every tunnel fails at once. An earlier watchdog
refused to rebuild above 80 percent failure. That rule stopped the run for 5
hours while the engines were healthy.

# Ten or fewer tunnels answer and the engines are idle

Symptom: low tunnel probe count, idle engines, and many rollouts in flight.

Cause: too many rollouts in flight. The tunnels are the limit, not the batch
size.

Repair: confirm that engine throughput is also low. The current
`max_inflight_rollouts` is 128. Do not change it or `batch_size` without an
experiment decision. The older 256/1024 measurements used 24 tunnels and do
not describe the current 48-port pool.

# The trainer waits and no batch arrives

Symptom: the trainer log stops. `time/wait_for_batch` grows. Port 29620 holds
several connections.

Cause: somebody killed the batch tunnel while the orchestrator was alive. The
zmq session broke and left dead connections behind.

Repair: stop the orchestrator, then rebuild the tunnel, then start the
orchestrator. The trainer waited 27 minutes when this happened.

Never touch port 29620 while the orchestrator runs. `repair("tunnels")` does
not touch it.

# The off-policy gate stays closed after the trainer advances

Symptom: the orchestrator prints `Pausing dispatcher` and remains at
`0/1024` with zero in flight after a newer adapter `STABLE` marker exists.
The trainer continues to complete steps. The H200 engines can show the new
adapter even though the watcher policy version does not advance.

Cause: vLLM completed `/load_lora_adapter`, but one HTTP response did not
finish. The weight watcher remained inside the inference update. An engine
restart can also outlast the old ten-attempt retry cap.

Repair: use the patched `client.py`, confirm both engines are healthy with
`engines()`, then restart only the edge orchestrator at the clean zero-inflight
boundary. With `ckpt.resume_step = -1` and `skip_progress = true`, it resumes
from the latest immutable rollout batch and latest `STABLE` policy. Do not
restart the trainer. Confirm the exact adapter SHA-256 on both H200 caches.

This happened in x56p after rollout batch 6. The orchestrator resumed batch 7
from policy version 3. The trainer continued without a restart.

# The orchestrator waits for a policy version that cannot exist

Symptom: these two lines in the orchestrator log.

```
Resuming orchestrator at rollout batch N from policy vM
Pausing dispatcher to prevent orchestrator from racing from trainer
```

Cause: the trainer restarted and the orchestrator did not. The orchestrator
now expects a weight version that the trainer will never produce.

Repair: start a new run name. `glm52_ops.start("x25", mode="cold")`. Do not try
to repair the old run directory.

# A trainer worker stopped

Symptom: `trainer DOWN` in `status`, and the trainer log ends with either
"Storage size calculation overflowed" or an NCCL collective timeout.

Cause: a corrupt micro-batch stopped rank 0. The other 7 ranks then waited for
the NCCL timeout. One micro-batch arrived with an `input_ids` shape of
`[1, 9142729177965406720]`.

Repair: `train.py` now tests each micro-batch shape and discards a bad batch,
so that corrupt shape must not stop the run again. The trainer has no
resumable optimizer checkpoint. The run supervisor therefore stops every
trainer worker and the edge watchdog stops rollout production. Start a fresh
run name after the cause is fixed. Do not restart the old run.

This fault held the run dead for 6 hours because no supervisor existed.

# Pipeline checkpoint recompute uses another microbatch's ring boundaries

Symptom: a Muse PP8 and CP4 trainer reaches its first backward and stops with
`CheckpointError`. The saved and recomputed `torch.int32` boundary tensors
have different shapes. Other pipeline ranks then stop after the first rank.
The hosts stay reachable and their kernel logs contain no OOM or NVIDIA Xid.

Cause: ring attention stores packed-sequence boundaries in process-global
state. Pipeline 1F1B runs a later microbatch forward before it recomputes an
earlier microbatch. The later forward replaces the earlier state.

Repair: use the ring-attention checkpoint context in `trainer/model.py`. Each
checkpoint invocation captures its forward boundaries and restores them for
recompute. Run `muse_cp_checkpoint_interleave_smoke.py` on CP4. It must finish
backward with finite gradients. Deploy `checkpointing.py`, `model.py`, and the
`model.py` launcher mount to all trainer nodes. Start a fresh run name with a
fresh rollout batch.

# INT4 expert combine runs out of GPU memory on a large packed batch

Symptom: one or more trainer ranks fail in `_streaming_int4_forward` while
allocating a 384 MiB BF16 tensor. Earlier steps can pass. x65p failed at step
8 with a packed batch size of 126 after steps with packed sizes from 92 to 99.

Cause: bounded all-to-all receive rounds reduced communication scratch, but
the combine still allocated a separate full BF16 result while the consumed
expert output remained live.

Repair: use the in-place four-round INT4 combine. It reuses the expert output
buffer for the BF16 result. Run `scripts/ops/test_int4_combine_streaming.py`
on eight GPUs. The constrained test must pass with 160 MiB free before the
combine. Start a fresh run because the failed trainer has no optimizer
checkpoint.

# FP8 combine backward cannot allocate its BF16 gradient

Symptom: `_INT4Dispatch.backward` fails at `grad_input.to(grad_output.dtype)`
with a 384 MiB allocation. x68p reached this at step 8 on n77 rank 1 with
367.5 MiB physically free and 911.12 MiB reserved but unused.

Cause: forward combine was bounded, but backward still held complete FP8 send
and receive tensors before allocating a second full BF16 gradient. The
trainer launcher also retained `max_split_size_mb=256`, which stranded the
large reserved blocks.

Repair: use four-round FP8 backward receive and dequantize into the consumed
incoming-gradient buffer when it has enough rows. Remove
`max_split_size_mb=256` from the trainer allocator configuration. The
eight-GPU smoke test must report exact forward and backward parity and pass
the constrained 32K by 6,144 case before a fresh run starts.

# FP8 expert row gather fails with unused allocator cache

Symptom: `_InplacePermutedExperts.forward` fails at `x.index_select` while it
allocates a 4,096 by 6,144 BF16 chunk. x74p failed at trainer step 6 on n77
local rank 1. The packed batch size was 110. The 48 MiB allocation had 7.5 MiB
physically free and 179.6 MiB reserved but unused.

Cause: the live set was near the 48 GiB limit and the unused allocator cache
did not have one contiguous block for the bounded gather. This was allocator
fragmentation, not the INT4 dispatch or combine path.

Repair: keep the 4,096-row chunk and make one `torch.cuda.empty_cache()` retry
for the row gather. Apply the same retry to the backward gather. A second OOM
must remain fatal. Run `scripts/ops/test_fp8_ep_inplace_permutation.py`; exact
forward and input-gradient parity must pass before a fresh run starts.

Set the trainer container core limit to zero. Otherwise, the 64 worker aborts
can fill `/var/lib/apport/coredump` and the full root filesystem prevents NCCL
from writing its diagnostic trace.

# INT4 dispatch cannot allocate an imbalanced BF16 receive tensor

Symptom: `_INT4Dispatch.forward` fails in the dispatch input hook while it
allocates `dequantized`. x69p failed at trainer step 8 on n77 local rank 5.
The rank needed 498 MiB for 42,445 rows and had 413.5 MiB physically free.

Cause: four-round transport applied only to expert-output combine. Dispatch
still retained the complete packed send and receive tensors before it
allocated the full BF16 receive tensor. EP routing imbalance made this tensor
larger than the 32K-row source on the failed rank.

Repair: apply the four-round path to dispatch and combine. When the result is
larger than the consumed source, allocate the BF16 result first and quantize
only one send round at a time. Apply the same order to the inverse FP8
all-to-all when its result is larger. The eight-GPU smoke test must pass exact
forward and backward parity and the unequal 32K by 6,144 constrained case
with 560 MiB initially free. Start a fresh run name.

Dispatch must not reuse its input even when that storage is large enough. The
input is an FSDP post-backward view. x70p returned and modified that view at
trainer step 1, and PyTorch stopped before computation because that action
would replace FSDP's custom backward. Only combine owns a reusable input.

# INT4 combine cannot allocate a larger inverse result

Symptom: `_streaming_int4_forward` fails in the combine output hook while it
allocates a 384 MiB BF16 tensor. The failed rank received fewer routed rows
during dispatch than it must return during combine. x73p failed at trainer
step 8 on n76 local rank 7 with 335.5 MiB physically free.

Cause: dispatch allocated only its received row count. The in-place native-FP8
expert path preserved that smaller storage, so combine could not reuse it for
the larger inverse result. The combine path also retained a complete INT4 send
copy before it expanded the BF16 result.

Repair: reserve `max(input_rows, output_rows)` BF16 storage in dispatch while
memory is available. Keep the smaller logical dispatch shape. During combine,
retain the compact INT4 receive payload, quantize one send round at a time in
the unused reserved tail, and dequantize into the reserved BF16 storage only
after all sends finish. The eight-GPU test must pass unequal dispatch, inverse
combine, and backward with 560 MiB initially free before a fresh run starts.

# FP8 backward receive fails with unused allocator cache

Symptom: `_streaming_fp8_backward` fails while it allocates one bounded FP8
receive round. x75p failed at trainer step 8 on n77 local rank 3. The 26 MiB
allocation had 19.5 MiB physically free and 284.7 MiB reserved but unused.

Cause: the live set was near the 48 GiB limit. The allocator cache did not
provide the small contiguous receive block. This was allocator fragmentation,
not an unbounded backward tensor or a rollout fault.

Repair: release unused cached allocator blocks and retry the identical receive
allocation once. Do not change its shape, precision, or round count. A true
live-memory OOM must remain fatal on the second attempt. Run the eight-GPU
unequal-routing parity and constrained-memory test before a fresh run starts.

The same repair applies to the full BF16 inverse-gradient result that is
allocated before the receive rounds. x76p failed at trainer step 5 on n77
local rank 2. The 488 MiB result had 389.5 MiB physically free and 613.8 MiB
reserved but unused. Do not reduce or quantize this result. Release the unused
cache once and retry its identical allocation.

x80p exposed a different 384 MiB inverse-gradient allocation at trainer step
4 on n77 local rank 7. The incoming gradient had fewer logical rows than the
32,768-row result, but its underlying storage already had 32,768-row capacity
reserved by dispatch. The old reuse check tested logical rows and tried to
allocate a second BF16 result with only 369.5 MiB physically free. Test
contiguous storage capacity and zero storage offset instead, quantize the
logical send rows before resizing, and dequantize the received rounds into the
reserved storage. The eight-GPU unequal-routing gate must show exact forward
and backward parity and pointer reuse on every rank.

x81p reached the same branch at step 4 with 32,802 returned rows. Its
contiguous incoming-gradient view had a nonzero storage offset, so the first
capacity repair rejected it and requested another 386 MiB BF16 tensor. Count
usable storage elements after `storage_offset()` for any contiguous view. Do
not require offset zero. The distributed gate must also pass an offset-view
backward case and preserve its data pointer.

# FP8 forward allocations fail with unused allocator cache

Symptom: `_streaming_int4_forward` fails while it allocates a 430 to 458 MiB
BF16 dispatch result, or `per_token_group_quant_fp8` fails while it allocates
a 24 to 48 MiB FP8 expert-input result. x77p reached both failures at trainer
step 8 after seven finite steps.

Cause: the live set was near the 48 GiB limit. PyTorch held more unused
reserved memory than the requested tensor size, but the allocator did not
have a suitable contiguous block. The later NCCL failures were consequences.

Repair: release unused cached allocator blocks and retry the identical
allocation or quantization once. Do not change the shape, precision, or
transport rounds. Run the eight-GPU unequal-routing parity and constrained
memory test before a fresh run starts. A second OOM must remain fatal.

# FP8 expert forward exceeds the live-memory headroom on a larger batch

Symptom: the cache retry also fails in `per_token_group_quant_fp8`, or
`_InplacePermutedExperts.forward` fails at `index_copy_`. x78p completed eight
finite steps, then step 9 used a packed batch of 112. The failed ranks needed
24 to 48 MiB with only 3.5 to 31.5 MiB physically free.

Cause: this is a live-set peak, not only allocator fragmentation. The
4,096-row expert workspace made a 24 MiB FP8 input for each projection.
Boolean indexing in the in-place writeback also made a hidden 48 MiB BF16
source-row copy.

Repair: use 2,048-row native-FP8 expert forward and backward chunks. Keep the
writeback on PyTorch `index_copy_`, where the boolean source gather is bounded
to 24 MiB, and retry it once after releasing unused allocator cache. Do not
use a custom Triton scatter here. x79p completed three finite steps, then a
previous custom scatter first loaded a CUDA module at the full step-4 live set
and failed with `Triton Error [CUDA]: out of memory` on n82 local rank 6. The
other ranks reported NCCL remote-process exits only after that first failure.
Run the SM89 exact forward and input-gradient gate before a fresh run starts.
The constrained gate must pass with approximately 1,024 MiB free before
forward and backward.

# The nodes run out of host memory

Symptom: fewer than 60 GB free on a node. The OOM killer stops the trainer.
The kernel trace shows `shmem_fault` and `shmem_alloc_folio`.

Cause: `ac_offloading` is not `"None"`. The default is
`ActivationOffloadingConfig()`, which puts activations into pinned host memory.
The host memory then follows the number of micro-batches in flight.

Repair: set `ac_offloading = "None"` in the trainer config. Host memory then
drops from 486 GB of 503 GB to about 3 GB.

This memory is real. An earlier note called it an accounting artifact. That
note was wrong and is retracted.

# The reward will not rise and the entropy is very high

Symptom: entropy near 9.2 nats against a vocabulary limit of 11.95. Gradient
norm 200000 to 330000 against a clip limit of 1.0. Mismatch KL 7 to 9 nats.

Cause: the trainer ran FP8. The engines use W8A16, which quantizes the weights
only. The trainer FP8 path uses W8A8, which also quantizes the activations, so
the trainer is less exact than the engines. The trust region then masks almost
every positive advantage.

Repair: set `dequantize_checkpoint = true`. The trainer then uses bf16 weights,
the mismatch KL drops to 0.2, and the entropy drops to 0.45.

A better fix is to make the trainer use W8A16. Start at `_dequant_weight` in
`fp8_ada.py`.

# The reward looks flat and nothing is broken

Symptom: `reward()` returns the verdict "inside the noise".

Cause: this is not a fault. The 10th percentile of the reward stays near zero
because the reward multiplies five factors and one factor near zero makes the
product near zero.

Action: report the verdict. Read the 90th percentile for the distribution
signal. Collect more steps. Do not report a percentage from a half window.

# A watchdog exists but does nothing

Symptom: `alive()` prints a process count above zero and a stale log line.

Cause: the process is stuck, or an old test used `pgrep -fc`. That command
counts the grep process, so its result is never zero and the test always
passes.

Repair: `glm52_ops.repair("supervisors")`. Always read the log timestamp as
well as the process count. This test hid three dead watchdogs for 3 hours.

# A source change has no effect

Symptom: the behaviour does not change after `deploy` reports success and the
trainer restarts.

Cause: `cluster.sh` does not mount that file, so the container uses the copy
inside the Docker image.

Repair: call `glm52_ops.mounted_files()`. Add a `-v` line to
`scripts/china/cluster.sh` if the path is missing, then deploy `cluster.sh`
itself.

Never copy a file out of the image into the repository to "restore" it. The
repository copies hold local changes that the image does not have. That action
destroyed `_FiniteGrad` and `PRESCALE_QK`.
