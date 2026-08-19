# GLM-5.2 routed-expert LoRA backward state

## Purpose

This document researches how to train both LoRA matrices for every routed
expert without replacing the validated native-FP8 expert path.

It answers three open questions:

1. How can backward recover the original expert input without keeping another
   large BF16 tensor on the GPU?
2. Which router gradients must remain?
3. Should H200 inference start with expert parallelism enabled or disabled?

This is a research reference. It is not permission to start a run.

## Short answer

The first implementation should use standard LoRA. Both A and B must train.

The expert input already crosses the node-local expert exchange as INT4 data
plus one scale for each row. The forward pass then converts that exact payload
to BF16 before the expert computation.

The best current design is:

1. Copy the exact received INT4 payload and its scales to pinned host memory
   during the existing streamed expert exchange.
2. Give each forward call its own payload object.
3. Save that object in the matching expert autograd context.
4. During backward, gather and copy only one 2,048-row compressed chunk to the
   GPU.
5. Convert that chunk to BF16 in the consumed output-gradient buffer.
6. Use the reconstructed input for the W1 and W3 LoRA gradients.
7. Recompute the small LoRA A results instead of retaining them for the whole
   forward and backward lifetime.

This preserves the exact input that the expert forward used. It does not repeat
the expert exchange. It adds no second full BF16 expert-input tensor on the GPU.

The router weights stay frozen. The selected route scores must remain
differentiable with respect to the layer input.

The H200 baseline must keep expert parallelism enabled. vLLM 0.26 requires the
`allgather_reducescatter` communication backend for fused MoE LoRA with expert
parallelism. Expert parallelism disabled is a comparison, not the baseline.

## Verified project facts

The following facts come from the current source and measured runs:

- GLM-5.2 has 78 decoder layers.
- It has 75 routed-expert layers.
- Each routed-expert layer has 256 experts.
- Each token uses eight experts.
- The model width is 6,144.
- The routed expert width is 2,048.
- Expert parallelism is node-local across eight RTX 4090 GPUs.
- The expert input crosses that group with INT4 values and one FP16 scale for
  each row.
- The current expert backward uses 2,048-row chunks.
- The current in-place expert forward overwrites its BF16 input with its output.
- The current backward retains the quantized W1 and W3 results in pinned host
  memory.
- The current route-score operation retains the unscaled BF16 expert output in
  pinned host memory.
- One measured rank received 42,445 routed rows.
- A BF16 tensor with shape `[42445, 6144]` is 497.402 MiB.
- A 24 MiB allocation has already failed when only 1.5 MiB was physically free.

Relevant project source:

- [`_InplacePermutedExperts`](../../source/src/prime_rl/trainer/models/layers/fp8_ada.py)
- [`_ScaleRoutedOutput`](../../source/src/prime_rl/trainer/models/layers/fp8_ada.py)
- [INT4 expert exchange](../../source/src/prime_rl/trainer/distributed/expert_parallel.py)
- [MoE routing](../../source/src/prime_rl/trainer/models/layers/moe.py)
- [Historical memory measurements](../PERFORMANCE.md)

## Why ordinary LoRA needs the original input

For one projection, standard LoRA is:

```text
y = base(x) + scale * (x A^T) B^T
```

Define:

```text
z = x A^T
```

Backward requires:

```text
grad_B = scale * grad_y^T z
grad_z = scale * grad_y B
grad_A = grad_z^T x
grad_x_lora = grad_z A
```

`grad_A` needs the original `x`.

The current native expert operation cannot refer to that BF16 `x` during
backward because forward overwrites the storage. The small `z` tensor is not a
replacement for `x` when A is trainable.

## Production-shape memory

Use two row counts for memory work:

```text
received_rows = rows owned by this expert-parallel rank after dispatch
source_rows = rows that this rank sent before dispatch and receives after combine
```

Routing imbalance makes these counts different. Expert padding can also add
rows to the W1 and W3 saved values.

The table uses 42,445 rows only as a common shape example. That count was
measured for received rows. The table does not claim that the same rank had
42,445 source rows at the same time.

| Saved value | Format | Size for one expert call |
|---|---:|---:|
| Original expert input | BF16 | 497.402 MiB |
| Original received expert payload | INT4 plus FP16 row scales | 124.432 MiB |
| Re-encoded expert input | FP8 plus FP32 row scales | 248.863 MiB |
| Existing W1 and W3 saved values | FP8 plus scales | 166.125 MiB |
| Existing unscaled expert output | BF16 | 497.402 MiB |
| Three rank-4 A results | BF16 | 0.971 MiB |

If received rows and source rows were both 42,445 and there were no padding
rows, the present expert and route-score path would retain approximately
663.527 MiB of host data for the call. Adding the exact received INT4 input
would produce approximately 787.959 MiB. This example is an increase of
approximately 18.8 percent.

The implementation must calculate the real value as:

```text
input payload = received_rows * (ceil(model_width / 2) + scale_bytes)
W1 and W3 state = padded_received_rows * expert_width * 2 FP8 values + scales
route-score state = source_rows * model_width * BF16 bytes
```

These shape examples describe one call on one rank. The host-memory high point
also depends on the pipeline stage, the number of live microbatches, and the
number of expert layers on that stage. The implementation must measure that
high point. It must not multiply this table by an assumed number of live calls
and present the result as a measurement.

The earlier `6 * rows * rank` calculation described only the small A results.
It did not include the input required for `grad_A`.

## Options

### Option 1: Retain the received INT4 payload on the host

Status: recommended for the first standard-LoRA implementation.

The streamed expert exchange already receives the compressed rows. It can copy
each received round into a preallocated pinned host tensor before it releases
the GPU receive buffer.

Benefits:

- It preserves the exact INT4 codes and scales used to create the BF16 forward
  input.
- It uses one quarter of the BF16 input storage.
- It does not repeat the node-local exchange.
- It can stage small backward chunks.
- It permits both A and B to train.

Costs:

- It adds approximately 124.432 MiB of host data for the measured large call.
- It adds device-to-host traffic in forward and host-to-device traffic in
  backward.
- It needs a safe ownership design for several live microbatches.
- It needs a bounded CPU gather because expert-sorted rows do not remain in the
  received source order.

This option has the best current balance of exactness, GPU memory, and
computation.

### Option 2: Retain the expert input as BF16 on the host

Status: reference implementation only.

The expert forward already gathers a BF16 chunk before W1 and W3. It could copy
that chunk to pinned host memory before overwriting the input.

Benefits:

- It is simple.
- It preserves exact BF16 values.
- It does not need access to the compressed exchange payload.

Costs:

- It adds approximately 497.402 MiB for the measured large call.
- It adds four times as much host storage and transfer as the INT4 payload.

Use this only as a correctness reference or if the compressed-payload ownership
change cannot be made safe.

### Option 3: Repeat the expert exchange during backward

Status: fallback benchmark, not the first choice.

Activation checkpointing can recreate the input by repeating dispatch and the
expert computation. PyTorch documents this normal memory and computation
tradeoff: checkpointed operations retain their inputs and rerun the selected
region during backward.

The project already has an expert activation-checkpoint path. It repeats the
node-local exchange when its cache is disabled.

Benefits:

- It avoids long-lived input storage.
- It can reproduce the same transport values.

Costs:

- It repeats communication and computation.
- Collective order must remain identical on all ranks.
- It increases the chance of a distributed failure.
- Prior measurements show that expert communication is a major cost.

This option should remain available as a measured comparison. It should not be
selected from theory alone.

### Option 4: Use the existing GPU expert-checkpoint cache

Status: reject.

`PRIME_EP_AC_CACHE=1` retains the received INT4 payload on the GPU. The project
measured approximately 3.8 GiB for each GPU. It also has one dispatch slot and
one combine slot for each layer. Several pipeline microbatches can overwrite a
slot before its matching backward work.

This cache has already caused memory failures. It is not a safe base for routed
LoRA.

### Option 5: Re-encode the BF16 input as FP8 or INT4

Status: experimental only.

The expert code could quantize the dequantized BF16 input again and retain that
new result. This avoids changes to the exchange code.

The second quantization is not automatically equal to the received payload.
It can change `grad_A`. An FP8 copy also uses about twice the memory of the
original INT4 payload.

Do not use this in the first exact implementation. Test it later only if output
and all gradients pass the accepted numerical comparison.

### Option 6: LoRA with frozen A

Status: valid method, but not equivalent to standard LoRA.

[LoRA-FA](https://arxiv.org/abs/2308.03303) freezes A and trains B. It needs the
small `x A^T` result for `grad_B`, but it does not need the full input for
`grad_A`. The paper reports lower activation memory and close results on its
tested models and tasks.

This changes the trainable parameter space. It does not meet a requirement to
train both LoRA matrices. Use it only as an explicit capacity or memory
experiment after the standard-LoRA result exists.

## Recommended input-state design

### Forward

The streamed INT4 dispatch should do this only when routed-expert LoRA is
enabled:

1. Allocate one pinned host payload and one pinned host scale tensor for the
   current forward call.
2. Receive the normal bounded GPU round.
3. Start a nonblocking copy of that round into its source-order host slice.
4. Record an event after the final copy.
5. Complete the existing BF16 conversion and return the normal expert input.
6. Pass ownership of the host payload, scales, shape, and event to the expert
   operation that immediately consumes that dispatch result.
7. Store that object in the autograd context for this call.

The payload must belong to the autograd call. Do not put it in one mutable slot
on the layer. Do not use a process-wide saved-tensor hook.

The expert context already owns the expert permutation and offsets. It should
also retain a small host copy of the permutation for backward staging. Padding
rows must remain zero and must not read a payload row.

### Backward

The current expert backward uses 2,048-row chunks. Keep that bound for the
first implementation.

For each expert-sorted chunk:

1. Wait for the forward host-copy event before the first use.
2. Use the host permutation to gather the compressed source rows into a bounded
   pinned staging buffer.
3. Copy the compressed chunk and its scales to the GPU.
4. Finish all W2 work that still needs the incoming output gradient.
5. Reuse the consumed `grad_chunk` BF16 storage as the reconstructed expert
   input.
6. Convert the staged INT4 rows directly into that storage.
7. Recompute the rank-sized W1 and W3 A results from the reconstructed input.
8. Calculate W1 and W3 `grad_A`, `grad_B`, and LoRA input gradients.
9. Add LoRA input gradients into the existing base input-gradient result.
10. Release the compressed staging tensors before the next chunk.

At 2,048 rows, a new BF16 input allocation would be 24 MiB. Reusing
`grad_chunk` removes that allocation. The compressed GPU staging data is about
6 MiB at this chunk size.

Use two bounded staging buffers only if a measured trace shows that overlapping
the next host-to-device copy improves time without reducing the required GPU
reserve.

### Low-rank A results

At rank 4, all three A results use less than 1 MiB for the measured large call.
However, the pipeline can keep many calls live.

The first implementation should recompute the A results in backward because:

- the exact input is already being reconstructed;
- rank-4 computation is small;
- recomputation removes another long-lived GPU value;
- dropout is zero;
- adapter parameters do not change between forward and backward.

Benchmark saving the A results against recomputation. Reconsider saving at rank
8 or rank 16 if the measured recomputation time becomes important.

## Parameter-gradient computation

Rows are already sorted by local expert after the expert permutation. Each
expert therefore owns one contiguous range in the logical sorted order.

The kernel should:

- read reconstructed input rows through that ordering;
- accumulate each A and B gradient tile in FP32;
- write BF16 parameter gradients after the tile reduction;
- add LoRA input gradients into the existing BF16 input-gradient destination;
- avoid a second full-width LoRA output or input-gradient tensor;
- use bounded split reduction if one expert has too many rows for one program.

The project should use the row-indirection and buffer-reuse ideas from
[ScatterMoE](https://arxiv.org/abs/2403.08245). It should not replace the
native FP8 expert computation with ScatterMoE. ScatterMoE saves the input and
expanded output for ordinary expert weight and route-score gradients. Its
published implementation does not solve this project's in-place FP8 input
lifetime by itself.

Latest Prime-RL already has a grouped-expert LoRA implementation at revision
[`1c0720d`](https://github.com/PrimeIntellect-ai/prime-rl/blob/1c0720d9753cd9c314b882c2020c19c47b82f78c/src/prime_rl/trainer/models/layers/lora/multi_moe.py).
It is a useful format and gradient reference. It is not the production kernel
for this model because it converts the base expert path to BF16. Its grouped
operation also disables itself for rank 4 and uses a per-expert fallback.

## Router-score gradients

Freezing the router weight does not remove every router-related gradient.

For one routed row:

```text
scaled_output = route_score * expert_output
```

Backward is:

```text
grad_expert_output = route_score * grad_scaled_output
grad_route_score = dot(grad_scaled_output, expert_output)
```

The selected expert identities are discrete. The selected scores are still
differentiable. The frozen router weight receives no parameter gradient, but
`grad_route_score` still contributes to the layer-input gradient through the
fixed router weight.

The current `_ScaleRoutedOutput.backward` returns both gradients. The current
router obtains its scores from the layer input without detaching them.

The first routed-LoRA implementation must preserve this behavior. Detaching
the selected scores would change the trainer. It would also remove the need to
retain the unscaled expert output, but memory savings alone are not a valid
reason to change the gradient.

The present exact design should keep the BF16 unscaled output in pinned host
memory. Compressing it or recomputing it would change or repeat the inverse
expert exchange unless a larger distributed refactor moves route-score work to
the expert-owning rank. That work is not part of the first implementation.

## H200 expert parallelism

The current project configuration uses one TP8 GLM replica on each H200 host
and has expert parallelism enabled.

[vLLM 0.26 expert-parallel documentation](https://docs.vllm.ai/en/v0.26.0/serving/expert_parallel_deployment/)
states that expert parallelism can improve MoE locality and efficiency. It
also labels the feature experimental.

[vLLM 0.26 fused MoE LoRA source](https://docs.vllm.ai/en/v0.26.0/api/vllm/lora/layers/fused_moe/)
has these restrictions when expert parallelism and fused MoE LoRA are both
enabled:

- The communication backend must be `allgather_reducescatter`.
- MoE sequence parallelism must be disabled.
- Fully sharded LoRA must be disabled.

The baseline qualification should therefore use:

```text
TP8
expert parallelism enabled
all-to-all backend = allgather_reducescatter
MoE sequence parallelism disabled
fully sharded LoRA disabled
```

Then compare the same adapter and requests with expert parallelism disabled.
Report these values separately:

- adapter load success and loaded tensor count;
- trainer and inference output comparison;
- prompt tokens per second;
- output tokens per second;
- first-token time;
- p50, p95, and p99 request completion time;
- GPU memory;
- expert communication time;
- complete real-rollout collection time.

Do not select either mode from a synthetic short request alone. Do not run this
comparison against an active production engine.

## Required measurements before implementation selection

### Host state

Measure these values for each trainer process and each node:

- current pinned host bytes for saved W1 and W3 values;
- current pinned host bytes for unscaled expert outputs;
- additional pinned bytes for retained expert inputs;
- maximum number of live expert autograd calls;
- system RAM high point;
- pinned allocator high point;
- device-to-host and host-to-device copy time;
- time waiting for the input payload in backward.

The code must release each payload after its matching backward call. A missing,
duplicate, or overwritten payload must stop the test with a clear error.

### GPU state

Measure around W1, W3, W2, and LoRA work:

- physically free GPU memory;
- allocated and reserved memory;
- compressed staging size;
- BF16 chunk size;
- base and LoRA input-gradient destinations;
- parameter-gradient reduction scratch;
- memory before and after kernel compilation.

Do not accept an implementation that relies on allocator cache release during
every ordinary chunk.

### Time

Report these times separately:

- native FP8 base expert forward;
- LoRA forward;
- native FP8 base expert backward;
- LoRA backward;
- input host copies;
- input host gathering;
- expert communication;
- complete trainer forward and backward.

## Required correctness tests

### One RTX 4090

- Compare W1, W3, and W2 outputs with a BF16 standard-LoRA reference.
- Compare input, A, and B gradients.
- Test rank 4, 8, and 16.
- Test empty experts and strongly unequal expert row counts.
- Verify that LoRA disabled leaves the native FP8 result unchanged.
- Verify that reconstructed input is bitwise equal to the BF16 input produced
  from the retained INT4 payload.

### One eight-GPU trainer node

- Use real expert dispatch and combine.
- Use unequal top-eight route scores.
- Keep several microbatches live before backward.
- Verify that every forward call receives its own payload.
- Verify exact route count conservation.
- Compare route-score gradients with the current implementation.
- Verify no second complete BF16 expert-input tensor appears.
- Verify no expert communication repeats during routed-LoRA backward.

### All trainer nodes

- Require all logical LoRA targets before the first forward.
- Require finite and nonzero gradients for attention, dense MLP, shared expert,
  and routed expert target classes.
- Require router and sparse-indexer parameters to remain frozen.
- Require the selected route-score input gradient to remain present.
- Run the trainer-only changing-policy qualification required by
  `TRAINING_EXPECTATIONS.md`.
- Apply its memory, precision, performance, and stability requirements.

### H200 serving

- Load and unload the complete adapter.
- Verify all routed experts and all three projections.
- Compare EP enabled and disabled under the same vLLM revision.
- Verify the exact allowed communication backend for EP plus LoRA.
- Compare trainer and inference outputs before a rollout test.
- Use real prompt and output-length distributions for the final measurement.

## Recommended order for the later build plan

1. Add instrumentation for live expert calls, pinned bytes, copy time, and GPU
   reserve.
2. Build a one-GPU full-LoRA reference with both A and B trainable.
3. Add per-call capture of the exact streamed INT4 receive payload.
4. Add bounded host gathering and input reconstruction.
5. Reuse `grad_chunk` as the reconstructed BF16 input.
6. Add W1 and W3 LoRA backward.
7. Add W2 LoRA backward.
8. Preserve and test selected route-score gradients.
9. Add the full target registry and `kv_b_proj` special path.
10. Pass one-GPU tests.
11. Pass EP8 tests with several live microbatches.
12. Pass the full 64-GPU trainer qualification.
13. Qualify vLLM 0.26 with EP enabled and the required backend.
14. Compare EP disabled.
15. Select rank only after correctness, memory, transfer, and learning tests.

## Decision for the future final plan

Use exact per-call INT4 input retention as the first standard-LoRA design.

Do not use the current GPU expert-checkpoint cache.

Do not freeze A unless the owner selects LoRA-FA as a separate experiment.

Do not detach selected route scores.

Do not disable H200 expert parallelism before measuring it against the current
known configuration.

## Primary sources

1. PyTorch activation checkpointing:
   [Current and New Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)
2. PyTorch saved tensor control and pinned host storage:
   [Hooks for autograd saved tensors](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
3. LoRA-FA:
   [LoRA-FA: Memory-efficient Low-rank Adaptation](https://arxiv.org/abs/2308.03303)
4. ScatterMoE:
   [paper](https://arxiv.org/abs/2403.08245) and
   [source](https://github.com/shawntan/scattermoe)
5. vLLM 0.26:
   [expert-parallel deployment](https://docs.vllm.ai/en/v0.26.0/serving/expert_parallel_deployment/) and
   [fused MoE LoRA](https://docs.vllm.ai/en/v0.26.0/api/vllm/lora/layers/fused_moe/)
6. Prime-RL grouped-expert LoRA reference:
   [revision `1c0720d`](https://github.com/PrimeIntellect-ai/prime-rl/blob/1c0720d9753cd9c314b882c2020c19c47b82f78c/src/prime_rl/trainer/models/layers/lora/multi_moe.py)
