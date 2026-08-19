# GLM-5.2 performance staging plan

Last updated: 2026-08-19

This document is a research and staging plan. It does not change the active
Muse run. It does not approve a new GLM production run.

## Short conclusion

Do not replace the current GLM expert path with ScatterMoE as a package.
ScatterMoE does not support the checkpoint FP8 format, the current expert
parallel layout, or the RTX 4090 production shape.

Use two ScatterMoE design ideas in a new routed-expert LoRA path:

- Compute the expert row mapping inside the LoRA matrix operations.
- Reuse grouped buffers during backward.

Keep the frozen base weights in native block FP8. Keep the LoRA weights and
gradients in BF16. Use FP32 accumulation inside the LoRA matrix operations.
The project must not create FP8 trainable LoRA weights.

The first system test should use a small rank on every required projection.
Uniform rank 4 is the simplest test because current vLLM expects one adapter
rank. This test checks full model coverage. It does not select the final rank.

## Measured starting point

| Item | Measured result |
|---|---:|
| x72p trainer forward and backward | 10 minutes 54 seconds to 13 minutes 11 seconds |
| x84p trainer forward and backward, step 6 | 11 minutes 57 seconds |
| x84p peak GPU allocation | 44.22 GiB on a 48 GiB GPU |
| x72p failure allocation | 24 MiB with 1.5 MiB physically free |
| x84p rollout generation, step 7 | 21 minutes 58 seconds |
| x84p adapter | approximately 397 MB |
| Fastest measured 279 MB adapter transfer | 76.8 seconds to one H200 host |

x72p trained four attention projections. x84p trained the same attention
projections and the always-on shared expert. Neither run trained the routed
experts. x84p also did not train the dense MLP layers.

The current base linear path is not FP32 backward on RTX 4090. The local
`_linear_input_grad()` function quantizes the output gradient and runs the
frozen base matrix operation as W8A8. The FP32 function is a fallback and a
reference implementation.

## Required model coverage

The first routed-expert test must keep the router frozen. It must adapt these
weights:

- `q_a_proj`
- `q_b_proj`
- `kv_a_proj_with_mqa`
- `kv_b_proj`
- `o_proj`
- Gate, up, and down projections in the three dense MLP layers
- Gate, up, and down projections in every shared expert
- Gate, up, and down projections in all 256 routed experts in all 75 MoE layers

The sparse attention indexer is under `torch.no_grad()`. The first experiment
must keep its query, key, and score projections frozen. The MoE router must
also stay frozen.

GLM-5.2 has approximately 58,224 base projection matrices in the list above.
The exact count must come from the loaded model. Startup must stop if the
actual count or target names differ. It must not silently ignore a target.

`kv_b_proj` needs special work. The trainer reads `kv_b_proj.weight` directly
during MLA absorption. Replacing the module with an ordinary LoRA wrapper does
not change this calculation. The MLA code must apply the LoRA update to the
absorbed key and value calculation. The H200 target list must also include
`kv_b_proj`.

## ScatterMoE review

The reviewed ScatterMoE revision is
`47b5e1502e5a10e82c8e5945d761b877849871e7`.

The [ScatterMoE paper](https://arxiv.org/abs/2403.08245) reports 38.1 percent
higher training throughput than sparse MegaBlocks in one integrated test. It
also reports 66.2 percent of the MegaBlocks training memory in one unit test.
These results do not predict the result on this cluster.

The integrated test used a 1.5B model, eight A100 GPUs, eight experts, and
top-2 routing. GLM-5.2 uses 256 experts, top-8 routing, block-FP8 checkpoint
weights, and RTX 4090 trainers.

The [ScatterMoE repository](https://github.com/shawntan/scattermoe) has these
limits for this project:

- Its tests cover FP32, eight experts, and small shapes.
- Its matrix operations do not read block-FP8 scales.
- It does not implement the current expert-parallel communication.
- It does not implement the current source-major row order.
- It does not implement the current W1, W3, and W2 streamed live-memory plan.
- It does not include multi-node training support.

Use ScatterMoE as a kernel reference. Do not make it the base expert module.

## Routed-expert LoRA kernel plan

### Parameter layout

Store each local adapter as contiguous expert-major tensors:

```text
A: [local experts, rank, input width]
B: [local experts, output width, rank]
```

Do not build a `ParameterList` for each expert. Do not stack expert parameters
inside each forward call.

### Forward

Use the current source-major row map once.

For W1 and W3, compute the frozen W8A8 base result and the BF16 LoRA result.
Add each LoRA result before the SiLU and multiply operations.

For W2, use the changed hidden value as the input to both the base operation
and the W2 LoRA operation. Add the two W2 results before the expert output is
returned.

Process rows in bounded chunks. Reuse the current gathered input and output
buffers. Do not make a second full routed-token copy for LoRA.

### Backward

Compute gradients for every A and B matrix. Accumulate matrix products in
FP32. Store parameter gradients in BF16.

Combine the frozen-base input gradient and the LoRA input gradient in one
destination buffer. Reuse the grouped row buffers when their earlier values
are no longer required.

Test three saved-value plans:

1. Save the low-rank A result on the GPU.
2. Recompute the A result during backward.
3. Move only the low-rank A result to bounded pinned host memory.

Do not move full expert inputs or full expert outputs to host memory. Do not
enable process-wide activation offload.

### Kernel selection

Benchmark these implementations with real routing histograms:

- The current Triton LoRA matrix operation
- A ScatterMoE-style row-mapped LoRA operation
- A grouped matrix operation when the RTX 4090 build supports the shape
- A simple per-expert reference implementation

Test ranks 1, 2, 4, 8, and 16. Test W1, W3, and W2 separately. Include forward,
input gradient, A gradient, and B gradient in the timing.

Compile and warm every production kernel shape before the full live-memory
test. A first-use compilation must not occur while the GPU is near its peak.

## Why upstream Prime-RL is not a direct fix

The reviewed Prime-RL main revision is
`1c0720d9753cd9c314b882c2020c19c47b82f78c`.

Latest Prime-RL has grouped expert LoRA. Its current expert forward converts
the base expert weights to BF16. Its LoRA parameter dtype also comes from the
base weight dtype. That behavior is not valid for this native-FP8 checkpoint.

Latest Prime-RL also sets trainer pipeline parallelism to 1. The project PP8
implementation is local code. A full update would remove the topology that
makes expert communication node-local.

Latest Prime-RL FP8 training uses DeepGEMM and needs SM90 or newer GPUs. The
trainer uses SM89 RTX 4090 GPUs. DeepEP also needs SM90 features and is not an
RTX 4090 option. Current expert communication is already node-local and was
measured at approximately 12 milliseconds for each layer.

Port selected changes only. Useful candidates are the vLLM 0.26 dependency,
the state-only optimizer offload, and adapter export fixes. Do not replace the
local PP8 or native Ada FP8 path.

## Pipeline parallel work

PP8 already removed the largest old problem. It changed expert communication
from a 64-GPU network operation to an eight-GPU node-local operation.

The current equal plan uses boundaries
`[0, 9, 18, 28, 38, 48, 58, 68, 78]`. Four boundaries need the sparse index
cache. The current global setting sends the cache at all seven boundaries.

Do this work in order:

1. Add a per-boundary cache-transfer setting.
2. Send the cache only before stages that start on a shared-index layer.
3. Measure P2P time and peak memory before and after this change.
4. Record per-layer forward, backward, FSDP gather, and memory cost.
5. Recompute boundaries from measured layer costs instead of layer count.
6. Add one bounded pinned side-input prefetch slot if the profile shows a CPU
   to GPU wait.
7. Test `reshard_after_forward=false` only after other work creates enough
   measured GPU space.

Interleaved 1F1B is later work. A normal x72p step had 94 microbatches and
eight stages. An ideal schedule can remove only about 7 percent of the simple
pipeline start and finish time at that microbatch count. The present side-input
FIFO also assumes forward calls stay in microbatch order. Interleaving needs
side inputs keyed by microbatch number and more than one local stage chunk.

The trainer configuration currently accepts `interleaved_1f1b`, but the
runtime rejects it. Remove this configuration mismatch before a later test.

## Memory work

The x72p failure was a live-value problem. Optimizer state was not the main
cause. The failed operation needed 24 MiB while only 1.5 MiB was physically
free.

Use this order:

1. Remove duplicate routed-token and LoRA outputs.
2. Reuse chunk buffers in forward and backward.
3. Measure saved LoRA A outputs and select save, recompute, or bounded offload.
4. Lower the W2 chunk size only when the trace shows the W2 output is still the
   largest short-lived allocation.
5. Test state-only Muon optimizer offload after the expert path is stable.
6. Keep `reshard_after_forward=true` until the measured peak permits a test.

State-only optimizer offload can recover some space. It does not remove the
expert activation peak. Full CPU optimizer work would also change optimizer
behavior and is not the first change.

## Rank and adapter transport

The old research estimates use rank 16 on dense and shared targets. The table
changes only the routed-expert rank.

| Routed rank | Adapter size | Estimated transfer at 3.63 MB/s |
|---:|---:|---:|
| 1 | 1.18 GB | 5.4 minutes |
| 2 | 2.16 GB | 9.9 minutes |
| 4 | 4.12 GB | 18.9 minutes |
| 8 | 8.05 GB | 37.0 minutes |
| 16 | 15.90 GB | 73.0 minutes |

The transfer times are estimates from the fastest measured 279 MB adapter
transfer. The link can change. Measure both H200 hosts for each format.

Current vLLM uses one rank and one scaling value for the complete adapter. It
does not read a PEFT `rank_pattern`. A mixed rank-16 dense and rank-4 expert
adapter therefore needs a loader change or local padding on each H200 host.
Padding before WAN transfer would remove the transport benefit.

Use this order for capacity tests:

1. Uniform rank 4 on all required projections for full-system correctness.
2. Uniform rank 8 if rank 4 is stable and learning is limited.
3. Mixed rank 16 for dense and shared weights with rank 4 or 8 for routed
   experts after variable-rank serving works.
4. Uniform routed rank 16 only after its transfer and load cost passes.

Keep alpha and rank recorded together. Do not change rank, alpha, learning
rate, and target coverage in one comparison.

Export only the adapter. Do not merge it into a full model checkpoint. Use
stacked expert tensors where the H200 loader can verify the layout. Test an
exact delta or shard transfer only after the ordinary adapter path is correct.
The receiver must reconstruct and verify the exact adapter hash.

## H200 inference work

The reviewed vLLM main revision is
`31305736307ecdfe3398b643240770651b523a84`.

Current vLLM lists `GlmMoeDsaForCausalLM` as supporting LoRA. It has a fused
MoE LoRA path for gate, up, and down projections. It supports per-expert and
stacked expert adapter layouts. It also has an FP8 MoE LoRA operation.

Prime-RL main pins vLLM 0.26.0. Test that version in a separate H200 image.
Do not change the active Muse engine image.

Use one TP8 GLM replica on each eight-H200 host. Run these tests:

1. Base model parity with the current image.
2. Dense and attention LoRA load, unload, and reload.
3. Routed-expert LoRA load, unload, and reload.
4. Exact adapter tensor count, shape, layout, and hash checks.
5. Direct trainer to inference log-probability comparison.
6. A real rollout-shape benchmark with prefix caching enabled.
7. `max_num_seqs` and `max_num_batched_tokens` sweeps.
8. The fused MoE LoRA dual-stream option.
9. A tuned fused MoE LoRA configuration for the selected rank.

Keep MTP disabled in the baseline. The previous GLM MTP test failed with an
integer stride limit. Test MTP only after the non-MTP adapter path passes. A
new vLLM version must also pass short-decode correctness because an open 2026
report shows bad GLM-5.2 output with one TP, EP, and DP combination.

The current measured rollout generation time is 21 minutes 58 seconds. The
project requirement is approximately 10 minutes. Report prompt processing,
generation, adapter transfer, and adapter load as separate times.

## Staging checklist

### Stage 1: Freeze the baseline

- [ ] Create a separate GLM staging branch and run directory.
- [ ] Keep the active Muse run unchanged.
- [ ] Record the exact local and upstream revisions.
- [ ] Add CUDA timing for attention, expert base work, LoRA work, expert
      dispatch, expert combine, PP send, PP receive, and FSDP gather.
- [ ] Record maximum GPU allocation on all 64 ranks.
- [ ] Record physical free memory before each expert W1, W3, and W2 operation.

### Stage 2: Prove target coverage

- [ ] Add a strict target registry.
- [ ] Add the special `kv_b_proj` LoRA calculation.
- [ ] Keep the indexer and router frozen.
- [ ] Check finite and nonzero gradients for every target class on every PP
      stage.
- [ ] Stop when any expected target is absent.

### Stage 3: Build the routed-expert LoRA kernel

- [ ] Build a standalone rank sweep on one RTX 4090.
- [ ] Run an exact-shape eight-GPU node test with real routing histograms.
- [ ] Compare output and all gradients with a BF16 reference.
- [ ] Add injected allocation-failure tests.
- [ ] Select the lowest-memory result that does not reduce throughput.

### Stage 4: Export and serve the adapter

- [ ] Export only adapter tensors.
- [ ] Verify all 75 MoE layers and all 256 experts.
- [ ] Load the adapter in an isolated vLLM 0.26 GLM server.
- [ ] Prove that changing one routed expert changes only prompts that visit it.
- [ ] Prove unload and reload restore the exact output and hash.
- [ ] Measure transfer and local load time on both H200 hosts.

### Stage 5: Improve PP8

- [ ] Stop the three unnecessary sparse-cache transfers.
- [ ] Use measured layer costs for boundaries.
- [ ] Test one bounded side-input prefetch slot.
- [ ] Measure 1F1B idle time before any schedule change.
- [ ] Do not add interleaving unless measured idle time can repay the code
      change.

### Stage 6: Full trainer qualification

- [ ] Use trainer-only fake data first. Do not collect a full rollout batch.
- [ ] Use PP8, CP8, node-local EP8, batch 1,024, and context 32,768.
- [ ] Run at least 20 changing trainer-only steps on all 64 RTX 4090 GPUs.
- [ ] Require finite gradients and no allocation failure.
- [ ] Require no trainer step above 15 minutes after step 1.
- [ ] Compare direct trainer log probabilities with H200 inference.
- [ ] Do not accept precision results worse than the prior validated x55a
      reference without an owner decision.

### Stage 7: Production acceptance

- [ ] Start a fresh run with fresh rollouts.
- [ ] Keep the learning rate at `8e-6` unless the owner changes it.
- [ ] Keep batch 1,024, group 16, and context 32,768.
- [ ] Report trainer compute, rollout generation, adapter transfer, adapter
      load, and waiting time separately.
- [ ] Complete at least 50 stable trainer steps.
- [ ] Use the reward trend verdict. Do not use an early reward change as proof.

## Research prompt for a second model

```text
You are doing a technical design review for a paused GLM-5.2 GRPO training
system. Do not implement code. Produce a source-backed report that can guide
implementation and benchmarks.

System facts:

- The model is GLM-5.2 with 78 decoder layers, 75 MoE layers, 256 routed
  experts per MoE layer, and top-8 routing.
- Training uses 8 nodes. Each node has 8 RTX 4090 GPUs. This gives 64 SM89
  trainer GPUs.
- The stable topology is PP8, CP8, and node-local EP8. Each PP stage is one
  physical node.
- Inference uses two separate hosts. Each host has 8 H200 GPUs and runs one
  TP8 GLM replica.
- The base checkpoint uses block FP8 expert weights and FP32 inverse scales.
- The local trainer has a custom SM89 W8A8 forward and input-backward path.
- Expert activation transport is INT4 in forward and FP8 in backward. Base
  weights are not INT4.
- The router is frozen. Router replay is disabled.
- x72p trained only four attention projections. x84p added the always-on
  shared expert. Neither run trained routed experts or dense MLP layers.
- x72p forward and backward took 10 minutes 54 seconds to 13 minutes 11
  seconds. x84p step 6 took 11 minutes 57 seconds.
- Peak allocation was about 44.22 to 46.55 GiB on 48 GiB GPUs. One step failed
  when a 24 MiB expert result had only 1.5 MiB physically free.
- x84p rollout generation took 21 minutes 58 seconds. The target is about 10
  minutes.
- The current 397 MB adapter does not include routed experts.
- Estimated adapter sizes are 4.12 GB for routed rank 4, 8.05 GB for rank 8,
  and 15.90 GB for rank 16 when dense and shared targets use rank 16.
- A 279 MB adapter needed 76.8 seconds in the fastest measured WAN transfer.
- Do not use QLoRA, INT4 base weights, or a merged full-model checkpoint.

Local code findings:

- `source/src/prime_rl/trainer/models/layers/fp8_ada.py` contains the current
  streamed native-FP8 expert forward and backward.
- `_linear_input_grad()` already uses W8A8 on SM89. The FP32 routine is the
  fallback and reference.
- `source/src/prime_rl/trainer/models/layers/lora/multi_linear.py` saves the
  full input and the A result. Its Triton matrix operation has simple fixed
  configurations.
- There is no correct routed-expert LoRA path for the local native-FP8 expert.
- Latest Prime-RL grouped expert LoRA converts base expert weights to BF16 in
  forward. It is not a direct fit for this checkpoint.
- `kv_b_proj` is read through `.weight` during the trainer MLA absorption.
  An ordinary wrapper does not apply a LoRA update to it.
- The current PP code sends the sparse index cache at all seven PP boundaries.
  Only four stage starts need it.
- A normal step has about 94 microbatches. An ideal schedule can remove only
  about 7 percent of the simple eight-stage start and finish time.
- Latest upstream Prime-RL sets trainer PP to 1. Its SM90 DeepGEMM and DeepEP
  paths cannot replace the local RTX 4090 path.

Research these sources first:

1. ScatterMoE paper and repository:
   https://arxiv.org/abs/2403.08245
   https://github.com/shawntan/scattermoe
2. Current Prime-RL main, especially grouped expert LoRA, FP8, optimizer
   offload, and expert communication:
   https://github.com/PrimeIntellect-ai/prime-rl
3. Current vLLM main, especially FusedMoEWithLoRA, its FP8 LoRA operation,
   GLM-5.2 support, expert parallel restrictions, adapter layouts, split-K,
   dual CUDA streams, and tuned kernel configurations:
   https://github.com/vllm-project/vllm
4. PyTorch pipeline schedules and FSDP2 behavior:
   https://docs.pytorch.org/docs/stable/distributed.pipelining.html
5. Relevant primary papers on fused LoRA training, MoE LoRA, activation
   memory, and low-rank capacity.

Answer these questions:

1. Which ScatterMoE ideas can be used for BF16 LoRA on top of a frozen
   block-FP8 expert without replacing the validated base expert operation?
2. Give a forward and backward kernel design for W1, W3, and W2. Show exact
   saved tensors, temporary tensors, reuse points, and gradient formulas.
3. Compare saving the LoRA A result, recomputing it, and bounded host offload.
   Estimate bytes for ranks 4, 8, and 16 at realistic routed-token counts.
4. Explain how to fuse row mapping with LoRA A and B without making a full
   grouped copy of routed inputs.
5. Explain how to keep source-major EP ordering and top-8 route scaling exact.
6. Find the best RTX 4090 kernel choices for skinny BF16 matrix operations.
   Do not recommend SM90-only software.
7. Check whether vLLM 0.26 can load and run GLM-5.2 expert LoRA with FP8 base
   weights. List exact limits for EP, sequence parallelism, adapter layout,
   rank, and scaling.
8. Determine how to serve mixed ranks, such as rank 16 on dense weights and
   rank 4 on routed experts, without padding the WAN file to rank 16.
9. Check all GLM attention projections. Explain the special work needed for
   `kv_b_proj` and whether the sparse indexer should remain frozen.
10. Estimate the possible PP improvement from per-boundary cache transfer,
    measured-cost stage boundaries, side-input prefetch, FSDP reshard changes,
    and interleaved schedules.
11. Give an inference benchmark plan for two independent TP8 H200 replicas.
    Include real rollout lengths, prefix reuse, max sequences, batched tokens,
    fused MoE LoRA tuning, and guarded MTP tests.
12. Give a staged test plan from one GPU to 8 GPUs, 64 GPUs, and two H200
    servers. Include correctness, memory, speed, adapter export, transfer,
    load, and trainer-to-inference parity.

Requirements for your report:

- Use primary sources for every external technical claim.
- Link each source next to the claim it supports.
- Separate measured project facts, upstream facts, estimates, and hypotheses.
- Do not transfer A100 or H100 performance claims directly to RTX 4090.
- Do not propose a full Prime-RL update unless you explain how local PP8 and
  the SM89 native-FP8 path remain intact.
- Do not propose FP8 trainable LoRA weights without direct numerical evidence.
- Keep the router frozen in the first experiment.
- End with a ranked implementation checklist and explicit stop conditions.
```

## Primary references

- [ScatterMoE paper](https://arxiv.org/abs/2403.08245)
- [ScatterMoE code](https://github.com/shawntan/scattermoe)
- [Prime-RL scaling documentation](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/scaling.md)
- [Prime-RL grouped expert LoRA](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/models/layers/lora/multi_moe.py)
- [vLLM LoRA documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/lora.md)
- [vLLM fused MoE LoRA](https://github.com/vllm-project/vllm/blob/main/vllm/lora/layers/fused_moe.py)
- [vLLM supported models](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)
- [PyTorch pipeline schedules](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [LoRA-FA](https://arxiv.org/abs/2308.03303)
