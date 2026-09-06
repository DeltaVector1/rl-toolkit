# Performance measurements

These are measured results. Keep trainer, rollout, adapter transfer, full-
weight transfer, H200 activation, judging, and save time separate.

## Current limits

- Trainer forward/backward reported time limit: 600 seconds. This metric
  includes existing statistics and optimizer work; it is not a pure kernel
  time.
- Rollout reported time limit: 720 seconds.
- Production settings remain batch 1,024, group 16, context 32,768, and output
  limit 16,384.
- No full 50-step neutral production run has passed.

## Neutral trainer qualification

`glm52-fixed-token` uses 64 GPUs, 8 packed microbatches x 32,768 tokens,
262,144 total tokens, 512 selected tokens, learning rate 0, and adapter B
zero at step 0.

| Utility state | Reported elapsed values | Peak reserved GPU memory |
|---|---:|---:|
| old `a2fd4c2` | 133.8006 s, 135.4288 s total | 41.1816 GiB |
| stats `894f2bb` | 136.0182 s, 137.6988 s total | 41.0645 GiB |
| current `eb54291` | 135.4668 s, 137.1223 s total | 41.1816 GiB |

Metrics were identical. These values do not show a compute-speed change. The
stopped `glm-perf-1` batch contained 16,520,157 tokens; it did not complete a
training step. It is not comparable with the historical 2,097,152-token
result, and no neutral 20-step or 50-step pass is established.

Two actual 8-GPU tests per rank passed in 3.58 to 3.62 s with nonfinite
gradients rejected.
An earlier PP16 qualification measured 227.13 s for pure PP work and 270.79 s
for the reported value including optimizer and profiler work, with 42.2 GiB
peak reserved GPU memory. It was not a full batch-1,024 result.

A one-step MTP fixed-token diagnostic with 512 selected tokens reported
134.9929 s forward/backward, 136.7785 s total, 41.1816 GiB peak reserved GPU
memory,
0.001953 masked fraction, 0.011633 masked K3, and 0.013462 unmasked K3. All
8 trainer hosts exited 0 with no OOM. Direct 64-token-selected parity measured
0.05851924324 mean absolute delta and 0.8908292055 maximum. This is a
zero-learning-rate diagnostic, not an acceptance result.

`glm52-native-small` now has `max_steps = 20` for a fresh synthetic memory
test. Each step uses 16 x 32,768 = 524,288 fresh synthetic tokens with a seed
and batch counter, real LR0 backward, no router replay, export, or checkpoint,
and a one-time rank-8 profiler. It is not full RL or an acceptance run.

## Stats component measurements

The 64-rank stats component test produced exact matching statistics:

| Path | Time | Peak reserved GPU memory |
|---|---:|---:|
| old padded gather | 2,652 ms | 5,904 MiB |
| uneven gather | 343 ms | 778 MiB |

The host-list component used 2.998 s and 4.89 GiB before the change, and
0.081 s and 0.968 GiB after it. These are component measurements, not full
trainer peak memory measurements.

## Parity and inference

- Direct 64-token-selected parity had mean absolute delta 0.04733 and maximum
  0.40776.
- The aggregate 512-token record from the old non-MTP reference had masked K3
  mean 0.1469, maximum 0.4887, and unmasked K3 mean 0.0107. Guards passed,
  but no trained nonzero-adapter parity result is established.
- The expert-target omission was confirmed and repaired at `9da34c2`; the
  load-boundary guard is at `a2fd4c2`.
- H200 A uses MTP 0. B uses MTP 1 from `6a3e756`. B was ready at 22:28:01 UTC
  with TP8, EP8, rank-32 expert LoRA, 32K context, and graphs 128. Weights,
  graph, and KV memory were 94.7 GiB, 2.11 GiB, and 396,160 tokens; baseline B
  KV capacity was 478,720 tokens. Three 32,512-prompt plus 256-output samples
  took 8.360, 6.743, and 6.873 s, or 34.95 tokens/s versus 27.08 before MTP,
  with 83% draft acceptance and no OOM or restart. This is a 29% single-
  request result with lower KV capacity, not an unchanged-capacity speed
  claim.
- The MTP run total was 21.976009 s for 768 finite generated tokens, with all
  HTTP responses successful.
- In an exact 8 x 32,512-input plus 256-output warm pair, B first took 34.429 s
  with 0 prefix hits; the immediate second took 8.285 s (247.18 tokens/s) with
  259,072 of 260,096 prefix hits. A warm run on a different host reached
  226.22 tokens/s. This is a component indication, not a controlled full-run
  improvement. The B graph and MTP path stayed intact. Immediate repeated
  inputs reused the prefix cache.
- A separate B concurrent test completed 3 rounds of 8 requests: all 24 HTTP
  responses were successful, with 6,144 finite generated tokens, zero errors,
  preemptions, or restarts, and 57.99, 55.59, and 59.74 tokens/s. KV drained;
  memory plateaued at 141,085 MiB after round 1. No concurrent MTP 0 baseline
  exists, so this is not a speed comparison. This remains a small MTP trial,
  not production approval. Baseline and MTP tests had differing measured
  activation allocations, so the KV-capacity difference does not support a
  pure causal claim.

A historical, different-workload H200 32K prefill test passed two 96-request
runs at 1,459.6 and 1,454.5 tokens/s with 32,768 prefill tokens, rank 32,
graphs 128, TP8, and EP8. The 65,536-token control crashed; its root cause is
unknown.

MTP is vLLM's Multi-Token Prediction path; see the [vLLM MTP
documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/).

## Adapter transfer and transport

Saved encoded artifact size for trained adapter steps 3, 4, and 5 was 3.20,
2.51, and 2.23 GB. This is not a WAN transfer measurement. Both H200 hosts
still have only cached `glm-perf-1` step 0, so full positive-step delivery is
pending.

Stronger zstd at level 9 on 16 chunks (raw 266,571,096 bytes) saved 2.27% and
added 29.6% CPU encode time; it was rejected. A 64 MiB workspace path
(18.04 s versus 10.50 s) was also rejected. No full new WAN result under 5
minutes is established.

Grouped 24-process, four-forward China tunnels are validated with 96
listeners. Restarting n82 `logind` cleared its exhausted session table (8192
limit), but the shared root cause of earlier SSH-session warnings remains
unresolved.

The full GLM FP8 model is 755,632,050,320 bytes, about 704 GiB. It is not an
adapter transfer. Warn before a full model copy and report that time separately.

## Historical measurements and warnings

- A 20-step batch-2,048 trainer test took 11m37s to 11m44s, with 43.2 GiB
  peak memory. This is historical, not the current neutral result.
- Historical W8A8 comparison measured mismatch KL 7-9 nats, entropy 9.2 nats,
  and gradient norm 200,000-330,000. The BF16 comparison was about 0.2 nats,
  0.45 nats, and 0.013-0.043. These are historical and not the current
  neutral path.
- The historical BF16 rank-32 adapter was 30,767,549,784 bytes while the base
  remained FP8. Historical save and broadcast time was 2m19s to 4m32s; keep
  it separate from network transfer and H200 activation time.
- Historical positive-delta network transfers took 3m50s, 4m01s, and 4m12s;
  matching H200 activation took 4m21s, 4m18s, and 4m40s.
- Activation offload used 486 GiB of 503 GiB host memory and stopped a node.
  With it disabled, host use was about 3 GiB. Current configs set
  `ac_offloading = "None"`. Historical `PRIME_EP_AC_CACHE=1` enabled an
  INT4/MXFP8 AllToAllExpertParallel single-slot cache. It cost about 3.8 GiB
  per GPU and had unsafe single-slot reuse under 1F1B. The current
  `bf16_forward_fp8_local_recompute_backward` path with
  `keep_routed_expert_activations_from_pp_stage=5` uses
  NativeFP8LoRAAllToAllExpertParallel: stages 0-4 checkpoint and recompute
  experts, while stages 5-7 retain activations. Do not disable this current
  retention based on the old cache warning.

## Version and topology notes

The measured trainer image is `prime-rl:glm52-fp8` with PyTorch 2.11.0+cu128
and NCCL 2.28.9. The measured H200 image is
`prime-rl:glm52-v080-glm52-h200-router` with vLLM 0.26. Upstream `26b3131`
was reviewed but is not upgraded.

The local `prime-rl/` overlay is copied to configured host paths at deploy;
launchers bind-mount files into `/app/src`. Dockerfile `COPY` is a separate
image build step, and deploy does not modify an existing image or hot-reload a
running process. The portable patch is not currently regenerated.

No Decensor environment or reward edits were made in these fixes.
