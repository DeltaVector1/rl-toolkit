# Performance measurements

Updated 2026-09-07 UTC.

These are measured results. Keep trainer, rollout, adapter transfer, full-
weight transfer, H200 activation, judging, and save time separate.

## Current limits

- Trainer forward/backward reported time limit: 600 seconds. This metric
  includes existing statistics and optimizer work; it is not a pure kernel
  time.
- Rollout reported time limit: 720 seconds.
- Production settings remain batch 1,024, group 16, context 32,768, and output
  limit 16,384.
- No full 20-step or 50-step neutral production run has passed.

## Current neutral production-shaped run

Latest attempt: 0 optimizer steps. Both H200s loaded the correct step-0
adapter (B at 17:26:15 UTC, A at 17:26:20). A received all bytes in
2,674.547 seconds and finished staging in 2,713.330 seconds. B received all
bytes in 82.705 seconds and finished staging in 116.598 seconds. These are
full startup-copy measurements, not positive-step update measurements.
The orchestrator then failed to write its receiver confirmation into the
root-owned H200A `broadcasts/step_0` directory. Training was stopped; engines
and complete adapters were preserved. Fresh rollout and compute timings for
this restart remain unmeasured.

`glm-perf-2` started on 2026-09-07; all eight trainers were up by 15:34 UTC.
The first attempt entered an API retry storm at 16:03:52 after about 1,800
seconds and stopped at 16:05:14 with 0 optimizer steps, 32/1,024 collected
rollouts, and
127 buffered rollouts. A had 105 preemptions versus B 0 at 15:51; both were
active with no OOM or restarts. The initial full uncompressed adapter was
30,767,549,784 bytes; A reached 44.2% at about 15:58 with a 10.94 MiB/s
interval, not positive step-update timing.

A same-run restart was invoked at 16:33:01 UTC and returned at 16:38:45.
It brought all eight trainers up at about 16:34-16:35 UTC and
the orchestrator at about 16:38 UTC. At 16:47 UTC, its initial adapter
bootstrap was pending and it still had 0 optimizer steps. The old 18.337 GB
partial was rejected because its source SHA differed (`2c2d5927...5072c`
versus `948490...b17f2`); it was not reused, and no value change is proven.
BF16-random A/zero B initialization is probable; the FP8 base was unchanged.
All eight trainers had expert activation offload 0 and `sm89-r32`; inference A
and B remained MTP 1 with 32K context and existing graphs. Different image
`nvcc` paths selected different FlashInfer/fusion paths, and this runtime-path
difference is unfixed. The first full step, performance, stability, and 20-step completion
remain pending; no production pass claim is made.

The first attempt's null harness used an SDK 600-second default timeout and two
retries inside the 3,600-second rollout limit; the dispatcher also retried every
timeout `HarnessError`. Commit `eb5ffad` disables the inner timeout/retries,
requires a recorded failed policy call to retry a harness timeout, and mounts
the updated harness.
Sixteen focused tests, `bash -n`, and patch-block reverse checks passed.

The earlier stopped `glm-perf-1` batch contained 16,520,157 effective tokens,
including 11,935,614 generated tokens. At about 2.1k tokens/s, that output
volume implies about 95 minutes; this is an estimate, not a measured run time.
The 524,288-token synthetic memory step took 3m25s, but it is not full-run
timing or a production result.

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
stopped `glm-perf-1` batch contained 16,520,157 effective tokens, including
11,935,614 generated tokens; it did not complete a training step. It is not
comparable with the historical 2,097,152-token result, and no full neutral
production RL 20-step or 50-step pass is established.

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
and batch counter, real LR0 backward, no adapter export, no RL rollout, no
router replay, and no checkpoint, with a one-time rank-8 profiler. The test
completed 20/20 steps on all 8 trainer hosts; all exited 0 with no OOM. Step 1
forward/backward took 272.7 s. Steps 2-20 had a median of 204.8 s and a range
of 201.3-208.4 s. Step total time had a median of 205.5 s and a range of
202.0-209.2 s. Peak reserved memory was 42.2168 GiB at step 1, 42.6543 GiB
at step 2, then flat at 42.9277 GiB for steps 3-20.
This is a synthetic memory test, not full RL or an acceptance run.

## Stats component measurements

The 64-rank stats component test produced exact matching statistics:

| Path | Time | Peak allocated GPU memory |
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
- H200 A and B now use MTP 1. B is configured from `6a3e756` and was ready at
  22:28:01 UTC with TP8, EP8, rank-32 expert LoRA, 32K context, and graphs 128.
  Weights,
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
- The prior A MTP 0 historical control took 225.89, 203.57, and 225.13
  tokens/s (median 225.13). With MTP 1, the same-host A test took 285.51,
  286.86, and 268.93 tokens/s (median 285.51, about 26.8% higher). Across
  32 requests including warmup, all requests returned HTTP 200; 8,192
  generated token log-probabilities were finite, with no errors, restarts, or
  preemptions. MTP accepted 3,294 of 4,844 draft tokens (about 68%). Graphs
  and context were unchanged. A's reported KV capacity was 368,960 tokens
  versus the prior 416,512; activation
  allocations differed, so this is not a pure causal speed claim.
- Both H200 services use active KV block size 64. Reported KV capacity was
  368,960 tokens (18.81 GiB) on A and 396,160 tokens (20.19 GiB) on B.
- A separate B concurrent test completed 3 rounds of 8 requests: all 24 HTTP
  responses were successful, with 6,144 finite generated tokens, zero errors,
  preemptions, or restarts, and 57.99, 55.59, and 59.74 tokens/s. KV drained;
  memory plateaued at 141,085 MiB after round 1. No concurrent MTP 0 baseline
  exists, so this is not a speed comparison. The trainer/inference MTP
  comparison is complete for this diagnostic. There is no full rollout-
  throughput claim or production approval. Baseline and MTP tests had
  differing measured activation allocations, so the KV-capacity difference
  does not support a pure causal claim.
- Three same-adapter B unload/reload probes took 4.186 to 4.193 s each. Each
  8 x 32,512-input plus 256-output probe returned HTTP 200 with finite output
  and no errors. These are cached adapter reload checks, not new policy or
  WAN timing.
- A short A concurrency benchmark completed 128 of 128 requests using 512
  input and at most 64 output tokens in 18.7 s: 8,071 generated tokens,
  431.43 tokens/s, no OOM, preemption, or restart, and peak concurrency 128.
  This is not long-rollout throughput.

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

CPU damped-compression used 16 chunks x 5 coefficients and passed 80 exact
round trips total. Deployed coefficient 1 produced the smallest
18,057,344-byte artifact; alternatives were 56% to 160% larger, so no
compression setting change is proposed.

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
running process. `prime-rl.patch` was refreshed against base
`f50ec5c4c8c45864ec535dead2c3b630de50d068` for four source files. Its other 47
  blocks, including dependency blocks, are unchanged. Source/package hunks
  passed apply-check against this base with `deps/*` excluded. Dependency
  hunks were preserved, not reapplied to fresh dependency trees. This patch
  does not apply unchanged to reviewed upstream `26b3131`; that upgrade is not
  deployed.

No Decensor environment or reward edits were made in these fixes.
