# Current handoff

Updated 2026-09-07 UTC. Use `configs/operations.toml` for every host, path,
port, container, and profile value.

## Current state

- Restart invoked at 23:45:14 UTC using `7fffd7d`. It is the same neutral
  `glm-perf-2` run, with fresh rollouts and a 20-step target. Completion is
  not yet confirmed. Root owns live operation and monitoring.
- Ownership repair `7912cc0` fixes both H200 broadcast-directory creators.
  The exact H200A directories now pass an actual UID 1000 write check.
  The complete saved step-0 adapter is readable on all eight trainer hosts.
  `7fffd7d` restores its per-expert tensors through `initial_adapter_path`
  after the run resets its adapter slot. CPU/Gloo tests passed BF16 native
  experts, sharded parameters, exact re-export, and rejection before mutation.
  This recovery is for the zero-B step-0 adapter only. Do not use it to resume
  a trained adapter: fresh-policy initialization still zeros B. The generic
  resume path also needs repair before use with a trained native adapter.
- The previous attempt stopped with 0 optimizer steps. Both H200s loaded the correct
  `glm-perf-2` step-0 adapter (B at 17:26:15 UTC, A at 17:26:20). A staging
  took 2,713.330 seconds; B took 116.598 seconds. The orchestrator then failed
  to write `broadcasts/step_0/.receivers-ready.json.<tmp>` on H200A because
  the step directory was root-owned with mode 0755. Trainers and orchestrator
  were stopped; inference engines and complete adapters were preserved.
  Directory ownership and its creators are now repaired. The restart reuses
  the saved complete adapter; confirm its source SHA and H200 cache remain
  unchanged before reporting that the full copy was avoided.
- No completed full production RL result is established. The neutral
  `glm-perf-1` run stopped after more than 43 minutes. Its batch contained
  16,520,157 effective tokens, including 11,935,614 generated tokens. At about
  2.1k tokens/s, that output volume implies about 95 minutes; this is an
  estimate, not a measured run time. The historical comparison processed
  2,097,152 tokens, so the runs are not comparable.
- A fresh production-shaped neutral `glm-perf-2` run started on 2026-09-07;
  all eight trainers were up by 15:34 UTC. The first attempt entered an API
  retry storm at 16:03:52 after about 1,800 seconds and stopped at 16:05:14
  with 0 optimizer steps, 32/1,024 collected rollouts, and 127 buffered
  rollouts. A had
  105 preemptions versus B 0 at 15:51; both were active with no OOM or
  restarts. The initial full uncompressed adapter was 30,767,549,784 bytes;
  A reached 44.2% at about 15:58 with a 10.94 MiB/s interval, not positive
  step-update timing.
- A same-run restart was invoked at 16:33:01 UTC and returned at 16:38:45.
  It brought all eight trainers up at about 16:34-16:35 UTC
  and the orchestrator at about 16:38 UTC. At 16:47 UTC, its initial adapter
  bootstrap was pending and it still had 0 optimizer steps. The old 18.337 GB
  partial was rejected because its source SHA differed (`2c2d5927...5072c`
  versus `948490...b17f2`); it was not reused, and no value change is proven.
  BF16-random A/zero B initialization is probable; the FP8 base was unchanged.
  All eight trainers had expert activation offload 0 and `sm89-r32`.
  Inference A and B remained MTP 1 with 32K context and existing graphs.
  Different image `nvcc` paths selected different FlashInfer/fusion paths; this
  runtime-path difference is unfixed. The first full step, performance, stability, and
  20-step completion remain pending, so this run has no production pass claim.
- The original cold start failed before trainers arrived: `install -d` assigned
  ownership only to the broadcasts leaf, leaving new run parents root-owned;
  creation of `control` and `.incoming` then failed with permission denied.
  The user-approved
  repair in `8bc0602` creates the run parent, service, and broadcasts
  directories explicitly as UID/GID 1000. The exact n82 directories were
  repaired before retry; 26 focused tests and `bash -n` passed.
- The null harness had an SDK 600-second default timeout and two retries inside
  the 3,600-second rollout limit; the dispatcher also retried every timeout
  `HarnessError`. `eb5ffad` disables the inner timeout/retries, requires a
  recorded failed policy call to retry a harness timeout, and mounts the
  updated harness. Sixteen focused
  tests, `bash -n`, and patch-block reverse checks passed.
- Trainer-only qualification is `glm52-fixed-token` on 64 GPUs (8 packed
  microbatches x 32,768 tokens, 262,144 total) with 512 selected tokens,
  learning rate 0, and adapter B zero at step 0.

## Trainer qualification

The following are elapsed qualification measurements with identical metrics:

- old utility (`a2fd4c2`): 133.8006 s and 135.4288 s total;
- stats utility (`894f2bb`): 136.0182 s and 137.6988 s total;
- current finite-gradient guard (`eb54291`): 135.4668 s and 137.1223 s total,
  peak 41.1816 GiB.

The old and stats qualification peaks were 41.1816 GiB and 41.0645 GiB.

These values do not establish a compute-speed change. Two actual 8-GPU tests
per rank passed in 3.58 to 3.62 s with nonfinite gradients rejected.

A one-step MTP fixed-token diagnostic with 512 selected tokens reported
134.9929 s forward/backward, 136.7785 s total, 41.1816 GiB peak memory,
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

An earlier PP16 qualification measured 227.13 s for pure PP work and 270.79 s
for the reported value including optimizer and profiler work, with 42.2 GiB
peak memory. It was not a full batch-1,024 result.

The 64-rank stats component test matched all statistics: the old path used
2,652 ms and 5,904 MiB, and the new path used 343 ms and 778 MiB. The host-list
component test used 2.998 s and 4.89 GiB before the change, and 0.081 s and
0.968 GiB after it. These are component measurements, not full trainer peak
memory measurements.

## Parity

- Direct 64-token-selected parity: mean absolute delta 0.04733 and maximum
  0.40776.
- Aggregate 512-token record from the old non-MTP reference: masked K3 mean
  0.1469, maximum 0.4887; unmasked K3 mean 0.0107. Guards passed, but no
  trained nonzero-adapter parity result is established.
- The parity reader now reports record and selected token counts and rejects
  invalid, empty, non-finite, or contradictory records.

## H200 inference

- The expert-target omission was confirmed and repaired at `9da34c2`; the
  load-boundary guard is at `a2fd4c2`.
- A and B are now configured with MTP 1. B is configured in
  `6a3e756`. B was ready at 22:28:01 UTC with TP8, EP8, rank-32 expert LoRA,
  32K context, and CUDA graphs 128. Reported weights, graph, and KV memory
  were 94.7 GiB, 2.11 GiB, and 396,160 tokens; baseline B KV capacity was
  478,720 tokens. Three 32,512-prompt plus 256-output samples took 8.360,
  6.743, and 6.873 s (34.95 tokens/s versus 27.08 before MTP), with 83%
  draft acceptance and no OOM or restart.
- In an exact 8 x 32,512-input plus 256-output warm pair, B first took
  34.429 s with 0 prefix hits; the immediate second took 8.285 s (247.18
  tokens/s) with 259,072 of 260,096 prefix hits. A warm run on a different
  host reached 226.22 tokens/s. This is a component indication, not a
  controlled full-run improvement. The B graph and MTP path stayed intact.
  Immediate repeated inputs reused the prefix cache.
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
- The MTP run total was 21.976009 s for 768 finite generated tokens, with all
  HTTP responses successful.
- A separate B concurrent test completed 3 rounds of 8 requests: all 24 HTTP
  responses were successful, with 6,144 finite generated tokens, zero errors,
  preemptions, or restarts, and 57.99, 55.59, and 59.74 tokens/s. KV drained;
  memory plateaued at 141,085 MiB after round 1. This has no concurrent MTP 0
  baseline, so it is not a speed comparison.
- Three same-adapter B unload/reload probes took 4.186 to 4.193 s each. Each
  8 x 32,512-input plus 256-output probe returned HTTP 200 with finite output
  and no errors. These are cached adapter reload checks, not new policy or
  WAN timing.
- A short A concurrency benchmark completed 128 of 128 requests using 512
  input and at most 64 output tokens in 18.7 s: 8,071 generated tokens,
  431.43 tokens/s, no OOM, preemption, or restart, and peak concurrency 128.
  This is not long-rollout throughput.
- The trainer/inference MTP comparison is complete for this diagnostic. There
  is no full rollout-throughput claim or production approval. Baseline and MTP
  tests had differing measured activation allocations, so the KV-capacity
  difference does not support a pure causal claim.
- A historical, different-workload H200 32K prefill test passed two 96-request
  runs at 1,459.6 and 1,454.5 tokens/s with 32,768 prefill tokens, rank 32,
  graphs 128, TP8, and EP8. The 65,536-token control crashed; its root cause
  is unknown.

MTP is vLLM's Multi-Token Prediction path; see the [vLLM MTP
documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/).

## Transfer and transport

- Saved encoded artifact size for trained adapter steps 3, 4, and 5 was 3.20,
  2.51, and 2.23 GB. This is not a WAN transfer measurement. Both H200 hosts
  still have only the cached `glm-perf-1` step 0; full positive-step delivery
  is pending.
- Stronger zstd at level 9 on 16 chunks (raw 266,571,096 bytes) saved 2.27%
  and added 29.6% CPU encode time, so it was rejected. A 64 MiB workspace path (18.04 s versus 10.50 s) was
  also rejected. No full new WAN result under 5 minutes is established.
- CPU damped-compression used 16 chunks x 5 coefficients and passed 80 exact
  round trips total. Deployed coefficient 1 produced the smallest
  18,057,344-byte artifact; alternatives were 56% to 160% larger, so no
  compression setting change is proposed.
- Grouped 24-process, four-forward China tunnels are validated with 96
  listeners. Restarting n82 `logind` cleared its exhausted session table
  (8192 limit); the shared root cause of earlier SSH-session warnings remains
  unresolved.
- Native cancellation now clears resources. The B relay is intentionally
  absent during trainer-only work.

## Versions and next checks

- Measured trainer image: `prime-rl:glm52-fp8`, PyTorch 2.11.0+cu128,
  NCCL 2.28.9.
- Measured H200 image: `prime-rl:glm52-v080-glm52-h200-router`, vLLM 0.26;
  the orchestrator image is `prime-rl:glm52-decensor`.
- Upstream `26b3131` was reviewed but is not upgraded.
- `prime-rl.patch` was refreshed against base
  `f50ec5c4c8c45864ec535dead2c3b630de50d068` for four source files. Its other
  47 blocks, including dependency blocks, are unchanged. Source/package hunks
  passed apply-check against this base with `deps/*` excluded. Dependency
  hunks were preserved, not reapplied to fresh dependency trees. This patch
  does not apply unchanged to reviewed upstream `26b3131`; that upgrade is not
  deployed.
- Keep the trainer timing limit at 600 seconds and the rollout limit at 720
  seconds. `watch()` reports only; it does not stop a test. The operator must
  stop and trace a test that exceeds the configured limit. A busy GPU is not
  proof of progress.
- Historical `glm-1` measurements are not the current neutral result. Do not
  restart that old run without an owner decision.
- No Decensor environment or reward edits were made in these fixes.
