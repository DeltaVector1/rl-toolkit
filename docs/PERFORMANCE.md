# Performance measurements

These are measured results. Keep trainer, rollout, adapter save, network
transfer, H200 activation, judging, and full model copy as separate values.

## Current limits

- Trainer forward and backward target: 10 to 15 minutes after step 1.
- Rollout target: about 10 minutes for 1,024 rollouts.
- Production settings: batch 1,024, group 16, context 32,768, output limit
  16,384.
- Final test: at least 50 complete production steps.

## Trainer

| Test | Result | Peak GPU memory |
|---|---:|---:|
| Batch-2,048 trainer-only, 20 steps | 11m37s to 11m44s; median 11m41s | 43.2 GiB |
| Production-shaped 20-step test | about 10 to 14 minutes | not recorded |
| Historical x24 step 3 | 14m36s; 2,645 tokens/s | 46.5 GiB |

The batch-2,048 test completed 20 of 20 steps without an OOM. The requested
3-to-5-minute trainer time has not been shown.

Do not enable activation offload. A prior test used 486 GiB of 503 GiB host
memory and stopped the node. With activation offload disabled, the host used
about 3 GiB and had about 460 GiB available.

The activation-checkpoint expert cache costs about 3.8 GiB for each GPU. It
also has unsafe single-slot reuse under the 1F1B schedule. Keep it disabled.

## Adapter movement

The rank-32 expert adapter is BF16 and is about 28.65 GiB, or 30,767,549,784
bytes. The base model remains FP8.

| Part | Measured time |
|---|---:|
| Trainer save and broadcast | 2m19s to 4m32s |
| Network transfer after relay repairs | 3m20s to 4m12s |
| H200 adapter activation | 4m18s to 4m40s |

Positive-delta network transfers on 2026-08-30 took 3m50s, 4m01s, and 4m12s.
The matching H200 activation times were 4m21s, 4m18s, and 4m40s.

The full GLM FP8 model is 755,632,050,320 bytes, about 704 GiB. This is not an
adapter transfer. Warn the owner before a full model copy, and report that time
separately.

Before the transport repairs, uncompressed bulk transfer was below 130 KiB/s.
Compressed transfer reached 4.9 MiB/s. Resumable transfer, file validation,
relay recovery, and staging the adapter once per host produced the times above.

## Rollouts and full cycles

| Test | Rollout time | Trainer time | Full cycle |
|---|---:|---:|---:|
| Historical x24 step 3 | 16m48s | 14m36s | 31m24s |
| Policy-7 measured cycle | 21m28s | 10m59s | overlap applies |
| Other recent batches | 17m36s to 21m55s | 10 to 14 minutes | not recorded |

Trainer work and rollout collection overlap. Do not add all values unless the
timestamps show that they ran in sequence.

The latest measured rollout times do not pass the target of about 10 minutes.
The current 48-tunnel inventory returns HTTP 200, but tunnel health alone does
not prove rollout throughput.

Historical WAN measurements had about 217 ms round-trip delay and packet loss
from 3 to 33 percent. Large request counts caused tunnel resets and read
timeouts. Compression, patient keepalive settings, bounded request timeouts,
and durable batch files reduced these failures.

## Numeric agreement

An old native W8A8 trainer test did not agree with H200 W8A16 inference:

| Metric | W8A8 trainer | BF16 trainer comparison |
|---|---:|---:|
| Mismatch KL | 7 to 9 nats | about 0.2 nats |
| Entropy | 9.2 nats | about 0.45 nats |
| Gradient norm | 200,000 to 330,000 | 0.013 to 0.043 |

The W8A8 result caused reward loss and was reverted. A low-memory W8A16
trainer path is still open work. Start in
`prime-rl/src/prime_rl/trainer/models/layers/fp8_ada.py`.

For the current path, compare selected-token FP32 log probabilities for the
same tokens and adapter. Test trainer scoring, inference prefill, eager decode,
CUDA-graph decode, long positions, and adapter reload. Do not change a parity
limit to hide a mismatch.

## Network settings

The old 64-rank network test favored one NCCL socket per rank. It measured
0.285 seconds for INT4 dispatch, 0.558 seconds for FP8 gradient traffic, and
0.885 seconds for BF16 traffic.

Eight NCCL channels measured 970 ms for each layer. Four channels measured
1,527 ms. The launcher still sets four channels, so this difference needs a
controlled current-topology test before a change.

Use `fq`, cubic congestion control, and MTU 1500. In the old test, `fq` reached
87 percent wire use; `fq_codel` reached 65 percent. BBR reached 36 percent.
MTU 9000 was slower than MTU 1500.

The current trainer uses PP8, CP8, and node-local EP8. Do not apply timings
from the removed global expert-parallel design to this topology.

## Open measurements

- A full 50-step production run has not passed.
- Rollout collection has not reached the 10-minute target.
- The complete MTP paired test at 32K context has not passed.
- The mixed H200 CUDA-graph memory difference is not explained. One host used
  10.89 GiB and the other used 2.04 GiB in the compared test.
- The measured reward slope from steps 11 to 40 was +0.0039 against a noise
  range of plus or minus 0.0165. This is not a reward trend.
