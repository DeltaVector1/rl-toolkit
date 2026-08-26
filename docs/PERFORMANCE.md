# GLM-5.2 DECENSOR RL. PERFORMANCE MEASUREMENTS

This document holds the transport and throughput measurements. It was moved out
of `AGENTS.md`, which is loaded into every session and must stay short.

Language: ASD-STE100 Simplified Technical English.

# READ THE ERA BEFORE YOU USE A NUMBER

The cluster changed topology. Numbers from one era do not apply to the other.

Era 1 is global expert parallelism. `global_ep = true`, `cp = 4`, no pipeline
parallelism, 88 rollouts each batch. Every MoE dispatch and combine crossed the
10 Gb/s node link. The all-to-all was 93 percent of a micro-batch.

Era 2 is the current trainer topology. `global_ep = false`, `cp = 8`, `pp = 8`, `ep = 8`
node-local, batch 1024, 16K context. The MoE all-to-all is node-local and about
80 times faster. The transport analysis in Era 1 no longer sets the step time.

Era 2 run x24 step 3 took 31.4 minutes for the full cycle: 14.6 minutes of
trainer compute and 16.8 minutes waiting for rollouts. Trainer throughput was
2645 tokens/s, MFU 3.2 percent, and peak memory 46.5 GiB.

The MFU is 3.2 percent. The step time has much space to decrease.

# X26 EDGE FILESYSTEM TEST

Test x26 ran on 2026-08-07. The orchestrator ran on h200a and sent completed
batches to n82. The trainer and rollout settings stayed at batch 1024, group
16, 128 rollouts in flight, and 16K context.

| Step | Collection | Attempts | Trainable | Errors | Batch file | Transfer |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20m 5s | 1,056 | 1,029 | 2.5% | 84.2 MB | 28.8 s |
| 2 | 18m 41s | 1,024 | 1,023 | 0.1% | 92.5 MB | 29.4 s |

The measured collection rates were 51.2 and 54.8 trainable rollouts each
minute. Both destination SHA-256 hashes matched the source. The transfer
overlapped the next collection.

These results measure rollout collection and file transfer. They do not
measure a completed trainer step. The trainer failed during its first forward
pass on two attempts. The first process failure was n80 rank 35. The remaining
ranks then reported NCCL remote-process errors.

# THE SETTINGS THAT STILL APPLY

These came from Era 1 but they tune the NIC, not the parallelism. `cluster.sh`
still applies them.

# NCCL socket count

NCCL opened `NCCL_SOCKET_NTHREADS=4` times `NCCL_NSOCKS_PERTHREAD=4`, which is
16 sockets for each rank. That made 128 TCP flows share one 10 Gb/s NIC on each
node. The flows then congested each other.

Measured on the real 64-rank expert-parallel group:

| Sockets per rank | INT4 dispatch | FP8 gradient | BF16 |
|---|---:|---:|---:|
| 16 (4 x 4, the default) | 0.506 s | 0.964 s | 1.551 s |
| 8 (2 x 2) | 0.437 s | 0.628 s | 1.236 s |
| 2 (1 x 2) | 0.329 s | 0.650 s | 1.101 s |
| 1 (1 x 1, current) | 0.285 s | 0.558 s | 0.885 s |

Use 1 socket for each rank.

# NCCL channel count

| `NCCL_*_NCHANNELS` | Time for each layer | Wire efficiency |
|---|---:|---:|
| default | 1835 ms | 44% |
| 2 | 1888 ms | 45% |
| 4 | 1527 ms | 55% |
| 8 (current) | 970 ms | 87% |
| 16 | 972 ms | 87% |

The measurement favored 8. The current launcher sets 4. This discrepancy is
unresolved; changing it changes trainer performance and needs a measured run.
One channel hangs and must not be used.

The value 87 percent is the measured ceiling of this NIC, so almost nothing is
left on the wire.

# Queueing discipline

`fq` gives 87 percent wire efficiency. The default `fq_codel` gives 65 percent.
`cluster.sh` applies `fq` before it starts the trainer.

BBR congestion control is worse here: 36 percent. Stay on `cubic`.

Jumbo frames were tested end to end. MTU 9000 passes the switch with no loss,
but the all-to-all became slower: 0.452 s against 0.285 s. Stay at MTU 1500.

# The expert activation-checkpoint cache

`PRIME_EP_AC_CACHE=1` keeps the INT4 payload that each rank received in the
forward pass and uses it again on the recompute. It costs about 3.8 GB for each
GPU.

Do not enable it on the current trainer. x26 failed twice when rank 35 needed a
308 MiB grouped-expert output with only 287.62 MiB free. The cache also has
only one dispatch and combine slot for each layer while 1F1B has multiple
micro-batches in flight. The slot can be overwritten before its matching
recompute.

`PRIME_EP_AC_CACHE=0` recomputes the same INT4 payload and repeats the
node-local all-to-all. This changes performance, not the model values, and
restores sufficient memory headroom. Selective activation checkpointing stays
enabled.

# THE LINK TO THE ENGINES

This is a long public link, not a datacentre network. Measured from n82: 217
milliseconds round trip, and packet loss between 3 and 33 percent that differs
between the two hosts.

Widening from 8 to 24 tunnels raised historical collection from about 4/min
to 13-16/min. x25 uses 48 ports, 18001-18048.

Uncompressed bulk transfer measured less than 130 KB/s. Compressed transfer
measured 4.9 MB/s. Therefore a batch can cross the link, but it must use a
resumable, integrity-checked bulk transport. Individual rollout responses and
WAN ZMQ do not provide that guarantee. See `docs/ROLLOUT_TRANSPORT.md`.

# THE TUNNEL CONCURRENCY CEILING

The tunnels limit the number of rollouts in flight. They do not limit the batch
size. The two values are independent.

| Rollouts in flight | Healthy tunnels | Engine concurrency |
|---|---:|---:|
| 256 | 20 to 22 of 24 | 45 to 70 |
| 1024 | 10 of 24 | 1 to 6 |

At 1024 the watchdog rebuilt 10 to 14 tunnels every 45 seconds and they failed
again at the same rate.

Set `batch_size` for the gradient quality. Set `max_inflight_rollouts` for the
tunnel load. The two are independent.

Read the two rows above with care. Both were sampled at a batch tail, when only
a few rollouts were in flight, so they overstate how healthy 256 was. Under
real load at 256 the count fell to 1 to 10 of 24. The current value is 128. See
"THE ROLLOUT TRANSPORT" below for the measurements that replace this table.

The healthy-tunnel count is also a weak signal on its own. The probe shares
each tunnel with rollout traffic, so it starves on links that still carry
rollouts. Confirm against the engines before acting on it.

# THE ROLLOUT TRANSPORT

The step time is set by rollout collection, not by the trainer. Measured over
steps 1 to 7 of run x24:

| Step | Total | Trainer compute | Waiting for rollouts |
|---|---:|---:|---:|
| 1 | 96.2m | 11.3m | 84.8m |
| 2 | 52.2m | 12.3m | 39.8m |
| 3 | 31.4m | 14.6m | 16.8m |
| 4 | 65.1m | 14.7m | 50.2m |
| 5 | 47.7m | 8.4m | 39.2m |
| 6 | 135.1m | 6.3m | 128.7m |
| 7 | 183.8m | 12.1m | 171.5m |

The trainer finishes in 6 to 15 minutes and then idles. Step 6 was 95 percent
idle. Do not tune the trainer to reduce the step time. A 15 minute step needs
68 rollouts each minute.

# What limited the rollouts

Every dropped rollout was a transport failure. Step 6 lost 78.4 percent of its
rollouts, and every signature was `ReadTimeout`. The `zero_advantage` filter
dropped 0.5 percent, gibberish 0.2 percent, and truncation 0 percent. Rollout
throughput and rollout loss are the same problem.

Three faults, in the order they were found:

The rollout tunnels had no compression, while the batch tunnel had it. The
batch tunnel measured under 130 KB/s without it and 4.9 MB/s with it.

The SSH keepalive was 10 x 3, so 30 seconds of lost keepalives killed a tunnel.
Keepalives are ordinary packets on a link that loses 10 to 33 percent of them,
so SSH killed live tunnels and every request in flight on them.

`build_async_openai` in verifiers left the OpenAI SDK defaults: `connect=5.0s`
and `read=600s`. A wedged request therefore held one of the fixed in-flight
slots for 10 minutes. Measured before the fix: 121 rollouts in flight while the
two engines ran 3 between them. Most of the in-flight count was dead requests
and the engines starved behind them.

# The effect

| Configuration | Collection | Full batch of 1024 | ReadTimeouts |
|---|---:|---:|---:|
| 256 in flight, no compression | 9.7/min | 106 min | 2.2/min |
| plus compression and patient keepalive | 13.0/min | 79 min | 0.75/min |
| plus 128 in flight and bounded timeouts | 15.7/min | 65 min | 0.22/min |

The step 6 error rate was 78.3 percent. Step 7 was 18.9 percent.

The current value is 128. A later 192-inflight measurement was not faster and
had about five times as many errors. Re-test concurrency only after the
orchestrator moves beside the engines; the WAN result does not measure local
engine capacity.

# WHAT ACTUALLY CROSSES THE LINK

Measured on a single `/inference/v1/generate` call, 2026-08-07:

| Field | Share of compressed bytes |
|---|---:|
| `routed_experts` | 95.5% |
| `logprobs` | ~4% |
| `token_ids`, `finish_reason`, `index` | <1% |

`routed_experts` is 711 bytes for each generated token. It is already encoded
as tightly as it can be: `dtype` is `uint8`, exactly 1 byte for each element,
and it gzips only 1.4 times because it is already binary. Its shape is
`[positions, 78 MoE layers, 8 experts]`, so each token costs 78 x 8 = 624
expert indices.

That is 2.79 MB for each rollout on the wire, and about 5.2 Mbit/s at 14
rollouts a minute, over a path that loses 7.5 percent of its packets.

The useful content, the tokens and their logprobs, is under 5 percent of it.

# OPT-IN ADJACENT ADAPTER DELTAS

The adapter delta is disabled by default.  Set `PRIME_ADAPTER_DELTA=1` only
for a measured qualification.  The exporter XORs the serialized adapter bytes
and compresses independent zlib frames.  The H200 checks the base SHA-256,
target SHA-256, adapter configuration hash, safetensors tensor-metadata hash,
frame codec, and byte count before it writes `READY`.

The wire-size break-even is exact:

```
delta_bytes + manifest_bytes < full_transfer_wire_bytes
```

The exporter uses a 0.90 delta-to-target-size guard.  At runtime the selector
also requires a current full-transfer wire estimate or a known complete-file
size, and compares `delta_bytes + manifest_bytes` with that value.  Without
that estimate it selects the current complete adapter transfer.  A missing or
mismatched base, an invalid frame, or a ratio at or above the guard selects the
same fallback.
For gzip or zstd, the estimate must be an encoded response length (from
`X-Prime-Wire-Length`, an encoded `Content-Length`, or
`PRIME_ADAPTER_FULL_WIRE_SIZE`); a raw target size is not used as a compressed
wire estimate.
The range server emits `X-Prime-Wire-Length`, `X-Prime-Wire-Codec`, and
`X-Prime-Raw-Length` on a full-file `HEAD`; the client rejects missing or
mismatched identity metadata.
The normal gzip transfer and production profiles are unchanged.

The digest-keyed H200 adapter cache has a 61,540,000,000-byte default cap
(two 30.77 GB generations) and keeps at most one additional stale generation.
The active target and the adjacent base considered for a transfer are
protected from pruning.  Operators can set `PRIME_LORA_CACHE_MAX_BYTES` and
`PRIME_LORA_CACHE_MAX_STALE_ENTRIES` for a smaller local limit; pruning removes
old stale entries first and never removes a protected entry.
Orphaned `.partial-*` directories older than one hour are removed; newer
partials count toward cache bytes.  Before a delta download, staging checks
free space for active adapter + base adapter + assembling target and fails
before transfer when that peak does not fit.

The measurement gate is exactly two adjacent real exports from one run:

1. Save step `S` and the immediately following step `S+1` with the normal
   safetensors exporter and `STABLE` markers.
2. Record the complete gzip wire bytes for `S+1`, the delta frames plus
   manifest for `S -> S+1`, transfer time for each, and all source/H200 SHA-256
   and tensor-metadata checks.
3. Repeat for the next adjacent pair `S+1 -> S+2`.  Both pairs must reconstruct
   byte-for-byte, pass metadata checks, and have delta wire bytes below the
   complete gzip wire bytes.  A single synthetic pair or one adjacent pair is
   not a production gate.

This described x24 with router replay enabled. x25 disables router replay and
routed-expert return together, so this field no longer limits batch size.

`enable_router_replay = true` is what asks for it. The trainer replays the
engine's expert choices instead of choosing its own, so the two forward passes
agree. Without it the payload falls by about 20 times.

The LoRA adapts `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, and `o_proj`
only. The router is not adapted and its weights are frozen and identical on
both sides, so the only source of routing disagreement is numeric: the trainer
runs bf16 with `moe_router_dtype = "float32"` and the engine runs W8A16. How
much that actually flips a top-8 selection is not known and has not been
measured.

Current mismatch KL with replay on: masked 0.19 to 0.21, unmasked 0.012,
entropy 0.47. Those are the numbers a run without replay must not spoil. For
comparison, the FP8 trainer experiment pushed masked mismatch KL to 7 to 9
nats and entropy to 9.2, and the reward collapsed.

# THE TOPOLOGY, AND WHAT IT RULES OUT

```
H200 (docker) --internet, 7.5% loss, 210 ms--> jump 202.101.73.12 --LAN--> n82
```

The engines reach n82 through `ProxyJump`, so each tunnel is SSH inside SSH and
the lossy leg carries two layers of channel flow control.

The bulk data travels H200 to n82, so the H200 is the sender and its congestion
control is the one that matters. BBR cannot be loaded there: the engine hosts
are Docker containers without `CAP_SYS_MODULE`, and the host has not loaded
`tcp_bbr` either.

A userspace FEC tunnel (kcptun, UDPspeeder, frp over KCP) would fix the loss
without any kernel module, but it needs a port. Measured from an engine host:
UDP and arbitrary TCP to the jump host both time out. Only SSH on port 31331
crosses. Ask the provider to open one UDP port if this route is wanted.

`scripts/network/tune_wan.sh` applies what is possible today: BBR on n82 for
the two engine routes only, and on both ends `tcp_slow_start_after_idle=0`,
`tcp_mtu_probing=1`, and a 32 MB buffer ceiling. Do not set BBR globally on
n82. It is a trainer node and NCCL crosses the same interface, where cubic
measured 87 percent against BBR's 36 percent.

# THE MEMORY MEASUREMENTS

| Condition | Host memory |
|---|---|
| Activation offloading on | 486 GB of 503 GB, then the node stopped |
| Activation offloading off | about 3 GB used, 460 GB available |

This memory is real. An earlier note called it an accounting artifact. That
note was wrong and is retracted. The kernel trace showed `shmem_fault`,
`shmem_alloc_folio`, `__alloc_pages`, and `out_of_memory`.

The pipeline stage balance moved the peak GPU memory from 24.6 GB to 21.9 GB
for each GPU. See `docs/PIPELINE_PARALLEL.md`.

# THE PRECISION MEASUREMENTS

| Metric | FP8 trainer weights | bf16 trainer weights |
|---|---|---|
| Mismatch KL | 7 to 9 nats | 0.2 nats |
| Positive advantage | 0.003 | 0.3 |
| Entropy | 9.2 nats | 0.45 nats |
| Gradient norm | 200000 to 330000 | 0.013 to 0.043 |

The vocabulary entropy limit is 11.95 nats. The gradient clip limit is 1.0.

The cause is that the engines use W8A16 and the trainer FP8 path uses W8A8. The
trainer was less exact than the engines, so the trust region masked almost
every positive advantage.

The open improvement is to make the trainer use W8A16. That recovers the FP8
memory and keeps the mismatch KL low. Start at `_dequant_weight` in
`fp8_ada.py`.

# ERA 1 ARCHIVE

Keep these numbers for reference only. They describe global expert parallelism,
which x25 and x26 did not use.

Each node has one 10 Gb/s link, `vlan130`, MTU 1500, shared by its 8 GPUs. The
second port is not cabled.

The 75 MoE layers hold 725 billion routed-expert parameters. Sharded across all
64 GPUs that is 11.3 GB for each GPU. Node-local expert parallelism would need
90.6 GB for each GPU, which does not fit, so Era 1 required `global_ep` and
every dispatch and combine crossed the network.

One micro-batch of 4096 tokens for each data rank, measured with per-layer CUDA
event timers:

| Phase | Time | Passes for each layer |
|---|---:|---|
| Forward dispatch all-to-all | 46.2 s | 2 |
| Forward combine all-to-all | 45.3 s | 2 |
| Backward dispatch all-to-all | 36.6 s | 1 |
| Backward combine all-to-all | 35.8 s | 1 |
| Expert count metadata all-to-all | 0.4 s | 2 |
| Attention, expert GEMM, router, output head | 12.9 s | |

All-to-all was 93 percent of the micro-batch, so kernel tuning could not move
the step time.

The remaining gap between the wire floor and the measured time was expert load
imbalance. `all_to_all_single` finishes at the pace of its busiest peer.

| Split skew | Time for each layer | Wire efficiency |
|---|---:|---:|
| 1 (uniform) | 942 ms | 90% |
| 2 | 1827 ms | 46% |
| 4 | 3567 ms | 24% |

The measured 103 to 108 seconds sat between skew 1 and skew 2, so live routing
carried about a 1.5 times imbalance. `enable_router_replay` is on, so the
trainer replays the routing of the inference engine and cannot flatten this.

Reproduce with `scripts/china/a2a_bench.py`. An earlier version of that
benchmark used idle GPUs and uniform splits. It reported a 1.89 times benefit
from extra NCCL channels where 1.06 times was delivered. Any transport
benchmark must model both contention and skew.
