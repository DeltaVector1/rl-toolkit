# GLM-5.2 cluster guide

## Connections

Connect to the China jump host first, then use the node aliases:

```bash
ssh jump-node
ssh n82
```

The same connection can be made in one command:

```bash
ssh -t jump-node ssh n82
```

Replace `n82` with any China node listed below.

| Node | Role |
|---|---|
| n76 | 8x RTX 4090 trainer worker |
| n77 | 8x RTX 4090 trainer worker |
| n78 | 8x RTX 4090 trainer worker |
| n79 | Unused; do not start or stop anything here |
| n80 | 8x RTX 4090 trainer worker |
| n81 | 8x RTX 4090 trainer worker |
| n82 | 8x RTX 4090 trainer worker, orchestrator, adapter server |
| n83 | 8x RTX 4090 trainer worker |
| n84 | 8x RTX 4090 trainer worker |
| n85 | Four Gemma user/judge servers and their proxy |

The two H200 rollout hosts are reached directly:

```bash
ssh -i ~/.ssh/h200_rollout_ed25519 root@152.236.142.239 -p 40050
ssh -i ~/.ssh/h200_rollout_ed25519 root@152.236.142.243 -p 40051
```

Do not copy weights or container layers through the jump host. The China nodes share `/data`; each H200 host has its own `/data`.

## Live run

```text
Deployment: /data/glm52-rl/deployment
Run:        /data/glm52-rl/runs/r35/run_decensor_fp8_r35
```

Start it with the launcher variables rather than editing the committed r33
configs:

```bash
RUN=r35 RUN_NAME=run_decensor_fp8_r35 CONFIG_TAG=.r35 \
  PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE=1 \
  bash /data/glm52-rl/deployment/scripts/china/cluster.sh trainer
```

Current rollout settings:

```text
Batch size:            1024 rollouts
GRPO group size:          8
Unique prompts/step:    128
Maximum live rollouts: 1024
Fixed evaluation:        32 rollouts every 10 steps
```

`max_inflight_rollouts` tracks `batch_size`. Leaving it at 88 while the batch is
1,024 would feed the two H200 engines in twelve sequential waves instead of
keeping them saturated; at 1,024 they run about 220 concurrent requests and
roughly 2,000 generated tokens per second, and a full batch takes about 20-28
minutes.

Measured at batch 1,024: 32 micro batches per optimizer step at about 126
seconds each, so roughly 67 minutes of trainer per step against about 28 minutes
of rollout collection, which overlaps. The trainer sets the pace.

Containers:

| Host | Containers |
|---|---|
| n76, n77, n78, n80, n81, n82, n83, n84 | `glm52-decensor-trainer` |
| n82 | `glm52-decensor-orchestrator`, `glm52-adapter-http` |
| n85 | `glm52-gemma-0` through `glm52-gemma-3`, `glm52-gemma-proxy` |
| Each H200 host | `glm52-decensor-h200` |

Useful checks:

```bash
sudo docker ps
sudo docker logs --tail 100 glm52-decensor-trainer
sudo docker logs --tail 100 glm52-decensor-orchestrator
nvidia-smi
```

### Rollout tunnels

The orchestrator reaches the two H200 engines through twenty-four reverse SSH
tunnels on n82, ports 18001-18024. Odd ports come from `152.236.142.243`, even
ports from `152.236.142.239`, twelve per host. Keep that odd/even split: the
link watcher derives the owning host from the port's parity.

It was eight tunnels originally. Widening to twenty-four raised rollout
collection from about 4 completions per minute to 13-16, roughly fourfold,
because aggregate throughput over a lossy path scales with the number of TCP
flows. Individual tunnels flap constantly on this link - six or so of the
twenty-four are typically down at any moment - but with twenty-four the
remaining eighteen carry the load, whereas with eight the same failure rate left
almost nothing. Start the extra ones with:

```bash
# on 152.236.142.243
bash /data/glm52-rl/deployment/scripts/h200/start_tunnels.sh 18009 18011 18013 18015 18017 18019 18021 18023
# on 152.236.142.239
bash /data/glm52-rl/deployment/scripts/h200/start_tunnels.sh 18010 18012 18014 18016 18018 18020 18022 18024
```

and list all of them in `orchestrator.*.toml` under `model.client.base_url`.

That path is a long-haul public link, not a datacentre network. Measured from
n82: about 216 ms round trip to both hosts, with packet loss that has been seen
between 3% and 20% and differing between the two hosts. At 216 ms round trip,
10% loss cuts a single cubic TCP flow to a small fraction of the link, so
rollout collection slows sharply and the orchestrator logs large numbers of
"Retrying transient rollout failure" while the engines themselves report 200 OK
and full generation throughput. Requests are completing and the tunnel is
dropping them.

Check the link before blaming the trainer or the tunnels:

```bash
ping -c 40 -i 0.25 -q 152.236.142.239
ping -c 40 -i 0.25 -q 152.236.142.243
```

**Keepalive tolerance is what actually kept the tunnels alive.** `start_tunnel.sh`
originally used `ServerAliveInterval=10` with `ServerAliveCountMax=3`, tearing
the SSH connection down after thirty seconds of unanswered keepalives. That is a
datacentre setting. On a link losing 13-17% of packets, three consecutive missed
keepalives happens routinely, so tunnels were being killed constantly while the
link was perfectly usable. Raising it to `ServerAliveInterval=15` with
`ServerAliveCountMax=20` - five minutes of tolerance - took healthy tunnels from
10 of 24 to 23 of 24.

That churn was also one way to wedge the orchestrator, but not the only one, and
the underlying fault was in the budget calculation rather than the network.

`Orchestrator.update_rollout_budget` sized fresh work as

```text
batch_size - accepted - buffered - queued - inflight - pending
```

Rollouts sitting in a *partial* group count in `buffered`. A group goes partial
whenever some of its eight members are dropped - the `zero_advantage` filter is
enforced, and steps routinely report only about 96% of rollouts trainable - or
lost to a dying tunnel. Once nothing is in flight, those buffered rollouts can
never be completed, yet they still subtract from the budget, so the budget sits
at zero, no fresh groups open, and the batch wedges forever. It shows up as
`Train batch 1000/1024 ... 0 inflight rollouts` with the trainer idle
indefinitely, and it recurs even with zero retries and healthy tunnels.

The fix, in `source/src/prime_rl/orchestrator/orchestrator.py`: when the
pipeline has fully drained (`queued + inflight + pending == 0`) and the batch is
still short, ignore the stuck buffer and open `batch_size - accepted` fresh
rollouts, logging `opening fresh groups to finish it`. It is guarded on
`accepted > 0` so the normal post-ship path, where the buffer legitimately
carries into the next batch, is untouched. If that warning appears regularly,
the run is healthy and self-healing; before the fix the same condition required
a manual orchestrator restart and lost the whole partial batch.

Loss here is the single largest factor in rollout wall time and none of the
trainer settings affect it. BBR would suit this path far better than cubic, but
the H200 containers ship no `tcp_bbr` module and their `net.ipv4` sysctls are
read only, so congestion control cannot be changed on the sending side. The
practical lever is more parallel tunnels: aggregate throughput over a lossy path
scales with the number of TCP flows, so widening past the current eight is the
way to buy back throughput when loss is high.

A tunnel can half-die: the socket keeps listening on n82, so `ss` still shows
it and clients still connect, but nothing reaches the H200 and every request
hangs until it times out. That looks like rollouts stuck "inflight" while both
H200 engines and the Gemma servers report zero running requests. Check with:

```bash
for p in 18001 18002 18003 18004 18005 18006 18007 18008; do
  printf "%s:" $p
  curl -s -m 8 -o /dev/null -w "%{http_code} %{time_total}s\n" http://127.0.0.1:$p/v1/models
done
```

Anything other than `200` in well under a second is broken. Distinguish the two
failures by how they fail: a *fast* refusal means `start_tunnel.sh` is already
reconnecting and will recover on its own; a probe that *hangs* for the full
timeout is the half-dead case, which never recovers by itself.

`scripts/network/watch_rollout_links.sh` exists to repair this automatically and
**must be running on n82**. It polls the health port forwarded beside each
tunnel (19001-19008) and kills the stale `sshd` holding that port so the tunnel
loop reconnects. It is not started by `cluster.sh`, so it is easy to leave off -
and with it off, half-dead tunnels persist and rollouts hang on them until the
3,600 second rollout timeout.

Its thresholds matter as much as running it at all. At 1,024 live rollouts the
health probe shares the tunnel with rollout traffic, so a busy link can miss
probes while it is perfectly healthy - and those misses arrive on every link at
once. The original six-failure threshold with no rate limit tore down seven of
the eight tunnels in the same second, destroying every in-flight rollout and
cutting group completion roughly threefold. The script now requires thirty
consecutive failures (about five minutes), reconnects at most one link per pass,
and waits two minutes between reconnects. Do not tighten these while running
large batches. A genuinely dead tunnel never recovers on its own, so a
five-minute detection window costs nothing.

Check and start it:

```bash
pgrep -af watch_rollout_links.sh
sudo sh -c "setsid nohup bash /data/glm52-rl/deployment/scripts/network/watch_rollout_links.sh \
  >/var/log/glm52-rollout-links.log 2>&1 </dev/null &"
```

The health ports are the precise signal; check them directly with
`curl -fsS --max-time 8 http://127.0.0.1:1900N`.

To restart one port by hand from the owning H200 host:

```bash
bash /data/glm52-rl/deployment/scripts/h200/start_tunnels.sh 18003
```

## Measured timing

Measurements below come from r33 with 88 rollouts, two 8x H200 rollout hosts, and 64 RTX 4090 trainer GPUs.

| Work | Measured time |
|---|---:|
| Gemma scoring per trace | 0.89 seconds average, 1.30 seconds p90 |
| H200 model generation per trace | 54.6 seconds average, 47.2 seconds median, 87.8 seconds p90 |
| H200 generation plus environment for one 88-rollout batch | About 3-8 minutes |
| Adapter weight transfer | About 0.7 seconds |
| One trainer microbatch, before tuning | 235.8 seconds |
| One trainer microbatch, after tuning | 105-129 seconds |
| Trainer forward plus backward for 88 traces, before tuning | 707.4 seconds |
| Trainer forward plus backward for 88 traces, after tuning | 389.6 seconds (includes about 80 seconds of one-time TileLang compilation) |
| Trainer step for 88 traces, before tuning | 7m54s-19m05s; recent average about 13m30s |

The trainer is the limiting stage. Gemma is not the bottleneck. The trainer uses global expert parallelism across 64 GPUs, CP4 ring attention, Torch collectives, and Ethernet because InfiniBand and GPU peer-to-peer communication are disabled. It reports only about 270-318 tokens/second across the cluster and 0.2-0.3% MFU.

## Where trainer time actually goes

Each node has a single 10 Gb/s link (`vlan130`, MTU 1500) shared by its eight
GPUs. The second port on the card is not cabled and only negotiates 1 Gb/s, so
10 Gb/s per node is a hard ceiling.

Expert weights force that link to carry everything. The 75 MoE layers hold 725
billion routed-expert parameters, which is 11.3 GB per GPU only when experts are
sharded across all 64 GPUs. Node-local expert parallelism would need 90.6 GB per
GPU, so `global_ep` is required and every MoE dispatch and combine crosses the
network.

Measured with per-layer CUDA-event timers (`PRIME_EP_PROFILE=1`), one micro
batch of 4,096 tokens per data rank breaks down as:

| Phase | Time | Passes per layer |
|---|---:|---|
| Forward dispatch all-to-all | 46.2 s | 2 (original plus recompute) |
| Forward combine all-to-all | 45.3 s | 2 (original plus recompute) |
| Backward dispatch all-to-all | 36.6 s | 1 |
| Backward combine all-to-all | 35.8 s | 1 |
| Expert count metadata all-to-all | 0.4 s | 2 |
| Everything else (attention, expert GEMM, router, output head) | 12.9 s | |

All-to-all is 93% of the micro batch. Attention and expert kernels together are
7%, so tuning those kernels cannot move the step time; only reducing bytes on
the wire or the number of transfers can.

## Trainer tuning applied

**Socket count.** NCCL was opening `NCCL_SOCKET_NTHREADS=4` x
`NCCL_NSOCKS_PERTHREAD=4` = 16 sockets per rank, so 128 TCP flows shared one
10 Gb/s NIC per node and collapsed under self-congestion. Measured on the real
64-rank expert-parallel group with the payload sizes the MoE actually sends:

| Sockets per rank | INT4 dispatch | FP8 gradient | BF16 |
|---|---:|---:|---:|
| 16 (4 x 4, previous) | 0.506 s | 0.964 s | 1.551 s |
| 8 (2 x 2) | 0.437 s | 0.628 s | 1.236 s |
| 2 (1 x 2) | 0.329 s | 0.650 s | 1.101 s |
| **1 (1 x 1, current)** | **0.285 s** | **0.558 s** | **0.885 s** |

Jumbo frames were tested end to end: MTU 9000 passes the switch with no loss,
but it made the all-to-all *slower* (0.452 s versus 0.285 s for INT4 dispatch),
so the cluster stays at MTU 1500.

**Channel count.** This is the largest single transport knob. The all-to-all is
limited by how many concurrent TCP flows share one NIC, and NCCL's automatic
choice for this topology leaves most of the link idle. Measured per MoE layer
(two INT4 plus two FP8 all-to-alls, 846 ms of wire time at line rate):

| `NCCL_*_NCHANNELS` | Per layer | Wire efficiency |
|---|---:|---:|
| NCCL default | 1835 ms | 44% |
| 2 | 1888 ms | 45% |
| 4 | 1527 ms | 55% |
| **8** | **970 ms** | **87%** |
| 16 | 972 ms | 87% |

One channel hangs; do not set it. Eight is the plateau, and 87% matches the
measured ceiling of this NIC, so there is very little left on the wire.

That ceiling was established by holding the off-node bytes per node fixed at
176 MB and varying only how many ranks per node participate:

| Ranks/node | Flows/node | Achieved | Efficiency |
|---|---:|---:|---:|
| 8 | 448 | 4.13 Gb/s | 41% |
| 4 | 112 | 5.11 Gb/s | 51% |
| 2 | 28 | 7.99 Gb/s | 80% |
| 1 | 7 | 8.84 Gb/s | 88% |

Fewer, larger transfers use the link far better. Eight NCCL channels recover
almost all of that without changing the parallelism, so the hierarchical
node-aggregated all-to-all this table would otherwise justify is unnecessary.

**Queueing discipline.** `fq` instead of the default `fq_codel` on the trainer
NIC is worth another large step once channels are right: 65% against 87%.
`cluster.sh` applies it before starting the trainer, so it survives a reboot.
BBR congestion control was tested and is clearly worse here (36%); the cluster
stays on `cubic`. `net.ipv4.tcp_slow_start_after_idle` makes no measurable
difference and is left at the default.

**Activation-checkpoint recompute.** Selective checkpointing targets
`routed_experts`, which wraps `MoE._run_routed_experts`. That method calls
`self.experts(...)`, and the dispatch and combine all-to-alls are pre- and
post-forward hooks on that module, so the backward recompute repeated both
transfers. `PRIME_EP_AC_CACHE=1` keeps the INT4 payload each rank *received*
during the forward pass and reuses it on the recompute. This is numerically
exact - the recompute would dequantize the same bytes - and the cached payload
is a quarter the size of the dequantized BF16 activation, costing about 3.8 GB
per GPU. Verified bitwise identical (outputs and gradients) against the
uncached path.

The guard here is easy to get wrong. `torch.autograd.Function.forward` always
runs with gradients disabled, so a `torch.is_grad_enabled()` test *inside*
`_INT4Dispatch.forward` never passes and nothing is ever cached - the run looks
correct and stays slow. The decision is made in `_token_dispatch` and
`_token_combine` instead, which pass `cache=None` when no graph is being built.
`scripts/china/check_ep_ac_cache.py` counts real `all_to_all_single` calls so
this failure mode cannot pass silently.

Result on the 88-rollout batch, measured per micro batch:

| Configuration | Micro batch | All-to-all | Throughput |
|---|---:|---:|---:|
| Before tuning | 235.8 s | 164.3 s | 277 tok/s |
| Sockets only | 177.2 s | 164.2 s | 351 tok/s |
| Sockets plus recompute cache | 128.6 s (steady state 105-115 s) | 110.7 s | 445 tok/s |

Peak memory rises from 37.1 GiB to 41.2 GiB of the 48 GiB card.

## Why the all-to-all still costs more than the wire

With the settings above, a synthetic 64-rank all-to-all using *uniform* splits
reaches 90% of line rate: 942 ms per layer against an 846 ms floor, or about 71
seconds of all-to-all per micro batch. The real trainer measures 103-108
seconds. The difference is not transport and not GPU contention - keeping the
SMs busy with GEMMs during the benchmark changes nothing (936 ms against 974 ms
idle).

It is expert load imbalance. `all_to_all_single` finishes at the pace of its
busiest peer, so a few hot experts stretch the whole collective:

| Split skew | Per layer | Wire efficiency | All-to-all per micro batch |
|---|---:|---:|---:|
| 1 (uniform) | 942 ms | 90% | 71 s |
| 2 | 1827 ms | 46% | 137 s |
| 4 | 3567 ms | 24% | 267 s |

The measured 103-108 seconds sits between skew 1 and skew 2, so live routing
carries roughly a 1.5x imbalance. Because `enable_router_replay` is on, the
trainer replays the inference engine's routing rather than choosing its own, so
this cannot be flattened from the trainer side, and flattening it at all would
change routing and therefore model behaviour. Treat the current all-to-all as
being at its practical floor and do not spend more time on NCCL or qdisc knobs.

Reproduce with `scripts/china/a2a_bench.py` (see the bench notes); the earlier
version of that benchmark ran on idle GPUs with uniform splits and overstated
the benefit of extra NCCL channels by 1.89x against 1.06x actually delivered.
Any future transport benchmark must model both contention and skew or it will
mislead in the same way.

## Remaining lever

After both changes the backward all-to-all is the largest single cost, about
70 seconds of a roughly 128-second micro batch. It sends FP8 at 6,144 bytes per
row while the forward sends INT4 at 3,072 bytes per row. Quantizing the
gradient to INT4 as well would cut roughly 25 seconds per micro batch, about
another 1.25x. That changes gradient precision rather than only scheduling, so
it is not applied here; it needs a side-by-side reward and gradient-norm
comparison before it goes into a real run.
