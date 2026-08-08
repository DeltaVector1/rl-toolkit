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
