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

This script runs on n82. It tests each of the 24 tunnel ports. It kills a
tunnel that does not answer two times. The supervisor on the H200 host then
rebuilds it.

An early version refused to rebuild when 80 percent of the tunnels failed.
That version stopped the run for 5 hours. The script now always rebuilds.

# scripts/ops/watchdog_smoketest.sh

This script tests the logic of the watchdogs. It uses false input data. It
does not touch the system.

The script has 18 tests. The tests include the two real failures.

# scripts/ops/notify.sh

This script sends a Telegram message for each step. It sends a message and a
telephone call when the run stops.

The script runs on h200a. The China nodes cannot reach Telegram.

A stopped run needs two conditions. The trainer step count must not change.
The orchestrator batch count must not change. One slow batch is normal. A slow
batch must not send an alarm.

# THE CHANGES TO THE SCRIPTS

# scripts/h200/start_tunnel.sh

`ServerAliveInterval` changed from 15 to 10. `ServerAliveCountMax` changed
from 20 to 3.

A dead tunnel is now removed in 30 seconds. The old value was 300 seconds. The
tunnels collected faster than SSH removed them.

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

# THE CONFIGURATION VALUES THAT MATTER

`ac_offloading = "None"`. Do not remove this line. The default is not `None`.

`dequantize_checkpoint = true`. The FP8 path uses W8A8. The engines use W8A16.
The difference breaks the importance ratio.

`max_inflight_rollouts = 256`. This value is not the batch size. The tunnels
fail above this value.

`batch_size = 1024` and `group_size = 16`. These values give 64 GRPO groups.

The orchestrator must run on n82. The batch cannot cross the link.

# THE MEASUREMENTS

Host memory with activation offloading on: 486 GB of 503 GB. The nodes
stopped.

Host memory with activation offloading off: 3 GB. The available memory was 460
GB.

Mismatch KL with FP8 weights: 7 to 9 nats. The positive advantage was 0.003.

Mismatch KL with bf16 weights: 0.2 nats. The positive advantage was 0.3.

Entropy with FP8 weights: 9.2 nats. The vocabulary limit is 11.95 nats.

Entropy with bf16 weights: 0.45 nats.

Gradient norm with FP8 weights: 200000 to 330000. The clip limit is 1.0.

Gradient norm with bf16 weights: 0.013 to 0.043.

Batch transfer over the tunnel: it did not complete. Over the loopback
interface: 0.45 seconds.

Tunnels healthy at 1024 rollouts in flight: 10 of 24. At 256: 20 of 24.

Engine concurrency at 1024 rollouts in flight: 1 to 6. At 256: 45 to 70.
