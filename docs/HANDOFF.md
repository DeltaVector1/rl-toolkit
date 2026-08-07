# GLM-5.2 DECENSOR RL TRAINING. HANDOFF NOTE

This note gives the reasons. `docs/RUNBOOK.md` gives the commands.
`docs/PERFORMANCE.md` gives the measurements. `.agents/skills/glm52-ops/`
holds the tools and the fault table.

Read the mistakes section before you change anything. Each item is a real
failure with a real cost.

Language: ASD-STE100 Simplified Technical English. One instruction for each
sentence. Active voice.

# WHAT THE SYSTEM IS

The system trains from the GLM-5.2-FP8 checkpoint, a 745 billion parameter
MoE, with GRPO and a rank-32 LoRA adapter. The H200 engines use W8A16 and FP8
KV cache. The trainer dequantizes the checkpoint to BF16 weights. The goal is to reduce
refusals.

The trainer computes gradients on 8 nodes of 8 RTX 4090: n76, n77, n78, n80,
n82, n83, n84, n85.

The inference engines make the rollouts on h200a and h200b, 8 H200 each.

The judge scores the rollouts on one external host of 8 RTX 5090, at
202.214.223.66 port 26406.

The orchestrator joins them and runs on n82.

# THE NETWORK, WHICH EXPLAINS ALMOST EVERYTHING

```
H200 (docker) --internet, measured 3-33% loss, ~217 ms--> jump --LAN--> n82
```

n82 has a private address, 10.92.20.82. The engines reach it with `ProxyJump`
through the jump host, so every tunnel is SSH inside SSH and the lossy leg
carries two layers of channel flow control.

The jump host is on 10.92.20.70, the same LAN as n82.

Only SSH on port 31331 crosses the lossy leg. UDP and every other TCP port
time out. This is measured, not assumed, and it is why no FEC tunnel is
possible today.

# THE RULES THAT THE OWNER GAVE

Do not touch n79.

Do not delete the data of the owner. The protected paths are `/data/ZiphuAI`,
`/data/trinity`, and `/data/nv-community_GLM-5.2-NVFP4`. The model at
`/data/glm52-rl/model` on n85 is the only copy.

Do not use QLoRA. Do not use INT4 weights.

Do not use cached batches. Every attempt is a full production run.

Do not use metaphors. Do not give opinions that the owner did not ask for.

Separate the trainer time and the rollout time. Do not give one combined
number.

Do not use automated monitor tools. Use a shell command with a sleep, which
prints the GPU data, the step data, and the rollout data together. Nothing may
sit still for more than 10 minutes.

Do not run load tests against the engines. The owner refused one on
2026-08-07. Read the vLLM metrics instead: `num_requests_running` and
`num_requests_waiting` answer the same question without adding load.

# THE STATE AT HANDOFF

There is no active run. x25 and x26 are stopped. Both H200 engines are healthy
and idle.

x26 used the edge filesystem transport with the settings below.

| Setting | Value |
|---|---|
| `batch_size` | 1024 |
| `group_size` | 16 |
| `max_inflight_rollouts` | 128 |
| orchestrator | h200a |
| trainer transport | filesystem plus verified LAN fanout |
| weight discovery | reverse `STABLE` marker pull |
| `enable_router_replay` | false |
| `enable_return_routed_experts` | false |

Rollout step 1 took 20 minutes 5 seconds. Step 2 took 18 minutes 41 seconds.
The batch transfers took 28.8 and 29.4 seconds. The trainer failed twice in
the first forward pass. n80 rank 35 was the first process failure. Do not state
a more specific cause without new evidence.

Do not use `run_supervisor.sh` for x26. It assumes that the orchestrator runs
on n82. Start the edge orchestrator before the trainer and verify that
`control/orch.toml` reached n82.

# THE ROUTER-REPLAY TEST

`enable_router_replay = false` is under test. x25 step 1 is now available.

Router replay makes the trainer replay the engine's expert choices so both
forward passes agree. It is also what asks for `routed_experts`, which was
95.5% of every byte crossing the link.

Step 1 measured masked mismatch KL mean 0.362 and max 6.252. Unmasked mean
was 0.0389, entropy 0.4948, and gradient norm 0.0161. The replay-on baseline
was 0.19 to 0.21 masked and 0.012 unmasked. This is not the several-nat
catastrophic failure, but it is not parity and its tail is large.

If later steps rise to several nats, revert both flags together:
`enable_router_replay = true` in `configs/china/trainer.x25.toml` and
`enable_return_routed_experts = true` in `configs/h200/inference.toml`, then
restart the trainer and both engines.

For scale, the FP8 trainer experiment pushed the same number to 7 to 9 nats,
entropy to 9.2, and the reward collapsed.

The router is not a LoRA target. `q_a_proj`, `q_b_proj`,
`kv_a_proj_with_mqa` and `o_proj` are. So the router weights are frozen and
identical on both sides, and the only divergence is numeric: bf16 with a
float32 router here against W8A16 on the engine.

# WHAT WAS FOUND ON 2026-08-07

On x24, trainer compute took 6 to 15 minutes and rollout collection set the
cycle time. On x25 step 1, trainer forward/backward took 17.2 minutes and
collection took 48.7 minutes. Keep both measurements separate.

Every dropped rollout was a transport failure. Step 6 lost 78.4 percent of its
rollouts and every signature was `ReadTimeout`. The `zero_advantage` filter
dropped 0.5 percent. Rollout speed and rollout loss were the same problem.

95.5 percent of the bytes on the link were `routed_experts`, an expert routing
table for every token: 711 bytes for each generated token, 2.79 MB for each
rollout. The tokens and their logprobs were under 5 percent. Measured after the
fix on h200a: the same 64-token response fell from 71064 to 10372 bytes.

The fixes, with their measured effect, are in `docs/PERFORMANCE.md`.

# THE MISTAKES THAT THE ASSISTANT MADE

# The assistant destroyed source files

The assistant copied `sparse_mla_attention.py` and `modeling_glm_moe_dsa.py`
out of the Docker image into the repository, losing `_FiniteGrad` and
`PRESCALE_QK`.

Read `scripts/china/cluster.sh` before you edit these paths, or call
`glm52_ops.mounted_files()`. Never copy a mounted path out of the image. When
`config.py` was extracted on 2026-08-07 the repository was checked first and no
local copy existed.

# The assistant lost ten steps of training

The assistant restarted the trainer at step 10 to change a flag.
`ckpt.interval` was 25, so no checkpoint had ever been written, and
`weights_only = true` cannot resume training even when one exists. The run
restarted at step 1.

Check that a checkpoint exists before you restart the trainer. x25 writes one
every 5 steps and they are resume-capable. The step-10 adapter still exists at
`runs/x24/run_x24/broadcasts/step_10/adapter_model.safetensors`.

# The assistant left supervisors from a dead run

Three `run_supervisor.sh` processes carrying `RUN=x24` were found running
against x25. The supervisor watches the container by name, so it stays quiet
until the trainer dies and then relaunches it under the OLD run name.

`glm52_ops.start()` now kills them first. Check the environment of the running
supervisor, not just its name:
`tr '\0' '\n' < /proc/<pid>/environ | grep RUN=`.

# The assistant used a liveness test that could not fail

`pgrep -fc` counts its own grep. The replacement was also wrong: `pgrep -f
"bash .*name.sh"` over SSH matches the remote shell, because that shell really
is bash and `.*` joins it to the name in the same command line. Bracketing the
first word does not help.

Bracket the LAST token: `pgrep -f "bash .*[t]unnel_watchdog.sh"`. Then read the
log timestamp as well.

Never put a kill and a start for the same process in one SSH command. That
killed all three watchdogs on 2026-08-07.

# The assistant trusted windowed measurements

Two 9-minute windows at the same setting gave 23.18 and 8.10 rollouts a minute.
A window lands at a random point in a 45 to 85 minute batch cycle, and the rate
collapses at the tail when the orchestrator stops opening work. Token
throughput has the same fault: 1369 then 883 tok/s at one setting.

Use the orchestrator's own per-batch cycle time. It averages the whole cycle
and it is the number that sets the step time.

# The assistant wrote a test suite that mirrored the code

`watchdog_smoketest.sh` re-implemented each watchdog's decision and tested the
copy. It passed 18 of 18 while three cases asserted the opposite of the shipped
behaviour. It now drives the real scripts against fake `curl`, `ss`, `kill`,
`sudo`, `docker` and `sleep`. 30 tests.

# The assistant used a watchdog rule that hid a fault

The rule was to stop rebuilding when 80 percent of tunnels failed. At batch
1024 all 24 wedged while the engines were healthy, and the run sat idle 5
hours. Always rebuild, but rate limited: 6 misses, at most 2 kills a pass, 120
seconds apart.

# The assistant tested the engine with the wrong probe

`/v1/models` answers while the EngineCore is dead. Test with a real completion
of 3 tokens.

# The assistant reported noise as a trend

A 25-step window showed "+26 percent"; at 36 steps it was 6.3 percent against a
standard deviation of 0.039. `glm52_ops.reward()` now returns a verdict and
withholds it below 10 steps.

# THE FAULTS THAT WERE FOUND IN THE SOFTWARE

Activation offloading defaults to on and fills pinned host memory. Set
`ac_offloading = "None"`. This stopped every 16K run for a week.

The adapter URL is resolved on the ENGINE hosts. Port 18100 now forwards to
n82 port 8100 everywhere. Before this fix no run could learn, so all reward
data before x22 is invalid.

FP8 trainer weights break the importance ratio. The engines use W8A16 and the
trainer FP8 path uses W8A8. Set `dequantize_checkpoint = true`.

The pipeline stage buffer took its dtype from the first parameter. Use
`DTYPE_MAP[config.model.optimization_dtype]`.

`apply_pp_boundaries` makes a ModuleDict, which broke `_context_parallel_state`,
slicing, `enumerate()` indexing of `routed_experts`, and `self.norm` on
non-final stages.

A corrupt micro-batch with shape `[1, 9142729177965406720]` stopped rank 0.
`train.py` now discards a bad batch.

`build_async_openai` in verifiers left the OpenAI SDK defaults, `connect=5.0s`
and `read=600s`. A wedged request held an in-flight slot for ten minutes. Now
300s and 30s, overridable by `PRIME_ROLLOUT_TIMEOUT_S` and
`PRIME_ROLLOUT_CONNECT_TIMEOUT_S`.

# WHAT TO DO NEXT

First, read the mismatch KL of x25 step 1 and decide the router replay
question. Everything else waits on that.

Then, in the order they are worth doing:

Move the orchestrator to the engine side and use a durable filesystem batch.
`docs/ROLLOUT_TRANSPORT.md` gives the design and the repository includes a
resumable, hash-checked publisher. Do not use WAN ZMQ: it has no application
acknowledgment, replay, or durable spool.

Stream the generations. A request that dies at 290 seconds today loses the
whole generation after the engine already did the work. vLLM streams natively.

Remove the SSH inside SSH. Terminate the reverse tunnel at the jump host and
let n82 pull over the LAN. No new ports and no sudo are needed.

Drain a tunnel before the watchdog rebuilds it, so a repair stops destroying
the requests in flight on it.

Lower the read timeout to about 150 seconds. Generation p90 is 87.8 seconds.
Consider hedged requests above p95.

Encode the logprobs in binary. They are now the largest field.

Ask the provider for one inbound UDP port. That unlocks kcptun or frp over KCP
with FEC, which is the only option that fixes the 7.5 percent loss instead of
working around it.

# WHAT IS ALREADY RULED OUT

More tunnels past 48 gave nothing at 128 in flight: 14.20 against 13.95 a
minute.

BBR on the sending side cannot be loaded. The engine hosts are Docker
containers with no `CAP_SYS_MODULE` and the host has not loaded `tcp_bbr`.

The judge is not a bottleneck. 24 concurrent requests returned in 2.85 seconds
of wall time.

The engines are not saturated. They run 26 to 46 requests against a
`max_num_seqs` of 128 with `num_requests_waiting` at zero.

Do not set BBR globally on n82. It is a trainer node and NCCL crosses the same
interface, where cubic measured 87 percent against BBR's 36 percent.
`scripts/network/tune_wan.sh` sets it per route instead.
