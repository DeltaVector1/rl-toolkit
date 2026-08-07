# GLM-5.2 DECENSOR RL TRAINING. HANDOFF NOTE

This note gives the reasons. `docs/RUNBOOK.md` gives the commands.
`docs/PERFORMANCE.md` gives the measurements.
`.agents/skills/glm52-ops/` holds the tools and the fault table.

Read the mistakes section before you change anything. Each item is a real
failure with a real cost.

Language: ASD-STE100 Simplified Technical English. One instruction for each
sentence. Active voice.

# WHAT THE SYSTEM IS

The system trains GLM-5.2-FP8, a 745 billion parameter MoE, with GRPO
reinforcement learning and a rank-32 LoRA adapter.

The trainer computes gradients on 8 nodes of 8 RTX 4090. The nodes are n76,
n77, n78, n80, n82, n83, n84, n85.

The inference engines make the rollouts on h200a and h200b, 8 H200 each.

The judge scores the rollouts on one external host of 8 RTX 5090, at
202.214.223.66 port 26406.

The orchestrator joins them and runs on n82.

The trainer nodes are in China. The engines and the judge are outside China.
The link drops 10 to 33 percent of the packets and adds 217 milliseconds. The
engines reach n82 through 24 reverse SSH tunnels on ports 18001 to 18024.

# THE RULES THAT THE OWNER GAVE

Do not touch n79.

Do not delete the data of the owner. The protected paths are `/data/ZiphuAI`,
`/data/trinity`, and `/data/nv-community_GLM-5.2-NVFP4`. The model at
`/data/glm52-rl/model` on n85 is the only copy.

Do not use QLoRA. Do not use int4.

Do not use cached batches. Every attempt is a full production run.

Do not use metaphors. Write plain technical text.

Do not give opinions that the owner did not ask for.

Separate the trainer time and the rollout time. Do not give one combined
number.

Do not use automated monitor tools. Use a shell command with a sleep, which
prints the GPU data, the step data, and the rollout data together. Nothing may
sit still for more than 10 minutes.

# THE ORDER IN WHICH THE PARTS START

This was written three different ways in three documents. All three were
partly right. The correct rule has two cases.

The trainer blocks at startup until `control/orch.toml` exists. Only the
orchestrator writes that file, at `orchestrator.py` line 214, early in
`setup()`.

For a NEW run name that file does not exist, so the orchestrator starts first.

For an EXISTING run name that file is already on disk, so the trainer starts
first. This matters because a zmq send to a dead trainer blocks the
orchestrator event loop. `run_supervisor.sh` handles only this case, which is
why it restarts the trainer first.

Never make the orchestrator wait for the trainer to bind port 29620. The
trainer waits for a file that only the orchestrator writes, so the two waits
form a deadlock.

`glm52_ops.start(run, mode=...)` puts both cases in code.

# THE MISTAKES THAT THE ASSISTANT MADE

# The assistant destroyed source files

The assistant copied `sparse_mla_attention.py` and `modeling_glm_moe_dsa.py`
out of the Docker image into the repository. The repository copies held local
changes. The image copies did not. The local changes were lost, including
`_FiniteGrad`, which removes NaN values from the gradients, and `PRESCALE_QK`.

Three of that day's faults came from writing again what the lost snapshot
already had.

Read `scripts/china/cluster.sh` before you edit these paths. It lists the files
that are mounted into the container. Those files hold local changes. Never copy
those paths out of the image. Edit the repository copy.
`glm52_ops.mounted_files()` prints the list.

# The assistant gave a wrong diagnosis for many hours

The assistant said the host memory was not real and called it phantom memory.
The OOM trace showed `shmem_fault`, `shmem_alloc_folio`, `__alloc_pages`, and
`out_of_memory`. That is real page allocation. `PHANTOM_RAM_PROOF.md` is
retracted.

Measure before you write a conclusion. Do not present a theory as a fact.

# The assistant used a liveness test that could not fail

The assistant used `pgrep -fc <name>` to test whether a process runs. That
command counts its own grep, so the result is never zero and the test always
passes. The tunnel watchdog was dead for 3 hours behind a green check.

The replacement was also wrong at first. `pgrep -f "bash .*name.sh"` sent over
SSH still matches, because the remote shell really is `bash` and the pattern
holds the name, so `.*` joins them inside one command line. Bracketing the
first word does not help for the same reason.

Bracket the LAST token: `pgrep -f "bash .*[t]unnel_watchdog.sh"`. Then read the
log file as well. A process that runs but writes no new log line does not work.
`glm52_ops.alive()` does both. `scripts/ops/watchdog_smoketest.sh` tests all
three forms so the fault cannot come back.

# The assistant killed the watchdogs with a restart command

On 2026-08-07 the assistant put a `pkill -f` and the restart into one SSH
command. The command line held the plain script name, so `pkill` matched its
own shell and the session died with status 255 before the restart ran. The
assistant repeated it once with a bracketed pattern, which still failed,
because the same command line also held the plain name in the start half.

All three watchdogs were then dead and the run was unprotected until the
assistant checked.

Put the kill and the start in separate connections. Verify with a third.

# The assistant wrote a test suite that mirrored the code

`scripts/ops/watchdog_smoketest.sh` re-implemented each watchdog's decision as
a local shell function and tested that copy. It reported 18 of 18 passed while
three of its cases asserted the opposite of the shipped behaviour: they still
expected the tunnel watchdog to refuse to rebuild on mass failure, a rule that
was removed after it froze the run for 5 hours.

A test that mirrors the logic cannot detect a change to the logic.

The suite now runs the real scripts against fake `curl`, `ss`, `kill`, `sudo`,
`docker` and `sleep`, and asserts on what they do. It was checked by putting
the old rule back, which turned 4 cases red.

# The assistant used a watchdog rule that hid a fault

The rule was: if 80 percent of the tunnels fail, do not rebuild them. The
reason was to expose a dead engine. At batch 1024 all 24 tunnels wedged at once
while the engines were healthy. The watchdog did nothing for 5 hours.

Always rebuild. A rebuild against a dead backend fails again on the next cycle
and costs nothing. A refusal to rebuild against a live backend stops the run
until an operator finds it.

# The assistant tested the engine with the wrong probe

`/v1/models` answers whenever the API server is alive. The EngineCore can be
dead at the same time. On 2026-08-06 h200a returned 200 on `/v1/models`, gave
no answer to a completion, and showed 64 requests with 0 tokens per second. The
container reported "Up" for 2.6 hours.

Neither the container state nor `/v1/models` is a health signal. Test with a
real completion of 3 tokens.

# The assistant broke the run with a tunnel restart

The assistant killed the batch tunnel while the orchestrator was alive. The zmq
session broke and left 7 dead connections on port 29620. The trainer waited 27
minutes.

Stop the orchestrator before you touch the batch tunnel.

# The assistant reported noise as a trend

The assistant reported that the reward increased 26 percent, measured over 25
steps. At 36 steps the increase was 6.3 percent, the standard deviation was
0.039, and the total trend was 0.013. The trend was inside the noise.

Compare the trend against the spread. Do not report a percentage from a half
window. `glm52_ops.reward()` returns a verdict and withholds it below 10 steps.

# THE FAULTS THAT WERE FOUND IN THE SOFTWARE

# Activation offloading was on by default

`ac_offloading` defaults to `ActivationOffloadingConfig()`, not to `None`. See
`configs/trainer.py` line 173. No configuration file set it.

It puts activations into pinned host memory, which then follows the number of
micro-batches in flight rather than the GPU memory. The host memory reached 486
GB of 503 GB and the nodes stopped.

This one fault stopped every 16K run for a week. It also explains two failed
experiments: FP8 weights halved the GPU memory and `expandable_segments` cut
the reserved memory by a third, and neither changed the host memory.

Set `ac_offloading = "None"`.

# The adapter URL must resolve on the engine hosts

The vLLM workers download the LoRA adapter. See
`inference/vllm/worker/filesystem.py`, `stage_lora_adapter`. `source_url` is
resolved on the H200 hosts, not on the orchestrator host.

The adapter server runs on n82 port 8100. The URL used port 18100, where
nothing answered. Every download failed, every failed download killed a worker,
and every dead worker killed the EngineCore.

Port 18100 now forwards to n82 port 8100 on all three hosts, so one URL works
everywhere.

Before this fix no run could learn. The engines never received new weights, so
every rollout came from the base model. All reward data before run x22 is not
valid.

# FP8 trainer weights break the importance ratio

The engines use W8A16. The trainer FP8 path uses W8A8: it also quantizes the
activations, in `fp8_ada.py`, `moe_kernel_quantize_input`. The trainer was
therefore less exact than the engines.

The mismatch KL reached 7 to 9 nats, the trust region masked almost every
positive advantage, the entropy rose to 9.2 nats, and the reward fell.

Set `dequantize_checkpoint = true`. The mismatch KL falls to 0.2.

A better fix is to make the trainer use W8A16. Start at `_dequant_weight`.

# The pipeline stage buffer used the wrong dtype

`train.py` took the dtype of the inter-stage buffers from the first parameter.
Those buffers hold activations. With FP8 weights the first parameter is
`float8_e4m3fn`, so every hidden-state buffer was declared FP8.

Use `DTYPE_MAP[config.model.optimization_dtype]`. Do not scan for the first
real float parameter, because an fp32 `weight_scale_inv` can come first.

# The pipeline code failed with a ModuleDict

`apply_pp_boundaries` replaces the layer ModuleList with an `nn.ModuleDict`
keyed by the global layer index as a string. See `pipeline_parallel.py` line
139. Four faults came from this.

`_context_parallel_state` used `self.layers[0]` and raised `KeyError: 0`.

The forward loop sliced `self.layers`, which a ModuleDict does not support.

The forward loop used `enumerate()`, which gives a local index where
`routed_experts` needs the global one. This fault raised no error. It replayed
the wrong router on every stage after the first.

`self.norm` is `None` on every stage except the last, and the forward called it
without a test.

# A corrupt batch stopped the run

On 2026-08-06 a micro-batch arrived with an `input_ids` shape of
`[1, 9142729177965406720]`, which is near 2^63. Torch raised "Storage size
calculation overflowed", rank 0 stopped, and the other 7 ranks waited for the
NCCL timeout. The run was dead for 6 hours because no supervisor existed.

`train.py` now tests each micro-batch shape against `seq_len * cp * 4` and
discards a bad batch.

# THE LIMITS OF THE SYSTEM

# The tunnels limit the concurrency, not the batch size

At 1024 rollouts in flight all 24 tunnels wedge and the engines fall to 1 to 6
concurrent requests. At 256 in flight, 20 to 22 stay healthy and the engines
run 45 to 70.

Set `batch_size` for the gradient quality. Set `max_inflight_rollouts` to 256.
The two values are independent.

# The batch cannot cross the link

The batch is one large stream and SSH gives each channel a window near 2 MB. On
a 217 millisecond link the transfer stops at that window and never completes.

The orchestrator must run on n82, where the batch goes over the loopback
interface in 0.45 seconds. The cost is that the rollout traffic crosses the bad
link instead. Accept that cost. The trainer cannot use a rollout that does not
arrive.

# The reward is multiplicative

The reward multiplies five factors: the judge score, the coherency, the
reasoning, the markdown, and the word count. One factor near zero makes the
whole reward near zero.

The 10th percentile therefore stays near zero and the headroom is at the
bottom, not the top. Read the 90th percentile for the distribution signal.

Read the `effective` reward, not the `all` reward. A failed rollout scores
zero, so the `all` series moves when the tunnels recover rather than when the
model learns. On x24 the error rate fell 42.5 to 13.7 percent across three
steps, which lifted `all` from 0.0408 to 0.0907 while `effective` moved 0.0693
to 0.1052.

# THE STATE AT HANDOFF

Run x24: batch 1024, group size 16, 256 rollouts in flight, 16K context, bf16
trainer weights, `pp=8`, `cp=8`, `ep=8` node-local.

Three steps are complete. The trainer step time is 83, then 51, then 31
minutes. The rollout collection time is 82, then 51, then 29 minutes. The error
rate is 42.5, then 27.4, then 13.7 percent, and falls because the tunnels
recover.

The effective reward is 0.0693, then 0.0702, then 0.1052. Three points are not
a trend. `glm52_ops.reward()` withholds a verdict below 10 steps.

Run x22 gave 36 steps at batch 256, mean reward 0.1205, standard deviation
0.039. That is the comparison.

# WHAT TO DO NEXT

Collect 10 to 15 clean steps at batch 1024. Compare the spread against 0.039.
That test shows whether the larger batch reduced the noise, which is the reason
the batch size was raised.

Make the trainer use W8A16. That recovers the FP8 memory and keeps the mismatch
KL low.

Add the SSH key to n81. The node is healthy and gives 72 GPUs. A change of the
pipeline layout is necessary.

The MFU is 3.2 percent, so the step time has much space to decrease.
