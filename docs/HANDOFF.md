# GLM-5.2 DECENSOR RL TRAINING. HANDOFF NOTE

This note tells you how to operate the training run. It also tells you which
mistakes were made, so that you do not make them again. Written 2026-08-07.

Language: ASD-STE100 Simplified Technical English. One instruction per
sentence. Active voice. No synonyms for the same thing.

# WHAT THE SYSTEM IS

The system trains GLM-5.2-FP8. The model has 745 billion parameters. The
method is GRPO reinforcement learning with a rank-32 LoRA adapter.

The system has three parts:

The trainer computes gradients. It runs on 8 nodes. Each node has 8 RTX 4090
GPUs. The nodes are n76, n77, n78, n80, n82, n83, n84, n85. The total is 64
GPUs.

The inference engines make the rollouts. They run on 2 hosts. Each host has 8
H200 GPUs. The hosts are h200a and h200b. The total is 16 GPUs.

The judge scores the rollouts. It runs on one external host with 8 RTX 5090
GPUs. It answers at 202.214.223.66 port 26406.

The orchestrator controls the run. It runs on n82.

# WHAT THE NETWORK IS

The trainer nodes sit in China. The engines and the judge sit outside China.
The link between them drops 10 to 33 percent of the packets. It adds 217
milliseconds of round trip time.

The engines connect to n82 with 24 SSH reverse tunnels. The ports are 18001 to
18024.

# HOW TO START THE RUN

Do these steps in this sequence. Do not change the sequence.

1. Make sure that all tunnels are healthy. Count the ports that answer.
2. Make sure that both engines make a real completion. Do not test with
   /v1/models. That test is not sufficient.
3. Start the trainer on all 8 nodes.
4. Start the orchestrator on n82.
5. Start the run supervisor on n82.
6. Start the tunnel watchdog on n82.

The commands are in `docs/RUNBOOK.md`.

# THE RULES THAT THE OWNER GAVE

Do not touch n79.

Do not delete the data of the owner. The protected paths are `/data/ZiphuAI`,
`/data/trinity`, and `/data/nv-community_GLM-5.2-NVFP4`. The model at
`/data/glm52-rl/model` on n85 is the only copy.

Do not use QLoRA. Do not use int4.

Do not use cached batches. Every attempt must be a full production run.

Do not use metaphors when you write to the owner. Write plain technical text.

Do not give opinions when the owner does not ask for them.

Separate the trainer time and the rollout time when you give an estimate. Do
not give one combined number.

Do not use automated monitor tools. Use a shell command with a sleep. The
command must print the GPU data, the step data, and the rollout data.

# THE MISTAKES THAT THE ASSISTANT MADE

Read this section before you change anything. Each item is a real failure.

# The assistant destroyed source files

The assistant copied `sparse_mla_attention.py` and `modeling_glm_moe_dsa.py`
out of the Docker image. The assistant then wrote them into the repository.
The repository copies had local changes. The Docker image copies did not have
those changes. The local changes were lost.

The lost code included `_FiniteGrad`. That function removes NaN values from
the gradients. The lost code also included `PRESCALE_QK`.

Read `scripts/china/cluster.sh` before you edit these paths. That file shows which files are mounted into the
container. Those files have local changes. Do not copy those paths out of the
image. Edit the repository copy.

# The assistant gave a wrong diagnosis for many hours

The assistant said that the host memory was not real. The assistant called it
phantom memory. That diagnosis was wrong. The document `PHANTOM_RAM_PROOF.md`
is retracted. The memory was real.

Measure before you write a conclusion. Do not present a theory as a fact.

# The assistant used a broken health test

The assistant used `pgrep -fc <name>` to test if a process runs. That command
counts the grep process. The result is always 1 or more. The test always
passed. The tunnel watchdog was dead for 3 hours.

Use `pgrep -f "<full command>" | wc -l` instead. Also read the log file. A
process that runs but does not write to its log does not work.

# The assistant used a watchdog that hid a fault

The assistant wrote a rule into the tunnel watchdog. The rule was: if 80
percent of the tunnels fail, do not rebuild them. The reason was to show a
dead engine. The rule was wrong. At batch 1024 all 24 tunnels failed at the
same time. The engines were healthy. The watchdog did nothing for 5 hours.

Always rebuild the tunnels. A rebuild against a dead engine fails again on the
next cycle. A refusal to rebuild against a live engine stops the run until an
operator finds it.

# The assistant tested the engine with the wrong probe

The assistant tested the engine with `/v1/models`. That endpoint answers when
the API server is alive. The EngineCore can be dead when the API server is
alive. On 2026-08-06 h200a gave HTTP 200 on `/v1/models`. The same host gave
no answer to a completion. The GPU showed 64 requests and 0 tokens per second.

Test the engine with a real completion. Use 3 tokens. Only a
completion proves that the engine works.

# The assistant broke the run with a tunnel restart

The assistant killed the batch tunnel while the orchestrator was alive. The
zmq session broke. Seven dead connections stayed on port 29620. The batches
did not arrive. The trainer waited 27 minutes.

Stop the orchestrator before you touch the batch tunnel.

# The assistant deadlocked the start sequence

The assistant made the orchestrator wait for the trainer to bind port 29620.
The trainer cannot start until the orchestrator writes `control/orch.toml`.
Both parts waited. Nothing happened.

Start the orchestrator first. The trainer starts after it.

# The assistant reported a trend that was noise

The assistant reported that the reward increased 26 percent. The measurement
used 25 steps. At 36 steps the increase was 6.3 percent. The standard
deviation was 0.039. The total trend was 0.013. The trend was inside the
noise.

Compare the trend against the standard deviation. Do not report a percentage
from a half window.

# THE FAULTS THAT WERE FOUND IN THE SOFTWARE

# Activation offloading was on by default

`ac_offloading` has the default value `ActivationOffloadingConfig()`. The
default is not `None`. See `configs/trainer.py` line 173. No configuration
file set it to `None`.

Activation offloading puts the activations into pinned host memory. The host
memory then follows the number of micro-batches in flight. It does not follow
the GPU memory.

This fault stopped every 16K run for one week. The host memory reached 486 GB
of 503 GB. The nodes stopped.

Set `ac_offloading = "None"`.

This fault also explains two failed experiments. FP8 weights cut the GPU
memory by half. The host memory did not change. `expandable_segments` cut the
reserved memory by one third. The host memory did not change.

# The adapter URL must resolve on the engine hosts

The vLLM engine workers download the LoRA adapter. See
`inference/vllm/worker/filesystem.py`, function `stage_lora_adapter`.

The `source_url` value is resolved on the H200 hosts. It is not resolved on
the orchestrator host.

The adapter server runs on n82 port 8100. The URL used port 18100. Nothing
answered on port 18100. Each download failed. Each failed download killed a
worker. Each dead worker killed the EngineCore.

Port 18100 now forwards to n82 port 8100 on all three hosts. One URL then works
on every host.

Before this fix no run could learn. The engines never received new weights.
All rollouts came from the base model. All reward data before run x22 is not
valid.

# FP8 trainer weights break the importance ratio

The engines use W8A16. The weights are 8-bit. The activations are 16-bit.

The trainer FP8 path uses W8A8. It also quantizes the activations. See
`fp8_ada.py`, function `moe_kernel_quantize_input`.

The trainer was therefore less exact than the engines. The mismatch KL reached
7 to 9 nats. The trust region then masked almost every positive advantage. The
entropy increased to 9.2 nats. The reward decreased.

Set `dequantize_checkpoint = true`. The trainer then uses bf16 weights. The
mismatch KL decreased to 0.2.

A better fix is possible. Make the trainer use W8A16. The function
`_dequant_weight` in `fp8_ada.py` is the start point.

# The pipeline stage buffer used the wrong dtype

`train.py` took the dtype of the inter-stage buffers from the first parameter.
Those buffers hold activations. With bf16 weights the two dtypes are the same.
With FP8 weights the first parameter is `float8_e4m3fn`. Every hidden-state
buffer was then declared FP8.

Use `DTYPE_MAP[config.model.optimization_dtype]`.

Do not scan for the first real float parameter. An fp32 `weight_scale_inv` can
come before the bf16 weights.

# The pipeline code failed with a ModuleDict

`apply_pp_boundaries` replaces the layer ModuleList with an `nn.ModuleDict`.
The keys are the global layer index as a string. See `pipeline_parallel.py`
line 139.

Three faults came from this:

`_context_parallel_state` used `self.layers[0]`. That call raised
`KeyError: 0`.

The forward loop sliced `self.layers`. A ModuleDict cannot be sliced.

The forward loop used `enumerate()`. That call gives a local index.
`routed_experts` needs the global index. This fault does not raise an error.
It replays the wrong router on each stage after the first.

`self.norm` is `None` on each stage except the last. The forward called it
without a test.

# A corrupt batch stopped the run

On 2026-08-06 a micro-batch arrived with an `input_ids` shape of
`[1, 9142729177965406720]`. That length is near 2^63. It is not a real length.

Torch raised "Storage size calculation overflowed". Rank 0 stopped. The other
7 ranks waited for the NCCL timeout. The run was dead for 6 hours.

`train.py` now tests each micro-batch shape. It discards a bad batch and the
run continues.

# THE LIMITS OF THE SYSTEM

# The tunnels limit the concurrency

The tunnels do not limit the batch size. They limit the number of rollouts in
flight.

At 1024 rollouts in flight all 24 tunnels failed. The watchdog rebuilt 10 to
14 tunnels each 45 seconds. They failed again at the same rate. The engines
fell to 1 to 6 concurrent requests.

At 256 rollouts in flight 20 of 24 tunnels stay healthy. The watchdog rebuilds
1 tunnel each cycle. The engines run 45 to 70 concurrent requests.

Set `batch_size` for the gradient quality. Set `max_inflight_rollouts` to 256.
These two values are independent.

# The batch transport cannot cross the link

The batch is one large stream. SSH gives each channel a window near 2 MB. On a
217 millisecond link the transfer stops at that window.

The orchestrator must run on n82. The batch then goes over the loopback
interface. The measured load time is 0.45 seconds.

The cost is that the rollout traffic then crosses the bad link. The rollout
collection is slower. Accept this cost. The trainer cannot use a rollout that
does not arrive.

# The reward is multiplicative

The reward multiplies five factors together: the judge score, the coherency,
the reasoning, the markdown, and the word count. One factor near zero makes the
whole reward near zero.

The 10th percentile of the reward stays near zero. The 90th percentile
increased from 0.276 to 0.648. The headroom is at the bottom, not the top.

# THE STATE AT HANDOFF

Run x24 uses batch 1024, group size 16, 256 rollouts in flight, 16K context,
bf16 trainer weights, and pipeline parallel 8.

Three steps are complete. The step time is 29 to 96 minutes and decreases. The
error rate is 42.5, then 27.4, then 13.7 percent. The error rate decreases
because the tunnels recover.

The reward is 0.0693, then 0.0702, then 0.1052. Do not use these three values
to make a conclusion. The first two batches had a high error rate.

Run x22 gave 36 steps at batch 256. The mean reward was 0.1205. The standard
deviation was 0.039. Use that data to compare.

# WHAT TO DO NEXT

Collect 10 to 15 clean steps at batch 1024. Then compare the standard
deviation against 0.039. That test shows if the larger batch decreased the
noise.

Add the SSH public key to n81. The key is in `docs/RUNBOOK.md`. The node is
healthy. The node has no authorized key. n81 gives 72 GPUs. A change of the
pipeline layout is necessary.

Make the trainer use W8A16. That change gives back the FP8 memory and keeps
the mismatch KL low.

The MFU is 2.2 to 3.0 percent. The step time has much space to decrease.
