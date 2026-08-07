# The rules the owner gave

These are instructions, not preferences. Some are in code in `glm52_ops`, which
raises `Refused` instead of obeying a request that breaks them. Language:
ASD-STE100 Simplified Technical English.

# Rules about the hardware

Do not touch n79. There is no exception. `glm52_ops` refuses any host argument
that names it.

Do not delete the data of the owner. The protected paths are `/data/ZiphuAI`,
`/data/trinity`, and `/data/nv-community_GLM-5.2-NVFP4`. The model at
`/data/glm52-rl/model` on n85 is the only copy.

Secrets are in `/data/trinity/training.env`. That file is outside the
repository. Keep it outside.

# Rules about the method

Do not use QLoRA. Do not use int4 weights.

Do not use a cached batch. Every attempt is a full production run. `start`
never forwards `PRIME_BATCH_CACHE`, because a leaked value replays one frozen
batch for every step while the run looks healthy.

# Rules about monitoring

Do not use an automated monitor tool. Use a foreground command with a sleep in
it. The command prints the GPU use, the generation time, the step count, and
the rollout count in one place.

`glm52_ops.watch()` is that command. It blocks. Do not move it to a thread or
a background cell.

Nothing must sit still for more than 10 minutes. `watch` compares each cycle
against the last one and prints `STALLED` when the step count and the batch
progress do not change.

# Rules about how to write to the owner

Do not use metaphors. Write plain technical text.

Do not give an opinion that the owner did not ask for.

Separate the trainer time and the rollout time in an estimate. Do not give one
combined number. The trainer step time and the rollout collection time overlap
and have different causes.

Do not report a trend that is inside the noise. `reward()` gives a verdict.
Report the verdict.

Measure before you write a conclusion. Do not present a theory as a fact. A
wrong diagnosis about host memory cost a week.

# The success criterion

The reward rises, or the reward distribution rises. Nothing else counts.

The target is a 15 minute step time at 16K context.

Do not compare run x22 or later against x21 or earlier. The adapter never
reached the engines before x22, so every rollout in those runs came from the
base model and their reward data is not valid.
