# Fault table

Each entry gives the symptom, the cause, and the one correct repair. Every
entry is a fault that happened. Language: ASD-STE100 Simplified Technical
English.

# Find the fault first

```python
s = glm52_ops.status()
s.warnings
```

`status` prints a warning for each condition below that it can detect. Match
the warning to an entry here. Do not repair a fault that `status` does not
show.

# The engine answers /v1/models but makes no tokens

Symptom: `gen=000` from `engines()`, or an engine at 200 on `/v1/models` with
`Running: 64` and `generation throughput: 0.0`.

Cause: the vLLM API server is alive and the EngineCore is dead. The two are
separate processes. The API server keeps answering metadata requests.

Repair: `glm52_ops.repair("engine:h200a")`. The model load needs 5 to 10
minutes. Test again with `engines()`, never with `/v1/models`.

This happened on 2026-08-06. The container showed "Up" for 2.6 hours with dead
engines. The container state is not a health signal either.

# The tunnels are listening but pass no data

Symptom: fewer than 18 of 48 ports answer. The probe hangs for the whole
timeout instead of refusing at once.

Cause: the SSH forward stays established while the link drops the packets. The
port accepts the connection and never answers.

Repair: `glm52_ops.repair("tunnels")`. This kills the n82 side sshd. The
supervisor on the H200 host rebuilds the tunnel in about 5 seconds.

Always rebuild, even when every tunnel fails at once. An earlier watchdog
refused to rebuild above 80 percent failure. That rule stopped the run for 5
hours while the engines were healthy.

# Ten or fewer tunnels answer and the engines are idle

Symptom: low tunnel probe count, idle engines, and many rollouts in flight.

Cause: too many rollouts in flight. The tunnels are the limit, not the batch
size.

Repair: confirm that engine throughput is also low. The current
`max_inflight_rollouts` is 128. Do not change it or `batch_size` without an
experiment decision. The older 256/1024 measurements used 24 tunnels and do
not describe the current 48-port pool.

# The trainer waits and no batch arrives

Symptom: the trainer log stops. `time/wait_for_batch` grows. Port 29620 holds
several connections.

Cause: somebody killed the batch tunnel while the orchestrator was alive. The
zmq session broke and left dead connections behind.

Repair: stop the orchestrator, then rebuild the tunnel, then start the
orchestrator. The trainer waited 27 minutes when this happened.

Never touch port 29620 while the orchestrator runs. `repair("tunnels")` does
not touch it.

# The off-policy gate stays closed after the trainer advances

Symptom: the orchestrator prints `Pausing dispatcher` and remains at
`0/1024` with zero in flight after a newer adapter `STABLE` marker exists.
The trainer continues to complete steps. The H200 engines can show the new
adapter even though the watcher policy version does not advance.

Cause: vLLM completed `/load_lora_adapter`, but one HTTP response did not
finish. The weight watcher remained inside the inference update. An engine
restart can also outlast the old ten-attempt retry cap.

Repair: use the patched `client.py`, confirm both engines are healthy with
`engines()`, then restart only the edge orchestrator at the clean zero-inflight
boundary. With `ckpt.resume_step = -1` and `skip_progress = true`, it resumes
from the latest immutable rollout batch and latest `STABLE` policy. Do not
restart the trainer. Confirm the exact adapter SHA-256 on both H200 caches.

This happened in x56p after rollout batch 6. The orchestrator resumed batch 7
from policy version 3. The trainer continued without a restart.

# The orchestrator waits for a policy version that cannot exist

Symptom: these two lines in the orchestrator log.

```
Resuming orchestrator at rollout batch N from policy vM
Pausing dispatcher to prevent orchestrator from racing from trainer
```

Cause: the trainer restarted and the orchestrator did not. The orchestrator
now expects a weight version that the trainer will never produce.

Repair: start a new run name. `glm52_ops.start("x25", mode="cold")`. Do not try
to repair the old run directory.

# The trainer stopped and nothing restarted it

Symptom: `trainer DOWN` in `status`, and the trainer log ends with either
"Storage size calculation overflowed" or an NCCL collective timeout.

Cause: a corrupt micro-batch stopped rank 0. The other 7 ranks then waited for
the NCCL timeout. One micro-batch arrived with an `input_ids` shape of
`[1, 9142729177965406720]`.

Repair: `train.py` now tests each micro-batch shape and discards a bad batch,
so this should not stop the run again. If the trainer is down, the run
supervisor should restart it. Check the supervisor first with
`alive("bash .*run_supervisor.sh")`.

This fault held the run dead for 6 hours because no supervisor existed.

# The nodes run out of host memory

Symptom: fewer than 60 GB free on a node. The OOM killer stops the trainer.
The kernel trace shows `shmem_fault` and `shmem_alloc_folio`.

Cause: `ac_offloading` is not `"None"`. The default is
`ActivationOffloadingConfig()`, which puts activations into pinned host memory.
The host memory then follows the number of micro-batches in flight.

Repair: set `ac_offloading = "None"` in the trainer config. Host memory then
drops from 486 GB of 503 GB to about 3 GB.

This memory is real. An earlier note called it an accounting artifact. That
note was wrong and is retracted.

# The reward will not rise and the entropy is very high

Symptom: entropy near 9.2 nats against a vocabulary limit of 11.95. Gradient
norm 200000 to 330000 against a clip limit of 1.0. Mismatch KL 7 to 9 nats.

Cause: the trainer ran FP8. The engines use W8A16, which quantizes the weights
only. The trainer FP8 path uses W8A8, which also quantizes the activations, so
the trainer is less exact than the engines. The trust region then masks almost
every positive advantage.

Repair: set `dequantize_checkpoint = true`. The trainer then uses bf16 weights,
the mismatch KL drops to 0.2, and the entropy drops to 0.45.

A better fix is to make the trainer use W8A16. Start at `_dequant_weight` in
`fp8_ada.py`.

# The reward looks flat and nothing is broken

Symptom: `reward()` returns the verdict "inside the noise".

Cause: this is not a fault. The 10th percentile of the reward stays near zero
because the reward multiplies five factors and one factor near zero makes the
product near zero.

Action: report the verdict. Read the 90th percentile for the distribution
signal. Collect more steps. Do not report a percentage from a half window.

# A watchdog exists but does nothing

Symptom: `alive()` prints a process count above zero and a stale log line.

Cause: the process is stuck, or an old test used `pgrep -fc`. That command
counts the grep process, so its result is never zero and the test always
passes.

Repair: `glm52_ops.repair("supervisors")`. Always read the log timestamp as
well as the process count. This test hid three dead watchdogs for 3 hours.

# A source change has no effect

Symptom: the behaviour does not change after `deploy` reports success and the
trainer restarts.

Cause: `cluster.sh` does not mount that file, so the container uses the copy
inside the Docker image.

Repair: call `glm52_ops.mounted_files()`. Add a `-v` line to
`scripts/china/cluster.sh` if the path is missing, then deploy `cluster.sh`
itself.

Never copy a file out of the image into the repository to "restore" it. The
repository copies hold local changes that the image does not have. That action
destroyed `_FiniteGrad` and `PRESCALE_QK`.
