---
name: glm52-ops
description: Operate the GLM-5.2 decensor GRPO run across the 8 China trainer nodes, the 2 H200 rollout hosts, and the external judge. Use it to read run status (tunnels, steps, GPU use, engine generation), to decide whether reward is really trending or only moving inside the noise, to repair wedged tunnels and dead engines, to push a changed source file to every node, and to start, restart, or stop a run. Use it for any question of the form "how is the run going", "is reward going up", "is anything stuck", or "why did it stop".
compatibility: Needs SSH aliases n76 n77 n78 n80 n82 n83 n84 n85 h200a h200b, passwordless sudo on the nodes, and Python 3.10 or later. Read-only functions are safe. The write functions change a live training run.
metadata:
  project: glm52-decensor
  scope: project
---

# GLM-5.2 decensor operations

The kernel imports this skill as `glm52_ops`. All functions are plain Python.
They print a report and also return the data.

```python
glm52_ops.status()          # one full picture of the run
glm52_ops.reward()          # reward series with a trend verdict
glm52_ops.watch(hours=2)    # the same picture on a loop, in the foreground
```

`await glm52_ops("status")` and `await glm52_ops("reward")` do the same thing.

Read `references/FAULTS.md` for the fault table. Read `references/RULES.md` for
the rules that the owner gave.

# The four rules that this skill puts into code

The skill refuses some actions. Do not work around the refusal.

n79 is never touched. Any host argument that names n79 raises an error.

No batch cache. `deploy` and `start` refuse to forward `PRIME_BATCH_CACHE`. A
cached batch makes a dead run look healthy.

No background monitor. `watch` blocks the kernel and prints each cycle. Do not
put it in a thread. Do not put it in a background cell. The owner asked for a
command with a sleep in it that prints the state. `watch` is that command.

No trend claim inside the noise. `reward` returns the verdict
`"inside the noise"` until the slope is larger than the step-to-step spread.
Report the verdict. Do not read the numbers yourself and give a better answer.

# How to answer "how is the run going"

```python
s = glm52_ops.status()
```

The report gives the tunnel count, the last trainer step, the last orchestrator
step, the batch progress, the rollout error rate, the judge state, the GPU
memory and use for each node, and a real generation probe for each engine.

Healthy values, measured on run x24:

Tunnels 20 to 22 of 24. Engine probe returns 200 in under 4000 ms. Engine
concurrency 45 to 70. Node GPU memory near 46 GiB of 48. Host memory 400 GB or
more free. Judge returns 200.

A tunnel count of 10 or less means the rollout load is too high. Check
`max_inflight_rollouts`. The value 256 works. The value 1024 does not.

# How to answer "is the reward going up"

```python
r = glm52_ops.reward()
```

Use `r.verdict`. It has three values.

`"too few steps"` means fewer than 10 steps. There is no answer yet. Say so.

`"inside the noise"` means the total change is smaller than twice the residual
spread. The reward did not move. Say so.

`"rising"` or `"falling"` means the change is larger than that. Give the number
and the residual spread together.

The report gives two reward series. Use the right one.

`effective` covers the rollouts that reached the trainer. It measures the
policy.

`all` covers every rollout. A failed rollout scores zero. This series moves
when the tunnels recover, not when the model learns. On x24 the `all` reward
went from 0.0408 to 0.0907 while the error rate went from 42.5 to 13.7 percent.
Almost all of that rise was transport.

The reward is a product of five factors. The 10th percentile stays near zero
because one factor near zero makes the whole product near zero. Watch the 90th
percentile for the real distribution signal. On x24 it went from 0.09 to 0.5034
in three steps.

# How to repair a fault

```python
glm52_ops.repair("tunnels")        # kill wedged tunnels; the H200 side rebuilds
glm52_ops.repair("engine:h200a")   # rebuild one engine container
glm52_ops.repair("supervisors")    # restart any supervisor that stopped
```

`repair` refuses to touch the batch tunnel on port 29620 while the orchestrator
runs. Stop the orchestrator first. A killed batch tunnel leaves dead zmq
connections and the trainer waits.

A stalled orchestrator has no repair. Start a new run name. See
`references/FAULTS.md`.

# How to push a changed file

```python
glm52_ops.deploy("source/src/prime_rl/trainer/models/layers/norms.py")
```

`deploy` reads the mount list out of `scripts/china/cluster.sh` and refuses a
path that the container does not mount. An unmounted file has no effect,
because the container then uses the copy inside the image.

Never copy a file out of the Docker image into the repository. The repository
copies hold local changes that the image does not have. That action destroyed
`sparse_mla_attention.py` and `modeling_glm_moe_dsa.py` once.

A source change needs a trainer restart. `deploy` says so and does not restart
by itself.

# How to start a run

```python
glm52_ops.start("x25", mode="cold")      # a new run name
glm52_ops.start("x24", mode="restart")   # an existing run name
```

The two modes start the two parts in opposite order. Both orders are correct
for their case.

A cold start has no `control/orch.toml` yet. Only the orchestrator writes that
file. The trainer blocks until the file exists. So the orchestrator starts
first.

A restart has `control/orch.toml` on disk already. The trainer can start alone.
The orchestrator must not start against a dead trainer, because a zmq send to a
dead peer blocks its event loop. So the trainer starts first.

Never make the orchestrator wait for the trainer to bind port 29620. The
trainer waits for a file that only the orchestrator writes. That pair of waits
is a deadlock. It stopped the run once.

# How to test that a process runs

```python
glm52_ops.alive("tunnel_watchdog.sh")
```

Do not use `pgrep -fc`. That command counts the grep process. The result is
never zero. This test reported three dead watchdogs as healthy for three hours.

`alive` uses the full command line and also reads the log file. A process that
runs but writes no new log line does not work.

# What this skill does not do

It does not read SwanLab. The address is
`https://swanlab.cn/@MangyMangaur/prod-run`.

It does not compare against run x21 or earlier. Those runs had a broken adapter
path, so the engines never received new weights. Their reward data is not
valid. Compare against x22: 36 steps at batch 256, mean reward 0.1205, standard
deviation 0.039.
