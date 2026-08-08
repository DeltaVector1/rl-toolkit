# GLM-5.2 DECENSOR RL. RUNBOOK

This document gives the commands. `docs/HANDOFF.md` gives the reasons.
`.agents/skills/glm52-ops/references/FAULTS.md` gives the symptom and the
repair for each fault.

Language: ASD-STE100 Simplified Technical English.

# USE THE SKILL FIRST

The `glm52-ops` skill does everything below and it applies the guards. Use the
raw commands only when the skill cannot run.

```python
glm52_ops.status()                  # tunnels, steps, GPU, engines, warnings
glm52_ops.reward()                  # the reward series with a trend verdict
glm52_ops.watch(hours=2, every=300) # the same picture on a loop
glm52_ops.repair("tunnels")
glm52_ops.repair("engine:h200a")
glm52_ops.repair("supervisors")
glm52_ops.deploy("scripts/china/tunnel_watchdog.sh")
glm52_ops.start("x25", mode="cold")
glm52_ops.stop()
```

Outside a Prime Agent kernel:

```
PYTHONPATH=.agents/skills/glm52-ops/src python3 -c "import glm52_ops; glm52_ops.status()"
```

# THE HOSTS

Trainer nodes: n76, n77, n78, n80, n82, n83, n84, n85. Each has 8 RTX 4090.

Engine hosts: h200a, h200b. Each has 8 H200.

Judge: 202.214.223.66 port 26406. It is reached on n82 as
`http://127.0.0.1:26406/v1`.

Do not use n79. n81 holds no authorized key.

# THE RAW STATUS COMMAND

Use this only if Python is not available. Add a sleep to repeat it.

```
echo "=========== $(date -u +%H:%M:%S) UTC ==========="
timeout 200 ssh n82 'ok=0
  for p in $(seq 18001 18048); do
    c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
    [ "$c" = "200" ] && ok=$((ok+1))
  done
  echo "tunnels: $ok/48"
  echo -n "trainer steps: "; sudo docker logs glm52-decensor-trainer 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -ac "SUCCESS Step"
  sudo docker logs glm52-decensor-trainer 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -a "SUCCESS Step" | tail -1
  sudo docker logs glm52-decensor-orchestrator 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -a "SUCCESS Step" | tail -1
  sudo docker logs --tail 5 glm52-decensor-orchestrator 2>&1 | grep -oE "batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight" | tail -1
  echo -n "fails/5m: "; sudo docker logs --since 5m glm52-decensor-orchestrator 2>&1 | grep -ac "Rollout failed"
  echo -n "judge: "; curl -s -o /dev/null -m 10 -w "%{http_code}\n" http://127.0.0.1:26406/v1/models'
for h in n76 n77 n78 n80 n82 n83 n84 n85; do
  printf "%-5s " "$h"
  timeout 20 ssh $h 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1 | tr "\n" " "
    free -g | sed -n 2p | awk "{print \"| \"\$7\"G avail \"\$5\"G shared\"}"'
done
for h in h200a h200b; do
  printf "%-6s " "$h"
  timeout 40 ssh $h 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | tr "\n" " "
    s=$(date +%s%N)
    c=$(curl -s -o /dev/null -m 20 -w "%{http_code}" -X POST http://127.0.0.1:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"/data/glm52-rl/model-fp8\",\"prompt\":\"hi\",\"max_tokens\":3}")
    e=$(date +%s%N)
    echo -n "| gen=$c/$(( (e-s)/1000000 ))ms | "
    sudo docker logs --tail 15 glm52-decensor-h200 2>&1 | grep -aoE "Running: [0-9]+|generation throughput: [0-9.]+" | tail -2 | tr "\n" " "
    echo'
done
```

# THE HEALTHY VALUES

The current pool has 48 tunnels. Probe counts fall under load, so confirm them against engine throughput. Engine generation probe 200 in under 4000 ms. Engine
concurrency 45 to 70. Node GPU memory near 46 GiB of 48. Host memory 400 GB or
more free. Judge 200. Adapter server 200.

# HOW TO START A RUN

Always use a new run name. Do not use the name of a run that stopped.

# Step 1. Make the configuration files

```
cd /home/mango/Misc/data-gen/glm52-decensor
python3 -c "
import pathlib
p=pathlib.Path
p('configs/china/trainer.xNN.toml').write_text(p('configs/china/trainer.x24.toml').read_text().replace('x24','xNN'))
p('configs/china/orchestrator.xNN.toml').write_text(p('configs/china/orchestrator.x24.toml').read_text().replace('x24','xNN'))"
scripts/ops/cl.py edit configs/china/trainer.xNN.toml
scripts/ops/cl.py -n n82 edit configs/china/orchestrator.xNN.toml
```

# Step 2. Test the engines

Test with a real completion. Do not test with `/v1/models`. That endpoint
answers while the EngineCore is dead.

```
for h in h200a h200b; do
  ssh $h 'curl -s -o /dev/null -m 25 -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/data/glm52-rl/model-fp8\",\"prompt\":\"hi\",\"max_tokens\":3}"'
done
```

Both hosts must give 200.

# Step 3. Test the tunnels

The current pool has 48 ports. A low probe count is not sufficient evidence of failure. If it is low, kill the ports that fail. The
supervisors on the H200 hosts rebuild them.

```
ssh n82 'k=0
  for p in $(seq 18001 18048); do
    c=$(curl -s -o /dev/null -m 6 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
    if [ "$c" != "200" ]; then
      pid=$(sudo ss -H -lntp "sport = :$p" | sed -n "s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
      [ -n "$pid" ] && sudo kill $pid && k=$((k+1))
    fi
  done
  echo "killed $k"'
```

# Step 4. Start the two parts in the correct order

The order depends on whether the run name is new. Both orders below are
correct. Do not use one order for both cases.

For a NEW run name, start the orchestrator first. `control/orch.toml` does not
exist yet, only the orchestrator writes it, and the trainer blocks until it
exists. A trainer started first sits at "No orchestrator config found" until
the orchestrator arrives.

For an EXISTING run name, start the trainer first. `control/orch.toml` is
already on disk, so the trainer starts alone. A zmq send to a dead trainer
blocks the orchestrator event loop, so the trainer must listen first.

Never make the orchestrator wait for the trainer to bind port 29620. The
trainer waits for a file that only the orchestrator writes. That pair of waits
is a deadlock.

Trainer:

```
TN="82 76 77 78 80 83 84 85"
for h in n82 n76 n77 n78 n80 n83 n84 n85; do
  ssh $h "sudo pkill -9 -f PRIME-RL; sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1"
done
for h in n82 n76 n77 n78 n80 n83 n84 n85; do
  ( ssh $h "TRAINER_NODES='$TN' RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN \
      PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE=0 \
      bash /data/glm52-rl/deployment/scripts/china/cluster.sh trainer" ) &
done
wait
```

Orchestrator:

```
ssh n82 'RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN \
  bash /data/glm52-rl/deployment/scripts/china/cluster.sh orchestrator'
```

# Step 5. Start the supervisors

```
ssh n82 'nohup setsid env RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN INTERVAL=120 \
  bash /data/glm52-rl/deployment/scripts/china/run_supervisor.sh \
  >> /data/glm52-rl/run_supervisor.log 2>&1 </dev/null & disown'

ssh n82 'nohup setsid env TIMEOUT=20 INTERVAL=45 HEARTBEAT=4 \
  bash /data/glm52-rl/deployment/scripts/china/tunnel_watchdog.sh \
  >> /data/glm52-rl/tunnel_watchdog.log 2>&1 </dev/null & disown'

for h in h200a h200b; do
  ssh $h 'nohup setsid bash /data/glm52-rl/deployment/scripts/h200/engine_watchdog.sh \
    >> /root/engine_watchdog.log 2>&1 </dev/null & disown'
done
```

# Step 6. Test that the supervisors run

A liveness probe must not match its own command line. `pgrep -fc` counts its
own grep. `pgrep -f "bash .*name.sh"` sent over SSH matches the remote shell,
because that shell really is bash and the pattern holds the name. Bracket the
LAST token, not the first.

```
ssh n82 'echo "supervisor: $(pgrep -f "bash .*[r]un_supervisor.sh" | wc -l)"
  echo "watchdog: $(pgrep -f "bash .*[t]unnel_watchdog.sh" | wc -l)"
  tail -1 /data/glm52-rl/tunnel_watchdog.log'
```

The log must show a new line. The tunnel watchdog writes a heartbeat every 4
cycles, which is about every 3 minutes. A process with no new log line does not
work.

Never put a kill and a start for the same process in one SSH command. The
command line then holds the plain name, the kill matches its own shell, and the
session dies before the start runs. That killed all three watchdogs on
2026-08-07. Use two separate connections.

# HOW TO REPAIR AN ENGINE

The engine watchdog does this without help. Do this only if the watchdog does
not run.

```
ssh h200a 'sudo docker rm -f glm52-decensor-h200
  bash /data/glm52-rl/deployment/scripts/h200/start_inference.sh'
```

The model load needs 5 to 10 minutes. Test with a completion.

# HOW TO REPAIR THE TUNNELS

Use the command in Step 3. The watchdog does this every 45 seconds.

Do not stop the batch tunnel on port 29620 while the orchestrator runs. Stop
the orchestrator first.

# HOW TO REPAIR A STALLED ORCHESTRATOR

Look for this text in the orchestrator log:

```
Resuming orchestrator at rollout batch N from policy vM
Pausing dispatcher to prevent orchestrator from racing from trainer
```

The orchestrator expects a policy version that the trainer cannot make. This
occurs when the trainer restarts and the orchestrator does not.

The repair is a new run name. Do not try to repair the old run directory.

# HOW TO TEST THE WATCHDOG LOGIC

```
bash scripts/ops/watchdog_smoketest.sh
```

This runs the real watchdog scripts against a fake environment. It touches
nothing real. It must report 26 passed and 0 failed.

# THE KEY FOR n81

n81 is healthy and holds no authorized key. Add this key to
`~/.ssh/authorized_keys` for the user `ubuntu`:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDplYZMm618PRgB/0C4bLNTPmzid5L+YB9YHNnbwxBsZ austral@Mac
```

n81 was installed again, so the host key changed. Remove the old one:

```
ssh-keygen -f ~/.ssh/known_hosts -R 10.92.20.81
```

The correct fingerprint is
`SHA256:FUQ/FcqY+fXt+e+GGVvI2fr90GDlV9ge/Shrwoqe2Q0`.

n81 gives 72 GPUs. A change of the pipeline layout is necessary.

# THE BACKUPS

Git tags: `working-16k-step` and `healthy-training`.

Git bundles: `~/Misc/data-gen/backups/`.

Node snapshot: `/data/glm52-rl/deployment-WORKING-16KSTEP-*.tar.gz` on n84.

Patches: `patches/glm52-fixes-all.patch` and `patches/series/`.

# THE SWANLAB PROJECT

`https://swanlab.cn/@MangyMangaur/prod-run`.

Do not compare run x22 and later against run x21 and earlier. The runs before
x22 had a broken adapter path. Their reward data is not valid.
