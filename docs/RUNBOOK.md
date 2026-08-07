# GLM-5.2 DECENSOR RL. RUNBOOK

This document gives the commands. Read `docs/HANDOFF.md` first. That document
gives the reasons.

Language: ASD-STE100 Simplified Technical English.

# THE HOSTS

Trainer nodes: n76, n77, n78, n80, n82, n83, n84, n85. Each node has 8 RTX
4090 GPUs.

Engine hosts: h200a, h200b. Each host has 8 H200 GPUs.

Judge host: `judge-ext`. The address is 202.214.223.66 port 26498.

Do not use n79. n81 has no authorized key.

# THE STATUS COMMAND

Use this command to see the state of the run. Add a sleep to repeat it.

```
echo "=========== $(date -u +%H:%M:%S) UTC ==========="
timeout 200 ssh n82 'ok=0
  for p in $(seq 18001 18024); do
    c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
    [ "$c" = "200" ] && ok=$((ok+1))
  done
  echo "tunnels: $ok/24"
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

# HOW TO START A RUN

Replace `xNN` with the new run name. Always use a new name. Do not use the
name of a run that stopped.

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

Test with a real completion. Do not test with `/v1/models`.

```
for h in h200a h200b; do
  ssh $h 'curl -s -o /dev/null -m 25 -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/data/glm52-rl/model-fp8\",\"prompt\":\"hi\",\"max_tokens\":3}"'
done
```

Both hosts must give 200. If a host does not give 200, go to "How to repair an
engine".

# Step 3. Test the tunnels

Count the ports that answer. The count must be 20 or more of 24.

If the count is low, kill the tunnels that fail. The supervisors on the H200
hosts then rebuild them.

```
ssh n82 'k=0
  for p in $(seq 18001 18024); do
    c=$(curl -s -o /dev/null -m 6 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
    if [ "$c" != "200" ]; then
      pid=$(sudo ss -H -lntp "sport = :$p" | sed -n "s/.*pid=\([0-9][0-9]*\).*/\1/p" | head -1)
      [ -n "$pid" ] && sudo kill $pid && k=$((k+1))
    fi
  done
  echo "killed $k"'
```

# Step 4. Start the trainer

```
TN="82 76 77 78 80 83 84 85"
for h in n82 n76 n77 n78 n80 n83 n84 n85; do
  ssh $h "sudo pkill -9 -f PRIME-RL; sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1"
done
for h in n82 n76 n77 n78 n80 n83 n84 n85; do
  ( ssh $h "TRAINER_NODES='$TN' RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN \
      PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE=1 \
      bash /data/glm52-rl/deployment/scripts/china/cluster.sh trainer" ) &
done
wait
```

# Step 5. Start the orchestrator

Start the orchestrator after the trainer. The trainer waits for
`control/orch.toml`. Only the orchestrator writes that file.

```
ssh n82 'RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN \
  bash /data/glm52-rl/deployment/scripts/china/cluster.sh orchestrator'
```

# Step 6. Start the supervisors

```
ssh n82 'nohup setsid env RUN=xNN RUN_NAME=run_xNN CONFIG_TAG=.xNN INTERVAL=120 \
  bash /data/glm52-rl/deployment/scripts/china/run_supervisor.sh \
  >> /data/glm52-rl/run_supervisor.log 2>&1 </dev/null & disown'

ssh n82 'nohup setsid env TIMEOUT=20 INTERVAL=45 \
  bash /data/glm52-rl/deployment/scripts/china/tunnel_watchdog.sh \
  >> /data/glm52-rl/tunnel_watchdog.log 2>&1 </dev/null & disown'

for h in h200a h200b; do
  ssh $h 'nohup setsid bash /data/glm52-rl/deployment/scripts/h200/engine_watchdog.sh \
    >> /root/engine_watchdog.log 2>&1 </dev/null & disown'
done
```

# Step 7. Test that the supervisors run

Do not use `pgrep -fc`. That command counts the grep process.

```
ssh n82 'echo "supervisor: $(pgrep -f "bash /data/glm52-rl/deployment/scripts/china/run_supervisor.sh" | wc -l)"
  echo "watchdog: $(pgrep -f "bash /data/glm52-rl/deployment/scripts/china/tunnel_watchdog.sh" | wc -l)"
  tail -1 /data/glm52-rl/tunnel_watchdog.log'
```

The log must show a new line with the current date. A process without a new
log line does not work.

# HOW TO REPAIR AN ENGINE

The engine watchdog does this without help. Do this only if the watchdog does
not run.

```
ssh h200a 'sudo docker rm -f glm52-decensor-h200
  bash /data/glm52-rl/deployment/scripts/h200/start_inference.sh'
```

The model load needs 5 to 10 minutes. Test with a completion, not with
`/v1/models`.

# HOW TO REPAIR THE TUNNELS

Use the command in "Step 3". The watchdog does this each 45 seconds.

Do not stop the batch tunnel on port 29620 while the orchestrator runs. Stop
the orchestrator first.

# HOW TO REPAIR A STALLED ORCHESTRATOR

Look for this text in the orchestrator log:

```
Resuming orchestrator at rollout batch N from policy vM
Pausing dispatcher to prevent orchestrator from racing from trainer
```

This text means that the orchestrator expects a policy version that the
trainer cannot make. This occurs when the trainer restarts and the
orchestrator does not.

The repair is a new run name. Do not try to repair the old run directory.

# THE KEY FOR n81

n81 is healthy. n81 has no authorized key. Add this key to
`~/.ssh/authorized_keys` for the user `ubuntu`:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDplYZMm618PRgB/0C4bLNTPmzid5L+YB9YHNnbwxBsZ austral@Mac
```

n81 was installed again. The host key changed. Remove the old host key:

```
ssh-keygen -f ~/.ssh/known_hosts -R 10.92.20.81
```

The correct fingerprint is
`SHA256:FUQ/FcqY+fXt+e+GGVvI2fr90GDlV9ge/Shrwoqe2Q0`.

# THE BACKUPS

Git tags: `working-16k-step` and `healthy-training`.

Git bundles: `~/Misc/data-gen/backups/`.

Node snapshot: `/data/glm52-rl/deployment-WORKING-16KSTEP-*.tar.gz` on n84.

Patches: `patches/glm52-fixes-all.patch` and `patches/series/`.

# THE SWANLAB PROJECT

The address is `https://swanlab.cn/@MangyMangaur/prod-run`.

Do not compare run x22 and later against run x21 and earlier. The runs before
x22 had a broken adapter path. Their reward data is not valid.
