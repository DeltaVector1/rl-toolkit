#!/usr/bin/env bash
# Run the orchestrator on h200a instead of n82.
#
# Why: every rollout turn is an HTTP round trip to an inference engine and to
# the judge, and none of those live in China. With the orchestrator on n82 all
# of that traffic crossed a 217ms link measuring 10-33% packet loss, through
# SSH tunnels that wedge silently -- observed 22 of 24 dead, engines at 8% KV
# cache while 256 rollouts sat blocked, ~90 min for a 256-rollout batch.
#
# Measured paths from h200a:
#   own engine        127.0.0.1:8000    5.8ms, no tunnel
#   h200b engine      LAN               0.48ms, 0% loss
#   judge             202.214.223.66    176ms, 0% loss, no tunnel
#   trainer (zmq)     n82:29620         crosses the lossy link, but it is one
#                                       bulk stream, not thousands of small
#                                       latency-sensitive requests
#
# ZMQTrainingBatchReceiver binds on the trainer and ZMQTrainingBatchSender
# connects from here, so a forward tunnel is the correct direction and the
# config sets host = 127.0.0.1. Ports +1/+2 are the trainer's internal
# micro-batch fan-out and are not used from this side.
set -uo pipefail

CONFIG_TAG=${CONFIG_TAG:-.x7h}
RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
deployment=/data/glm52-rl/deployment
config=$deployment/configs/h200
image=prime-rl:glm52-decensor

tunnel() { # local_port  target_host:port  ssh_target  [compress]
  local lp=$1 tgt=$2 via=$3 comp=${4:-no}
  # Match on the forward spec, not a SetEnv tag: any tunnel already holding
  # this port counts, however it was started. Two supervisors on one port means
  # the loser respawns every 3s forever under ExitOnForwardFailure.
  pgrep -f -- "-L 127.0.0.1:$lp:" >/dev/null && return 0
  setsid nohup bash -c "while true; do
      ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
          -o IPQoS=none -o Compression=$comp \
          -L 127.0.0.1:$lp:$tgt $via
      sleep 3
    done" >/dev/null 2>&1 < /dev/null &
}

# h200b's engine over the 0.48ms host-to-host link.
tunnel 18101 127.0.0.1:8000 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
# Trainer batch socket and the adapter weight source, both on n82.
# Compression is mandatory on 29620, not a tuning knob: the China link does
# <130 KB/s of bulk TCP (20 MB failed to cross in 150s) because throughput
# collapses as MSS/(RTT*sqrt(loss)) at 217ms and 10-33% loss. Uncompressed, the
# trainer starved for 12 minutes while the orchestrator ran 3 steps ahead. The
# same 20 MB crossed in 9s with -C, and the live tunnel measured 4.9 MB/s --
# token ids and logprobs compress well.
tunnel 29620 10.92.20.82:29620 china-trainer yes
tunnel 18100 10.92.20.82:18100 china-trainer yes

sleep 6
for p in 18101 29620 18100; do
  ss -tln 2>/dev/null | grep -q "127.0.0.1:$p" && echo "tunnel :$p up" || echo "tunnel :$p DOWN"
done

sudo docker rm -f glm52-decensor-orchestrator >/dev/null 2>&1 || true
sudo docker run -d --name glm52-decensor-orchestrator \
  --network host --restart no \
  --env-file /data/trinity/training.env \
  -e VLLM_API_KEY=unused \
  -e PYTHONPATH=/data/glm52-rl/decensor-v1:/app/src \
  -e HF_HOME=/data/glm52-rl/hf-cache \
  -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -v /data/glm52-rl:/data/glm52-rl \
  -v "$deployment/source/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
  -v "$deployment/source/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
  -v "$deployment/source/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/algo/grpo.py":/app/src/prime_rl/orchestrator/algo/grpo.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/dispatcher.py":/app/src/prime_rl/orchestrator/dispatcher.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/orchestrator.py":/app/src/prime_rl/orchestrator/orchestrator.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
  -v "$deployment/source/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
  --entrypoint /app/.venv/bin/orchestrator "$image" \
  @ "$config/orchestrator${CONFIG_TAG}.toml"
echo "orchestrator started on $(hostname) with ${CONFIG_TAG}"

# The trainer reads the orchestrator's control file off its OWN filesystem
# (runs.py:232). That was implicit when both ran on n82; with the orchestrator
# here it writes control/orch.toml locally and the trainer waits forever with
# "No orchestrator config found". Ship it over once it appears -- the file is
# written at orchestrator startup and does not change afterwards.
ctl=/data/glm52-rl/runs/$RUN/$RUN_NAME/control/orch.toml
for _ in $(seq 1 60); do
  [ -f "$ctl" ] && break
  sleep 2
done
if [ -f "$ctl" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=20 china-trainer \
      "mkdir -p /data/glm52-rl/runs/$RUN/$RUN_NAME/control" \
    && scp -q -o BatchMode=yes "$ctl" "china-trainer:$ctl" \
    && echo "control/orch.toml -> trainer ok" \
    || echo "WARNING: could not ship orch.toml to the trainer; it will not start" >&2
else
  echo "WARNING: $ctl never appeared; trainer will not start" >&2
fi
