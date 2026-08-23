#!/usr/bin/env bash
# Run the orchestrator on h200a instead of n82.
# Engine traffic stays local. Only adapter and completed batch data cross the
# China link.
set -uo pipefail

CONFIG_TAG=${CONFIG_TAG:-.x7h}
RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
deployment=${prime_deployment:-/data/glm52-rl/deployment-glm52-optims}
config=$deployment/configs/h200
image=${prime_image:-prime-rl:glm52-v080-full}
model_volume=${prime_model_volume:-glm52-fp8-model}
orchestrator_container=${prime_orchestrator_container:-glm52-optims-orchestrator}
env_container=${prime_env_container:-glm52-optims-env-train}
trainer_container=${prime_trainer_container:-glm52-optims-trainer}
config_file=${ORCH_CONFIG:-$config/orchestrator${CONFIG_TAG}.toml}
env_config_file=${ENV_CONFIG:-$config/env${CONFIG_TAG}.toml}
orchestrator_output=$(python3 - "$config_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    print(tomllib.load(source)["output_dir"])
PY
)
transport=$(awk '
  $0 == "[rollout_transport]" { active=1; next }
  /^\[/ { active=0 }
  active && $1 == "type" { gsub(/[ "\047]/, "", $3); print $3; exit }
' "$config_file")
adapter_admin_urls=$(python3 - "$config_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
for url in config["model"]["client"]["admin_base_url"]:
    print(url.removesuffix("/v1"), end=" ")
PY
)

tunnel() { # local_port  target_host:port  ssh_target  [compress]
  local lp=$1 tgt=$2 via=$3 comp=${4:-no}
  pgrep -f -- "-L 127.0.0.1:$lp:" >/dev/null && return 0
  setsid nohup bash -c "while true; do
      ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
          -o IPQoS=none -o Compression=$comp \
          -L 127.0.0.1:$lp:$tgt $via
      sleep 3
    done" >/dev/null 2>&1 < /dev/null &
}

tunnel 18100 127.0.0.1:8100 china-trainer yes
ports=(18100)
if [[ " $adapter_admin_urls " == *":18111 "* ]]; then
  tunnel 18101 127.0.0.1:8000 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
  tunnel 18111 127.0.0.1:8001 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
  tunnel 18112 127.0.0.1:8002 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
  tunnel 18113 127.0.0.1:8003 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
  tunnel 18114 127.0.0.1:8004 "-p 40051 -i /root/.ssh/china_rollout_ed25519 root@152.236.142.243"
  ports+=(18101 18111 18112 18113 18114)
fi
if [ "$transport" = zmq ]; then
  tunnel 29620 10.92.20.82:29620 china-trainer yes
  ports+=(29620)
fi

sleep 6
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN"
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN/$RUN_NAME"
sudo install -d -o 1000 -g 1000 "$(dirname "$orchestrator_output")"
sudo install -d -o 1000 -g 1000 "$orchestrator_output"
for p in "${ports[@]}"; do
  ss -tln 2>/dev/null | grep -q "127.0.0.1:$p" && echo "tunnel :$p up" || echo "tunnel :$p DOWN"
done

if [ ! -r "$env_config_file" ]; then
  echo "ERROR: environment server config not found: $env_config_file" >&2
  exit 1
fi

sudo docker rm -f "$orchestrator_container" "$env_container" >/dev/null 2>&1 || true
sudo docker run -d --name "$env_container" \
  --network host --restart no --user root \
  --env-file /data/trinity/training.env \
  -e VLLM_API_KEY=unused \
  -e PYTHONPATH="$deployment/env":/app/src \
  -e HF_HOME=/data/glm52-rl/hf-cache \
  -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -v /data/glm52-rl:/data/glm52-rl \
  -v "$model_volume":/models/glm52-fp8:ro \
  -v "$deployment/source/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
  -v "$deployment/source/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
  -v "$deployment/source/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
  -v "$deployment/source/deps/verifiers/verifiers/v1/harnesses/null/harness.py":/app/deps/verifiers/verifiers/v1/harnesses/null/harness.py:ro \
  -v "$deployment/patches/verifiers/v1-serialize-env-replies.patch":/tmp/v1-serialize-env-replies.patch:ro \
  --entrypoint /bin/sh "$image" \
  -c 'cd /app/deps/verifiers && patch -s -p1 < /tmp/v1-serialize-env-replies.patch && exec runuser -u appuser -- /app/.venv/bin/env-server "$@"' \
  sh @ "$env_config_file" >/dev/null
sudo docker run -d --name "$orchestrator_container" \
  --network host --restart no --user root \
  --env-file /data/trinity/training.env \
  -e VLLM_API_KEY=unused \
  -e PYTHONPATH="$deployment/env":/app/src \
  -e HF_HOME=/data/glm52-rl/hf-cache \
  -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -v /data/glm52-rl:/data/glm52-rl \
  -v "$model_volume":/models/glm52-fp8:ro \
  -v /data/muse-glimmer/model:/data/muse-glimmer/model:ro \
  -v "$deployment/source/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
  -v "$deployment/source/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
  -v "$deployment/source/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
  -v "$deployment/patches/verifiers/v1-serialize-env-replies.patch":/tmp/v1-serialize-env-replies.patch:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/algo/grpo.py":/app/src/prime_rl/orchestrator/algo/grpo.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/dispatcher.py":/app/src/prime_rl/orchestrator/dispatcher.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/orchestrator.py":/app/src/prime_rl/orchestrator/orchestrator.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/train_sink.py":/app/src/prime_rl/orchestrator/train_sink.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
  -v "$deployment/source/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
  -v "$deployment/source/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
  -v "$deployment/source/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  --entrypoint /bin/sh "$image" \
  -c 'cd /app/deps/verifiers && patch -s -p1 < /tmp/v1-serialize-env-replies.patch && exec runuser -u appuser -- /app/.venv/bin/orchestrator "$@"' \
  sh @ "$config_file"
echo "environment and orchestrator started on $(hostname) with $(basename "$config_file")"

# The China-side supervisor refuses to restart a failed trainer because the
# optimizer has no resumable checkpoint. Stop rollout production here as soon
# as any trainer worker is gone, so the run cannot accumulate unusable batches.
pkill -f '[t]rainer_failure_watchdog.sh' 2>/dev/null || true
nohup setsid env RUN="$RUN" RUN_NAME="$RUN_NAME" \
  prime_trainer_container="$trainer_container" \
  prime_orchestrator_container="$orchestrator_container" \
  prime_env_container="$env_container" \
  bash "$deployment/scripts/h200/trainer_failure_watchdog.sh" \
  >>"/data/glm52-rl/trainer-failure-watchdog-$RUN.log" 2>&1 </dev/null &
trainer_watchdog_pid=$!
sleep 1
if ! kill -0 "$trainer_watchdog_pid" 2>/dev/null; then
  echo "ERROR: trainer failure watchdog exited during startup" >&2
  sudo docker rm -f "$orchestrator_container" "$env_container" >/dev/null 2>&1 || true
  exit 1
fi
echo "trainer failure watchdog started (pid $trainer_watchdog_pid)"

ctl=$orchestrator_output/control/orch.toml
if [ "$transport" = filesystem ]; then
  publisher_dir="$orchestrator_output"
  publisher_pid_file="$publisher_dir/.batch-sync.pid"
  publisher_lock="$publisher_dir/.batch-sync.lock"
  if [ -r "$publisher_pid_file" ]; then
    old_publisher_pid=$(cat "$publisher_pid_file")
    case "$old_publisher_pid" in
      ''|*[!0-9]*) ;;
      *)
        # The publisher runs under setsid. Stop the complete group so an rsync
        # child cannot inherit fd 9 and keep the flock after its parent exits.
        kill -- "-$old_publisher_pid" 2>/dev/null || kill "$old_publisher_pid" 2>/dev/null || true
        ;;
    esac
  fi
  pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null || true
  # A terminated publisher can remove its pid file before a timeout/rsync/ssh
  # child exits. That child inherits fd 9 and keeps the lock, so a normal
  # process-name kill cannot find it. Stop only process groups that still hold
  # this exact run's lock before starting the replacement publisher.
  if ! flock -n "$publisher_lock" -c true && command -v lsof >/dev/null 2>&1; then
    lock_holders=$(lsof -t -- "$publisher_lock" 2>/dev/null | sort -u || true)
    for holder in $lock_holders; do
      holder_pgid=$(ps -o pgid= -p "$holder" 2>/dev/null | tr -d ' ')
      case "$holder_pgid" in
        ''|*[!0-9]*) ;;
        *) kill -- "-$holder_pgid" 2>/dev/null || kill "$holder" 2>/dev/null || true ;;
      esac
    done
  fi
  publisher_lock_ready=0
  for _ in $(seq 1 30); do
    if flock -n "$publisher_lock" -c true; then
      publisher_lock_ready=1
      break
    fi
    sleep 1
  done
  if [ "$publisher_lock_ready" != 1 ]; then
    echo "ERROR: old durable batch publisher still owns $publisher_lock" >&2
    exit 1
  fi
  nohup setsid env SOURCE_RUN_DIR="$orchestrator_output" DEST_RUN_DIR="$orchestrator_output" \
    DEST_HOST=china-trainer \
    ADAPTER_SOURCE_BASE="http://127.0.0.1:18100/$RUN/$RUN_NAME" \
    ADAPTER_ADMIN_URLS="$adapter_admin_urls" \
    "$deployment/scripts/h200/sync_rollout_batches.sh" \
    >>"/data/glm52-rl/batch-sync-$RUN.log" 2>&1 </dev/null &
  publisher_pid=$!
  sleep 1
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    echo "ERROR: durable batch publisher exited during startup" >&2
    exit 1
  fi
  echo "durable batch publisher started (pid $publisher_pid)"
  exit
fi

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
