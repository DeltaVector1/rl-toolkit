#!/usr/bin/env bash
# Restart the local vLLM engine when a real completion fails.
set -uo pipefail

PORT=${PORT:-8000}
MODEL=${MODEL:-/data/glm52-rl/model-fp8}
INTERVAL=${INTERVAL:-60}
TIMEOUT=${TIMEOUT:-15}
FAILS_BEFORE_RESTART=${FAILS_BEFORE_RESTART:-2}
# A deliberate engine replacement can leave the base-model endpoint absent
# while the checkpoint loads. The repair command starts a fresh watchdog with
# this grace so stale misses from the old engine cannot trigger a second
# replacement during that load.
STARTUP_GRACE=${STARTUP_GRACE:-0}
# A LoRA collective can temporarily stop token progress while requests remain
# registered in the engine. Two such stalls at policy updates on 2026-08-11
# finished on the peer engine but made this watchdog discard h200b after only
# two minutes. Give a busy engine a longer bounded grace. A dead or empty
# EngineCore still uses FAILS_BEFORE_RESTART.
BUSY_STALL_FAILS_BEFORE_RESTART=${BUSY_STALL_FAILS_BEFORE_RESTART:-5}
# A blocked LoRA collective can report zero running requests while the complete
# engine remains resident. This happened at x64p policy v3: h200b held
# 132155 MiB but the ordinary two-probe limit restarted it. Treat a resident
# engine like a busy one for the bounded grace. Empty dead engines use the
# ordinary limit and still restart quickly.
RESIDENT_ENGINE_MIN_MIB=${RESIDENT_ENGINE_MIN_MIB:-65536}
# Tests can replace these commands without changing this file.
# against a fake launcher instead of re-implementing its decisions.
START=${START:-/data/glm52-rl/deployment/.agents/skills/training-ops/runtime/h200/start_inference.sh}
ENGINE_CONTAINER=${ENGINE_CONTAINER:-glm52-decensor-h200}
RELOAD_TRIES=${RELOAD_TRIES:-40}
RELOAD_WAIT=${RELOAD_WAIT:-30}
misses=0
last_generation_tokens=-1

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "engine watchdog started (port=$PORT interval=${INTERVAL}s timeout=${TIMEOUT}s)"
if [ "$STARTUP_GRACE" -gt 0 ]; then
  log "engine startup grace ${STARTUP_GRACE}s"
  sleep "$STARTUP_GRACE"
fi

while true; do
  # A tiny real generation. max_tokens=3 keeps it cheap; a hung EngineCore
  # never answers it even though /v1/models still returns 200.
  code=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" \
    -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":3}" 2>/dev/null)
  metrics=
  running=0
  generated=0
  if [ "$code" = "200" ]; then
    [ "$misses" -gt 0 ] && log "engine recovered (was $misses miss(es))"
    misses=0
  else
    metrics=$(curl -s -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null || true)
    running=$(printf '%s\n' "$metrics" | awk '/^vllm:num_requests_running[{]/ {sum += $NF} END {print sum + 0}')
    generated=$(printf '%s\n' "$metrics" | awk '/^vllm:generation_tokens_total[{]/ {sum += $NF} END {print sum + 0}')
    if [ "$running" != 0 ] &&
       { [ "$last_generation_tokens" = -1 ] || awk "BEGIN {exit !($generated > $last_generation_tokens)}"; }; then
      # A tiny completion can wait behind a saturated production queue for more
      # than TIMEOUT. A rising token counter proves the EngineCore is working;
      # restarting here discards every live rollout.
      log "engine busy but progressing (running=$running generated=$generated); probe deferred"
      misses=0
    else
      misses=$((misses + 1))
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
      gpu_mib=$(printf '%s\n' "$gpu" | awk '{print $1 + 0}')
      restart_after=$FAILS_BEFORE_RESTART
      if [ "$running" != 0 ] || [ "$gpu_mib" -ge "$RESIDENT_ENGINE_MIN_MIB" ]; then
        restart_after=$BUSY_STALL_FAILS_BEFORE_RESTART
      fi
      log "engine probe failed (code=${code:-none} running=$running generated=$generated gpu0=${gpu:-unknown}) miss=$misses/$restart_after"
    if [ "$misses" -ge "$restart_after" ]; then
      log "RESTARTING ENGINE (gpu0=${gpu:-unknown})"
      sudo docker rm -f "$ENGINE_CONTAINER" >/dev/null 2>&1
      if bash "$START" >/dev/null 2>&1; then
        log "start_inference.sh returned ok; waiting for engine to load"
      else
        log "WARNING start_inference.sh returned non-zero"
      fi
      # Loading the 745B checkpoint takes minutes; do not thrash it.
      for _ in $(seq 1 "$RELOAD_TRIES"); do
        sleep "$RELOAD_WAIT"
        c=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" \
          -X POST "http://127.0.0.1:$PORT/v1/completions" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":3}" 2>/dev/null)
        [ "$c" = "200" ] && { log "engine back up (generation verified)"; break; }
      done
      misses=0
      fi
    fi
  fi
  [ -n "$metrics" ] && last_generation_tokens=$generated
  sleep "$INTERVAL"
done
