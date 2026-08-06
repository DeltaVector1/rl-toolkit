#!/usr/bin/env bash
# Restart the local vLLM engine when it dies. Runs ON each H200 host.
#
# vLLM's EngineCore can die while the CONTAINER stays up: a worker RPC times
# out ("TimeoutError: RPC call to sample_tokens timed out", preceded by "No
# available shared memory broadcast block found in 60 seconds"), EngineCore
# raises EngineDeadError, the API server shuts down, and docker still reports
# the container as Up with the GPUs at 4 MiB. Observed 2026-08-06 03:13:30 on
# both hosts with 63 requests in flight; the run then sat dead for 2.6 hours.
#
# The tunnel watchdog on n82 CANNOT see this -- the SSH forward still accepts
# connections, so from n82 a dead engine is indistinguishable from a healthy
# one until the request fails. And n82 has no ssh access to these hosts, so it
# could not restart anything anyway. Hence a local self-healing loop.
#
# Neither container "Up" NOR /v1/models is a sufficient health signal. Observed
# 2026-08-06 06:05 on h200a: models=200 while gen=000, "Running: 64" with
# "throughput: 0.0" -- the API server was alive and answering while EngineCore
# was hung and generating nothing. Only a real completion proves the engine
# works, so that is what this probes.
set -uo pipefail

PORT=${PORT:-8000}
MODEL=${MODEL:-/data/glm52-rl/model-fp8}
INTERVAL=${INTERVAL:-60}
TIMEOUT=${TIMEOUT:-15}
FAILS_BEFORE_RESTART=${FAILS_BEFORE_RESTART:-2}
START=/data/glm52-rl/deployment/scripts/h200/start_inference.sh
misses=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "engine watchdog started (port=$PORT interval=${INTERVAL}s timeout=${TIMEOUT}s)"

while true; do
  # A tiny real generation. max_tokens=3 keeps it cheap; a hung EngineCore
  # never answers it even though /v1/models still returns 200.
  code=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" \
    -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":3}" 2>/dev/null)
  if [ "$code" = "200" ]; then
    [ "$misses" -gt 0 ] && log "engine recovered (was $misses miss(es))"
    misses=0
  else
    misses=$((misses + 1))
    log "engine probe failed (code=${code:-none}) miss=$misses/$FAILS_BEFORE_RESTART"
    if [ "$misses" -ge "$FAILS_BEFORE_RESTART" ]; then
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
      log "RESTARTING ENGINE (gpu0=${gpu:-unknown})"
      sudo docker rm -f glm52-decensor-h200 >/dev/null 2>&1
      if bash "$START" >/dev/null 2>&1; then
        log "start_inference.sh returned ok; waiting for engine to load"
      else
        log "WARNING start_inference.sh returned non-zero"
      fi
      # Loading the 745B checkpoint takes minutes; do not thrash it.
      for _ in $(seq 1 40); do
        sleep 30
        c=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" \
          -X POST "http://127.0.0.1:$PORT/v1/completions" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":3}" 2>/dev/null)
        [ "$c" = "200" ] && { log "engine back up (generation verified)"; break; }
      done
      misses=0
    fi
  fi
  sleep "$INTERVAL"
done
