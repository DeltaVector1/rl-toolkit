#!/usr/bin/env bash
# Stop edge rollout production when any trainer worker is absent.
set -uo pipefail

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
TRAINER_NODES=${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}
INTERVAL=${INTERVAL:-30}
STARTUP_GRACE=${STARTUP_GRACE:-900}
MAX_MISSES=${MAX_MISSES:-2}
TRAINER_CT=glm52-decensor-trainer
ORCH_CT=glm52-decensor-orchestrator
run_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_failures() {
  local output
  if ! output=$(ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "failures=''; for id in $TRAINER_NODES; do
       host=10.92.20.\$id
       state=\$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \"\$host\" \\
         \"sudo docker ps --filter name=${TRAINER_CT} --format up 2>/dev/null | head -1\" \\
         2>/dev/null || true)
       [ \"\$state\" = up ] || failures=\"\$failures n\$id\"
     done
     printf '%s\\n' \"\${failures# }\"" 2>/dev/null); then
    printf '%s\n' probe-failed
    return 2
  fi
  printf '%s\n' "$output"
}

stop_edge() {
  local publisher_pid=""
  if [ -r "$run_dir/.batch-sync.pid" ]; then
    publisher_pid=$(cat "$run_dir/.batch-sync.pid")
  fi
  case "$publisher_pid" in
    ''|*[!0-9]*) ;;
    *) kill -- "-$publisher_pid" 2>/dev/null || kill "$publisher_pid" 2>/dev/null || true ;;
  esac
  pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null || true
  sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
}

log "trainer failure watchdog started (RUN=$RUN interval=${INTERVAL}s)"
deadline=$(( $(date +%s) + STARTUP_GRACE ))
while true; do
  probe_status=0
  failed=$(trainer_failures) || probe_status=$?
  if [ "$probe_status" -eq 0 ] && [ -z "$failed" ]; then
    break
  fi
  if [ "$probe_status" -ne 0 ]; then
    log "startup: trainer state unknown; health query failed"
  elif [ "$(date +%s)" -ge "$deadline" ]; then
    log "FATAL: trainer cluster did not become complete; missing:$failed"
    stop_edge
    exit 1
  else
    log "startup: waiting for trainer nodes:$failed"
  fi
  sleep 15
done

log "all trainer containers are up"
misses=0
while true; do
  probe_status=0
  failed=$(trainer_failures) || probe_status=$?
  if [ "$probe_status" -ne 0 ]; then
    misses=0
    log "trainer state unknown; health query failed; no failure recorded"
  elif [ -n "$failed" ]; then
    misses=$((misses + 1))
    log "trainer probe miss $misses/$MAX_MISSES:$failed"
  else
    misses=0
    log "healthy: all trainer containers are up"
  fi
  if [ "$misses" -ge "$MAX_MISSES" ]; then
    log "FATAL: trainer cluster incomplete; stopping orchestrator and durable batch publisher"
    stop_edge
    exit 1
  fi
  sleep "$INTERVAL"
done
