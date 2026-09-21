#!/usr/bin/env bash
# Stop edge rollout production when any trainer worker is absent.
set -uo pipefail

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
MODE=${MODE:-ssh}
TRAINER_ADDRESSES=${TRAINER_ADDRESSES:-}
INTERVAL=${INTERVAL:-30}
STARTUP_GRACE=${STARTUP_GRACE:-900}
MAX_MISSES=${MAX_MISSES:-2}
TRAINER_CT=${TRAINER_CT:?set TRAINER_CT}
ORCH_CT=${ORCH_CT:?set ORCH_CT}
run_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"
LEASE_FILE=${LEASE_FILE:-"$run_dir/.trainer-health-lease"}
LEASE_MAX_AGE=${LEASE_MAX_AGE:-90}

case "$MODE" in
  ssh) [ -n "$TRAINER_ADDRESSES" ] || { echo "set TRAINER_ADDRESSES" >&2; exit 2; } ;;
  lease) ;;
  *) echo "MODE must be ssh or lease" >&2; exit 2 ;;
esac

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_failures() {
  local output
  if ! output=$(ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "failures=''; for host in $TRAINER_ADDRESSES; do
       id=\${host##*.}
       state=\$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \"\$host\" \\
         \"sudo docker ps --filter name=^${TRAINER_CT}$ --format up 2>/dev/null | head -1\" \\
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
  name=$(sudo docker inspect --format '{{.Name}}' "$ORCH_CT" 2>/dev/null || true)
  [ -z "$name" ] || [ "$name" = "/$ORCH_CT" ] || return 1
  [ -z "$name" ] || sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
}

lease_fresh() {
  local value age
  value=$(cat "$LEASE_FILE" 2>/dev/null || true)
  case "$value" in ''|*[!0-9]*) return 1 ;; esac
  age=$(( $(date +%s) - value ))
  [ "$age" -ge 0 ] && [ "$age" -le "$LEASE_MAX_AGE" ]
}

log "trainer failure watchdog started (RUN=$RUN mode=$MODE interval=${INTERVAL}s)"
deadline=$(( $(date +%s) + STARTUP_GRACE ))
if [ "$MODE" = lease ]; then
  while ! lease_fresh; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "FATAL: trainer health lease did not become fresh"
      stop_edge
      exit 1
    fi
    sleep 15
  done
  misses=0
  while true; do
    if lease_fresh; then
      misses=0
      log "healthy: trainer health lease is fresh"
    else
      misses=$((misses + 1))
      log "trainer lease miss $misses/$MAX_MISSES"
    fi
    if [ "$misses" -ge "$MAX_MISSES" ]; then
      log "FATAL: trainer health lease is stale; stopping orchestrator"
      stop_edge
      exit 1
    fi
    sleep "$INTERVAL"
  done
fi

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
