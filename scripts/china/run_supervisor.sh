#!/usr/bin/env bash
# Fail the complete run closed if any trainer worker exits. Runs on n82.
#
# This trainer has no resumable optimizer checkpoint. An automatic container
# restart would silently create a new optimizer while the orchestrator keeps
# advancing the old run. Stop every surviving worker instead and require an
# explicit fresh run name.
set -uo pipefail

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
TRAINER_NODES=${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}
INTERVAL=${INTERVAL:-30}
STARTUP_GRACE=${STARTUP_GRACE:-900}
MAX_MISSES=${MAX_MISSES:-2}
TRAINER_CT=glm52-decensor-trainer
ORCHESTRATOR_CT=glm52-decensor-orchestrator
STOP_LOCAL_ORCHESTRATOR=${STOP_LOCAL_ORCHESTRATOR:-0}
failure_marker="/data/glm52-rl/runs/$RUN/$RUN_NAME/control/trainer-failed"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_failures() {
  local id host state failures=""
  for id in $TRAINER_NODES; do
    if [ "$id" = 82 ]; then
      state=$(sudo docker ps --filter name="$TRAINER_CT" --format up 2>/dev/null | head -1)
    else
      host="10.92.20.$id"
      state=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$host" \
        "sudo docker ps --filter name=${TRAINER_CT} --format up 2>/dev/null | head -1" \
        2>/dev/null || true)
    fi
    if [ "$state" != up ]; then
      failures="$failures n$id"
    fi
  done
  printf '%s\n' "${failures# }"
}

stop_trainers() {
  local id host
  for id in $TRAINER_NODES; do
    host="10.92.20.$id"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$host" \
      "sudo docker rm -f $TRAINER_CT >/dev/null 2>&1 || true" \
      >/dev/null 2>&1 &
  done
  wait
}

stop_local_orchestrator() {
  if [ "$STOP_LOCAL_ORCHESTRATOR" = 1 ]; then
    sudo docker rm -f "$ORCHESTRATOR_CT" >/dev/null 2>&1 || true
    log "stopped local orchestrator after trainer failure"
  fi
}

log "fail-closed run supervisor started (RUN=$RUN interval=${INTERVAL}s)"
deadline=$(( $(date +%s) + STARTUP_GRACE ))
while true; do
  failed=$(trainer_failures)
  if [ -z "$failed" ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "FATAL: trainer cluster did not become complete within ${STARTUP_GRACE}s; missing:$failed"
    mkdir -p "$(dirname "$failure_marker")"
    printf '%s missing:%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failed" >"$failure_marker"
    stop_trainers
    stop_local_orchestrator
    exit 1
  fi
  log "startup: waiting for trainer nodes:$failed"
  sleep 15
done

rm -f "$failure_marker"
log "all trainer containers are up"
misses=0
while true; do
  failed=$(trainer_failures)
  if [ -n "$failed" ]; then
    misses=$((misses + 1))
    log "trainer probe miss $misses/$MAX_MISSES:$failed"
  else
    misses=0
    log "healthy: all trainer containers are up"
  fi
  if [ "$misses" -ge "$MAX_MISSES" ]; then
    log "FATAL: trainer worker loss confirmed:$failed; stopping every trainer and refusing automatic restart"
    mkdir -p "$(dirname "$failure_marker")"
    printf '%s missing:%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failed" >"$failure_marker"
    stop_trainers
    stop_local_orchestrator
    exit 1
  fi
  sleep "$INTERVAL"
done
