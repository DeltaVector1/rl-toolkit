#!/usr/bin/env bash
# Stop the run if a trainer leaves.
set -uo pipefail

run=${run:?set run}
run_name=${run_name:?set run_name}
runs_path=${runs_path:?set runs_path}
trainer_hosts=${trainer_hosts:?set trainer_hosts}
local_host=${local_host:?set local_host}
trainer_container_name=${trainer_container_name:?set trainer_container_name}
orchestrator_container_name=${orchestrator_container_name:?set orchestrator_container_name}
interval=${interval:-30}
startup_grace=${startup_grace:-900}
max_misses=${max_misses:-2}
stop_local_orchestrator=${stop_local_orchestrator:-0}
failure_marker="$runs_path/$run/$run_name/control/trainer-failed"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_failures() {
  local host state failures=""
  for host in $trainer_hosts; do
    if [ "$host" = "$local_host" ]; then
      state=$(sudo docker ps --filter name="$trainer_container_name" --format up 2>/dev/null | head -1)
    else
      state=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$host" \
        "sudo docker ps --filter name='$trainer_container_name' --format up 2>/dev/null | head -1" \
        2>/dev/null || true)
    fi
    if [ "$state" != up ]; then
      failures="$failures $host"
    fi
  done
  printf '%s\n' "${failures# }"
}

stop_trainers() {
  local host
  for host in $trainer_hosts; do
    if [ "$host" = "$local_host" ]; then
      sudo docker rm -f "$trainer_container_name" >/dev/null 2>&1 || true
    else
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$host" \
        "sudo docker rm -f '$trainer_container_name' >/dev/null 2>&1 || true" \
        >/dev/null 2>&1 &
    fi
  done
  wait
}

stop_local_orchestrator() {
  if [ "$stop_local_orchestrator" = 1 ]; then
    sudo docker rm -f "$orchestrator_container_name" >/dev/null 2>&1 || true
    log "stopped local orchestrator after trainer failure"
  fi
}

log "run supervisor started (run=$run interval=${interval}s)"
deadline=$(( $(date +%s) + startup_grace ))
while true; do
  failed=$(trainer_failures)
  if [ -z "$failed" ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "FATAL: trainer cluster did not become complete within ${startup_grace}s; missing:$failed"
    mkdir -p "$(dirname "$failure_marker")"
    printf '%s missing:%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failed" >"$failure_marker"
    stop_trainers
    stop_local_orchestrator
    exit 1
  fi
  log "startup: waiting for trainer hosts:$failed"
  sleep 15
done

rm -f "$failure_marker"
log "all trainer containers are up"
misses=0
while true; do
  failed=$(trainer_failures)
  if [ -n "$failed" ]; then
    misses=$((misses + 1))
    log "trainer probe miss $misses/$max_misses:$failed"
  else
    misses=0
    log "healthy: all trainer containers are up"
  fi
  if [ "$misses" -ge "$max_misses" ]; then
    log "FATAL: trainer worker loss confirmed:$failed; stopping every trainer and refusing automatic restart"
    mkdir -p "$(dirname "$failure_marker")"
    printf '%s missing:%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failed" >"$failure_marker"
    stop_trainers
    stop_local_orchestrator
    exit 1
  fi
  sleep "$interval"
done
