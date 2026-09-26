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
STATUS_FILE=${STATUS_FILE:-"$run_dir/.trainer-health-status"}
LEASE_MAX_AGE=${LEASE_MAX_AGE:-90}
WAIT_FOR_INITIAL_HEALTH=${WAIT_FOR_INITIAL_HEALTH:-0}

case "$WAIT_FOR_INITIAL_HEALTH" in
  0|1) ;;
  *) echo "WAIT_FOR_INITIAL_HEALTH must be 0 or 1" >&2; exit 2 ;;
esac

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

health_status() {
  local state epoch extra owner age
  [ -f "$STATUS_FILE" ] && [ ! -L "$STATUS_FILE" ] || return 1
  owner=$(stat -c '%u:%g:%a' "$STATUS_FILE" 2>/dev/null) || return 1
  [ "$owner" = "1000:1000:600" ] || return 1
  read -r state epoch extra <"$STATUS_FILE" || return 1
  [ -z "${extra:-}" ] || return 1
  case "$state" in
    healthy|missing|unknown) ;;
    *) return 1 ;;
  esac
  case "$epoch" in ''|*[!0-9]*) return 1 ;; esac
  age=$(( $(date +%s) - epoch ))
  [ "$age" -ge 0 ] && [ "$age" -le "$LEASE_MAX_AGE" ] || return 1
  printf '%s %s\n' "$state" "$epoch"
}

log "trainer failure watchdog started (RUN=$RUN mode=$MODE interval=${INTERVAL}s)"
deadline=$(( $(date +%s) + STARTUP_GRACE ))
if [ "$MODE" = lease ]; then
  missing_epoch=
  missing_count=0
  armed=$((1 - WAIT_FOR_INITIAL_HEALTH))
  startup_deadline=$(( $(date +%s) + STARTUP_GRACE ))
  while true; do
    snapshot=$(health_status 2>/dev/null || true)
    if [ "$armed" = 0 ]; then
      if [ -n "$snapshot" ]; then
        read -r state epoch <<<"$snapshot"
        if [ "$state" = healthy ] && lease_fresh; then
          armed=1
          missing_epoch=
          missing_count=0
          log "trainer health watchdog armed after fresh healthy status and lease"
        elif [ "$(date +%s)" -ge "$startup_deadline" ]; then
          log "FATAL: trainer health did not become healthy during startup grace; stopping orchestrator"
          stop_edge
          exit 1
        else
          log "startup: waiting for fresh healthy trainer status and lease (state=$state)"
        fi
      elif [ "$(date +%s)" -ge "$startup_deadline" ]; then
        log "FATAL: trainer health status stayed unknown during startup grace; stopping orchestrator"
        stop_edge
        exit 1
      else
        log "startup: trainer health status unknown; waiting for fresh healthy status"
      fi
      sleep "$INTERVAL"
      continue
    fi
    if [ -z "$snapshot" ]; then
      missing_epoch=
      missing_count=0
      log "trainer health status unknown; no failure recorded"
    else
      read -r state epoch <<<"$snapshot"
      case "$state" in
        healthy)
          missing_epoch=
          missing_count=0
          if lease_fresh; then
            log "healthy: trainer health lease is fresh"
          else
            log "trainer health lease is stale; status is unknown"
          fi
          ;;
        unknown)
          missing_epoch=
          missing_count=0
          log "trainer health probe unknown; no failure recorded"
          ;;
        missing)
          if [ -z "$missing_epoch" ] || [ "$epoch" -gt "$missing_epoch" ]; then
            missing_epoch=$epoch
            missing_count=$((missing_count + 1))
            log "confirmed trainer probe missing epoch $epoch ($missing_count/$MAX_MISSES)"
          elif [ "$epoch" = "$missing_epoch" ]; then
            log "repeated trainer probe missing epoch $epoch ($missing_count/$MAX_MISSES)"
          else
            log "ignored backward trainer probe missing epoch $epoch (latest $missing_epoch)"
          fi
          if [ "$missing_count" -ge "$MAX_MISSES" ]; then
            log "FATAL: trainer probes confirmed missing; stopping orchestrator"
            stop_edge
            exit 1
          fi
          ;;
      esac
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
