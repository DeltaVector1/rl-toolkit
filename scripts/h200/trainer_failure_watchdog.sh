#!/usr/bin/env bash
# Stop edge rollout production when any trainer worker is absent.
set -uo pipefail

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
trainer_nodes=${trainer_nodes:-"82 76 77 78 80 83 84 85"}
interval=${interval:-30}
startup_grace=${startup_grace:-900}
max_misses=${max_misses:-2}
max_unknown=${max_unknown:-20}
trainer_container=${prime_trainer_container:-glm52-optims-trainer}
orchestrator_container=${prime_orchestrator_container:-glm52-optims-orchestrator}
environment_container=${prime_env_container:-glm52-optims-env-train}
run_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_failures() {
  local output
  if ! output=$(ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "failures=''; unknown=''; for id in $trainer_nodes; do
       host=10.92.20.\$id
       state=\$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \"\$host\" \\
         \"sudo docker ps --filter name=${trainer_container} --format up 2>/dev/null | head -1\" \\
         2>/dev/null); rc=\$?
       if [ \"\$rc\" -ne 0 ]; then
         unknown=\"\$unknown n\$id\"
       elif [ \"\$state\" != up ]; then
         failures=\"\$failures n\$id\"
       fi
     done
     printf 'missing:%s\\nunknown:%s\\n' \"\${failures# }\" \"\${unknown# }\"" 2>/dev/null); then
    printf '%s\n' 'missing:' 'unknown:china-trainer'
    return
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
  sudo docker rm -f "$orchestrator_container" "$environment_container" >/dev/null 2>&1 || true
}

log "trainer failure watchdog started (RUN=$RUN interval=${interval}s)"
deadline=$(( $(date +%s) + startup_grace ))
while true; do
  probe=$(trainer_failures || true)
  failed=$(printf '%s\n' "$probe" | sed -n 's/^missing://p')
  unknown=$(printf '%s\n' "$probe" | sed -n 's/^unknown://p')
  if [ -z "$failed" ] && [ -z "$unknown" ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "FATAL: trainer cluster did not become complete; missing:${failed:-none} unknown:${unknown:-none}"
    stop_edge
    exit 1
  fi
  log "startup: waiting for trainer nodes; missing:${failed:-none} unknown:${unknown:-none}"
  sleep 15
done

log "all trainer containers are up"
misses=0
unknown_misses=0
while true; do
  probe=$(trainer_failures || true)
  failed=$(printf '%s\n' "$probe" | sed -n 's/^missing://p')
  unknown=$(printf '%s\n' "$probe" | sed -n 's/^unknown://p')
  if [ -n "$failed" ]; then
    misses=$((misses + 1))
    unknown_misses=0
    log "trainer confirmed missing $misses/$max_misses:$failed"
  elif [ -n "$unknown" ]; then
    unknown_misses=$((unknown_misses + 1))
    log "trainer check unavailable $unknown_misses/$max_unknown:$unknown"
  else
    misses=0
    unknown_misses=0
    log "healthy: all trainer containers are up"
  fi
  if [ "$misses" -ge "$max_misses" ]; then
    log "FATAL: trainer cluster incomplete; stopping orchestrator and durable batch publisher"
    stop_edge
    exit 1
  fi
  if [ "$unknown_misses" -ge "$max_unknown" ]; then
    log "FATAL: trainer state could not be checked for 10 minutes; stopping rollout production"
    stop_edge
    exit 1
  fi
  sleep "$interval"
done
