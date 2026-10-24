#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
watchdog="$repo/training_ops/runtime/h200/trainer_failure_watchdog.sh"
tmp=$(mktemp -d)
publisher_pid=
cleanup() {
  if [ -n "${publisher_pid:-}" ]; then
    kill "$publisher_pid" 2>/dev/null || true
    wait "$publisher_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

mkdir -p "$tmp/bin" "$tmp/run"
publisher_script="$tmp/sync_rollout_batches.sh"
cat >"$publisher_script" <<'EOF'
#!/usr/bin/env bash
while :; do sleep 1; done
EOF
chmod +x "$publisher_script"

cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
cat >"$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  inspect)
    count=$(cat "${DOCKER_STATE:?}")
    count=$((count + 1))
    printf '%s\n' "$count" >"$DOCKER_STATE"
    format="${3:-}"
    if [ "$count" -le 3 ]; then
      case "$format" in
        *'.Id'*) printf 'old-container-id\n' ;;
        *'.Name'*) printf '/orchestrator\n' ;;
        *'Config.Labels'*) printf 'old-launch-identity\n' ;;
        *) exit 1 ;;
      esac
    else
      case "$format" in
        *'.Id'*) printf 'new-container-id\n' ;;
        *'.Name'*) printf '/orchestrator\n' ;;
        *'Config.Labels'*) printf 'new-launch-identity\n' ;;
        *) exit 1 ;;
      esac
    fi
    ;;
  rm)
    : >"${DOCKER_REMOVED:?}"
    ;;
  *) exit 1 ;;
esac
EOF
cat >"$tmp/bin/pkill" <<'EOF'
#!/usr/bin/env bash
: >"${PKILL_CALLED:?}"
EOF
chmod +x "$tmp/bin"/*

setsid env SOURCE_RUN_DIR="$tmp/run" "$publisher_script" >/dev/null 2>&1 &
publisher_pid=$!
printf '%s\n' "$publisher_pid" >"$tmp/run/.batch-sync.pid"
printf '0\n' >"$tmp/docker-state"
printf 'missing %s\n' "$(date +%s)" >"$tmp/status"
chmod 600 "$tmp/status"
chown 1000:1000 "$tmp/status"
printf '%s\n' "$(date +%s)" >"$tmp/lease"

set +e
PATH="$tmp/bin:$PATH" DOCKER_STATE="$tmp/docker-state" DOCKER_REMOVED="$tmp/docker-removed" \
  PKILL_CALLED="$tmp/pkill-called" RUN=fixture RUN_NAME=run RUN_DIR="$tmp/run" MODE=lease \
  LEASE_FILE="$tmp/lease" STATUS_FILE="$tmp/status" TRAINER_CT=trainer ORCH_CT=orchestrator \
  ORCH_CONTAINER_ID=old-container-id ORCH_LAUNCH_IDENTITY=old-launch-identity \
  PUBLISHER_SCRIPT="$publisher_script" INTERVAL=0.1 LEASE_MAX_AGE=30 MAX_MISSES=1 \
  STARTUP_GRACE=5 WAIT_FOR_INITIAL_HEALTH=0 bash "$watchdog" >"$tmp/watchdog.log" 2>&1
watchdog_status=$?
set -e
test "$watchdog_status" -eq 1
test ! -e "$tmp/docker-removed"
test ! -e "$tmp/pkill-called"
kill -0 "$publisher_pid"

printf 'trainer failure watchdog race tests passed\n'
