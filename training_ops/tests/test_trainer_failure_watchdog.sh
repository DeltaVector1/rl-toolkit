#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
watchdog="$repo/training_ops/runtime/h200/trainer_failure_watchdog.sh"
tmp=$(mktemp -d)
trap 'if [ -n "${watchdog_pid:-}" ]; then kill "$watchdog_pid" 2>/dev/null || true; wait "$watchdog_pid" 2>/dev/null || true; fi; rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
cat >"$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  inspect) printf '/orchestrator\n' ;;
  rm) : >"${DOCKER_STOP:?}" ;;
  *) exit 1 ;;
esac
EOF
cat >"$tmp/bin/pkill" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$tmp/bin"/*

status_file="$tmp/health-status"
lease_file="$tmp/health-lease"
stop_file="$tmp/docker-stop"
log_file="$tmp/watchdog.log"
export PATH="$tmp/bin:$PATH" DOCKER_STOP="$stop_file"

write_status() {
  printf '%s %s\n' "$1" "$2" >"$status_file"
  chmod 600 "$status_file"
  chown 1000:1000 "$status_file"
}

start_watchdog() {
  RUN=fixture RUN_NAME=run MODE=lease LEASE_FILE="$lease_file" STATUS_FILE="$status_file" \
    TRAINER_CT=trainer ORCH_CT=orchestrator INTERVAL=0.5 LEASE_MAX_AGE=30 MAX_MISSES=2 \
    STARTUP_GRACE="${STARTUP_GRACE:-900}" WAIT_FOR_INITIAL_HEALTH="${WAIT_FOR_INITIAL_HEALTH:-0}" \
    bash "$watchdog" >"$log_file" 2>&1 &
  watchdog_pid=$!
}

wait_for_stop() {
  for _ in $(seq 1 40); do
    [ -e "$stop_file" ] && return 0
    sleep 0.1
  done
  return 1
}

# The startup flag is strict; repair/runtime watchdogs use the armed default.
if WAIT_FOR_INITIAL_HEALTH=2 RUN=fixture RUN_NAME=run MODE=lease \
  LEASE_FILE="$lease_file" STATUS_FILE="$status_file" TRAINER_CT=trainer ORCH_CT=orchestrator \
  bash "$watchdog" >/dev/null 2>&1; then
  echo "invalid startup flag was accepted" >&2
  exit 1
fi

# Cold start waits through grace and stops if no fresh healthy status appears.
WAIT_FOR_INITIAL_HEALTH=1 STARTUP_GRACE=1 start_watchdog
wait_for_stop
set +e
wait "$watchdog_pid"
watchdog_status=$?
set -e
test "$watchdog_status" -eq 1
test -e "$stop_file"
watchdog_pid=
WAIT_FOR_INITIAL_HEALTH=0 STARTUP_GRACE=900
rm -f "$status_file" "$stop_file"

# A fresh healthy status and lease arm cold start without stopping the edge.
now=$(date +%s)
write_status healthy "$now"
printf '%s\n' "$now" >"$lease_file"
WAIT_FOR_INITIAL_HEALTH=1 STARTUP_GRACE=3 start_watchdog
sleep 1.2
test ! -e "$stop_file"
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
rm -f "$status_file" "$stop_file"

# A stale lease without a fresh, authenticated status is unknown.
printf '1\n' >"$lease_file"
chmod 600 "$lease_file"
start_watchdog
sleep 2.2
test ! -e "$stop_file"
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
rm -f "$status_file" "$stop_file"

# Two distinct fresh missing epochs are a confirmed failure.
epoch_base=$(date +%s)
write_status missing "$((epoch_base - 4))"
start_watchdog
sleep 1.2
write_status missing "$((epoch_base - 3))"
wait_for_stop
set +e
wait "$watchdog_pid"
watchdog_status=$?
set -e
test "$watchdog_status" -eq 1
test -e "$stop_file"
watchdog_pid=
rm -f "$status_file" "$stop_file"

# Repeated and backward missing epochs do not count; unknown remains safe.
now=$(date +%s)
write_status healthy "$now"
printf '%s\n' "$now" >"$lease_file"
start_watchdog
sleep 0.7
write_status missing "$now"
sleep 0.7
write_status missing "$((now - 1))"
sleep 0.7
write_status unknown "$now"
sleep 0.7
test ! -e "$stop_file"
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
rm -f "$status_file" "$stop_file"

# A healthy status with a fresh lease resets a prior missing epoch.
write_status missing "$(date +%s)"
start_watchdog
sleep 0.2
now=$(date +%s)
write_status healthy "$now"
printf '%s\n' "$now" >"$lease_file"
sleep 1.5
test ! -e "$stop_file"
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=

printf 'trainer failure watchdog tests passed\n'
