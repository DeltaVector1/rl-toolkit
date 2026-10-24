#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
watchdog="$repo/training_ops/runtime/h200/trainer_failure_watchdog.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
cat >"$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  inspect)
    format="${3:-}"
    case "$format" in
      *'.Id'*) printf 'fresh-container-id\n' ;;
      *'.Name'*) printf '/orchestrator\n' ;;
      *'Config.Labels'*) printf 'fresh-launch-identity\n' ;;
      *) exit 1 ;;
    esac
    ;;
  rm)
    printf '%s\n' "${3:-}" >"${DOCKER_REMOVED:?}"
    ;;
  *) exit 1 ;;
esac
EOF
cat >"$tmp/bin/pkill" <<'EOF'
#!/usr/bin/env bash
: >"${PKILL_CALLED:?}"
EOF
chmod +x "$tmp/bin"/*

run_case() {
  local expected_id=$1 expected_identity=$2
  local status_file="$tmp/status-$expected_id" lease_file="$tmp/lease-$expected_id"
  local removed_file="$tmp/removed-$expected_id" pkill_file="$tmp/pkill-$expected_id"
  local log_file="$tmp/log-$expected_id"
  printf 'missing %s\n' "$(date +%s)" >"$status_file"
  chmod 600 "$status_file"
  chown 1000:1000 "$status_file"
  printf '%s\n' "$(date +%s)" >"$lease_file"
  set +e
  PATH="$tmp/bin:$PATH" DOCKER_REMOVED="$removed_file" PKILL_CALLED="$pkill_file" \
    RUN=fixture RUN_NAME=run MODE=lease LEASE_FILE="$lease_file" STATUS_FILE="$status_file" \
    TRAINER_CT=trainer ORCH_CT=orchestrator ORCH_CONTAINER_ID="$expected_id" \
    ORCH_LAUNCH_IDENTITY="$expected_identity" INTERVAL=0.1 LEASE_MAX_AGE=30 \
    MAX_MISSES=1 STARTUP_GRACE=5 WAIT_FOR_INITIAL_HEALTH=0 \
    bash "$watchdog" >"$log_file" 2>&1
  local status=$?
  set -e
  test "$status" -eq 1
  test ! -e "$removed_file"
  test ! -e "$pkill_file"
}

# The fixed name is reused by a fresh container. A stale watchdog has the old
# ID and must not stop or remove the fresh container.
run_case stale-container-id stale-launch-identity
# Even with the right ID, a relabelled/reused launch must not be stopped.
run_case fresh-container-id stale-launch-identity

printf 'trainer failure watchdog ownership tests passed\n'
