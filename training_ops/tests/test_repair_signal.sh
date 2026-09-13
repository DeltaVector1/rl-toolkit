#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$repo/training_ops/runtime/h200/start_orchestrator.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fixture="$tmp/repair-fixture.sh"
cat >"$fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$LAUNCHER")

tunnels_only=1
repair_created_tunnels=()
repair_failed=0

# The fixture keeps the real register, EXIT, and signal functions. These
# narrow stubs only replace the tunnel-specific ownership and stop functions
# so the fixture can test record parsing without a live SSH connection.
generic_tunnel_stale_supervisor_match() { return 0; }
peer_tunnel_supervisor_match() { return 0; }
china_tunnel_supervisor_match() { return 0; }
generic_tunnel_stop_stale() {
  printf 'generic|%s|%s|%s\n' "$1" "$3" "$4" >>"$CLEANUP_LOG"
  kill "$1" 2>/dev/null || true
}
peer_kill_owned() {
  printf 'peer|%s|%s|\n' "$1" "$4" >>"$CLEANUP_LOG"
  kill "$1" 2>/dev/null || true
}
china_tunnel_stop_supervisor() {
  printf 'china|%s|%s|\n' "$1" "$3" >>"$CLEANUP_LOG"
  kill "$1" 2>/dev/null || true
}

spawn_fixture_process() {
  setsid sleep 1000 >/dev/null 2>&1 &
  printf '%s\n' "$!"
}

if [ "${MODE:-signal}" = register-failure ]; then
  pid=$(spawn_fixture_process)
  process_start_time() { return 1; }
  generic_tunnel_stale_group_safe() { return 0; }
  generic_tunnel_stale_supervisor_match() { return 0; }
  set +e
  repair_register_tunnel generic "$pid" failed-marker 127.0.0.1:29500:target:22
  status=$?
  set -e
  test "$status" -ne 0
  for _ in $(seq 1 20); do
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$state" in ''|Z*) break ;; esac
    sleep 0.05
  done
  state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  case "$state" in ''|Z*) ;; *)
    echo "registration failure left an unregistered process" >&2
    kill "$pid" 2>/dev/null || true
    exit 1
    ;;
  esac
  exit 0
fi

created_pid=$(spawn_fixture_process)
peer_pid=$(spawn_fixture_process)
china_pid=$(spawn_fixture_process)
preexisting_pid=$(spawn_fixture_process)
repair_register_tunnel generic "$created_pid" created-marker 127.0.0.1:29500:target:22
repair_register_tunnel peer "$peer_pid" peer-marker
repair_register_tunnel china "$china_pid" china-marker
printf '%s %s %s %s\n' "$created_pid" "$peer_pid" "$china_pid" "$preexisting_pid" >"$READY_FILE"
while :; do sleep 1; done
EOF
chmod +x "$fixture"

run_signal() {
  local signal=$1 ready log runner status
  ready="$tmp/ready-$signal"
  log="$tmp/log-$signal"
  : >"$log"
  LAUNCHER="$launcher" MODE=signal READY_FILE="$ready" CLEANUP_LOG="$log" \
    bash "$fixture" &
  runner=$!
  for _ in $(seq 1 50); do
    [ -s "$ready" ] && break
    sleep 0.1
  done
  test -s "$ready"
  read -r created_pid peer_pid china_pid preexisting_pid <"$ready"
  kill -s "$signal" "$runner"
  set +e
  wait "$runner"
  status=$?
  set -e
  test "$status" -ne 0
  for pid in "$created_pid" "$peer_pid" "$china_pid"; do
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$state" in ''|Z*) continue ;; esac
    if kill -0 "$pid" 2>/dev/null; then
      echo "$signal left registered process $pid" >&2
      kill "$pid" 2>/dev/null || true
      return 1
    fi
  done
  kill -0 "$preexisting_pid" 2>/dev/null
  grep -Fqx "generic|$created_pid|created-marker|127.0.0.1:29500:target:22" "$log"
  grep -Fqx "peer|$peer_pid|peer-marker|" "$log"
  grep -Fqx "china|$china_pid|china-marker|" "$log"
  kill "$preexisting_pid" 2>/dev/null || true
  wait "$preexisting_pid" 2>/dev/null || true
}

run_signal TERM
run_signal HUP
LAUNCHER="$launcher" MODE=register-failure READY_FILE="$tmp/unused" \
  CLEANUP_LOG="$tmp/register-failure-log" bash "$fixture"
echo "repair signal and registration tests passed"
