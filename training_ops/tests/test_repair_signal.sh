#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$repo/training_ops/runtime/h200/start_orchestrator.sh"
tmp=$(mktemp -d)
cleanup_test_processes() {
  local ready pid
  for ready in "$tmp"/ready-* "$tmp"/failure-*; do
    [ -s "$ready" ] || continue
    while read -r pid _; do
      case "$pid" in ''|*[!0-9]*) continue ;; esac
      /bin/kill -- "-$pid" 2>/dev/null || /bin/kill "$pid" 2>/dev/null || true
    done <"$ready"
  done
  rm -rf "$tmp"
}
trap cleanup_test_processes EXIT

fixture="$tmp/repair-fixture.sh"
cat >"$fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$LAUNCHER")

tunnels_only=1
tunnel_repair_deadline=$((SECONDS + 120))
repair_created_tunnels=()
repair_unregistered_tunnels=()
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

fixture_pids=()
fixture_keep_pid=
spawn_fixture_process() {
  setsid sleep 1000 >/dev/null 2>&1 &
  spawned_pid=$!
  fixture_pids+=("$spawned_pid")
}
fixture_cleanup() {
  local pid
  for pid in "${fixture_pids[@]}"; do
    [ "$pid" = "$fixture_keep_pid" ] && continue
    /bin/kill -- "-$pid" 2>/dev/null || /bin/kill "$pid" 2>/dev/null || true
  done
}

fixture_status_trap() {
  local status=$?
  if ! repair_cleanup_created_tunnels; then status=125; fi
  if ! repair_cleanup_unregistered_tunnels; then status=125; fi
  fixture_cleanup
  exit "$status"
}
trap fixture_status_trap EXIT

if [ "${MODE:-signal}" = register-failure ]; then
  spawn_fixture_process
  pid=$spawned_pid
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

if [ "${MODE:-signal}" = register-disappears ]; then
  spawn_fixture_process
  pid=$spawned_pid
  /bin/kill "$pid"
  wait "$pid" 2>/dev/null || true
  process_start_time() { return 1; }
  set +e
  repair_register_tunnel generic "$pid" disappeared-marker 127.0.0.1:29500:target:22
  status=$?
  set -e
  test "$status" -ne 0
  exit 0
fi

if [ "${MODE:-signal}" = register-reused ]; then
  spawn_fixture_process
  pid=$spawned_pid
  printf '%s\n' "$pid" >"$FAILURE_PID_FILE"
  generic_tunnel_stale_supervisor_match() { return 1; }
  kill() { printf '%s\n' reached >"$KILL_REACHED_FILE"; return 1; }
  set +e
  repair_stop_unregistered_tunnel generic "$pid" reused-marker 127.0.0.1:29500:target:22
  status=$?
  set -e
  test "$status" -ne 0
  test ! -s "$KILL_REACHED_FILE"
  exit 0
fi

if [ "${MODE:-signal}" = register-kill-failure ]; then
  spawn_fixture_process
  pid=$spawned_pid
  printf '%s\n' "$pid" >"$FAILURE_PID_FILE"
  process_start_time() { return 1; }
  generic_tunnel_stale_group_safe() { return 0; }
  generic_tunnel_stale_supervisor_match() { return 0; }
  kill() {
    attempts=$(cat "$KILL_ATTEMPTS_FILE" 2>/dev/null || printf '0')
    attempts=$((attempts + 1))
    printf '%s\n' "$attempts" >"$KILL_ATTEMPTS_FILE"
    [ "$attempts" -gt 1 ] || return 1
    command kill "$@"
  }
  set +e
  repair_register_tunnel generic "$pid" failed-marker 127.0.0.1:29500:target:22
  status=$?
  set -e
  test "$status" -ne 0
  exit "$status"
fi

spawn_fixture_process
created_pid=$spawned_pid
spawn_fixture_process
peer_pid=$spawned_pid
spawn_fixture_process
china_pid=$spawned_pid
spawn_fixture_process
preexisting_pid=$spawned_pid
fixture_keep_pid=$preexisting_pid
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
  test "$(wc -l <"$log")" -eq 3
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

run_register_failure() {
  local mode=$1 status pid state
  local failure_file="$tmp/failure-$mode"
  case "$mode" in
    register-kill-failure)
      set +e
      LAUNCHER="$launcher" MODE="$mode" FAILURE_PID_FILE="$failure_file" \
        KILL_ATTEMPTS_FILE="$tmp/kill-attempts" CLEANUP_LOG="$tmp/kill-log" \
        bash "$fixture"
      status=$?
      set -e
      test "$status" -ne 0
      test "$(cat "$tmp/kill-attempts")" -ge 2
      ;;
    register-reused)
      LAUNCHER="$launcher" MODE="$mode" FAILURE_PID_FILE="$failure_file" \
        KILL_REACHED_FILE="$tmp/kill-reached" CLEANUP_LOG="$tmp/reused-log" \
        bash "$fixture"
      test ! -s "$tmp/kill-reached"
      ;;
    *)
      LAUNCHER="$launcher" MODE="$mode" FAILURE_PID_FILE="$failure_file" \
        CLEANUP_LOG="$tmp/disappeared-log" bash "$fixture"
      ;;
  esac
  if [ -s "$failure_file" ]; then
    read -r pid <"$failure_file"
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$state" in ''|Z*) ;; *)
      echo "$mode left a live child" >&2
      return 1
      ;;
    esac
  fi
}

# Full-mode transactions use real local process groups and the production
# register, identity, signal, commit, and EXIT functions. Sleep groups stand
# in for SSH groups; the narrow stop fixture proves their exact identities.
full_fixture="$tmp/full-repair-fixture.sh"
cat >"$full_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
test "$tunnels_only" -eq 0 || exit 1
setsid sleep 1000 >/dev/null 2>&1 &
old_pid=$!
setsid sleep 1000 >/dev/null 2>&1 &
new_pid=$!
printf '%s\n' "$old_pid" "$new_pid" >"$full_ready"
for _ in $(seq 1 50); do
  [ "$(ps -o pgid= -p "$new_pid" | tr -d ' ')" = "$new_pid" ] && break
  sleep 0.01
done
new_start=$(process_start_time "$new_pid") || exit 1
repair_direct_child_owned "$new_pid" || exit 1
generic_tunnel_stale_supervisor_match() {
  [ "$1" = 'sleep 1000' ] && [ "$2" = full-created ] && [ "$3" = fixture-forward ]
}
generic_tunnel_stop_stale() {
  [ "$1" = "$new_pid" ] || return 1
  generic_tunnel_stale_supervisor_match "$2" "$3" "$4" || return 1
  process_identity_matches "$1" "$new_pid" "$new_start" 'sleep 1000' || return 1
  printf '%s\n' "$1" >>"$full_log"
  kill -- "-$new_pid" || return 1
  for _ in $(seq 1 50); do
    kill -0 "$new_pid" 2>/dev/null || return 0
    sleep 0.01
  done
  return 1
}
repair_register_tunnel generic "$new_pid" full-created fixture-forward || exit 1
test "${#repair_created_tunnels[@]}" -eq 1 || exit 1
case "$full_mode" in
  fail) exit 7 ;;
  signal) repair_signal ;;
  committed)
    adapter_tunnel_readiness() { return 0; }
    eval "$(sed -n '/^if ! adapter_tunnel_readiness;/,/^repair_release_lock .*repair_fail/p' "$launcher")"
    test "${#repair_created_tunnels[@]}" -eq 0 || exit 1
    exit 7
    ;;
esac
EOF
for full_mode in fail signal committed; do
  full_ready="$tmp/ready-full-$full_mode"
  full_log="$tmp/full-$full_mode-log"
  set +e
  launcher="$launcher" full_mode="$full_mode" full_ready="$full_ready" full_log="$full_log" \
    bash "$full_fixture"
  full_status=$?
  set -e
  if [ "$full_mode" = signal ]; then test "$full_status" -eq 124; else test "$full_status" -eq 7; fi
  readarray -t full_pids <"$full_ready"
  old_pid=${full_pids[0]}
  new_pid=${full_pids[1]}
  kill -0 "$old_pid"
  if [ "$full_mode" = committed ]; then
    kill -0 "$new_pid"
    test ! -s "$full_log"
  else
    test "$(cat "$full_log")" = "$new_pid"
    if kill -0 "$new_pid" 2>/dev/null; then
      echo "full-mode rollback left its new group" >&2
      exit 1
    fi
  fi
  /bin/kill -- "-$old_pid" 2>/dev/null || true
  /bin/kill -- "-$new_pid" 2>/dev/null || true
done

run_signal TERM
run_signal HUP
LAUNCHER="$launcher" MODE=register-failure READY_FILE="$tmp/unused" \
  CLEANUP_LOG="$tmp/register-failure-log" bash "$fixture"
run_register_failure register-disappears
run_register_failure register-reused
run_register_failure register-kill-failure
echo "repair signal and registration tests passed"
