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
  test "$status" -eq 125
  test "${#repair_unidentified_tunnels[@]}" -eq 1
  test "${#repair_unregistered_tunnels[@]}" -eq 0
  command kill -0 "$pid"
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
      test "$status" -eq 125
      test ! -e "$tmp/kill-attempts"
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

# Inject real signals at deterministic boundaries in the shared launch helper.
# Sleep groups replace SSH only; registration, pending signals, and EXIT are real.
boundary_fixture="$tmp/launch-boundary-fixture.sh"
cat >"$boundary_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
setsid sleep 1000 >/dev/null 2>&1 &
old_pid=$!
printf '%s\n' "$old_pid" >"$boundary_ready"
fixture_stop() {
  local pid=$1 pgid start_time command_line
  [ "$pid" != "$old_pid" ] || return 1
  repair_direct_child_owned "$pid" || return 1
  command_line=$(process_command_line "$pid") || return 1
  [ "$command_line" = 'sleep 1000' ] || return 1
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ') || return 1
  [ "$pgid" = "$pid" ] || return 1
  start_time=$(process_start_time "$pid") || return 1
  process_identity_matches "$pid" "$pgid" "$start_time" "$command_line" || return 1
  printf '%s\n' "$pid" >>"$boundary_log"
  kill -- "-$pgid" || return 1
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.01
  done
  return 1
}
generic_tunnel_stale_supervisor_match() { [ "$1" = 'sleep 1000' ] && [ "$2" = "$fixture_marker" ]; }
peer_tunnel_supervisor_match() { [ "$1" = 'sleep 1000' ] && [ "$3" = "$fixture_marker" ]; }
china_tunnel_supervisor_match() { [ "$1" = 'sleep 1000' ] && [ "$3" = "$fixture_marker" ]; }
generic_tunnel_stop_stale() { fixture_stop "$1"; }
peer_kill_owned() { fixture_stop "$1"; }
china_tunnel_stop_supervisor() { fixture_stop "$1"; }
repair_started_command_matches() {
  [ "$2" = 'sleep 1000' ] && [ "$3" = "$fixture_marker" ] && [ "$5" = 'exec sleep 1000' ]
}
case "$boundary_kind" in
  generic) fixture_marker=zmq-tunnel-29500 ;;
  peer) fixture_marker=peer-tunnel-lora-0 ;;
  china) fixture_marker=china-tunnel-lora-0 ;;
esac
case "$boundary_point" in
  before) interrupt_command='repair_launch_active=1' ;;
  spawned|transition|failed) interrupt_command='launch_pid=$!' ;;
  registered) interrupt_command='repair_launch_active=0' ;;
  after) interrupt_command='return "$launch_status"' ;;
esac
if [ "$boundary_point" = failed ]; then
  # Model a failed registration that has safely stopped its unregistered child.
  repair_register_tunnel() {
    for _ in $(seq 1 100); do
      [ "$(process_command_line "$2")" = 'sleep 1000' ] && break
      sleep 0.01
    done
    fixture_stop "$2" || return 1
    return 1
  }
fi
if [ "$boundary_point" = transition ]; then
  setsid() {
    : >"$boundary_ready.transition"
    while [ ! -e "$boundary_ready.release" ]; do :; done
    exec setsid "$@"
  }
fi
boundary_hook() {
  [ "$1" = "$interrupt_command" ] || return 0
  trap - DEBUG
  if [ "$boundary_point" != before ]; then
    new_pid=$!
    printf '%s\n' "$new_pid" >>"$boundary_ready"
    if [ "$boundary_point" = transition ]; then
      for _ in $(seq 1 100); do
        [ -e "$boundary_ready.transition" ] && break
        sleep 0.01
      done
      test -e "$boundary_ready.transition" || exit 1
      [ "$(process_command_line "$new_pid")" != 'sleep 1000' ] || exit 1
    fi
  fi
  kill -s "$boundary_signal" "$BASHPID"
  [ "$boundary_point" != transition ] || : >"$boundary_ready.release"
}
set -T
trap 'boundary_hook "$BASH_COMMAND"' DEBUG
repair_launch_tunnel "$boundary_kind" "$fixture_marker" fixture-forward 'exec sleep 1000'
exit 9
EOF
for boundary_kind in generic peer china; do
  for boundary_signal in TERM; do
    for boundary_point in before spawned transition registered after failed; do
      boundary_ready="$tmp/ready-boundary-$boundary_kind-$boundary_signal-$boundary_point"
      boundary_log="$tmp/boundary-$boundary_kind-$boundary_signal-$boundary_point-log"
      set +e
      launcher="$launcher" boundary_kind="$boundary_kind" boundary_signal="$boundary_signal" \
        boundary_point="$boundary_point" boundary_ready="$boundary_ready" boundary_log="$boundary_log" \
        bash "$boundary_fixture"
      boundary_status=$?
      set -e
      test "$boundary_status" -eq 124
      readarray -t boundary_pids <"$boundary_ready"
      old_pid=${boundary_pids[0]}
      kill -0 "$old_pid"
      if [ "$boundary_point" = before ]; then
        test "${#boundary_pids[@]}" -eq 1
        test ! -s "$boundary_log"
      else
        new_pid=${boundary_pids[1]}
        test "$(cat "$boundary_log")" = "$new_pid"
        if kill -0 "$new_pid" 2>/dev/null; then
          echo "signal boundary left a $boundary_kind group" >&2
          exit 1
        fi
      fi
      /bin/kill -- "-$old_pid" 2>/dev/null || true
    done
  done
done

orphan_fixture="$tmp/orphan-start-fixture.sh"
cat >"$orphan_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
launch_command='sleep 1000 & wait'
source <(sed -n '/^tunnel_channel_timeout_normalize() {/,/^}/p' "$launcher")
if [ "$orphan_mode" = turnover ]; then
  launch_command='sleep 1000 & first=$!; trap '\''kill "$first"; wait "$first"; sleep 1000 &'\'' USR1; wait; wait'
fi
setsid bash -c "$launch_command" >/dev/null 2>&1 &
pid=$!
setsid sleep 1000 >/dev/null 2>&1 &
other_pid=$!
trap 'command kill -KILL -- "-$pid" "-$other_pid" 2>/dev/null || true; wait "$pid" "$other_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
  child=$(command ps -eo pid=,ppid=,args= | awk -v parent="$pid" '$2 == parent && $3 == "sleep" {print $1}')
  [ -n "$child" ] && break
  sleep 0.01
done
test -n "$child" || exit 1
start_time=$(process_start_time "$pid") || exit 1
tunnel_supervisor_ssh_command() { printf 'sleep 1000\n'; }
repair_started_command_matches() {
  [ "$1" = "$orphan_kind" ] && [ "$2" = "bash -c $launch_command" ] &&
    [ "$3" = orphan-marker ] && [ "$4" = fixture-forward ] && [ "$5" = expected ]
}
if repair_starting_group "$pid" "$start_time" expected orphan-marker '' capture "$orphan_kind" wrong-forward; then
  exit 1
fi
repair_starting_group "$pid" "$start_time" expected orphan-marker '' capture "$orphan_kind" fixture-forward || exit 1
bound=$repair_starting_members
test -n "$bound" || exit 1
if [ "$orphan_mode" = turnover ]; then
  first_child=$child
  command kill -USR1 "$pid"
  for _ in $(seq 1 100); do
    child=$(command ps -eo pid=,ppid=,args= | awk -v parent="$pid" '$2 == parent && $3 == "sleep" {print $1}')
    [ -n "$child" ] && [ "$child" != "$first_child" ] && break
    sleep 0.01
  done
  test -n "$child" && test "$child" != "$first_child" || exit 1
fi
command kill -KILL "$pid"
wait "$pid" 2>/dev/null || true
test ! -e "/proc/$pid" || exit 1
signals=0
kill() { signals=$((signals + 1)); command kill "$@"; }
case "$orphan_mode" in
  registration)
    repair_register_tunnel "$orphan_kind" "$pid" orphan-marker fixture-forward expected
    test "$?" -eq 125 || exit 1
    test "${#repair_starting_tunnels[@]}" -eq 1 || exit 1
    test "$signals" -eq 0 || exit 1
    bound=
    ;;
  unbound) bound="$other_pid:$(process_start_time "$other_pid")," ;;
  unknown) tunnel_supervisor_ssh_command() { printf 'sleep 999\n'; } ;;
  ps-error) ps() { command ps "$@"; return 1; } ;;
  awk-error) awk() { command awk "$@"; return 2; } ;;
esac
repair_starting_tunnels=("$pid|$start_time|orphan-marker|$orphan_kind|fixture-forward|expected|$bound")
repair_cleanup_starting_tunnels
status=$?
if [ "$orphan_mode" = success ]; then
  test "$status" -eq 0 && test "$signals" -eq 1 || exit 1
  state=$(command ps -o stat= -p "$child" | tr -d ' ')
  case "$state" in ''|Z*) ;; *) exit 1 ;; esac
else
  test "$status" -ne 0 && test "$signals" -eq 0 || exit 1
  command kill -0 "$child" || exit 1
fi
command kill -0 "$other_pid" || exit 1
EOF
for orphan_kind in generic peer china; do
  for orphan_mode in success registration turnover unbound unknown ps-error awk-error; do
    launcher="$launcher" orphan_kind="$orphan_kind" orphan_mode="$orphan_mode" bash "$orphan_fixture"
  done
done

command_fixture="$tmp/starting-command-fixture.sh"
cat >"$command_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
source <(sed -n '/^tunnel_channel_timeout_normalize() {/,/^}/p; /^tunnel_supervisor_ssh_command() {/,/^}/p; /^china_tunnel_proxy_ssh_command() {/,/^}/p' "$launcher")
kind=peer
route=infinity
actual_ssh='ssh -NT -o ChannelTimeout=*-tcpip=45s'
proxy_ssh='ssh -W [fixture.test]:22'
case "$command_mode" in
  escaped-near) actual_ssh='ssh -NT -o ChannelTimeout=*-tcpip=46s' ;;
  proxy-success|proxy-near) kind=china ;;
esac
[ "$command_mode" != proxy-near ] || proxy_ssh='ssh -W [near.test]:22'
expected='while true; do ssh -NT -o ChannelTimeout=\*-tcpip=45s infinity; sleep 3; done # cmd-marker'
launch_command='bash -c '\''exec -a "$1" sleep infinity'\'' fixture "$actual_ssh" &'
case "$command_mode" in proxy-*) launch_command+=' bash -c '\''exec -a "$1" sleep infinity'\'' fixture "$proxy_ssh" &' ;; esac
launch_command+=' wait'
export actual_ssh proxy_ssh
setsid bash -c "$launch_command" >/dev/null 2>&1 &
pid=$!
trap 'command kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
  children=$(command ps -eo ppid=,args= | awk -v parent="$pid" '$1 == parent && $2 == "ssh" {count++} END {print count+0}')
  wanted=1
  case "$command_mode" in proxy-*) wanted=2 ;; esac
  [ "$children" -eq "$wanted" ] && break
  sleep 0.01
done
test "$children" -eq "$wanted" || exit 1
start_time=$(process_start_time "$pid") || exit 1
repair_started_command_matches() {
  [ "$1" = "$kind" ] && [ "$2" = "bash -c $launch_command" ] && [ "$3" = cmd-marker ] && [ "$5" = "$expected" ]
}
ssh() {
  [ "$*" = '-G infinity' ] || return 1
  printf 'hostname fixture.test\nport 22\nproxyjump infinity\n'
}
signals=0
kill() { signals=$((signals + 1)); command kill "$@"; }
if [ "$command_mode" = capture-empty ]; then
  ps() {
    [ "$*" != '-eo pid=,ppid=,pgid=,sid=,stat=,args=' ] || return 0
    command ps "$@"
  }
fi
repair_starting_group "$pid" "$start_time" "$expected" cmd-marker '' capture "$kind" fixture-forward "$route"
status=$?
case "$command_mode" in
  capture-empty)
    test "$status" -ne 0 && test "$signals" -eq 0 || exit 1
    repair_register_tunnel "$kind" "$pid" cmd-marker fixture-forward "$expected"
    test "$?" -ne 0 || exit 1
    test "${#repair_created_tunnels[@]}" -eq 0 || exit 1
    ;;
  escaped-success|proxy-success)
    test "$status" -eq 0 || exit 1
    bound=$repair_starting_members
    test -n "$bound" || exit 1
    command kill -KILL "$pid"
    wait "$pid" 2>/dev/null || true
    repair_starting_tunnels=("$pid|$start_time|cmd-marker|$kind|fixture-forward|$expected|$bound|$route")
    repair_cleanup_starting_tunnels || exit 1
    test "$signals" -eq 1 || exit 1
    ;;
  *) test "$status" -ne 0 && test "$signals" -eq 0 || exit 1 ;;
esac
EOF
for command_mode in escaped-success escaped-near proxy-success proxy-near proxy-kind capture-empty; do
  launcher="$launcher" command_mode="$command_mode" bash "$command_fixture"
done

# Resume after SIGSTOP failures only when a fresh start-time check succeeds.
stop_fixture="$tmp/stop-fault-fixture.sh"
cat >"$stop_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
sleep 1000 &
pid=$!
trap 'command kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
start_time=$(process_start_time "$pid") || exit 1
eval "$(declare -f process_start_time | sed '1s/process_start_time/fault_real_start_time/')"
stopped=0
continued=0
kill() {
  if [ "${1:-}" = -STOP ]; then
    command kill "$@" || return 1
    stopped=1
  elif [ "${1:-}" = -KILL ] && [ "$stop_fault" = kill ]; then
    return 1
  else
    [ "${1:-}" != -CONT ] || continued=1
    command kill "$@"
  fi
}
process_start_time() {
  case "$stopped:$stop_fault" in
    1:start) return 1 ;;
    1:empty) return 0 ;;
    1:mismatch) printf '%s\n' "$((start_time + 1))"; return 0 ;;
  esac
  fault_real_start_time "$@"
}
ps() {
  if [ "$stopped" = 1 ]; then
    case "$stop_fault:${2:-}" in
      stat:stat=|sid:sid=|pgid:pgid=|children:ppid=) return 1 ;;
      child:ppid=) printf '%s\n' "$pid"; return 0 ;;
    esac
  fi
  command ps "$@"
}
set +e
repair_stop_starting_tunnel "$pid" "$start_time"
stop_status=$?
set -e
test "$stop_status" -ne 0
command kill -0 "$pid"
case "$stop_fault" in
  start|empty|mismatch)
    test "$continued" -eq 0
    exit 0
    ;;
esac
test "$continued" -eq 1
for _ in $(seq 1 50); do
  state=$(command ps -o stat= -p "$pid" | tr -d ' ')
  case "$state" in T*) sleep 0.01 ;; *) break ;; esac
done
case "$state" in ''|T*|Z*) exit 1 ;; esac
EOF
for stop_fault in stat start empty mismatch sid pgid children child kill; do
  launcher="$launcher" stop_fault="$stop_fault" bash "$stop_fixture"
done

# A pending signal cannot turn a missing initial identity into a group kill.
baseline_fixture="$tmp/baseline-failure-fixture.sh"
cat >"$baseline_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
setsid sleep 1000 >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$baseline_ready"
repair_launch_active=1
tunnels_only=1
tunnel_repair_deadline=$((SECONDS + 1))
process_start_time() { printf 'read\n' >>"$baseline_reads"; return 1; }
kill() { printf '%s\n' "$*" >>"$baseline_kills"; return 1; }
command kill -TERM "$$"
repair_register_tunnel generic "$pid" baseline-marker fixture-forward expected-command
test "$?" -eq 125 || exit 1
test "${#repair_unidentified_tunnels[@]}" -eq 1 || exit 1
test "${#repair_unregistered_tunnels[@]}" -eq 0 || exit 1
repair_launch_active=0
repair_signal
EOF
baseline_ready="$tmp/baseline-ready"
baseline_reads="$tmp/baseline-reads"
baseline_kills="$tmp/baseline-kills"
set +e
baseline_output=$(launcher="$launcher" baseline_ready="$baseline_ready" \
  baseline_reads="$baseline_reads" baseline_kills="$baseline_kills" \
  bash "$baseline_fixture" 2>&1)
baseline_status=$?
set -e
read -r baseline_pid <"$baseline_ready"
test "$baseline_status" -eq 125
test "$(wc -l <"$baseline_reads")" -gt 1
test ! -e "$baseline_kills"
[[ "$baseline_output" == *"unresolved unidentified tunnel PID $baseline_pid marker baseline-marker; no signal sent"* ]]
command kill -0 "$baseline_pid"
/bin/kill -- "-$baseline_pid" 2>/dev/null || /bin/kill "$baseline_pid" 2>/dev/null || true

# A failed startup stop retains its PID/start record for the real EXIT retry.
retry_fixture="$tmp/start-retry-fixture.sh"
cat >"$retry_fixture" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
source <(sed '/^RUN=\${RUN:/,$d' "$launcher")
setsid sleep 1000 >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$retry_ready"
start_time=$(process_start_time "$pid") || exit 1
eval "$(declare -f repair_stop_starting_tunnel | sed '1s/repair_stop_starting_tunnel/retry_real_stop/')"
repair_wait_started_tunnel() { return 1; }
stop_calls=0
repair_stop_starting_tunnel() {
  [ "$1" = "$pid" ] && [ "$2" = "$start_time" ] || return 1
  stop_calls=$((stop_calls + 1))
  printf '%s\n' "$stop_calls" >>"$retry_log"
  [ "$retry_mode" = once ] && [ "$stop_calls" -gt 1 ] || return 1
  retry_real_stop "$@"
}
repair_register_tunnel generic "$pid" retry-marker fixture-forward expected-command
test "$?" -eq 125 || exit 1
test "${#repair_starting_tunnels[@]}" -eq 1 || exit 1
repair_signal
EOF
for retry_mode in once persistent; do
  retry_ready="$tmp/ready-retry-$retry_mode"
  retry_log="$tmp/retry-$retry_mode-log"
  set +e
  retry_output=$(launcher="$launcher" retry_mode="$retry_mode" retry_ready="$retry_ready" \
    retry_log="$retry_log" bash "$retry_fixture" 2>&1)
  retry_status=$?
  set -e
  read -r retry_pid <"$retry_ready"
  test "$(cat "$retry_log")" = $'1\n2'
  if [ "$retry_mode" = persistent ]; then
    test "$retry_status" -eq 125
    [[ "$retry_output" == *"unresolved starting tunnel PID $retry_pid marker retry-marker"* ]]
    /bin/kill -- "-$retry_pid" 2>/dev/null || /bin/kill "$retry_pid" 2>/dev/null || true
  else
    test "$retry_status" -eq 124
    if kill -0 "$retry_pid" 2>/dev/null; then exit 1; fi
  fi
done

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
    eval "$(sed -n '/^repair_committed=1$/,/^repair_release_lock .*repair_fail/p' "$launcher")"
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
