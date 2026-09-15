#!/usr/bin/env bash
# Run the orchestrator on h200a instead of n82.
# Engine traffic stays local. Only adapter and completed batch data cross the
# China link.
set -uo pipefail

tunnels_only=0
restart_tunnels=0
for argument in "$@"; do
  case "$argument" in
    --tunnels-only) tunnels_only=1 ;;
    --restart-tunnels) restart_tunnels=1 ;;
    *) echo "unknown launcher argument: $argument" >&2; exit 2 ;;
  esac
done

tunnel_repair_deadline=0
repair_total_deadline=0
repair_cleanup_spent_us=0
repair_cleanup_active_us=0
repair_cleanup_completed=0
repair_cleanup_limit_us=20000000
repair_exit_cleanup_limit_us=20000000
repair_stage=initialization
repair_started_seconds=$SECONDS
repair_failed=0
repair_created_tunnels=()
repair_unregistered_tunnels=()
repair_starting_tunnels=()
repair_unidentified_tunnels=()
repair_committed=0
repair_launch_active=0
repair_pending_signal=0
peer_readiness_failure=
peer_phase_failure=
repair_lock_held=0
repair_lock_file=${ADAPTER_TUNNEL_LOCK_FILE:-/tmp/prime-rl-adapter-tunnel-repair.lock}
if [ "$tunnels_only" = 1 ]; then
  tunnel_repair_deadline_s=${TUNNEL_REPAIR_DEADLINE_S:-300}
  case "$tunnel_repair_deadline_s" in
    ''|*[!0-9]*) echo "TUNNEL_REPAIR_DEADLINE_S must be a positive integer" >&2; exit 2 ;;
  esac
  [ "$tunnel_repair_deadline_s" -gt 0 ] || {
    echo "TUNNEL_REPAIR_DEADLINE_S must be a positive integer" >&2
    exit 2
  }
  tunnel_repair_deadline=$((SECONDS + tunnel_repair_deadline_s))
  repair_total_deadline=$tunnel_repair_deadline
  repair_reserve_seconds=$((tunnel_repair_deadline_s / 5))
  [ "$repair_reserve_seconds" -le 30 ] || repair_reserve_seconds=30
  tunnel_repair_deadline=$((tunnel_repair_deadline - repair_reserve_seconds))
  repair_cleanup_limit_us=$(((tunnel_repair_deadline_s - repair_reserve_seconds) * 1000000))
  repair_exit_cleanup_limit_us=$((repair_reserve_seconds * 1000000))
fi

repair_cleanup_remaining() {
  local now=${EPOCHREALTIME/./} used=${repair_cleanup_spent_us:-0} deadline=${repair_total_deadline:-0}
  if [ "${repair_cleanup_active_us:-0}" -gt 0 ]; then
    used=$((used + now - repair_cleanup_active_us))
  fi
  [ "$used" -lt "${repair_cleanup_limit_us:-20000000}" ] || return 1
  if [ "${repair_stage:-}" != cleanup ] && [ "${tunnel_repair_deadline:-0}" -gt 0 ]; then
    deadline=$tunnel_repair_deadline
  fi
  [ "$deadline" = 0 ] || [ "$SECONDS" -lt "$deadline" ]
}

repair_cleanup_call() {
  local stage=$1 status finished
  repair_cleanup_remaining || {
    echo "tunnel cleanup time limit reached: stage=$stage completed=${repair_cleanup_completed:-0}" >&2
    return 1
  }
  if [ "${repair_cleanup_active_us:-0}" -gt 0 ]; then
    "$@"
    return $?
  fi
  repair_cleanup_active_us=${EPOCHREALTIME/./}
  if "$@"; then status=0; else status=$?; fi
  finished=${EPOCHREALTIME/./}
  repair_cleanup_spent_us=$(( ${repair_cleanup_spent_us:-0} + finished - repair_cleanup_active_us ))
  repair_cleanup_active_us=0
  repair_cleanup_completed=$(( ${repair_cleanup_completed:-0} + 1 ))
  repair_cleanup_remaining || {
    echo "tunnel cleanup time limit reached: stage=$stage completed=$repair_cleanup_completed" >&2
    return 1
  }
  return "$status"
}

repair_remaining() {
  local remaining
  if [ "$tunnels_only" != 1 ]; then
    printf '2147483647\n'
    return 0
  fi
  remaining=$((tunnel_repair_deadline - SECONDS))
  if [ "$remaining" -gt 0 ]; then
    printf '%s\n' "$remaining"
    return 0
  fi
  echo "adapter tunnel reconciliation deadline exceeded" >&2
  return 1
}

repair_timeout() {
  local requested=$1 remaining
  case "$requested" in
    ''|*[!0-9]*) return 1 ;;
  esac
  if [ "$tunnels_only" != 1 ]; then
    printf '%s\n' "$requested"
    return 0
  fi
  remaining=$(repair_remaining) || return 1
  [ "$requested" -lt "$remaining" ] && printf '%s\n' "$requested" || printf '%s\n' "$remaining"
}

repair_sleep() {
  local requested=$1 duration
  if [[ "$requested" =~ ^0\.[0-9]+$ ]]; then
    repair_remaining >/dev/null || return 1
    sleep "$requested" || return 1
    repair_remaining >/dev/null
    return $?
  fi
  duration=$(repair_timeout "$requested") || return 1
  sleep "$duration"
  repair_remaining >/dev/null
}

repair_fail() {
  repair_failed=1
  echo "adapter tunnel reconciliation failed: $*" >&2
  exit 1
}

repair_acquire_lock() {
  local lock_timeout
  [ "$repair_lock_held" = 1 ] && return 0
  lock_timeout=$(repair_timeout 30) || return 1
  exec 8>"$repair_lock_file" || return 1
  if ! flock -w "$lock_timeout" 8; then
    exec 8>&-
    echo "could not acquire adapter tunnel repair lock" >&2
    return 1
  fi
  repair_lock_held=1
}

repair_release_lock() {
  [ "$repair_lock_held" = 1 ] || return 0
  repair_lock_held=0
  if ! flock -u 8; then
    exec 8>&-
    return 1
  fi
  exec 8>&-
}

repair_signal() {
  repair_failed=1
  if [ "$repair_launch_active" = 1 ]; then
    repair_pending_signal=1
    return 0
  fi
  trap - TERM HUP INT
  exit 124
}

repair_launch_tunnel() {
  local kind=$1 marker=$2 forward=$3 command_line=$4 launch_pid launch_status
  repair_launch_active=1
  setsid nohup bash -c "$command_line" >/dev/null 2>&1 < /dev/null 8>&- &
  launch_pid=$!
  if repair_register_tunnel "$kind" "$launch_pid" "$marker" "$forward" "$command_line"; then
    launch_status=0
  else
    launch_status=$?
  fi
  repair_launch_active=0
  [ "${repair_pending_signal:-0}" = 0 ] || repair_signal
  return "$launch_status"
}

process_start_time() {
  local pid=$1
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

process_command_line() {
  local pid=$1 output
  if ! output=$(ps -o args= -p "$pid" 2>/dev/null); then
    [ ! -e "/proc/$pid" ] && return 2
    return 1
  fi
  sed 's/[[:space:]]*$//' <<<"$output"
}

process_identity_matches() {
  local pid=$1 expected_pgid=$2 expected_start=$3 expected_command=$4
  local current_pgid current_start current_command
  current_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  [ "$current_pgid" = "$expected_pgid" ] || return 1
  current_start=$(process_start_time "$pid") || return 1
  [ -n "$current_start" ] && [ "$current_start" = "$expected_start" ] || return 1
  current_command=$(process_command_line "$pid") || return 1
  [ "$current_command" = "$expected_command" ]
}

repair_direct_child_owned() {
  local expected_pid=$1 child_pid child_pids
  child_pids=$(jobs -pr) || return 1
  while read -r child_pid; do
    [ "$child_pid" = "$expected_pid" ] && return 0
  done <<<"$child_pids"
  return 1
}

repair_stop_unregistered_tunnel() {
  repair_cleanup_call repair_stop_unregistered_tunnel_inner "$@"
}

repair_stop_unregistered_tunnel_inner() {
  local kind=$1 pid=$2 marker=$3 forward=${4:-} command_line pgid shape=current
  local start_time current_start current_command current_pgid process_status group_output
  [ -e "/proc/$pid" ] || return 0
  repair_direct_child_owned "$pid" || return 1
  command_line=$(process_command_line "$pid")
  process_status=$?
  case "$process_status" in
    0) [ -n "$command_line" ] || return 1 ;;
    2) return 0 ;;
    *) return 0 ;;
  esac
  case "$kind" in
    generic)
      generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward" || return 1
      ;;
    peer)
      [ "$marker" = peer-tunnel-api ] && shape=api
      peer_tunnel_stale_supervisor_match "$command_line" "$shape" "$marker" || return 1
      ;;
    china)
      china_tunnel_supervisor_match "$command_line" current "$marker" || return 1
      ;;
    *) return 1 ;;
  esac
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || {
    [ ! -e "/proc/$pid" ] && return 0
    return 1
  }
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$") return 1 ;;
  esac
  case "$kind" in
    generic) generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward" || return 1 ;;
    peer) peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker" || return 1 ;;
    china) china_tunnel_group_safe "$pgid" "$marker" || return 1 ;;
  esac
  # A new, direct child can be stopped when registration could not read its
  # start time. Recheck the job, exact command, and group before that stop.
  if ! start_time=$(process_start_time "$pid" 2>/dev/null); then
    [ ! -e "/proc/$pid" ] && return 0
    start_time=
  fi
  repair_direct_child_owned "$pid" || return 1
  current_command=$(process_command_line "$pid")
  process_status=$?
  case "$process_status" in
    0) [ "$current_command" = "$command_line" ] || return 1 ;;
    2) return 0 ;;
    *) return 1 ;;
  esac
  case "$kind" in
    generic) generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward" || return 1 ;;
    peer) peer_tunnel_stale_supervisor_match "$command_line" "$shape" "$marker" || return 1 ;;
    china) china_tunnel_supervisor_match "$command_line" current "$marker" || return 1 ;;
  esac
  current_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || {
    [ ! -e "/proc/$pid" ] && return 0
    return 1
  }
  [ "$current_pgid" = "$pgid" ] || return 1
  if [ -n "$start_time" ]; then
    current_start=$(process_start_time "$pid") || {
      [ ! -e "/proc/$pid" ] && return 0
      return 1
    }
    [ "$current_start" = "$start_time" ] || return 1
  fi
  case "$kind" in
    generic) generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward" || return 1 ;;
    peer) peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker" || return 1 ;;
    china) china_tunnel_group_safe "$pgid" "$marker" || return 1 ;;
  esac
  repair_cleanup_remaining || return 1
  if kill -- "-$pgid" 2>/dev/null; then
    for _ in $(seq 1 30); do
      repair_cleanup_remaining || return 1
      # Check the complete proven group, not only its supervisor. Zombies
      # have exited and cannot own a listener; their parent must reap them.
      group_output=$(ps -eo pgid=,stat= 2>/dev/null) || return 1
      if awk -v group="$pgid" '$1 == group && $2 !~ /^Z/ {found=1} END {exit !found}' \
          <<<"$group_output"; then
        :
      else
        case "$?" in 1) return 0 ;; *) return 1 ;; esac
      fi
      sleep 0.1
      repair_cleanup_remaining || return 1
    done
    return 1
  fi
  [ ! -e "/proc/$pid" ]
}

repair_register_tunnel() {
  local kind=$1 pid=$2 marker=$3 forward=${4:-} expected_command=${5:-} start_time= observed_start
  local starting_index=${#repair_starting_tunnels[@]} route=
  [ "$kind" != china ] || route=${trainer_ssh_target:-}
  expected_command=${expected_command//$'\n'/ }
  expected_command=${expected_command//$'\t'/ }
  repair_starting_tunnels+=("$pid||$marker|$kind|$forward|$expected_command||$route")
  for _ in $(seq 1 100); do
    if observed_start=$(process_start_time "$pid") && [ -n "$observed_start" ]; then
      start_time=$observed_start
      break
    fi
    if [ ! -e "/proc/$pid" ]; then
      repair_failed=1
      repair_stop_starting_tunnel "$pid" "" "$kind" "$marker" "$forward" "$expected_command" "" "$route" || return 125
      return 1
    fi
    [ "$repair_launch_active" = 1 ] || break
    repair_remaining >/dev/null || break
    sleep 0.05
  done
  if [ -z "$start_time" ]; then
    echo "unidentified starting tunnel PID $pid marker $marker; no signal sent" >&2
    repair_unidentified_tunnels+=("$pid|$marker")
    repair_failed=1
    return 125
  fi
  if [ -n "$expected_command" ]; then
    repair_starting_tunnels[$starting_index]="$pid|$start_time|$marker|$kind|$forward|$expected_command||$route"
    repair_starting_members=
    if ! repair_wait_started_tunnel "$kind" "$pid" "$marker" "$forward" "$start_time" "$expected_command" "$route"; then
      repair_starting_tunnels[$starting_index]="$pid|$start_time|$marker|$kind|$forward|$expected_command|$repair_starting_members|$route"
      repair_failed=1
      repair_stop_starting_tunnel "$pid" "$start_time" "$kind" "$marker" "$forward" "$expected_command" "$repair_starting_members" "$route" || {
        echo "could not safely stop starting tunnel PID $pid marker $marker" >&2
        return 125
      }
      return 1
    fi
    repair_starting_tunnels[$starting_index]="$pid|$start_time|$marker|$kind|$forward|$expected_command|$repair_starting_members|$route"
  else
    repair_starting_tunnels[$starting_index]="$pid|$start_time|$marker|$kind|$forward|||$route"
  fi
  repair_created_tunnels+=("$kind|$pid|$marker|$forward|$start_time")
}

repair_cleanup_starting_tunnels() {
  local item pid start_time marker kind forward expected members route failed=0 processed=0
  for item in "${repair_unidentified_tunnels[@]}"; do
    IFS='|' read -r pid marker <<<"$item"
    if [ -e "/proc/$pid" ]; then
      echo "unresolved unidentified tunnel PID $pid marker $marker; no signal sent" >&2
      failed=1
    fi
  done
  for item in "${repair_starting_tunnels[@]}"; do
    repair_cleanup_remaining || {
      echo "tunnel cleanup stopped: stage=starting remaining=$((${#repair_starting_tunnels[@]} - processed))" >&2
      return 1
    }
    processed=$((processed + 1))
    IFS='|' read -r pid start_time marker kind forward expected members route <<<"$item"
    if ! repair_stop_starting_tunnel "$pid" "$start_time" "$kind" "$marker" "$forward" "$expected" "$members" "$route"; then
      echo "unresolved starting tunnel PID $pid marker $marker" >&2
      failed=1
    fi
  done
  [ "$failed" = 0 ]
}

repair_starting_group() {
  local pid=$1 start_time=$2 expected=$3 marker=$4 bound=$5 mode=$6
  local kind=${7:-} forward=${8:-} trainer_ssh_target=${9:-} proxy_ssh
  local output rows member parent pgid sid state command_line member_start expected_ssh anchor=0 found=0 leader_seen=0
  output=$(ps -eo pid=,ppid=,pgid=,sid=,stat=,args=) || return 1
  rows=$(awk -v group="$pid" '$3 == group {print}' <<<"$output") || return 1
  repair_starting_members=
  [ -n "$rows" ] || { [ "$mode" = stop ]; return "$?"; }
  while read -r member parent pgid sid state command_line; do
    [ "$pgid" = "$pid" ] && [ "$sid" = "$pid" ] || return 1
    case "$state" in Z*) continue ;; esac
    found=1
    [ -n "$start_time" ] && [ -n "$expected" ] || return 1
    member_start=$(process_start_time "$member") || return 1
    [ -n "$member_start" ] && [ "$member_start" -ge "$start_time" ] || return 1
    if [ "$member" = "$pid" ]; then
      [ "$mode" = capture ] && [ "$member_start" = "$start_time" ] || return 1
      repair_started_command_matches "$kind" "$command_line" "$marker" "$forward" "$expected" || return 1
      leader_seen=1
    else
      expected_ssh=$(tunnel_supervisor_ssh_command "bash -c $expected" "$marker") || return 1
      expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
      command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
      case "$command_line" in /usr/bin/ssh\ *) command_line="ssh ${command_line#/usr/bin/ssh }" ;; esac
      case "$command_line" in
        "$expected_ssh"|'sleep 3') ;;
        *)
          [ "$kind" = china ] && [ -n "$trainer_ssh_target" ] || return 1
          proxy_ssh=$(china_tunnel_proxy_ssh_command) || return 1
          [ "$command_line" = "$proxy_ssh" ] || return 1
          ;;
      esac
      case ",$bound," in *",$member:$member_start,"*) anchor=1 ;; esac
      repair_starting_members+="$member:$member_start,"
    fi
  done <<<"$rows"
  if [ "$mode" = capture ]; then
    [ "$leader_seen" = 1 ]
  else
    [ "$found" = 0 ] || [ "$anchor" = 1 ]
  fi
}

repair_stop_starting_group() {
  local pid=$1 start_time=$2 expected=$3 marker=$4 bound=$5 kind=${6:-} route=${7:-} output remaining
  repair_starting_group "$pid" "$start_time" "$expected" "$marker" "$bound" stop "$kind" "" "$route" || return 1
  [ -n "$repair_starting_members" ] || return 0
  repair_starting_group "$pid" "$start_time" "$expected" "$marker" "$bound" stop "$kind" "" "$route" || return 1
  [ -n "$repair_starting_members" ] || return 0
  repair_cleanup_remaining || return 1
  kill -KILL -- "-$pid" || return 1
  for _ in $(seq 1 40); do
    repair_cleanup_remaining || return 1
    output=$(ps -eo pgid=,stat=) || return 1
    remaining=$(awk -v group="$pid" '$1 == group && $2 !~ /^Z/ {print}' <<<"$output") || return 1
    [ -n "$remaining" ] || return 0
    sleep 0.05
    repair_cleanup_remaining || return 1
  done
  return 1
}

repair_started_command_matches() {
  local kind=$1 command_line=$2 marker=$3 forward=$4 expected=$5 shape=current actual_ssh expected_ssh
  case "$command_line" in 'bash -c '*) ;; *) return 1 ;; esac
  case "$kind" in
    generic) generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward" || return 1 ;;
    peer)
      [ "$marker" != peer-tunnel-api ] || shape=api
      peer_tunnel_supervisor_match "$command_line" "$shape" "$marker" || return 1
      ;;
    china) china_tunnel_supervisor_match "$command_line" current "$marker" || return 1 ;;
    *) return 1 ;;
  esac
  actual_ssh=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  expected_ssh=$(tunnel_supervisor_ssh_command "bash -c $expected" "$marker") || return 1
  actual_ssh=$(tunnel_channel_timeout_normalize "$actual_ssh") || return 1
  expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
  [ "$actual_ssh" = "$expected_ssh" ]
}

repair_wait_started_tunnel() {
  local kind=$1 pid=$2 marker=$3 forward=$4 start_time=$5 expected=$6 route=${7:-}
  local command_line current_start pgid startup_deadline=$((SECONDS + 10))
  while [ "$SECONDS" -lt "$startup_deadline" ]; do
    repair_remaining >/dev/null || return 1
    if ! repair_direct_child_owned "$pid"; then
      [ -e "/proc/$pid" ] || return 1
      sleep 0.05
      continue
    fi
    current_start=$(process_start_time "$pid") || return 1
    [ "$current_start" = "$start_time" ] || return 1
    command_line=$(process_command_line "$pid") || return 1
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ') || return 1
    if [ "$pgid" = "$pid" ] &&
        repair_started_command_matches "$kind" "$command_line" "$marker" "$forward" "$expected"; then
      process_identity_matches "$pid" "$pid" "$start_time" "$command_line" || return 1
      repair_starting_group "$pid" "$start_time" "$expected" "$marker" "" capture "$kind" "$forward" "$route" || return 1
      return 0
    fi
    sleep 0.05
  done
  return 1
}

repair_stop_starting_tunnel() {
  repair_cleanup_call repair_stop_starting_tunnel_inner "$@"
}

repair_stop_starting_tunnel_inner() {
  local pid=$1 start_time=$2 command_line pgid current_start
  local kind=${3:-} marker=${4:-} forward=${5:-} expected=${6:-} members=${7:-} route=${8:-}
  if [ ! -e "/proc/$pid" ]; then
    repair_stop_starting_group "$pid" "$start_time" "$expected" "$marker" "$members" "$kind" "$route"
    return "$?"
  fi
  repair_direct_child_owned "$pid" || return 1
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ') || return 1
  command_line=$(process_command_line "$pid") || return 1
  process_identity_matches "$pid" "$pgid" "$start_time" "$command_line" || return 1
  kill -STOP "$pid" || return 1
  if repair_kill_stopped_tunnel "$pid" "$start_time"; then
    repair_stop_starting_group "$pid" "$start_time" "$expected" "$marker" "$members" "$kind" "$route"
    return "$?"
  fi
  # Resume after a failed stop only while the same identity is still proven.
  if [ ! -e "/proc/$pid" ]; then
    repair_stop_starting_group "$pid" "$start_time" "$expected" "$marker" "$members" "$kind" "$route"
    return "$?"
  fi
  current_start=$(process_start_time "$pid") || return 1
  [ "$current_start" = "$start_time" ] || return 1
  kill -CONT "$pid" 2>/dev/null || {
    if [ ! -e "/proc/$pid" ]; then
      repair_stop_starting_group "$pid" "$start_time" "$expected" "$marker" "$members" "$kind" "$route"
      return "$?"
    fi
    return 1
  }
  return 1
}

repair_kill_stopped_tunnel() {
  local pid=$1 start_time=$2 current_start pgid sid process_output children state
  for _ in $(seq 1 40); do
    repair_cleanup_remaining || return 1
    state=$(ps -o stat= -p "$pid" | tr -d ' ') || return 1
    case "$state" in T*) break ;; Z*) return 0 ;; esac
    sleep 0.01
    repair_cleanup_remaining || return 1
  done
  case "$state" in T*) ;; *) return 1 ;; esac
  current_start=$(process_start_time "$pid") || return 1
  [ "$current_start" = "$start_time" ] || return 1
  sid=$(ps -o sid= -p "$pid" | tr -d ' ') || return 1
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ') || return 1
  if [ "$sid" = "$pid" ] && [ "$pgid" = "$pid" ]; then
    kill -KILL -- "-$pid" || return 1
  else
    process_output=$(ps -eo ppid=) || return 1
    children=$(awk -v parent="$pid" '$1 == parent {print}' <<<"$process_output") || return 1
    [ -z "$children" ] || return 1
    kill -KILL "$pid" || return 1
  fi
  for _ in $(seq 1 40); do
    repair_cleanup_remaining || return 1
    [ -e "/proc/$pid" ] || return 0
    sleep 0.05
    repair_cleanup_remaining || return 1
  done
  return 1
}

repair_cleanup_unregistered_tunnels() {
  local item kind pid marker forward cleanup_failed=0 processed=0
  for item in "${repair_unregistered_tunnels[@]}"; do
    repair_cleanup_remaining || {
      echo "tunnel cleanup stopped: stage=unregistered remaining=$((${#repair_unregistered_tunnels[@]} - processed))" >&2
      return 1
    }
    processed=$((processed + 1))
    IFS='|' read -r kind pid marker forward <<<"$item"
    if ! repair_stop_unregistered_tunnel "$kind" "$pid" "$marker" "$forward"; then
      echo "adapter tunnel rollback failed to stop unregistered $marker" >&2
      cleanup_failed=1
    fi
  done
  [ "$cleanup_failed" = 0 ]
}

repair_cleanup_created_tunnels() {
  local item kind pid marker forward start_time command_line current_start shape rollback_failed=0 processed=0
  [ "$repair_failed" = 1 ] || return 0
  for item in "${repair_created_tunnels[@]}"; do
    repair_cleanup_remaining || {
      echo "tunnel cleanup stopped: stage=created remaining=$((${#repair_created_tunnels[@]} - processed))" >&2
      return 1
    }
    processed=$((processed + 1))
    IFS='|' read -r kind pid marker forward start_time <<<"$item"
    command_line=$(process_command_line "$pid")
    case "$?" in
      0) [ -n "$command_line" ] || { rollback_failed=1; continue; } ;;
      2) continue ;;
      *)
        echo "adapter tunnel rollback failed to inspect $marker" >&2
        rollback_failed=1
        continue
        ;;
    esac
    current_start=$(process_start_time "$pid") || {
      echo "adapter tunnel rollback failed to read process identity for $marker" >&2
      rollback_failed=1
      continue
    }
    if [ "$current_start" != "$start_time" ]; then
      echo "adapter tunnel rollback refused reused PID for $marker" >&2
      rollback_failed=1
      continue
    fi
    case "$kind" in
      china)
        if ! china_tunnel_supervisor_match "$command_line" current "$marker"; then
          echo "adapter tunnel rollback failed ownership proof for $marker" >&2
          rollback_failed=1
          continue
        fi
        if ! china_tunnel_stop_supervisor "$pid" "$command_line" "$marker" 0; then
          echo "adapter tunnel rollback failed to stop $marker" >&2
          rollback_failed=1
        fi
        ;;
      peer)
        shape=current
        [ "$marker" != peer-tunnel-api ] || shape=api
        if ! peer_tunnel_supervisor_match "$command_line" "$shape" "$marker"; then
          echo "adapter tunnel rollback failed ownership proof for $marker" >&2
          rollback_failed=1
          continue
        fi
        if ! peer_kill_owned "$pid" "$command_line" "$shape" "$marker"; then
          echo "adapter tunnel rollback failed to stop $marker" >&2
          rollback_failed=1
        fi
        ;;
      generic)
        if ! generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward"; then
          echo "adapter tunnel rollback failed ownership proof for $marker" >&2
          rollback_failed=1
          continue
        fi
        if ! generic_tunnel_stop_stale "$pid" "$command_line" "$marker" "$forward"; then
          echo "adapter tunnel rollback failed to stop $marker" >&2
          rollback_failed=1
        fi
        ;;
    esac
  done
  [ "$rollback_failed" = 0 ]
}

repair_preserve_created_tunnels() {
  local item pid start_time created_item created_pid created_start matched
  local -a preserved_starting=()
  for item in "${repair_starting_tunnels[@]}"; do
    IFS='|' read -r pid start_time _ <<<"$item"
    matched=0
    for created_item in "${repair_created_tunnels[@]}"; do
      IFS='|' read -r _ created_pid _ _ created_start <<<"$created_item"
      if [ "$pid" = "$created_pid" ] && [ "$start_time" = "$created_start" ]; then
        matched=1
        break
      fi
    done
    [ "$matched" = 1 ] || preserved_starting+=("$item")
  done
  repair_starting_tunnels=("${preserved_starting[@]}")
  repair_created_tunnels=()
}

repair_cleanup_all() {
  local status=0
  if ! repair_cleanup_created_tunnels; then status=125; fi
  if ! repair_cleanup_unregistered_tunnels; then status=125; fi
  if ! repair_cleanup_starting_tunnels; then status=125; fi
  return "$status"
}

repair_exit() {
  local status=$1 cause_status=$1 failure_stage=${repair_stage:-unknown} cleanup_result=not-needed
  local migration_stops=${repair_cleanup_completed:-0} rollback_stops=0
  if [ "$status" -ne 0 ]; then repair_failed=1; fi
  if [ "$repair_committed" = 0 ]; then
    if [ "$status" -ne 0 ]; then
      echo "adapter tunnel exit status=$status stage=$failure_stage elapsed=$((SECONDS - repair_started_seconds))s cleanup=starting" >&2
    fi
    repair_stage=cleanup
    repair_cleanup_spent_us=0
    repair_cleanup_active_us=0
    repair_cleanup_completed=0
    repair_cleanup_limit_us=${repair_exit_cleanup_limit_us:-20000000}
    if repair_cleanup_call repair_cleanup_all; then
      cleanup_result=passed
    else
      cleanup_result=failed
      # Incomplete cleanup takes priority; cause_status keeps the first failure.
      status=125
    fi
    rollback_stops=$repair_cleanup_completed
  fi
  if ! repair_release_lock; then status=125; cleanup_result=failed; fi
  if [ "$status" -ne 0 ]; then
    echo "adapter tunnel exit status=$status cause_status=$cause_status stage=$failure_stage elapsed=$((SECONDS - repair_started_seconds))s cleanup=$cleanup_result migration_stops=$migration_stops rollback_stops=$rollback_stops" >&2
  fi
  exit "$status"
}

trap 'repair_exit "$?"' EXIT
trap 'repair_signal' TERM HUP INT

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
TRAINER_ADDRESSES=${TRAINER_ADDRESSES:?set TRAINER_ADDRESSES}
TRAINER_CT=${TRAINER_CT:?set TRAINER_CT}
ORCH_CT=${ORCH_CT:?set ORCH_CT}
deployment=/data/glm52-rl/deployment
config_file=${ORCH_CONFIG:-$deployment/configs/general-agent/orchestrator.toml}
nodes_file="$deployment/configs/nodes.toml"
test -f "$config_file" || {
  echo "orchestrator config is missing: $config_file" >&2
  exit 2
}
image=prime-rl:glm52-decensor
if ! lora_source_port_output=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
ports = config["ports"]
first = ports["lora_source_first"]
last = ports["lora_source_last"]
if not isinstance(first, int) or not isinstance(last, int) or first < 1 or last < first:
    raise SystemExit("invalid lora source port range")
print(f"RANGE {first} {last} {last - first + 1}")
for port in range(first, last + 1):
    print(port)
PY
); then
  echo "could not read LoRA source ports" >&2
  exit 2
fi
readarray -t lora_source_port_lines <<<"$lora_source_port_output"
range_marker= source_first= source_last= source_count= extra_range_field=
read -r range_marker source_first source_last source_count extra_range_field <<<"${lora_source_port_lines[0]:-}"
case "$range_marker:$source_first:$source_last:$source_count:$extra_range_field" in
  RANGE:*:*:*:) ;;
  *) echo "nodes config returned an invalid LoRA source range" >&2; exit 2 ;;
esac
case "$source_first:$source_last:$source_count" in
  ''|*[!0-9:]* ) echo "nodes config returned an invalid LoRA source range" >&2; exit 2 ;;
esac
[ "$source_first" -ge 1 ] && [ "$source_last" -ge "$source_first" ] &&
  [ "$source_count" -eq "$((source_last - source_first + 1))" ] || {
  echo "nodes config returned an invalid LoRA source range" >&2
  exit 2
}
lora_source_ports=("${lora_source_port_lines[@]:1}")
if [ "${#lora_source_ports[@]}" -ne "$source_count" ]; then
  echo "nodes config has no LoRA source ports" >&2
  exit 2
fi
lora_source_port_index=0
for lora_source_port in "${lora_source_ports[@]}"; do
  case "$lora_source_port" in ''|*[!0-9]*) echo "nodes config returned an invalid LoRA source port" >&2; exit 2 ;; esac
  [ "$lora_source_port" -eq "$((source_first + lora_source_port_index))" ] || {
    echo "nodes config returned an incomplete LoRA source range" >&2
    exit 2
  }
  lora_source_port_index=$((lora_source_port_index + 1))
done
adapter_port=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
value = config["ports"].get("adapter")
if not isinstance(value, int) or not 1 <= value <= 65535:
    raise SystemExit("nodes config has no valid adapter port")
print(value)
PY
)
if [ -z "$adapter_port" ]; then
  echo "nodes config has no adapter port" >&2
  exit 2
fi
if ! relay_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import sys
import tomllib
from urllib.parse import urlsplit

with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    edge = tomllib.load(stream)
ports = nodes.get("ports")
if not isinstance(ports, dict):
    raise SystemExit("nodes config has no port inventory")
relay_port = ports.get("adapter_relay")
source_first = ports.get("lora_source_first")
source_last = ports.get("lora_source_last")
relay_container = nodes.get("containers", {}).get("adapter_relay")
cache_dir = nodes.get("paths", {}).get("adapter_cache")
source_url = edge.get("weight_broadcast", {}).get("source_url")
model_name = edge.get("model", {}).get("name")
glm_model = nodes.get("models", {}).get("glm52")
if not isinstance(relay_port, int) or not 1 <= relay_port <= 65535:
    raise SystemExit("nodes config has no valid adapter relay port")
if (
    not isinstance(source_first, int)
    or not isinstance(source_last, int)
    or not 1 <= source_first <= source_last <= 65535
):
    raise SystemExit("nodes config has no valid adapter source range")
port_values = {}
for name, value in ports.items():
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise SystemExit(f"nodes port {name} is invalid")
    port_values[name] = value
source_ports = set(range(source_first, source_last + 1))
reserved_names = set(port_values) - {"lora_source_first", "lora_source_last"}
reserved_ports = {name: port_values[name] for name in reserved_names}
collisions = {
    name: value
    for name, value in reserved_ports.items()
    if value in source_ports
}
if collisions:
    raise SystemExit(f"configured port overlaps the adapter source range: {collisions}")
duplicate_ports = {}
for name, value in reserved_ports.items():
    duplicate_ports.setdefault(value, []).append(name)
duplicates = [names for names in duplicate_ports.values() if len(names) > 1]
if duplicates:
    raise SystemExit(f"configured ports are duplicated: {duplicates}")
if not isinstance(relay_container, str) or not relay_container:
    raise SystemExit("nodes config has no adapter relay container")
if not isinstance(cache_dir, str) or not cache_dir.startswith("/"):
    raise SystemExit("nodes config has no absolute adapter cache path")
if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
    raise SystemExit("orchestrator config has no canonical adapter source URL")
if not isinstance(model_name, str) or not isinstance(glm_model, str):
    raise SystemExit("model names are missing from the nodes or orchestrator config")
parsed = urlsplit(source_url)
if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.query or parsed.fragment:
    raise SystemExit("canonical adapter source URL must be loopback without query or fragment")
if parsed.port != source_first:
    raise SystemExit("canonical adapter source URL must use lora_source_first")
print(relay_port, relay_container, cache_dir, source_url, int(model_name == glm_model))
PY
); then
  echo "cannot read the configured adapter relay" >&2
  exit 2
fi
read -r relay_port relay_container relay_cache_dir relay_source_url relay_is_glm <<<"$relay_settings"
trainer_ssh_target=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
target = config["hosts"]["trainer_ssh"]
if not isinstance(target, str) or not target:
    raise SystemExit("nodes config has no trainer SSH target")
print(target)
PY
)
if [ -z "$trainer_ssh_target" ]; then
  echo "nodes config has no trainer SSH target" >&2
  exit 2
fi
transport=$(awk '
  $0 == "[rollout_transport]" { active=1; next }
  /^\[/ { active=0 }
  active && $1 == "type" { gsub(/[ "\047]/, "", $3); print $3; exit }
' "$config_file")

if ! peer_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import ipaddress
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    edge = tomllib.load(stream)

hosts = nodes.get("hosts")
inference_hosts = hosts.get("inference") if isinstance(hosts, dict) else None
orchestrator = hosts.get("orchestrator") if isinstance(hosts, dict) else None
if not isinstance(inference_hosts, list) or not all(isinstance(item, str) for item in inference_hosts):
    raise SystemExit("nodes config has no valid inference host list")
peers = [item for item in inference_hosts if item != orchestrator]
if len(peers) > 1:
    raise SystemExit("two-H200 routing supports one inference peer")
if not peers:
    print(f"inference peer: none; config={sys.argv[2]}; nodes={sys.argv[1]}", file=sys.stderr)
    print("none - 0 0 - 0 0")
    raise SystemExit

peer_name = peers[0]
peer = nodes.get("inference", {}).get(peer_name)
if not isinstance(peer, dict):
    raise SystemExit(f"inference settings for {peer_name} are missing")
host = peer.get("ssh_host")
port = peer.get("ssh_port")
user = peer.get("ssh_user")
identity = peer.get("peer_ssh_identity_file")
api_port = peer.get("api_tunnel_port")
if not isinstance(host, str):
    raise SystemExit(f"inference.{peer_name}.ssh_host is missing")
try:
    ipaddress.ip_address(host)
except ValueError as error:
    raise SystemExit(f"inference.{peer_name}.ssh_host must be an IP address") from error
if not isinstance(port, int) or not 1 <= port <= 65535:
    raise SystemExit(f"inference.{peer_name}.ssh_port is invalid")
if not isinstance(user, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", user) is None:
    raise SystemExit(f"inference.{peer_name}.ssh_user is invalid")
if not isinstance(identity, str) or not identity.startswith("/") or any(part in {"", ".", ".."} for part in identity.split("/")[1:]):
    raise SystemExit(f"inference.{peer_name}.peer_ssh_identity_file is invalid")
if not isinstance(api_port, int) or not 1 <= api_port <= 65535:
    raise SystemExit(f"inference.{peer_name}.api_tunnel_port is invalid")

client = edge.get("model", {}).get("client", {})
urls = []
for field in ("base_url", "admin_base_url"):
    value = client.get(field, []) if isinstance(client, dict) else []
    if isinstance(value, str):
        urls.append(value)
    elif isinstance(value, list):
        urls.extend(item for item in value if isinstance(item, str))
expected = f"http://127.0.0.1:{api_port}/v1"
enabled = int(expected in urls)
if not enabled:
    raise SystemExit(f"inference peer {peer_name} is configured in {sys.argv[1]}, but {sys.argv[2]} does not use {expected}; no tunnels changed")
print(f"inference peer: {peer_name} enabled; config={sys.argv[2]}; API={expected}", file=sys.stderr)
print(peer_name, host, port, user, identity, api_port, enabled)
PY
); then
  echo "cannot read the configured inference peer" >&2
  exit 2
fi
read -r peer_name peer_host peer_ssh_port peer_ssh_user peer_ssh_identity peer_api_port peer_enabled <<<"$peer_settings"
if [ "$peer_enabled" = 1 ] && [ "$relay_is_glm" != 1 ]; then
  echo "adapter relay peer mode is only supported for the configured GLM model" >&2
  exit 2
fi

declare -A peer_supervisor_pids=()
declare -A peer_supervisor_starts=()
declare -A peer_supervisor_commands=()
declare -A peer_child_pids=()
declare -A peer_child_starts=()
declare -A peer_child_commands=()
declare -A peer_child_shapes=()
declare -A peer_child_forwards=()

if ! secondary_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import ipaddress
import re
import sys
import tomllib
from urllib.parse import urlsplit


def port_value(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise SystemExit(f"judges.secondary.{name} is invalid")
    return value


with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    active_config = tomllib.load(stream)

judges = nodes.get("judges")
external = judges.get("external") if isinstance(judges, dict) else None
secondary = judges.get("secondary") if isinstance(judges, dict) else None
if not isinstance(external, dict) or not isinstance(secondary, dict):
    raise SystemExit("nodes config has no complete judge inventory")

host = secondary.get("host")
try:
    ipaddress.ip_address(host)
except (TypeError, ValueError) as error:
    raise SystemExit("judges.secondary.host must be an IP address") from error
ssh_port = port_value(secondary.get("ssh_port"), "ssh_port")
local_api_port = port_value(secondary.get("local_api_port"), "local_api_port")
remote_api_port = port_value(secondary.get("remote_api_port"), "remote_api_port")
ssh_user = secondary.get("ssh_user")
if not isinstance(ssh_user, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", ssh_user) is None:
    raise SystemExit("judges.secondary.ssh_user is invalid")
identity = secondary.get("ssh_identity_file")
if (
    not isinstance(identity, str)
    or not identity.startswith("/")
    or re.fullmatch(r"/[A-Za-z0-9_./-]+", identity) is None
    or any(part in {"", ".", ".."} for part in identity.split("/")[1:])
):
    raise SystemExit("judges.secondary.ssh_identity_file is invalid")
api_url = secondary.get("api_url")
expected_api_url = f"http://127.0.0.1:{local_api_port}/v1"
parsed_api_url = urlsplit(api_url) if isinstance(api_url, str) else None
if (
    api_url != expected_api_url
    or parsed_api_url is None
    or parsed_api_url.query
    or parsed_api_url.fragment
):
    raise SystemExit("judges.secondary.api_url must be the configured local API URL")
model = secondary.get("model")
external_model = external.get("model")
if not isinstance(model, str) or model != external_model:
    raise SystemExit("judges.secondary.model must match judges.external.model")
data_parallel_size = secondary.get("data_parallel_size")
tensor_parallel_size = secondary.get("tensor_parallel_size")
if (
    not isinstance(data_parallel_size, int)
    or isinstance(data_parallel_size, bool)
    or data_parallel_size != 2
):
    raise SystemExit("judges.secondary.data_parallel_size must be 2")
if (
    not isinstance(tensor_parallel_size, int)
    or isinstance(tensor_parallel_size, bool)
    or tensor_parallel_size != 4
    or tensor_parallel_size * data_parallel_size != 8
):
    raise SystemExit("judges.secondary.tensor_parallel_size must be 4 for eight GPUs")


def values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from values(item)
    elif isinstance(value, list):
        for item in value:
            yield from values(item)
    else:
        yield value


active = int(any(value == api_url for value in values(active_config)))
print(host, ssh_port, ssh_user, identity, local_api_port, remote_api_port, api_url, active)
PY
); then
  echo "cannot read the configured secondary judge" >&2
  exit 2
fi
read -r secondary_host secondary_ssh_port secondary_ssh_user secondary_ssh_identity \
  secondary_local_api_port secondary_remote_api_port secondary_api_url secondary_enabled <<<"$secondary_settings"
secondary_ssh_target="$secondary_ssh_user@$secondary_host"
secondary_ssh_options=(
  -o IdentitiesOnly=yes
  -i "$secondary_ssh_identity"
  -p "$secondary_ssh_port"
)

if [ "$tunnels_only" = 0 ] && grep -q 'refusals-classifier-eval-v1' "$config_file"; then
  bash "$deployment/training_ops/runtime/h200/start_refusals_classifier.sh"
fi

generic_tunnel_ssh_match() {
  local command_line=$1 via=$2 forward=$3 comp=$4 index
  shift 4
  local -a ssh_options=("$@")
  local -a command_args expected_args
  expected_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=$comp
    "${ssh_options[@]}"
    -L "$forward"
  )
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  [ "${#command_args[@]}" -eq "$(( ${#expected_args[@]} + 2 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[-1]}" = "$via" ]
}

generic_tunnel_supervisor_match() {
  local command_line=$1 marker=$2 via=$3 forward=$4 comp=$5
  shift 5
  local -a ssh_options=("$@")
  local ssh_command
  case "$command_line" in *"while true; do"*"done # $marker") ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  generic_tunnel_ssh_match "$ssh_command" "$via" "$forward" "$comp" \
    "${ssh_options[@]}"
}

pgrep_matches() {
  local pattern=$1 status
  pgrep -f -- "$pattern"
  status=$?
  [ "$status" -eq 1 ] && return 0
  return "$status"
}

tunnel_live_group() {
  local process_output
  process_output=$(ps -eo pid=,ppid=,pgid=,stat= 2>/dev/null) || return 1
  awk -v group="$1" '$3 == group && $4 !~ /^Z/ {print $1, $2}' <<<"$process_output"
}

tunnel_process_argv() {
  base64 -w0 "/proc/$1/cmdline"
}

tunnel_stop_identity_matches() {
  local pgid=$1 output pid ppid current seen=0
  [ "${tunnel_stop_pgid:-}" = "$pgid" ] || return 1
  process_identity_matches "$tunnel_stop_pid" "$pgid" "$tunnel_stop_start" "$tunnel_stop_command" || return 1
  output=$(tunnel_live_group "$pgid") || return 1
  while read -r pid ppid; do
    [ -n "$pid" ] || continue
    [ "$pid" != "$tunnel_stop_pid" ] || seen=1
    [ -n "${tunnel_stop_starts[$pid]:-}" ] || return 1
    [ "$(process_start_time "$pid")" = "${tunnel_stop_starts[$pid]}" ] || return 1
    current=$(tunnel_process_argv "$pid") || return 1
    [ -n "$current" ] && [ "$current" = "${tunnel_stop_args[$pid]}" ] || return 1
    [ "$ppid" = "${tunnel_stop_parents[$pid]}" ] || return 1
  done <<<"$output"
  [ "$seen" = 1 ]
}

tunnel_prepare_group_stop() {
  local pgid=$1 pid=$2 start=$3 command_line=$4 output member ppid
  shift 4
  [ -n "$command_line" ] || command_line=$(process_command_line "$pid") || return 1
  declare -gA tunnel_stop_args=() tunnel_stop_starts=() tunnel_stop_parents=()
  tunnel_stop_pgid=$pgid
  tunnel_stop_pid=$pid
  tunnel_stop_start=$start
  tunnel_stop_command=$command_line
  output=$(tunnel_live_group "$pgid") || return 1
  while read -r member ppid; do
    [ -n "$member" ] || continue
    tunnel_stop_args[$member]=$(tunnel_process_argv "$member") || return 1
    tunnel_stop_starts[$member]=$(process_start_time "$member") || return 1
    tunnel_stop_parents[$member]=$ppid
  done <<<"$output"
  "$@" || return 1
  tunnel_stop_identity_matches "$pgid"
}

tunnel_wait_group_exit() {
  local pgid=$1 process_output attempt phase
  case "$pgid" in ''|*[!0-9]*|0|1) return 1 ;; esac
  for phase in term kill; do
    for attempt in $(seq 1 20); do
      repair_cleanup_remaining || {
        echo "tunnel cleanup time limit reached: stage=$phase group=$pgid" >&2
        return 1
      }
      process_output=$(tunnel_live_group "$pgid") || return 1
      [ -n "$process_output" ] || return 0
      sleep 0.25
      repair_cleanup_remaining || return 1
    done
    [ "$phase" = term ] || break
    if ! tunnel_stop_identity_matches "$pgid"; then
      echo "tunnel group $pgid ownership changed; no KILL sent; remaining PIDs: $(awk 'NR <= 16 {printf "%s ", $1}' <<<"$process_output")" >&2
      return 1
    fi
    repair_cleanup_remaining || return 1
    kill -KILL -- "-$pgid" || return 1
  done
  echo "tunnel group $pgid did not exit; remaining PIDs: $(awk 'NR <= 16 {printf "%s ", $1}' <<<"$process_output")" >&2
  return 1
}

generic_tunnel_listener_ready() {
  local lp=$1 marker=$2 via=$3 forward=$4 comp=$5
  shift 5
  local -a ssh_options=("$@")
  local listener pid ppid owner_command supervisor_command ss_output
  generic_listener_supervisor_pid=
  ss_output=$(ss -H -ltnp 2>/dev/null) || return 2
  listener=$(awk -v port=":$lp" '$4 ~ port "$" {print; exit}' <<<"$ss_output") || return 2
  [ -n "$listener" ] || return 1
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$pid" in ''|*[!0-9]*) return 2 ;; esac
  owner_command=$(ps -o args= -p "$pid" 2>/dev/null) || return 2
  [ -n "$owner_command" ] || return 2
  generic_tunnel_ssh_match "$owner_command" "$via" "$forward" "$comp" "${ssh_options[@]}" || return 2
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || return 2
  case "$ppid" in ''|*[!0-9]*) return 2 ;; esac
  [ "$ppid" != 1 ] || return 2
  supervisor_command=$(ps -o args= -p "$ppid" 2>/dev/null) || return 2
  [ -n "$supervisor_command" ] || return 2
  generic_tunnel_supervisor_match "$supervisor_command" "$marker" "$via" "$forward" "$comp" \
    "${ssh_options[@]}" || return 2
  generic_listener_supervisor_pid=$ppid
  local probe_seconds
  probe_seconds=$(repair_timeout 5) || return 1
  if timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      bash -c ":</dev/tcp/127.0.0.1/$lp" 2>/dev/null; then
    return 0
  fi
  return 1
}

tunnel_inner() { # local_port  target_host:port  ssh_target  [compress]
  local lp=$1 tgt=$2 via=$3 comp=${4:-no}
  shift $(( $# >= 4 ? 4 : $# ))
  local -a ssh_options=("$@")
  local ssh_text=""
  [ "${#ssh_options[@]}" -eq 0 ] || ssh_text=$(printf ' %q' "${ssh_options[@]}")
  local marker="zmq-tunnel-$lp" forward="127.0.0.1:$lp:$tgt"
  local pid command_line matching_pids owner_index=-1 index owner_status
  local listener_owner_pid= saved_owner_pid saved_owner_start final_pid final_count final_command
  local -a exact_pids=() exact_commands=()
  matching_pids=$(pgrep_matches "$marker") || return 1
  while read -r pid; do
    [ -n "$pid" ] || continue
    if ! command_line=$(ps -o args= -p "$pid" 2>/dev/null); then
      [ ! -e "/proc/$pid" ] && continue
      echo "could not inspect generic tunnel process $pid for $marker" >&2
      return 1
    fi
    if generic_tunnel_supervisor_match "$command_line" "$marker" "$via" "$forward" "$comp" \
      "${ssh_options[@]}"; then
      exact_pids+=("$pid")
      exact_commands+=("$command_line")
      if generic_tunnel_listener_ready "$lp" "$marker" "$via" "$forward" "$comp" \
        "${ssh_options[@]}"; then
        listener_owner_pid=$generic_listener_supervisor_pid
        [ "$generic_listener_supervisor_pid" = "$pid" ] &&
          [ "$owner_index" -lt 0 ] && owner_index=$((${#exact_pids[@]} - 1))
      else
        owner_status=$?
        [ "$owner_status" -eq 1 ] || return 1
      fi
      continue
    fi
    generic_tunnel_stop_stale "$pid" "$command_line" "$marker" "$forward" || return 1
  done <<<"$matching_pids"
  if [ "${#exact_pids[@]}" -gt 0 ]; then
    if [ "$owner_index" -lt 0 ] && [ -n "$listener_owner_pid" ]; then
      for index in "${!exact_pids[@]}"; do
        if [ "${exact_pids[index]}" = "$listener_owner_pid" ]; then
          owner_index=$index
          break
        fi
      done
      [ "$owner_index" -ge 0 ] || {
        echo "generic tunnel listener :$lp is owned by an unrelated process" >&2
        return 1
      }
    fi
    for index in "${!exact_pids[@]}"; do
      [ "$index" -eq "$owner_index" ] && continue
      generic_tunnel_stop_stale "${exact_pids[index]}" "${exact_commands[index]}" \
        "$marker" "$forward" || return 1
    done
    if [ "$owner_index" -ge 0 ]; then
      saved_owner_pid=${exact_pids[owner_index]}
      saved_owner_start=$(process_start_time "$saved_owner_pid") || return 1
      matching_pids=$(pgrep_matches "$marker") || return 1
      final_pid=
      final_count=0
      final_command=
      while read -r pid; do
        [ -n "$pid" ] || continue
        if ! command_line=$(ps -o args= -p "$pid" 2>/dev/null); then
          [ ! -e "/proc/$pid" ] && continue
          return 1
        fi
        if generic_tunnel_supervisor_match "$command_line" "$marker" "$via" "$forward" "$comp" \
            "${ssh_options[@]}"; then
          final_count=$((final_count + 1))
          final_pid=$pid
          final_command=$command_line
        fi
      done <<<"$matching_pids"
      [ "$final_count" -eq 1 ] || return 1
      [ "$final_pid" = "$saved_owner_pid" ] || return 1
      [ "$(process_start_time "$final_pid")" = "$saved_owner_start" ] || return 1
      generic_tunnel_listener_ready "$lp" "$marker" "$via" "$forward" "$comp" \
        "${ssh_options[@]}" || return 1
      [ "$generic_listener_supervisor_pid" = "$saved_owner_pid" ] || return 1
      return 0
    fi
  fi
  repair_launch_tunnel generic "$marker" "$forward" "while true; do
      ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
          -o IPQoS=none -o Compression=$comp$ssh_text \
          -L 127.0.0.1:$lp:$tgt $via
      sleep 3
    done # $marker" || return 1
  for _ in $(seq 1 30); do
    repair_remaining >/dev/null || return 1
    if generic_tunnel_listener_ready "$lp" "$marker" "$via" "$forward" "$comp" \
      "${ssh_options[@]}"; then
      return 0
    fi
    repair_sleep 1 || return 1
  done
  echo "generic tunnel :$lp is not owned by the configured supervisor or is not usable" >&2
  return 1
}

tunnel() {
  local lock_status
  if [ "${repair_lock_held:-0}" = 1 ]; then
    tunnel_inner "$@"
    return $?
  fi
  repair_lock_file=${TUNNEL_LOCK_FILE:-$repair_lock_file}
  repair_acquire_lock || return 1
  tunnel_inner "$@"
  lock_status=$?
  if ! repair_release_lock; then
    echo "could not release generic tunnel repair lock" >&2
    lock_status=1
  fi
  return "$lock_status"
}

generic_tunnel_stale_supervisor_match() {
  local command_line=$1 marker=$2 forward=$3 ssh_command
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  [ "$(grep -o -- '-L ' <<<"$ssh_command" | wc -l)" -eq 1 ] || return 1
  tunnel_command_has "$ssh_command" "$forward"
}

generic_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 marker=$3 forward=$4
  local group_command expected_ssh group_listing group_seen=0
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        generic_tunnel_stale_supervisor_match "$group_command" "$marker" "$forward" || return 1
        [ "$(tunnel_supervisor_ssh_command "$group_command" "$marker")" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        [ "$group_command" = "$expected_ssh" ] || return 1
        ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ]
}

generic_tunnel_stop_stale() {
  repair_cleanup_call generic_tunnel_stop_stale_inner "$@"
}

generic_tunnel_stop_stale_inner() {
  local pid=$1 command_line=$2 marker=$3 forward=$4 pgid expected_start
  generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward" || return 0
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  expected_start=$(process_start_time "$pid") || {
    echo "could not read stale generic tunnel process identity for $marker" >&2
    return 1
  }
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$")
      echo "could not prove stale generic tunnel group for $marker" >&2
      return 1
      ;;
    *)
      if ! generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward"; then
        echo "stale generic tunnel group $pgid failed ownership proof for $marker" >&2
        return 1
      fi
      if ! generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward"; then
        echo "stale generic tunnel group changed before stop for $marker" >&2
        return 1
      fi
      process_identity_matches "$pid" "$pgid" "$expected_start" "$command_line" || {
        echo "stale generic tunnel process changed before stop for $marker" >&2
        return 1
      }
      tunnel_prepare_group_stop "$pgid" "$pid" "$expected_start" "$command_line" \
        generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward" || return 1
      repair_cleanup_remaining || return 1
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop stale generic tunnel group $pgid for $marker" >&2
        return 1
      }
      tunnel_wait_group_exit "$pgid" || {
        echo "stale generic tunnel group $pgid did not exit for $marker" >&2
        return 1
      }
      ;;
  esac
}

peer_tunnel_listener_owner() {
  local candidate_pid=$1 expected_command=$2 listener owner_pid owner_ppid owner_command ss_output
  ss_output=$(ss -H -ltnp 2>/dev/null) || return 2
  listener=$(awk -v port=":$peer_api_port" '$4 ~ port "$" {print; exit}' <<<"$ss_output") || return 2
  [ -n "$listener" ] || return 1
  owner_pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$owner_pid" in ''|*[!0-9]*) return 2 ;; esac
  owner_command=$(ps -o args= -p "$owner_pid" 2>/dev/null) || return 2
  [ -n "$owner_command" ] || return 2
  peer_tunnel_api_variant_match "$owner_command" || return 2
  owner_ppid=$(ps -o ppid= -p "$owner_pid" 2>/dev/null | tr -d ' ') || return 2
  [ "$owner_ppid" = "$candidate_pid" ] || return 1
  peer_tunnel_supervisor_forwards_match "$expected_command" api peer-tunnel-api \
    "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" || return 2
}

peer_tunnel_api_listener_owner() {
  local listener owner_pid owner_ppid owner_command supervisor_command ss_output owner_start
  peer_api_listener_owner_pid=
  peer_api_listener_owner_start=
  ss_output=$(ss -H -ltnp 2>/dev/null) || return 2
  listener=$(awk -v port=":$peer_api_port" '$4 ~ port "$" {print; exit}' <<<"$ss_output") || return 2
  [ -n "$listener" ] || return 1
  owner_pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$owner_pid" in ''|*[!0-9]*) return 2 ;; esac
  owner_command=$(ps -o args= -p "$owner_pid" 2>/dev/null) || return 2
  [ -n "$owner_command" ] || return 2
  peer_tunnel_ssh_args_match "$owner_command" api || return 2
  owner_ppid=$(ps -o ppid= -p "$owner_pid" 2>/dev/null | tr -d ' ') || return 2
  case "$owner_ppid" in ''|*[!0-9]*|1) return 2 ;; esac
  supervisor_command=$(ps -o args= -p "$owner_ppid" 2>/dev/null) || return 2
  [ -n "$supervisor_command" ] || return 2
  peer_tunnel_supervisor_forwards_match "$supervisor_command" api peer-tunnel-api \
    "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" || return 2
  owner_start=$(process_start_time "$owner_ppid") || return 2
  [ -n "$owner_start" ] || return 2
  peer_api_listener_owner_pid=$owner_ppid
  peer_api_listener_owner_start=$owner_start
}

peer_tunnel_connection() {
  local connection_name=$1 signature=$2
  shift 2
  local -a forwards=("$@")

  local -a ssh_args=(
    -NT
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -i "$peer_ssh_identity"
    -p "$peer_ssh_port"
  )
  if [ "$connection_name" = api ]; then
    peer_tunnel_api_args current
    ssh_args=("${peer_api_args[@]:1:${#peer_api_args[@]}-4}")
  fi
  local forward_text ssh_text peer_target marker pid command_line final_command
  local matching_pids matching_count keep_index=0 owner_index=-1 final_count owner_status final_pid child_status
  local listener_status listener_owner_pid= listener_owner_start= current_owner_start
  local owner_count=0 index
  local -a candidate_pids=() candidate_commands=() live_pids=() live_commands=()
  local expected_shape=current
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  marker="peer-tunnel-$connection_name"
  repair_remaining >/dev/null || return 1
  [ "$connection_name" = api ] && expected_shape=api
  if [ "${restart_tunnels:-0}" = 0 ]; then
    if [ "$connection_name" = api ]; then
      listener_status=0
      peer_tunnel_api_listener_owner || listener_status=$?
      case "$listener_status" in
        0) listener_owner_pid=$peer_api_listener_owner_pid ;;
        1) listener_owner_pid= ;;
        *) return 1 ;;
      esac
    fi
    matching_pids=$(pgrep_matches "$signature") || return 1
    while read -r pid; do
      [ -n "$pid" ] || continue
      if ! command_line=$(ps -o args= -p "$pid" 2>/dev/null); then
        [ ! -e "/proc/$pid" ] && continue
        echo "could not inspect peer tunnel process $pid for $marker" >&2
        return 1
      fi
      if peer_tunnel_supervisor_forwards_match "$command_line" "$expected_shape" "$marker" "${forwards[@]}"; then
        candidate_pids+=("$pid")
        candidate_commands+=("$command_line")
        if [ "$connection_name" = api ] && [ "$pid" = "$listener_owner_pid" ]; then
          owner_index=$(( ${#candidate_pids[@]} - 1 ))
          owner_count=$((owner_count + 1))
        fi
        continue
      fi
      peer_kill_owned "$pid" "$command_line" "$expected_shape" "$marker" "${forwards[@]}" || return 1
    done <<<"$matching_pids"
    if [ "$connection_name" != api ] && [ "${#candidate_pids[@]}" -gt 0 ]; then
      for index in "${!candidate_pids[@]}"; do
        if peer_tunnel_supervisor_child_identity "${candidate_pids[index]}" "$expected_shape" "$marker" \
          "${forwards[@]}"; then
          child_status=0
        else
          child_status=$?
        fi
        case "$child_status" in
          0)
            live_pids+=("${candidate_pids[index]}")
            live_commands+=("${candidate_commands[index]}")
            ;;
          1)
            peer_kill_owned "${candidate_pids[index]}" "${candidate_commands[index]}" \
              "$expected_shape" "$marker" "${forwards[@]}" || return 1
            ;;
          *)
            echo "could not inspect peer tunnel child for $marker" >&2
            return 1
            ;;
        esac
      done
      candidate_pids=("${live_pids[@]}")
      candidate_commands=("${live_commands[@]}")
    fi
    matching_count=${#candidate_pids[@]}
    if [ "$connection_name" = api ] && [ -n "$listener_owner_pid" ]; then
      [ "$owner_count" -eq 1 ] || {
        echo "peer API listener owner is not one exact candidate" >&2
        return 1
      }
      current_owner_start=$(process_start_time "$listener_owner_pid") || return 1
      [ "$current_owner_start" = "$peer_api_listener_owner_start" ] || {
        echo "peer API listener supervisor changed during inspection" >&2
        return 1
      }
    fi
    if [ "$matching_count" -gt 0 ]; then
      if [ "$owner_count" -gt 1 ]; then
        echo "multiple peer tunnel supervisors own the API listener" >&2
        return 1
      fi
      if [ "$connection_name" = api ] && [ -n "$listener_owner_pid" ] && [ "$owner_count" -eq 0 ]; then
        echo "peer API listener belongs to an unexpected supervisor" >&2
        return 1
      fi
      if [ "$connection_name" = api ] && [ "$owner_count" = 0 ]; then
        for index in "${!candidate_pids[@]}"; do
          peer_kill_owned "${candidate_pids[index]}" "${candidate_commands[index]}" \
            "$expected_shape" "$marker" "${forwards[@]}" || return 1
        done
        matching_count=0
      fi
    fi
    if [ "$matching_count" -gt 0 ]; then
      [ "$owner_index" -ge 0 ] && keep_index=$owner_index
      for index in "${!candidate_pids[@]}"; do
        [ "$index" -eq "$keep_index" ] && continue
        peer_kill_owned "${candidate_pids[index]}" "${candidate_commands[index]}" \
          "$expected_shape" "$marker" "${forwards[@]}" || return 1
      done
      matching_pids=$(pgrep_matches "$signature") || return 1
      final_count=0
      final_command=
      final_pid=
      while read -r pid; do
        [ -n "$pid" ] || continue
        if ! command_line=$(ps -o args= -p "$pid" 2>/dev/null); then
          [ ! -e "/proc/$pid" ] && continue
          echo "could not inspect final peer tunnel process $pid for $marker" >&2
          return 1
        fi
        if peer_tunnel_supervisor_forwards_match "$command_line" "$expected_shape" \
            "$marker" "${forwards[@]}"; then
          final_count=$((final_count + 1))
          final_pid=$pid
          final_command=$command_line
        fi
      done <<<"$matching_pids"
      [ "$final_count" -eq 1 ] || {
        echo "peer tunnel reconciliation did not leave one exact supervisor for $marker" >&2
        return 1
      }
      if [ "$owner_index" -ge 0 ]; then
        current_owner_start=$(process_start_time "$final_pid") || return 1
        [ "$final_pid" = "$listener_owner_pid" ] || return 1
        [ "$current_owner_start" = "$peer_api_listener_owner_start" ] || return 1
        peer_tunnel_listener_owner "$final_pid" "$final_command" || return 1
      fi
      return 0
    fi
  fi
  repair_launch_tunnel peer "$marker" "" "while true; do
      ssh$ssh_text$forward_text $peer_target
      sleep 3
    done # $marker" || return 1
}

peer_tunnel_forward_shape() {
  local command_line=$1 shape=$2
  local forward_total spec source_port destination_port
  local -a specs seen=()
  forward_total=$(grep -o -- '-[LR] ' <<<"$command_line" | wc -l)
  case "$shape:$forward_total" in
    cleanup:1|cleanup:2|cleanup:4|current:2|api:1) ;;
    *) return 1 ;;
  esac
  mapfile -t specs < <(grep -oE -- '-[LR] 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#specs[@]}" -eq "$forward_total" ] || return 1
  for spec in "${specs[@]}"; do
    case "$spec" in
      -R\ 127.0.0.1:*:127.0.0.1:*)
        source_port=${spec#-R 127.0.0.1:}
        source_port=${source_port%%:*}
        destination_port=${spec##*:}
        case "$destination_port" in
          ''|*[!0-9]*) return 1 ;;
        esac
        [ "$destination_port" -ge 1 ] && [ "$destination_port" -le 65535 ] || return 1
        if [ "$shape" != cleanup ] && [ "$destination_port" != "$relay_port" ]; then
          return 1
        fi
        case " ${lora_source_ports[*]} " in
          *" $source_port "*) ;;
          *) return 1 ;;
        esac
        case " ${seen[*]} " in
          *" $source_port "*) return 1 ;;
          *) seen+=("$source_port") ;;
        esac
        ;;
      -L\ 127.0.0.1:"$peer_api_port":127.0.0.1:8000)
        [ "$forward_total" -eq 1 ] || return 1
        ;;
      *) return 1 ;;
    esac
  done
}

peer_tunnel_marker_match() {
  local command_line=$1 marker=${2:-}
  if [ -n "$marker" ]; then
    case "$command_line" in *"# $marker") ;; *) return 1 ;; esac
  else
    [[ "$command_line" =~ \#\ peer-tunnel-(api|lora-[0-9]+)[[:space:]]*$ ]] || return 1
  fi
}

peer_tunnel_supervisor_match() {
  local command_line=$1 shape=$2 marker=${3:-}
  local ssh_command
  peer_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  if [ -z "$marker" ]; then
    marker=$(sed -n 's/.*# \(peer-tunnel-\(api\|lora-[0-9][0-9]*\)\)$/\1/p' <<<"$command_line")
  fi
  [ -n "$marker" ] || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  peer_tunnel_ssh_args_match "$ssh_command" "$shape"
}

peer_tunnel_supervisor_forwards_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  peer_tunnel_supervisor_match "$command_line" "$shape" "$marker" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

peer_tunnel_stale_supervisor_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  peer_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  peer_tunnel_forward_shape "$command_line" "$shape" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

tunnel_supervisor_ssh_command() {
  local command_line=$1 marker=$2 normalized prefix body suffix ssh_command
  normalized=$(tr '\n\t' '  ' <<<"$command_line" | sed -E 's/[[:space:]]+/ /g; s/^ +//; s/ +$//')
  case "$normalized" in *"while true; do "*) ;; *) return 1 ;; esac
  prefix=${normalized%%while true; do *}
  case "$prefix" in
    *';'*|*'|'*|*'&'*|*'`'*|*'$('*|*'<'*|*'>'*|*'('*|*')'*|*'{'*|*'}'*) return 1 ;;
    ""|"bash -c "|"nohup bash -c ") ;;
    *) return 1 ;;
  esac
  body=${normalized#*while true; do }
  case "$body" in
    "ssh "*"; sleep 3; done # $marker") suffix="; sleep 3; done # $marker" ;;
    "ssh "*" sleep 3; done # $marker") suffix=" sleep 3; done # $marker" ;;
    "ssh "*" sleep 3 done # $marker") suffix=" sleep 3 done # $marker" ;;
    *) return 1 ;;
  esac
  ssh_command=${body%"$suffix"}
  case "$ssh_command" in
    *';'*|*'|'*|*'&'*|*'`'*|*'$('*|*'<'*|*'>'*|*'('*|*')'*|*'{'*|*'}'*|*'#'*) return 1 ;;
  esac
  printf '%s\n' "$ssh_command"
}

tunnel_channel_timeout_normalize() {
  sed 's/ChannelTimeout=\\\*-tcpip=45s/ChannelTimeout=*-tcpip=45s/g' <<<"$1"
}

china_tunnel_proxy_ssh_command() {
  local route_target=${trainer_ssh_target:-} hostname= port= proxyjump=
  local key value route_output
  if [ "${china_tunnel_proxy_target:-}" = "$route_target" ]; then
    [ -n "${china_tunnel_proxy_command:-}" ] || return 1
    printf '%s\n' "$china_tunnel_proxy_command"
    return 0
  fi
  china_tunnel_proxy_command=
  [ -n "$route_target" ] || return 1
  route_output=$(ssh -G "$route_target" 2>/dev/null) || return 2
  china_tunnel_proxy_target=$route_target
  while read -r key value _; do
    case "$key" in
      hostname) hostname=$value ;;
      port) port=$value ;;
      proxyjump) proxyjump=$value ;;
    esac
  done <<<"$route_output"
  case "$hostname" in ''|*[!A-Za-z0-9_.:-]*) return 1 ;; esac
  case "$port" in ''|*[!0-9]*) return 1 ;; esac
  [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || return 1
  case "$proxyjump" in
    ''|none|*,*|*@*|*:*|*[!A-Za-z0-9_.-]*) return 1 ;;
  esac
  china_tunnel_proxy_command="ssh -W [$hostname]:$port $proxyjump"
  printf '%s\n' "$china_tunnel_proxy_command"
}

china_tunnel_proxy_child_match() {
  local command_line=$1 expected normalized=$1
  expected=$(china_tunnel_proxy_ssh_command) || return 1
  case "$normalized" in
    /usr/bin/ssh\ *) normalized="ssh ${normalized#/usr/bin/ssh }" ;;
  esac
  [ "$normalized" = "$expected" ]
}

peer_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 shape=$3 marker=$4
  local group_command expected_ssh actual_ssh group_seen=0 normalized_command process_output group_listing
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
  process_output=$(ps -eo pgid=,args=) || return 1
  group_listing=$(awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' <<<"$process_output") || return 1
  while IFS= read -r group_command; do
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        peer_tunnel_stale_supervisor_match "$group_command" "$shape" "$marker" || return 1
        actual_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$marker") || return 1
        actual_ssh=$(tunnel_channel_timeout_normalize "$actual_ssh") || return 1
        [ "$actual_ssh" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        [ "$normalized_command" = "$expected_ssh" ] || return 1
        ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ]
}

peer_tunnel_ssh_args_match() {
  local command_line=$1 shape=$2
  if [ "$shape" = api ]; then
    peer_tunnel_api_variant_match "$command_line" current
    return $?
  fi
  local index forward_count forward_start target_index remaining
  local -a command_args expected_args
  command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
  expected_args=(
    -NT
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -i "$peer_ssh_identity"
    -p "$peer_ssh_port"
  )
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  target_index=$((${#command_args[@]} - 1))
  [ "$target_index" -gt "$(( ${#expected_args[@]} + 1 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[target_index]}" = "$peer_ssh_user@$peer_host" ] || return 1
  forward_start=$((1 + ${#expected_args[@]}))
  remaining=$((target_index - forward_start))
  [ "$remaining" -gt 0 ] && [ "$((remaining % 2))" -eq 0 ] || return 1
  forward_count=$((remaining / 2))
  for ((index = forward_start; index < target_index; index += 2)); do
    case "${command_args[index]}" in -L|-R) ;; *) return 1 ;; esac
    case "${command_args[index + 1]}" in
      127.0.0.1:[0-9]*:127.0.0.1:[0-9]*) ;;
      *) return 1 ;;
    esac
  done
  peer_tunnel_forward_shape "$command_line" "$shape" || return 1
  [ "$forward_count" -eq "$(grep -o -- '-[LR] ' <<<"$command_line" | wc -l)" ]
}

peer_tunnel_api_args() {
  case "$1" in
    current)
      peer_api_args=(ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3
        -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no
        -i "$peer_ssh_identity" -p "$peer_ssh_port"
        -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" "$peer_ssh_user@$peer_host") ;;
    legacy-timeout)
      peer_api_args=(ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3
        -o 'ChannelTimeout=*-tcpip=45s' -o TCPKeepAlive=yes -o IPQoS=none
        -o Compression=no -i "$peer_ssh_identity" -p "$peer_ssh_port"
        -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" "$peer_ssh_user@$peer_host") ;;
    compression)
      peer_api_args=(ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3
        -o 'ChannelTimeout=*-tcpip=45s' -o TCPKeepAlive=yes -o IPQoS=none
        -i "$peer_ssh_identity" -p "$peer_ssh_port"
        -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" "$peer_ssh_user@$peer_host") ;;
    channel-compression)
      peer_api_args=(ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3
        -o TCPKeepAlive=yes -o IPQoS=none -i "$peer_ssh_identity" -p "$peer_ssh_port"
        -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" "$peer_ssh_user@$peer_host") ;;
    *) return 1 ;;
  esac
}

peer_tunnel_api_payload() {
  peer_tunnel_api_args "$1" || return 1
  local ssh_text
  printf -v ssh_text ' %q' "${peer_api_args[@]:1}"
  printf 'while true; do\n      ssh%s\n      sleep 3\n    done # peer-tunnel-api' "$ssh_text"
}

peer_tunnel_api_variant_match() {
  local command_line=$1 variant expected
  command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
  for variant in ${2:-current legacy-timeout compression channel-compression}; do
    peer_tunnel_api_args "$variant" || return 1
    expected="${peer_api_args[*]}"
    [ "$command_line" != "$expected" ] || return 0
  done
  return 1
}

peer_tunnel_api_supervisor_variant_match() {
  local command_line=$1 marker=$2 forward=$3 variant payload
  [ "$marker" = peer-tunnel-api ] || return 1
  [ "$forward" = "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" ] || return 1
  for variant in current legacy-timeout compression channel-compression; do
    payload=$(peer_tunnel_api_payload "$variant") || return 1
    [ "$command_line" != "bash -c $payload" ] || return 0
  done
  return 1
}

peer_tunnel_api_read_args() {
  mapfile -d '' -t peer_api_actual <"/proc/$1/cmdline" || return 1
  [ "${#peer_api_actual[@]}" -gt 0 ]
}

peer_tunnel_api_pid_match() {
  local pid=$1 kind=$2 variant index equal payload
  peer_tunnel_api_read_args "$pid" || return 1
  for variant in ${3:-current legacy-timeout compression channel-compression}; do
    peer_tunnel_api_args "$variant" || return 1
    if [ "$kind" = supervisor ]; then
      payload=$(peer_tunnel_api_payload "$variant") || return 1
      peer_api_args=(bash -c "$payload")
    fi
    [ "${#peer_api_actual[@]}" -eq "${#peer_api_args[@]}" ] || continue
    equal=1
    for index in "${!peer_api_args[@]}"; do
      [ "${peer_api_actual[index]}" = "${peer_api_args[index]}" ] || { equal=0; break; }
    done
    [ "$equal" = 1 ] || continue
    peer_api_variant=$variant
    return 0
  done
  return 1
}

peer_tunnel_supervisor_child_identity() {
  local supervisor_pid=$1 shape=$2 marker=$3
  shift 3
  local -a forwards=("$@")
  local child_pid child_ppid child_command child_count=0 direct_ssh_count=0
  local expected_start process_output
  if [ "${peer_process_snapshot_active:-0}" = 1 ]; then
    process_output=$peer_process_snapshot
  else
    process_output=$(ps -eo pid=,ppid=,args=) || return 2
  fi
  while read -r child_pid child_ppid child_command; do
    [ "$child_ppid" = "$supervisor_pid" ] || continue
    case "$child_command" in
      ssh\ *|/usr/bin/ssh\ *) direct_ssh_count=$((direct_ssh_count + 1)) ;;
      *) continue ;;
    esac
    peer_tunnel_ssh_args_match "$child_command" "$shape" || continue
    for forward in "${forwards[@]}"; do
      tunnel_command_has "$child_command" "$forward" || return 1
    done
    child_count=$((child_count + 1))
    peer_snapshot_child_pid=$child_pid
    peer_snapshot_child_command=$child_command
  done <<<"$process_output"
  [ "$direct_ssh_count" -eq 1 ] && [ "$child_count" -eq 1 ] || return 1
  expected_start=$(process_start_time "$peer_snapshot_child_pid") || return 2
  peer_snapshot_child_start=$expected_start
}

peer_tunnel_wait_child_identity() {
  local supervisor_pid=$1 shape=$2 marker=$3
  shift 3
  local child_status
  for _ in $(seq 1 10); do
    if [ "${peer_process_snapshot_active:-0}" = 1 ]; then
      peer_process_snapshot=$(ps -eo pid=,ppid=,args=) || return 2
    fi
    peer_snapshot_child_pid=
    peer_snapshot_child_command=
    peer_snapshot_child_start=
    if peer_tunnel_supervisor_child_identity "$supervisor_pid" "$shape" "$marker" "$@"; then
      child_status=0
    else
      child_status=$?
    fi
    if [ "$child_status" -eq 0 ]; then
      return 0
    fi
    [ "$child_status" -eq 1 ] || return "$child_status"
    repair_sleep 1 || return 1
  done
  return 1
}

peer_tunnel_children_unchanged() {
  local marker child_pid child_start child_command supervisor_pid supervisor_start supervisor_command
  local current_command current_start current_ppid expected_forward
  local -a expected_forwards
  [ "$peer_enabled" = 1 ] || return 0
  for marker in "${!peer_child_pids[@]}"; do
    child_pid=${peer_child_pids[$marker]}
    child_start=${peer_child_starts[$marker]}
    child_command=${peer_child_commands[$marker]}
    supervisor_pid=${peer_supervisor_pids[$marker]}
    supervisor_start=${peer_supervisor_starts[$marker]}
    supervisor_command=${peer_supervisor_commands[$marker]}
    current_start=$(process_start_time "$supervisor_pid") || return 1
    [ "$current_start" = "$supervisor_start" ] || return 1
    current_command=$(process_command_line "$supervisor_pid") || return 1
    [ "$current_command" = "$supervisor_command" ] || return 1
    current_ppid=$(ps -o ppid= -p "$child_pid" 2>/dev/null | tr -d ' ') || return 1
    [ "$current_ppid" = "$supervisor_pid" ] || return 1
    current_start=$(process_start_time "$child_pid") || return 1
    [ "$current_start" = "$child_start" ] || return 1
    current_command=$(process_command_line "$child_pid") || return 1
    [ "$current_command" = "$child_command" ] || return 1
    peer_tunnel_ssh_args_match "$current_command" "${peer_child_shapes[$marker]}" || return 1
    read -r -a expected_forwards <<<"${peer_child_forwards[$marker]}"
    for expected_forward in "${expected_forwards[@]}"; do
      tunnel_command_has "$current_command" "$expected_forward" || return 1
    done
  done
}

peer_tunnel_supervisors_unchanged() {
  local marker supervisor_pid supervisor_start supervisor_command current_command current_start
  [ "$peer_enabled" = 1 ] || return 0
  for marker in "${!peer_supervisor_pids[@]}"; do
    supervisor_pid=${peer_supervisor_pids[$marker]}
    supervisor_start=${peer_supervisor_starts[$marker]}
    supervisor_command=${peer_supervisor_commands[$marker]}
    current_start=$(process_start_time "$supervisor_pid") || return 1
    [ "$current_start" = "$supervisor_start" ] || return 1
    current_command=$(process_command_line "$supervisor_pid") || return 1
    [ "$current_command" = "$supervisor_command" ] || return 1
  done
}

peer_tunnel_current_children_ready() {
  local marker shape child_pid child_command
  local -a expected_forwards
  for marker in "${!peer_supervisor_pids[@]}"; do
    shape=${peer_child_shapes[$marker]}
    read -r -a expected_forwards <<<"${peer_child_forwards[$marker]}"
    peer_tunnel_supervisor_child_identity "${peer_supervisor_pids[$marker]}" "$shape" \
      "$marker" "${expected_forwards[@]}" || return 1
  done
}

peer_tunnel_group_safe() {
  local pgid=$1 group_command group_seen=0 process_output group_listing
  process_output=$(ps -eo pgid=,args=) || return 1
  group_listing=$(awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' <<<"$process_output") || return 1
  while IFS= read -r group_command; do
    group_seen=1
    case "$group_command" in
      *"while true; do"*) peer_tunnel_supervisor_match "$group_command" cleanup || return 1 ;;
      ssh\ *|/usr/bin/ssh\ *) peer_tunnel_ssh_args_match "$group_command" cleanup || return 1 ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ]
}

peer_tunnel_remove_legacy() {
  local keep_current=${1:-0}
  local pid command_line marker process_output
  process_output=$(ps -eo pid=,args=) || return 1
  while read -r pid command_line; do
    repair_remaining >/dev/null || return 1
    marker=$(sed -n 's/.*# \(peer-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
    [ -n "$marker" ] || continue
    peer_tunnel_stale_supervisor_match "$command_line" cleanup "$marker" || continue
    if [ "$keep_current" = 1 ]; then
      if peer_tunnel_supervisor_match "$command_line" api \
          || peer_tunnel_supervisor_match "$command_line" current; then
        continue
      fi
    fi
    peer_kill_owned "$pid" "$command_line" cleanup "$marker" || return 1
  done <<<"$process_output"
}

peer_kill_owned() {
  repair_cleanup_call peer_kill_owned_inner "$@"
}

peer_kill_owned_inner() {
  local pid=$1 command_line=$2 shape=${3:-} marker=${4:-} pgid expected_start
  if [ -n "$shape" ] && [ -n "$marker" ]; then
    peer_tunnel_stale_supervisor_match "$command_line" "$shape" "$marker" || return 0
  else
    peer_tunnel_supervisor_match "$command_line" cleanup || return 0
  fi
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  expected_start=$(process_start_time "$pid") || {
    echo "could not read peer tunnel process identity for $marker" >&2
    return 1
  }
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$")
      echo "could not prove peer tunnel group for ${marker:-unknown}" >&2
      return 1
      ;;
    *)
      if [ -n "$shape" ] && [ -n "$marker" ]; then
        if ! peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker"; then
          echo "peer tunnel group $pgid failed ownership proof for $marker" >&2
          return 1
        fi
        if ! peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker"; then
          echo "peer tunnel group $pgid changed before stop for $marker" >&2
          return 1
        fi
        process_identity_matches "$pid" "$pgid" "$expected_start" "$command_line" || {
          echo "peer tunnel process changed before stop for $marker" >&2
          return 1
        }
        tunnel_prepare_group_stop "$pgid" "$pid" "$expected_start" "$command_line" \
          peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker" || return 1
        repair_cleanup_remaining || return 1
        kill -- "-$pgid" 2>/dev/null || {
          echo "could not stop peer tunnel group $pgid for $marker" >&2
          return 1
        }
        tunnel_wait_group_exit "$pgid" || {
          echo "peer tunnel group $pgid did not exit for $marker" >&2
          return 1
        }
        return 0
      fi
      if ! peer_tunnel_group_safe "$pgid"; then
        echo "peer tunnel group $pgid failed ownership proof" >&2
        return 1
      fi
      if ! peer_tunnel_group_safe "$pgid"; then
        echo "peer tunnel group changed before stop" >&2
        return 1
      fi
      process_identity_matches "$pid" "$pgid" "$expected_start" "$command_line" || {
        echo "peer tunnel process changed before stop" >&2
        return 1
      }
      tunnel_prepare_group_stop "$pgid" "$pid" "$expected_start" "$command_line" \
        peer_tunnel_group_safe "$pgid" || return 1
      repair_cleanup_remaining || return 1
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop peer tunnel group $pgid" >&2
        return 1
      }
      tunnel_wait_group_exit "$pgid" || {
        echo "peer tunnel group $pgid did not exit" >&2
        return 1
      }
      ;;
  esac
}

peer_tunnel_remove_orphan_children() {
  local pid ppid command_line executable process_output found=0 pgid start_time
  process_output=$(ps -eo pid=,ppid=,args=) || return 1
  while read -r pid ppid command_line; do
    [ "$ppid" = 1 ] || continue
    peer_tunnel_ssh_args_match "$command_line" cleanup || continue
    executable=$(readlink "/proc/$pid/exe" 2>/dev/null) || {
      [ ! -e "/proc/$pid" ] && continue
      return 1
    }
    [ "$executable" = /usr/bin/ssh ] || continue
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
    start_time=$(process_start_time "$pid") || return 1
    process_identity_matches "$pid" "$pgid" "$start_time" "$command_line" || return 1
    kill "$pid" 2>/dev/null || return 1
    found=1
  done <<<"$process_output"
  [ "$found" = 0 ]
}

peer_tunnel_remove_orphan_api_listener() {
  local listener pid ppid start_time
  listener=$(ss -H -ltnp 2>/dev/null | awk -v port=":$peer_api_port" '$4 ~ port "$" {print; exit}') || return 1
  [ -n "$listener" ] || return 0
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  peer_tunnel_api_pid_match "$pid" child || return 1
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  if [ "$ppid" != 1 ]; then
    peer_tunnel_api_pid_match "$pid" child current || return 1
    peer_tunnel_api_pid_match "$ppid" supervisor current || return 1
    return 0
  fi
  start_time=$(process_start_time "$pid") || return 1
  peer_tunnel_api_pid_match "$pid" child || return 1
  [ "$start_time" = "$(process_start_time "$pid")" ] || return 1
  [ "$(ps -o ppid= -p "$pid" | tr -d ' ')" = 1 ] || return 1
  kill "$pid" 2>/dev/null || return 1
  for _ in $(seq 1 10); do
    ss -H -ltnp 2>/dev/null | awk -v port=":$peer_api_port" '$4 ~ port "$" {found=1} END {exit found}' && return 0
    sleep 0.1
    repair_cleanup_remaining || return 1
  done
  return 1
}

peer_tunnel_remove_duplicate_api_supervisors() {
  local keep_current=${1:-1}
  local owner_pid= owner_child= pid matching_pids listener output variant
  output=$(ss -H -ltnp) || return 1
  listener=$(awk -v port=":$peer_api_port" '$4 ~ port "$" {print; exit}' <<<"$output")
  if [ -n "$listener" ]; then
    owner_child=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
    case "$owner_child" in ''|*[!0-9]*) return 1 ;; esac
    peer_tunnel_api_pid_match "$owner_child" child || return 1
    variant=$peer_api_variant
    owner_pid=$(ps -o ppid= -p "$owner_child" | tr -d ' ') || return 1
    if [ "$owner_pid" != 1 ]; then
      peer_tunnel_api_pid_match "$owner_pid" supervisor "$variant" || return 1
    else
      owner_pid=
    fi
  fi
  matching_pids=$(pgrep_matches peer-tunnel-api) || return 1
  while read -r pid; do
    [ -n "$pid" ] || continue
    peer_tunnel_api_pid_match "$pid" supervisor || return 1
    [ "$keep_current" = 1 ] && [ "$pid" = "$owner_pid" ] && \
      [ "$peer_api_variant" = current ] && continue
    peer_tunnel_api_stop_group "$pid" "$peer_api_variant" || return 1
    [ "$pid" != "$owner_pid" ] || owner_pid=
  done <<<"$matching_pids"
  if [ -n "$owner_pid" ] && ! peer_tunnel_api_pid_match "$owner_pid" supervisor current; then
    for _ in $(seq 1 10); do
      output=$(ss -H -ltnp) || return 1
      if ! awk -v port=":$peer_api_port" '$4 ~ port "$" {found=1} END {exit !found}' <<<"$output"; then
        return 0
      fi
      repair_sleep 0.1 || return 1
    done
    return 1
  fi
}

peer_tunnel_api_group_safe() {
  local pgid=$1 pid=$2 start=$3 variant=$4 member member_group member_parent output count
  peer_tunnel_api_pid_match "$pid" supervisor "$variant" || return 1
  [ "$(process_start_time "$pid")" = "$start" ] || return 1
  output=$(ps -eo pid=,ppid=,pgid=) || return 1
  count=0
  while read -r member member_parent member_group; do
    [ "$member_group" = "$pgid" ] || continue
    count=$((count + 1))
    [ "$member" = "$pid" ] && continue
    [ "$member_parent" = "$pid" ] || return 1
    peer_tunnel_api_read_args "$member" || return 1
    if [ "${#peer_api_actual[@]}" = 2 ] && [ "${peer_api_actual[0]}" = sleep ] && [ "${peer_api_actual[1]}" = 3 ]; then
      continue
    fi
    peer_tunnel_api_pid_match "$member" child "$variant" || return 1
  done <<<"$output"
  [ "$count" -ge 1 ] || return 1
}

peer_tunnel_api_stop_group() {
  repair_cleanup_call peer_tunnel_api_stop_group_inner "$@"
}

peer_tunnel_api_stop_group_inner() {
  local pid=$1 variant=$2 pgid start
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ') || return 1
  [ "$pgid" = "$pid" ] && [ "$pgid" -gt 1 ] && [ "$pgid" != "$$" ] || return 1
  start=$(process_start_time "$pid") || return 1
  peer_tunnel_api_group_safe "$pgid" "$pid" "$start" "$variant" || return 1
  tunnel_prepare_group_stop "$pgid" "$pid" "$start" "" \
    peer_tunnel_api_group_safe "$pgid" "$pid" "$start" "$variant" || return 1
  peer_tunnel_api_pid_match "$pid" supervisor "$variant" || return 1
  [ "$(process_start_time "$pid")" = "$start" ] || return 1
  [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pgid" ] || return 1
  repair_cleanup_remaining || return 1
  kill -- "-$pgid" || return 1
  tunnel_wait_group_exit "$pgid"
}

peer_remote_ports_ready() {
  local probe_seconds=${1:-10} peer_target verified_ports probe_output probe_status
  peer_missing_remote_ports=
  probe_seconds=$(repair_timeout "$probe_seconds") || return 1
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  probe_output=$(timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout="$probe_seconds" \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
      -o IPQoS=none -i "$peer_ssh_identity" -p "$peer_ssh_port" \
      "$peer_target" "python3 - ${lora_source_ports[0]} ${lora_source_ports[-1]}" <<'PY'
import http.client
import sys
from concurrent.futures import ThreadPoolExecutor

first = int(sys.argv[1])
last = int(sys.argv[2])


def is_relay(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/adapter-relay-status", headers={"Host": f"127.0.0.1:{port}"})
        response = connection.getresponse()
        body = response.read(257)
        if response.status != 404 or body != b"source URL is not configured\n":
            return port, response.status, body[:256], len(body) > 256
    except OSError as error:
        return port, "error", repr(error).encode()[:256], False
    finally:
        connection.close()
    return None


ports = list(range(first, last + 1))
with ThreadPoolExecutor(max_workers=min(32, len(ports))) as executor:
    failed = [result for result in executor.map(is_relay, ports) if result is not None]
if failed:
    print("wrong adapter relay ports: " + ",".join(str(port) for port, _, _, _ in failed))
    port, status, body, truncated = failed[0]
    print(f"first wrong adapter relay: port={port} status={status} body={body!r} truncated={truncated}")
    raise SystemExit(1)
print(len(ports))
PY
  )
  probe_status=$?
  if [ "$probe_status" -ne 0 ]; then
    peer_missing_remote_ports=$(sed -n 's/^wrong adapter relay ports: //p' <<<"$probe_output")
    [ -n "$probe_output" ] && printf '%s\n' "$probe_output" >&2
    return 1
  fi
  verified_ports=$(tail -n 1 <<<"$probe_output")
  [ "$verified_ports" = "${#lora_source_ports[@]}" ] || return 1
  peer_tunnel_supervisors_unchanged
}

peer_repair_missing_remote_lanes() {
  local port lane lane_start lane_ports lane_forwards marker pid command_line matching_pids
  local -a missing_ports
  local -A seen_lanes=()
  [[ "${peer_missing_remote_ports:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || return 1
  IFS=, read -r -a missing_ports <<<"$peer_missing_remote_ports"
  for port in "${missing_ports[@]}"; do
    [ "${#port}" -le 5 ] || return 1
    port=$((10#$port))
    case " ${lora_source_ports[*]} " in *" $port "*) ;; *) return 1 ;; esac
    lane=$(( (port - lora_source_ports[0]) / 2 ))
    [ "$lane" -ge 0 ] || return 1
    [ "$lane" -lt "$(( (${#lora_source_ports[@]} + 1) / 2 ))" ] || return 1
    seen_lanes[$lane]=1
  done
  for lane in "${!seen_lanes[@]}"; do
    lane_start=$((lane * 2))
    lane_ports=("${lora_source_ports[@]:lane_start:2}")
    lane_forwards=()
    for port in "${lane_ports[@]}"; do
      lane_forwards+=("-R 127.0.0.1:$port:127.0.0.1:$relay_port")
    done
    marker="peer-tunnel-lora-$lane"
    matching_pids=$(pgrep_matches "$marker") || return 1
    while read -r pid; do
      [ -n "$pid" ] || continue
      command_line=$(process_command_line "$pid") || return 1
      peer_tunnel_supervisor_forwards_match "$command_line" current "$marker" \
        "${lane_forwards[@]}" || continue
      peer_kill_owned "$pid" "$command_line" current "$marker" "${lane_forwards[@]}" || return 1
    done <<<"$matching_pids"
    peer_tunnel_connection "lora-$lane" \
      "-R 127.0.0.1:${lane_ports[0]}:127.0.0.1:$relay_port" \
      "${lane_forwards[@]}" || return 1
  done
}

china_source_ports_ready() {
  local probe_seconds=40 expected_lanes observed_lanes verified_ports
  expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  observed_lanes=$(ps -eo args= | sed -n 's/.*# china-tunnel-lora-\([0-9][0-9]*\)$/\1/p' | sort -n) || return 1
  [ "$observed_lanes" = "$(seq 0 $((expected_lanes - 1)))" ] || return 1
  if ! verified_ports=$(timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      python3 - "${lora_source_ports[0]}" "${lora_source_ports[-1]}" <<'PY'
import asyncio
import sys
import time

first = int(sys.argv[1])
last = int(sys.argv[2])
budget = 40.0


async def is_adapter_server(port, socket_seconds, semaphore):
    writer = None
    try:
        async with semaphore:
            async with asyncio.timeout(socket_seconds):
                reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=16384)
                writer.write(b"HEAD / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                await writer.drain()
                lines = (await reader.readuntil(b"\r\n\r\n")).decode("iso-8859-1").split("\r\n")
                status = lines[0].split(" ", 2)
                server = next((line.split(":", 1)[1].strip() for line in lines[1:]
                               if line.lower().startswith("server:")), "")
                if len(status) < 2 or status[0] not in {"HTTP/1.0", "HTTP/1.1"} or status[1] != "200":
                    return port, f"status={lines[0][:48]!r}"
                if not server.startswith("SimpleHTTP/"):
                    return port, f"server={server[:48]!r}"
    except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
        return port, type(error).__name__
    finally:
        if writer is not None:
            writer.close()
    return port, None


ports = list(range(first, last + 1))


async def check_ports():
    failed = dict.fromkeys(ports, "not checked")
    for attempt, (workers, socket_seconds, stage_seconds) in enumerate(((32, 2, 10), (4, 5, 12), (2, 5, 12)), 1):
        if attempt > 1:
            await asyncio.sleep(0.25)
        started = time.monotonic()
        semaphore = asyncio.Semaphore(workers)
        tasks = {asyncio.create_task(is_adapter_server(port, socket_seconds, semaphore)): port for port in failed}
        try:
            done, pending = await asyncio.wait(tasks, timeout=stage_seconds)
            failed = {tasks[task]: "stage timeout" for task in pending}
            for task in done:
                port, reason = task.result()
                if reason is not None:
                    failed[port] = reason
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        details = ",".join(f"{port}:{reason}" for port, reason in sorted(failed.items())[:16])
        elapsed = time.monotonic() - started
        print(f"adapter source check attempt={attempt} elapsed={elapsed:.2f}s failed={len(failed)} {details}", file=sys.stderr)
        if not failed:
            break
    return failed


async def main():
    return await asyncio.wait_for(check_ports(), timeout=budget)


try:
    failed = asyncio.run(main())
except TimeoutError:
    print("adapter source check exceeded its time limit", file=sys.stderr)
    raise SystemExit(1)
if failed:
    print("persistent adapter source failures: " + ",".join(map(str, list(failed)[:16])), file=sys.stderr)
    raise SystemExit(1)
print(len(ports))
PY
  ); then
    return 1
  fi
  [ "$verified_ports" = "${#lora_source_ports[@]}" ]
}

tunnel_command_has() {
  local command_line=$1 token=$2
  case " $command_line " in
    *" $token "*|*" $token;"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Validate source forwards from the configured inventory.  The caller chooses
# the accepted shape so cleanup can handle the one- and four-forward layouts,
# while readiness accepts only the current two-forward layout.
china_tunnel_forward_shape() {
  local command_line=$1 shape=$2
  local source_port forward_total spec
  local -a specs seen=()
  forward_total=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
  case "$shape:$forward_total" in
    cleanup:1|cleanup:2|cleanup:4|current:2) ;;
    *) return 1 ;;
  esac
  mapfile -t specs < <(grep -oE -- '-L 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#specs[@]}" -eq "$forward_total" ] || return 1
  for spec in "${specs[@]}"; do
    case "$spec" in
      -L\ 127.0.0.1:*:127.0.0.1:"$adapter_port") ;;
      *) return 1 ;;
    esac
    source_port=${spec#-L 127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${lora_source_ports[*]} " in
      *" $source_port "*) ;;
      *) return 1 ;;
    esac
    case " ${seen[*]} " in
      *" $source_port "*) return 1 ;;
      *) seen+=("$source_port") ;;
    esac
  done
}

china_tunnel_marker_match() {
  local command_line=$1 marker=${2:-}
  if [ -n "$marker" ]; then
    case "$command_line" in *"# $marker") ;; *) return 1 ;; esac
  else
    [[ "$command_line" =~ \#\ china-tunnel-lora-[0-9]+[[:space:]]*$ ]] || return 1
  fi
}

china_tunnel_supervisor_match() {
  local command_line=$1 shape=$2 marker=${3:-}
  local ssh_command
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  if [ -z "$marker" ]; then
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  fi
  [ -n "$marker" ] || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_ssh_args_match "$ssh_command" "$shape"
}

china_tunnel_supervisor_forwards_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  china_tunnel_supervisor_match "$command_line" "$shape" "$marker" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_known_old_supervisor_forwards_match() {
  local command_line=$1 marker=$2 forward ssh_command
  shift 2
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_known_old_ssh_args_match "$ssh_command" current || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_known_old_supervisor_match() {
  local command_line=$1 marker ssh_command
  marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  [ -n "$marker" ] || return 1
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  china_tunnel_forward_shape "$command_line" current || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_known_old_ssh_args_match "$ssh_command" current
}

china_tunnel_stale_supervisor_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  china_tunnel_forward_shape "$command_line" "$shape" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_ssh_args_match_mode() {
  local command_line=$1 shape=$2 mode=${3:-current}
  local source_port forward_count index
  local -a command_args expected_args
  expected_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
  )
  case "$mode" in
    current) expected_args+=( -o 'ChannelTimeout=*-tcpip=45s' ) ;;
    old) ;;
    *) return 1 ;;
  esac
  expected_args+=(
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=no
  )
  command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  [ "${#command_args[@]}" -ge "$(( ${#expected_args[@]} + 4 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[-1]}" = "$trainer_ssh_target" ] || return 1
  forward_count=$(( (${#command_args[@]} - ${#expected_args[@]} - 2) / 2 ))
  china_tunnel_forward_shape "$command_line" "$shape" || return 1
  [ "$forward_count" -eq "$(grep -o -- '-L ' <<<"$command_line" | wc -l)" ] || return 1
  for ((index = ${#expected_args[@]} + 1; index < ${#command_args[@]} - 1; index += 2)); do
    [ "${command_args[index]}" = "-L" ] || return 1
    source_port=${command_args[index + 1]#127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${lora_source_ports[*]} " in
      *" $source_port "*) ;;
      *) return 1 ;;
    esac
  done
}

china_tunnel_ssh_args_match() {
  china_tunnel_ssh_args_match_mode "$1" "$2" current
}

china_tunnel_known_old_ssh_args_match() {
  china_tunnel_ssh_args_match_mode "$1" "$2" old
}

china_tunnel_group_safe() {
  local pgid=$1 expected_marker=${2:-} group_command
  local group_seen=0 supervisor_seen=0 direct_seen=0 proxy_seen=0 sleep_seen=0
  local expected_ssh proxy_ssh normalized_command
  local group_listing
  if [ "${tunnel_group_process_snapshot_active:-0}" = 1 ]; then
    group_listing=$(awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' \
      <<<"$tunnel_group_process_snapshot" | sed -E 's/[[:space:]]+$//') || return 1
  else
    group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  fi
  expected_ssh=
  if proxy_ssh=$(china_tunnel_proxy_ssh_command); then
    :
  else
    # A route without a proxy is valid, but an unreadable route is unknown.
    [ "$?" -eq 1 ] || return 1
    proxy_ssh=
  fi
  while IFS= read -r group_command; do
    case "$group_command" in
      *"while true; do"*)
        [ "$supervisor_seen" = 0 ] || return 1
        supervisor_seen=1
        china_tunnel_supervisor_match "$group_command" cleanup "$expected_marker" || return 1
        expected_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$expected_marker") || return 1
        expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
        ;;
    esac
  done <<<"$group_listing"
  [ "$supervisor_seen" = 1 ] || return 1
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        if [ "$normalized_command" = "$expected_ssh" ]; then
          [ "$direct_seen" = 0 ] || return 1
          direct_seen=1
        else
          [ -n "$proxy_ssh" ] && [ "$normalized_command" = "$proxy_ssh" ] || return 1
          [ "$proxy_seen" = 0 ] || return 1
          proxy_seen=1
        fi
        ;;
      "sleep 3")
        [ "$sleep_seen" = 0 ] || return 1
        sleep_seen=1
        ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ] && [ "$supervisor_seen" = 1 ]
}

china_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 marker=$3
  local group_command expected_ssh actual_ssh group_listing group_seen=0
  local supervisor_seen=0 direct_seen=0 proxy_seen=0 sleep_seen=0 normalized_command
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
  if ! china_tunnel_ssh_args_match "$expected_ssh" cleanup; then
    china_tunnel_known_old_ssh_args_match "$expected_ssh" cleanup || return 1
  fi
  if [ "${tunnel_group_process_snapshot_active:-0}" = 1 ]; then
    group_listing=$(awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' \
      <<<"$tunnel_group_process_snapshot" | sed -E 's/[[:space:]]+$//') || return 1
  else
    group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[[:space:]]*[0-9]+[[:space:]]+/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  fi
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        [ "$supervisor_seen" = 0 ] || return 1
        supervisor_seen=1
        china_tunnel_stale_supervisor_match "$group_command" cleanup "$marker" || return 1
        actual_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$marker") || return 1
        actual_ssh=$(tunnel_channel_timeout_normalize "$actual_ssh") || return 1
        [ "$actual_ssh" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        if [ "$normalized_command" != "$expected_ssh" ]; then
          china_tunnel_proxy_child_match "$normalized_command" || return 1
          [ "$proxy_seen" = 0 ] || return 1
          proxy_seen=1
        else
          [ "$direct_seen" = 0 ] || return 1
          direct_seen=1
        fi
        ;;
      "sleep 3")
        [ "$sleep_seen" = 0 ] || return 1
        sleep_seen=1
        ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ] && [ "$supervisor_seen" = 1 ]
}

tunnel_listener_ready() {
  local port=$1 marker=$2 target=$3 forward_count=$4
  shift 4
  local -a forwards=("$@")
  local listener pid ppid owner_command supervisor_command forward forward_total
  listener=$(ss -H -ltnp 2>/dev/null | awk -v port=":$port" '$4 ~ port "$" {print; exit}') || return 1
  [ -n "$listener" ] || return 1
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  owner_command=$(process_command_line "$pid") || return 1
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  case "$ppid" in ''|*[!0-9]*|1) return 1 ;; esac
  case "$marker" in
    china-tunnel-*) china_tunnel_ssh_args_match "$owner_command" current || return 1 ;;
    peer-tunnel-api) peer_tunnel_ssh_args_match "$owner_command" api || return 1 ;;
    peer-tunnel-*) peer_tunnel_ssh_args_match "$owner_command" current || return 1 ;;
  esac
  forward_total=$(grep -o -- '-[LR] ' <<<"$owner_command" | wc -l)
  [ "$forward_total" -eq "$forward_count" ] || return 1
  tunnel_command_has "$owner_command" ssh || return 1
  tunnel_command_has "$owner_command" "$target" || return 1
  for forward in "${forwards[@]}"; do
    tunnel_command_has "$owner_command" "$forward" || return 1
  done
  supervisor_command=$(process_command_line "$ppid") || return 1
  case "$supervisor_command" in *"while true; do"*) ;; *) return 1 ;; esac
  tunnel_command_has "$supervisor_command" "$target" || return 1
  tunnel_command_has "$supervisor_command" "# $marker" || return 1
  forward_total=$(grep -o -- '-[LR] ' <<<"$supervisor_command" | wc -l)
  [ "$forward_total" -eq "$forward_count" ] || return 1
  for forward in "${forwards[@]}"; do
    tunnel_command_has "$supervisor_command" "$forward" || return 1
  done
  case "$marker" in
    china-tunnel-*) china_tunnel_supervisor_forwards_match "$supervisor_command" current "$marker" "${forwards[@]}" || return 1 ;;
    peer-tunnel-api) peer_tunnel_supervisor_forwards_match "$supervisor_command" api "$marker" "${forwards[@]}" || return 1 ;;
    peer-tunnel-*) peer_tunnel_supervisor_forwards_match "$supervisor_command" current "$marker" "${forwards[@]}" || return 1 ;;
  esac
}

china_tunnel_supervisor_owns_forwards() {
  local supervisor_pid=$1 supervisor_command=$2 mode=${3:-current}
  local listener forward source_port child_pid child_ppid owner_command index ssh_command marker owner_pid= listener_output
  local -a forward_specs=()
  shift 3
  while [ "$#" -gt 0 ]; do
    if [ "$1" = -L ]; then
      shift
      [ "$#" -gt 0 ] || return 1
      forward_specs+=("$1")
    else
      case "$1" in -L\ *) forward_specs+=("${1#-L }") ;; *) return 1 ;; esac
    fi
    shift
  done
  [ "${#forward_specs[@]}" -gt 0 ] || return 1
  marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$supervisor_command")
  ssh_command=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  if [ "$mode" = current ]; then
    china_tunnel_ssh_args_match "$ssh_command" current || return 1
  else
    china_tunnel_known_old_ssh_args_match "$ssh_command" current || return 1
  fi
  if [ "${china_listener_snapshot_active:-0}" = 1 ]; then
    listener_output=$china_listener_snapshot
  else
    listener_output=$(ss -H -ltnp 2>/dev/null) || return 2
  fi
  for forward in "${forward_specs[@]}"; do
    tunnel_command_has "$supervisor_command" "$forward" || return 1
    source_port=${forward#127.0.0.1:}
    source_port=${source_port%%:*}
    listener=$(awk -v port=":$source_port" '$4 ~ port "$" {print; exit}' <<<"$listener_output") || return 2
    [ -n "$listener" ] || return 1
    child_pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener") || return 2
    case "$child_pid" in ''|*[!0-9]*) return 2 ;; esac
    if [ -z "$owner_pid" ]; then
      owner_pid=$child_pid
    else
      [ "$child_pid" = "$owner_pid" ] || return 1
    fi
    child_ppid=$(ps -o ppid= -p "$child_pid" 2>/dev/null | tr -d ' ') || return 2
    case "$child_ppid" in ''|*[!0-9]*) return 2 ;; esac
    [ "$child_ppid" = "$supervisor_pid" ] || return 1
    owner_command=$(process_command_line "$child_pid")
    case "$?" in
      0) [ -n "$owner_command" ] || return 2 ;;
      *) return 2 ;;
    esac
    if [ "$mode" = current ]; then
      china_tunnel_ssh_args_match "$owner_command" current || return 1
    else
      china_tunnel_known_old_ssh_args_match "$owner_command" current || return 1
    fi
    tunnel_command_has "$owner_command" "$forward" || return 1
  done
}

china_tunnel_grouped_legacy_match() {
  local command_line=$1 forward_total marker ssh_command
  china_tunnel_marker_match "$command_line" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  tunnel_command_has "$command_line" "$trainer_ssh_target" || return 1
  china_tunnel_forward_shape "$command_line" cleanup || return 1
  marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  [ -n "$marker" ] || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  # The unscoped cleanup runs before the current two-forward lanes start.
  # Only the old SSH shape is safe to remove here. A current-shaped supervisor
  # may use a cleanup-shaped forward count during a migration.
  china_tunnel_known_old_ssh_args_match "$ssh_command" cleanup || return 1
  case "$command_line" in
    *"while true; do"*"$trainer_ssh_target"*"# china-tunnel-lora-"*) ;;
    *) return 1 ;;
  esac
  forward_total=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
  [ "$forward_total" -eq 1 ] || [ "$forward_total" -eq 4 ]
}

china_tunnel_ssh_child_match() {
  local pid=$1 shape=${2:-cleanup} command_line executable
  executable=$(readlink "/proc/$pid/exe" 2>/dev/null) || {
    [ ! -e "/proc/$pid" ] && return 1
    return 2
  }
  case "$executable" in
    /usr/bin/ssh) ;;
    *) return 1 ;;
  esac
  command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | sed 's/[[:space:]]*$//') || return 1
  china_tunnel_ssh_args_match "$command_line" "$shape" ||
    china_tunnel_known_old_ssh_args_match "$command_line" "$shape" ||
    china_tunnel_proxy_child_match "$command_line"
}

china_tunnel_remove_orphan_children() {
  local requested source_port listener listener_output state receive_queue send_queue local_address remote_address details pid ppid seen_pid pgid start_time command_line process_status
  local -a requested_ports=() listener_pids=()
  local -A requested_port=()
  for requested in "$@"; do
    case "$requested" in
      127.0.0.1:*:127.0.0.1:*)
        requested=${requested#127.0.0.1:}
        requested_ports+=("${requested%%:*}")
        ;;
      [0-9]*) requested_ports+=("$requested") ;;
    esac
  done
  [ "${#requested_ports[@]}" -gt 0 ] || requested_ports=("${lora_source_ports[@]}")
  for source_port in "${requested_ports[@]}"; do
    requested_port[$source_port]=1
  done
  listener_output=$(ss -H -ltnp 2>/dev/null) || return 1
  while IFS= read -r listener; do
    read -r state receive_queue send_queue local_address remote_address details <<<"$listener"
    [ "${local_address%:*}" = "127.0.0.1" ] || continue
    source_port=${local_address##*:}
    [ -n "${requested_port[$source_port]+set}" ] || continue
    pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    seen_pid=0
    for listener_pid in "${listener_pids[@]}"; do
      [ "$listener_pid" = "$pid" ] && seen_pid=1
    done
    [ "$seen_pid" = 0 ] || continue
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    process_status=$?
    if [ "$process_status" -ne 0 ]; then
      [ ! -e "/proc/$pid" ] && continue
      return 1
    fi
    case "$ppid" in ''|*[!0-9]*) return 1 ;; esac
    [ "$ppid" = 1 ] || continue
    if china_tunnel_ssh_child_match "$pid"; then
      :
    else
      [ "$?" -eq 1 ] && continue
      return 1
    fi
    listener_pids+=("$pid")
    china_legacy_cleanup_required=1
    china_legacy_cleanup_pids+=("$pid")
    china_legacy_cleanup_orphan_pids+=("$pid")
    case " ${china_legacy_cleanup_ports[*]} " in
      *" $source_port "*) ;;
      *) china_legacy_cleanup_ports+=("$source_port") ;;
    esac
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || {
      [ ! -e "/proc/$pid" ] && continue
      return 1
    }
    command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | sed 's/[[:space:]]*$//') || return 1
    start_time=$(process_start_time "$pid") || return 1
    process_identity_matches "$pid" "$pgid" "$start_time" "$command_line" || return 1
    kill "$pid" 2>/dev/null || return 1
  done <<<"$listener_output"
}

china_tunnel_ports_clear() {
  local requested source_port listener_output
  local -a requested_ports=()
  for requested in "$@"; do
    case "$requested" in
      127.0.0.1:*:127.0.0.1:*)
        requested=${requested#127.0.0.1:}
        requested_ports+=("${requested%%:*}")
        ;;
    esac
  done
  [ "${#requested_ports[@]}" -gt 0 ] || return 1
  listener_output=$(ss -H -ltn 2>/dev/null) || return 2
  for source_port in "${requested_ports[@]}"; do
    if awk -v local_port="127.0.0.1:$source_port" '$4 == local_port {found=1} END {exit !found}' <<<"$listener_output"; then
      return 1
    else
      case "$?" in 1) ;; *) return 2 ;; esac
    fi
  done
}

china_tunnel_force_kill_orphan() {
  local pid=$1 pgid command_line start_time current_start current_command state kill_status remaining process_status
  if [ ! -e "/proc/$pid" ]; then
    return 0
  fi
  china_tunnel_ssh_child_match "$pid" || return 1
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | sed 's/[[:space:]]*$//') || return 1
  start_time=$(process_start_time "$pid") || return 1
  process_identity_matches "$pid" "$pgid" "$start_time" "$command_line" || return 1
  if kill -KILL "$pid" 2>/dev/null; then
    kill_status=0
  else
    kill_status=1
  fi
  for _ in $(seq 1 30); do
    if [ ! -e "/proc/$pid" ]; then
      return 0
    fi
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
    process_status=$?
    if [ "$process_status" -ne 0 ]; then
      [ ! -e "/proc/$pid" ] && return 0
      return 1
    fi
    case "$state" in Z*) return 0 ;; esac
    current_start=$(process_start_time "$pid") || {
      [ ! -e "/proc/$pid" ] && return 0
      return 1
    }
    [ "$current_start" = "$start_time" ] || return 0
    current_command=$(process_command_line "$pid")
    process_status=$?
    case "$process_status" in
      0) [ "$current_command" = "$command_line" ] || return 1 ;;
      2) return 0 ;;
      *) return 1 ;;
    esac
    [ "$kill_status" = 0 ] || return 1
    remaining=$(repair_remaining) || return 1
    [ "$remaining" -gt 0 ] || return 1
    repair_sleep 1 || return 1
  done
  return 1
}

china_tunnel_stop_supervisor() {
  repair_cleanup_call china_tunnel_stop_supervisor_inner "$@"
}

china_tunnel_stop_supervisor_inner() {
  local pid=$1 command_line=$2 expected_marker=${3:-} stale=${4:-0} pgid forward source_port expected_start
  local -a owned_forwards=()
  mapfile -t owned_forwards < <(grep -oE -- '-L 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#owned_forwards[@]}" -gt 0 ] || return 1
  china_legacy_cleanup_required=1
  china_legacy_cleanup_pids+=("$pid")
  for forward in "${owned_forwards[@]}"; do
    source_port=${forward#-L 127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${china_legacy_cleanup_ports[*]} " in
      *" $source_port "*) ;;
      *) china_legacy_cleanup_ports+=("$source_port") ;;
    esac
  done
  if [ -z "$expected_marker" ]; then
    expected_marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  fi
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  expected_start=$(process_start_time "$pid") || {
    echo "could not read China tunnel process identity for ${expected_marker:-unknown}" >&2
    return 1
  }
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$")
      echo "could not prove China tunnel group for ${expected_marker:-unknown}" >&2
      return 1
      ;;
    *)
      case " ${china_legacy_cleanup_pgids[*]} " in
        *" $pgid "*) ;;
        *) china_legacy_cleanup_pgids+=("$pgid") ;;
      esac
      if [ "$stale" = 1 ]; then
        if ! china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker"; then
          echo "stale China tunnel group $pgid failed ownership proof for ${expected_marker:-unknown}" >&2
          return 1
        fi
        if ! china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker"; then
          echo "stale China tunnel group changed before stop for ${expected_marker:-unknown}" >&2
          return 1
        fi
      elif ! china_tunnel_group_safe "$pgid" "$expected_marker"; then
        echo "China tunnel group $pgid failed ownership proof for ${expected_marker:-unknown}" >&2
        return 1
      elif ! china_tunnel_group_safe "$pgid" "$expected_marker"; then
        echo "China tunnel group changed before stop for ${expected_marker:-unknown}" >&2
        return 1
      fi
      process_identity_matches "$pid" "$pgid" "$expected_start" "$command_line" || {
        echo "China tunnel process changed before stop for ${expected_marker:-unknown}" >&2
        return 1
      }
      if [ "$stale" = 1 ]; then
        tunnel_prepare_group_stop "$pgid" "$pid" "$expected_start" "$command_line" \
          china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker" || return 1
      else
        tunnel_prepare_group_stop "$pgid" "$pid" "$expected_start" "$command_line" \
          china_tunnel_group_safe "$pgid" "$expected_marker" || return 1
      fi
      repair_cleanup_remaining || return 1
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop China tunnel group $pgid for ${expected_marker:-unknown}" >&2
        return 1
      }
      china_tunnel_stopped_pgid=$pgid
      china_tunnel_wait_group_exit "$pgid" || {
        echo "China tunnel group $pgid did not exit for ${expected_marker:-unknown}" >&2
        return 1
      }
      ;;
  esac
}

china_tunnel_preflight_group() {
  local pid=$1 command_line=$2 stale=${3:-0} expected_marker=${4:-} pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || return 1
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$") return 1 ;;
  esac
  case " ${china_tunnel_preflight_pgids[*]:-} " in
    *" $pgid "*) return 1 ;;
  esac
  if [ "$stale" = 1 ]; then
    china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker" || return 1
  else
    china_tunnel_group_safe "$pgid" "$expected_marker" || return 1
  fi
  china_tunnel_preflight_pgids+=("$pgid")
}

china_tunnel_remove_all_owned() {
  local pid command_line marker process_output invalid=0 index
  local -a current_pids=() current_commands=() old_pids=() old_commands=()
  process_output=$(ps -eo pid=,args=) || return 1
  while read -r pid command_line; do
    repair_remaining >/dev/null || return 1
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
    [ -n "$marker" ] || continue
    if china_tunnel_supervisor_match "$command_line" cleanup "$marker"; then
      current_pids+=("$pid")
      current_commands+=("$command_line")
    elif china_tunnel_known_old_supervisor_match "$command_line"; then
      old_pids+=("$pid")
      old_commands+=("$command_line")
    else
      invalid=1
    fi
  done <<<"$process_output"
  [ "$invalid" -eq 0 ] || return 1
  china_tunnel_preflight_pgids=()
  for index in "${!current_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"${current_commands[index]}")
    china_tunnel_preflight_group "${current_pids[index]}" "${current_commands[index]}" 0 "$marker" || return 1
  done
  for index in "${!old_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"${old_commands[index]}")
    china_tunnel_preflight_group "${old_pids[index]}" "${old_commands[index]}" 1 "$marker" || return 1
  done
  for index in "${!current_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    china_tunnel_stop_supervisor "${current_pids[index]}" "${current_commands[index]}" || return 1
  done
  for index in "${!old_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    china_tunnel_stop_supervisor "${old_pids[index]}" "${old_commands[index]}" "" 1 || return 1
  done
}

china_tunnel_remove_legacy() {
  local only_marker=${1:-}
  local lora_source_port pattern pid command_line cleanup_port matching_pids
  local grouped_legacy stale
  local -a cleanup_ports=()
  if [ -z "$only_marker" ]; then
    local process_output marker
    process_output=$(ps -eo pid=,args=) || return 1
    while read -r pid command_line; do
      repair_remaining >/dev/null || return 1
      marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
      [ -n "$marker" ] || continue
      china_tunnel_grouped_legacy_match "$command_line" || continue
      china_tunnel_stop_supervisor "$pid" "$command_line" "$marker" 1 || return 1
    done <<<"$process_output"
    return 0
  fi
  for cleanup_port in "${@:2}"; do
    case "$cleanup_port" in
      127.0.0.1:*:127.0.0.1:*)
        cleanup_port=${cleanup_port#127.0.0.1:}
        cleanup_ports+=("${cleanup_port%%:*}")
        ;;
      [0-9]*) cleanup_ports+=("$cleanup_port") ;;
    esac
  done
  [ "${#cleanup_ports[@]}" -gt 0 ] || cleanup_ports=("${lora_source_ports[@]}")
  for lora_source_port in "${cleanup_ports[@]}"; do
    repair_remaining >/dev/null || return 1
    pattern="-L 127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
    matching_pids=$(pgrep_matches "$pattern") || return 1
    while read -r pid; do
      [ -n "$pid" ] || continue
      repair_remaining >/dev/null || return 1
      command_line=$(process_command_line "$pid")
      case "$?" in
        0) ;;
        2) continue ;;
        *) return 1 ;;
      esac
      grouped_legacy=0
      if [ -n "$only_marker" ]; then
        china_tunnel_marker_match "$command_line" "$only_marker" || continue
        case "$command_line" in *"while true; do"*) ;; *) continue ;; esac
        tunnel_command_has "$command_line" "$trainer_ssh_target" || continue
        china_tunnel_forward_shape "$command_line" cleanup || continue
        tunnel_command_has "$command_line" "$pattern" || continue
        grouped_legacy=1
      else
        china_tunnel_grouped_legacy_match "$command_line" || continue
        grouped_legacy=1
      fi
      if [ "$grouped_legacy" = 1 ]; then
        stale=1
        if [ -n "$only_marker" ] && china_tunnel_supervisor_match "$command_line" cleanup "$only_marker"; then
          stale=0
        fi
        china_tunnel_stop_supervisor "$pid" "$command_line" "$only_marker" "$stale" || return 1
      fi
    done <<<"$matching_pids"
  done
}

china_tunnel_wait_legacy_clear() {
  local pid pgid group_processes command_line process_status listener_output
  for wait_cycle in $(seq 1 30); do
    repair_remaining >/dev/null || return 1
    china_tunnel_remove_orphan_children "${china_legacy_cleanup_ports[@]}" || return 1
    if [ "$wait_cycle" -eq 15 ]; then
      for pid in "${china_legacy_cleanup_orphan_pids[@]}"; do
        repair_remaining >/dev/null || return 1
        china_tunnel_force_kill_orphan "$pid" || return 1
      done
    fi
    listener_remaining=0
    for pid in "${china_legacy_cleanup_pids[@]}"; do
      command_line=$(process_command_line "$pid")
      process_status=$?
      case "$process_status" in
        0)
          [ -z "$command_line" ] || {
            listener_remaining=1
            break
          }
          ;;
        2) ;;
        *) return 1 ;;
      esac
    done
    if [ "$listener_remaining" = 0 ]; then
      for pgid in "${china_legacy_cleanup_pgids[@]}"; do
        listener_output=$(ps -eo pgid=,args= 2>/dev/null) || return 1
        group_processes=$(awk -v group="$pgid" '$1 == group {print; exit}' <<<"$listener_output") || return 1
        [ -z "$group_processes" ] || {
          listener_remaining=1
          break
        }
      done
    fi
    [ "$listener_remaining" = 0 ] && return 0
    repair_sleep 1 || return 1
  done
  echo "stale China adapter tunnels did not stop before migration" >&2
  return 1
}

china_tunnel_wait_group_exit() {
  tunnel_wait_group_exit "$1"
}

china_tunnel_reconcile_connection() {
  local marker=$1
  shift
  local -a forwards=("$@")
  local pid command_line current_count=0 keeper= old_keeper= pgid invalid=0 index replace_all=0 matching_pids
  local -a current_pids=() current_commands=() old_pids=() old_commands=()
  repair_remaining >/dev/null || return 1
  matching_pids=$(pgrep_matches "$marker") || return 1
  while read -r pid; do
    [ -n "$pid" ] || continue
    repair_remaining >/dev/null || return 1
    command_line=$(process_command_line "$pid")
    case "$?" in
      0) ;;
      2) continue ;;
      *)
        echo "could not inspect China tunnel process $pid for $marker" >&2
        return 1
        ;;
    esac
    if china_tunnel_supervisor_forwards_match "$command_line" current "$marker" "${forwards[@]}"; then
      current_count=$((current_count + 1))
      current_pids+=("$pid")
      current_commands+=("$command_line")
    elif china_tunnel_known_old_supervisor_forwards_match "$command_line" "$marker" "${forwards[@]}"; then
      old_pids+=("$pid")
      old_commands+=("$command_line")
    elif [ -n "$command_line" ] && china_tunnel_marker_match "$command_line" "$marker"; then
      invalid=1
    fi
  done <<<"$matching_pids"
  [ "$invalid" -eq 0 ] || return 1
  if [ "$current_count" -gt 0 ]; then
    for index in "${!current_pids[@]}"; do
      repair_remaining >/dev/null || return 1
      if china_tunnel_supervisor_owns_forwards "${current_pids[index]}" \
          "${current_commands[index]}" current "${forwards[@]}"; then
        keeper=${current_pids[index]}
        break
      else
        [ "$?" -eq 1 ] || return 1
      fi
    done
    if [ -z "$keeper" ]; then
      for index in "${!old_pids[@]}"; do
        if china_tunnel_supervisor_owns_forwards "${old_pids[index]}" \
            "${old_commands[index]}" old "${forwards[@]}"; then
          old_keeper=${old_pids[index]}
          break
        else
          [ "$?" -eq 1 ] || return 1
        fi
      done
      replace_all=1
    fi
  fi
  china_tunnel_preflight_pgids=()
  for index in "${!old_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    china_tunnel_preflight_group "${old_pids[index]}" "${old_commands[index]}" 1 "$marker" || return 1
  done
  for index in "${!current_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    china_tunnel_preflight_group "${current_pids[index]}" "${current_commands[index]}" 0 "$marker" || return 1
  done
  for index in "${!old_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    china_tunnel_stop_supervisor "${old_pids[index]}" "${old_commands[index]}" "$marker" 1 || return 1
  done
  for index in "${!current_pids[@]}"; do
    repair_remaining >/dev/null || return 1
    [ "$replace_all" = 0 ] && [ "${current_pids[index]}" = "$keeper" ] && continue
    china_tunnel_stop_supervisor "${current_pids[index]}" "${current_commands[index]}" "$marker" 1 || return 1
  done
  if [ -n "$keeper" ] && [ "$replace_all" = 0 ]; then
    return 0
  fi
  return 2
}

china_tunnel_connection() {
  local connection_name=$1
  shift
  local -a forwards=("$@")

  local -a ssh_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=no
  )
  local forward_text ssh_text target marker pid command_line ports_clear reconcile_status
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  target=$(printf '%q' "$trainer_ssh_target")
  marker="china-tunnel-$connection_name"
  repair_remaining >/dev/null || return 1
  if [ "${restart_tunnels:-0}" = 0 ]; then
    if china_tunnel_reconcile_connection "$marker" "${forwards[@]}"; then
      reconcile_status=0
    else
      reconcile_status=$?
    fi
    case "$reconcile_status" in
      0) return 0 ;;
      2) ;;
      *) return 1 ;;
    esac
    china_tunnel_remove_legacy "$marker" "${forwards[@]}" || return 1
    ports_clear=0
    for _ in $(seq 1 10); do
      repair_remaining >/dev/null || return 1
      china_tunnel_remove_orphan_children "${forwards[@]}" || return 1
      if china_tunnel_ports_clear "${forwards[@]}"; then
        ports_clear=1
        break
      else
        [ "$?" -eq 1 ] || return 1
      fi
      repair_sleep 1 || return 1
    done
    if [ "$ports_clear" != 1 ]; then
      echo "old adapter tunnel ports did not clear for $marker" >&2
      return 1
    fi
  fi
  repair_launch_tunnel china "$marker" "" "while true; do
      ssh$ssh_text$forward_text $target
      sleep 3
    done # $marker" || return 1
}

peer_tunnel() {
  local peer_keep_current peer_orphans_clear phase_start
  peer_keep_current=$peer_enabled
  [ "${restart_tunnels:-0}" = 1 ] && peer_keep_current=0
  phase_start=$SECONDS
  repair_stage=peer-legacy
  peer_tunnel_remove_legacy "$peer_keep_current" || {
    echo "peer phase=legacy stage=remove elapsed=$((SECONDS - phase_start))" >&2
    return 1
  }
  peer_orphans_clear=0
  for _ in $(seq 1 10); do
    repair_remaining >/dev/null || {
      echo "peer phase=orphan stage=deadline elapsed=$((SECONDS - phase_start))" >&2
      return 1
    }
    if peer_tunnel_remove_orphan_children; then
      peer_orphans_clear=1
      break
    fi
    repair_sleep 1 || {
      echo "peer phase=orphan stage=wait elapsed=$((SECONDS - phase_start))" >&2
      return 1
    }
  done
  if [ "$peer_orphans_clear" != 1 ]; then
    echo "old peer tunnel connections did not stop" >&2
    echo "peer phase=orphan stage=clear elapsed=$((SECONDS - phase_start))" >&2
    return 1
  fi
  [ "$peer_enabled" = 1 ] || return 0
  repair_stage=peer-api
  peer_tunnel_remove_duplicate_api_supervisors "$peer_keep_current" || {
    echo "peer phase=api stage=duplicate-cleanup elapsed=$((SECONDS - phase_start))" >&2
    return 1
  }
  peer_tunnel_remove_orphan_api_listener || {
    echo "peer phase=api stage=orphan-listener elapsed=$((SECONDS - phase_start))" >&2
    return 1
  }
  peer_tunnel_connection "api" \
    "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" \
    -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" || {
      echo "peer phase=api stage=connection elapsed=$((SECONDS - phase_start))" >&2
      return 1
    }

  # Keep each SSH connection to two source ports so all configured source listeners
  # have independent bounded workers.
  local peer_lane_size=2
  local lane_start=0 lane_number=0
  local -a lane_ports lane_forwards
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    repair_stage=peer-lane-$lane_number
    lane_ports=("${lora_source_ports[@]:lane_start:peer_lane_size}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=(
        -R "127.0.0.1:$lora_source_port:127.0.0.1:$relay_port"
      )
    done
    if ! peer_tunnel_connection "lora-$lane_number" \
      "-R 127.0.0.1:${lane_ports[0]}:127.0.0.1:$relay_port" \
      "${lane_forwards[@]}"; then
      echo "peer phase=lane lane=$lane_number ports=${lane_ports[0]}-${lane_ports[-1]} stage=connection elapsed=$((SECONDS - phase_start))" >&2
      return 1
    fi
    lane_start=$((lane_start + peer_lane_size))
    lane_number=$((lane_number + 1))
    [ "$lane_start" -lt "${#lora_source_ports[@]}" ] && sleep 0.1
  done
  if [ "${tunnels_only:-0}" = 0 ]; then
    echo "peer tunnel :$peer_api_port -> $peer_name:8000; adapter source reverse ports ${lora_source_ports[0]}-${lora_source_ports[-1]} in $lane_number SSH lanes"
  fi
}

supervisor_lane_inventory_ready() {
  local kind=$1 expected_lanes=$2 allow_missing=${3:-0}
  local pid command_line marker lane prefix match_function
  local inventory_count=0 process_output
  local -a seen_lanes=()
  case "$kind" in
    china) prefix=china-tunnel-lora; match_function=china_tunnel_supervisor_match ;;
    peer) prefix=peer-tunnel-lora; match_function=peer_tunnel_supervisor_match ;;
    *) return 1 ;;
  esac
  if [ "${supervisor_process_snapshot_active:-0}" = 1 ]; then
    process_output=$supervisor_process_snapshot
  else
    process_output=$(ps -eo pid=,args=) || return 1
  fi
  while read -r pid command_line; do
    if [[ "$command_line" =~ \#\ ($prefix-[^[:space:]]+)[[:space:]]*$ ]]; then
      marker=${BASH_REMATCH[1]}
      case "$pid" in ''|*[!0-9]*) return 1 ;; esac
      "$match_function" "$command_line" current "$marker" || return 1
      lane=${marker#${prefix}-}
      case "$lane" in ''|*[!0-9]*) return 1 ;; esac
      [ "$lane" -lt "$expected_lanes" ] || return 1
      case " ${seen_lanes[*]} " in
        *" $lane "*) return 1 ;;
        *) seen_lanes+=("$lane") ;;
      esac
      inventory_count=$((inventory_count + 1))
    fi
  done <<<"$process_output"
  if [ "$inventory_count" -eq "$expected_lanes" ]; then
    return 0
  fi
  [ "$allow_missing" = 1 ] && [ "$inventory_count" -lt "$expected_lanes" ]
}

china_tunnel_supervisors_ready() {
  local lane_start=0 lane_number=0 lane_ports=() lane_forwards=() pid command_line
  local marker ready lora_source_port matching_pid matching_command match_count invalid matching_pids
  local expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  local supervisor_process_snapshot tunnel_group_process_snapshot
  local china_listener_snapshot
  local supervisor_process_snapshot_active=1 tunnel_group_process_snapshot_active=1
  local china_listener_snapshot_active=1
  supervisor_process_snapshot=$(ps -eo pid=,args=) || return 1
  tunnel_group_process_snapshot=$(ps -eo pgid=,args=) || return 1
  china_listener_snapshot=$(ss -H -ltnp 2>/dev/null) || return 1
  supervisor_lane_inventory_ready china "$expected_lanes" || return 1
  china_tunnel_preflight_pgids=()
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    lane_ports=("${lora_source_ports[@]:lane_start:2}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=("-L 127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port")
    done
    marker="china-tunnel-lora-$lane_number"
    ready=0
    matching_pid=
    match_count=0
    invalid=0
    matching_pids=$(awk -v suffix="# $marker" 'index($0, suffix) {print $1}' \
      <<<"$supervisor_process_snapshot") || return 1
    while read -r pid; do
      [ -n "$pid" ] || continue
      command_line=$(process_command_line "$pid")
      case "$?" in
        0) ;;
        2) continue ;;
        *) return 1 ;;
      esac
      if china_tunnel_supervisor_forwards_match "$command_line" current "$marker" "${lane_forwards[@]}"; then
        match_count=$((match_count + 1))
        matching_pid=$pid
        matching_command=$command_line
      elif [ -n "$command_line" ] && china_tunnel_marker_match "$command_line" "$marker"; then
        invalid=1
      fi
    done <<<"$matching_pids"
    [ "$invalid" -eq 0 ] || return 1
    [ "$match_count" -eq 1 ] || return 1
    china_tunnel_supervisor_owns_forwards "$matching_pid" "$matching_command" current \
      "${lane_forwards[@]}" || return 1
    china_tunnel_preflight_group "$matching_pid" "$matching_command" 0 "$marker" || return 1
    lane_start=$((lane_start + 2))
    lane_number=$((lane_number + 1))
  done
  supervisor_lane_inventory_ready china "$expected_lanes"
}

peer_tunnel_lane_supervisor() {
  local marker=$1
  shift
  local matching_pids pid command_line matching_count=0
  peer_ready_pid=
  peer_ready_command=
  if [ "${peer_process_snapshot_active:-0}" = 1 ]; then
    matching_pids=$(awk -v suffix="# $marker" 'index($0, suffix) {print $1}' \
      <<<"$peer_process_snapshot") || return 1
  else
    matching_pids=$(pgrep_matches "$marker") || return 1
  fi
  while read -r pid; do
    [ -n "$pid" ] || continue
    command_line=$(process_command_line "$pid") || return 1
    if peer_tunnel_supervisor_forwards_match "$command_line" current "$marker" "$@"; then
      matching_count=$((matching_count + 1))
      peer_ready_pid=$pid
      peer_ready_command=$command_line
    fi
  done <<<"$matching_pids"
  [ "$matching_count" -eq 1 ] || return 2
}

peer_tunnel_supervisors_ready() {
  local lane_start=0 lane_number=0 lane_ports=() lane_forwards=() pid command_line
  local marker lora_source_port matching_pid matching_command supervisor_start child_status
  local allow_repair=${1:-1}
  local supervisor_process_snapshot peer_process_snapshot
  local supervisor_process_snapshot_active=1 peer_process_snapshot_active=1
  [ "$peer_enabled" = 1 ] || return 0
  supervisor_process_snapshot=$(ps -eo pid=,args=) || return 1
  peer_process_snapshot=$(ps -eo pid=,ppid=,args=) || return 1
  local expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  peer_supervisor_pids=()
  peer_supervisor_starts=()
  peer_supervisor_commands=()
  peer_child_pids=()
  peer_child_starts=()
  peer_child_commands=()
  peer_child_shapes=()
  peer_child_forwards=()
  supervisor_lane_inventory_ready peer "$expected_lanes" 1 || return 1
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    lane_ports=("${lora_source_ports[@]:lane_start:2}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=("-R 127.0.0.1:$lora_source_port:127.0.0.1:$relay_port")
    done
    marker="peer-tunnel-lora-$lane_number"
    if peer_tunnel_lane_supervisor "$marker" "${lane_forwards[@]}"; then
      matching_pid=$peer_ready_pid
      matching_command=$peer_ready_command
    else
      child_status=$?
      [ "$child_status" -eq 2 ] || return 1
      [ "$allow_repair" = 1 ] || return 1
      peer_tunnel_connection "lora-$lane_number" \
        "-R 127.0.0.1:${lane_ports[0]}:127.0.0.1:$relay_port" \
        "${lane_forwards[@]}" || return 1
      supervisor_process_snapshot=$(ps -eo pid=,args=) || return 1
      peer_process_snapshot=$(ps -eo pid=,ppid=,args=) || return 1
      peer_tunnel_lane_supervisor "$marker" "${lane_forwards[@]}" || return 1
      matching_pid=$peer_ready_pid
      matching_command=$peer_ready_command
      peer_tunnel_wait_child_identity "$matching_pid" current "$marker" \
        "${lane_forwards[@]}" || return 1
    fi
    supervisor_start=$(process_start_time "$matching_pid") || return 1
    peer_snapshot_child_pid=
    peer_snapshot_child_command=
    peer_snapshot_child_start=
    if peer_tunnel_supervisor_child_identity "$matching_pid" current "$marker" "${lane_forwards[@]}"; then
      child_status=0
    else
      child_status=$?
    fi
    if [ "$child_status" -ne 0 ]; then
      [ "$child_status" -eq 1 ] || return 1
      [ "$allow_repair" = 1 ] || return 1
      if peer_tunnel_wait_child_identity "$matching_pid" current "$marker" \
          "${lane_forwards[@]}"; then
        child_status=0
      else
        child_status=$?
        [ "$child_status" -eq 1 ] || return 1
        peer_tunnel_connection "lora-$lane_number" \
          "-R 127.0.0.1:${lane_ports[0]}:127.0.0.1:$relay_port" \
          "${lane_forwards[@]}" || return 1
        supervisor_process_snapshot=$(ps -eo pid=,args=) || return 1
        peer_process_snapshot=$(ps -eo pid=,ppid=,args=) || return 1
        peer_tunnel_lane_supervisor "$marker" "${lane_forwards[@]}" || return 1
        matching_pid=$peer_ready_pid
        matching_command=$peer_ready_command
        peer_tunnel_wait_child_identity "$matching_pid" current "$marker" \
          "${lane_forwards[@]}" || return 1
        supervisor_start=$(process_start_time "$matching_pid") || return 1
      fi
    fi
    peer_supervisor_pids[$marker]=$matching_pid
    peer_supervisor_starts[$marker]=$supervisor_start
    peer_supervisor_commands[$marker]=$matching_command
    peer_child_pids[$marker]=$peer_snapshot_child_pid
    peer_child_starts[$marker]=$peer_snapshot_child_start
    peer_child_commands[$marker]=$peer_snapshot_child_command
    peer_child_shapes[$marker]=current
    peer_child_forwards[$marker]="${lane_forwards[*]}"
    lane_start=$((lane_start + 2))
    lane_number=$((lane_number + 1))
  done
  supervisor_lane_inventory_ready peer "$expected_lanes"
}

peer_tunnel_api_supervisor_ready() {
  local marker=peer-tunnel-api pid command_line matching_pid= matching_command= matching_pids
  local match_count=0 supervisor_start
  local api_forward="-L 127.0.0.1:$peer_api_port:127.0.0.1:8000"
  [ "$peer_enabled" = 1 ] || return 0
  matching_pids=$(pgrep_matches "$marker") || return 2
  while read -r pid; do
    [ -n "$pid" ] || continue
    command_line=$(process_command_line "$pid") || return 2
    if peer_tunnel_supervisor_forwards_match "$command_line" api "$marker" "$api_forward"; then
      match_count=$((match_count + 1))
      matching_pid=$pid
      matching_command=$command_line
    fi
  done <<<"$matching_pids"
  [ "$match_count" -eq 1 ] || return 1
  supervisor_start=$(process_start_time "$matching_pid") || return 2
  peer_snapshot_child_pid=
  peer_snapshot_child_command=
  peer_snapshot_child_start=
  peer_tunnel_supervisor_child_identity "$matching_pid" api "$marker" "$api_forward" || return $?
  peer_supervisor_pids[$marker]=$matching_pid
  peer_supervisor_starts[$marker]=$supervisor_start
  peer_supervisor_commands[$marker]=$matching_command
  peer_child_pids[$marker]=$peer_snapshot_child_pid
  peer_child_starts[$marker]=$peer_snapshot_child_start
  peer_child_commands[$marker]=$peer_snapshot_child_command
  peer_child_shapes[$marker]=api
  peer_child_forwards[$marker]=$api_forward
}

adapter_tunnel_local_snapshot() {
  local allow_peer_repair=${1:-1}
  repair_remaining >/dev/null || return 1
  china_tunnel_supervisors_ready || return 1
  if [ "$peer_enabled" = 1 ]; then
    peer_tunnel_supervisors_ready "$allow_peer_repair" || return 1
    tunnel_listener_ready "$peer_api_port" peer-tunnel-api \
      "$peer_ssh_user@$peer_host" 1 \
      -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" || return 1
    peer_tunnel_api_supervisor_ready || return 1
  fi
  if [ "$secondary_enabled" = 1 ]; then
    generic_tunnel_listener_ready "$secondary_local_api_port" \
      "zmq-tunnel-$secondary_local_api_port" "$secondary_ssh_target" \
      "127.0.0.1:$secondary_local_api_port:127.0.0.1:$secondary_remote_api_port" \
      no "${secondary_ssh_options[@]}" || return 1
  fi
}

adapter_tunnel_remote_snapshot() {
  [ "$peer_enabled" = 1 ] || return 0
  peer_remote_ports_ready 10
}

adapter_tunnel_exact_snapshot() {
  adapter_tunnel_local_snapshot 0 || return 1
  adapter_tunnel_remote_snapshot || return 1
}

adapter_tunnel_stability_snapshot() {
  peer_tunnel_supervisors_unchanged || return 1
  peer_tunnel_children_unchanged || return 1
  if [ "$peer_enabled" = 1 ]; then
    peer_tunnel_api_listener_owner || return 1
  fi
  if [ "$secondary_enabled" = 1 ]; then
    generic_tunnel_listener_ready "$secondary_local_api_port" \
      "zmq-tunnel-$secondary_local_api_port" "$secondary_ssh_target" \
      "127.0.0.1:$secondary_local_api_port:127.0.0.1:$secondary_remote_api_port" \
      no "${secondary_ssh_options[@]}" || return 1
  fi
  adapter_tunnel_remote_snapshot
}

adapter_tunnel_readiness() {
  local tunnels_ready=0 readiness_port="" listener_status=0 remote_ready=1 attempt api_repair_attempted=0
  for attempt in $(seq 1 30); do
    repair_stage=local-peer-readiness
    repair_remaining >/dev/null || return 1
    tunnels_ready=1
    if [ "$tunnels_ready" = 1 ] && [ "$peer_enabled" = 1 ]; then
      repair_stage=local-peer-readiness
      peer_tunnel_supervisors_ready || {
        peer_readiness_failure="supervisor inventory/identity attempt $attempt"
        tunnels_ready=0
        readiness_port=peer
      }
      if [ "$tunnels_ready" = 1 ]; then
        repair_stage=local-api-readiness
        listener_status=0
        peer_tunnel_api_listener_owner || listener_status=$?
        if [ "$listener_status" = 0 ]; then
          peer_tunnel_api_supervisor_ready || listener_status=$?
        fi
        if [ "$listener_status" -gt 1 ]; then
          echo "peer API ownership could not be verified; no repair started" >&2
          return 1
        fi
        if [ "$listener_status" -ne 0 ]; then
          tunnels_ready=0
          readiness_port=$peer_api_port
          if [ "$api_repair_attempted" = 0 ]; then
            api_repair_attempted=1
            repair_stage=peer-api-repair
            echo "adapter tunnel retry stage=$repair_stage attempt=1 port=$peer_api_port" >&2
            peer_tunnel_connection api \
              "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" \
              -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" || return 1
          fi
        fi
      fi
    fi
    if [ "$tunnels_ready" = 1 ] && [ "$secondary_enabled" = 1 ]; then
      repair_stage=local-secondary-readiness
      listener_status=0
      generic_tunnel_listener_ready "$secondary_local_api_port" "zmq-tunnel-$secondary_local_api_port" \
        "$secondary_ssh_target" \
        "127.0.0.1:$secondary_local_api_port:127.0.0.1:$secondary_remote_api_port" \
        no "${secondary_ssh_options[@]}" || listener_status=$?
      if [ "$listener_status" -ne 0 ]; then
        tunnels_ready=0
        readiness_port=$secondary_local_api_port
      fi
    fi
    [ "$tunnels_ready" = 1 ] && break
    echo "adapter tunnel retry stage=$repair_stage attempt=$attempt port=${readiness_port:-source}" >&2
    repair_sleep 1 || return 1
  done
  if [ "$tunnels_ready" != 1 ]; then
    if [ -n "$readiness_port" ]; then
      echo "adapter tunnel readiness failed for local port $readiness_port: listener owner is not the expected SSH supervisor" >&2
    else
      echo "adapter tunnel reconciliation did not establish all configured forwards" >&2
    fi
    return 1
  fi
  if [ "$peer_enabled" = 1 ]; then
    remote_ready=0
    for attempt in $(seq 1 10); do
      repair_stage=remote-readiness
      repair_remaining >/dev/null || return 1
      if peer_remote_ports_ready 10; then
        sleep 0.2
        repair_remaining >/dev/null || return 1
        if peer_remote_ports_ready 10; then
          if peer_tunnel_current_children_ready; then
            remote_ready=1
            break
          else
            peer_readiness_failure="post-probe child check attempt $attempt"
          fi
        else
          peer_readiness_failure="E2E probe pass2 attempt $attempt"
        fi
      else
        peer_readiness_failure="E2E probe pass1 attempt $attempt"
      fi
      if [ -n "${peer_missing_remote_ports:-}" ]; then
        peer_repair_missing_remote_lanes || {
          peer_readiness_failure="port repair attempt $attempt"
          return 1
        }
        peer_tunnel_supervisors_ready || {
          peer_readiness_failure="supervisor inventory/identity after port repair attempt $attempt"
          return 1
        }
      fi
      echo "adapter tunnel retry stage=$repair_stage attempt=$attempt reason=${peer_readiness_failure:-unknown}" >&2
      repair_sleep 1 || return 1
    done
  fi
  if [ "$remote_ready" != 1 ]; then
    echo "adapter tunnel reconciliation did not establish all remote reverse forwards" \
      "(${peer_readiness_failure:-unknown stage})" >&2
    return 1
  fi
  for _ in $(seq 1 10); do
    repair_stage=final-stability
    if adapter_tunnel_stability_snapshot; then
      return 0
    fi
    peer_readiness_failure="final local/remote snapshot attempt $_"
    echo "adapter tunnel retry stage=$repair_stage attempt=$_" >&2
    repair_sleep 1 || return 1
  done
  echo "adapter tunnel final stability check failed" >&2
  return 1
}

adapter_relay_start() {
  local running relay_ready=0 relay_probe relay_status
  if [ "$peer_enabled" != 1 ]; then
    [ "$tunnels_only" = 1 ] || sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
    return 0
  fi
  sudo install -d -o 1000 -g 1000 -m 0755 "$relay_cache_dir"
  running=$(sudo docker inspect "$relay_container" --format '{{.State.Running}}' 2>/dev/null || true)
    relay_probe=$(curl -sS --max-time 2 -H "Host: 127.0.0.1:${lora_source_ports[0]}" \
      -w '%{http_code}' "http://127.0.0.1:$relay_port/" 2>/dev/null || true)
    relay_status=${relay_probe: -3}
    relay_probe=${relay_probe:0:${#relay_probe}-3}
    if [ "$running" = true ] && ss -tln 2>/dev/null | grep -q "127.0.0.1:$relay_port" && \
        [ "$relay_status" = 404 ] && [ "$relay_probe" = $'source URL is not configured\n' ]; then
      return 0
  fi
  sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
  sudo docker run -d --name "$relay_container" --restart unless-stopped \
    --user 0:0 \
    --network host \
    --log-opt max-size=128m --log-opt max-file=2 \
    -v "$relay_cache_dir":"$relay_cache_dir":ro \
    -v "$deployment/prime-rl/src/prime_rl/inference/relay.py":/app/relay.py:ro \
    --entrypoint /app/.venv/bin/python \
    "$image" /app/relay.py \
    --host 127.0.0.1 --port "$relay_port" \
    --cache-dir "$relay_cache_dir" \
    --source-port-first "${lora_source_ports[0]}" \
    --source-port-last "${lora_source_ports[-1]}" \
    --canonical-source-url "$relay_source_url" >/dev/null || return 1
  for _ in $(seq 1 60); do
    running=$(sudo docker inspect "$relay_container" --format '{{.State.Running}}' 2>/dev/null || true)
    relay_probe=$(curl -sS --max-time 2 -H "Host: 127.0.0.1:${lora_source_ports[0]}" \
      -w '%{http_code}' "http://127.0.0.1:$relay_port/" 2>/dev/null || true)
    relay_status=${relay_probe: -3}
    relay_probe=${relay_probe:0:${#relay_probe}-3}
    if [ "$running" = true ] && ss -tln 2>/dev/null | grep -q "127.0.0.1:$relay_port" && \
        [ "$relay_status" = 404 ] && [ "$relay_probe" = $'source URL is not configured\n' ]; then
      relay_ready=1
      break
    fi
    sleep 1
  done
  [ "$relay_ready" = 1 ] || {
    sudo docker logs --tail 100 "$relay_container" >&2 || true
    sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
    return 1
  }
}

china_legacy_cleanup_required=0
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
china_legacy_cleanup_orphan_pids=()
china_legacy_cleanup_ports=()
repair_acquire_lock || repair_fail "could not acquire adapter tunnel repair lock"
if ! adapter_relay_start; then
  repair_fail "adapter relay unavailable"
fi
tunnels_prevalidated=0
if [ "$restart_tunnels" = 0 ]; then
  repair_stage=existing-local
  if adapter_tunnel_local_snapshot 0; then
    repair_stage=existing-source
    if china_source_ports_ready; then
      repair_stage=existing-remote
      if adapter_tunnel_remote_snapshot; then
        repair_stage=existing-stability
        if repair_sleep 0.2 && adapter_tunnel_stability_snapshot; then
          tunnels_prevalidated=1
        fi
      fi
    fi
  fi
  if [ "$tunnels_prevalidated" != 1 ]; then
    echo "existing adapter tunnel check failed at $repair_stage; running bounded repair" >&2
  fi
fi
if [ "$tunnels_prevalidated" != 1 ]; then
repair_stage=china-legacy
if [ "$restart_tunnels" = 1 ]; then
  china_tunnel_remove_all_owned || repair_fail "full reset cleanup failed"
else
  china_tunnel_remove_legacy || repair_fail "legacy cleanup failed"
fi
repair_remaining >/dev/null || repair_fail "deadline exceeded during cleanup"
china_tunnel_remove_orphan_children || repair_fail "orphan China tunnel cleanup failed"
if [ "$china_legacy_cleanup_required" = 1 ] && ! china_tunnel_wait_legacy_clear; then
  repair_fail "old China tunnel groups did not stop"
fi
repair_remaining >/dev/null || repair_fail "deadline exceeded after cleanup"
china_lane_size=2
china_lane_start=0 china_lane_number=0
china_lane_ports=() china_lane_forwards=()
while [ "$china_lane_start" -lt "${#lora_source_ports[@]}" ]; do
  repair_stage=china-lane-$china_lane_number
  china_lane_ports=("${lora_source_ports[@]:china_lane_start:china_lane_size}")
  china_lane_forwards=()
  for lora_source_port in "${china_lane_ports[@]}"; do
    china_lane_forwards+=(
      -L "127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
    )
  done
  china_tunnel_connection "lora-$china_lane_number" \
    "${china_lane_forwards[@]}" || repair_fail "China lane $china_lane_number failed"
  china_lane_start=$((china_lane_start + china_lane_size))
  china_lane_number=$((china_lane_number + 1))
done
repair_stage=china-source
china_source_ports_ready || repair_fail "China adapter source check failed"
if [ "$tunnels_only" = 0 ]; then
  echo "China adapter forwards ${lora_source_ports[0]}-${lora_source_ports[-1]} in $china_lane_number SSH lanes"
fi
if ! peer_tunnel; then
  # Keep established China and peer lanes available when one peer lane fails.
  # The EXIT rollback still handles unregistered and starting processes.
  repair_preserve_created_tunnels
  repair_fail "peer tunnel reconciliation failed"
fi
if [ "$secondary_enabled" = 1 ]; then
  repair_stage=secondary
  if [ "$tunnels_only" = 1 ]; then
    tunnel "$secondary_local_api_port" "127.0.0.1:$secondary_remote_api_port" \
      "$secondary_ssh_target" no "${secondary_ssh_options[@]}" || repair_fail "secondary tunnel failed"
  elif ! tunnel "$secondary_local_api_port" "127.0.0.1:$secondary_remote_api_port" \
      "$secondary_ssh_target" no "${secondary_ssh_options[@]}"; then
    echo "secondary judge tunnel failed" >&2
    exit 1
  fi
fi
if ! adapter_tunnel_readiness; then
  repair_fail "exact tunnel readiness checks failed"
fi
fi
repair_committed=1
repair_created_tunnels=()
repair_unregistered_tunnels=()
repair_starting_tunnels=()
repair_unidentified_tunnels=()
repair_release_lock || repair_fail "could not release adapter tunnel repair lock"
if [ "$tunnels_only" = 1 ]; then
  echo "adapter tunnel reconciliation complete"
  exit 0
fi
ports=("${lora_source_ports[@]}")
if [ "$peer_enabled" = 1 ]; then
  ports+=("$peer_api_port")
fi
if [ "$secondary_enabled" = 1 ]; then
  ports+=("$secondary_local_api_port")
fi
if [ "$peer_enabled" = 1 ]; then
  ports+=("$relay_port")
fi
if [ "$transport" = zmq ]; then
  tunnel 29620 10.92.20.82:29620 "$trainer_ssh_target" yes
  ports+=(29620)
fi

sleep 6
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN"
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN/$RUN_NAME"
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN/$RUN_NAME/hf-cache"
broadcast_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME/broadcasts"
for path in "$broadcast_dir" "$broadcast_dir/step_0"; do
  [ ! -L "$path" ] || {
    echo "broadcast path is a symlink: $path" >&2
    exit 1
  }
done
for step_dir in "$broadcast_dir"/step_*; do
  [ -d "$step_dir" ] || continue
  [[ "$step_dir" =~ /step_[0-9]+$ ]] || continue
  [ ! -L "$step_dir" ] || {
    echo "broadcast step path is a symlink: $step_dir" >&2
    exit 1
  }
done
sudo install -d -o 1000 -g 1000 -m 0755 "$broadcast_dir" "$broadcast_dir/step_0"
for step_dir in "$broadcast_dir"/step_*; do
  [ -d "$step_dir" ] || continue
  [[ "$step_dir" =~ /step_[0-9]+$ ]] || continue
  sudo install -d -o 1000 -g 1000 -m 0755 "$step_dir"
  sudo -u '#1000' test -w "$step_dir" || {
    echo "broadcast step path is not writable by UID 1000: $step_dir" >&2
    exit 1
  }
done
sudo -u '#1000' test -w "$broadcast_dir" && sudo -u '#1000' test -w "$broadcast_dir/step_0" || {
  echo "broadcast paths are not writable by UID 1000: $broadcast_dir" >&2
  exit 1
}
for p in "${ports[@]}"; do
  ss -tln 2>/dev/null | grep -q "127.0.0.1:$p" && echo "tunnel :$p up" || echo "tunnel :$p DOWN"
done

sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
sudo docker run -d --name "$ORCH_CT" \
  --network host --restart no \
  --env-file /data/trinity/training.env \
  -e VLLM_API_KEY=unused \
  -e PYTHONPATH="$deployment/environments/decensor/src:$deployment/environments/general-agent:/app/src" \
  -e HF_HOME="/data/glm52-rl/runs/$RUN/$RUN_NAME/hf-cache" \
  -v /data/glm52-rl:/data/glm52-rl \
  -v glm52-fp8-model:/models/glm52-fp8:ro \
  -v "$deployment/prime-rl/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
  -v "$deployment/prime-rl/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
  -v "$deployment/prime-rl/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
  -v "$deployment/prime-rl/deps/verifiers/verifiers/v1/harnesses/null/harness.py":/app/deps/verifiers/verifiers/v1/harnesses/null/harness.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/algo/grpo.py":/app/src/prime_rl/orchestrator/algo/grpo.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/dispatcher.py":/app/src/prime_rl/orchestrator/dispatcher.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/orchestrator.py":/app/src/prime_rl/orchestrator/orchestrator.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/train_sink.py":/app/src/prime_rl/orchestrator/train_sink.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
  -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/shared.py":/app/packages/prime-rl-configs/src/prime_rl/configs/shared.py:ro \
  -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py":/app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro \
  --entrypoint /app/.venv/bin/orchestrator "$image" \
  @ "$config_file"
echo "orchestrator started on $(hostname) with $(basename "$config_file")"

# The China-side supervisor refuses to restart a failed trainer because the
# optimizer has no resumable checkpoint. Stop rollout production here as soon
# as any trainer worker is gone, so the run cannot accumulate unusable batches.
pkill -f '[t]rainer_failure_watchdog.sh' 2>/dev/null || true
nohup setsid env RUN="$RUN" RUN_NAME="$RUN_NAME" \
  TRAINER_ADDRESSES="$TRAINER_ADDRESSES" \
  TRAINER_CT="$TRAINER_CT" ORCH_CT="$ORCH_CT" \
  bash "$deployment/training_ops/runtime/h200/trainer_failure_watchdog.sh" \
  >>"/data/glm52-rl/trainer-failure-watchdog-$RUN.log" 2>&1 </dev/null &
trainer_watchdog_pid=$!
sleep 1
if ! kill -0 "$trainer_watchdog_pid" 2>/dev/null; then
  echo "ERROR: trainer failure watchdog exited during startup" >&2
  sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
  exit 1
fi
echo "trainer failure watchdog started (pid $trainer_watchdog_pid)"

ctl=/data/glm52-rl/runs/$RUN/$RUN_NAME/control/orch.toml
if [ "$transport" = filesystem ]; then
  publisher_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"
  publisher_pid_file="$publisher_dir/.batch-sync.pid"
  publisher_lock="$publisher_dir/.batch-sync.lock"
  if [ -r "$publisher_pid_file" ]; then
    old_publisher_pid=$(cat "$publisher_pid_file")
    case "$old_publisher_pid" in
      ''|*[!0-9]*) ;;
      *)
        # The publisher runs under setsid. Stop the complete group so an rsync
        # child cannot inherit fd 9 and keep the flock after its parent exits.
        kill -- "-$old_publisher_pid" 2>/dev/null || kill "$old_publisher_pid" 2>/dev/null || true
        ;;
    esac
  fi
  pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null || true
  # A terminated publisher can remove its pid file before a timeout/rsync/ssh
  # child exits. That child inherits fd 9 and keeps the lock, so a normal
  # process-name kill cannot find it. Stop only process groups that still hold
  # this exact run's lock before starting the replacement publisher.
  if ! flock -n "$publisher_lock" -c true && command -v lsof >/dev/null 2>&1; then
    lock_holders=$(lsof -t -- "$publisher_lock" 2>/dev/null | sort -u || true)
    for holder in $lock_holders; do
      holder_pgid=$(ps -o pgid= -p "$holder" 2>/dev/null | tr -d ' ')
      case "$holder_pgid" in
        ''|*[!0-9]*) ;;
        *) kill -- "-$holder_pgid" 2>/dev/null || kill "$holder" 2>/dev/null || true ;;
      esac
    done
  fi
  publisher_lock_ready=0
  for _ in $(seq 1 30); do
    if flock -n "$publisher_lock" -c true; then
      publisher_lock_ready=1
      break
    fi
    sleep 1
  done
  if [ "$publisher_lock_ready" != 1 ]; then
    echo "ERROR: old durable batch publisher still owns $publisher_lock" >&2
    exit 1
  fi
  nohup setsid env SOURCE_RUN_DIR="/data/glm52-rl/runs/$RUN/$RUN_NAME" \
    DEST_HOST="$trainer_ssh_target" \
    "$deployment/training_ops/runtime/h200/sync_rollout_batches.sh" \
    >>"/data/glm52-rl/batch-sync-$RUN.log" 2>&1 </dev/null &
  publisher_pid=$!
  sleep 1
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    echo "ERROR: durable batch publisher exited during startup" >&2
    exit 1
  fi
  echo "durable batch publisher started (pid $publisher_pid)"
  exit
fi

for _ in $(seq 1 60); do
  [ -f "$ctl" ] && break
  sleep 2
done
if [ -f "$ctl" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$trainer_ssh_target" \
      "mkdir -p /data/glm52-rl/runs/$RUN/$RUN_NAME/control" \
    && scp -q -o BatchMode=yes "$ctl" "$trainer_ssh_target:$ctl" \
    && echo "control/orch.toml -> trainer ok" \
    || echo "WARNING: could not ship orch.toml to the trainer; it will not start" >&2
else
  echo "WARNING: $ctl never appeared; trainer will not start" >&2
fi
