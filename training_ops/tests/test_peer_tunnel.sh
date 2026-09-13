#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="$root/training_ops/runtime/h200/start_orchestrator.sh"
nodes_file="$root/configs/nodes.toml"
readarray -t configured_lora_ports < <(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    ports = tomllib.load(stream)["ports"]
for port in range(ports["lora_source_first"], ports["lora_source_last"] + 1):
    print(port)
PY
)
configured_relay_port=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    print(tomllib.load(stream)["ports"]["adapter_relay"])
PY
)
[ "${#configured_lora_ports[@]}" -eq 128 ] || {
  echo "expected 128 configured LoRA source ports" >&2
  exit 1
}
test_source_first=${configured_lora_ports[0]}
test_source_last=${configured_lora_ports[-1]}
channel_timeout=$(ssh -G -o 'ChannelTimeout=*-tcpip=45s' example.invalid 2>/dev/null |
  awk '$1 == "channeltimeout" {print $2; exit}')
[ "$channel_timeout" = "*-tcpip=45s" ] || {
  echo "OpenSSH does not accept the adapter channel timeout" >&2
  exit 1
}
case "$configured_relay_port" in ''|*[!0-9]*)
  echo "invalid configured adapter relay port" >&2
  exit 1
  ;;
esac
export TEST_SOURCE_FIRST="$test_source_first" TEST_SOURCE_LAST="$test_source_last" \
  TEST_SOURCE_COUNT="${#configured_lora_ports[@]}"
relay_block=$(sed -n '/sudo docker run -d --name/,/--canonical-source-url/p' "$script")
[[ "$relay_block" == *"--user 0:0"* && "$relay_block" == *'-v "$relay_cache_dir":"$relay_cache_dir":ro'* ]] || {
  echo "relay must run as root with a read-only adapter cache" >&2
  exit 1
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
real_cat=$(command -v cat)
real_ssh=$(command -v ssh)
cat >"$tmp/bin/cat" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  /proc/*/cmdline)
    pid=${1#/proc/}
    pid=${pid%/cmdline}
    fixture="${CHINA_PROC_FIXTURES:?missing China proc fixtures}/$pid"
    [ -s "$fixture" ] || exit 1
    exec "$REAL_CAT" "$fixture"
    ;;
  *) exec "$REAL_CAT" "$@" ;;
esac
EOF
cat >"$tmp/bin/pgrep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
pattern=${3:?missing pgrep pattern}
printf '%s' "$pattern" >"${PEER_QUERY:?}"
if [ -r "${PEER_TEST_LOG:?}" ] && grep -Fq -- "$pattern" "$PEER_TEST_LOG"; then
  case "$pattern" in
    china-tunnel-lora-*|peer-tunnel-lora-*)
      if [ "${PEER_DUPLICATE_MARKER:-}" = "$pattern" ]; then
        echo 10000
        echo 10001
        exit 0
      fi
      echo $((10000 + ${pattern##*-}))
      ;;
    '-L 127.0.0.1:'*)
      port=${pattern#-L 127.0.0.1:}
      port=${port%%:*}
      if [ "${PEER_STALE_GROUP_ACTIVE:-0}" = 1 ]; then
        echo 10000
        exit 0
      fi
      if [ "${PEER_CHINA_MODE:-0}" = 1 ] && [ "$port" -ge "${TEST_SOURCE_FIRST:?}" ] && [ "$port" -le "${TEST_SOURCE_LAST:?}" ]; then
        echo $((10000 + (port - TEST_SOURCE_FIRST)))
      else
        echo 1234
      fi
      ;;
    *)
      echo 1234
      ;;
  esac
  exit 0
fi
exit 1
EOF
cat >"$tmp/bin/setsid" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
line=$(printf '%s\n' "$*" | tr '\n' ' ')
printf '%s\n' "$line" >>"${PEER_TEST_LOG:?}"
EOF
cat >"$tmp/bin/ps" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-o" ] && [ "${2:-}" = "args=" ]; then
  pid=${4:?missing process id}
  if [ "${TUNNEL_OWNER_PID:-}" = "$pid" ]; then
    cat "${TUNNEL_OWNER_COMMAND:?}"
  elif [ "${TUNNEL_EXTRA_OWNER_PID:-}" = "$pid" ]; then
    cat "${TUNNEL_OWNER_COMMAND:?}"
  elif [ "${TUNNEL_SUPERVISOR_PID:-}" = "$pid" ]; then
    cat "${TUNNEL_SUPERVISOR_COMMAND:?}"
  elif [ -n "${CHINA_PROC_FIXTURES:-}" ] && [ -s "$CHINA_PROC_FIXTURES/$pid" ]; then
    "$REAL_CAT" "$CHINA_PROC_FIXTURES/$pid" | tr '\0' ' '
  elif [ "${PEER_CHINA_OWNER_MODE:-0}" = 1 ] && [ "$pid" -ge 30000 ] 2>/dev/null; then
    lane=$((pid - 30000))
    line=$(grep -E "done # (china|peer)-tunnel-lora-${lane}[[:space:]]*$" "${PEER_TEST_LOG:?}" | tail -n 1)
    line=${line#*while true; do }
    line=${line%% sleep 3*}
    printf '%s\n' "$line"
  elif [ -n "${PEER_DUPLICATE_COMMAND:-}" ] && [ "$pid" -ge 10000 ] 2>/dev/null; then
    cat "$PEER_DUPLICATE_COMMAND"
  elif [ "$pid" -ge 10000 ] 2>/dev/null; then
    if [ "${PEER_STALE_GROUP_ACTIVE:-0}" = 1 ] && [ -r "${PEER_STALE_GROUP_COMMAND:-}" ]; then
      cat "$PEER_STALE_GROUP_COMMAND"
      exit 0
    fi
    lane=$((pid - 10000))
    grep -E "done # (china|peer)-tunnel-lora-${lane}[[:space:]]*$" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  else
    query=$(<"${PEER_QUERY:?}")
    grep -F -- "$query" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  fi
elif [ "${1:-}" = "-o" ] && [ "${2:-}" = "ppid=" ]; then
  pid=${4:?missing process id}
  if [ "${TUNNEL_OWNER_PID:-}" = "$pid" ]; then
    printf '%s\n' "${TUNNEL_OWNER_PPID:?}"
  elif [ "${TUNNEL_EXTRA_OWNER_PID:-}" = "$pid" ]; then
    printf '%s\n' "${TUNNEL_OWNER_PPID:?}"
  elif [ "${PEER_CHINA_OWNER_MODE:-0}" = 1 ] && [ "$pid" -ge 30000 ] 2>/dev/null; then
    echo $((10000 + pid - 30000))
  else
    printf '%s\n' 1
  fi
elif [ "${1:-}" = "-o" ] && [ "${2:-}" = "pgid=" ]; then
  if [ "${PEER_TEST_PGID_PID:-}" = "${4:?missing process id}" ]; then
    echo "${PEER_TEST_PGID:?missing test process group id}"
  else
    echo "$4"
  fi
elif [ "${1:-}" = "-eo" ]; then
  if [ "${2:-}" = "pid=,args=" ] && [ -n "${PEER_INVENTORY_COUNTER:-}" ]; then
    inventory_count=$(cat "$PEER_INVENTORY_COUNTER" 2>/dev/null || echo 0)
    inventory_count=$((inventory_count + 1))
    printf '%s\n' "$inventory_count" >"$PEER_INVENTORY_COUNTER"
    if [ "$inventory_count" -ge 2 ] && [ -r "${PEER_INVENTORY_SECOND:-}" ]; then
      cat "$PEER_INVENTORY_SECOND"
      exit 0
    fi
  fi
  if [ -n "${PEER_GROUP_COMMANDS:-}" ]; then
    cat "$PEER_GROUP_COMMANDS"
  elif [ "${PEER_STALE_GROUP_ACTIVE:-0}" = 1 ] && [ -r "${PEER_STALE_GROUP_COMMAND:?}" ]; then
    printf '10000 %s\n' "$(<"$PEER_STALE_GROUP_COMMAND")"
  elif [ -r "${PEER_TEST_LOG:-}" ]; then
    while IFS= read -r group_command; do
      marker=$(sed -n -E 's/.*# ((china|peer)-tunnel-lora-[0-9][0-9]*)[[:space:]]*$/\1/p' <<<"$group_command")
      lane=${marker##*-}
      [ -n "$lane" ] || continue
      printf '%s %s\n' "$((10000 + lane))" "$group_command"
    done <"$PEER_TEST_LOG"
  fi
fi
EOF
cat >"$tmp/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ -n "${TUNNEL_SS_OUTPUT:-}" ] && [ "${PEER_STALE_LISTENERS_ACTIVE:-}" != 0 ]; then
  cat "${TUNNEL_SS_OUTPUT:?missing tunnel ss output}"
elif [ "${PEER_CHINA_OWNER_MODE:-0}" = 1 ] && [[ "$*" == *ltnp* ]]; then
  cat "${CHINA_OWNER_SS_OUTPUT:?missing China owner fixture}"
fi
EOF
cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = -G ] && [ -n "${SSH_G_OUTPUT:-}" ]; then
  cat "$SSH_G_OUTPUT"
else
  exec "$REAL_SSH" "$@"
fi
EOF
chmod +x "$tmp/bin/cat" "$tmp/bin/pgrep" "$tmp/bin/setsid" "$tmp/bin/ssh"
chmod +x "$tmp/bin/ps" "$tmp/bin/ss"
export REAL_CAT="$real_cat" REAL_SSH="$real_ssh"

start=$(grep -n '^generic_tunnel_ssh_match() {' "$script" | cut -d: -f1)
end=$(grep -n '^china_legacy_cleanup_required=0$' "$script" | cut -d: -f1)
source <(sed -n "${start},$((end - 1))p" "$script")

# The launcher preamble supplies these helpers in production. Keep this
# function-level fixture independent from its remote configuration parser.
tunnels_only=0
repair_remaining() {
  if [ "$tunnels_only" != 1 ]; then
    printf '2147483647\n'
    return 0
  fi
  remaining=$((tunnel_repair_deadline - SECONDS))
  [ "$remaining" -gt 0 ] || return 1
  printf '%s\n' "$remaining"
}
repair_timeout() {
  requested=$1
  remaining=$(repair_remaining) || return 1
  [ "$tunnels_only" != 1 ] || [ "$requested" -lt "$remaining" ] && printf '%s\n' "$requested" || printf '%s\n' "$remaining"
}
repair_sleep() {
  duration=$(repair_timeout "$1") || return 1
  sleep "$duration"
  repair_remaining >/dev/null
}
repair_register_tunnel() { :; }
repair_fail() { echo "adapter tunnel reconciliation failed: $*" >&2; exit 1; }

lora_source_ports=("${configured_lora_ports[@]}")
expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
expected_peer_lines=$((expected_lanes + 1))
peer_enabled=1
peer_api_port=18049
relay_port=$configured_relay_port
peer_name=h200b
peer_host=38.255.28.19
peer_ssh_port=20099
peer_ssh_user=root
peer_ssh_identity=/data/trinity/h200_rollout_ed25519
china_legacy_cleanup_required=0
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
log="$tmp/peer-tunnel.log"
query="$tmp/peer-query"

PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
for _ in $(seq 1 50); do
  [ -f "$log" ] && [ "$(wc -l <"$log")" -eq "$expected_peer_lines" ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq "$expected_peer_lines"

api_pattern='-L 127.0.0.1:18049:127.0.0.1:8000'
test "$(grep -F -c -- "$api_pattern" "$log")" -eq 1
test "$(grep -F -c -- '-R 127.0.0.1:' "$log")" -eq "$expected_lanes"
while IFS= read -r line; do
  case "$line" in
    *"$api_pattern"*"-R 127.0.0.1:"*)
      echo "API and reverse forwards share an SSH connection" >&2
      exit 1
      ;;
  esac
done <"$log"
for port in "${lora_source_ports[@]}"; do
  test "$(grep -F -c -- "-R 127.0.0.1:$port:127.0.0.1:$relay_port" "$log")" -eq 1
done
while IFS= read -r line; do
  reverse_count=$(grep -o -- '-R 127.0.0.1:' <<<"$line" | wc -l)
  test "$reverse_count" -eq 2
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done < <(grep -F -- '-R 127.0.0.1:' "$log")

# SSH matching rejects an extra token even when the forwards and target match.
peer_exact_command="ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port root@38.255.28.19"
peer_tunnel_ssh_args_match "$peer_exact_command" current
if peer_tunnel_ssh_args_match "${peer_exact_command/root@38.255.28.19/junk root@38.255.28.19}" current; then
  echo "peer SSH matcher accepted an extra token" >&2
  exit 1
fi
peer_supervisor_command="bash -c while true; do $peer_exact_command; sleep 3; done # peer-tunnel-lora-0"
peer_escaped_supervisor_command=$(sed 's/ChannelTimeout=\*-tcpip=45s/ChannelTimeout=\\*-tcpip=45s/' <<<"$peer_supervisor_command")
peer_tunnel_supervisor_match "$peer_supervisor_command" current peer-tunnel-lora-0
for hostile_target in \
  "root@38.255.28.19.evil" \
  "" \
  "root@38.255.28.19 extra"; do
  hostile_peer_supervisor=${peer_supervisor_command/root@38.255.28.19/$hostile_target}
  if peer_tunnel_supervisor_match "$hostile_peer_supervisor" current peer-tunnel-lora-0; then
    echo "peer supervisor accepted hostile target token: $hostile_target" >&2
    exit 1
  fi
done
for hostile_peer_supervisor in \
  "${peer_supervisor_command/; sleep 3; done/; evil; sleep 3; done}" \
  "${peer_supervisor_command/; sleep 3; done/; sleep 3; evil; done}" \
  "${peer_supervisor_command/#*# peer-tunnel-lora-0/*# peer-tunnel-lora-0 extra}" \
  "${peer_supervisor_command/while true; do/while false; do}" \
  "${peer_supervisor_command/while true; do/evil bash -c while true; do}"; do
  if peer_tunnel_supervisor_match "$hostile_peer_supervisor" current peer-tunnel-lora-0; then
    echo "peer supervisor accepted injected shell syntax" >&2
    exit 1
  fi
done

# A second launcher invocation finds the exact API and lane signatures.
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
test "$(wc -l <"$log")" -eq "$expected_peer_lines"

china_log="$tmp/china-tunnel.log"
trainer_ssh_target=china-trainer
adapter_port=8100
china_exact_command="ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer"
china_old_command_prefix="ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no"
china_supervisor_command="bash -c while true; do $china_exact_command; sleep 3; done # china-tunnel-lora-0"
china_escaped_supervisor_command=$(sed 's/ChannelTimeout=\*-tcpip=45s/ChannelTimeout=\\*-tcpip=45s/' <<<"$china_supervisor_command")
china_tunnel_supervisor_match "$china_supervisor_command" current china-tunnel-lora-0
for hostile_target in \
  "china-trainer.evil" \
  "" \
  "china-trainer extra"; do
  hostile_china_supervisor=${china_supervisor_command/china-trainer/$hostile_target}
  if china_tunnel_supervisor_match "$hostile_china_supervisor" current china-tunnel-lora-0; then
    echo "China supervisor accepted hostile target token: $hostile_target" >&2
    exit 1
  fi
done
for hostile_china_supervisor in \
  "${china_supervisor_command/; sleep 3; done/; evil; sleep 3; done}" \
  "${china_supervisor_command/; sleep 3; done/; sleep 3; evil; done}" \
  "${china_supervisor_command/#*# china-tunnel-lora-0/*# china-tunnel-lora-0 extra}" \
  "${china_supervisor_command/while true; do/while false; do}" \
  "${china_supervisor_command/while true; do/evil bash -c while true; do}"; do
  if china_tunnel_supervisor_match "$hostile_china_supervisor" current china-tunnel-lora-0; then
    echo "China supervisor accepted injected shell syntax" >&2
    exit 1
  fi
done
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
china_lane_size=2
china_lane_start=0
china_lane_number=0
china_lane_ports=()
china_lane_forwards=()
run_china_lanes() {
  china_lane_start=0
  china_lane_number=0
  while [ "$china_lane_start" -lt "${#lora_source_ports[@]}" ]; do
    china_lane_ports=("${lora_source_ports[@]:china_lane_start:china_lane_size}")
    china_lane_forwards=()
    for lora_source_port in "${china_lane_ports[@]}"; do
      china_lane_forwards+=(
        -L "127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
      )
    done
    PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_connection "lora-$china_lane_number" \
      "${china_lane_forwards[@]}" || return 1
    china_lane_start=$((china_lane_start + china_lane_size))
    china_lane_number=$((china_lane_number + 1))
  done
}

# A migration launch removes an old four-port China supervisor before
# creating sixty-four two-forward supervisors.
stale_group_command="$tmp/stale-group-command"
stale_group_kill_log="$tmp/stale-group-kill"
kill() {
  printf '%s\n' "$*" >>"$stale_group_kill_log"
  PEER_STALE_GROUP_ACTIVE=0
  export PEER_STALE_GROUP_ACTIVE
  sed -i '/# china-tunnel-lora-[01]$/d' "$china_log"
  case " $* " in
    *" 20001"*) : >"$CHINA_PROC_FIXTURES/20001" ;;
  esac
}

unrelated_group_command="$tmp/unrelated-group-command"
printf '%s\n' 'while true; do ssh -NT -L 127.0.0.1:18104:127.0.0.1:8100 -L 127.0.0.1:18105:127.0.0.1:8100 -L 127.0.0.1:18106:127.0.0.1:8100 -L 127.0.0.1:18107:127.0.0.1:8100 china-trainer; sleep 3; done # unrelated-supervisor' >"$unrelated_group_command"
printf '%s\n' "$(<"$unrelated_group_command")" >"$china_log"
export PEER_STALE_GROUP_ACTIVE=1 PEER_STALE_GROUP_COMMAND="$unrelated_group_command"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
test "${PEER_STALE_GROUP_ACTIVE:-0}" -eq 1
test ! -s "$stale_group_kill_log"
unset PEER_STALE_GROUP_ACTIVE PEER_STALE_GROUP_COMMAND
: >"$china_log"

# A migration launch also removes an old one-forward owned supervisor.
printf '%s\n' "while true; do $china_old_command_prefix -L 127.0.0.1:18100:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-0" >"$stale_group_command"
printf '%s\n' "$(<"$stale_group_command")" >"$china_log"
export PEER_STALE_GROUP_ACTIVE=1 PEER_STALE_GROUP_COMMAND="$stale_group_command"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
test "${PEER_STALE_GROUP_ACTIVE:-1}" -eq 0
test -s "$stale_group_kill_log"
unset PEER_STALE_GROUP_ACTIVE PEER_STALE_GROUP_COMMAND
: >"$china_log"

printf '%s\n' "while true; do $china_old_command_prefix -L 127.0.0.1:18104:127.0.0.1:8100 -L 127.0.0.1:18105:127.0.0.1:8100 -L 127.0.0.1:18106:127.0.0.1:8100 -L 127.0.0.1:18107:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-1" >"$stale_group_command"
printf '%s\n' "$(<"$stale_group_command")" >"$china_log"
export PEER_STALE_GROUP_ACTIVE=1 PEER_STALE_GROUP_COMMAND="$stale_group_command"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
test "${PEER_STALE_GROUP_ACTIVE:-1}" -eq 0
test -s "$stale_group_kill_log"
unset PEER_STALE_GROUP_ACTIVE PEER_STALE_GROUP_COMMAND

proc_fixtures="$tmp/proc-cmdline"
mkdir -p "$proc_fixtures"
write_child_cmdline() {
  local output=$1
  shift
  printf '%s\0' \
    /usr/bin/ssh -NT \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    -o 'ChannelTimeout=*-tcpip=45s' \
    -o TCPKeepAlive=yes \
    -o IPQoS=none \
    -o Compression=no \
    "$@" >"$output"
}
write_child_cmdline "$proc_fixtures/20001" \
  -L 127.0.0.1:18104:127.0.0.1:8100 \
  -L 127.0.0.1:18105:127.0.0.1:8100 \
  -L 127.0.0.1:18106:127.0.0.1:8100 \
  -L 127.0.0.1:18107:127.0.0.1:8100 \
  china-trainer
write_child_cmdline "$proc_fixtures/20002" \
  -L 127.0.0.1:18110:127.0.0.1:8100 \
  -L 127.0.0.1:18111:127.0.0.1:8100 \
  -L 127.0.0.1:18112:127.0.0.1:8100 \
  -L 127.0.0.1:18113:127.0.0.1:8100 \
  unrelated-trainer
write_child_cmdline "$proc_fixtures/20003" \
  -L 127.0.0.1:18114:127.0.0.1:8100 \
  -L 127.0.0.1:18195:127.0.0.1:8100 \
  -L "127.0.0.1:$((test_source_last + 1)):127.0.0.1:8100" \
  -L 127.0.0.1:18116:127.0.0.1:8100 \
  china-trainer
orphan_ss="$tmp/orphan-ss"
printf 'LISTEN 0 128 127.0.0.1:18104 0.0.0.0:* users:(("ssh",pid=20001,fd=5))\n' >"$orphan_ss"
printf 'LISTEN 0 128 127.0.0.1:18105 0.0.0.0:* users:(("ssh",pid=20001,fd=6))\n' >>"$orphan_ss"
printf 'LISTEN 0 128 127.0.0.1:18106 0.0.0.0:* users:(("ssh",pid=20001,fd=7))\n' >>"$orphan_ss"
printf 'LISTEN 0 128 127.0.0.1:18107 0.0.0.0:* users:(("ssh",pid=20001,fd=8))\n' >>"$orphan_ss"
printf 'LISTEN 0 128 127.0.0.1:18110 0.0.0.0:* users:(("ssh",pid=20002,fd=5))\n' >>"$orphan_ss"
printf 'LISTEN 0 128 127.0.0.1:18114 0.0.0.0:* users:(("ssh",pid=20003,fd=5))\n' >>"$orphan_ss"
export REAL_CAT="$real_cat" CHINA_PROC_FIXTURES="$proc_fixtures" TUNNEL_SS_OUTPUT="$orphan_ss"
PATH="$tmp/bin:$PATH" china_tunnel_remove_orphan_children
test ! -s "$proc_fixtures/20001"
test -s "$proc_fixtures/20002"
test -s "$proc_fixtures/20003"
test ! -s "$stale_group_kill_log" || ! grep -Fq -- '20002' "$stale_group_kill_log"
test ! -s "$stale_group_kill_log" || ! grep -Fq -- '20003' "$stale_group_kill_log"
unset CHINA_PROC_FIXTURES TUNNEL_SS_OUTPUT
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
china_legacy_cleanup_orphan_pids=()

stale_listener_output="$tmp/stale-listeners"
: >"$stale_listener_output"
for port in "${lora_source_ports[@]}"; do
  printf 'LISTEN 0 128 127.0.0.1:%s 0.0.0.0:*\n' "$port" >>"$stale_listener_output"
done
export TUNNEL_SS_OUTPUT="$stale_listener_output" PEER_STALE_LISTENERS_ACTIVE=1 PEER_TEARDOWN_POLLS=2
printf '%s\n' 'ssh -NT stale-listener-owner' >"$tmp/tunnel-owner"
china_legacy_cleanup_pids=(4242)
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_COMMAND="$tmp/tunnel-owner"
sleep() {
  if [ "${PEER_TEARDOWN_POLLS:-0}" -gt 0 ]; then
    PEER_TEARDOWN_POLLS=$((PEER_TEARDOWN_POLLS - 1))
    export PEER_TEARDOWN_POLLS
    if [ "$PEER_TEARDOWN_POLLS" -eq 0 ]; then
      PEER_STALE_LISTENERS_ACTIVE=0
      export PEER_STALE_LISTENERS_ACTIVE
      unset TUNNEL_OWNER_PID
    fi
  fi
}
PATH="$tmp/bin:$PATH" china_tunnel_wait_legacy_clear
test "${PEER_TEARDOWN_POLLS:-1}" -eq 0
PEER_STALE_LISTENERS_ACTIVE=1
export TUNNEL_OWNER_PID=4242 PEER_STALE_LISTENERS_ACTIVE
set +e
PATH="$tmp/bin:$PATH" china_tunnel_wait_legacy_clear
wait_status=$?
set -e
test "$wait_status" -ne 0
unset PEER_STALE_LISTENERS_ACTIVE PEER_TEARDOWN_POLLS TUNNEL_SS_OUTPUT \
  TUNNEL_OWNER_PID TUNNEL_OWNER_COMMAND
china_legacy_cleanup_pids=()
unset -f sleep kill

# A fresh launch creates sixty-four two-forward lanes.
kill() { return 0; }
china_owner_ss="$tmp/china-owner-ss"
: >"$china_owner_ss"
for ((owner_index=0; owner_index<${#lora_source_ports[@]}; owner_index++)); do
  printf 'LISTEN 0 128 127.0.0.1:%s 0.0.0.0:* users:(("ssh",pid=%s,fd=5))\n' \
    "${lora_source_ports[owner_index]}" "$((30000 + owner_index / 2))" >>"$china_owner_ss"
done
export PEER_CHINA_OWNER_MODE=1 CHINA_OWNER_SS_OUTPUT="$china_owner_ss"
run_china_lanes
for _ in $(seq 1 50); do
  [ -f "$china_log" ] && [ "$(wc -l <"$china_log")" -eq "$expected_lanes" ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq "$expected_lanes"
test "$(awk '/18104:127.0.0.1:8100/ && /18105:127.0.0.1:8100/ {count++} END {print count + 0}' "$china_log")" -eq 1
while IFS= read -r line; do
  forward_count=$(grep -o -- '-L 127.0.0.1:' <<<"$line" | wc -l)
  test "$forward_count" -eq 2
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done <"$china_log"
# A repeat launch preserves all matching groups without adding processes.
run_china_lanes
test "$(wc -l <"$china_log")" -eq "$expected_lanes"
# A target change fails closed and does not start a second owner.
trainer_ssh_target=china-trainer-regrouped
set +e
run_china_lanes
regroup_status=$?
set -e
test "$regroup_status" -ne 0
test "$(wc -l <"$china_log")" -eq "$expected_lanes"
trainer_ssh_target=china-trainer
unset PEER_CHINA_OWNER_MODE CHINA_OWNER_SS_OUTPUT

# A same-signature process with the wrong peer target is not reused.
printf '%s\n' "while true; do ssh -R 127.0.0.1:18100:127.0.0.1:$relay_port wrong@peer; sleep 3; done # peer-tunnel-lora-0" >"$log"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel_connection lora-0 \
  "-R 127.0.0.1:18100:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18100:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18101:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18102:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18103:127.0.0.1:$relay_port"
for _ in $(seq 1 50); do
  [ "$(wc -l <"$log")" -eq 2 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 2
grep -Fq -- 'root@38.255.28.19' "$log"

# A process missing a lane forward is not reused.
printf '%s\n' "while true; do ssh -R 127.0.0.1:18100:127.0.0.1:$relay_port root@38.255.28.19; sleep 3; done # peer-tunnel-lora-0" >"$log"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel_connection lora-0 \
  "-R 127.0.0.1:18100:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18100:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18101:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18102:127.0.0.1:$relay_port" \
  -R "127.0.0.1:18103:127.0.0.1:$relay_port"
for _ in $(seq 1 50); do
  [ "$(wc -l <"$log")" -eq 2 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 2
test "$(grep -F -c -- "-R 127.0.0.1:18103:127.0.0.1:$relay_port" "$log")" -eq 1

# A peer marker does not authorize killing an unrelated process in its group.
group_commands="$tmp/group-commands"
kill_log="$tmp/kill-log"
printf '%s\n' \
  "1234 bash -c while true; do ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port root@38.255.28.19; sleep 3; done # peer-tunnel-lora-0" \
  "1234 ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port root@38.255.28.19" \
  '1234 evil sleep 3' >"$group_commands"
kill() { printf '%s\n' "$*" >>"$KILL_LOG"; }
export KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group_commands"
export PEER_TEST_PGID_PID=1235 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH"
export PATH
set +e
peer_kill_owned 1235 \
  "while true; do ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port root@38.255.28.19; sleep 3; done # peer-tunnel-lora-0"
peer_current_status=$?
set -e
test "$peer_current_status" -ne 0
test ! -s "$kill_log"
unset -f kill
unset KILL_LOG PEER_GROUP_COMMANDS
unset PEER_TEST_PGID_PID PEER_TEST_PGID

# A wrong-target legacy supervisor is owned by its exact peer marker and
# forward shape, so its complete process group is stopped safely.
legacy_group_commands="$tmp/legacy-group-commands"
legacy_kill_log="$tmp/legacy-kill-log"
printf '%s\n' \
  "1234 bash -c while true; do ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port wrong@peer; sleep 3; done # peer-tunnel-lora-0" \
  "1234 ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:$relay_port -R 127.0.0.1:18101:127.0.0.1:$relay_port wrong@peer" \
  '1234 sleep 3' \
  >"$legacy_group_commands"
kill() { printf '%s\n' "$*" >>"$LEGACY_KILL_LOG"; }
export LEGACY_KILL_LOG="$legacy_kill_log" PEER_GROUP_COMMANDS="$legacy_group_commands"
export PEER_TEST_PGID_PID=1234 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" peer_tunnel_remove_legacy
test "$(grep -F -c -- '-1234' "$legacy_kill_log")" -eq 1
unset -f kill
unset LEGACY_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

# A stale peer group with a shell-like sleep member is not owned.
peer_stale_evil_group="$tmp/peer-stale-evil-group"
peer_stale_evil_kill_log="$tmp/peer-stale-evil-kill"
printf '%s\n' \
  "1234 $peer_supervisor_command" \
  "1234 $peer_exact_command" \
  '1234 evil sleep 3' >"$peer_stale_evil_group"
kill() { printf '%s\n' "$*" >>"$PEER_STALE_EVIL_KILL_LOG"; return 0; }
export PEER_STALE_EVIL_KILL_LOG="$peer_stale_evil_kill_log" PEER_GROUP_COMMANDS="$peer_stale_evil_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
set +e
PATH="$tmp/bin:$PATH" peer_kill_owned 42 "$peer_supervisor_command" cleanup peer-tunnel-lora-0
peer_stale_evil_status=$?
set -e
test "$peer_stale_evil_status" -ne 0
test ! -s "$peer_stale_evil_kill_log"
unset -f kill
unset PEER_STALE_EVIL_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

# The supervisor command escapes the wildcard for its shell, while its SSH
# child has the literal wildcard.  Only that exact timeout argument is
# normalized for group comparison.
peer_group_accepts_escaped_timeout() {
  local group="$tmp/peer-escaped-group" kill_log="$tmp/peer-escaped-kill"
  printf '%s\n' \
    "1234 $peer_escaped_supervisor_command" \
    "1234 $peer_exact_command" \
    '1234 sleep 3' >"$group"
  kill() { printf '%s\n' "$*" >>"$PEER_ESCAPED_KILL_LOG"; return 0; }
  export PEER_ESCAPED_KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group"
  export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
  PATH="$tmp/bin:$PATH" peer_kill_owned 42 "$peer_escaped_supervisor_command" \
    cleanup peer-tunnel-lora-0
  test "$(grep -F -c -- '-- -1234' "$kill_log")" -eq 1
  unset -f kill
  unset PEER_ESCAPED_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID
}

peer_group_rejects_child() {
  local child=$1 name=$2 group="$tmp/peer-$2-group" kill_log="$tmp/peer-$2-kill"
  printf '%s\n' \
    "1234 $peer_escaped_supervisor_command" \
    "1234 $child" \
    '1234 sleep 3' >"$group"
  kill() { printf '%s\n' "$*" >>"$PEER_REJECT_KILL_LOG"; return 0; }
  export PEER_REJECT_KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group"
  export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
  set +e
  PATH="$tmp/bin:$PATH" peer_kill_owned 42 "$peer_escaped_supervisor_command" \
    cleanup peer-tunnel-lora-0
  local status=$?
  set -e
  test "$status" -ne 0
  test ! -s "$kill_log"
  unset -f kill
  unset PEER_REJECT_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID
}

peer_group_accepts_escaped_timeout
peer_group_rejects_child "${peer_exact_command/ChannelTimeout=*-tcpip=45s/ChannelTimeout=*-tcpip=60s}" timeout
peer_group_rejects_child "$peer_exact_command extra" extra
peer_group_rejects_child "${peer_exact_command/root@38.255.28.19/wrong@peer}" target

# Group cleanup accepts only an exact owned China supervisor and child.
trainer_ssh_target=china-trainer
china_owned_group="$tmp/china-owned-group"
china_owned_kill_log="$tmp/china-owned-kill"
printf '%s\n' \
  "1234 $china_supervisor_command" \
  "1234 $china_exact_command" \
  '1234 sleep 3' >"$china_owned_group"
kill() { printf '%s\n' "$*" >>"$CHINA_OWNED_KILL_LOG"; return 0; }
export CHINA_OWNED_KILL_LOG="$china_owned_kill_log" PEER_GROUP_COMMANDS="$china_owned_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 1
test "$(grep -F -c -- '-- -1234' "$china_owned_kill_log")" -eq 1

china_unrelated_group="$tmp/china-unrelated-group"
: >"$china_owned_kill_log"
printf '%s\n' \
  "1234 $china_supervisor_command" \
  '1234 ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 unrelated-trainer' \
  >"$china_unrelated_group"
export PEER_GROUP_COMMANDS="$china_unrelated_group"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 1
china_unrelated_status=$?
set -e
test "$china_unrelated_status" -ne 0
test ! -s "$china_owned_kill_log"
unset -f kill
unset CHINA_OWNED_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

# Normal China cleanup also accepts only the exact standalone sleep argv.
china_normal_group="$tmp/china-normal-group"
china_normal_kill_log="$tmp/china-normal-kill"
printf '%s\n' \
  "1234 $china_supervisor_command" \
  "1234 $china_exact_command" \
  '1234 sleep 3' >"$china_normal_group"
kill() { printf '%s\n' "$*" >>"$CHINA_NORMAL_KILL_LOG"; return 0; }
export CHINA_NORMAL_KILL_LOG="$china_normal_kill_log" PEER_GROUP_COMMANDS="$china_normal_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 0
test "$(grep -F -c -- '-- -1234' "$china_normal_kill_log")" -eq 1

china_normal_evil_group="$tmp/china-normal-evil-group"
: >"$china_normal_kill_log"
printf '%s\n' \
  "1234 $china_supervisor_command" \
  "1234 $china_exact_command" \
  '1234 evil sleep 3' >"$china_normal_evil_group"
export PEER_GROUP_COMMANDS="$china_normal_evil_group"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 0
china_normal_evil_status=$?
set -e
test "$china_normal_evil_status" -ne 0
test ! -s "$china_normal_kill_log"
unset -f kill
unset CHINA_NORMAL_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

china_group_accepts_escaped_timeout() {
  local group="$tmp/china-escaped-group" kill_log="$tmp/china-escaped-kill"
  printf '%s\n' \
    "1234 $china_escaped_supervisor_command" \
    "1234 $china_exact_command" \
    '1234 sleep 3' >"$group"
  kill() { printf '%s\n' "$*" >>"$CHINA_ESCAPED_KILL_LOG"; return 0; }
  export CHINA_ESCAPED_KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group"
  export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
  PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_escaped_supervisor_command" \
    china-tunnel-lora-0 1
  test "$(grep -F -c -- '-- -1234' "$kill_log")" -eq 1
  unset -f kill
  unset CHINA_ESCAPED_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID
}

china_group_rejects_child() {
  local child=$1 name=$2 group="$tmp/china-$2-group" kill_log="$tmp/china-$2-kill"
  printf '%s\n' \
    "1234 $china_escaped_supervisor_command" \
    "1234 $child" \
    '1234 sleep 3' >"$group"
  kill() { printf '%s\n' "$*" >>"$CHINA_REJECT_KILL_LOG"; return 0; }
  export CHINA_REJECT_KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group"
  export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
  set +e
  PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_escaped_supervisor_command" \
    china-tunnel-lora-0 1
  local status=$?
  set -e
  test "$status" -ne 0
  test ! -s "$kill_log"
  unset -f kill
  unset CHINA_REJECT_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID
}

china_group_accepts_escaped_timeout
china_group_rejects_child "${china_exact_command/ChannelTimeout=*-tcpip=45s/ChannelTimeout=*-tcpip=60s}" timeout
china_group_rejects_child "$china_exact_command extra" extra
china_group_rejects_child "${china_exact_command/china-trainer/wrong-trainer}" target

# Captured ProxyJump groups accept only the route-derived transport child.
proxy_route="$tmp/proxy-route"
printf '%s\n' 'hostname 10.92.20.82' 'port 22' 'proxyjump china-jump' >"$proxy_route"
trainer_ssh_target=china-proxy
export SSH_G_OUTPUT="$proxy_route"
proxy_exact_command=${china_exact_command/china-trainer/china-proxy}
proxy_supervisor="bash -c while true; do $proxy_exact_command; sleep 3; done # china-tunnel-lora-0"
proxy_child='ssh -W [10.92.20.82]:22 china-jump'
proxy_group="$tmp/proxy-group"
proxy_kill_log="$tmp/proxy-kill"
printf '%s\n' "1234 $proxy_supervisor" "1234 $proxy_exact_command" \
  "1234 $proxy_child" '1234 sleep 3' >"$proxy_group"
kill() { printf '%s\n' "$*" >>"$PROXY_KILL_LOG"; return 0; }
export PROXY_KILL_LOG="$proxy_kill_log" PEER_GROUP_COMMANDS="$proxy_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$proxy_supervisor" \
  china-tunnel-lora-0 1
test "$(grep -F -c -- '-- -1234' "$proxy_kill_log")" -eq 1

china_proxy_rejects_child() {
  local child=$1 name=$2 group="$tmp/proxy-$2-group" kill_log="$tmp/proxy-$2-kill"
  printf '%s\n' "1234 $proxy_supervisor" "1234 $proxy_exact_command" \
    "1234 $child" '1234 sleep 3' >"$group"
  kill() { printf '%s\n' "$*" >>"$PROXY_REJECT_KILL_LOG"; return 0; }
  : >"$kill_log"
  export PROXY_REJECT_KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group"
  set +e
  PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$proxy_supervisor" \
    china-tunnel-lora-0 1
  local status=$?
  set -e
  test "$status" -ne 0
  test ! -s "$kill_log"
}
china_proxy_rejects_child 'ssh -W [10.92.20.83]:22 china-jump' wrong-host
china_proxy_rejects_child 'ssh -W [10.92.20.82]:22 china-jump extra' extra
china_proxy_rejects_child 'ssh -W [10.92.20.82]:23 china-jump' wrong-port
unset PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID SSH_G_OUTPUT

# The captured old profile is exact except for ChannelTimeout; altered options fail.
trainer_ssh_target=china-trainer
old_exact_command="$china_old_command_prefix -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer"
old_supervisor="bash -c while true; do $old_exact_command; sleep 3; done # china-tunnel-lora-0"
old_group="$tmp/old-group"
old_kill_log="$tmp/old-kill"
printf '%s\n' "1234 $old_supervisor" "1234 $old_exact_command" '1234 sleep 3' >"$old_group"
kill() { printf '%s\n' "$*" >>"$OLD_KILL_LOG"; return 0; }
export OLD_KILL_LOG="$old_kill_log" PEER_GROUP_COMMANDS="$old_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$old_supervisor" \
  china-tunnel-lora-0 1
test "$(grep -F -c -- '-- -1234' "$old_kill_log")" -eq 1
old_bad_command="${old_exact_command/ServerAliveCountMax=3/ServerAliveCountMax=4}"
old_bad_supervisor="${old_supervisor/$old_exact_command/$old_bad_command}"
printf '%s\n' "1234 $old_bad_supervisor" "1234 $old_bad_command" '1234 sleep 3' >"$old_group"
: >"$old_kill_log"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$old_bad_supervisor" \
  china-tunnel-lora-0 1
old_bad_status=$?
set -e
test "$old_bad_status" -ne 0
test ! -s "$old_kill_log"

# Reconcile also preflights all old candidates before it kills any group.
reconcile_log="$tmp/china-reconcile-log"
reconcile_group="$tmp/china-reconcile-group"
reconcile_kill_log="$tmp/china-reconcile-kill"
printf '%s\n' "$old_supervisor" >"$reconcile_log"
printf '%s\n' \
  "10000 $old_supervisor" \
  "10000 $old_exact_command" \
  '10000 sleep 3' \
  "10001 $old_supervisor" \
  "10001 $old_exact_command" \
  '10001 evil sleep 3' >"$reconcile_group"
kill() { printf '%s\n' "$*" >>"$RECONCILE_KILL_LOG"; return 0; }
printf '%s\n' "$old_supervisor" >"$tmp/reconcile-command"
export RECONCILE_KILL_LOG="$reconcile_kill_log" PEER_TEST_LOG="$reconcile_log" \
  PEER_DUPLICATE_MARKER=china-tunnel-lora-0 \
  PEER_DUPLICATE_COMMAND="$tmp/reconcile-command" PEER_GROUP_COMMANDS="$reconcile_group"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_reconcile_connection china-tunnel-lora-0 \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -L 127.0.0.1:18101:127.0.0.1:8100
reconcile_status=$?
set -e
test "$reconcile_status" -ne 0
test ! -s "$reconcile_kill_log"
unset -f kill
unset RECONCILE_KILL_LOG PEER_TEST_LOG PEER_DUPLICATE_MARKER \
  PEER_DUPLICATE_COMMAND PEER_GROUP_COMMANDS

# A listener-owning keeper is also rejected when its group has an unknown member.
keeper_log="$tmp/china-keeper-log"
keeper_group="$tmp/china-keeper-group"
keeper_ss="$tmp/china-keeper-ss"
keeper_kill_log="$tmp/china-keeper-kill"
printf '%s\n' "$china_supervisor_command" >"$keeper_log"
printf '%s\n' \
  "10000 $china_supervisor_command" \
  "10000 $china_exact_command" \
  '10000 evil sleep 3' >"$keeper_group"
printf '%s\n' \
  'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(('"ssh"',pid=4242,fd=5))' \
  'LISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(('"ssh"',pid=4242,fd=6))' >"$keeper_ss"
kill() { printf '%s\n' "$*" >>"$KEEPER_KILL_LOG"; return 0; }
printf '%s\n' "$china_supervisor_command" >"$tmp/keeper-command"
export KEEPER_KILL_LOG="$keeper_kill_log" PEER_TEST_LOG="$keeper_log" \
  PEER_DUPLICATE_MARKER=china-tunnel-lora-0 \
  PEER_DUPLICATE_COMMAND="$tmp/keeper-command" PEER_GROUP_COMMANDS="$keeper_group" \
  PEER_STALE_LISTENERS_ACTIVE=1 TUNNEL_SS_OUTPUT="$keeper_ss" \
  TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=10000 TUNNEL_OWNER_COMMAND="$tmp/keeper-command"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_reconcile_connection china-tunnel-lora-0 \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -L 127.0.0.1:18101:127.0.0.1:8100
keeper_status=$?
set -e
test "$keeper_status" -ne 0
test ! -s "$keeper_kill_log"
unset -f kill
unset KEEPER_KILL_LOG PEER_TEST_LOG PEER_DUPLICATE_MARKER \
  PEER_DUPLICATE_COMMAND PEER_GROUP_COMMANDS PEER_STALE_LISTENERS_ACTIVE \
  TUNNEL_SS_OUTPUT TUNNEL_OWNER_PID TUNNEL_OWNER_PPID TUNNEL_OWNER_COMMAND

# A supervisor-only or sleep-only transition is safe and has no unknown member.
sleep_group="$tmp/sleep-group"
printf '%s\n' "1234 $china_supervisor_command" '1234 sleep 3' >"$sleep_group"
kill() { printf '%s\n' "$*" >>"$SLEEP_KILL_LOG"; return 0; }
export SLEEP_KILL_LOG="$tmp/sleep-kill" PEER_GROUP_COMMANDS="$sleep_group"
: >"$SLEEP_KILL_LOG"
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 0
test "$(grep -F -c -- '-- -1234' "$SLEEP_KILL_LOG")" -eq 1
printf '%s\n' "1234 $china_supervisor_command" >"$sleep_group"
: >"$SLEEP_KILL_LOG"
PATH="$tmp/bin:$PATH" china_tunnel_stop_supervisor 42 "$china_supervisor_command" \
  china-tunnel-lora-0 0
test "$(grep -F -c -- '-- -1234' "$SLEEP_KILL_LOG")" -eq 1

# Group cleanup preflights every candidate before it kills any group.
two_candidate_group="$tmp/china-two-candidate-group"
two_candidate_kill_log="$tmp/china-two-candidate-kill"
printf '%s\n' \
  "10000 $china_supervisor_command" \
  "10000 $china_exact_command" \
  '10000 sleep 3' \
  "10001 $china_supervisor_command" \
  "10001 $china_exact_command" \
  '10001 evil sleep 3' >"$two_candidate_group"
kill() { printf '%s\n' "$*" >>"$TWO_CANDIDATE_KILL_LOG"; return 0; }
export TWO_CANDIDATE_KILL_LOG="$two_candidate_kill_log" PEER_GROUP_COMMANDS="$two_candidate_group"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_remove_all_owned
two_candidate_status=$?
set -e
test "$two_candidate_status" -ne 0
test ! -s "$two_candidate_kill_log"
unset -f kill
unset TWO_CANDIDATE_KILL_LOG PEER_GROUP_COMMANDS

# Group exit waits reject a live PGID and accept an absent PGID.
wait_group="$tmp/wait-group"
printf '%s\n' '1234 sleep 3' >"$wait_group"
export PEER_GROUP_COMMANDS="$wait_group"
sleep() { :; }
set +e
PATH="$tmp/bin:$PATH" china_tunnel_wait_group_exit 1234
wait_live_status=$?
set -e
test "$wait_live_status" -ne 0
: >"$wait_group"
PATH="$tmp/bin:$PATH" china_tunnel_wait_group_exit 1234
unset -f sleep
unset PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

# Readiness rejects two exact same-marker supervisors even when one owns both listeners.
duplicate_command="$tmp/duplicate-command"
printf '%s\n' "$china_supervisor_command" >"$duplicate_command"
duplicate_log="$tmp/duplicate-log"
printf '%s\n' "$china_supervisor_command" "$china_supervisor_command" >"$duplicate_log"
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=30000,fd=5))\n' >"$tmp/duplicate-ss"
printf 'LISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(("ssh",pid=30001,fd=6))\n' >>"$tmp/duplicate-ss"
lora_source_ports=(18100 18101)
export PEER_TEST_LOG="$duplicate_log" PEER_DUPLICATE_MARKER=china-tunnel-lora-0 \
  PEER_DUPLICATE_COMMAND="$duplicate_command"
export TUNNEL_OWNER_PPID=10000 TUNNEL_SS_OUTPUT="$tmp/duplicate-ss" PEER_STALE_LISTENERS_ACTIVE=1
set +e
PATH="$tmp/bin:$PATH" china_tunnel_supervisors_ready
duplicate_ready_status=$?
set -e
test "$duplicate_ready_status" -ne 0
unset PEER_TEST_LOG PEER_DUPLICATE_MARKER PEER_DUPLICATE_COMMAND TUNNEL_OWNER_PPID \
  TUNNEL_SS_OUTPUT PEER_STALE_LISTENERS_ACTIVE

# Readiness rejects one lane plus an otherwise valid out-of-range marker.
extra_marker_log="$tmp/extra-marker-log"
extra_marker_command="${china_supervisor_command%# china-tunnel-lora-0}# china-tunnel-lora-64"
printf '%s\n' "$china_supervisor_command" "$extra_marker_command" >"$extra_marker_log"
lora_source_ports=(18100 18101)
export PEER_TEST_LOG="$extra_marker_log"
set +e
PATH="$tmp/bin:$PATH" china_tunnel_supervisors_ready
extra_marker_status=$?
set -e
test "$extra_marker_status" -ne 0
unset PEER_TEST_LOG

# Peer readiness rejects an otherwise valid out-of-range lane marker.
peer_extra_log="$tmp/peer-extra-marker-log"
peer_extra_command="${peer_supervisor_command%# peer-tunnel-lora-0}# peer-tunnel-lora-64"
printf '%s\n' "$peer_supervisor_command" "$peer_extra_command" >"$peer_extra_log"
export PEER_TEST_LOG="$peer_extra_log" PEER_QUERY="$query"
set +e
PATH="$tmp/bin:$PATH" peer_tunnel_supervisors_ready
peer_extra_status=$?
set -e
test "$peer_extra_status" -ne 0
unset PEER_TEST_LOG PEER_QUERY

# A peer supervisor added after the first inventory is rejected by the final pass.
peer_race_first="$tmp/peer-race-first"
peer_race_second="$tmp/peer-race-second"
peer_race_counter="$tmp/peer-race-counter"
printf '%s\n' "$peer_supervisor_command" >"$peer_race_first"
printf '%s\n' "$peer_supervisor_command" "$peer_extra_command" >"$peer_race_second"
printf '0\n' >"$peer_race_counter"
export PEER_TEST_LOG="$peer_race_first" PEER_QUERY="$query" \
  PEER_INVENTORY_COUNTER="$peer_race_counter" PEER_INVENTORY_SECOND="$peer_race_second"
set +e
PATH="$tmp/bin:$PATH" peer_tunnel_supervisors_ready
peer_race_status=$?
set -e
test "$peer_race_status" -ne 0
test "$(<"$peer_race_counter")" -ge 2
unset PEER_TEST_LOG PEER_QUERY PEER_INVENTORY_COUNTER PEER_INVENTORY_SECOND
lora_source_ports=("${configured_lora_ports[@]}")

# Both source listeners must belong to one SSH child PID, not two split children.
shared_owner="$tmp/shared-owner"
printf '%s\n' "$china_exact_command" >"$shared_owner"
shared_ss="$tmp/shared-ss"
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n' >"$shared_ss"
printf 'LISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(("ssh",pid=4242,fd=6))\n' >>"$shared_ss"
export TUNNEL_SS_OUTPUT="$shared_ss" PEER_STALE_LISTENERS_ACTIVE=1 \
  TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241 TUNNEL_OWNER_COMMAND="$shared_owner"
PATH="$tmp/bin:$PATH" china_tunnel_supervisor_owns_forwards 4241 \
  "$china_supervisor_command" current \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -L 127.0.0.1:18101:127.0.0.1:8100
printf 'LISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(("ssh",pid=4243,fd=6))\n' >"$shared_ss"
export TUNNEL_EXTRA_OWNER_PID=4243
set +e
PATH="$tmp/bin:$PATH" china_tunnel_supervisor_owns_forwards 4241 \
  "$china_supervisor_command" current \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -L 127.0.0.1:18101:127.0.0.1:8100
split_owner_status=$?
set -e
test "$split_owner_status" -ne 0
unset TUNNEL_SS_OUTPUT PEER_STALE_LISTENERS_ACTIVE TUNNEL_OWNER_PID \
  TUNNEL_EXTRA_OWNER_PID TUNNEL_OWNER_PPID TUNNEL_OWNER_COMMAND

# Readiness accepts a listener only when its SSH child and two-port
# supervisor both match the configured target.
trainer_ssh_target=china-trainer
china_forwards=(-L "127.0.0.1:18100:127.0.0.1:8100" -L "127.0.0.1:18101:127.0.0.1:8100")
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer' \
  >"$tmp/tunnel-owner"
printf '%s\n' \
  'bash -c while true; do ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-0' \
  >"$tmp/tunnel-supervisor"
: >"$tmp/tunnel-ss"
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n' >>"$tmp/tunnel-ss"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241 TUNNEL_SUPERVISOR_PID=4241 \
  TUNNEL_OWNER_COMMAND="$tmp/tunnel-owner" TUNNEL_SUPERVISOR_COMMAND="$tmp/tunnel-supervisor" \
  TUNNEL_SS_OUTPUT="$tmp/tunnel-ss"
PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 2 "${china_forwards[@]}"

# A single-port supervisor is rejected when its SSH child uses the wrong target.
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 wrong-trainer' \
  >"$tmp/tunnel-owner"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241
if PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 2 "${china_forwards[@]}"; then
  echo "wrong child target was accepted" >&2
  exit 1
fi

# An orphan listener is rejected without readiness-time cleanup.
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer' \
  >"$tmp/tunnel-owner"
orphan_kill_log="$tmp/orphan-kill"
kill() { printf '%s\n' "$*" >"$orphan_kill_log"; return 0; }
export TUNNEL_OWNER_PPID=1
if PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 2 \
    "${china_forwards[@]}"; then
  echo "orphan listener was accepted" >&2
  exit 1
else
  test "$?" -eq 1
fi
test ! -s "$orphan_kill_log"
unset -f kill
unset TUNNEL_OWNER_PID TUNNEL_OWNER_PPID TUNNEL_SUPERVISOR_PID \
  TUNNEL_OWNER_COMMAND TUNNEL_SUPERVISOR_COMMAND TUNNEL_SS_OUTPUT

# The secondary judge uses the supervised generic tunnel with a separate SSH
# identity and port.  Its child, supervisor, and listener checks are exact.
secondary_target=root@202.214.223.66
secondary_forward='127.0.0.1:26407:127.0.0.1:8000'
secondary_options=(-o IdentitiesOnly=yes -i /data/trinity/h200_rollout_ed25519 -p 30599)
secondary_owner="ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no ${secondary_options[*]} -L $secondary_forward $secondary_target"
secondary_supervisor="bash -c while true; do $secondary_owner; sleep 3; done # zmq-tunnel-26407"
generic_tunnel_ssh_match "$secondary_owner" "$secondary_target" "$secondary_forward" no \
  "${secondary_options[@]}"
generic_tunnel_supervisor_match "$secondary_supervisor" zmq-tunnel-26407 \
  "$secondary_target" "$secondary_forward" no "${secondary_options[@]}"
for hostile_target in \
  "root@202.214.223.66.evil" \
  "" \
  "root@202.214.223.66 extra"; do
  hostile_generic_supervisor=${secondary_supervisor/root@202.214.223.66/$hostile_target}
  if generic_tunnel_supervisor_match "$hostile_generic_supervisor" zmq-tunnel-26407 \
      "$secondary_target" "$secondary_forward" no "${secondary_options[@]}"; then
    echo "generic supervisor accepted hostile target token: $hostile_target" >&2
    exit 1
  fi
done
for hostile_generic_supervisor in \
  "${secondary_supervisor/; sleep 3; done/; evil; sleep 3; done}" \
  "${secondary_supervisor/; sleep 3; done/; sleep 3; evil; done}" \
  "${secondary_supervisor/#*# zmq-tunnel-26407/*# zmq-tunnel-26407 extra}" \
  "${secondary_supervisor/while true; do/while false; do}" \
  "${secondary_supervisor/while true; do/evil bash -c while true; do}"; do
  if generic_tunnel_supervisor_match "$hostile_generic_supervisor" zmq-tunnel-26407 \
      "$secondary_target" "$secondary_forward" no "${secondary_options[@]}"; then
    echo "generic supervisor accepted injected shell syntax" >&2
    exit 1
  fi
done

# Generic stale cleanup requires every member of the exact process group.
generic_owned_group="$tmp/generic-owned-group"
generic_owned_kill_log="$tmp/generic-owned-kill"
printf '%s\n' \
  "1234 $secondary_supervisor" \
  "1234 $secondary_owner" \
  '1234 sleep 3' >"$generic_owned_group"
kill() { printf '%s\n' "$*" >>"$GENERIC_OWNED_KILL_LOG"; return 0; }
export GENERIC_OWNED_KILL_LOG="$generic_owned_kill_log" PEER_GROUP_COMMANDS="$generic_owned_group"
export PEER_TEST_PGID_PID=42 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH" generic_tunnel_stop_stale 42 "$secondary_supervisor" \
  zmq-tunnel-26407 "$secondary_forward"
test "$(grep -F -c -- '-- -1234' "$generic_owned_kill_log")" -eq 1

generic_unrelated_group="$tmp/generic-unrelated-group"
: >"$generic_owned_kill_log"
printf '%s\n' \
  "1234 $secondary_supervisor" \
  '1234 ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:26407:127.0.0.1:8000 unrelated-target' \
  >"$generic_unrelated_group"
export PEER_GROUP_COMMANDS="$generic_unrelated_group"
set +e
PATH="$tmp/bin:$PATH" generic_tunnel_stop_stale 42 "$secondary_supervisor" \
  zmq-tunnel-26407 "$secondary_forward"
generic_unrelated_status=$?
set -e
test "$generic_unrelated_status" -ne 0
test ! -s "$generic_owned_kill_log"
unset -f kill
unset GENERIC_OWNED_KILL_LOG PEER_GROUP_COMMANDS PEER_TEST_PGID_PID PEER_TEST_PGID

if generic_tunnel_ssh_match "${secondary_owner/30599/30598}" "$secondary_target" \
    "$secondary_forward" no "${secondary_options[@]}"; then
  echo "generic tunnel accepted the wrong SSH port" >&2
  exit 1
fi
printf '%s\n' "$secondary_owner" >"$tmp/tunnel-owner"
printf '%s\n' "$secondary_supervisor" >"$tmp/tunnel-supervisor"
printf 'LISTEN 0 128 127.0.0.1:26407 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n' \
  >"$tmp/tunnel-ss"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241 TUNNEL_SUPERVISOR_PID=4241 \
  TUNNEL_OWNER_COMMAND="$tmp/tunnel-owner" TUNNEL_SUPERVISOR_COMMAND="$tmp/tunnel-supervisor" \
  TUNNEL_SS_OUTPUT="$tmp/tunnel-ss" PEER_STALE_LISTENERS_ACTIVE=1
timeout() { return 0; }
PATH="$tmp/bin:$PATH" generic_tunnel_listener_ready 26407 zmq-tunnel-26407 \
  "$secondary_target" "$secondary_forward" no "${secondary_options[@]}"
unset -f timeout
unset TUNNEL_OWNER_PID TUNNEL_OWNER_PPID TUNNEL_SUPERVISOR_PID TUNNEL_OWNER_COMMAND \
  TUNNEL_SUPERVISOR_COMMAND TUNNEL_SS_OUTPUT PEER_STALE_LISTENERS_ACTIVE

# Execute the normal-path launcher block with a failing reconciliation.  The
# service-start marker must not be reached, while tunnels-only keeps its
# existing non-fatal handoff to the readiness loop.
secondary_launch_block=$(awk '
  /^if \[ "\$secondary_enabled" = 1 \]; then$/ {
    matches++
    if (matches == 1) {
      active=1
      depth=1
      print
      next
    }
  }
  active {
    print
    if ($0 ~ /^[[:space:]]*if[[:space:]]/) depth++
    if ($0 ~ /^[[:space:]]*fi[[:space:]]*$/) {
      depth--
      if (depth == 0) exit
    }
  }
' "$script")
test -n "$secondary_launch_block"
if output=$(bash -c '
  set -u
  secondary_enabled=1
  tunnels_only=0
  secondary_local_api_port=26407
  secondary_remote_api_port=8000
  secondary_ssh_target=root@202.214.223.66
  secondary_ssh_options=(-o IdentitiesOnly=yes -i /data/trinity/h200_rollout_ed25519 -p 30599)
  repair_fail() { echo "adapter tunnel reconciliation failed: $*" >&2; exit 1; }
  tunnel() { return 1; }
'"$secondary_launch_block"'
  printf "%s\n" service-started
' 2>&1); then
  echo "normal secondary tunnel failure was ignored" >&2
  exit 1
fi
if grep -Fq service-started <<<"$output" || ! grep -Fq 'secondary judge tunnel failed' <<<"$output"; then
  echo "normal secondary tunnel failure reached the service path" >&2
  exit 1
fi
if tunnels_only_output=$(bash -c '
  set -u
  secondary_enabled=1
  tunnels_only=1
  secondary_local_api_port=26407
  secondary_remote_api_port=8000
  secondary_ssh_target=root@202.214.223.66
  secondary_ssh_options=(-o IdentitiesOnly=yes -i /data/trinity/h200_rollout_ed25519 -p 30599)
  repair_fail() { echo "adapter tunnel reconciliation failed: $*" >&2; exit 1; }
  tunnel() { return 1; }
'"$secondary_launch_block"'
  printf "%s\n" tunnels-only-continued
'); then
  echo "tunnels-only secondary failure was ignored" >&2
  exit 1
fi
if grep -Fq tunnels-only-continued <<<"$tunnels_only_output"; then
  echo "tunnels-only secondary failure reached the readiness path" >&2
  exit 1
fi

cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ -n "${PEER_REMOTE_SLEEP:-}" ]; then
  sleep "$PEER_REMOTE_SLEEP"
fi
if [ "${PEER_REMOTE_FAIL:-0}" = 1 ]; then
  exit 1
fi
if [ "${PEER_REMOTE_MISSING:-0}" = 1 ]; then
  printf '%s\n' "$((TEST_SOURCE_COUNT - 1))"
else
  printf '%s\n' "${TEST_SOURCE_COUNT:?}"
fi
EOF
chmod +x "$tmp/bin/ssh"
PATH="$tmp/bin:$PATH" PEER_REMOTE_FAIL=0 PEER_REMOTE_MISSING=0 peer_remote_ports_ready
if PATH="$tmp/bin:$PATH" PEER_REMOTE_FAIL=0 PEER_REMOTE_MISSING=1 peer_remote_ports_ready; then
  echo "remote missing reverse port was accepted" >&2
  exit 1
fi
if PATH="$tmp/bin:$PATH" PEER_REMOTE_FAIL=1 PEER_REMOTE_MISSING=0 peer_remote_ports_ready; then
  echo "remote SSH failure was accepted" >&2
  exit 1
fi
export PEER_REMOTE_SLEEP=20
started=$(date +%s)
if PATH="$tmp/bin:$PATH" peer_remote_ports_ready 1; then
  echo "hung remote probe was accepted" >&2
  exit 1
fi
elapsed=$(( $(date +%s) - started ))
printf 'hung remote probe wall_seconds=%s\n' "$elapsed"
test "$elapsed" -le 3
unset PEER_REMOTE_SLEEP

# A failed probe receives the remaining absolute repair deadline, not its full
# requested timeout. This keeps one failed remote check inside the outer SSH
# timeout and prevents nested retry windows from multiplying.
tunnels_only=1
tunnel_repair_deadline=$((SECONDS + 2))
export PEER_REMOTE_SLEEP=20
started=$SECONDS
if PATH="$tmp/bin:$PATH" peer_remote_ports_ready 10; then
  echo "deadline-limited remote probe was accepted" >&2
  exit 1
fi
elapsed=$((SECONDS - started))
printf 'deadline-limited remote probe elapsed_seconds=%s\n' "$elapsed"
test "$elapsed" -le 3
unset PEER_REMOTE_SLEEP
printf 'peer tunnel uses one API connection and sixty-four bounded two-port lanes\n'
