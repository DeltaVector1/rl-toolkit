#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="$root/training_ops/runtime/h200/start_orchestrator.sh"
relay_block=$(sed -n '/sudo docker run -d --name/,/--canonical-source-url/p' "$script")
[[ "$relay_block" == *"--user 0:0"* && "$relay_block" == *'-v "$relay_cache_dir":"$relay_cache_dir":ro'* ]] || {
  echo "relay must run as root with a read-only adapter cache" >&2
  exit 1
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
real_cat=$(command -v cat)
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
    china-tunnel-lora-*)
      echo $((10000 + ${pattern##*-}))
      ;;
    '-L 127.0.0.1:'*)
      port=${pattern#-L 127.0.0.1:}
      port=${port%%:*}
      if [ "${PEER_STALE_GROUP_ACTIVE:-0}" = 1 ]; then
        echo 10000
        exit 0
      fi
      if [ "${PEER_CHINA_MODE:-0}" = 1 ] && [ "$port" -ge 18100 ] && [ "$port" -le 18195 ]; then
        echo $((10000 + (port - 18100)))
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
  elif [ "${TUNNEL_SUPERVISOR_PID:-}" = "$pid" ]; then
    cat "${TUNNEL_SUPERVISOR_COMMAND:?}"
  elif [ -n "${CHINA_PROC_FIXTURES:-}" ] && [ -s "$CHINA_PROC_FIXTURES/$pid" ]; then
    "$REAL_CAT" "$CHINA_PROC_FIXTURES/$pid" | tr '\0' ' '
  elif [ "$pid" -ge 10000 ] 2>/dev/null; then
    if [ "${PEER_STALE_GROUP_ACTIVE:-0}" = 1 ] && [ -r "${PEER_STALE_GROUP_COMMAND:-}" ]; then
      cat "$PEER_STALE_GROUP_COMMAND"
      exit 0
    fi
    lane=$((pid - 10000))
    grep -E "done # china-tunnel-lora-${lane}[[:space:]]*$" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  else
    query=$(<"${PEER_QUERY:?}")
    grep -F -- "$query" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  fi
elif [ "${1:-}" = "-o" ] && [ "${2:-}" = "ppid=" ]; then
  pid=${4:?missing process id}
  if [ "${TUNNEL_OWNER_PID:-}" = "$pid" ]; then
    printf '%s\n' "${TUNNEL_OWNER_PPID:?}"
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
  [ -n "${PEER_GROUP_COMMANDS:-}" ] && cat "$PEER_GROUP_COMMANDS"
fi
EOF
cat >"$tmp/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ -n "${TUNNEL_SS_OUTPUT:-}" ] && [ "${PEER_STALE_LISTENERS_ACTIVE:-}" != 0 ]; then
  cat "${TUNNEL_SS_OUTPUT:?missing tunnel ss output}"
fi
EOF
chmod +x "$tmp/bin/cat" "$tmp/bin/pgrep" "$tmp/bin/setsid"
chmod +x "$tmp/bin/ps" "$tmp/bin/ss"
export REAL_CAT="$real_cat"

start=$(grep -n '^generic_tunnel_ssh_match() {' "$script" | cut -d: -f1)
end=$(grep -n '^china_legacy_cleanup_required=0$' "$script" | cut -d: -f1)
source <(sed -n "${start},$((end - 1))p" "$script")

lora_source_ports=()
for port in $(seq 18100 18195); do
  lora_source_ports+=("$port")
done
peer_enabled=1
peer_api_port=18049
relay_port=18196
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
  [ -f "$log" ] && [ "$(wc -l <"$log")" -eq 49 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 49

api_pattern='-L 127.0.0.1:18049:127.0.0.1:8000'
test "$(grep -F -c -- "$api_pattern" "$log")" -eq 1
test "$(grep -F -c -- '-R 127.0.0.1:' "$log")" -eq 48
while IFS= read -r line; do
  case "$line" in
    *"$api_pattern"*"-R 127.0.0.1:"*)
      echo "API and reverse forwards share an SSH connection" >&2
      exit 1
      ;;
  esac
done <"$log"
for port in $(seq 18100 18195); do
  test "$(grep -F -c -- "-R 127.0.0.1:$port:127.0.0.1:$relay_port" "$log")" -eq 1
done
while IFS= read -r line; do
  reverse_count=$(grep -o -- '-R 127.0.0.1:' <<<"$line" | wc -l)
  test "$reverse_count" -eq 2
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done < <(grep -F -- '-R 127.0.0.1:' "$log")

# A second launcher invocation finds the exact API and lane signatures.
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
test "$(wc -l <"$log")" -eq 49

china_log="$tmp/china-tunnel.log"
trainer_ssh_target=china-trainer
adapter_port=8100
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
      "${china_lane_forwards[@]}"
    china_lane_start=$((china_lane_start + china_lane_size))
    china_lane_number=$((china_lane_number + 1))
  done
}

# A migration launch removes an old four-port China supervisor before
# creating forty-eight two-forward supervisors.
stale_group_command="$tmp/stale-group-command"
stale_group_kill_log="$tmp/stale-group-kill"
kill() {
  printf '%s\n' "$*" >>"$stale_group_kill_log"
  PEER_STALE_GROUP_ACTIVE=0
  export PEER_STALE_GROUP_ACTIVE
  sed -i '/# china-tunnel-lora-1$/d' "$china_log"
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
printf '%s\n' 'while true; do ssh -NT -L 127.0.0.1:18100:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-0' >"$stale_group_command"
printf '%s\n' "$(<"$stale_group_command")" >"$china_log"
export PEER_STALE_GROUP_ACTIVE=1 PEER_STALE_GROUP_COMMAND="$stale_group_command"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
test "${PEER_STALE_GROUP_ACTIVE:-1}" -eq 0
test -s "$stale_group_kill_log"
unset PEER_STALE_GROUP_ACTIVE PEER_STALE_GROUP_COMMAND
: >"$china_log"

printf '%s\n' 'while true; do ssh -NT -L 127.0.0.1:18104:127.0.0.1:8100 -L 127.0.0.1:18105:127.0.0.1:8100 -L 127.0.0.1:18106:127.0.0.1:8100 -L 127.0.0.1:18107:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-1' >"$stale_group_command"
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
  -L 127.0.0.1:18196:127.0.0.1:8100 \
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
for port in $(seq 18100 18195); do
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

# A fresh launch creates forty-eight two-forward lanes.
run_china_lanes
for _ in $(seq 1 50); do
  [ -f "$china_log" ] && [ "$(wc -l <"$china_log")" -eq 48 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 48
test "$(awk '/18104:127.0.0.1:8100/ && /18105:127.0.0.1:8100/ {count++} END {print count + 0}' "$china_log")" -eq 1
while IFS= read -r line; do
  forward_count=$(grep -o -- '-L 127.0.0.1:' <<<"$line" | wc -l)
  test "$forward_count" -eq 2
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done <"$china_log"
# A repeat launch preserves all matching groups without adding processes.
run_china_lanes
test "$(wc -l <"$china_log")" -eq 48
# A target change removes only the owned old groups before regrouping.
trainer_ssh_target=china-trainer-regrouped
run_china_lanes
for _ in $(seq 1 50); do
  [ "$(wc -l <"$china_log")" -eq 96 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 96

# A same-signature process with the wrong peer target is not reused.
printf '%s\n' 'while true; do ssh -R 127.0.0.1:18100:127.0.0.1:18196 wrong@peer; sleep 3; done # peer-tunnel-lora-0' >"$log"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel_connection lora-0 \
  '-R 127.0.0.1:18100:127.0.0.1:18196' \
  -R 127.0.0.1:18100:127.0.0.1:18196 \
  -R 127.0.0.1:18101:127.0.0.1:18196 \
  -R 127.0.0.1:18102:127.0.0.1:18196 \
  -R 127.0.0.1:18103:127.0.0.1:18196
for _ in $(seq 1 50); do
  [ "$(wc -l <"$log")" -eq 2 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 2
grep -Fq -- 'root@38.255.28.19' "$log"

# A process missing a lane forward is not reused.
printf '%s\n' 'while true; do ssh -R 127.0.0.1:18100:127.0.0.1:18196 root@38.255.28.19; sleep 3; done # peer-tunnel-lora-0' >"$log"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel_connection lora-0 \
  '-R 127.0.0.1:18100:127.0.0.1:18196' \
  -R 127.0.0.1:18100:127.0.0.1:18196 \
  -R 127.0.0.1:18101:127.0.0.1:18196 \
  -R 127.0.0.1:18102:127.0.0.1:18196 \
  -R 127.0.0.1:18103:127.0.0.1:18196
for _ in $(seq 1 50); do
  [ "$(wc -l <"$log")" -eq 2 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 2
test "$(grep -F -c -- '-R 127.0.0.1:18103:127.0.0.1:18196' "$log")" -eq 1

# A peer marker does not authorize killing an unrelated process in its group.
group_commands="$tmp/group-commands"
kill_log="$tmp/kill-log"
printf '%s\n' \
  '1234 bash -c while true; do ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:18196 -R 127.0.0.1:18101:127.0.0.1:18196 root@38.255.28.19; done # peer-tunnel-lora-0' \
  '1234 ssh -i /other/key -p 20099 -R 127.0.0.1:18100:127.0.0.1:18196 unrelated@peer' >"$group_commands"
kill() { printf '%s\n' "$*" >>"$KILL_LOG"; }
export KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group_commands"
export PEER_TEST_PGID_PID=1235 PEER_TEST_PGID=1234
PATH="$tmp/bin:$PATH"
export PATH
peer_kill_owned 1235 \
  'while true; do ssh -NT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -i /data/trinity/h200_rollout_ed25519 -p 20099 -R 127.0.0.1:18100:127.0.0.1:18196 -R 127.0.0.1:18101:127.0.0.1:18196 root@38.255.28.19; done # peer-tunnel-lora-0'
test "$(grep -F -c -- '1235' "$kill_log")" -eq 1
if grep -Fq -- '-1235' "$kill_log"; then
  echo "unrelated peer group received a group kill" >&2
  exit 1
fi
unset -f kill
unset KILL_LOG PEER_GROUP_COMMANDS
unset PEER_TEST_PGID_PID PEER_TEST_PGID

# Readiness accepts a listener only when its SSH child and two-port
# supervisor both match the configured target.
trainer_ssh_target=china-trainer
china_forwards=(-L "127.0.0.1:18100:127.0.0.1:8100" -L "127.0.0.1:18101:127.0.0.1:8100")
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer' \
  >"$tmp/tunnel-owner"
printf '%s\n' \
  'bash -c while true; do ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-0' \
  >"$tmp/tunnel-supervisor"
: >"$tmp/tunnel-ss"
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n' >>"$tmp/tunnel-ss"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241 TUNNEL_SUPERVISOR_PID=4241 \
  TUNNEL_OWNER_COMMAND="$tmp/tunnel-owner" TUNNEL_SUPERVISOR_COMMAND="$tmp/tunnel-supervisor" \
  TUNNEL_SS_OUTPUT="$tmp/tunnel-ss"
PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 2 "${china_forwards[@]}"

# A single-port supervisor is rejected when its SSH child uses the wrong target.
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 wrong-trainer' \
  >"$tmp/tunnel-owner"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241
if PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 2 "${china_forwards[@]}"; then
  echo "wrong child target was accepted" >&2
  exit 1
fi

# An orphan listener is rejected without readiness-time cleanup.
printf '%s\n' \
  'ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer' \
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
tunnels_only_output=$(bash -c '
  set -u
  secondary_enabled=1
  tunnels_only=1
  secondary_local_api_port=26407
  secondary_remote_api_port=8000
  secondary_ssh_target=root@202.214.223.66
  secondary_ssh_options=(-o IdentitiesOnly=yes -i /data/trinity/h200_rollout_ed25519 -p 30599)
  tunnel() { return 1; }
'"$secondary_launch_block"'
  printf "%s\n" tunnels-only-continued
')
grep -Fq tunnels-only-continued <<<"$tunnels_only_output"

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
  printf '95\n'
else
  printf '96\n'
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
printf 'peer tunnel uses one API connection and forty-eight bounded two-port lanes\n'
