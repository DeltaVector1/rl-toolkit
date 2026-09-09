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
  echo "${4:?missing process id}"
elif [ "${1:-}" = "-eo" ]; then
  [ -n "${PEER_GROUP_COMMANDS:-}" ] && cat "$PEER_GROUP_COMMANDS"
fi
EOF
cat >"$tmp/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cat "${TUNNEL_SS_OUTPUT:?missing tunnel ss output}"
EOF
chmod +x "$tmp/bin/pgrep" "$tmp/bin/setsid"
chmod +x "$tmp/bin/ps" "$tmp/bin/ss"

start=$(grep -n '^tunnel() {' "$script" | cut -d: -f1)
end=$(grep -n '^china_tunnel_remove_legacy$' "$script" | cut -d: -f1)
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
log="$tmp/peer-tunnel.log"
query="$tmp/peer-query"

PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
for _ in $(seq 1 50); do
  [ -f "$log" ] && [ "$(wc -l <"$log")" -eq 25 ] && break
  sleep 0.1
done
test "$(wc -l <"$log")" -eq 25

api_pattern='-L 127.0.0.1:18049:127.0.0.1:8000'
test "$(grep -F -c -- "$api_pattern" "$log")" -eq 1
test "$(grep -F -c -- '-R 127.0.0.1:' "$log")" -eq 24
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
  test "$reverse_count" -eq 4
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done < <(grep -F -- '-R 127.0.0.1:' "$log")

# A second launcher invocation finds the exact API and lane signatures.
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
test "$(wc -l <"$log")" -eq 25

china_log="$tmp/china-tunnel.log"
trainer_ssh_target=china-trainer
adapter_port=8100
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
china_lane_size=1
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
# creating ninety-six one-port supervisors.
stale_group_command="$tmp/stale-group-command"
printf '%s\n' 'while true; do ssh -NT -L 127.0.0.1:18104:127.0.0.1:8100 -L 127.0.0.1:18105:127.0.0.1:8100 -L 127.0.0.1:18106:127.0.0.1:8100 -L 127.0.0.1:18107:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-1' >"$stale_group_command"
printf '%s\n' "$(<"$stale_group_command")" >"$china_log"
stale_group_kill_log="$tmp/stale-group-kill"
kill() {
  printf '%s\n' "$*" >>"$stale_group_kill_log"
  PEER_STALE_GROUP_ACTIVE=0
  export PEER_STALE_GROUP_ACTIVE
  sed -i '/# china-tunnel-lora-1$/d' "$china_log"
}
export PEER_STALE_GROUP_ACTIVE=1 PEER_STALE_GROUP_COMMAND="$stale_group_command"
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
test "${PEER_STALE_GROUP_ACTIVE:-1}" -eq 0
test -s "$stale_group_kill_log"
unset PEER_STALE_GROUP_ACTIVE PEER_STALE_GROUP_COMMAND
unset -f kill

# A fresh launch creates ninety-six one-forward lanes.
run_china_lanes
for _ in $(seq 1 50); do
  [ -f "$china_log" ] && [ "$(wc -l <"$china_log")" -eq 96 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 96
test "$(awk '/18104:127.0.0.1:8100/ && /18107:127.0.0.1:8100/ {count++} END {print count + 0}' "$china_log")" -eq 0
while IFS= read -r line; do
  forward_count=$(grep -o -- '-L 127.0.0.1:' <<<"$line" | wc -l)
  test "$forward_count" -eq 1
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done <"$china_log"
# A repeat launch preserves all matching groups without adding processes.
run_china_lanes
test "$(wc -l <"$china_log")" -eq 96
# A target change removes only the owned old groups before regrouping.
trainer_ssh_target=china-trainer-regrouped
run_china_lanes
for _ in $(seq 1 50); do
  [ "$(wc -l <"$china_log")" -eq 192 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 192

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
  '1234 bash -c while true; do ssh -R 127.0.0.1:18100:127.0.0.1:18196 root@38.255.28.19; done # peer-tunnel-lora-0' \
  '1234 ssh -i /other/key -p 20099 -R 127.0.0.1:18100:127.0.0.1:18196 unrelated@peer' >"$group_commands"
kill() { printf '%s\n' "$*" >>"$KILL_LOG"; }
export KILL_LOG="$kill_log" PEER_GROUP_COMMANDS="$group_commands"
PATH="$tmp/bin:$PATH"
export PATH
peer_kill_owned 1234 \
  'while true; do ssh -R 127.0.0.1:18100:127.0.0.1:18196 root@38.255.28.19; done # peer-tunnel-lora-0'
test "$(grep -F -c -- '1234' "$kill_log")" -eq 1
if grep -Fq -- '-1234' "$kill_log"; then
  echo "unrelated peer group received a group kill" >&2
  exit 1
fi
unset -f kill
unset KILL_LOG PEER_GROUP_COMMANDS

# Readiness accepts a listener only when its SSH child and single-port
# supervisor both match the configured target.
china_forwards=(-L "127.0.0.1:18100:127.0.0.1:8100")
printf '%s\n' \
  'ssh -NT -L 127.0.0.1:18100:127.0.0.1:8100 china-trainer' \
  >"$tmp/tunnel-owner"
printf '%s\n' \
  'bash -c while true; do ssh -NT -L 127.0.0.1:18100:127.0.0.1:8100 china-trainer; sleep 3; done # china-tunnel-lora-0' \
  >"$tmp/tunnel-supervisor"
: >"$tmp/tunnel-ss"
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n' >>"$tmp/tunnel-ss"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241 TUNNEL_SUPERVISOR_PID=4241 \
  TUNNEL_OWNER_COMMAND="$tmp/tunnel-owner" TUNNEL_SUPERVISOR_COMMAND="$tmp/tunnel-supervisor" \
  TUNNEL_SS_OUTPUT="$tmp/tunnel-ss"
PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 1 "${china_forwards[@]}"

# A single-port supervisor is rejected when its SSH child uses the wrong target.
printf '%s\n' \
  'ssh -NT -L 127.0.0.1:18100:127.0.0.1:8100 wrong-trainer' \
  >"$tmp/tunnel-owner"
export TUNNEL_OWNER_PID=4242 TUNNEL_OWNER_PPID=4241
if PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 1 "${china_forwards[@]}"; then
  echo "wrong child target was accepted" >&2
  exit 1
fi

# An orphan listener is rejected without readiness-time cleanup.
printf '%s\n' \
  'ssh -NT -L 127.0.0.1:18100:127.0.0.1:8100 china-trainer' \
  >"$tmp/tunnel-owner"
orphan_kill_log="$tmp/orphan-kill"
kill() { printf '%s\n' "$*" >"$orphan_kill_log"; return 0; }
export TUNNEL_OWNER_PPID=1
if PATH="$tmp/bin:$PATH" tunnel_listener_ready 18100 china-tunnel-lora-0 china-trainer 1 \
    -L "127.0.0.1:18100:127.0.0.1:8100"; then
  echo "orphan listener was accepted" >&2
  exit 1
else
  test "$?" -eq 1
fi
test ! -s "$orphan_kill_log"
unset -f kill
unset TUNNEL_OWNER_PID TUNNEL_OWNER_PPID TUNNEL_SUPERVISOR_PID \
  TUNNEL_OWNER_COMMAND TUNNEL_SUPERVISOR_COMMAND TUNNEL_SS_OUTPUT

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
  for port in $(seq 18100 18194); do printf '127.0.0.1:%s\n' "$port"; done
else
  for port in $(seq 18100 18195); do printf '127.0.0.1:%s\n' "$port"; done
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
printf 'peer tunnel uses one API connection and ninety-six bounded one-port lanes\n'
