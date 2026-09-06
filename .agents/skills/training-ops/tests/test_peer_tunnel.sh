#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
script="$root/.agents/skills/training-ops/runtime/h200/start_orchestrator.sh"
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
      if [ "${PEER_CHINA_MODE:-0}" = 1 ] && [ "$port" -ge 18100 ] && [ "$port" -le 18195 ]; then
        echo $((10000 + (port - 18100) / 4))
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
  if [ "$pid" -ge 10000 ] 2>/dev/null; then
    lane=$((pid - 10000))
    grep -E "done # china-tunnel-lora-${lane}[[:space:]]*$" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  else
    query=$(<"${PEER_QUERY:?}")
    grep -F -- "$query" "${PEER_TEST_LOG:?}" | tail -n 1 | sed 's/[[:space:]]*$//' || true
  fi
elif [ "${1:-}" = "-o" ] && [ "${2:-}" = "pgid=" ]; then
  echo "${4:?missing process id}"
elif [ "${1:-}" = "-eo" ]; then
  [ -n "${PEER_GROUP_COMMANDS:-}" ] && cat "$PEER_GROUP_COMMANDS"
fi
EOF
chmod +x "$tmp/bin/pgrep" "$tmp/bin/setsid"
chmod +x "$tmp/bin/ps"

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
  test "$reverse_count" -le 4
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done < <(grep -F -- '-R 127.0.0.1:' "$log")

# A second launcher invocation finds the exact API and lane signatures.
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" PEER_QUERY="$query" peer_tunnel
test "$(wc -l <"$log")" -eq 25

china_log="$tmp/china-tunnel.log"
trainer_ssh_target=china-trainer
adapter_port=8100
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_remove_legacy
china_lane_start=0
china_lane_number=0
china_lane_ports=()
china_lane_forwards=()
run_china_lanes() {
  china_lane_start=0
  china_lane_number=0
  while [ "$china_lane_start" -lt "${#lora_source_ports[@]}" ]; do
    china_lane_ports=("${lora_source_ports[@]:china_lane_start:4}")
    china_lane_forwards=()
    for lora_source_port in "${china_lane_ports[@]}"; do
      china_lane_forwards+=(
        -L "127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
      )
    done
    PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" PEER_QUERY="$query" PEER_CHINA_MODE=1 china_tunnel_connection "lora-$china_lane_number" \
      "${china_lane_forwards[@]}"
    china_lane_start=$((china_lane_start + 4))
    china_lane_number=$((china_lane_number + 1))
  done
}

# A fresh launch creates twenty-four four-forward lanes.
run_china_lanes
for _ in $(seq 1 50); do
  [ -f "$china_log" ] && [ "$(wc -l <"$china_log")" -eq 24 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 24
while IFS= read -r line; do
  forward_count=$(grep -o -- '-L 127.0.0.1:' <<<"$line" | wc -l)
  test "$forward_count" -le 4
  grep -Fq -- '-o ExitOnForwardFailure=yes' <<<"$line"
done <"$china_log"
# A repeat launch preserves all matching groups without adding processes.
run_china_lanes
test "$(wc -l <"$china_log")" -eq 24
# A target change removes only the owned old groups before regrouping.
trainer_ssh_target=china-trainer-regrouped
run_china_lanes
for _ in $(seq 1 50); do
  [ "$(wc -l <"$china_log")" -eq 48 ] && break
  sleep 0.1
done
test "$(wc -l <"$china_log")" -eq 48

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
printf 'peer tunnel uses one API connection and twenty-four bounded self-healing lanes\n'
