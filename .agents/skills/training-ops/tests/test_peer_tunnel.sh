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
if [ -r "${PEER_TEST_LOG:?}" ] && grep -Fq -- "$pattern" "$PEER_TEST_LOG"; then
  exit 0
fi
exit 1
EOF
cat >"$tmp/bin/setsid" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" | tr '\n' ' ' >>"${PEER_TEST_LOG:?}"
printf '\n' >>"${PEER_TEST_LOG:?}"
EOF
chmod +x "$tmp/bin/pgrep" "$tmp/bin/setsid"

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

PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" peer_tunnel
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
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$log" peer_tunnel
test "$(wc -l <"$log")" -eq 25

china_log="$tmp/china-tunnel.log"
trainer_ssh_target=china-trainer
adapter_port=8100
PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" china_tunnel_remove_legacy
china_lane_size=4
china_lane_start=0
china_lane_number=0
china_lane_ports=()
china_lane_forwards=()
while [ "$china_lane_start" -lt "${#lora_source_ports[@]}" ]; do
  china_lane_ports=("${lora_source_ports[@]:china_lane_start:china_lane_size}")
  china_lane_forwards=()
  for lora_source_port in "${china_lane_ports[@]}"; do
    china_lane_forwards+=(
      -L "127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
    )
  done
  PATH="$tmp/bin:$PATH" PEER_TEST_LOG="$china_log" china_tunnel_connection "lora-$china_lane_number" \
    "-L 127.0.0.1:${china_lane_ports[0]}:127.0.0.1:$adapter_port" \
    "${china_lane_forwards[@]}"
  china_lane_start=$((china_lane_start + china_lane_size))
  china_lane_number=$((china_lane_number + 1))
done
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
printf 'peer tunnel uses one API connection and twenty-four bounded self-healing lanes\n'
