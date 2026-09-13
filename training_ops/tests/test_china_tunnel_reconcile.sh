#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="$root/training_ops/runtime/h200/start_orchestrator.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/args" "$tmp/ppid"
cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = -G ]; then
  printf '%s\n' 'hostname 10.92.20.82' 'port 22' 'proxyjump china-jump'
fi
EOF
cat >"$tmp/bin/ps" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${WAIT_MODE:-0}" = 1 ] && [ "${1:-}" = -o ] && [ "${2:-}" = args= ]; then
  count=0
  [ -r "${WAIT_COUNT:?}" ] && count=$(<"$WAIT_COUNT")
  count=$((count + 1))
  printf '%s\n' "$count" >"$WAIT_COUNT"
  [ "$count" -eq 1 ] || exit 0
  cat "${PS_ARGS:?}/${4:?}" 2>/dev/null || true
  exit 0
fi
case "${1:-}" in
  -eo) cat "${PS_LIST:?}" ;;
  -o)
    case "${2:-}" in
      pgid=) printf '%s\n' "${4:?}" ;;
      args=) cat "${PS_ARGS:?}/${4:?}" 2>/dev/null || true ;;
      ppid=) cat "${PS_PPID:?}/${4:?}" 2>/dev/null || printf '1\n' ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 1 ;;
esac
EOF
cat >"$tmp/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cat "${SS_LIST:?}"
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/ps" "$tmp/bin/ss"

start=$(grep -n '^generic_tunnel_ssh_match() {' "$script" | cut -d: -f1)
end=$(grep -n '^china_legacy_cleanup_required=0$' "$script" | cut -d: -f1)
source <(sed -n "${start},$((end - 1))p" "$script")

lora_source_ports=(18100 18101)
adapter_port=8100
trainer_ssh_target=china-trainer
export PS_ARGS="$tmp/args" PS_PPID="$tmp/ppid" SS_LIST="$tmp/ss"
PATH="$tmp/bin:$PATH"

current_ssh='ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ChannelTimeout=*-tcpip=45s -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer'
old_ssh='ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o IPQoS=none -o Compression=no -L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100 china-trainer'
current_supervisor="bash -c while true; do $current_ssh; sleep 3; done # china-tunnel-lora-0"
old_supervisor="bash -c while true; do $old_ssh; sleep 3; done # china-tunnel-lora-0"
proxy_child='ssh -W [10.92.20.82]:22 china-jump'
forwards=(-L 127.0.0.1:18100:127.0.0.1:8100 -L 127.0.0.1:18101:127.0.0.1:8100)

test "$(china_tunnel_proxy_child_expected)" = "$proxy_child"
for wrong in \
  'ssh -W [10.92.20.81]:22 china-jump' \
  'ssh -W [10.92.20.82]:23 china-jump' \
  'ssh -W [10.92.20.82]:22 other-jump' \
  'ssh -W [10.92.20.82]:22 china-jump extra'; do
  if china_tunnel_proxy_child_match "$wrong"; then
    echo "accepted wrong ProxyJump child: $wrong" >&2
    exit 1
  fi
done

china_tunnel_ssh_args_match "$current_ssh" current
china_tunnel_ssh_args_match "$old_ssh" old
if china_tunnel_ssh_args_match "$old_ssh" current; then exit 1; fi
if china_tunnel_ssh_args_match "${old_ssh/china-trainer/wrong-trainer}" old; then exit 1; fi
china_tunnel_supervisor_match "$old_supervisor" cleanup china-tunnel-lora-0

write_group() {
  local file=$1 pgid=$2 supervisor=$3
  shift 3
  {
    printf '%s %s\n' "$pgid" "$supervisor"
    for member in "$@"; do printf '%s %s\n' "$pgid" "$member"; done
  } >"$file"
}

group="$tmp/group"
write_group "$group" 100 "$current_supervisor" "$current_ssh" "$proxy_child" 'sleep 3'
export PS_LIST="$group"
china_tunnel_group_safe 100 china-tunnel-lora-0

write_group "$group" 100 "$current_supervisor" "$current_ssh" "$proxy_child" "$proxy_child"
if china_tunnel_group_safe 100 china-tunnel-lora-0; then exit 1; fi
write_group "$group" 100 "$current_supervisor" "$current_ssh" 'evil child'
if china_tunnel_group_safe 100 china-tunnel-lora-0; then exit 1; fi
write_group "$group" 100 "$old_supervisor"
china_tunnel_group_safe 100 china-tunnel-lora-0

kill_log="$tmp/kill"
kill() { printf '%s\n' "$*" >>"$kill_log"; }

cat >"$tmp/ss" <<EOF
LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=201,fd=5))
LISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(("ssh",pid=201,fd=6))
EOF
printf '%s\n' "$current_ssh" >"$tmp/args/201"
printf '101\n' >"$tmp/ppid/201"
printf '%s\n' "$current_supervisor" >"$tmp/args/101"
printf '%s\n' "$current_supervisor" >"$tmp/args/102"
write_group "$group" 101 "$current_supervisor" "$current_ssh"
{
  cat "$group"
  printf '102 %s\n' "$current_supervisor"
} >"$group.duplicates"
export PS_LIST="$group.duplicates"
china_legacy_cleanup_required=0
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
china_legacy_cleanup_ports=()
restart_tunnels=0
china_tunnel_connection lora-0 "${forwards[@]}"
test "$(grep -F -c -- '-- -102' "$kill_log")" -eq 1
test "$(grep -F -c -- '-- -101' "$kill_log")" -eq 0

# Two one-port owners are split ownership, not a ready two-port lane.
printf 'LISTEN 0 128 127.0.0.1:18100 0.0.0.0:* users:(("ssh",pid=202,fd=5))\nLISTEN 0 128 127.0.0.1:18101 0.0.0.0:* users:(("ssh",pid=203,fd=6))\n' >"$tmp/ss"
printf '%s\n' "${current_ssh/-L 127.0.0.1:18101:127.0.0.1:8100/}" >"$tmp/args/202"
printf '%s\n' "${current_ssh/-L 127.0.0.1:18100:127.0.0.1:8100/}" >"$tmp/args/203"
printf '101\n' >"$tmp/ppid/202"
printf '101\n' >"$tmp/ppid/203"
if china_tunnel_lane_listener_owner 101 china-tunnel-lora-0 china-trainer "${forwards[@]}"; then
  echo 'split listeners were accepted as one lane' >&2
  exit 1
fi

# A supervisor and a sleep member are valid during transition.
write_group "$group" 100 "$current_supervisor" 'sleep 3'
export PS_LIST="$group"
china_tunnel_group_safe 100 china-tunnel-lora-0

# A terminating process group clears after exactly two polls.
: >"$tmp/ss"
printf '%s\n' "$current_ssh" >"$tmp/args/4242"
: >"$tmp/wait-count"
export PS_LIST="$tmp/empty" WAIT_MODE=1 WAIT_COUNT="$tmp/wait-count"
: >"$PS_LIST"
china_legacy_cleanup_pids=(4242)
china_legacy_cleanup_pgids=(4242)
china_legacy_cleanup_orphan_pids=()
china_legacy_cleanup_ports=(18100)
sleep() { :; }
china_tunnel_wait_legacy_clear
test "$(<"$WAIT_COUNT")" -eq 2
unset -f sleep
unset WAIT_MODE WAIT_COUNT

echo 'China tunnel reconcile fixtures passed'
