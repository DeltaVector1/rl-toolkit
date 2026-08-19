#!/usr/bin/env bash
set -euo pipefail

# Run on each H200 rollout host. Each local port gets an independent SSH TCP
# connection to n82's checkpoint server. This avoids one congestion window for
# a 60 GB BF16 policy update on the lossy 217 ms China link.
first_port=${FIRST_PORT:-19001}
last_port=${LAST_PORT:-19128}

for port in $(seq "$first_port" "$last_port"); do
  marker="muse-weight-tunnel-$port"
  if pgrep -f -- "$marker" >/dev/null; then
    continue
  fi
  setsid nohup bash -c '
    marker=$0
    port=$1
    while true; do
      ssh -NT \
        -o BatchMode=yes \
        -o ConnectTimeout=30 \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -o IPQoS=none \
        -o Compression=no \
        -L "127.0.0.1:$port:127.0.0.1:8100" \
        china-trainer
      sleep 3
    done
  ' "$marker" "$port" >/dev/null 2>&1 </dev/null &
done

sleep 5
up=0
for port in $(seq "$first_port" "$last_port"); do
  if ss -tln 2>/dev/null | grep -q "127.0.0.1:$port "; then
    up=$((up + 1))
  fi
done
expected=$((last_port - first_port + 1))
echo "Muse checkpoint tunnels: $up/$expected listening"
test "$up" -eq "$expected"
