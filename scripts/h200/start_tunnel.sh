#!/usr/bin/env bash
set -euo pipefail

remote_port=${1:?usage: start_tunnel.sh REMOTE_PORT LOCAL_PORT}
local_port=${2:?usage: start_tunnel.sh REMOTE_PORT LOCAL_PORT}
health_port=$((remote_port + 1000))

while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "sudo ss -H -lntp 'sport = :$remote_port' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | xargs -r sudo kill; sudo ss -H -lntp 'sport = :$health_port' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | xargs -r sudo kill" || true
  ssh -NT \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=20 \
    -o TCPKeepAlive=yes \
    -o IPQoS=none \
    -R "127.0.0.1:$remote_port:127.0.0.1:$local_port" \
    -R "127.0.0.1:$health_port:127.0.0.1:8000" \
    china-trainer || true
  sleep 5
done
