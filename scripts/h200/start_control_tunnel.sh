#!/usr/bin/env bash
set -euo pipefail

while true; do
  ssh -NT \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o IPQoS=none \
    -L 127.0.0.1:18100:127.0.0.1:8100 \
    china-trainer || true
  sleep 5
done
