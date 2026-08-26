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
    -o Compression=yes \
    -L 127.0.0.1:18100:127.0.0.1:8100 \
    china-trainer &
  tunnel_pid=$!
  misses=0
  while kill -0 "$tunnel_pid" 2>/dev/null; do
    sleep 30
    if curl --head --fail --silent --show-error --max-time 15 \
      http://127.0.0.1:18100/ -o /dev/null; then
      misses=0
    else
      misses=$((misses + 1))
      echo "$(date -u +%FT%TZ) adapter control tunnel probe failed ($misses/2)" >&2
      if [ "$misses" -ge 2 ]; then
        echo "$(date -u +%FT%TZ) recycling stalled adapter control tunnel" >&2
        kill "$tunnel_pid" 2>/dev/null || true
        break
      fi
    fi
  done
  wait "$tunnel_pid" 2>/dev/null || true
  sleep 5
done
