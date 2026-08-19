#!/usr/bin/env bash
set -euo pipefail

remote_port=${1:-26407}
judge_host=${JUDGE_HOST:-202.214.223.66}
judge_port=${JUDGE_PORT:-26406}

while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "sudo ss -H -lntp 'sport = :$remote_port' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | xargs -r sudo kill" \
    || true
  ssh -NT \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=8 \
    -o TCPKeepAlive=yes \
    -o IPQoS=none \
    -o Compression=yes \
    -R "127.0.0.1:$remote_port:$judge_host:$judge_port" \
    china-trainer || true
  sleep 5
done
