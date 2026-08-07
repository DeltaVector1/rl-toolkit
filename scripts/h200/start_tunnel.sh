#!/usr/bin/env bash
set -euo pipefail

remote_port=${1:?usage: start_tunnel.sh REMOTE_PORT LOCAL_PORT}
local_port=${2:?usage: start_tunnel.sh REMOTE_PORT LOCAL_PORT}
health_port=$((remote_port + 1000))

# Keepalive: 15s x 8 = a dead TCP path is reaped in ~120s.
#
# This value has now been set three ways, and each was right for the watchdog
# that existed at the time. 15 x 20 (300s) was chosen when nothing else could
# detect a wedge, and it let wedged tunnels pile up. 10 x 3 (30s) replaced it,
# but SSH keepalives are ordinary packets on a link that loses 10-33% of them,
# so three consecutive misses happen routinely on a tunnel that is working --
# and under load they miss on every tunnel at once. SSH was then killing live
# tunnels and every request in flight on them.
#
# scripts/china/tunnel_watchdog.sh now owns wedge detection from the n82 side,
# where it can see the thing SSH cannot: forward alive, no data flowing. So
# SSH's only job here is to notice a genuinely dead TCP path, and it should be
# patient about it. Exactly one of the two may be impatient; right now neither
# is, because a wedge is expensive to misdiagnose in both directions.
#
# Compression is the reason this link works at all. On the batch tunnel it took
# throughput from under 130 KB/s to 4.9 MB/s. Rollout traffic is JSON holding
# generated text, which compresses just as well, and on a loss-limited path
# fewer bytes means proportionally fewer lost packets. It costs a little CPU on
# hosts that have plenty.
while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=30 china-trainer \
    "sudo ss -H -lntp 'sport = :$remote_port' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | xargs -r sudo kill; sudo ss -H -lntp 'sport = :$health_port' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | xargs -r sudo kill" || true
  ssh -NT \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=8 \
    -o TCPKeepAlive=yes \
    -o IPQoS=none \
    -o Compression=yes \
    -R "127.0.0.1:$remote_port:127.0.0.1:$local_port" \
    -R "127.0.0.1:$health_port:127.0.0.1:8000" \
    china-trainer || true
  sleep 5
done
