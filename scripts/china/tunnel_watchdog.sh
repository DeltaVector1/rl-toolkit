#!/usr/bin/env bash
# Detect engine tunnels that are listening but not passing data, and kill them.
#
# The H200-side supervisor (scripts/h200/start_tunnel.sh) only rebuilds a tunnel
# when its ssh process EXITS. Over the lossy China<->H200 path (10-33% packet
# loss measured) a tunnel routinely wedges with the forward still established:
# the port on n82 accepts connections and then never answers. With
# ServerAliveCountMax=20 x Interval=15 that state persists for 300s, and wedged
# tunnels accumulate -- observed 22 of 24 dead, with h200a taking zero traffic
# while h200b carried the whole run.
#
# Killing the n82-side sshd that holds the port makes the remote ssh exit, so
# the existing supervisor loop rebuilds it within ~5s. Nothing to change on the
# H200 hosts.
#
# Two consecutive failures are required before killing: a single slow reply
# under load is normal and must not trigger a rebuild.
set -uo pipefail

PORTS=${PORTS:-$(seq 18001 18024)}
TIMEOUT=${TIMEOUT:-10}
INTERVAL=${INTERVAL:-45}
declare -A misses

while true; do
  killed=0
  for p in $PORTS; do
    code=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" "http://127.0.0.1:$p/v1/models" 2>/dev/null)
    if [ "$code" = "200" ]; then
      misses[$p]=0
      continue
    fi
    misses[$p]=$(( ${misses[$p]:-0} + 1 ))
    if [ "${misses[$p]}" -ge 2 ]; then
      pid=$(sudo ss -H -lntp "sport = :$p" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
      if [ -n "$pid" ]; then
        sudo kill "$pid" 2>/dev/null && killed=$((killed+1))
        echo "$(date -u +%H:%M:%S) killed wedged tunnel on :$p (pid $pid, code=${code:-none})"
      fi
      misses[$p]=0
    fi
  done
  [ "$killed" -gt 0 ] && echo "$(date -u +%H:%M:%S) rebuilt $killed tunnel(s)"
  sleep "$INTERVAL"
done
