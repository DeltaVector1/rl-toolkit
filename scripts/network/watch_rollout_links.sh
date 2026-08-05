#!/usr/bin/env bash
# Repair half-dead rollout tunnels on n82.
#
# A reverse tunnel can keep LISTENing here while nothing reaches the H200, so
# rollouts routed to it hang until the 3,600 second rollout timeout. Killing the
# owning sshd makes start_tunnel.sh reconnect it.
#
# Two rules keep the cure from being worse than the disease. At 1,024 live
# rollouts the health probe shares the tunnel with rollout traffic and can time
# out on a link that is merely busy, and those timeouts arrive on every link at
# once. So: require a long run of consecutive failures before acting, and
# reconnect at most one link per pass. Without both, a burst of congestion
# tears down every tunnel simultaneously and destroys all in-flight rollouts.
set -uo pipefail

PORTS="$(seq 18001 18024 | tr "\n" " ")"
TIMEOUT=${LINK_TIMEOUT:-15}       # seconds per health probe
# /run is root-owned: unprivileged writes fail silently and the failure
# counter never advances, making the watcher a no-op. Use a writable dir
# and refuse to run if it is not.
STATE_DIR=${LINK_STATE_DIR:-/data/glm52-rl/cache/links}
mkdir -p "$STATE_DIR" 2>/dev/null
[ -w "$STATE_DIR" ] || { echo "FATAL: $STATE_DIR not writable" >&2; exit 1; }
THRESHOLD=${LINK_THRESHOLD:-30}   # consecutive failures (~5 min at 10s cadence)
COOLDOWN=${LINK_COOLDOWN:-120}    # seconds to wait after a reconnect

probe() {  # 0 = healthy
  curl -fsS --max-time "$TIMEOUT" "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

last_action=0

while true; do
  worst_port=""; worst_count=0
  for port in $PORTS; do
    health=$((port + 1000))
    state="$STATE_DIR/glm52-rollout-link-$port.failures"
    if probe "$health"; then
      rm -f "$state"
      continue
    fi
    failures=$(( $(cat "$state" 2>/dev/null || echo 0) + 1 ))
    echo "$failures" >"$state"
    if (( failures >= THRESHOLD && failures > worst_count )); then
      worst_count=$failures; worst_port=$port
    fi
  done

  now=$(date +%s)
  if [[ -n $worst_port ]] && (( now - last_action >= COOLDOWN )); then
    health=$((worst_port + 1000))
    pid=$(sudo ss -H -lntp "sport = :$health" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)
    if [[ -n ${pid:-} ]]; then
      echo "$(date -u +%FT%TZ) reconnecting rollout link $worst_port after $worst_count consecutive failures"
      sudo kill "$pid"
      rm -f "/run/glm52-rollout-link-$worst_port.failures"
      last_action=$now
    fi
  fi

  sleep 10
done
