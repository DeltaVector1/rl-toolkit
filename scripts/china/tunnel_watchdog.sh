#!/usr/bin/env bash
# Rebuild engine tunnels that are listening but not passing data. Runs on n82.
#
# The H200-side supervisor (scripts/h200/start_tunnel.sh) only rebuilds a
# tunnel when its ssh process EXITS. Over the lossy China<->H200 path (10-33%
# packet loss measured) a tunnel routinely wedges with the forward still
# established: the port accepts connections and then never answers. With
# ServerAliveCountMax=20 x Interval=15 that persists for 300s, and wedged
# tunnels accumulate -- observed 22 of 24 dead while h200a took zero traffic.
#
# Killing the n82-side sshd that holds the port makes the remote ssh exit, so
# the existing supervisor rebuilds it within ~5s.
#
# CRITICAL: a dead ENGINE looks exactly like a wedged tunnel from here -- the
# SSH forward still accepts the connection either way. On 2026-08-06 both vLLM
# engines died (EngineCore RPC timeout) and this script spent 2.6 hours
# cheerfully rebuilding all 24 tunnels against a dead backend while the run
# sat stalled. n82 has no ssh access to the H200s, so it cannot check or fix
# the engines directly; scripts/h200/engine_watchdog.sh does that locally.
# What this script must do is NOT mask the problem: when nearly every tunnel
# is failing at once the cause is almost certainly upstream, so it stops
# churning and says so loudly instead.
set -uo pipefail

PORTS=${PORTS:-$(seq 18001 18024)}
TIMEOUT=${TIMEOUT:-20}
INTERVAL=${INTERVAL:-60}
# Heartbeat. Without it this script logs only when it kills something, so a
# healthy idle watchdog and a wedged one look identical from outside -- and the
# only other liveness test, `pgrep -fc`, counts its own grep and can never
# return zero. A periodic line makes "still checking" observable.
HEARTBEAT=${HEARTBEAT:-5}
# Above this fraction of simultaneous failures, blame the backend, not the
# tunnels. 24 ports -> 20 failures trips it.
MASS_FAILURE_PCT=${MASS_FAILURE_PCT:-80}
declare -A misses
total=0
for _p in $PORTS; do total=$((total + 1)); done
mass_streak=0
cycle=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "tunnel watchdog started (${total} ports, timeout=${TIMEOUT}s, interval=${INTERVAL}s)"

while true; do
  failed=""
  nfail=0
  for p in $PORTS; do
    code=$(curl -s -o /dev/null -m "$TIMEOUT" -w "%{http_code}" "http://127.0.0.1:$p/v1/models" 2>/dev/null)
    if [ "$code" = "200" ]; then
      misses[$p]=0
    else
      nfail=$((nfail + 1))
      failed="$failed $p"
    fi
  done

  # ALWAYS rebuild. An earlier version refused when >=80% failed, on the theory
  # that mass failure meant a dead backend it could not fix. That was wrong and
  # it froze the run: at batch 1024 all 24 tunnels wedged at once under 4x the
  # rollout load while the engines were healthy and generating, and the
  # watchdog sat idle for five hours. The asymmetry is decisive -- rebuilding
  # against a dead backend is harmless (it just fails again next cycle), while
  # refusing to rebuild against a live backend stalls everything. So warn
  # loudly on mass failure, then rebuild anyway.
  if [ "$total" -gt 0 ] && [ $((nfail * 100 / total)) -ge "$MASS_FAILURE_PCT" ]; then
    mass_streak=$((mass_streak + 1))
    [ $((mass_streak % 5)) -eq 1 ] && log "*** $nfail/$total tunnels failing at once -- check the engines too, but rebuilding regardless"
  else
    mass_streak=0
  fi

  killed=0
  for p in $failed; do
    misses[$p]=$(( ${misses[$p]:-0} + 1 ))
    # Two consecutive misses: one slow reply under load must not trigger a rebuild.
    if [ "${misses[$p]}" -ge 2 ]; then
      pid=$(sudo ss -H -lntp "sport = :$p" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
      if [ -n "$pid" ]; then
        sudo kill "$pid" 2>/dev/null && { killed=$((killed + 1)); log "killed wedged tunnel on :$p (pid $pid)"; }
      else
        log "tunnel :$p failing but nothing holds the port (remote supervisor down?)"
      fi
      misses[$p]=0
    fi
  done
  [ "$killed" -gt 0 ] && log "rebuilt $killed tunnel(s)"

  cycle=$((cycle + 1))
  if [ "$HEARTBEAT" -gt 0 ] && [ $((cycle % HEARTBEAT)) -eq 0 ]; then
    log "heartbeat: $((total - nfail))/$total tunnels healthy"
  fi

  sleep "$INTERVAL"
done
