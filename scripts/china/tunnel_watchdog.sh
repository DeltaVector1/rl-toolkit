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

PORTS=${PORTS:-$(seq 18001 18048)}
TIMEOUT=${TIMEOUT:-20}
INTERVAL=${INTERVAL:-60}
# Heartbeat. Without it this script logs only when it kills something, so a
# healthy idle watchdog and a wedged one look identical from outside -- and the
# only other liveness test, `pgrep -fc`, counts its own grep and can never
# return zero. A periodic line makes "still checking" observable.
HEARTBEAT=${HEARTBEAT:-5}
# Rate limits. The /v1/models probe shares each tunnel with rollout traffic, so
# a BUSY tunnel misses probes while working perfectly -- and it misses them on
# every link at once, because the load arrives everywhere at once. Measured on
# 2026-08-07 at 251 rollouts in flight: healthy-tunnel counts tracked the batch
# phase (23/24 at the batch tail, 1/24 mid-batch) while both engines held 100%
# GPU and ~500 tok/s. A 2-miss threshold with no cap then killed up to 9 live
# tunnels in one pass, destroying every rollout in flight on them: 68 kills and
# 139 rollout failures in one hour.
#
# A dead tunnel never recovers on its own, so a slow detection window costs
# nothing. A live tunnel killed under load costs every rollout on it. Prefer
# the slow window. This is the same conclusion scripts/network/watch_rollout_links.sh
# reached (30 misses, one link per pass, 2 minutes apart); do not tighten these
# while running large batches.
MISSES_BEFORE_KILL=${MISSES_BEFORE_KILL:-6}
MAX_KILLS_PER_PASS=${MAX_KILLS_PER_PASS:-2}
KILL_COOLDOWN=${KILL_COOLDOWN:-120}
# Above this fraction of simultaneous failures, blame the backend, not the
# tunnels. 24 ports -> 20 failures trips it.
MASS_FAILURE_PCT=${MASS_FAILURE_PCT:-80}
declare -A misses
total=0
for _p in $PORTS; do total=$((total + 1)); done
mass_streak=0
cycle=0
last_kill=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "tunnel watchdog started (${total} ports, timeout=${TIMEOUT}s, interval=${INTERVAL}s)"

while true; do
  # Probe every port at once. Serially this loop takes 24 x TIMEOUT in the bad
  # case, so one cycle ran 2 to 7 minutes instead of the nominal interval and
  # MISSES_BEFORE_KILL meant half an hour rather than a few minutes. Measured
  # 2026-08-07: no log line at all for 10 minutes because the first heartbeat
  # was still 4 cycles away.
  results=$( for p in $PORTS; do
      ( printf '%s %s\n' "$p" \
          "$(curl -s -o /dev/null -m "$TIMEOUT" -w '%{http_code}' "http://127.0.0.1:$p/v1/models" 2>/dev/null)" ) &
    done
    wait )

  failed=""
  nfail=0
  # A here-string, not a pipe: a pipe puts the loop in a subshell and every
  # update to the misses array is then discarded.
  while read -r p code; do
    [ -z "$p" ] && continue
    if [ "$code" = "200" ]; then
      misses[$p]=0
    else
      nfail=$((nfail + 1))
      failed="$failed $p"
    fi
  done <<< "$(printf '%s\n' "$results" | sort -n)"

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

  now=$(date +%s)
  cooling=0
  [ $((now - last_kill)) -lt "$KILL_COOLDOWN" ] && cooling=1

  killed=0
  deferred=0
  for p in $failed; do
    misses[$p]=$(( ${misses[$p]:-0} + 1 ))
    # Not yet convinced this port is really dead rather than just busy.
    [ "${misses[$p]}" -lt "$MISSES_BEFORE_KILL" ] && continue
    # Past the threshold but rate limited. Do NOT reset the miss count here:
    # the port stays a candidate and gets rebuilt on a later pass.
    if [ "$cooling" -eq 1 ] || [ "$killed" -ge "$MAX_KILLS_PER_PASS" ]; then
      deferred=$((deferred + 1))
      continue
    fi
    pid=$(sudo ss -H -lntp "sport = :$p" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
    if [ -n "$pid" ]; then
      sudo kill "$pid" 2>/dev/null && {
        killed=$((killed + 1)); misses[$p]=0
        log "killed wedged tunnel on :$p (pid $pid) after ${MISSES_BEFORE_KILL} misses"
      }
    else
      log "tunnel :$p failing but nothing holds the port (remote supervisor down?)"
      misses[$p]=0
    fi
  done
  if [ "$killed" -gt 0 ]; then
    last_kill=$now
    log "rebuilt $killed tunnel(s)${deferred:+, deferred $deferred}"
  elif [ "$deferred" -gt 0 ]; then
    log "$deferred tunnel(s) past threshold, deferred (rate limit: max ${MAX_KILLS_PER_PASS}/pass, ${KILL_COOLDOWN}s apart)"
  fi

  cycle=$((cycle + 1))
  if [ "$HEARTBEAT" -gt 0 ] && [ $((cycle % HEARTBEAT)) -eq 0 ]; then
    log "heartbeat: $((total - nfail))/$total tunnels healthy"
  fi

  sleep "$INTERVAL"
done
