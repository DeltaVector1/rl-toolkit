#!/usr/bin/env bash
# Rebuild engine tunnels that are listening but not passing data. Runs on n82.
# Killing the n82 sshd makes the H200 supervisor rebuild the forward.
# A dead engine looks the same from n82, so mass failures also log a warning.
set -uo pipefail

PORTS=${PORTS:-$(seq 18001 18048)}
TIMEOUT=${TIMEOUT:-20}
INTERVAL=${INTERVAL:-60}
HEARTBEAT=${HEARTBEAT:-5}
# Busy tunnels can miss the shared HTTP probe. Require repeated misses and
# limit repairs so the watchdog does not destroy live rollouts.
MISSES_BEFORE_KILL=${MISSES_BEFORE_KILL:-6}
MAX_KILLS_PER_PASS=${MAX_KILLS_PER_PASS:-2}
KILL_COOLDOWN=${KILL_COOLDOWN:-120}
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
  # Probe in parallel so a failed cycle takes TIMEOUT, not ports * TIMEOUT.
  results=$( for p in $PORTS; do
      ( printf '%s %s\n' "$p" \
          "$(curl -s -o /dev/null -m "$TIMEOUT" -w '%{http_code}' "http://127.0.0.1:$p/v1/models" 2>/dev/null)" ) &
    done
    wait )

  failed=""
  nfail=0
  # Keep the loop in this shell so updates to misses persist.
  while read -r p code; do
    [ -z "$p" ] && continue
    if [ "$code" = "200" ]; then
      misses[$p]=0
    else
      nfail=$((nfail + 1))
      failed="$failed $p"
    fi
  done <<< "$(printf '%s\n' "$results" | sort -n)"

  # Warn on mass failure, but continue the rate-limited repair.
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
