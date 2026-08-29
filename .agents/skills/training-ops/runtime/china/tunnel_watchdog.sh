#!/usr/bin/env bash
# Rebuild local engine tunnels that stop passing requests.
set -uo pipefail

ports=${ports:?set ports}
service_host=${service_host:-127.0.0.1}
timeout=${timeout:-20}
interval=${interval:-60}
heartbeat=${heartbeat:-5}
misses_before_kill=${misses_before_kill:-6}
max_kills_per_pass=${max_kills_per_pass:-2}
kill_cooldown=${kill_cooldown:-120}
mass_failure_pct=${mass_failure_pct:-80}
declare -A misses
total=0
for port in $ports; do total=$((total + 1)); done
mass_streak=0
cycle=0
last_kill=0

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "tunnel watchdog started (${total} ports, timeout=${timeout}s, interval=${interval}s)"

while true; do
  results=$( for port in $ports; do
      ( printf '%s %s\n' "$port" \
          "$(curl -s -o /dev/null -m "$timeout" -w '%{http_code}' "http://$service_host:$port/v1/models" 2>/dev/null)" ) &
    done
    wait )

  failed=""
  nfail=0
  while read -r port code; do
    [ -z "$port" ] && continue
    if [ "$code" = "200" ]; then
      misses[$port]=0
    else
      nfail=$((nfail + 1))
      failed="$failed $port"
    fi
  done <<< "$(printf '%s\n' "$results" | sort -n)"

  if [ "$total" -gt 0 ] && [ $((nfail * 100 / total)) -ge "$mass_failure_pct" ]; then
    mass_streak=$((mass_streak + 1))
    [ $((mass_streak % 5)) -eq 1 ] && log "$nfail/$total tunnels failing at once"
  else
    mass_streak=0
  fi

  now=$(date +%s)
  cooling=0
  [ $((now - last_kill)) -lt "$kill_cooldown" ] && cooling=1

  killed=0
  deferred=0
  for port in $failed; do
    misses[$port]=$(( ${misses[$port]:-0} + 1 ))
    [ "${misses[$port]}" -lt "$misses_before_kill" ] && continue
    if [ "$cooling" -eq 1 ] || [ "$killed" -ge "$max_kills_per_pass" ]; then
      deferred=$((deferred + 1))
      continue
    fi
    pid=$(sudo ss -H -lntp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
    if [ -n "$pid" ]; then
      sudo kill "$pid" 2>/dev/null && {
        killed=$((killed + 1)); misses[$port]=0
        log "killed wedged tunnel on :$port (pid $pid) after ${misses_before_kill} misses"
      }
    else
      log "tunnel :$port failing but nothing holds the port"
      misses[$port]=0
    fi
  done
  if [ "$killed" -gt 0 ]; then
    last_kill=$now
    log "rebuilt $killed tunnel(s)${deferred:+, deferred $deferred}"
  elif [ "$deferred" -gt 0 ]; then
    log "$deferred tunnel(s) past threshold, deferred (max ${max_kills_per_pass}/pass, ${kill_cooldown}s apart)"
  fi

  cycle=$((cycle + 1))
  if [ "$heartbeat" -gt 0 ] && [ $((cycle % heartbeat)) -eq 0 ]; then
    log "heartbeat: $((total - nfail))/$total tunnels healthy"
  fi

  sleep "$interval"
done
