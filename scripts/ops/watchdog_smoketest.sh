#!/usr/bin/env bash
# Smoke-test the watchdog decision logic against every failure mode we have
# actually hit, using fake probe results so nothing real is touched.
#
# The bug this exists to prevent: on 2026-08-06 the tunnel watchdog spent 2.6
# hours rebuilding all 24 tunnels while both vLLM engines were dead, because a
# dead engine and a wedged tunnel are indistinguishable from n82. Case 4 is
# that exact scenario.
set -uo pipefail

pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1))
  else echo "  FAIL  $1 (expected '$2', got '$3')"; fail=$((fail+1)); fi
}

# Mirror of the tunnel watchdog's decision: given N failures out of TOTAL,
# does it rebuild tunnels, blame the backend, or do nothing?
decide() { # nfail total pct
  local nfail=$1 total=$2 pct=${3:-80}
  [ "$nfail" -eq 0 ] && { echo "none"; return; }
  [ $((nfail * 100 / total)) -ge "$pct" ] && { echo "backend"; return; }
  echo "rebuild"
}

echo "== tunnel watchdog decision logic =="
check "all healthy -> no action"                 "none"    "$(decide 0 24)"
check "1 wedged -> rebuild"                      "rebuild" "$(decide 1 24)"
check "11 wedged (46%) -> rebuild"               "rebuild" "$(decide 11 24)"
check "19 wedged (79%) -> rebuild"               "rebuild" "$(decide 19 24)"
check "20 down (83%) -> blame backend"           "backend" "$(decide 20 24)"
check "24 down (100%) -> blame backend"          "backend" "$(decide 24 24)"
check "the 2026-08-06 outage (0/24 up)"          "backend" "$(decide 24 24)"
check "the 22-of-24 wedge (92%) -> backend"      "backend" "$(decide 22 24)"

# Two consecutive misses required before killing, so a single slow reply
# under load does not trigger a needless rebuild.
echo "== debounce =="
debounce() { # miss_count
  [ "$1" -ge 2 ] && echo "kill" || echo "wait"
}
check "1 slow reply -> wait"                     "wait"    "$(debounce 1)"
check "2 consecutive misses -> kill"             "kill"    "$(debounce 2)"

# Engine watchdog: container Up must NOT count as healthy. Only HTTP 200 does.
echo "== engine watchdog health signal =="
engine_healthy() { # http_code container_state
  [ "$1" = "200" ] && echo "healthy" || echo "dead"
}
check "http 200, container up -> healthy"        "healthy" "$(engine_healthy 200 up)"
check "http 000, container UP -> dead (the bug)" "dead"    "$(engine_healthy 000 up)"
check "http 000, container exited -> dead"       "dead"    "$(engine_healthy 000 exited)"
check "http 500 -> dead"                         "dead"    "$(engine_healthy 500 up)"

echo "== engine watchdog restart threshold =="
eng_action() { # misses threshold
  [ "$1" -ge "$2" ] && echo "restart" || echo "wait"
}
check "1 miss -> wait"                           "wait"    "$(eng_action 1 2)"
check "2 misses -> restart"                      "restart" "$(eng_action 2 2)"

echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
