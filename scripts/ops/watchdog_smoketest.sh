#!/usr/bin/env bash
# Drive the REAL watchdog scripts against a fake environment and assert on what
# they actually do. Nothing real is touched: curl, ss, kill, sudo, docker,
# nvidia-smi and sleep are all shadowed on PATH.
#
# The previous version of this file re-implemented each watchdog's decision as
# a local shell function and tested THAT. It passed 18 of 18 while asserting
# the opposite of the shipped behaviour: three cases still expected the tunnel
# watchdog to refuse to rebuild on mass failure, a rule that was removed after
# it froze the run for five hours. A test that mirrors the logic cannot catch a
# change to the logic. Every case below runs the real script.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TUNNEL_WD=$ROOT/scripts/china/tunnel_watchdog.sh
ENGINE_WD=$ROOT/scripts/h200/engine_watchdog.sh

pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1))
  else echo "  FAIL  $1 (expected '$2', got '$3')"; fail=$((fail+1)); fi
}
contains() { # name haystack-file needle
  if grep -q -- "$3" "$2" 2>/dev/null; then echo "  PASS  $1"; pass=$((pass+1))
  else echo "  FAIL  $1 (log has no '$3')"; fail=$((fail+1)); fi
}
absent() { # name haystack-file needle
  if grep -q -- "$3" "$2" 2>/dev/null; then echo "  FAIL  $1 (log unexpectedly has '$3')"; fail=$((fail+1))
  else echo "  PASS  $1"; pass=$((pass+1)); fi
}

# ---------------------------------------------------------------- fake world

FIX=$(mktemp -d)
trap 'rm -rf "$FIX"' EXIT
BIN=$FIX/bin; mkdir -p "$BIN"

# curl: hands the URL to the per-case responder and prints only an HTTP code,
# which is all these scripts ask of it (-o /dev/null -w '%{http_code}').
cat > "$BIN/curl" <<'EOF'
#!/usr/bin/env bash
url=""
for a in "$@"; do case "$a" in http*) url=$a ;; esac; done
port=$(printf '%s' "$url" | sed -n 's|.*://[^:]*:\([0-9]*\)/.*|\1|p')
kind=tunnel
case "$url" in *completions*) kind=engine ;; esac
echo "$kind $port" >> "$FIX/probed"
# shellcheck disable=SC1090
. "$FIX/respond.sh"
respond "$kind" "$port"
EOF

# ss: reports which pid holds a listening port. Ports listed in $FIX/unheld
# have nothing behind them, which is the "remote supervisor down" case.
cat > "$BIN/ss" <<'EOF'
#!/usr/bin/env bash
port=$(printf '%s' "$*" | sed -n 's/.*sport = :\([0-9]*\).*/\1/p')
grep -qx "$port" "$FIX/unheld" 2>/dev/null && exit 0
echo "LISTEN 0 128 127.0.0.1:$port 0.0.0.0:* users:((\"sshd\",pid=$((9000+port%1000)),fd=9))"
EOF

cat > "$BIN/kill" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in [0-9]*) echo "$a" >> "$FIX/killed" ;; esac; done
EOF

cat > "$BIN/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF

cat > "$BIN/docker" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$FIX/docker"
EOF

cat > "$BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "1 MiB"
EOF

# sleep: counts loop iterations and stops the script under test after MAXCYCLES,
# so each case runs an exact, known number of cycles instead of a timeout race.
cat > "$BIN/sleep" <<'EOF'
#!/usr/bin/env bash
n=$(( $(cat "$FIX/cycles" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$FIX/cycles"
[ "$n" -ge "${MAXCYCLES:-2}" ] && kill -TERM "$PPID" 2>/dev/null
exit 0
EOF

chmod +x "$BIN"/*
export FIX PATH="$BIN:$PATH"

reset() { : > "$FIX/probed"; : > "$FIX/killed"; : > "$FIX/docker"; : > "$FIX/unheld"; rm -f "$FIX/cycles"; }
nkilled() { wc -l < "$FIX/killed" | tr -d ' '; }

# ------------------------------------------------------- tunnel watchdog

run_tunnel() { # ports... ; needs $FIX/respond.sh and $MAXCYCLES
  # The brace group swallows the shell's "Terminated" notice: the fake sleep
  # ends each case by signalling the script under test, and the reporting shell
  # is this one, not the child.
  { PORTS="$*" TIMEOUT=1 INTERVAL=1 HEARTBEAT="${HEARTBEAT:-1}" MASS_FAILURE_PCT=80 \
      timeout 30 bash "$TUNNEL_WD" > "$FIX/log" 2>&1; } 2>/dev/null
}
P24=$(seq 18001 18024 | tr '\n' ' ')

echo "== tunnel watchdog: real script, faked probes =="

reset; echo 'respond() { echo 200; }' > "$FIX/respond.sh"
MAXCYCLES=2 run_tunnel $P24
check "24 healthy -> nothing killed"              "0"  "$(nkilled)"
contains "24 healthy -> heartbeat proves it ran"  "$FIX/log" "heartbeat: 24/24 tunnels healthy"

reset; echo 'respond() { [ "$2" = 18007 ] && echo 000 || echo 200; }' > "$FIX/respond.sh"
MAXCYCLES=2 run_tunnel $P24
check "1 wedged, 2 cycles -> exactly 1 kill"      "1"  "$(nkilled)"
contains "1 wedged -> names the port"             "$FIX/log" "killed wedged tunnel on :18007"

# The regression that matters. An earlier rule refused to rebuild above 80%
# failure; at batch 1024 all 24 wedged with healthy engines and the run sat
# idle five hours. Mass failure must still rebuild.
reset; echo 'respond() { echo 000; }' > "$FIX/respond.sh"
MAXCYCLES=2 run_tunnel $P24
check "24 wedged -> rebuilds ALL of them"         "24" "$(nkilled)"
contains "24 wedged -> warns about the backend"   "$FIX/log" "check the engines too, but rebuilding regardless"
absent  "24 wedged -> does not refuse to act"     "$FIX/log" "not rebuilding"

reset; echo 'respond() { case "$2" in 18001|18002) echo 200 ;; *) echo 000 ;; esac; }' > "$FIX/respond.sh"
MAXCYCLES=2 run_tunnel $P24
check "the 22-of-24 wedge -> 22 kills"            "22" "$(nkilled)"

# Debounce: one slow reply under load must not cost a rebuild.
reset
cat > "$FIX/respond.sh" <<'EOF'
respond() {
  c=$(cat "$FIX/cycles" 2>/dev/null || echo 0)
  if [ "$2" = 18007 ] && [ "$c" = 0 ]; then echo 000; else echo 200; fi
}
EOF
MAXCYCLES=3 run_tunnel $P24
check "1 miss then healthy -> no kill"            "0"  "$(nkilled)"

reset; echo 'respond() { [ "$2" = 18007 ] && echo 000 || echo 200; }' > "$FIX/respond.sh"
echo 18007 > "$FIX/unheld"
MAXCYCLES=2 run_tunnel $P24
check "wedged with nothing on the port -> no kill" "0" "$(nkilled)"
contains "wedged, unheld -> says so"              "$FIX/log" "nothing holds the port"

# The batch tunnel carries the zmq stream to the trainer. Killing it under a
# live orchestrator left seven dead connections and starved the trainer for 27
# minutes, so this watchdog must never probe or touch it.
reset; echo 'respond() { echo 000; }' > "$FIX/respond.sh"
MAXCYCLES=2 run_tunnel $P24
absent "batch port 29620 is never probed"         "$FIX/probed" "29620"
absent "batch port 29620 is never killed"         "$FIX/killed" "29620"

# ------------------------------------------------------- engine watchdog

run_engine() { # needs $FIX/respond.sh and $MAXCYCLES
  { PORT=8000 INTERVAL=1 TIMEOUT=1 FAILS_BEFORE_RESTART=2 \
    START="$FIX/fake_start.sh" RELOAD_TRIES=2 RELOAD_WAIT=1 \
      timeout 30 bash "$ENGINE_WD" > "$FIX/log" 2>&1; } 2>/dev/null
}
printf '#!/usr/bin/env bash\necho started >> "$FIX/docker"\n' > "$FIX/fake_start.sh"
chmod +x "$FIX/fake_start.sh"

echo "== engine watchdog: real script, faked engine =="

reset; echo 'respond() { echo 200; }' > "$FIX/respond.sh"
MAXCYCLES=3 run_engine
absent "generation ok -> no restart"              "$FIX/log" "RESTARTING ENGINE"
contains "generation ok -> probes a completion"   "$FIX/probed" "engine 8000"

# The 2026-08-06 06:05 failure on h200a: /v1/models answered 200 while the
# EngineCore was hung at "Running: 64, throughput: 0.0". Container state and
# /v1/models are both useless here; only a generation proves anything.
reset; echo 'respond() { [ "$1" = engine ] && echo 000 || echo 200; }' > "$FIX/respond.sh"
MAXCYCLES=3 run_engine
contains "gen 000 while models 200 -> restarts"   "$FIX/log" "RESTARTING ENGINE"
contains "restart removes the container first"    "$FIX/docker" "rm -f glm52-decensor-h200"
contains "restart runs the launcher"              "$FIX/docker" "started"

reset; echo 'respond() { [ "$1" = engine ] && echo 500 || echo 200; }' > "$FIX/respond.sh"
MAXCYCLES=3 run_engine
contains "gen 500 -> restarts"                    "$FIX/log" "RESTARTING ENGINE"

reset
cat > "$FIX/respond.sh" <<'EOF'
respond() {
  c=$(cat "$FIX/cycles" 2>/dev/null || echo 0)
  if [ "$1" = engine ] && [ "$c" = 0 ]; then echo 000; else echo 200; fi
}
EOF
MAXCYCLES=3 run_engine
absent  "1 miss then recovery -> no restart"      "$FIX/log" "RESTARTING ENGINE"
contains "1 miss then recovery -> logs recovery"  "$FIX/log" "engine recovered"

# After a restart it must confirm with a generation, not with /v1/models.
reset
cat > "$FIX/respond.sh" <<'EOF'
respond() {
  n=$(( $(cat "$FIX/n" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FIX/n"
  [ "$n" -le 2 ] && echo 000 || echo 200
}
EOF
rm -f "$FIX/n"
MAXCYCLES=4 run_engine
contains "post-restart health is a generation"    "$FIX/log" "engine back up (generation verified)"

# ------------------------------------------- liveness probes must not self-match

# `pgrep -f X` run through `ssh host "..."` or `bash -c "..."` matches the very
# shell running it, because X sits in that command line. The count is then never
# zero and every liveness test passes. `pgrep -fc` has the same fault plus one:
# it counts its own grep. This hid three dead watchdogs for 3 hours, and killed
# all three again on 2026-08-07 when a `pkill -f` matched its own ssh session.
#
# Bracketing the FIRST word is not enough. The shell really is `bash`, so
# `[b]ash .*name.sh` still matches through the literal leading `bash ` and the
# `.*`. The distinctive last token is what must be bracketed.
echo "== liveness probes must not match themselves =="

naive=$(bash -c 'pgrep -f "bash .*no_such_proc_zzz.sh" | wc -l' | tr -d ' ')
check "the naive pattern matches its own shell"  "1" "$naive"

first=$(bash -c 'pgrep -f "[b]ash .*no_such_proc_zzz.sh" | wc -l' | tr -d ' ')
check "bracketing the first word does not help"  "1" "$first"

fixed=$(bash -c 'pgrep -f "bash .*[n]o_such_proc_zzz.sh" | wc -l' | tr -d ' ')
check "bracketing the last token works"          "0" "$fixed"

if command -v python3 >/dev/null; then
  helper=$(PYTHONPATH="$ROOT/.agents/skills/glm52-ops/src" python3 -c \
    'import glm52_ops; print(glm52_ops._no_self_match("bash .*tunnel_watchdog.sh"))' 2>/dev/null)
  check "the skill helper brackets the last token" "bash .*[t]unnel_watchdog.sh" "$helper"
fi

echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
