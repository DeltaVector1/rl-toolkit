#!/usr/bin/env bash
# Measure rollout collection rate at one max_inflight_rollouts value.
# Runs from the operator workstation. One value per invocation.
#
#   VALUE=192 WARMUP=180 WINDOW=480 bash scripts/ops/sweep_inflight.sh
#
# Changing max_inflight_rollouts needs an orchestrator restart, and a restart
# discards whatever the current batch has collected, so each measurement costs
# one partial batch. Skip the warmup window: a fresh batch ramps from zero and
# the opening burst reads high.
#
# Rate is accepted rollouts per minute. Accepted advances in group_size steps,
# so a window shorter than a few groups quantizes badly -- keep WINDOW >= 480s.
set -uo pipefail

VALUE=${VALUE:?set VALUE, the max_inflight_rollouts to test}
# Warm up until the batch is really running, not for a fixed time. A restarted
# orchestrator starts at accepted=0, each rollout takes minutes, and acceptance
# is group-granular, so nothing is accepted for the first several minutes. A
# fixed 180s warmup opened the window at accepted=0 and measured the ramp: it
# read 2/min where the steady rate was 14/min.
MIN_ACCEPTED=${MIN_ACCEPTED:-64}
WARMUP_MAX=${WARMUP_MAX:-900}
WINDOW=${WINDOW:-480}
TAG=${CONFIG_TAG:-.x24}
HEAD=${HEAD:-n82}
DEPLOY=/data/glm52-rl/deployment
CFG=$DEPLOY/configs/china/orchestrator$TAG.toml
ORCH=glm52-decensor-orchestrator
RESULTS=${RESULTS:-/tmp/inflight_sweep.tsv}

say() { echo "$(date -u +%H:%M:%S) $*"; }
sh_() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HEAD" "$@"; }

accepted() {
  sh_ "sudo docker logs --tail 60 $ORCH 2>&1 \
       | grep -oE 'batch [0-9]+/[0-9]+' | tail -1 | grep -oE '[0-9]+' | head -1"
}
timeouts_since() { # minutes
  sh_ "sudo docker logs --since ${1}m $ORCH 2>&1 | grep -ca 'ReadTimeout' || true"
}
engine_running() {
  for h in h200a h200b; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" \
      "sudo docker logs --tail 30 glm52-decensor-h200 2>&1 \
       | grep -aoE 'Running: [0-9]+' | tail -1 | grep -oE '[0-9]+'" 2>/dev/null
  done | paste -sd+ - | bc 2>/dev/null || echo 0
}

# Measure the value already running, without paying for a restart. Use this
# when a previous run left the orchestrator at the value under test.
if [ "${SKIP_RESTART:-0}" = "1" ]; then
  say "SKIP_RESTART: measuring the running orchestrator at $VALUE"
  sh_ "grep -m1 max_inflight_rollouts $CFG"
else

say "setting max_inflight_rollouts = $VALUE"
sh_ "sudo sed -i 's/^max_inflight_rollouts = .*/max_inflight_rollouts = $VALUE/' $CFG && grep -m1 max_inflight_rollouts $CFG"

# Let run_supervisor.sh own the restart rather than racing it.
say "restarting the orchestrator"
sh_ "sudo docker rm -f $ORCH >/dev/null 2>&1; echo removed"
for _ in $(seq 1 24); do
  sleep 10
  [ "$(sh_ "sudo docker ps --filter name=$ORCH --format up | head -1")" = "up" ] && break
done
if [ "$(sh_ "sudo docker ps --filter name=$ORCH --format up | head -1")" != "up" ]; then
  say "ABORT: orchestrator did not come back"
  exit 1
fi
fi

say "warming up until accepted >= $MIN_ACCEPTED (max ${WARMUP_MAX}s)"
warm_start=$(date -u +%s)
while : ; do
  a=$(accepted); a=${a:-0}
  now=$(date -u +%s)
  [ "$a" -ge "$MIN_ACCEPTED" ] && { say "warm at accepted=$a after $((now - warm_start))s"; break; }
  if [ $((now - warm_start)) -ge "$WARMUP_MAX" ]; then
    say "WARNING: only accepted=$a after ${WARMUP_MAX}s; measuring anyway"
    break
  fi
  sleep 30
done

s=$(accepted); t0=$(date -u +%s)
say "window open at accepted=$s"
sleep "$WINDOW"
e=$(accepted); t1=$(date -u +%s)
run=$(engine_running)
to=$(timeouts_since $(( (WINDOW + 59) / 60 )) )

if [ -z "$s" ] || [ -z "$e" ] || [ "$e" -lt "$s" ]; then
  say "DISCARD: batch rolled over during the window (start=$s end=$e). Re-run."
  exit 2
fi

python3 - "$VALUE" "$s" "$e" "$((t1 - t0))" "$to" "$run" "$RESULTS" <<'PY'
import sys, pathlib
val, s, e, dt, to, run, out = sys.argv[1:]
s, e, dt, to, run = int(s), int(e), int(dt), int(to), int(run)
rate = (e - s) / (dt / 60)
line = f"{val}\t{e-s}\t{dt/60:.1f}\t{rate:.2f}\t{1024/rate if rate else 0:.0f}\t{to/(dt/60):.2f}\t{run}"
print(f"inflight={val}  accepted={e-s} in {dt/60:.1f}min  "
      f"rate={rate:.2f}/min  batch1024={1024/rate if rate else 0:.0f}min  "
      f"timeouts={to/(dt/60):.2f}/min  engines_running={run}")
p = pathlib.Path(out)
if not p.exists():
    p.write_text("inflight\taccepted\tminutes\trate_per_min\tbatch1024_min\ttimeouts_per_min\tengines_running\n")
with p.open("a") as f:
    f.write(line + "\n")
PY
