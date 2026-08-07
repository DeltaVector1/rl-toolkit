#!/usr/bin/env bash
# Wait until the trainer finishes a given step, then restart the orchestrator
# so it re-reads its config. Runs on n82.
#
# The orchestrator reads orchestrator.<tag>.toml once, at startup. Changing a
# value such as max_inflight_rollouts therefore needs a restart, and a restart
# discards whatever the current batch has collected. Restarting immediately
# after a step completes costs almost nothing, because the next batch has only
# just begun.
#
# This does not restart the trainer. The trainer keeps its checkpoint and its
# policy version, so the orchestrator resumes against a policy that exists.
# run_supervisor.sh performs the same orchestrator-only restart whenever the
# orchestrator dies, so this is a supported path rather than a new one.
set -uo pipefail

STEP=${STEP:?set STEP -- the step number to wait for}
TAG=${CONFIG_TAG:?set CONFIG_TAG}
RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
EXPECT=${EXPECT:-}          # optional "key = value" line to verify in the config
GIVE_UP_HOURS=${GIVE_UP_HOURS:-6}
INTERVAL=${INTERVAL:-60}
DEPLOY=/data/glm52-rl/deployment
CFG=$DEPLOY/configs/china/orchestrator${TAG}.toml
TRAINER_CT=glm52-decensor-trainer
ORCH_CT=glm52-decensor-orchestrator

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

log "waiting for trainer step $STEP, then restarting the orchestrator (tag=$TAG)"

# Refuse to arm if the change is not actually staged. Restarting for nothing
# would discard a batch and change no behaviour.
if [ -n "$EXPECT" ]; then
  if grep -qF -- "$EXPECT" "$CFG"; then
    log "config check ok: '$EXPECT' present in $CFG"
  else
    log "ABORT: '$EXPECT' is not in $CFG. Nothing to apply."
    exit 1
  fi
fi

deadline=$(( $(date +%s) + GIVE_UP_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if sudo docker logs "$TRAINER_CT" 2>&1 | strip | grep -qa "SUCCESS Step $STEP "; then
    log "trainer step $STEP is complete"
    break
  fi
  sleep "$INTERVAL"
done

if ! sudo docker logs "$TRAINER_CT" 2>&1 | strip | grep -qa "SUCCESS Step $STEP "; then
  log "ABORT: gave up after ${GIVE_UP_HOURS}h; step $STEP never completed. Nothing was restarted."
  exit 1
fi

before=$(sudo docker logs --tail 40 "$ORCH_CT" 2>&1 | grep -oE 'batch [0-9]+/[0-9]+' | tail -1)
log "orchestrator was at: ${before:-unknown}. Removing it now."
sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1

# Prefer the supervisor: it owns the restart and would otherwise race with us.
# Bracket the last token so this probe cannot match its own command line.
if pgrep -f 'bash .*[r]un_supervisor.sh' >/dev/null; then
  log "run supervisor is alive; it restarts the orchestrator within its interval"
  for _ in $(seq 1 20); do
    sleep 15
    [ "$(sudo docker ps --filter "name=$ORCH_CT" --format up | head -1)" = "up" ] && { log "orchestrator is up again"; break; }
  done
else
  log "no run supervisor; starting the orchestrator directly"
  RUN=$RUN RUN_NAME=$RUN_NAME CONFIG_TAG=$TAG bash "$DEPLOY/scripts/china/cluster.sh" orchestrator >/dev/null 2>&1
fi

sleep 60
if [ "$(sudo docker ps --filter "name=$ORCH_CT" --format up | head -1)" != "up" ]; then
  log "WARNING: the orchestrator is still not up. Start it by hand."
  exit 1
fi

# The one restart failure that cannot be repaired in place: the orchestrator
# waiting for a policy version the trainer cannot produce. The fix is a new run
# name, so say so loudly rather than pretend the restart worked.
if sudo docker logs --tail 200 "$ORCH_CT" 2>&1 | grep -qa "Pausing dispatcher"; then
  log "WARNING: 'Pausing dispatcher' in the orchestrator log. It expects a policy version"
  log "WARNING: the trainer cannot make. This needs a NEW RUN NAME; it cannot be repaired here."
fi

log "done. orchestrator restarted with $CFG"
sudo docker logs --tail 20 "$ORCH_CT" 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1
