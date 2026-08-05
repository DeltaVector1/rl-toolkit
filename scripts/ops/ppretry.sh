#!/usr/bin/env bash
# PRIME_BATCH_CACHE (if exported) makes the trainer replay a saved batch
# instead of waiting for the rollout hosts -- retest costs a model load.
# ppretry.sh <RUN> -- restart only the trainer for the current PP test run.
#
# ONLY safe while the trainer has never completed a step. Once it has, the
# orchestrator has advanced its policy version and buffered batches that a
# restarted trainer will never receive: the orchestrator then pauses at its
# off-policy limit waiting for a policy the trainer cannot produce, and both
# sides wait forever. After a completed step, restart BOTH or start a new run.
#
# The orchestrator keeps running and keeps its already-collected batch buffered.
# That is safe precisely because a failing trainer never completes a step, so
# the orchestrator does not advance and the two stay in sync -- the mismatch
# that forces a fresh run directory only appears once a step lands. Saves the
# ~15 minutes of rollout collection per debug cycle.
set -uo pipefail
RUN=${1:?usage: ppretry.sh <RUN>}
NAME=run_$RUN
NODES="n76 n77 n78 n80 n82 n83 n84 n85"
TRAINER_NODES=${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}
DBG=${PRIME_STATS_DEBUG:-}
CAP=${PRIME_PP_MAX_MICRO:-}
# pre-gate cleanup: the gate must judge the node AFTER the old containers die,
# or zombie ranks from a dead attempt fail the availability check.
for h in $NODES; do
  ( ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" "sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1" ) &
done
wait
sleep 3

# swap-hangover gate: a node still paging back a previous incarnation's
# spill joins the rendezvous sluggish, stalls its ranks, and the NCCL
# watchdog kills the whole cohort ~30 min later. Refuse such nodes.
unfit=""
for h in $NODES; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" 'sw=$(free -g | sed -n 3p | awk "{print \$3}"); fr=$(free -g | sed -n 2p | awk "{print \$7}"); echo "$sw $fr"' 2>/dev/null)
  set -- $st
  sw=${1:-99}; fr=${2:-0}
  if [ "$sw" -gt 10 ] 2>/dev/null || [ "$fr" -lt 300 ] 2>/dev/null; then
    unfit="$unfit $h(swap=${sw}G,avail=${fr}G)"
  fi
done
if [ -n "$unfit" ]; then
  echo "UNFIT nodes:$unfit -- drain swap (swapoff/swapon) before launching" >&2
  exit 1
fi

fail=""
for h in $NODES; do
  ( ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" "sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1; \
      TRAINER_NODES='$TRAINER_NODES' \
      PRIME_STATS_DEBUG='$DBG' PRIME_PP_MAX_MICRO='$CAP' \
      RUN=$RUN RUN_NAME=$NAME CONFIG_TAG=.$RUN PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE='${PRIME_EP_AC_CACHE:-1}' \
      bash /data/glm52-rl/deployment/scripts/china/cluster.sh trainer >/dev/null 2>&1" \
    || echo "$h" >> /tmp/ppretry_failed.$$ ) &
done
wait
# A silent partial restart leaves torchrun waiting for nodes that never come
# (observed: 3/7 joined, rendezvous timeout after 901s). Verify every node.
sleep 5
for h in $NODES; do
  up=$(ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" 'sudo docker ps --filter name=glm52-decensor-trainer --format up | head -1' 2>/dev/null)
  [ "$up" = "up" ] || fail="$fail $h"
done
rm -f /tmp/ppretry_failed.$$
if [ -n "$fail" ]; then
  echo "FAILED to start trainer on:$fail -- rerun or investigate before expecting a step" >&2
  exit 1
fi
echo "trainer restarted for $RUN on all nodes (orchestrator left running)"
