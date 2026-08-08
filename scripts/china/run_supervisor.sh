#!/usr/bin/env bash
# Keep the trainer and orchestrator alive. Runs on n82.
# Bring the trainer back before the orchestrator. ZMQ can block an
# orchestrator that starts against a dead trainer.
set -uo pipefail

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
TAG=${CONFIG_TAG:?set CONFIG_TAG}
TRAINER_NODES=${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}
NODES=${NODES:-"n76 n77 n78 n80 n82 n83 n84 n85"}
INTERVAL=${INTERVAL:-120}
CLUSTER=/data/glm52-rl/deployment/scripts/china/cluster.sh

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
up() { sudo docker ps --filter "name=$1" --format up 2>/dev/null | head -1; }

log "run supervisor started (RUN=$RUN TAG=$TAG interval=${INTERVAL}s)"

while true; do
  if [ "$(up glm52-decensor-trainer)" != "up" ]; then
    log "TRAINER DOWN -- restarting across $NODES"
    for h in $NODES; do
      ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" \
        "sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1 || true" 2>/dev/null
    done
    sleep 20
    declare -A launches
    for h in $NODES; do
      ( ssh -o BatchMode=yes -o ConnectTimeout=30 "$h" \
          "TRAINER_NODES='$TRAINER_NODES' RUN=$RUN RUN_NAME=$RUN_NAME CONFIG_TAG=$TAG \
           PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE=0 bash $CLUSTER trainer" ) \
          >>"/data/glm52-rl/run-supervisor-$h.log" 2>&1 &
      launches[$h]=$!
    done
    failed=""
    for h in $NODES; do
      wait "${launches[$h]}" || failed="$failed $h"
    done
    if [ -n "$failed" ]; then
      log "trainer relaunch failed on:$failed"
    else
      log "trainer relaunch issued"
    fi
    sleep 60
  fi

  if [ "$(up glm52-decensor-orchestrator)" != "up" ]; then
    if [ "$(up glm52-decensor-trainer)" = "up" ]; then
      log "ORCHESTRATOR DOWN -- restarting (trainer is up)"
      RUN=$RUN RUN_NAME=$RUN_NAME CONFIG_TAG=$TAG bash "$CLUSTER" orchestrator >/dev/null 2>&1
      log "orchestrator relaunch issued"
    else
      log "orchestrator down but trainer not up yet; waiting"
    fi
  fi

  sleep "$INTERVAL"
done
