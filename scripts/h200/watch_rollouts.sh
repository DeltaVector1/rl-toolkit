#!/usr/bin/env bash
set -euo pipefail

# Foreground monitor for the edge orchestrator. Show one live tqdm rollout bar
# and one short line for each completed rollout step. The H200 host does not
# install Python packages, so use the tqdm package in the orchestrator image.
container=${1:-glm52-decensor-orchestrator}
since=${SINCE:-2m}
deployment=${DEPLOYMENT:-/data/glm52-rl/deployment}
renderer=$deployment/scripts/h200/watch_rollouts_tqdm.py

test -r "$renderer" || {
  echo "missing rollout progress renderer: $renderer" >&2
  exit 1
}

sudo docker exec "$container" /app/.venv/bin/python -c 'import tqdm' \
  >/dev/null 2>&1 || {
    echo "the orchestrator container is not running or does not contain tqdm" >&2
    exit 1
  }

# Read the current state once. Then follow only new messages. Replaying the
# complete SINCE window can briefly label an old batch with the current step.
stream_since=$(date -u +'%Y-%m-%dT%H:%M:%S.%NZ')
completed_step=$(
  sudo docker logs "$container" 2>&1 |
    sed $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep -aE 'SUCCESS[[:space:]]+Step' |
    tail -1 |
    sed -nE 's/.*SUCCESS[[:space:]]+Step[[:space:]]+([0-9]+).*/\1/p' || true
)
completed_step=${completed_step:-0}
collecting_step=$((completed_step + 1))
latest_batch=$(
  sudo docker logs --since "$since" "$container" 2>&1 |
    sed $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep -aE 'Train batch' |
    tail -1 || true
)

{
  if [ -n "$latest_batch" ]; then
    printf '%s\n' "$latest_batch"
  fi
  sudo docker logs --follow --since "$stream_since" "$container" 2>&1 |
    sed -u $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep --line-buffered -aE 'Train batch|SUCCESS[[:space:]]+Step'
} |
  sudo docker exec -i "$container" \
    /app/.venv/bin/python -u "$renderer" "$collecting_step" 2>/dev/null
