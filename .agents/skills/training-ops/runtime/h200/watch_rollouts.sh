#!/usr/bin/env bash
set -euo pipefail

container=${1:?orchestrator container is required}
renderer=${2:?renderer path is required}
since=${SINCE:-2m}

test -r "$renderer"
sudo docker exec "$container" /app/.venv/bin/python -c 'import tqdm'

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
