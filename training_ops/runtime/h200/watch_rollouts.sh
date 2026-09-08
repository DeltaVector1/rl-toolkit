#!/usr/bin/env bash
set -euo pipefail

container=${1:?orchestrator container is required}
renderer=${2:?renderer path is required}
since=${SINCE:-2m}

test -r "$renderer"
if ! sudo docker inspect "$container" >/dev/null 2>&1; then
  runtime_dir=$(cd -- "$(dirname -- "$0")" && pwd)
  repository=$(cd -- "$runtime_dir/../../../../.." && pwd)
  python3 - "$repository/configs/nodes.toml" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

with open(sys.argv[1], "rb") as stream:
    run_root = Path(tomllib.load(stream)["paths"]["runs"])
files = list(run_root.glob("*/*/rollout-metrics.jsonl"))
if not files:
    print("Rollout collection is not running. No saved rollout step was found.")
else:
    path = max(files, key=lambda item: item.stat().st_mtime)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    step = json.loads(lines[-1]).get("step", "unknown") if lines else "unknown"
    print(f"Rollout collection is not running. Last saved rollout step: {step}.")
print("Run training_ops.status() to confirm completion or failure.")
PY
  exit 0
fi
if [ "$(sudo docker inspect --format '{{.State.Running}}' "$container")" != true ]; then
  echo "Rollout collection stopped. Run training_ops.status() for the exact result."
  exit 0
fi
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
