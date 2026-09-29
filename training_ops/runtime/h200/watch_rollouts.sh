#!/usr/bin/env bash
set -euo pipefail

container=${1:?orchestrator container is required}
renderer=${2:?renderer path is required}

test -r "$renderer"
inspect_error=
container_id=
container_started_at=
container_running=
container_status=
container_exit_code=
container_oom_killed=
container_error=

inspect_container_state() {
  local raw
  if ! raw=$(sudo docker inspect --format '{{.Id}}|{{.State.StartedAt}}|{{.State.Running}}|{{.State.Status}}|{{.State.ExitCode}}|{{.State.OOMKilled}}|{{.State.Error}}' "$container" 2>&1); then
    inspect_error=$raw
    return 1
  fi
  IFS='|' read -r container_id container_started_at container_running \
    container_status container_exit_code container_oom_killed container_error <<<"$raw"
  if [ -z "$container_id" ] || [ -z "$container_started_at" ] || \
    [ -z "$container_running" ] || [ -z "$container_status" ] || \
    [ -z "$container_exit_code" ] || [ -z "$container_oom_killed" ]; then
    inspect_error="unexpected docker inspect output: $raw"
    return 1
  fi
}

print_container_state() {
  printf 'Rollout container state: id=%s started_at=%s status=%s exit_code=%s oom_killed=%s' \
    "$container_id" "$container_started_at" "$container_status" \
    "$container_exit_code" "$container_oom_killed"
  if [ -n "$container_error" ]; then
    printf ' error=%s' "$container_error"
  fi
  printf '\n'
}

is_clean_exit() {
  [ "$container_running" != true ] && [ "$container_status" = exited ] && \
    [ "$container_exit_code" = 0 ] && [ "$container_oom_killed" = false ] && \
    [ -z "$container_error" ]
}

same_container_instance() {
  [ "$container_id" = "$initial_container_id" ] && \
    [ "$container_started_at" = "$initial_container_started_at" ]
}

has_clean_completion_marker() {
  sudo docker logs --since "$initial_container_started_at" --tail 20000 "$container_id" 2>&1 |
    grep -aFq 'Orchestrator finished.'
}

show_saved_rollout() {
  runtime_dir=$(cd -- "$(dirname -- "$0")" && pwd)
  repository=$(cd -- "$runtime_dir/../../.." && pwd)
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
}

if ! inspect_container_state; then
  if [[ "$inspect_error" == *"No such object"* ]]; then
    show_saved_rollout
    exit 0
  fi
  printf 'Rollout container state unavailable: %s\n' "$inspect_error" >&2
  exit 1
fi
initial_container_id=$container_id
initial_container_started_at=$container_started_at

if [ "$container_running" != true ]; then
  print_container_state
  if is_clean_exit; then
    echo "Rollout collection completed normally."
    exit 0
  fi
  echo "Rollout collection stopped abnormally." >&2
  exit 1
fi

sudo docker exec "$container" /app/.venv/bin/python -c 'import tqdm'

stream_since=$(date -u +'%Y-%m-%dT%H:%M:%S.%NZ')
step_event=$(
  sudo docker logs --since "$initial_container_started_at" --tail 20000 "$container_id" 2>&1 |
    sed $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep -aE 'SUCCESS[[:space:]]+Step[[:space:]]+[0-9]+|Resuming orchestrator at rollout batch[[:space:]]+[0-9]+[[:space:]]+from policy v[0-9]+[[:space:]]*$' |
    tail -1 |
    sed -nE \
      -e 's/.*SUCCESS[[:space:]]+Step[[:space:]]+([0-9]+).*/success \1/p' \
      -e 's/.*Resuming orchestrator at rollout batch[[:space:]]+([0-9]+)[[:space:]]+from policy v[0-9]+[[:space:]]*$/resume \1/p' || true
)
collecting_step=1
case "$step_event" in
  success\ *) collecting_step=$(( ${step_event#success } + 1 )) ;;
  resume\ *) collecting_step=${step_event#resume } ;;
esac
latest_line=$(
  sudo docker logs --since "$initial_container_started_at" --tail 200 "$container_id" 2>&1 |
    sed $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep -aE 'PIPELINE_TELEMETRY[[:space:]]|Train batch|SUCCESS[[:space:]]+Step' |
    tail -1 || true
)

producer_status=0
renderer_status=0
set +e
{
  if [ -n "$latest_line" ]; then
    printf '%s\n' "$latest_line"
  fi
  sudo docker logs --follow --since "$stream_since" "$container" 2>&1 |
    sed -u $'s/\033\\[[0-9;]*[[:alpha:]]//g' |
    grep --line-buffered -aE 'PIPELINE_TELEMETRY[[:space:]]|Train batch|SUCCESS[[:space:]]+Step'
  producer_chain_statuses=("${PIPESTATUS[@]}")
  producer_chain_exit=0
  if [ "${producer_chain_statuses[0]:-1}" != 0 ]; then
    producer_chain_exit=${producer_chain_statuses[0]}
  elif [ "${producer_chain_statuses[1]:-1}" != 0 ]; then
    producer_chain_exit=${producer_chain_statuses[1]}
  elif [ "${producer_chain_statuses[2]:-1}" != 0 ] && \
    [ "${producer_chain_statuses[2]:-1}" != 1 ]; then
    producer_chain_exit=${producer_chain_statuses[2]}
  fi
  if [ "$producer_chain_exit" != 0 ]; then
    exit "$producer_chain_exit"
  fi
} |
  sudo docker exec -i "$container" \
    /app/.venv/bin/python -u "$renderer" "$collecting_step" 2>/dev/null
pipeline_statuses=("${PIPESTATUS[@]}")
set -e
producer_status=${pipeline_statuses[0]:-1}
renderer_status=${pipeline_statuses[1]:-1}

pipeline_failure=$producer_status
if [ "$pipeline_failure" = 0 ]; then
  pipeline_failure=$renderer_status
fi
if [ "$pipeline_failure" = 0 ]; then
  pipeline_failure=1
fi

if [ "$producer_status" != 0 ]; then
  printf 'Rollout producer failed with exit code %s; renderer exit code %s.\n' \
    "$producer_status" "$renderer_status" >&2
fi

if ! inspect_container_state; then
  printf 'Rollout container state unavailable after producer exit %s and renderer exit %s: %s\n' \
    "$producer_status" "$renderer_status" "$inspect_error" >&2
  exit "$pipeline_failure"
fi

print_container_state
if ! same_container_instance; then
  printf 'Rollout container identity changed during watch: initial_id=%s final_id=%s initial_started_at=%s final_started_at=%s\n' \
    "$initial_container_id" "$container_id" "$initial_container_started_at" \
    "$container_started_at" >&2
  exit "$pipeline_failure"
fi
if [ "$producer_status" = 0 ] && \
  { [ "$renderer_status" = 137 ] || [ "$renderer_status" = 0 ]; }; then
  if is_clean_exit && has_clean_completion_marker; then
    if ! inspect_container_state; then
      printf 'Rollout container state unavailable after completion marker: %s\n' \
        "$inspect_error" >&2
    else
      print_container_state
      if ! same_container_instance; then
        printf 'Rollout container identity changed after completion marker: initial_id=%s final_id=%s initial_started_at=%s final_started_at=%s\n' \
          "$initial_container_id" "$container_id" "$initial_container_started_at" \
          "$container_started_at" >&2
      elif is_clean_exit; then
        echo "Rollout collection completed normally."
        exit 0
      fi
    fi
  fi
fi
if [ "$producer_status" = 0 ] && [ "$renderer_status" = 0 ]; then
  echo "Rollout collection stopped before clean completion." >&2
else
  echo "Rollout collection stopped abnormally." >&2
fi
exit "$pipeline_failure"
