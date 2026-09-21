#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$repo/training_ops/runtime/h200/start_orchestrator.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
cat >"$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail

case "${1:-}" in
  inspect)
    printf 'inspect %s\n' "${2:-}" >>"${CALL_LOG:?}"
    target=${2:-}
    format=${4:-}
    case "$format" in
      "{{.Id}}")
        if [ "$target" = orchestrator ] && [ -s "${CONTAINER_STATE_FILE:?}" ]; then
          cat "$CONTAINER_STATE_FILE"
        else
          printf 'inference-container-id\n'
        fi
        ;;
      "{{.Name}}")
        case "$target" in
          container-id|failed-container-id|unrelated-container-id) printf '/orchestrator\n' ;;
          *) printf '/configured-inference\n' ;;
        esac
        ;;
      '{{index .Config.Labels "com.glm52.orchestrator.launch"}}')
        if [ "${DOCKER_MODE:-}" = unrelated ]; then
          printf 'run/run_run/different-nonce\n'
        elif [ -s "${CONTAINER_LABEL_FILE:?}" ]; then
          cat "$CONTAINER_LABEL_FILE"
        else
          printf 'run/run_run/unknown\n'
        fi
        ;;
      '{{index .Config.Labels "com.glm52.adapter-relay.source-url"}}')
        printf 'http://127.0.0.1:18100\n'
        ;;
      "{{.State.Running}}")
        if [ "${DOCKER_MODE:-}" = immediate-exit ]; then
          printf 'false\n'
        else
          printf 'true\n'
        fi
        ;;
      "{{.Image}}")
        if [ "${DOCKER_MODE:-}" = relay-wrong-image ] && [ "$target" = relay ]; then
          printf '%s\n' sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        elif [ "$target" = container-id ] || [ "$target" = failed-container-id ] || [ "$target" = unrelated-container-id ]; then
          printf '%s\n' "${EXPECTED_ORCHESTRATOR_IMAGE:?}"
        else
          printf '%s\n' "${EXPECTED_IMAGE:?}"
        fi
        ;;
      "{{.State.Status}}")
        if [ "${DOCKER_MODE:-}" = immediate-exit ]; then
          printf 'exited\n'
        else
          printf 'running\n'
        fi
        ;;
      'orchestrator state: {{json .State}}') printf 'orchestrator state: {"Status":"exited","ExitCode":1}\n' ;;
      *) printf 'unexpected inspect format: %s\n' "$format" >&2; exit 1 ;;
    esac
    ;;
  image)
    [ "${2:-}" = inspect ] || exit 1
    if [ "${3:-}" = --format ]; then
      [ "${4:-}" = '{{.Id}}' ] || exit 1
      [ "${5:-}" = configured-orchestrator ] || exit 1
      [ "${DOCKER_MODE:-}" != missing-orchestrator ] || exit 1
      if [ "${DOCKER_MODE:-}" = invalid-orchestrator-id ]; then
        printf 'not-an-image-id\n'
      else
        printf '%s\n' "${EXPECTED_ORCHESTRATOR_IMAGE:?}"
      fi
    else
      target=${3:-}
      if [ "$target" = "${EXPECTED_IMAGE:?}" ]; then
        [ "${DOCKER_MODE:-}" != missing-image ]
      elif [ "$target" = "${EXPECTED_ORCHESTRATOR_IMAGE:?}" ]; then
        [ "${DOCKER_MODE:-}" != missing-orchestrator-id ]
      else
        exit 1
      fi
    fi
    ;;
  run)
    run_args="$*"
    launch_label=
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --label ] && [ "$2" != '' ]; then
        launch_label=${2#com.glm52.orchestrator.launch=}
      fi
      shift
    done
    printf 'run %s\n' "$run_args" >>"${CALL_LOG:?}"
    if [ -n "$launch_label" ]; then
      printf '%s\n' "$launch_label" >"${CONTAINER_LABEL_FILE:?}"
      printf '%s\n' "$launch_label" >>"${LAUNCH_LABEL_LOG:?}"
    fi
    if [ "${DOCKER_MODE:-}" = run-failure ]; then
      printf 'failed-container-id\n' >"${CONTAINER_STATE_FILE:?}"
      printf 'daemon rejected orchestrator launch\n' >&2
      exit 42
    fi
    if [ "${DOCKER_MODE:-}" = empty-id ] || [ "${DOCKER_MODE:-}" = unrelated ]; then
      if [ "${DOCKER_MODE:-}" = empty-id ]; then
        printf 'failed-container-id\n' >"${CONTAINER_STATE_FILE:?}"
      else
        printf 'unrelated-container-id\n' >"${CONTAINER_STATE_FILE:?}"
        printf 'run/run_run/different-nonce\n' >"${CONTAINER_LABEL_FILE:?}"
      fi
      exit 0
    fi
    printf 'container-id\n' >"${CONTAINER_STATE_FILE:?}"
    printf 'container-id\n'
    ;;
  logs)
    printf 'orchestrator exited: fatal startup error\n'
    ;;
  rm)
    printf 'rm %s\n' "${3:-}" >>"${CALL_LOG:?}"
    if [ -s "${CONTAINER_STATE_FILE:?}" ] && [ "$(cat "$CONTAINER_STATE_FILE")" = "${3:-}" -o "${3:-}" = orchestrator ]; then
      : >"$CONTAINER_STATE_FILE"
      : >"$CONTAINER_LABEL_FILE"
    fi
    ;;
  *)
    printf 'unexpected docker command: %s\n' "$*" >&2
    exit 1
    ;;
esac
EOF
chmod +x "$tmp/bin"/*

source <(sed -n '/^resolve_inference_image() {/,/^}$/p' "$launcher")
source <(sed -n '/^resolve_orchestrator_image() {/,/^}$/p' "$launcher")
source <(sed -n '/^resolve_orchestrator_image_for_launch() {/,/^}$/p' "$launcher")
source <(sed -n '/^remove_failed_orchestrator() {/,/^}$/p' "$launcher")
source <(sed -n '/^start_orchestrator_container() {/,/^}$/p' "$launcher")
source <(sed -n '/^adapter_relay_start() {/,/^}$/p' "$launcher")

nodes_file="$tmp/nodes.toml"
printf '%s\n' '[containers]' 'glm52_inference = "configured-inference"' \
  'orchestrator_image = "configured-orchestrator"' >"$nodes_file"
export PATH="$tmp/bin:$PATH"
export CALL_LOG="$tmp/docker-calls"
export CONTAINER_STATE_FILE="$tmp/container-state"
export CONTAINER_LABEL_FILE="$tmp/container-label"
export LAUNCH_LABEL_LOG="$tmp/launch-labels"
export EXPECTED_IMAGE=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
export EXPECTED_ORCHESTRATOR_IMAGE=sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
: >"$CALL_LOG"

inference_state=$(DOCKER_MODE=ready resolve_inference_image)
read -r inference_id image <<<"$inference_state"
test "$inference_id" = inference-container-id
test "$image" = "$EXPECTED_IMAGE"

set +e
missing_output=$(DOCKER_MODE=missing-image resolve_inference_image 2>"$tmp/missing-image.err")
missing_status=$?
set -e
test "$missing_status" -ne 0
test -z "$missing_output"
grep -Fx "configured GLM inference image is missing: $EXPECTED_IMAGE" "$tmp/missing-image.err"

orchestrator_state=$(DOCKER_MODE=ready resolve_orchestrator_image)
read -r orchestrator_reference orchestrator_image_id <<<"$orchestrator_state"
test "$orchestrator_reference" = configured-orchestrator
test "$orchestrator_image_id" = "$EXPECTED_ORCHESTRATOR_IMAGE"

set +e
missing_output=$(DOCKER_MODE=missing-orchestrator resolve_orchestrator_image 2>"$tmp/missing-orchestrator.err")
missing_status=$?
set -e
test "$missing_status" -ne 0
test -z "$missing_output"
grep -Fx 'configured orchestrator image is missing: configured-orchestrator' "$tmp/missing-orchestrator.err"

set +e
invalid_output=$(DOCKER_MODE=invalid-orchestrator-id resolve_orchestrator_image 2>"$tmp/invalid-orchestrator.err")
invalid_status=$?
set -e
test "$invalid_status" -ne 0
test -z "$invalid_output"
grep -Fx 'configured orchestrator image has an invalid immutable id: not-an-image-id' "$tmp/invalid-orchestrator.err"

nodes_without_orchestrator_image="$tmp/nodes-without-orchestrator-image.toml"
printf '%s\n' '[containers]' 'glm52_inference = "configured-inference"' \
  >"$nodes_without_orchestrator_image"
nodes_file=$nodes_without_orchestrator_image
tunnels_only=1
unset orchestrator_image_reference orchestrator_image_id
resolve_orchestrator_image_for_launch
test -z "${orchestrator_image_reference:-}"
test -z "${orchestrator_image_id:-}"

tunnels_only=0
set +e
launch_output=$(resolve_orchestrator_image_for_launch 2>"$tmp/full-launch-image.err")
launch_status=$?
set -e
test "$launch_status" -ne 0
test -z "$launch_output"
grep -Fx 'cannot read the configured orchestrator image' "$tmp/full-launch-image.err"
nodes_file="$tmp/nodes.toml"

deployment=/fixture
RUN=run
RUN_NAME=run_run
ORCH_CT=orchestrator
image=$EXPECTED_IMAGE
orchestrator_image_id=$EXPECTED_ORCHESTRATOR_IMAGE
config_file=/fixture/orchestrator.toml

set +e
export DOCKER_MODE=run-failure
start_output=$(start_orchestrator_container 2>"$tmp/run-failure.err")
run_status=$?
set -e
test "$run_status" -ne 0
test -z "$start_output"
grep -Fx 'daemon rejected orchestrator launch' "$tmp/run-failure.err"
grep -Fx 'ERROR: orchestrator container launch failed: orchestrator' "$tmp/run-failure.err"
grep -Fx 'rm failed-container-id' "$CALL_LOG"
test ! -s "$CONTAINER_STATE_FILE"

set +e
export DOCKER_MODE=empty-id
start_output=$(start_orchestrator_container 2>"$tmp/empty-id.err")
empty_status=$?
set -e
test "$empty_status" -ne 0
test -z "$start_output"
grep -Fx 'ERROR: orchestrator container launch returned no container id: orchestrator' "$tmp/empty-id.err"
grep -Fx 'rm failed-container-id' "$CALL_LOG"
test ! -s "$CONTAINER_STATE_FILE"
test "$(sed -n '1p' "$LAUNCH_LABEL_LOG")" != "$(sed -n '2p' "$LAUNCH_LABEL_LOG")"
grep -Eq '^run/run_run/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' "$LAUNCH_LABEL_LOG"

set +e
export DOCKER_MODE=unrelated
start_output=$(start_orchestrator_container 2>"$tmp/unrelated.err")
unrelated_status=$?
set -e
test "$unrelated_status" -ne 0
test -z "$start_output"
grep -Fx 'ERROR: orchestrator container launch returned no container id: orchestrator' "$tmp/unrelated.err"
! grep -Fx 'rm unrelated-container-id' "$CALL_LOG"
grep -Fx 'unrelated-container-id' "$CONTAINER_STATE_FILE"
grep -Fx 'run/run_run/different-nonce' "$CONTAINER_LABEL_FILE"

set +e
export DOCKER_MODE=immediate-exit
start_output=$(start_orchestrator_container 2>"$tmp/immediate-exit.err")
exit_status=$?
set -e
test "$exit_status" -ne 0
test -z "$start_output"
grep -Fx 'ERROR: orchestrator container is not running: orchestrator' "$tmp/immediate-exit.err"
grep -F 'orchestrator state: {"Status":"exited","ExitCode":1}' "$tmp/immediate-exit.err"
grep -Fx 'orchestrator exited: fatal startup error' "$tmp/immediate-exit.err"
grep -Fx 'inspect container-id' "$CALL_LOG"

sleep_calls=0
sleep() { sleep_calls=$((sleep_calls + 1)); }
export DOCKER_MODE=stable
start_orchestrator_container
test "$sleep_calls" -eq 20
grep -F -- "$EXPECTED_ORCHESTRATOR_IMAGE" "$CALL_LOG"

peer_enabled=1
tunnels_only=0
relay_container=relay
relay_cache_dir="$tmp/relay-cache"
relay_source_url=http://127.0.0.1:18100
relay_port=18228
lora_source_ports=(18100)
curl() { printf 'source URL is not configured\n404'; }
ss() { printf 'LISTEN 0 128 127.0.0.1:18228 0.0.0.0:*\n'; }
export DOCKER_MODE=relay-wrong-image
: >"$CALL_LOG"
adapter_relay_start
grep -Fx 'rm relay' "$CALL_LOG"
grep -F 'run -d --name relay --pull=never' "$CALL_LOG"
grep -F -- "$EXPECTED_IMAGE" "$CALL_LOG"
! grep -F -- "$EXPECTED_ORCHESTRATOR_IMAGE" "$CALL_LOG"

printf 'orchestrator launch guard checks immutable images, safe cleanup, immediate exit, relay reuse, and a 10s stable window\n'
