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
          printf 'other/run\n'
        else
          printf 'run/run_run\n'
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
    [ "${DOCKER_MODE:-}" != missing-image ]
    ;;
  run)
    printf 'run %s\n' "$*" >>"${CALL_LOG:?}"
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
source <(sed -n '/^remove_failed_orchestrator() {/,/^}$/p' "$launcher")
source <(sed -n '/^start_orchestrator_container() {/,/^}$/p' "$launcher")
source <(sed -n '/^adapter_relay_start() {/,/^}$/p' "$launcher")

nodes_file="$tmp/nodes.toml"
printf '%s\n' '[containers]' 'glm52_inference = "configured-inference"' >"$nodes_file"
export PATH="$tmp/bin:$PATH"
export CALL_LOG="$tmp/docker-calls"
export CONTAINER_STATE_FILE="$tmp/container-state"
export EXPECTED_IMAGE=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
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

deployment=/fixture
RUN=run
RUN_NAME=run_run
ORCH_CT=orchestrator
image=$EXPECTED_IMAGE
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

printf 'orchestrator launch guard checks immutable images, safe cleanup, immediate exit, relay reuse, and a 10s stable window\n'
