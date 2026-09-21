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
    format=${4:-}
    case "$format" in
      "{{.Name}}")
        if [ "${DOCKER_MODE:-}" = stable ] || [ "${DOCKER_MODE:-}" = immediate-exit ]; then
          printf '/orchestrator\n'
        else
          printf '/configured-inference\n'
        fi
        ;;
      "{{.Id}}") printf 'inference-container-id\n' ;;
      "{{.State.Running}}")
        if [ "${DOCKER_MODE:-}" = immediate-exit ]; then
          printf 'false\n'
        else
          printf 'true\n'
        fi
        ;;
      "{{.Image}}") printf '%s\n' "${EXPECTED_IMAGE:?}" ;;
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
    if [ "${DOCKER_MODE:-}" = run-failure ]; then
      printf 'daemon rejected orchestrator launch\n' >&2
      exit 42
    fi
    printf 'container-id\n'
    ;;
  logs)
    printf 'orchestrator exited: fatal startup error\n'
    ;;
  rm)
    printf 'rm\n' >>"${CALL_LOG:?}"
    ;;
  *)
    printf 'unexpected docker command: %s\n' "$*" >&2
    exit 1
    ;;
esac
EOF
chmod +x "$tmp/bin"/*

source <(sed -n '/^resolve_inference_image() {/,/^}$/p' "$launcher")
source <(sed -n '/^start_orchestrator_container() {/,/^}$/p' "$launcher")

nodes_file="$tmp/nodes.toml"
printf '%s\n' '[containers]' 'glm52_inference = "configured-inference"' >"$nodes_file"
export PATH="$tmp/bin:$PATH"
export CALL_LOG="$tmp/docker-calls"
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
grep -Fx 'ERROR: orchestrator container launch failed: orchestrator' "$tmp/run-failure.err"

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
test "$sleep_calls" -eq 19

printf 'orchestrator launch guard rejects missing image, run failure, and immediate exit; stable launch waits\n'
