#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
sync_script="$root/.agents/skills/training-ops/runtime/h200/sync_rollout_batches.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
command=${args[$#-1]}
if [ "${SYNC_TEST_MODE:-}" = statusfailure ] && [[ "$command" == *"-name 'part-*'"* ]]; then
  exit 1
fi
bash -c "$command"
EOF
cat >"$tmp/bin/rsync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
source=${args[$#-2]}
destination=${args[$#-1]#remote:}
mkdir -p "$(dirname "$destination")"
if [ "${SYNC_TEST_MODE:-}" = interrupted ] && [ ! -e "$FAIL_MARKER" ]; then
  (set -C; : >"$FAIL_MARKER") 2>/dev/null || true
  exit 1
fi
exec 9>"$ACTIVE_LOCK"; flock 9
active=$(sed -n '1p' "$ACTIVE_STATE"); max=$(sed -n '2p' "$ACTIVE_STATE")
active=$((active + 1)); [ "$active" -gt "$max" ] && max=$active
printf '%s\n%s\n' "$active" "$max" >"$ACTIVE_STATE"; flock -u 9
sleep 0.05; cp "$source" "$destination"
flock 9; active=$(sed -n '1p' "$ACTIVE_STATE")
printf '%s\n%s\n' "$((active - 1))" "$max" >"$ACTIVE_STATE"; flock -u 9
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/rsync"

run_case() {
  local mode=$1 source_dir=$2 remote_dir=$3 log=$4
  mkdir -p "$source_dir/rollouts/step_1"
  head -c 256 /dev/zero >"$source_dir/rollouts/step_1/train_rollouts.bin"
  printf '0\n0\n' >"$tmp/active-state"
  : >"$tmp/active-lock"
  rm -f -- "$tmp/fail-marker"
  PATH="$tmp/bin:$PATH" \
    SOURCE_RUN_DIR="$source_dir" \
    DEST_RUN_DIR="$remote_dir" \
    DEST_HOST=remote \
    PARALLEL_THRESHOLD=1 \
    parallel_active_limit=4 \
    PARALLEL_LAUNCH_DELAY=0 \
    STRIPE_ATTEMPT_TIMEOUT=2 \
    SYNC_TEST_MODE="$mode" \
    ACTIVE_STATE="$tmp/active-state" \
    ACTIVE_LOCK="$tmp/active-lock" \
    FAIL_MARKER="$tmp/fail-marker" \
    ONCE=1 \
    "$sync_script" >"$log" 2>&1
}

success_source="$tmp/success-source"
success_remote="$tmp/success-remote"
run_case success "$success_source" "$success_remote" "$tmp/success.log"
test -f "$success_remote/rollouts/step_1/train_rollouts.bin"
test -f "$success_source/rollouts/step_1/train_rollouts.bin.delivered"
test "$(sed -n '2p' "$tmp/active-state")" -le 4

run_case success "$success_source" "$success_remote" "$tmp/resume.log"
test "$(wc -l <"$tmp/resume.log")" -lt 2
test "$(sed -n '2p' "$tmp/active-state")" -eq 0

run_case interrupted "$tmp/interrupted-source" "$tmp/interrupted-remote" "$tmp/interrupted.log"
test -f "$tmp/interrupted-remote/rollouts/step_1/train_rollouts.bin"
grep -q 'parallel transfer attempt 1 incomplete' "$tmp/interrupted.log"

run_case statusfailure "$tmp/status-source" "$tmp/status-remote" "$tmp/status.log"
! test -e "$tmp/status-source/rollouts/step_1/train_rollouts.bin.delivered"
grep -q 'cannot read remote part sizes' "$tmp/status.log"
! test -e "$tmp/status-remote/rollouts/step_1/train_rollouts.bin"

printf 'rollout transfer success, resume, interrupted retry, status failure, and max-active bound pass\n'
