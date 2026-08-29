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
printf '%s\n' "${SOURCE_RUN_DIR:?}/broadcasts/step_1/STABLE"
EOF
cat >"$tmp/bin/rsync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
source=${args[$#-2]}
destination=${args[$#-1]}
case "$source" in
  remote:*)
    mkdir -p "$(dirname "$destination")"
    : >"$destination"
    ;;
  *)
    printf 'unexpected rsync source: %s\n' "$source" >&2
    exit 1
    ;;
esac
EOF
cat >"$tmp/bin/curl" <<'EOF'
#!/usr/bin/env bash
sleep 10
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/rsync" "$tmp/bin/curl"

log="$tmp/sync.log"
SECONDS=0
PATH="$tmp/bin:$PATH" \
  SOURCE_RUN_DIR="$tmp/run" \
  DEST_HOST=remote \
  ADAPTER_SOURCE_BASE=http://source \
  ADAPTER_ADMIN_URLS=http://admin \
  ADAPTER_PREFETCH_TIMEOUT=1 \
  ADAPTER_REQUEST_TIMEOUT=1 \
  ONCE=1 \
  "$sync_script" >"$log" 2>&1
elapsed=$SECONDS

test -f "$tmp/run/broadcasts/step_1/STABLE"
test "$elapsed" -lt 3
sleep 2
grep -q 'adapter prefetch started' "$log"
grep -q 'adapter prefetch failed' "$log"
grep -q 'policy watcher will stage it' "$log"
printf 'stable marker delivered in %ss while prefetch failed asynchronously\n' "$elapsed"
