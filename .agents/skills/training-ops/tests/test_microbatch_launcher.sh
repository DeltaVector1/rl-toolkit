#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cluster="$root/.agents/skills/training-ops/runtime/china/cluster.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/storage"
cat >"$tmp/bin/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'filesystem %s 0\n' "${TEST_TRAINER_OUTPUT:?}"
EOF
cat >"$tmp/bin/sudo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
cat >"$tmp/bin/nohup" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
"$@"
EOF
cat >"$tmp/bin/setsid" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  printf '<%s>\n' "$arg"
done
EOF
cat >"$tmp/bin/ip" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
chmod +x "$tmp/bin"/*

TEST_TRAINER_OUTPUT="$tmp/storage/run_x85" \
PATH="$tmp/bin:$PATH" \
deployment_root="$root" \
storage_root="$tmp/storage" \
runs_path="$tmp/runs" \
trainer_container_name=test-trainer \
master_node_id=82 \
node_id=82 \
trainer_nodes='82 76 77 78 80 83 84 85' \
RUN=x85 \
RUN_NAME=run_x85 \
TRAINER_CONFIG="$root/configs/paused/glm52/trainer.toml" \
bash "$cluster" trainer >/dev/null 2>&1

sync_log="$tmp/storage/microbatch-sync-x85.log"
test -f "$sync_log"
grep -Fx '<run_dir='"$tmp/storage/run_x85"'>' "$sync_log"
grep -Fx '<nodes=n76 n77 n78 n80 n83 n84 n85>' "$sync_log"
! grep -Fq '<nodes=n82' "$sync_log"
! grep -Fq '<RUN_DIR=' "$sync_log"
printf 'microbatch launcher passes lowercase run_dir and remote trainer nodes\n'
