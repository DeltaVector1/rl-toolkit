#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bin="$tmp/bin"
mkdir -p "$bin" "$tmp/deployment/configs/china"

cat >"$tmp/deployment/configs/china/trainer.toml" <<'EOF'
output_dir = "/tmp/nsys-launch-test"

[rollout_transport]
type = "nccl"

[model]
pp = 8
cp = 1
EOF

cat >"$bin/sudo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "docker" ] && [ "${2:-}" = "run" ]; then
  shift 2
  printf '%s\n' "$@" >"$NSYS_DOCKER_ARGS"
fi
exit 0
EOF
chmod +x "$bin/sudo"
cat >"$bin/ip" <<'EOF'
#!/usr/bin/env bash
echo '1: vlan130@eth0:'
EOF
chmod +x "$bin/ip"

run_launcher() {
  local node=$1
  local nodes=$2
  local args_file=$3
  NODE_ID="$node" \
  TRAINER_NODES="$nodes" \
  PRIME_NSYS_PROFILE=1 \
  PRIME_CUDA_PROFILE=1 \
  PRIME_CUDA_PROFILE_RANK=24 \
  PRIME_CUDA_PROFILE_STEPS=2 \
  PRIME_CUDA_PROFILE_SECONDS=60 \
  TRAINER_CONFIG="$tmp/deployment/configs/china/trainer.toml" \
  prime_deployment="$tmp/deployment" \
  NSYS_DOCKER_ARGS="$args_file" \
  PATH="$bin:$PATH" \
    bash "$root/scripts/china/cluster.sh" trainer
}

n78_args="$tmp/n78.args"
run_launcher 78 "82 76 77 78 80" "$n78_args"
grep -Fx -- '--entrypoint' "$n78_args" >/dev/null
grep -Fx -- '/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys' "$n78_args" >/dev/null
grep -Fx -- 'profile' "$n78_args" >/dev/null

n80_args="$tmp/n80.args"
n80_output=$(run_launcher 80 "82 76 77 78 80" "$n80_args" 2>&1)
grep -F 'Nsight disabled on n80; capture owner is n78 rank 24' <<<"$n80_output" >/dev/null
grep -Fx -- '--entrypoint' "$n80_args" >/dev/null
grep -Fx -- '/app/.venv/bin/torchrun' "$n80_args" >/dev/null
if grep -F '/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys' "$n80_args" >/dev/null; then
  echo "non-owner unexpectedly used nsys" >&2
  exit 1
fi
if grep -Fx -- 'profile' "$n80_args" >/dev/null; then
  echo "non-owner unexpectedly received nsys arguments" >&2
  exit 1
fi

bad_args="$tmp/bad.args"
if run_launcher 80 "82 76 77 80 78" "$bad_args" 2>"$tmp/bad.stderr"; then
  echo "reordered trainer nodes unexpectedly passed" >&2
  exit 1
fi
grep -F 'Nsight rank 24 must map to n78 in trainer_nodes order' "$tmp/bad.stderr" >/dev/null
if [ -e "$bad_args" ]; then
  echo "reordered trainer nodes reached Docker" >&2
  exit 1
fi
