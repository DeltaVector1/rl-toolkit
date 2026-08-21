#!/usr/bin/env bash
set -euo pipefail

source_file=${1:?usage: distribute_resume_adapter.sh /absolute/path/adapter_model.safetensors}
nodes=(76 77 78 80 83 84 85)

test -f "$source_file"
test "${source_file#/data/glm52-rl/runs/}" != "$source_file"
source_hash=$(sha256sum "$source_file" | cut -d' ' -f1)
source_dir=$(dirname "$source_file")

copy_one() {
  local node=$1 host=10.92.20.$1
  ssh -o BatchMode=yes "$host" "mkdir -p '$source_dir'"
  if ssh -o BatchMode=yes "$host" 'command -v rsync >/dev/null'; then
    rsync -t --partial --whole-file "$source_file" "$host:$source_file"
  else
    scp -q "$source_file" "$host:$source_file"
  fi
  test "$(ssh -o BatchMode=yes "$host" "sha256sum '$source_file' | cut -d' ' -f1")" = "$source_hash"
  echo "n$node verified"
}

pids=()
for node in "${nodes[@]}"; do
  copy_one "$node" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0
echo "adapter $source_hash verified on all eight trainer nodes"
