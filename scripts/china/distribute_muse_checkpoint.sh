#!/usr/bin/env bash
set -euo pipefail

# Run on n82 after bootstrap_muse_checkpoint.sh has completed and the verified
# bootstrap directory has been renamed to /data/muse-glimmer/model.
source_root=${SOURCE_ROOT:-/data/muse-glimmer}
model=$source_root/model
nodes=(76 77 78 80 83 84 85)

shard1=8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694
shard2=b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381

test -f "$model/model-00001-of-00002.safetensors"
test -f "$model/model-00002-of-00002.safetensors"
test "$(sha256sum "$model/model-00001-of-00002.safetensors" | cut -d' ' -f1)" = "$shard1"
test "$(sha256sum "$model/model-00002-of-00002.safetensors" | cut -d' ' -f1)" = "$shard2"
test -d "$source_root/transformers/transformers"
test -d "$source_root/python-overrides/safetensors"

copy_one() {
  local node=$1
  local host=10.92.20.$node
  echo "n$node: preparing destination"
  ssh -o BatchMode=yes "$host" \
    'sudo install -d -o "$(id -u)" -g "$(id -g)" -m 0755 /data/muse-glimmer /data/muse-glimmer/model /data/muse-glimmer/transformers /data/muse-glimmer/python-overrides'

  if ssh -o BatchMode=yes "$host" 'command -v rsync >/dev/null'; then
    rsync -rlt --partial --whole-file "$model/" "$host:/data/muse-glimmer/model/"
    rsync -rlt --partial --whole-file "$source_root/transformers/" "$host:/data/muse-glimmer/transformers/"
    rsync -rlt --partial --whole-file "$source_root/python-overrides/" "$host:/data/muse-glimmer/python-overrides/"
  else
    # n85 has OpenSSH but no rsync. The China LAN is stable, and final hashes
    # make a partial or damaged copy fail closed.
    scp -q -rp "$model/." "$host:/data/muse-glimmer/model/"
    scp -q -rp "$source_root/transformers/." "$host:/data/muse-glimmer/transformers/"
    scp -q -rp "$source_root/python-overrides/." "$host:/data/muse-glimmer/python-overrides/"
  fi

  ssh -o BatchMode=yes "$host" \
    "test \"\$(sha256sum /data/muse-glimmer/model/model-00001-of-00002.safetensors | cut -d' ' -f1)\" = '$shard1' && \
     test \"\$(sha256sum /data/muse-glimmer/model/model-00002-of-00002.safetensors | cut -d' ' -f1)\" = '$shard2' && \
     test -f /data/muse-glimmer/model/model.safetensors.index.json && \
     test -f /data/muse-glimmer/transformers/transformers/models/muse_glimmer/modeling_muse_glimmer.py && \
     test -f /data/muse-glimmer/python-overrides/safetensors/__init__.py"
  echo "n$node: verified"
}

pids=()
for node in "${nodes[@]}"; do
  copy_one "$node" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
test "$failed" -eq 0
echo "Muse checkpoint and dependencies verified on all eight 4090 trainer nodes"
