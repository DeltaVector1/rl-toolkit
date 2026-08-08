#!/usr/bin/env bash
set -euo pipefail

source_dir=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}
dest_dir=${DEST_RUN_DIR:-$source_dir}
dest_host=${DEST_HOST:-china-trainer}
interval=${INTERVAL:-5}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=8)
mkdir -p "$source_dir"
exec 9>"$source_dir/.batch-sync.lock"
flock -n 9 || exit 0
echo "$$" >"$source_dir/.batch-sync.pid"
trap 'rm -f "$source_dir/.batch-sync.pid"' EXIT

remote() {
  if [ "$dest_host" = local ]; then
    bash -c "$1"
  else
    ssh "${ssh_opts[@]}" "$dest_host" "$1"
  fi
}

# One WAN TCP stream collapses under the measured loss. Even ordinary
# no-router batches are 80-100 MiB, so stripe them too.
parallel_threshold=${PARALLEL_THRESHOLD:-67108864}
parallel_streams=${PARALLEL_STREAMS:-16}

copy_parallel() {
  local source=$1 destination=$2
  local size chunk_size parts_dir remote_parts status pid part part_name part_size
  local -a pids

  size=$(stat -c %s "$source")
  chunk_size=$(( (size + parallel_streams - 1) / parallel_streams ))
  parts_dir="$source.parallel"
  remote_parts="$destination.parallel"
  rm -rf "$parts_dir"
  mkdir -p "$parts_dir"
  split -b "$chunk_size" -d -a 3 "$source" "$parts_dir/part-"
  remote "mkdir -p '$remote_parts'"
  remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n'" \
    >"$parts_dir/remote-sizes" || : >"$parts_dir/remote-sizes"

  status=0
  pids=()
  for part in "$parts_dir"/part-*; do
    part_name=$(basename "$part")
    part_size=$(stat -c %s "$part")
    if grep -qx "$part_name $part_size" "$parts_dir/remote-sizes"; then
      continue
    fi
    rsync -az --partial --append-verify --timeout=120 \
      -e "ssh ${ssh_opts[*]}" "$part" "$dest_host:$remote_parts/$part_name" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [ "$status" = 0 ] || return 1

  remote "rm -f '$destination.assembling'; cat '$remote_parts'/part-* >'$destination.assembling'; mv '$destination.assembling' '$destination'; rm -rf '$remote_parts'"
  rm -rf "$parts_dir"
}

copy_partial() {
  local source=$1 destination=$2 size
  if [ "$dest_host" = local ]; then
    cp "$source" "$destination"
    return
  fi
  size=$(stat -c %s "$source")
  if [ "$size" -ge "$parallel_threshold" ]; then
    copy_parallel "$source" "$destination"
  else
    rsync -az --partial --append-verify --timeout=120 \
      -e "ssh ${ssh_opts[*]}" "$source" "$dest_host:$destination"
  fi
}

publish() {
  local source=$1 relative=${1#"$source_dir"/}
  local final="$dest_dir/$relative" partial="$dest_dir/.incoming/$relative"
  local digest remote_digest

  digest=$(sha256sum "$source" | cut -d' ' -f1)
  remote_digest=$(remote "sha256sum '$final' 2>/dev/null | cut -d' ' -f1" || true)
  if [ "$remote_digest" = "$digest" ]; then
    printf '%s\n' "$digest" >"$source.delivered"
    return
  fi
  if [ -n "$remote_digest" ]; then
    echo "refusing to replace $final: destination hash differs" >&2
    return 1
  fi

  remote "mkdir -p '$(dirname "$partial")' '$(dirname "$final")'"
  copy_partial "$source" "$partial"
  remote_digest=$(remote "sha256sum '$partial' | cut -d' ' -f1")
  [ "$remote_digest" = "$digest" ] || {
    echo "checksum failed for $relative" >&2
    return 1
  }
  remote "mv '$partial' '$final'"
  printf '%s\n' "$digest" >"$source.delivered"
  echo "published $relative $digest"
}

scan() {
  local file
  if [ -f "$source_dir/control/orch.toml" ] &&
     [ $(( $(date +%s) - $(stat -c %Y "$source_dir/control/orch.toml") )) -ge 2 ] &&
     [ ! -f "$source_dir/control/orch.toml.delivered" ]; then
    publish "$source_dir/control/orch.toml" || true
  fi
  while IFS= read -r -d '' file; do
    [ -f "$file.delivered" ] || publish "$file" || true
  done < <(find "$source_dir/rollouts" -type f -name train_rollouts.bin -print0 2>/dev/null | sort -z)
}

pull_stable_markers() {
  local remote_file relative final partial
  [ "$dest_host" = local ] && return
  while IFS= read -r remote_file; do
    [ -n "$remote_file" ] || continue
    relative=${remote_file#"$dest_dir"/}
    final="$source_dir/$relative"
    [ -f "$final" ] && continue
    partial="$source_dir/.incoming/$relative"
    mkdir -p "$(dirname "$partial")" "$(dirname "$final")"
    rsync -a --partial --timeout=60 -e "ssh ${ssh_opts[*]}" \
      "$dest_host:$remote_file" "$partial" || continue
    mv "$partial" "$final"
    echo "pulled $relative"
  done < <(remote "find '$dest_dir/broadcasts' -type f -name STABLE -print 2>/dev/null" || true)
}

while true; do
  scan
  pull_stable_markers
  [ "${ONCE:-0}" = 1 ] && exit
  sleep "$interval"
done
