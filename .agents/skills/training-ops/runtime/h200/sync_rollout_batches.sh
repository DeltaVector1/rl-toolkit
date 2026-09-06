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

# Use parallel streams for material batches on the WAN path.
parallel_threshold=${PARALLEL_THRESHOLD:-16777216}
parallel_streams=${PARALLEL_STREAMS:-16}
parallel_active_limit=${parallel_active_limit:-16}
stripe_io_timeout=${STRIPE_IO_TIMEOUT:-60}
stripe_attempt_timeout=${STRIPE_ATTEMPT_TIMEOUT:-90}
case "$parallel_active_limit" in
  ''|*[!0-9]*|0) echo "parallel_active_limit must be a positive integer" >&2; exit 2 ;;
esac
# Spread SSH handshakes across four seconds.
parallel_launch_delay=${PARALLEL_LAUNCH_DELAY:-0.25}
copy_parallel() {
  local source=$1 destination=$2
  local size chunk_size parts_dir remote_parts status pid part part_name part_size attempt
  local expected_sizes remote_sizes
  local -a pids

  size=$(stat -c %s "$source")
  chunk_size=$(( (size + parallel_streams - 1) / parallel_streams ))
  parts_dir="$source.parallel"
  remote_parts="$destination.parallel"
  rm -rf "$parts_dir"
  mkdir -p "$parts_dir"
  split -b "$chunk_size" -d -a 3 "$source" "$parts_dir/part-"
  if ! remote "mkdir -p '$remote_parts'"; then
    rm -rf "$parts_dir"
    return 1
  fi
  expected_sizes=$(find "$parts_dir" -maxdepth 1 -type f -name 'part-*' -printf '%f %s\n' | sort)

  for attempt in 1 2 3; do
    if ! remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n'" \
      >"$parts_dir/remote-sizes"; then
      echo "parallel transfer attempt $attempt cannot read remote part sizes for $source" >&2
      status=1
      continue
    fi
    pids=()
    status=0
    for part in "$parts_dir"/part-*; do
      part_name=$(basename "$part")
      part_size=$(stat -c %s "$part")
      if grep -qx "$part_name $part_size" "$parts_dir/remote-sizes"; then
        continue
      fi
      while [ "${#pids[@]}" -ge "$parallel_active_limit" ]; do
        if ! wait "${pids[0]}"; then
          status=1
        fi
        pids=("${pids[@]:1}")
      done
      timeout --signal=TERM --kill-after=10 "$stripe_attempt_timeout" \
        rsync -az --partial --append-verify --timeout="$stripe_io_timeout" \
        -e "ssh ${ssh_opts[*]}" "$part" "$dest_host:$remote_parts/$part_name" &
      pids+=("$!")
      sleep "$parallel_launch_delay"
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    if ! remote_sizes=$(remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n' | sort"); then
      echo "parallel transfer attempt $attempt cannot verify remote part sizes for $source" >&2
      status=1
      continue
    fi
    if [ "$status" = 0 ] && [ "$remote_sizes" = "$expected_sizes" ]; then
      status=0
      break
    fi
    status=1
    echo "parallel transfer attempt $attempt incomplete for $source" >&2
  done
  if [ "$status" != 0 ]; then
    rm -rf "$parts_dir"
    return 1
  fi

  if ! remote "rm -f '$destination.assembling'; cat '$remote_parts'/part-* >'$destination.assembling' && mv '$destination.assembling' '$destination' && rm -rf '$remote_parts'"; then
    rm -rf "$parts_dir"
    return 1
  fi
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
    timeout --signal=TERM --kill-after=10 "$stripe_attempt_timeout" \
      rsync -az --partial --append-verify --timeout="$stripe_io_timeout" \
      -e "ssh ${ssh_opts[*]}" "$source" "$dest_host:$destination"
  fi
}

publish() {
  local source=$1 relative=${1#"$source_dir"/}
  local final="$dest_dir/$relative" partial="$dest_dir/.incoming/$relative"
  local digest remote_digest probe_status

  digest=$(sha256sum "$source" | cut -d' ' -f1)
  if remote_digest=$(remote "if [ -f '$final' ]; then sha256sum '$final'; else exit 3; fi"); then
    remote_digest=${remote_digest%% *}
  else
    probe_status=$?
    if [ "$probe_status" -ne 3 ]; then
      echo "cannot probe destination checksum for $relative" >&2
      return 1
    fi
    remote_digest=
  fi
  if [ "$remote_digest" = "$digest" ]; then
    printf '%s\n' "$digest" >"$source.delivered"
    return
  fi
  if [ -n "$remote_digest" ]; then
    echo "refusing to replace $final: destination hash differs" >&2
    return 1
  fi

  if ! remote "mkdir -p '$(dirname "$partial")' '$(dirname "$final")'"; then
    return 1
  fi
  if ! copy_partial "$source" "$partial"; then
    return 1
  fi
  if ! remote_digest=$(remote "sha256sum '$partial'"); then
    echo "cannot read transferred checksum for $relative" >&2
    return 1
  fi
  remote_digest=${remote_digest%% *}
  [ "$remote_digest" = "$digest" ] || {
    echo "checksum failed for $relative" >&2
    return 1
  }
  if ! remote "mv '$partial' '$final'"; then
    return 1
  fi
  printf '%s\n' "$digest" >"$source.delivered"
  echo "published $relative $digest"
}

scan() {
  local file relative status=0
  if [ -f "$source_dir/control/orch.toml" ] &&
     [ $(( $(date +%s) - $(stat -c %Y "$source_dir/control/orch.toml") )) -ge 2 ] &&
     [ ! -f "$source_dir/control/orch.toml.delivered" ]; then
    publish "$source_dir/control/orch.toml" || status=1
  fi
  while IFS= read -r -d '' file; do
    relative=${file#"$source_dir"/}
    [[ "$relative" =~ ^broadcasts/step_[0-9]+/receivers-ready\.json$ ]] || continue
    [ -f "$file.delivered" ] || publish "$file" || status=1
  done < <(find "$source_dir/broadcasts" -mindepth 2 -maxdepth 2 -type f -name receivers-ready.json -print0 2>/dev/null | sort -z)
  while IFS= read -r -d '' file; do
    [ -f "$file.delivered" ] || publish "$file" || status=1
  done < <(find "$source_dir/rollouts" -type f -name train_rollouts.bin -print0 2>/dev/null | sort -z)
  return "$status"
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
  if ! scan; then
    if [ "${ONCE:-0}" = 1 ]; then
      echo "$(date -u +%FT%TZ) batch scan failed" >&2
      exit 1
    fi
    echo "$(date -u +%FT%TZ) batch scan failed; retrying" >&2
  fi
  pull_stable_markers || echo "$(date -u +%FT%TZ) stable-marker pull failed; retrying" >&2
  [ "${ONCE:-0}" = 1 ] && exit
  sleep "$interval"
done
