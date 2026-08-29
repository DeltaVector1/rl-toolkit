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

# One WAN TCP stream collapses under the measured loss. Shorter x64p batches
# can be 55-60 MiB and took more than four minutes on one rsync stream, so
# stripe every material production batch rather than only files above 64 MiB.
parallel_threshold=${PARALLEL_THRESHOLD:-16777216}
parallel_streams=${PARALLEL_STREAMS:-16}
# A stripe that makes no I/O progress for one minute is dead on this path.
# Retry its partial file promptly instead of holding the trainer for the old
# two-minute rsync timeout. Active slow transfers are not timed out.
stripe_io_timeout=${STRIPE_IO_TIMEOUT:-60}
# rsync's I/O timeout did not terminate a wedged SSH child on x72p step 16.
# Bound each small stripe attempt by wall time as well. A timeout keeps the
# remote partial file, and the next attempt resumes it with --append-verify.
stripe_attempt_timeout=${STRIPE_ATTEMPT_TIMEOUT:-90}
# Starting all 16 SSH handshakes in one instant produced repeatable
# kex/reset failures on the jump path. Keep the independent TCP streams, but
# spread their handshakes across four seconds so a failed first wave does not
# force a complete retry pass.
parallel_launch_delay=${PARALLEL_LAUNCH_DELAY:-0.25}
adapter_source_base=${ADAPTER_SOURCE_BASE:-}
adapter_admin_urls=${ADAPTER_ADMIN_URLS:-}

prefetch_adapter() {
  local relative=$1 adapter_relative source_url admin status pid
  local -a pids
  [ -n "$adapter_source_base" ] || return 0
  [ -n "$adapter_admin_urls" ] || return 0
  case "$relative" in
    broadcasts/step_*/STABLE) ;;
    *) return 0 ;;
  esac
  adapter_relative=${relative%/STABLE}
  source_url="${adapter_source_base%/}/$adapter_relative"
  pids=()
  for admin in $adapter_admin_urls; do
    curl --fail --silent --show-error --max-time 600 \
      --retry 3 --retry-all-errors --retry-delay 2 \
      -H 'Content-Type: application/json' \
      -d "{\"lora_path\":\"$source_url\"}" \
      "${admin%/}/stage_lora_adapter" >/dev/null &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if [ "$status" != 0 ]; then
    echo "$(date -u +%FT%TZ) adapter prefetch incomplete for $relative" >&2
    return 1
  fi
  echo "$(date -u +%FT%TZ) prefetched adapter for $relative"
}

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

  # A single SSH key-exchange reset occurred in x58g. Retry only absent or
  # short stripes, then require the complete size manifest before assembly.
  status=1
  for attempt in 1 2 3; do
    if ! remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n'" \
      >"$parts_dir/remote-sizes"; then
      : >"$parts_dir/remote-sizes"
    fi
    pids=()
    for part in "$parts_dir"/part-*; do
      part_name=$(basename "$part")
      part_size=$(stat -c %s "$part")
      if grep -qx "$part_name $part_size" "$parts_dir/remote-sizes"; then
        continue
      fi
      timeout --signal=TERM --kill-after=10 "$stripe_attempt_timeout" \
        rsync -az --partial --append-verify --timeout="$stripe_io_timeout" \
        -e "ssh ${ssh_opts[*]}" "$part" "$dest_host:$remote_parts/$part_name" &
      pids+=("$!")
      sleep "$parallel_launch_delay"
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    remote_sizes=$(remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n' | sort" || true)
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

  if ! remote "mkdir -p '$(dirname "$partial")' '$(dirname "$final")'"; then
    return 1
  fi
  if ! copy_partial "$source" "$partial"; then
    return 1
  fi
  if ! remote_digest=$(remote "sha256sum '$partial' 2>/dev/null | cut -d' ' -f1"); then
    echo "cannot read transferred checksum for $relative" >&2
    return 1
  fi
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
    prefetch_adapter "$relative" || continue
    mv "$partial" "$final"
    echo "pulled $relative"
  done < <(remote "find '$dest_dir/broadcasts' -type f -name STABLE -print 2>/dev/null" || true)
}

while true; do
  scan || echo "$(date -u +%FT%TZ) batch scan failed; retrying" >&2
  pull_stable_markers || echo "$(date -u +%FT%TZ) stable-marker pull failed; retrying" >&2
  [ "${ONCE:-0}" = 1 ] && exit
  sleep "$interval"
done
