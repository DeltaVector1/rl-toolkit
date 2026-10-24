#!/usr/bin/env bash
set -euo pipefail

run_dir=${run_dir:?set run_dir}
nodes=${nodes:?set nodes}
interval=${interval:-1}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
mkdir -p "$run_dir"
exec 9>"$run_dir/.microbatch-sync.lock"
flock -n 9 || exit 0
echo "$$" >"$run_dir/.microbatch-sync.pid"
trap 'rm -f "$run_dir/.microbatch-sync.pid"' EXIT

publish() {
  local source=$1 host=$2 relative=${1#"$run_dir"/}
  local final="$run_dir/$relative" partial="$run_dir/.incoming/$relative"
  local digest remote_digest

  digest=$(sha256sum "$source" | cut -d' ' -f1)
  remote_digest=$(ssh -n "${ssh_opts[@]}" "$host" \
    "sha256sum '$final' 2>/dev/null | cut -d' ' -f1" || true)
  [ "$remote_digest" = "$digest" ] && return
  if [ -n "$remote_digest" ]; then
    # A restarted run can leave the previous generation at this exact step.
    # Move it out of the receiver's path before publishing the new immutable
    # file. If the copy fails, the receiver sees no file instead of consuming
    # the previous generation.
    stale="$final.stale.$remote_digest"
    ssh -n "${ssh_opts[@]}" "$host" \
      "mv -f '$final' '$stale'" || {
      echo "$host: cannot quarantine stale $relative ($remote_digest)" >&2
      return 1
    }
    echo "$host: quarantined stale $relative ($remote_digest)" >&2
  fi

  ssh -n "${ssh_opts[@]}" "$host" \
    "mkdir -p '$(dirname "$partial")' '$(dirname "$final")'"
  if ssh -n "${ssh_opts[@]}" "$host" "command -v rsync >/dev/null"; then
    rsync -a --partial --append-verify --timeout=60 \
      -e "ssh ${ssh_opts[*]}" "$source" "$host:$partial"
  else
    scp "${ssh_opts[@]}" -p "$source" "$host:$partial"
  fi
  remote_digest=$(ssh -n "${ssh_opts[@]}" "$host" \
    "sha256sum '$partial' | cut -d' ' -f1")
  [ "$remote_digest" = "$digest" ] || {
    echo "$host: checksum failed for $relative" >&2
    return 1
  }
  ssh -n "${ssh_opts[@]}" "$host" "mv '$partial' '$final'"
  echo "published $relative to $host $digest"
}

scan() {
  local file host
  while IFS= read -r -d '' file; do
    for host in $nodes; do
      publish "$file" "$host" || return 1
    done
  done < <(find "$run_dir/rollouts" -type f -name 'rank_*.bin' -print0 2>/dev/null | sort -z)
}

write_marker() {
  local source generation count digest marker tmp host
  source=$(find "$run_dir/rollouts" -type f -name 'rank_0.bin' -print 2>/dev/null | sort -V | tail -1)
  [ -n "$source" ] || return 0
  generation=${source#"$run_dir/"}
  generation=${generation%/*}
  count=$(find "$(dirname "$source")" -maxdepth 1 -type f -name 'rank_*.bin' -printf '%f\n' 2>/dev/null | wc -l)
  digest=$(sha256sum "$source" | cut -d' ' -f1)
  marker="generation=$generation count=$count sha256=$digest"
  tmp="$run_dir/.microbatch-sync.marker.tmp.$$"
  printf '%s\n' "$marker" >"$tmp"
  mv -f "$tmp" "$run_dir/.microbatch-sync.marker"
  for host in $nodes; do
    publish "$run_dir/.microbatch-sync.marker" "$host" || return 1
  done
}

while true; do
  scan || {
    echo "microbatch sync stopped: publication failed" >&2
    exit 1
  }
  write_marker || {
    echo "microbatch sync stopped: marker publication failed" >&2
    exit 1
  }
  [ "${once:-0}" = 1 ] && exit
  sleep "$interval"
done
