#!/usr/bin/env bash
set -euo pipefail

run_dir=${RUN_DIR:?set RUN_DIR}
nodes=${NODES:-"10.92.20.76 10.92.20.77 10.92.20.78 10.92.20.80 10.92.20.83 10.92.20.84 10.92.20.85"}
interval=${INTERVAL:-1}
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
    echo "$host: refusing to replace $relative: destination hash differs" >&2
    return 1
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
      publish "$file" "$host" || true
    done
  done < <(find "$run_dir/rollouts" -type f -name 'rank_*.bin' -print0 2>/dev/null | sort -z)
}

while true; do
  scan
  [ "${ONCE:-0}" = 1 ] && exit
  sleep "$interval"
done
