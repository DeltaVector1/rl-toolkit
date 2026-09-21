#!/usr/bin/env bash
set -euo pipefail

action=${1:-run}
source_dir=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}
dest_dir=${DEST_RUN_DIR:-$source_dir}
source_host=${SOURCE_HOST:?set SOURCE_HOST}
source_port=${SOURCE_PORT:?set SOURCE_PORT}
source_user=${SOURCE_USER:?set SOURCE_USER}
source_identity=${SOURCE_IDENTITY_FILE:?set SOURCE_IDENTITY_FILE}
source_known_hosts=${SOURCE_KNOWN_HOSTS_FILE:?set SOURCE_KNOWN_HOSTS_FILE}
run=${RUN:?set RUN}
trainer_addresses=${TRAINER_ADDRESSES:?set TRAINER_ADDRESSES}
trainer_container=${TRAINER_CT:?set TRAINER_CT}
orchestrator_container=${ORCH_CT:?set ORCH_CT}
interval=${INTERVAL:-5}
parallel_threshold=${PARALLEL_THRESHOLD:-16777216}
parallel_streams=${PARALLEL_STREAMS:-16}
case "$parallel_streams" in ''|*[!0-9]*|0) echo "PARALLEL_STREAMS must be a positive integer" >&2; exit 2 ;; esac
[ "$parallel_streams" -le 16 ] || { echo "PARALLEL_STREAMS must not exceed 16" >&2; exit 2; }
script=$(readlink -f "$0")
pid_file="$dest_dir/.rollout-pull.pid"
lock_file="$dest_dir/.rollout-pull.lock"
log_file=${LOG_FILE:-"${dest_dir%/*}/rollout-pull.log"}
source_target="$source_user@$source_host"
ssh_opts=(
  -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$source_known_hosts"
  -o GlobalKnownHostsFile=/dev/null -o UpdateHostKeys=no
  -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=8
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no
  -i "$source_identity" -p "$source_port"
)
printf -v rsync_ssh 'ssh %q ' "${ssh_opts[@]}"
rsync_ssh=${rsync_ssh% }

remote() {
  ssh "${ssh_opts[@]}" "$source_target" "$1"
}

local_path_safe() {
  python3 - "$1" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute():
    raise SystemExit(1)
current = Path(path.anchor)
for part in path.parts[1:]:
    current /= part
    if current.is_symlink():
        raise SystemExit(f"symlink path refused: {current}")
PY
}

remote_path_safe() {
  remote "python3 -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); assert p.is_absolute(); c=Path(p.anchor); [(lambda q: (_ for _ in ()).throw(SystemExit(1)) if q.is_symlink() else None)(c := c / part) for part in p.parts[1:]]' '$1'"
}

process_matches() {
  local pid=$1 command group environment
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  [ "$pid" -gt 1 ] && [ -r "/proc/$pid/cmdline" ] && [ -r "/proc/$pid/environ" ] || return 1
  group=$(awk '{print $5}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [ "$group" = "$pid" ] || return 1
  command=$(tr '\0' ' ' <"/proc/$pid/cmdline" | sed 's/[[:space:]]*$//') || return 1
  case "$command" in
    "bash $script run"|"/bin/bash $script run"|"/usr/bin/bash $script run") ;;
    *) return 1 ;;
  esac
  environment=$(tr '\0' '\n' <"/proc/$pid/environ") || return 1
  for expected in \
    "ROLLOUT_TRANSFER_DIRECTION=trainer_pull" \
    "SOURCE_RUN_DIR=$source_dir" "DEST_RUN_DIR=$dest_dir" \
    "SOURCE_HOST=$source_host" "SOURCE_PORT=$source_port" \
    "SOURCE_USER=$source_user" "SOURCE_IDENTITY_FILE=$source_identity" \
    "SOURCE_KNOWN_HOSTS_FILE=$source_known_hosts"; do
    grep -Fxq "$expected" <<<"$environment" || return 1
  done
}

status() {
  local pid proc candidate count=0
  pid=$(cat "$pid_file" 2>/dev/null || true)
  process_matches "$pid" || return 1
  for proc in /proc/[0-9]*; do
    candidate=${proc##*/}
    process_matches "$candidate" && count=$((count + 1))
  done
  [ "$count" -eq 1 ]
}

stop() {
  local pid group_live
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if ! process_matches "$pid"; then
    rm -f -- "$pid_file"
    return 0
  fi
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    group_live=$(ps -eo pgid=,stat= | awk -v group="$pid" '$1 == group && $2 !~ /^Z/ {print 1; exit}')
    [ -n "$group_live" ] || break
    sleep 0.1
  done
  group_live=$(ps -eo pgid=,stat= | awk -v group="$pid" '$1 == group && $2 !~ /^Z/ {print 1; exit}')
  if [ -n "$group_live" ]; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 50); do
    group_live=$(ps -eo pgid=,stat= | awk -v group="$pid" '$1 == group && $2 !~ /^Z/ {print 1; exit}')
    [ -n "$group_live" ] || break
    sleep 0.1
  done
  [ -z "$group_live" ] || return 1
  flock -n "$lock_file" -c true || return 1
  rm -f -- "$pid_file"
}

start() {
  local_path_safe "$dest_dir" || return 1
  mkdir -p "$dest_dir"
  if status; then
    echo "rollout puller already running"
    return
  fi
  stop
  nohup setsid env \
    ROLLOUT_TRANSFER_DIRECTION=trainer_pull \
    SOURCE_RUN_DIR="$source_dir" DEST_RUN_DIR="$dest_dir" \
    SOURCE_HOST="$source_host" SOURCE_PORT="$source_port" SOURCE_USER="$source_user" \
    SOURCE_IDENTITY_FILE="$source_identity" SOURCE_KNOWN_HOSTS_FILE="$source_known_hosts" \
    LOG_FILE="$log_file" \
    bash "$script" run >>"$log_file" 2>&1 </dev/null &
  local pid=$!
  for _ in $(seq 1 20); do
    status && { echo "rollout puller started (pid $pid)"; return; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  echo "rollout puller did not start" >&2
  return 1
}

case "$action" in
  start) start; exit ;;
  stop) stop; exit ;;
  status) status; exit ;;
  run) ;;
  *) echo "usage: $0 {run|start|stop|status}" >&2; exit 2 ;;
esac

local_path_safe "$dest_dir"
remote_path_safe "$source_dir"
mkdir -p "$dest_dir"
exec 9>"$lock_file"
flock -n 9 || exit 0
echo "$$" >"$pid_file"
trap 'rm -f -- "$pid_file"' EXIT

valid_pull_path() {
  local relative=$1
  [[ "$relative" == control/orch.toml ||
     "$relative" =~ ^broadcasts/step_[0-9]+/receivers-ready\.json$ ||
     "$relative" =~ ^rollouts/step_[0-9]+/train_rollouts\.bin$ ]]
}

write_receipt() {
  local source=$1 digest=$2
  remote_path_safe "$(dirname "$source")" || return 1
  remote "marker='$source.delivered'; [ ! -L \"\$marker\" ] || exit 1; if [ -e \"\$marker\" ]; then [ -f \"\$marker\" ] && [ \"\$(cat \"\$marker\")\" = '$digest' ]; else partial=\$(mktemp -- \"\${marker}.tmp.XXXXXX\") || exit 1; if ! printf '%s\\n' '$digest' >\"\$partial\"; then rm -f -- \"\$partial\"; exit 1; fi; if ! ln \"\$partial\" \"\$marker\" 2>/dev/null; then equal=0; [ ! -L \"\$marker\" ] && [ -f \"\$marker\" ] && [ \"\$(cat \"\$marker\")\" = '$digest' ] && equal=1; rm -f -- \"\$partial\"; [ \"\$equal\" = 1 ]; else rm -f -- \"\$partial\"; fi; fi"
}

pull_bytes() {
  local source=$1 partial=$2 digest=$3 size chunk remote_parts local_parts listing
  local part name expected status=0
  local -a pids=()
  size=$(remote "stat -c %s -- '$source'") || return 1
  if [ "$size" -lt "$parallel_threshold" ]; then
    rsync -az --partial --append-verify --timeout=60 -e "$rsync_ssh" \
      "$source_target:$source" "$partial"
    return
  fi
  chunk=$(( (size + parallel_streams - 1) / parallel_streams ))
  remote_parts="$source.pull-parts-$digest"
  local_parts="$partial.parallel"
  remote_path_safe "$remote_parts" || return 1
  local_path_safe "$local_parts" || return 1
  mkdir -p "$local_parts"
  remote "mkdir -p '$remote_parts'; if [ ! -f '$remote_parts/READY' ]; then find '$remote_parts' -maxdepth 1 -type f -delete; split -b '$chunk' -d -a 3 '$source' '$remote_parts/part-'; : >'$remote_parts/READY'; fi" || return 1
  listing=$(remote "find '$remote_parts' -maxdepth 1 -type f -name 'part-*' -printf '%f %s\\n' | sort") || return 1
  [ -n "$listing" ] || return 1
  while read -r name expected; do
    [[ "$name" =~ ^part-[0-9]{3}$ && "$expected" =~ ^[0-9]+$ ]] || return 1
    part="$local_parts/$name"
    [ -f "$part" ] && [ "$(stat -c %s "$part")" = "$expected" ] && continue
    while [ "${#pids[@]}" -ge "$parallel_streams" ]; do
      wait "${pids[0]}" || status=1
      pids=("${pids[@]:1}")
    done
    rsync -az --partial --append-verify --timeout=60 -e "$rsync_ssh" \
      "$source_target:$remote_parts/$name" "$part" &
    pids+=("$!")
  done <<<"$listing"
  for part in "${pids[@]}"; do wait "$part" || status=1; done
  [ "$status" = 0 ] || return 1
  : >"$partial"
  while read -r name expected; do
    part="$local_parts/$name"
    [ "$(stat -c %s "$part")" = "$expected" ] || return 1
    cat "$part" >>"$partial"
  done <<<"$listing"
  remote "find '$remote_parts' -maxdepth 1 -type f -delete; rmdir '$remote_parts'" || return 1
  find "$local_parts" -maxdepth 1 -type f -delete
  rmdir "$local_parts"
}

pull_one() {
  local source=$1 relative=${1#"$source_dir"/} final partial first_digest local_digest last_digest
  valid_pull_path "$relative" || { echo "refusing unexpected rollout path: $relative" >&2; return 1; }
  remote_path_safe "$source" || return 1
  final="$dest_dir/$relative"
  partial="$dest_dir/.incoming/$relative"
  local_path_safe "$final" || return 1
  local_path_safe "$(dirname "$final")" || return 1
  local_path_safe "$(dirname "$partial")" || return 1
  first_digest=$(remote "sha256sum -- '$source'" | awk '{print $1}') || return 1
  [[ "$first_digest" =~ ^[a-f0-9]{64}$ ]] || return 1
  if [ -f "$final" ]; then
    local_digest=$(sha256sum -- "$final" | awk '{print $1}') || return 1
    if [ "$local_digest" = "$first_digest" ]; then
      write_receipt "$source" "$local_digest"
      return
    fi
    echo "refusing to replace $final: destination hash differs" >&2
    return 1
  fi
  mkdir -p "$(dirname "$partial")" "$(dirname "$final")"
  pull_bytes "$source" "$partial" "$first_digest" || return 1
  local_digest=$(sha256sum -- "$partial" | awk '{print $1}') || return 1
  last_digest=$(remote "sha256sum -- '$source'" | awk '{print $1}') || return 1
  if [ "$local_digest" != "$first_digest" ] || [ "$last_digest" != "$first_digest" ]; then
    echo "source changed while pulling $relative" >&2
    return 1
  fi
  if ! ln -- "$partial" "$final" 2>/dev/null; then
    local_digest=$(sha256sum -- "$final" 2>/dev/null | awk '{print $1}') || return 1
    [ "$local_digest" = "$first_digest" ] || return 1
  fi
  rm -f -- "$partial"
  local_digest=$(sha256sum -- "$final" | awk '{print $1}') || return 1
  [ "$local_digest" = "$first_digest" ] || return 1
  write_receipt "$source" "$local_digest" || return 1
  echo "pulled $relative $local_digest"
}

push_stable() {
  local local_file=$1 relative=${1#"$dest_dir"/} remote_file partial digest remote_digest
  [[ "$relative" =~ ^broadcasts/step_[0-9]+/STABLE$ ]] || return 1
  remote_file="$source_dir/$relative"
  partial="$source_dir/.incoming/$relative"
  local_path_safe "$local_file" || return 1
  remote_path_safe "$(dirname "$remote_file")" || return 1
  digest=$(sha256sum -- "$local_file" | awk '{print $1}') || return 1
  remote_digest=$(remote "sha256sum -- '$remote_file' 2>/dev/null | awk '{print \$1}'" || true)
  [ "$remote_digest" = "$digest" ] && return 0
  [ -z "$remote_digest" ] || { echo "refusing to replace $remote_file: destination hash differs" >&2; return 1; }
  remote "mkdir -p '$(dirname "$partial")' '$(dirname "$remote_file")'" || return 1
  remote_path_safe "$partial" || return 1
  rsync -az --partial --timeout=60 -e "$rsync_ssh" "$local_file" "$source_target:$partial" || return 1
  remote_digest=$(remote "sha256sum -- '$partial' | awk '{print \$1}'") || return 1
  [ "$remote_digest" = "$digest" ] || return 1
  remote "digest='$digest'; partial='$partial'; final='$remote_file'; [ -f \"\$partial\" ] && [ ! -L \"\$partial\" ] && [ ! -L \"\$final\" ] || exit 1; if ! ln \"\$partial\" \"\$final\" 2>/dev/null; then [ \"\$(sha256sum -- \"\$final\" 2>/dev/null | awk '{print \$1}')\" = \"\$digest\" ] || exit 1; fi; rm -f -- \"\$partial\"" || return 1
  echo "sent $relative $digest"
}

scan() {
  local remote_file local_file status=0
  while IFS= read -r remote_file; do
    [ -n "$remote_file" ] || continue
    pull_one "$remote_file" || status=1
  done < <(remote "control='$source_dir/control/orch.toml'; if [ -f \"\$control\" ] && [ \$(( \$(date +%s) - \$(stat -c %Y \"\$control\") )) -ge 2 ]; then printf '%s\\n' \"\$control\"; fi; find '$source_dir/broadcasts' '$source_dir/rollouts' -type f \( -path '*/broadcasts/step_*/receivers-ready.json' -o -path '*/rollouts/step_*/train_rollouts.bin' \) -print 2>/dev/null" || true)
  while IFS= read -r -d '' local_file; do
    push_stable "$local_file" || status=1
  done < <(find "$dest_dir/broadcasts" -mindepth 2 -maxdepth 2 -type f -name STABLE -print0 2>/dev/null | sort -z)
  return "$status"
}

refresh_health_lease() {
  local host state now
  for host in $trainer_addresses; do
    state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$host" \
      "sudo docker ps --filter name=^${trainer_container}$ --format '{{.Names}}' 2>/dev/null | head -1" \
      2>/dev/null || true)
    [ "$state" = "$trainer_container" ] || return 0
  done
  now=$(date +%s)
  remote "mkdir -p '$source_dir'; lease='$source_dir/.trainer-health-lease'; partial=\$(mktemp -- \"\${lease}.tmp.XXXXXX\") || exit 1; printf '%s\\n' '$now' >\"\$partial\" && mv -- \"\$partial\" \"\$lease\""
}

while true; do
  status=0
  scan || {
    echo "$(date -u +%FT%TZ) rollout pull failed; retrying" >&2
    status=1
  }
  refresh_health_lease || {
    echo "$(date -u +%FT%TZ) trainer health lease update failed; retrying" >&2
    status=1
  }
  [ "${ONCE:-0}" = 1 ] && exit "$status"
  sleep "$interval"
done
