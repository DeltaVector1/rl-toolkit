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
pull_part_timeout=${PULL_PART_TIMEOUT:-300}
case "$pull_part_timeout" in ''|*[!0-9]*|0) echo "PULL_PART_TIMEOUT must be a positive integer" >&2; exit 2 ;; esac
[ "$pull_part_timeout" -le 3600 ] || { echo "PULL_PART_TIMEOUT must not exceed 3600 seconds" >&2; exit 2; }
script=$(readlink -f "$0")
rollout_backend=${ROLLOUT_TRANSFER_BACKEND:-ssh}
case "$rollout_backend" in ssh|b2_s3) ;; *) echo "invalid rollout backend" >&2; exit 2 ;; esac
b2_python=${B2_ROLLOUT_PYTHON:-/data/trinity/b2-transport-venv/bin/python}
b2_helper=${B2_ROLLOUT_HELPER:-${script%/china/*}/h200/b2_rollout.py}
b2_local_helper=${script%/china/*}/h200/b2_rollout.py
b2_download_workers=${B2_DOWNLOAD_WORKERS:-1}
b2_download_threshold=${B2_DOWNLOAD_THRESHOLD_BYTES:-67108864}
[[ "$b2_download_workers" =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || { echo "invalid B2 download workers" >&2; exit 2; }
[[ "$b2_download_threshold" =~ ^[1-9][0-9]{0,12}$ ]] && [ "$b2_download_threshold" -le 1099511627776 ] || { echo "invalid B2 download threshold" >&2; exit 2; }
pid_file="$dest_dir/.rollout-pull.pid"
lock_file="$dest_dir/.rollout-pull.lock"
control_file="$dest_dir/.rollout-pull-control.lock"
pause_file="$dest_dir/.rollout-pull-paused"
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
rsync_ssh=
rsync_ssh_dir=
rsync_ssh_wrapper=
pid_owned=0
health_lease_pid=
health_probe_pids=()
stop_signalled=0
stop_signalled_pids=()
stop_target_pid=

pull_rsync() {
  timeout --signal=TERM --kill-after=10s "$pull_part_timeout" rsync "$@"
}

remote() {
  ssh -n "${ssh_opts[@]}" "$source_target" "$1"
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
  if grep -q '^ROLLOUT_TRANSFER_BACKEND=' <<<"$environment"; then
    grep -Fxq "ROLLOUT_TRANSFER_BACKEND=$rollout_backend" <<<"$environment" || return 1
  else
    [ "$rollout_backend" = ssh ] || return 1
  fi
}

status() {
  local pid proc candidate count=0 environment workers threshold
  pid=$(cat "$pid_file" 2>/dev/null || true)
  process_matches "$pid" || return 1
  # Desired tuning is not process ownership: stop can replace older settings.
  if [ "$rollout_backend" = b2_s3 ]; then
    environment=$(tr '\0' '\n' <"/proc/$pid/environ") || return 1
    workers=$(sed -n 's/^B2_DOWNLOAD_WORKERS=//p' <<<"$environment")
    threshold=$(sed -n 's/^B2_DOWNLOAD_THRESHOLD_BYTES=//p' <<<"$environment")
    [ "${workers:-1}" = "$b2_download_workers" ] || return 1
    [ "${threshold:-67108864}" = "$b2_download_threshold" ] || return 1
  fi
  for proc in /proc/[0-9]*; do
    candidate=${proc##*/}
    process_matches "$candidate" && count=$((count + 1))
  done
  [ "$count" -eq 1 ]
}

state() {
  local -a pids=()
  local pid
  local_path_safe "$dest_dir" || return 1
  if status; then
    echo running
    return 0
  fi
  mapfile -t pids < <(matching_pids)
  [ "${#pids[@]}" -eq 0 ] || return 1
  [ ! -e "$pid_file" ] && [ ! -L "$pid_file" ] || return 1
  if [ -e "$lock_file" ] || [ -L "$lock_file" ]; then
    [ -f "$lock_file" ] && [ ! -L "$lock_file" ] || return 1
    flock -n "$lock_file" -c true || return 1
  fi
  echo stopped
}

puller_identity() {
  python3 - "$dest_dir" "${1:-0}" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

directory = Path(sys.argv[1])
pid = sys.argv[2]
info = os.stat(directory, follow_symlinks=False)
if not stat.S_ISDIR(info.st_mode):
    raise SystemExit(1)
fields = [str(info.st_dev), str(info.st_ino), str(info.st_uid), str(info.st_gid)]
if pid == "0":
    fields.extend(("0", "0", "0"))
else:
    if not pid.isdigit() or int(pid) <= 1:
        raise SystemExit(1)
    process = Path("/proc") / pid
    stat_fields = (process / "stat").read_bytes().rsplit(b") ", 1)[1].split()
    if len(stat_fields) <= 19 or not stat_fields[19].isdigit():
        raise SystemExit(1)
    command = (process / "cmdline").read_bytes()
    environment = (process / "environ").read_bytes()
    fields.extend((
        stat_fields[19].decode("ascii"),
        hashlib.sha256(command).hexdigest(),
        hashlib.sha256(environment).hexdigest(),
    ))
print(" ".join(fields))
PY
}

receipt_valid() {
  local current mode_owner
  [ "${#receipt[@]}" -eq 12 ] && \
    [ "${receipt[0]}" = "$1" ] && [ "${receipt[1]}" = "$dest_dir" ] && \
    [[ "${receipt[2]}" == running || "${receipt[2]}" == stopped ]] && \
    [[ "${receipt[4]}" == yes || "${receipt[4]}" == no ]] || return 1
  current=$(puller_identity 0) || return 1
  read -r -a current_fields <<<"$current"
  [ "${#current_fields[@]}" -eq 7 ] || return 1
  for index in 0 1 2 3; do
    [ "${receipt[index + 5]}" = "${current_fields[index]}" ] || return 1
  done
  mode_owner=$(stat -c '%u:%g:%a' -- "$pause_file") || return 1
  [ "$mode_owner" = "${receipt[7]}:${receipt[8]}:600" ] || return 1
  if [ "${receipt[2]}" = stopped ]; then
    [ "${receipt[3]}" = 0 ] && [ "${receipt[4]}" = no ] && \
      [ "${receipt[9]}" = 0 ] && [ "${receipt[10]}" = 0 ] && \
      [ "${receipt[11]}" = 0 ] || return 1
  else
    [[ "${receipt[3]}" =~ ^[0-9]+$ ]] && [[ "${receipt[8]}" =~ ^[0-9]+$ ]] && \
      [[ "${receipt[9]}" =~ ^[0-9]+$ ]] && \
      [[ "${receipt[10]}" =~ ^[a-f0-9]{64}$ ]] && \
      [[ "${receipt[11]}" =~ ^[a-f0-9]{64}$ ]] || return 1
  fi
}

receipt_process_matches() {
  local current
  current=$(puller_identity "${receipt[3]}") || return 1
  read -r -a current_fields <<<"$current"
  [ "${#current_fields[@]}" -eq 7 ] || return 1
  for index in 0 1 2 3 4 5 6; do
    [ "${receipt[index + 5]}" = "${current_fields[index]}" ] || return 1
  done
}

write_pause_receipt() {
  local replace=$1 temporary
  shift
  if [ "$replace" = no ]; then
    (
      umask 077
      set -o noclobber
      printf '%s\n' "$@" >"$pause_file"
    )
    return
  fi
  temporary=$(mktemp -- "$dest_dir/.rollout-pull-pause.XXXXXX") || return 1
  chmod 600 -- "$temporary" || { rm -f -- "$temporary"; return 1; }
  if ! printf '%s\n' "$@" >"$temporary" || ! mv -fT -- "$temporary" "$pause_file"; then
    rm -f -- "$temporary"
    return 1
  fi
}

matching_pids() {
  local proc candidate
  for proc in /proc/[0-9]*; do
    candidate=${proc##*/}
    process_matches "$candidate" && printf '%s\n' "$candidate"
  done
}

group_live() {
  ps -eo pgid=,stat= | awk -v group="$1" '$1 == group && $2 !~ /^Z/ {found=1} END {exit !found}'
}

pid_live() {
  local pid=$1 state
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  [ "$pid" -gt 1 ] || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [ "$state" != Z ]
}

lock_owner_safe() {
  local listing line pid path
  listing=$(lslocks -n -o PID,PATH --notruncate 2>/dev/null) || return 1
  [ -n "$listing" ] || return 0
  while IFS= read -r line; do
    [ -n "$line" ] || return 1
    read -r pid path <<<"$line"
    [ -n "${path:-}" ] || return 1
    [ "$path" = "$lock_file" ] || continue
    case "$pid" in
      -) ;;
      ''|*[!0-9]*) return 1 ;;
      *) pid_live "$pid" && return 1 ;;
    esac
  done <<<"$listing"
  return 0
}

cleanup_pid() {
  local flock_status command
  [ ! -L "$lock_file" ] || return 1
  if [ -e "$lock_file" ] && [ ! -f "$lock_file" ]; then
    return 1
  fi
  printf -v command \
    'if [ -L %q ] || { [ -e %q ] && [ ! -f %q ]; }; then exit 74; fi; rm -f -- %q || exit 74' \
    "$pid_file" "$pid_file" "$pid_file" "$pid_file"
  if flock -n -E 75 "$lock_file" -c "$command"; then
    return 0
  else
    flock_status=$?
  fi
  [ "$flock_status" -eq 75 ] || return 1

  # A contended lock is safe only when its kernel owner is gone or is the
  # explicit unresolved marker. A live owner may be another run and must stop
  # the cleanup.
  lock_owner_safe
}

stop_impl() {
  local pid any_live
  local -a pids=()
  stop_signalled=0
  stop_signalled_pids=()
  local_path_safe "$dest_dir" || return 1
  mapfile -t pids < <(matching_pids)
  for pid in "${pids[@]}"; do
    [ -z "$stop_target_pid" ] || [ "$pid" = "$stop_target_pid" ] || continue
    process_matches "$pid" || continue
    if [ -n "$stop_target_pid" ]; then receipt_process_matches || return 1; fi
    if kill -- "-$pid" 2>/dev/null; then
      stop_signalled=1
      stop_signalled_pids+=("$pid")
    elif group_live "$pid"; then
      return 1
    fi
  done
  for _ in $(seq 1 50); do
    any_live=0
    for pid in "${pids[@]}"; do group_live "$pid" && any_live=1; done
    [ "$any_live" = 1 ] || break
    sleep 0.1
  done
  for pid in "${pids[@]}"; do
    [ -z "$stop_target_pid" ] || [ "$pid" = "$stop_target_pid" ] || continue
    group_live "$pid" || continue
    process_matches "$pid" || return 1
    if [ -n "$stop_target_pid" ]; then receipt_process_matches || return 1; fi
    if kill -KILL -- "-$pid" 2>/dev/null; then
      stop_signalled=1
      stop_signalled_pids+=("$pid")
    else
      return 1
    fi
  done
  for _ in $(seq 1 50); do
    any_live=0
    for pid in "${pids[@]}"; do group_live "$pid" && any_live=1; done
    [ "$any_live" = 1 ] || break
    sleep 0.1
  done
  for pid in "${pids[@]}"; do group_live "$pid" && return 1; done
  [ -z "$(matching_pids)" ] || return 1
  if [ ! -e "$dest_dir" ]; then
    return 0
  fi
  [ -d "$dest_dir" ] && [ ! -L "$dest_dir" ] || return 1
  cleanup_pid
}

start_impl() {
  local_path_safe "$dest_dir" || return 1
  mkdir -p "$dest_dir"
  if status; then
    echo "rollout puller already running"
    return
  fi
  stop_impl
  nohup setsid env \
    ROLLOUT_TRANSFER_DIRECTION=trainer_pull \
    ROLLOUT_TRANSFER_BACKEND="$rollout_backend" \
    B2_ROLLOUT_PYTHON="$b2_python" B2_ROLLOUT_HELPER="$b2_helper" \
    B2_DOWNLOAD_WORKERS="$b2_download_workers" B2_DOWNLOAD_THRESHOLD_BYTES="$b2_download_threshold" \
    SOURCE_RUN_DIR="$source_dir" DEST_RUN_DIR="$dest_dir" \
    SOURCE_HOST="$source_host" SOURCE_PORT="$source_port" SOURCE_USER="$source_user" \
    SOURCE_IDENTITY_FILE="$source_identity" SOURCE_KNOWN_HOSTS_FILE="$source_known_hosts" \
    LOG_FILE="$log_file" \
    bash "$script" run 7>&- >>"$log_file" 2>&1 </dev/null &
  local pid=$!
  local deadline=$((SECONDS + 30))
  local failure="timeout"
  while :; do
    status && { echo "rollout puller started (pid $pid)"; return; }
    if ! kill -0 "$pid" 2>/dev/null; then
      failure="child exited"
      break
    fi
    [ "$SECONDS" -lt "$deadline" ] || break
    sleep 0.1
  done
  if ! stop_impl || [ -n "$(matching_pids)" ] || [ -e "$pid_file" ] || ! flock -n "$lock_file" -c true; then
    echo "rollout puller failed to start ($failure); cleanup failed" >&2
    return 1
  fi
  echo "rollout puller did not start ($failure)" >&2
  return 1
}

management_lock() {
  local old_umask
  local_path_safe "$dest_dir" || return 1
  mkdir -p "$dest_dir"
  [ ! -L "$control_file" ] || return 1
  if [ -e "$control_file" ] && [ ! -f "$control_file" ]; then return 1; fi
  old_umask=$(umask)
  umask 077
  exec 7>>"$control_file"
  umask "$old_umask"
  flock -x 7 || return 1
  python3 - "$control_file" "$BASHPID" <<'PY'
import os
import stat
import sys

path, pid = sys.argv[1:]
try:
    named = os.lstat(path)
    opened = os.stat(f"/proc/{pid}/fd/7")
except OSError:
    raise SystemExit(1)
if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
    raise SystemExit(1)
if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
    raise SystemExit(1)
PY
}

remove_pause_receipt() {
  local identity current
  [ -e "$pause_file" ] || [ -L "$pause_file" ] || return 0
  [ -f "$pause_file" ] && [ ! -L "$pause_file" ] || return 1
  identity=$(stat -c '%d:%i' -- "$pause_file") || return 1
  current=$(stat -c '%d:%i' -- "$pause_file") || return 1
  [ "$identity" = "$current" ] || return 1
  rm -- "$pause_file"
}

pause_action() {
  local token=$1 current_state was_running recorded_pid identity pid
  local stop_status=0 owns_restore
  local -a identity_fields=()
  [[ "$token" =~ ^[a-f0-9]{32}$ ]] || return 2
  if [ -e "$pause_file" ] || [ -L "$pause_file" ]; then
    [ -f "$pause_file" ] && [ ! -L "$pause_file" ] || return 1
    mapfile -t receipt <"$pause_file"
    receipt_valid "$token" || return 1
    identity_fields=("${receipt[@]:5:7}")
    current_state=$(state) || return 1
    if [ "$current_state" = stopped ]; then
      echo paused
      return 0
    fi
    [ "${receipt[2]}" = running ] || return 1
    recorded_pid=${receipt[3]}
    [ "$(cat "$pid_file" 2>/dev/null || true)" = "$recorded_pid" ] && \
      process_matches "$recorded_pid" && receipt_process_matches || return 1
  else
    current_state=$(state) || return 1
    was_running=stopped
    recorded_pid=0
    if [ "$current_state" = stopped ]; then
      identity=$(puller_identity 0) || return 1
    else
      was_running=running
      recorded_pid=$(cat "$pid_file") || return 1
      [[ "$recorded_pid" =~ ^[0-9]+$ ]] || return 1
      process_matches "$recorded_pid" || return 1
      identity=$(puller_identity "$recorded_pid") || return 1
    fi
    read -r -a identity_fields <<<"$identity"
    [ "${#identity_fields[@]}" -eq 7 ] || return 1
    receipt=("$token" "$dest_dir" "$was_running" "$recorded_pid" no "${identity_fields[@]}")
    write_pause_receipt no "${receipt[@]}" || return 1
    if [ "$was_running" = stopped ]; then
      echo paused
      return 0
    fi
  fi
  stop_target_pid=$recorded_pid
  if stop_impl; then
    stop_status=0
  else
    stop_status=$?
  fi
  stop_target_pid=
  owns_restore=no
  for pid in "${stop_signalled_pids[@]}"; do
    [ "$pid" = "$recorded_pid" ] && owns_restore=yes
  done
  if [ "$owns_restore" = yes ]; then
    [ "${#identity_fields[@]}" -eq 7 ] || return 1
    receipt[4]=yes
    write_pause_receipt yes "${receipt[@]}" || return 1
  fi
  if [ "$stop_status" != 0 ]; then
    echo "rollout puller pause did not finish; receipt retained" >&2
    return 1
  fi
  echo paused
}

resume_action() {
  local token=$1 state_value was_running recorded_pid
  [[ "$token" =~ ^[a-f0-9]{32}$ ]] || return 2
  if [ ! -e "$pause_file" ] && [ ! -L "$pause_file" ]; then
    echo not-owned
    return 0
  fi
  [ -f "$pause_file" ] && [ ! -L "$pause_file" ] || return 1
  mapfile -t receipt <"$pause_file"
  receipt_valid "$token" || return 1
  was_running=${receipt[2]}
  recorded_pid=${receipt[3]}
  if [ "$was_running" = stopped ] || [ "${receipt[4]}" = no ]; then
    remove_pause_receipt || return 1
    echo already-stopped
    return 0
  fi
  state_value=$(state) || return 1
  if [ "$state_value" = stopped ]; then
    start_impl >/dev/null || return 1
    status || return 1
    remove_pause_receipt || return 1
    echo restored
    return 0
  fi
  if [ "$(cat "$pid_file" 2>/dev/null || true)" != "$recorded_pid" ] || \
    ! process_matches "$recorded_pid" || ! receipt_process_matches; then
    remove_pause_receipt || return 1
    echo already-running
    return 0
  fi
  remove_pause_receipt || return 1
  echo already-running
}

case "$action" in
  state) state; exit ;;
  status) status; exit ;;
  start|stop|pause|resume)
    management_lock || { echo "rollout puller control lock is unsafe" >&2; exit 1; }
    case "$action" in
      start)
        if [ -e "$pause_file" ] || [ -L "$pause_file" ]; then echo "rollout puller is paused by a namespace transaction" >&2; exit 1; fi
        start_impl
        ;;
      stop)
        if [ -e "$pause_file" ] || [ -L "$pause_file" ]; then echo "rollout puller is paused by a namespace transaction" >&2; exit 1; fi
        stop_impl
        ;;
      pause) pause_action "${2:-}" ;;
      resume) resume_action "${2:-}" ;;
    esac
    exit
    ;;
  run) ;;
  *) echo "usage: $0 {run|start|stop|pause|resume|status|state}" >&2; exit 2 ;;
esac

local_path_safe "$dest_dir"
remote_path_safe "$source_dir"

cleanup() {
  if [ -n "$health_lease_pid" ]; then
    kill -- "-$health_lease_pid" 2>/dev/null || true
    wait "$health_lease_pid" 2>/dev/null || true
    health_lease_pid=
  fi
  if [ "$pid_owned" = 1 ]; then
    rm -f -- "$pid_file"
  fi
  [ -z "$rsync_ssh_wrapper" ] || rm -f -- "$rsync_ssh_wrapper"
  [ -z "$rsync_ssh_dir" ] || rmdir -- "$rsync_ssh_dir" 2>/dev/null || true
}

trap cleanup EXIT

rsync_ssh_dir=$(mktemp -d -- /tmp/rollout-pull-ssh.XXXXXX)
rsync_ssh_wrapper="$rsync_ssh_dir/ssh"
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf 'exec ssh'
  printf ' %q' "${ssh_opts[@]}"
  printf ' "$@"\n'
} >"$rsync_ssh_wrapper"
chmod 700 "$rsync_ssh_wrapper"
rsync_ssh="$rsync_ssh_wrapper"

mkdir -p "$dest_dir"
exec 8<"$dest_dir"
flock -s 8
exec 9>"$lock_file"
flock -n 9 || exit 0
echo "$$" >"$pid_file"
pid_owned=1

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

b2_remote() {
  local command status
  printf -v command '%q ' timeout --signal=TERM --kill-after=5s 300s "$b2_python" "$b2_helper" "$@"
  if timeout --signal=TERM --kill-after=5s 310s ssh -n "${ssh_opts[@]}" "$source_target" "$command"; then
    return 0
  else
    status=$?
  fi
  [ "$status" -eq 124 ] && return 75
  return "$status"
}

b2_retry() {
  local label=$1; shift
  local attempt status started elapsed bytes
  for attempt in 1 2 3; do
    started=$(date +%s)
    if "$@"; then
      return 0
    else
      status=$?
    fi
    [ "$status" -eq 75 ] || return "$status"
    elapsed=$(( $(date +%s) - started ))
    bytes=0
    if [ -n "${b2_retry_partial:-}" ] && [ -f "$b2_retry_partial" ]; then
      bytes=$(stat -c %s -- "$b2_retry_partial" 2>/dev/null || echo 0)
    fi
    echo "B2 transient failure label=$label attempt=$attempt elapsed=${elapsed}s bytes=$bytes error=timeout" >&2
    [ "$attempt" -lt 3 ] || return 75
    sleep "$attempt"
  done
  return 75
}

b2_finish() {
  local pending=$1 final=${1%.b2-ack.json} receipt digest relative source status
  local_path_safe "$pending" || return 76
  receipt=$(python3 "$b2_local_helper" ack-json --receipt "$pending" --final "$final") || { status=$?; [ "$status" -eq 75 ] && return 75; return 76; }
  relative=${final#"$dest_dir"/}
  [[ "$relative" =~ ^rollouts/step_[0-9]+/train_rollouts\.bin$ ]] || return 76
  source="$source_dir/$relative"
  digest=$(sha256sum -- "$final" | awk '{print $1}') || return 76
  write_receipt "$source" "$digest" || return 76
  b2_remote ack --receipt "$receipt" || { status=$?; [ "$status" -eq 75 ] && return 75; return 76; }
  python3 "$b2_local_helper" cleanup-parts --receipt "$pending" --final "$final" || { status=$?; [ "$status" -eq 75 ] && return 75; return 76; }
  rm -- "$pending"
}

pull_b2() {
  # Presigned URLs must not enter shell tracing, command arguments, or logs.
  set +x
  local source=$1 relative=${1#"$source_dir"/} final partial metadata pending
  final="$dest_dir/$relative"
  partial="$dest_dir/.incoming/$relative"
  pending="$final.b2-ack.json"
  local_path_safe "$final" || return 76
  local_path_safe "$partial" || return 76
  if [ -f "$pending" ]; then
    b2_finish "$pending" || return $?
    echo "pulled $relative via b2_s3"
    return 0
  fi
  metadata=$(b2_remote publish --source "$source") || { local status=$?; [ "$status" -eq 75 ] && return 75; return 76; }
  printf '%s' "$metadata" | python3 "$b2_local_helper" receive \
    --source "$source" --partial "$partial" --final "$final" \
    --download-workers "$b2_download_workers" --download-threshold-bytes "$b2_download_threshold" || { local status=$?; [ "$status" -eq 75 ] && return 75; return 76; }
  unset metadata
  b2_finish "$final.b2-ack.json" || return $?
  echo "pulled $relative via b2_s3"
}

pull_bytes() {
  local source=$1 partial=$2 digest=$3 size chunk remote_parts local_parts listing
  local part name expected status=0
  local -a pids=()
  size=$(remote "stat -c %s -- '$source'") || return 1
  if [ "$size" -lt "$parallel_threshold" ]; then
    pull_rsync -az --partial --append-verify --timeout=60 -e "$rsync_ssh" \
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
    pull_rsync -az --partial --append-verify --timeout=60 -e "$rsync_ssh" \
      "$source_target:$remote_parts/$name" "$part" 9>&- &
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
  if [ "$rollout_backend" = b2_s3 ] && [[ "$relative" =~ ^rollouts/step_[0-9]+/train_rollouts\.bin$ ]]; then
    pull_b2 "$source"
    return
  fi
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
  local local_file=$1 relative=${1#"$dest_dir"/} remote_file remote_step partial digest remote_digest
  [[ "$relative" =~ ^broadcasts/step_[0-9]+/STABLE$ ]] || return 1
  remote_file="$source_dir/$relative"
  remote_step=$(dirname "$remote_file")
  partial="$source_dir/.incoming/$relative"
  local_path_safe "$local_file" || return 1
  remote_path_safe "$remote_step" || return 1
  remote "directory='$remote_step'; parent=\${directory%/*}; [ -d \"\$parent\" ] && [ ! -L \"\$parent\" ] || exit 1; if [ -e \"\$directory\" ]; then [ -d \"\$directory\" ] && [ ! -L \"\$directory\" ] || exit 1; else mkdir -m 0755 -- \"\$directory\" || exit 1; chown 1000:1000 \"\$directory\" || exit 1; fi; [ \"\$(stat -c '%u:%g:%a' \"\$directory\")\" = 1000:1000:755 ]" || return 1
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
  local listing remote_file local_file step status=0 attempt pushed
  local -a rollout_files=()
  if [ "$rollout_backend" = b2_s3 ] && [ -d "$dest_dir/rollouts" ]; then
    while IFS= read -r -d '' local_file; do
      b2_retry_partial="${local_file%.b2-ack.json}" b2_retry "ack:${local_file#"$dest_dir"/}" b2_finish "$local_file" || return $?
    done < <(find "$dest_dir/rollouts" -type f -name 'train_rollouts.bin.b2-ack.json' -print0)
  fi
  while IFS= read -r -d '' local_file; do
    pushed=0
    for attempt in 1 2 3; do
      if push_stable "$local_file"; then
        pushed=1
        break
      fi
      [ "$attempt" -lt 3 ] && sleep 1
    done
    if [ "$pushed" = 0 ]; then
      echo "stable marker push failed after 3 attempts: ${local_file#"$dest_dir"/}" >&2
      return 1
    fi
  done < <(find "$dest_dir/broadcasts" -mindepth 2 -maxdepth 2 -type f -name STABLE -print0 2>/dev/null | sort -z)
  listing=$(remote "{ printf '%s\\n' '$source_dir/control/orch.toml'; find '$source_dir/broadcasts' '$source_dir/rollouts' -type f \( -path '*/broadcasts/step_*/receivers-ready.json' -o -path '*/rollouts/step_*/train_rollouts.bin' \) -print 2>/dev/null; } | while IFS= read -r file; do [ -f \"\$file\" ] || continue; marker=\"\$file.delivered\"; [ ! -L \"\$marker\" ] || exit 1; if [ -e \"\$marker\" ]; then [ -f \"\$marker\" ] || exit 1; continue; fi; if [ \"\$file\" = '$source_dir/control/orch.toml' ] && [ \$(( \$(date +%s) - \$(stat -c %Y \"\$file\") )) -lt 2 ]; then continue; fi; printf '%s\\n' \"\$file\"; done") || return 1
  while IFS= read -r remote_file; do
    [ -n "$remote_file" ] || continue
    if [[ "$remote_file" =~ /rollouts/step_[0-9]+/train_rollouts\.bin$ ]]; then
      rollout_files+=("$remote_file")
    else
      pull_one "$remote_file" || status=1
    fi
  done <<<"$listing"
  [ "$status" = 0 ] || return "$status"
  while IFS=$'\t' read -r step remote_file; do
    [ -n "$remote_file" ] || continue
    if [[ "$remote_file" =~ /rollouts/step_[0-9]+/train_rollouts\.bin$ ]] && [ "$rollout_backend" = b2_s3 ]; then
      b2_retry_partial="$dest_dir/.incoming/${remote_file#"$source_dir"/}" b2_retry "pull:${remote_file#"$source_dir"/}" pull_one "$remote_file" || return $?
    else
      pull_one "$remote_file" || return $?
    fi
  done < <(
    for remote_file in "${rollout_files[@]}"; do
      step=${remote_file##*/step_}
      step=${step%/train_rollouts.bin}
      printf '%s\t%s\n' "$step" "$remote_file"
    done | sort -n -k1,1 -k2,2
  )
  return "$status"
}

refresh_health_lease() {
  local host state now probe_status unknown=0 missing=0 health_state
  health_probe_pids=()
  for host in $trainer_addresses; do
    (
      if state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$host" \
        "sudo docker ps --filter name=^${trainer_container}$ --format '{{.Names}}' 2>/dev/null" \
        2>/dev/null); then
        [ "$state" = "$trainer_container" ] || exit 1
      else
        exit 2
      fi
    ) 9>&- &
    health_probe_pids+=("$!")
  done
  for pid in "${health_probe_pids[@]}"; do
    if wait "$pid"; then
      continue
    else
      probe_status=$?
      case "$probe_status" in
        1) missing=1 ;;
        *) unknown=1 ;;
      esac
    fi
  done
  health_probe_pids=()
  health_state=healthy
  [ "$unknown" = 0 ] || health_state=unknown
  [ "$unknown" = 1 ] || { [ "$missing" = 0 ] || health_state=missing; }
  now=$(date +%s)
  health_publish_state "$health_state" "$now" || return 1
  [ "$health_state" = healthy ]
}

health_publish_state() {
  local health_state=$1 now=$2
  health_remote "set -eu; status='$source_dir/.trainer-health-status'; partial=\$(mktemp -- \"\${status}.tmp.XXXXXX\"); printf '%s %s\\n' '$health_state' '$now' >\"\$partial\"; chmod 600 \"\$partial\"; chown 1000:1000 \"\$partial\"; mv -- \"\$partial\" \"\$status\"; if [ '$health_state' = healthy ]; then lease='$source_dir/.trainer-health-lease'; lease_partial=\$(mktemp -- \"\${lease}.tmp.XXXXXX\"); printf '%s\\n' '$now' >\"\$lease_partial\"; chown 1000:1000 \"\$lease_partial\"; mv -- \"\$lease_partial\" \"\$lease\"; fi"
}

health_lease_shutdown() {
  local pid
  trap - TERM INT EXIT
  for pid in "${health_probe_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${health_probe_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  health_probe_pids=()
  exit 0
}

health_lease_loop() {
  trap health_lease_shutdown TERM INT EXIT
  while :; do
    if ! refresh_health_lease; then
      echo "$(date -u +%FT%TZ) trainer health lease update failed; retrying" >&2
    fi
    sleep "${HEALTH_LEASE_INTERVAL:-30}" || exit 0
  done
}

start_health_lease() {
  [ "${ONCE:-0}" = 1 ] && return
  export source_dir source_host source_port source_user source_identity source_known_hosts
  export trainer_addresses trainer_container
  export -f health_remote health_publish_state refresh_health_lease health_lease_shutdown health_lease_loop
  setsid bash -c 'exec -a rollout-pull-health bash -c health_lease_loop' 9>&- &
  health_lease_pid=$!
}

health_remote() {
  ssh -n -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$source_known_hosts" \
    -o GlobalKnownHostsFile=/dev/null -o UpdateHostKeys=no \
    -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
    -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
    -i "$source_identity" -p "$source_port" "$source_user@$source_host" "$1"
}

if [ "$rollout_backend" = b2_s3 ]; then
  [ -f "$b2_local_helper" ] || { echo "B2 receiver helper is missing" >&2; exit 76; }
  command -v curl >/dev/null || { echo "curl is missing" >&2; exit 76; }
  b2_retry preflight b2_remote preflight >/dev/null || { status=$?; exit "$status"; }
fi
start_health_lease
while true; do
  status=0
  scan || {
    failure=$?
    if [ "$failure" = 75 ]; then
      echo "B2 rollout transfer exhausted its bounded timeout retries" >&2
      exit 75
    fi
    if [ "$failure" = 76 ]; then
      echo "B2 rollout transfer failed; stopped without retry" >&2
      exit 76
    fi
    echo "$(date -u +%FT%TZ) rollout pull failed; retrying" >&2
    status=1
  }
  if [ "${ONCE:-0}" = 1 ]; then
    refresh_health_lease || {
      echo "$(date -u +%FT%TZ) trainer health lease update failed; retrying" >&2
      status=1
    }
  fi
  [ "${ONCE:-0}" = 1 ] && exit "$status"
  sleep "$interval"
done
