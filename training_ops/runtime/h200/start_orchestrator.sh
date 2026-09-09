#!/usr/bin/env bash
# Run the orchestrator on h200a instead of n82.
# Engine traffic stays local. Only adapter and completed batch data cross the
# China link.
set -uo pipefail

tunnels_only=0
for argument in "$@"; do
  case "$argument" in
    --tunnels-only) tunnels_only=1 ;;
    *) echo "unknown launcher argument: $argument" >&2; exit 2 ;;
  esac
done

RUN=${RUN:?set RUN}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
TRAINER_ADDRESSES=${TRAINER_ADDRESSES:?set TRAINER_ADDRESSES}
TRAINER_CT=${TRAINER_CT:?set TRAINER_CT}
ORCH_CT=${ORCH_CT:?set ORCH_CT}
deployment=/data/glm52-rl/deployment
config_file=${ORCH_CONFIG:-$deployment/configs/general-agent/orchestrator.toml}
nodes_file="$deployment/configs/nodes.toml"
test -f "$config_file" || {
  echo "orchestrator config is missing: $config_file" >&2
  exit 2
}
image=prime-rl:glm52-decensor
readarray -t lora_source_ports < <(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
ports = config["ports"]
first = ports["lora_source_first"]
last = ports["lora_source_last"]
if not isinstance(first, int) or not isinstance(last, int) or first < 1 or last < first:
    raise SystemExit("invalid lora source port range")
for port in range(first, last + 1):
    print(port)
PY
)
if [ "${#lora_source_ports[@]}" -lt 1 ]; then
  echo "nodes config has no LoRA source ports" >&2
  exit 2
fi
adapter_port=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
value = config["ports"].get("adapter")
if not isinstance(value, int) or not 1 <= value <= 65535:
    raise SystemExit("nodes config has no valid adapter port")
print(value)
PY
)
if [ -z "$adapter_port" ]; then
  echo "nodes config has no adapter port" >&2
  exit 2
fi
if ! relay_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import sys
import tomllib
from urllib.parse import urlsplit

with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    edge = tomllib.load(stream)
ports = nodes.get("ports")
if not isinstance(ports, dict):
    raise SystemExit("nodes config has no port inventory")
relay_port = ports.get("adapter_relay")
source_first = ports.get("lora_source_first")
source_last = ports.get("lora_source_last")
relay_container = nodes.get("containers", {}).get("adapter_relay")
cache_dir = nodes.get("paths", {}).get("adapter_cache")
source_url = edge.get("weight_broadcast", {}).get("source_url")
model_name = edge.get("model", {}).get("name")
glm_model = nodes.get("models", {}).get("glm52")
if not isinstance(relay_port, int) or not 1 <= relay_port <= 65535:
    raise SystemExit("nodes config has no valid adapter relay port")
if (
    not isinstance(source_first, int)
    or not isinstance(source_last, int)
    or not 1 <= source_first <= source_last <= 65535
):
    raise SystemExit("nodes config has no valid adapter source range")
port_values = {}
for name, value in ports.items():
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise SystemExit(f"nodes port {name} is invalid")
    port_values[name] = value
source_ports = set(range(source_first, source_last + 1))
reserved_names = set(port_values) - {"lora_source_first", "lora_source_last"}
reserved_ports = {name: port_values[name] for name in reserved_names}
collisions = {
    name: value
    for name, value in reserved_ports.items()
    if value in source_ports
}
if collisions:
    raise SystemExit(f"configured port overlaps the adapter source range: {collisions}")
duplicate_ports = {}
for name, value in reserved_ports.items():
    duplicate_ports.setdefault(value, []).append(name)
duplicates = [names for names in duplicate_ports.values() if len(names) > 1]
if duplicates:
    raise SystemExit(f"configured ports are duplicated: {duplicates}")
if not isinstance(relay_container, str) or not relay_container:
    raise SystemExit("nodes config has no adapter relay container")
if not isinstance(cache_dir, str) or not cache_dir.startswith("/"):
    raise SystemExit("nodes config has no absolute adapter cache path")
if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
    raise SystemExit("orchestrator config has no canonical adapter source URL")
if not isinstance(model_name, str) or not isinstance(glm_model, str):
    raise SystemExit("model names are missing from the nodes or orchestrator config")
parsed = urlsplit(source_url)
if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.query or parsed.fragment:
    raise SystemExit("canonical adapter source URL must be loopback without query or fragment")
if parsed.port != source_first:
    raise SystemExit("canonical adapter source URL must use lora_source_first")
print(relay_port, relay_container, cache_dir, source_url, int(model_name == glm_model))
PY
); then
  echo "cannot read the configured adapter relay" >&2
  exit 2
fi
read -r relay_port relay_container relay_cache_dir relay_source_url relay_is_glm <<<"$relay_settings"
trainer_ssh_target=$(python3 - "$nodes_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
target = config["hosts"]["trainer_ssh"]
if not isinstance(target, str) or not target:
    raise SystemExit("nodes config has no trainer SSH target")
print(target)
PY
)
if [ -z "$trainer_ssh_target" ]; then
  echo "nodes config has no trainer SSH target" >&2
  exit 2
fi
transport=$(awk '
  $0 == "[rollout_transport]" { active=1; next }
  /^\[/ { active=0 }
  active && $1 == "type" { gsub(/[ "\047]/, "", $3); print $3; exit }
' "$config_file")

if ! peer_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import ipaddress
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    edge = tomllib.load(stream)

hosts = nodes.get("hosts")
inference_hosts = hosts.get("inference") if isinstance(hosts, dict) else None
orchestrator = hosts.get("orchestrator") if isinstance(hosts, dict) else None
if not isinstance(inference_hosts, list) or not all(isinstance(item, str) for item in inference_hosts):
    raise SystemExit("nodes config has no valid inference host list")
peers = [item for item in inference_hosts if item != orchestrator]
if len(peers) > 1:
    raise SystemExit("two-H200 routing supports one inference peer")
if not peers:
    print("none - 0 0 - 0 0")
    raise SystemExit

peer_name = peers[0]
peer = nodes.get("inference", {}).get(peer_name)
if not isinstance(peer, dict):
    raise SystemExit(f"inference settings for {peer_name} are missing")
host = peer.get("ssh_host")
port = peer.get("ssh_port")
user = peer.get("ssh_user")
identity = peer.get("peer_ssh_identity_file")
api_port = peer.get("api_tunnel_port")
if not isinstance(host, str):
    raise SystemExit(f"inference.{peer_name}.ssh_host is missing")
try:
    ipaddress.ip_address(host)
except ValueError as error:
    raise SystemExit(f"inference.{peer_name}.ssh_host must be an IP address") from error
if not isinstance(port, int) or not 1 <= port <= 65535:
    raise SystemExit(f"inference.{peer_name}.ssh_port is invalid")
if not isinstance(user, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", user) is None:
    raise SystemExit(f"inference.{peer_name}.ssh_user is invalid")
if not isinstance(identity, str) or not identity.startswith("/") or any(part in {"", ".", ".."} for part in identity.split("/")[1:]):
    raise SystemExit(f"inference.{peer_name}.peer_ssh_identity_file is invalid")
if not isinstance(api_port, int) or not 1 <= api_port <= 65535:
    raise SystemExit(f"inference.{peer_name}.api_tunnel_port is invalid")

client = edge.get("model", {}).get("client", {})
urls = []
for field in ("base_url", "admin_base_url"):
    value = client.get(field, []) if isinstance(client, dict) else []
    if isinstance(value, str):
        urls.append(value)
    elif isinstance(value, list):
        urls.extend(item for item in value if isinstance(item, str))
expected = f"http://127.0.0.1:{api_port}/v1"
enabled = int(expected in urls)
print(peer_name, host, port, user, identity, api_port, enabled)
PY
); then
  echo "cannot read the configured inference peer" >&2
  exit 2
fi
read -r peer_name peer_host peer_ssh_port peer_ssh_user peer_ssh_identity peer_api_port peer_enabled <<<"$peer_settings"
if [ "$peer_enabled" = 1 ] && [ "$relay_is_glm" != 1 ]; then
  echo "adapter relay peer mode is only supported for the configured GLM model" >&2
  exit 2
fi

if [ "$tunnels_only" = 0 ] && grep -q 'refusals-classifier-eval-v1' "$config_file"; then
  bash "$deployment/training_ops/runtime/h200/start_refusals_classifier.sh"
fi

tunnel() { # local_port  target_host:port  ssh_target  [compress]
  local lp=$1 tgt=$2 via=$3 comp=${4:-no}
  pgrep -f -- "-L 127.0.0.1:$lp:" >/dev/null && return 0
  setsid nohup bash -c "while true; do
      ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
          -o IPQoS=none -o Compression=$comp \
          -L 127.0.0.1:$lp:$tgt $via
      sleep 3
    done" >/dev/null 2>&1 < /dev/null &
}

peer_tunnel_connection() {
  local connection_name=$1 signature=$2
  shift 2
  local -a forwards=("$@")

  local -a ssh_args=(
    -NT
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -i "$peer_ssh_identity"
    -p "$peer_ssh_port"
  )
  local forward_text ssh_text peer_target marker pid command_line process_matches forward_count
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  marker="peer-tunnel-$connection_name"
  for pid in $(pgrep -f -- "$signature" || true); do
    command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
    process_matches=1
    for forward in "${forwards[@]}"; do
      case "$command_line" in *"$forward"*) ;; *) process_matches=0; break ;; esac
    done
    forward_count=$(grep -o -- '-[LR] ' <<<"$command_line" | wc -l)
    [ "$forward_count" -eq "$(( ${#forwards[@]} / 2 ))" ] || process_matches=0
    case "$command_line" in
      *"ssh -NT "*"-o IdentitiesOnly=yes"*"-o ExitOnForwardFailure=yes"*"-i $peer_ssh_identity"*"-p $peer_ssh_port "*"$peer_target"*"# $marker") ;;
      *) process_matches=0 ;;
    esac
    [ "$process_matches" = 1 ] && return 0
    peer_kill_owned "$pid" "$command_line"
  done
  setsid nohup bash -c "while true; do
      ssh$ssh_text$forward_text $peer_target
      sleep 3
    done # $marker" >/dev/null 2>&1 < /dev/null &
}

peer_tunnel_remove_legacy() {
  local keep_current=${1:-0}
  local api_signature="-L 127.0.0.1:$peer_api_port:127.0.0.1:8000"
  local reverse_signature="-R 127.0.0.1:${lora_source_ports[0]}:"
  local current_signature pid command_line lora_source_port
  for pid in $(pgrep -f -- "$api_signature" || true); do
    command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
    case "$command_line" in
      *"$api_signature"*)
        case "$command_line" in *"while true; do"*"# peer-tunnel-"*) ;; *) continue ;; esac
        if [ "$keep_current" = 1 ]; then
          case "$command_line" in *"$reverse_signature"*) ;; *) continue ;; esac
        fi
        peer_kill_owned "$pid" "$command_line"
        sleep 1
        ;;
    esac
  done
  for lora_source_port in "${lora_source_ports[@]}"; do
    reverse_signature="-R 127.0.0.1:${lora_source_port}:"
    current_signature="-R 127.0.0.1:${lora_source_port}:127.0.0.1:$relay_port"
    for pid in $(pgrep -f -- "$reverse_signature" || true); do
      command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
      case "$command_line" in
        *"$reverse_signature"*)
          case "$command_line" in *"while true; do"*"# peer-tunnel-"*) ;; *) continue ;; esac
          case "$command_line" in
            *"$current_signature"*) [ "$keep_current" = 1 ] && continue ;;
          esac
          peer_kill_owned "$pid" "$command_line"
          ;;
      esac
    done
  done
}

peer_kill_owned() {
  local pid=$1 command_line=$2 pgid group_command group_safe group_seen peer_forward
  case "$command_line" in
    *"while true; do"*"# peer-tunnel-"*) ;;
    *) return 0 ;;
  esac
  peer_forward=$(grep -o -- '-[LR] [^ ]*' <<<"$command_line" | head -n 1)
  [ -n "$peer_forward" ] || { kill "$pid" 2>/dev/null || true; return 0; }
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$") kill "$pid" 2>/dev/null || true ;;
    *)
      group_safe=1
      group_seen=0
      while IFS= read -r group_command; do
        group_seen=1
        case "$group_command" in
          *"# peer-tunnel-"*|*" sleep 3"*) ;;
          *"ssh "*"-i $peer_ssh_identity"*"-p $peer_ssh_port "*"$peer_forward"*) ;;
          *) group_safe=0; break ;;
        esac
      done < <(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}')
      if [ "$group_safe" = 1 ] && [ "$group_seen" = 1 ]; then
        kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      else
        kill "$pid" 2>/dev/null || true
      fi
      ;;
  esac
}

peer_remote_ports_ready() {
  local probe_seconds=${1:-5} peer_target remote_ports port
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  if ! remote_ports=$(timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout="$probe_seconds" \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
      -o IPQoS=none -i "$peer_ssh_identity" -p "$peer_ssh_port" \
      "$peer_target" "ss -H -ltn | awk '{print \$4}'" 2>/dev/null); then
    return 1
  fi
  for port in "${lora_source_ports[@]}"; do
    printf '%s\n' "$remote_ports" | grep -Eq "([.:])$port$" || return 1
  done
}

tunnel_command_has() {
  local command_line=$1 token=$2
  case " $command_line " in
    *" $token "*|*" $token;"*) return 0 ;;
    *) return 1 ;;
  esac
}

tunnel_listener_ready() {
  local port=$1 marker=$2 target=$3 forward_count=$4
  shift 4
  local -a forwards=("$@")
  local listener pid ppid owner_command supervisor_command forward forward_total
  listener=$(ss -H -ltnp 2>/dev/null | awk -v port=":$port" '$4 ~ port "$" {print; exit}')
  [ -n "$listener" ] || return 1
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  owner_command=$(ps -o args= -p "$pid" 2>/dev/null || true)
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$ppid" != 1 ] || return 1
  forward_total=$(grep -o -- '-[LR] ' <<<"$owner_command" | wc -l)
  [ "$forward_total" -eq "$forward_count" ] || return 1
  tunnel_command_has "$owner_command" ssh || return 1
  tunnel_command_has "$owner_command" "$target" || return 1
  for forward in "${forwards[@]}"; do
    tunnel_command_has "$owner_command" "$forward" || return 1
  done
  supervisor_command=$(ps -o args= -p "$ppid" 2>/dev/null || true)
  case "$supervisor_command" in *"while true; do"*) ;; *) return 1 ;; esac
  tunnel_command_has "$supervisor_command" "$target" || return 1
  tunnel_command_has "$supervisor_command" "# $marker" || return 1
  forward_total=$(grep -o -- '-[LR] ' <<<"$supervisor_command" | wc -l)
  [ "$forward_total" -eq "$forward_count" ] || return 1
  for forward in "${forwards[@]}"; do
    tunnel_command_has "$supervisor_command" "$forward" || return 1
  done
}

china_tunnel_grouped_legacy_match() {
  local command_line=$1 source_port forward_total configured_total
  case "$command_line" in
    *"while true; do"*"$trainer_ssh_target"*"# china-tunnel-lora-"*) ;;
    *) return 1 ;;
  esac
  forward_total=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
  [ "$forward_total" -eq 4 ] || return 1
  configured_total=0
  for source_port in "${lora_source_ports[@]}"; do
    case "$command_line" in
      *"-L 127.0.0.1:$source_port:127.0.0.1:$adapter_port"*)
        configured_total=$((configured_total + 1))
        ;;
    esac
  done
  [ "$configured_total" -eq 4 ]
}

china_tunnel_remove_legacy() {
  local only_marker=${1:-}
  local lora_source_port pattern pid command_line pgid group_command group_safe group_seen cleanup_port
  local grouped_legacy
  local -a cleanup_ports=()
  for cleanup_port in "${@:2}"; do
    case "$cleanup_port" in
      127.0.0.1:*:127.0.0.1:*)
        cleanup_port=${cleanup_port#127.0.0.1:}
        cleanup_ports+=("${cleanup_port%%:*}")
        ;;
      [0-9]*) cleanup_ports+=("$cleanup_port") ;;
    esac
  done
  [ "${#cleanup_ports[@]}" -gt 0 ] || cleanup_ports=("${lora_source_ports[@]}")
  for lora_source_port in "${cleanup_ports[@]}"; do
    pattern="-L 127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
    for pid in $(pgrep -f -- "$pattern" || true); do
      command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
      grouped_legacy=0
      if [ -n "$only_marker" ]; then
        case "$command_line" in
          *"while true; do"*"$pattern"*"# $only_marker") ;;
          *) continue ;;
        esac
      else
        case "$command_line" in
          *"while true; do"*"$pattern"*"$trainer_ssh_target"*)
            china_tunnel_grouped_legacy_match "$command_line" || continue
            grouped_legacy=1
            ;;
          *) continue ;;
        esac
      fi
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ "$grouped_legacy" = 1 ]; then
        china_legacy_cleanup_required=1
        china_legacy_cleanup_pids+=("$pid")
        case "$pgid" in
          ''|*[!0-9]*|0|1) ;;
          *)
            case " ${china_legacy_cleanup_pgids[*]} " in
              *" $pgid "*) ;;
              *) china_legacy_cleanup_pgids+=("$pgid") ;;
            esac
            ;;
        esac
      fi
      case "$pgid" in
        ''|*[!0-9]*|0|1|"$$") kill "$pid" 2>/dev/null || true ;;
        *)
          group_safe=1
          group_seen=0
          while IFS= read -r group_command; do
            group_seen=1
            if [ -n "$only_marker" ]; then
              case "$group_command" in
                *"$pattern"*|*"# $only_marker"|*"sleep 3"*) ;;
                *) group_safe=0; break ;;
              esac
            else
              case "$group_command" in
                *"$pattern"*"$trainer_ssh_target"*|*"sleep 3"*) ;;
                *) group_safe=0; break ;;
              esac
            fi
          done < <(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}')
          if [ "$group_safe" = 1 ] && [ "$group_seen" = 1 ]; then
            kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
          else
            kill "$pid" 2>/dev/null || true
          fi
          ;;
      esac
    done
  done
}

china_tunnel_wait_legacy_clear() {
  local pid pgid group_processes source_port listener_remaining
  for _ in $(seq 1 30); do
    listener_remaining=0
    for pid in "${china_legacy_cleanup_pids[@]}"; do
      [ -z "$(ps -o args= -p "$pid" 2>/dev/null || true)" ] || {
        listener_remaining=1
        break
      }
    done
    if [ "$listener_remaining" = 0 ]; then
      for pgid in "${china_legacy_cleanup_pgids[@]}"; do
        group_processes=$(ps -eo pgid=,args= 2>/dev/null | awk -v group="$pgid" '$1 == group {print; exit}' || true)
        [ -z "$group_processes" ] || {
          listener_remaining=1
          break
        }
      done
    fi
    if [ "$listener_remaining" = 0 ]; then
      for source_port in "${lora_source_ports[@]}"; do
        if ss -H -ltn 2>/dev/null | awk -v port=":$source_port" '$4 ~ port "$" {found=1} END {exit !found}'; then
          listener_remaining=1
          break
        fi
      done
    fi
    [ "$listener_remaining" = 0 ] && return 0
    sleep 1
  done
  echo "stale China adapter tunnels did not stop before one-port migration" >&2
  return 1
}

china_tunnel_connection() {
  local connection_name=$1
  shift
  local -a forwards=("$@")

  local -a ssh_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=no
  )
  local forward_text ssh_text target marker pid command_line process_matches forward forward_count
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  target=$(printf '%q' "$trainer_ssh_target")
  marker="china-tunnel-$connection_name"
  for pid in $(pgrep -f -- "$marker" || true); do
    command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
    process_matches=1
    for forward in "${forwards[@]}"; do
      case "$command_line" in *"$forward"*) ;; *) process_matches=0; break ;; esac
    done
    forward_count=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
    [ "$forward_count" -eq "$(( ${#forwards[@]} / 2 ))" ] || process_matches=0
    case "$command_line" in *"$target"*"# $marker") ;; *) process_matches=0 ;; esac
    [ "$process_matches" = 1 ] && return 0
  done
  china_tunnel_remove_legacy "$marker" "${forwards[@]}"
  setsid nohup bash -c "while true; do
      ssh$ssh_text$forward_text $target
      sleep 3
    done # $marker" >/dev/null 2>&1 < /dev/null &
}

peer_tunnel() {
  peer_tunnel_remove_legacy "$peer_enabled"
  [ "$peer_enabled" = 1 ] || return 0
  peer_tunnel_connection "api" \
    "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" \
    -L "127.0.0.1:$peer_api_port:127.0.0.1:8000"

  # Keep each SSH cipher worker to four source ports without creating one SSH
  # connection for every port.
  local peer_lane_size=4
  local lane_start=0 lane_number=0
  local -a lane_ports lane_forwards
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    lane_ports=("${lora_source_ports[@]:lane_start:peer_lane_size}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=(
        -R "127.0.0.1:$lora_source_port:127.0.0.1:$relay_port"
      )
    done
    peer_tunnel_connection "lora-$lane_number" \
      "-R 127.0.0.1:${lane_ports[0]}:127.0.0.1:$relay_port" \
      "${lane_forwards[@]}"
    lane_start=$((lane_start + peer_lane_size))
    lane_number=$((lane_number + 1))
  done
  if [ "${tunnels_only:-0}" = 0 ]; then
    echo "peer tunnel :$peer_api_port -> $peer_name:8000; adapter source reverse ports ${lora_source_ports[0]}-${lora_source_ports[-1]} in $lane_number SSH lanes"
  fi
}

china_legacy_cleanup_required=0
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
china_tunnel_remove_legacy
if [ "$china_legacy_cleanup_required" = 1 ] && ! china_tunnel_wait_legacy_clear; then
  exit 1
fi
china_lane_size=1
china_lane_start=0 china_lane_number=0
china_lane_ports=() china_lane_forwards=()
while [ "$china_lane_start" -lt "${#lora_source_ports[@]}" ]; do
  china_lane_ports=("${lora_source_ports[@]:china_lane_start:china_lane_size}")
  china_lane_forwards=()
  for lora_source_port in "${china_lane_ports[@]}"; do
    china_lane_forwards+=(
      -L "127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port"
    )
  done
  china_tunnel_connection "lora-$china_lane_number" \
    "${china_lane_forwards[@]}"
  china_lane_start=$((china_lane_start + china_lane_size))
  china_lane_number=$((china_lane_number + 1))
done
if [ "$tunnels_only" = 0 ]; then
  echo "China adapter forwards ${lora_source_ports[0]}-${lora_source_ports[-1]} in $china_lane_number SSH lanes"
fi
peer_tunnel
if [ "$tunnels_only" = 1 ]; then
  tunnels_ready=0
  readiness_port=""
  for _ in $(seq 1 30); do
    tunnels_ready=1
    for port in "${lora_source_ports[@]}"; do
      port_index=$((port - lora_source_ports[0]))
      lane_number=$((port_index / china_lane_size))
      lane_start=$((lane_number * china_lane_size))
      lane_ports=("${lora_source_ports[@]:lane_start:china_lane_size}")
      lane_forwards=()
      for lane_port in "${lane_ports[@]}"; do
        lane_forwards+=(
          -L "127.0.0.1:$lane_port:127.0.0.1:$adapter_port"
        )
      done
      listener_status=0
      tunnel_listener_ready "$port" "china-tunnel-lora-$lane_number" \
        "$trainer_ssh_target" "${#lane_ports[@]}" "${lane_forwards[@]}" || listener_status=$?
      if [ "$listener_status" -ne 0 ]; then
        tunnels_ready=0
        readiness_port=$port
        break
      fi
    done
    if [ "$tunnels_ready" = 1 ] && [ "$peer_enabled" = 1 ]; then
      listener_status=0
      tunnel_listener_ready "$peer_api_port" peer-tunnel-api \
        "$peer_ssh_user@$peer_host" 1 \
        -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" || listener_status=$?
      if [ "$listener_status" -ne 0 ]; then
        tunnels_ready=0
        readiness_port=$peer_api_port
      fi
    fi
    [ "$tunnels_ready" = 1 ] && break
    sleep 1
  done
  if [ "$tunnels_ready" != 1 ]; then
    if [ -n "$readiness_port" ]; then
      echo "adapter tunnel readiness failed for local port $readiness_port: listener owner is not the expected SSH supervisor" >&2
    else
      echo "adapter tunnel reconciliation did not establish all configured forwards" >&2
    fi
    exit 1
  fi
  remote_ready=1
  if [ "$peer_enabled" = 1 ]; then
    remote_ready=0
    for _ in $(seq 1 10); do
      if peer_remote_ports_ready 5; then
        remote_ready=1
        break
      fi
      sleep 1
    done
  fi
  if [ "$remote_ready" != 1 ]; then
    echo "adapter tunnel reconciliation did not establish all remote reverse forwards" >&2
    exit 1
  fi
  echo "adapter tunnel reconciliation complete"
  exit 0
fi
ports=("${lora_source_ports[@]}")
if [ "$peer_enabled" = 1 ]; then
  ports+=("$peer_api_port")
fi
sudo install -d -o 1000 -g 1000 -m 0755 "$relay_cache_dir"
if [ "$peer_enabled" = 1 ]; then
  sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
  if ! sudo docker run -d --name "$relay_container" --restart unless-stopped \
    --user 0:0 \
    --network host \
    --log-opt max-size=128m --log-opt max-file=2 \
    -v "$relay_cache_dir":"$relay_cache_dir":ro \
    -v "$deployment/prime-rl/src/prime_rl/inference/relay.py":/app/relay.py:ro \
    --entrypoint /app/.venv/bin/python \
    "$image" /app/relay.py \
    --host 127.0.0.1 --port "$relay_port" \
    --cache-dir "$relay_cache_dir" \
    --source-port-first "${lora_source_ports[0]}" \
    --source-port-last "${lora_source_ports[-1]}" \
    --canonical-source-url "$relay_source_url" >/dev/null; then
    echo "adapter relay failed to start" >&2
    exit 1
  fi
  relay_ready=0
  for _ in $(seq 1 60); do
    running=$(sudo docker inspect "$relay_container" --format '{{.State.Running}}' 2>/dev/null || true)
    if [ "$running" = true ] && ss -tln 2>/dev/null | grep -q "127.0.0.1:$relay_port"; then
      relay_ready=1
      break
    fi
    sleep 1
  done
  if [ "$relay_ready" != 1 ]; then
    echo "adapter relay did not become ready on 127.0.0.1:$relay_port" >&2
    sudo docker logs --tail 100 "$relay_container" >&2 || true
    sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
    exit 1
  fi
  ports+=("$relay_port")
else
  sudo docker rm -f "$relay_container" >/dev/null 2>&1 || true
fi
if [ "$transport" = zmq ]; then
  tunnel 29620 10.92.20.82:29620 "$trainer_ssh_target" yes
  ports+=(29620)
fi

sleep 6
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN"
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN/$RUN_NAME"
sudo install -d -o 1000 -g 1000 "/data/glm52-rl/runs/$RUN/$RUN_NAME/hf-cache"
broadcast_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME/broadcasts"
for path in "$broadcast_dir" "$broadcast_dir/step_0"; do
  [ ! -L "$path" ] || {
    echo "broadcast path is a symlink: $path" >&2
    exit 1
  }
done
for step_dir in "$broadcast_dir"/step_*; do
  [ -d "$step_dir" ] || continue
  [[ "$step_dir" =~ /step_[0-9]+$ ]] || continue
  [ ! -L "$step_dir" ] || {
    echo "broadcast step path is a symlink: $step_dir" >&2
    exit 1
  }
done
sudo install -d -o 1000 -g 1000 -m 0755 "$broadcast_dir" "$broadcast_dir/step_0"
for step_dir in "$broadcast_dir"/step_*; do
  [ -d "$step_dir" ] || continue
  [[ "$step_dir" =~ /step_[0-9]+$ ]] || continue
  sudo install -d -o 1000 -g 1000 -m 0755 "$step_dir"
  sudo -u '#1000' test -w "$step_dir" || {
    echo "broadcast step path is not writable by UID 1000: $step_dir" >&2
    exit 1
  }
done
sudo -u '#1000' test -w "$broadcast_dir" && sudo -u '#1000' test -w "$broadcast_dir/step_0" || {
  echo "broadcast paths are not writable by UID 1000: $broadcast_dir" >&2
  exit 1
}
for p in "${ports[@]}"; do
  ss -tln 2>/dev/null | grep -q "127.0.0.1:$p" && echo "tunnel :$p up" || echo "tunnel :$p DOWN"
done

sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
sudo docker run -d --name "$ORCH_CT" \
  --network host --restart no \
  --env-file /data/trinity/training.env \
  -e VLLM_API_KEY=unused \
  -e PYTHONPATH="$deployment/environments/decensor/src:$deployment/environments/general-agent:/app/src" \
  -e HF_HOME="/data/glm52-rl/runs/$RUN/$RUN_NAME/hf-cache" \
  -v /data/glm52-rl:/data/glm52-rl \
  -v glm52-fp8-model:/models/glm52-fp8:ro \
  -v "$deployment/prime-rl/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
  -v "$deployment/prime-rl/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
  -v "$deployment/prime-rl/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
  -v "$deployment/prime-rl/deps/verifiers/verifiers/v1/harnesses/null/harness.py":/app/deps/verifiers/verifiers/v1/harnesses/null/harness.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/algo/grpo.py":/app/src/prime_rl/orchestrator/algo/grpo.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/dispatcher.py":/app/src/prime_rl/orchestrator/dispatcher.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/orchestrator.py":/app/src/prime_rl/orchestrator/orchestrator.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/train_sink.py":/app/src/prime_rl/orchestrator/train_sink.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
  -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/shared.py":/app/packages/prime-rl-configs/src/prime_rl/configs/shared.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro \
  --entrypoint /app/.venv/bin/orchestrator "$image" \
  @ "$config_file"
echo "orchestrator started on $(hostname) with $(basename "$config_file")"

# The China-side supervisor refuses to restart a failed trainer because the
# optimizer has no resumable checkpoint. Stop rollout production here as soon
# as any trainer worker is gone, so the run cannot accumulate unusable batches.
pkill -f '[t]rainer_failure_watchdog.sh' 2>/dev/null || true
nohup setsid env RUN="$RUN" RUN_NAME="$RUN_NAME" \
  TRAINER_ADDRESSES="$TRAINER_ADDRESSES" \
  TRAINER_CT="$TRAINER_CT" ORCH_CT="$ORCH_CT" \
  bash "$deployment/training_ops/runtime/h200/trainer_failure_watchdog.sh" \
  >>"/data/glm52-rl/trainer-failure-watchdog-$RUN.log" 2>&1 </dev/null &
trainer_watchdog_pid=$!
sleep 1
if ! kill -0 "$trainer_watchdog_pid" 2>/dev/null; then
  echo "ERROR: trainer failure watchdog exited during startup" >&2
  sudo docker rm -f "$ORCH_CT" >/dev/null 2>&1 || true
  exit 1
fi
echo "trainer failure watchdog started (pid $trainer_watchdog_pid)"

ctl=/data/glm52-rl/runs/$RUN/$RUN_NAME/control/orch.toml
if [ "$transport" = filesystem ]; then
  publisher_dir="/data/glm52-rl/runs/$RUN/$RUN_NAME"
  publisher_pid_file="$publisher_dir/.batch-sync.pid"
  publisher_lock="$publisher_dir/.batch-sync.lock"
  if [ -r "$publisher_pid_file" ]; then
    old_publisher_pid=$(cat "$publisher_pid_file")
    case "$old_publisher_pid" in
      ''|*[!0-9]*) ;;
      *)
        # The publisher runs under setsid. Stop the complete group so an rsync
        # child cannot inherit fd 9 and keep the flock after its parent exits.
        kill -- "-$old_publisher_pid" 2>/dev/null || kill "$old_publisher_pid" 2>/dev/null || true
        ;;
    esac
  fi
  pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null || true
  # A terminated publisher can remove its pid file before a timeout/rsync/ssh
  # child exits. That child inherits fd 9 and keeps the lock, so a normal
  # process-name kill cannot find it. Stop only process groups that still hold
  # this exact run's lock before starting the replacement publisher.
  if ! flock -n "$publisher_lock" -c true && command -v lsof >/dev/null 2>&1; then
    lock_holders=$(lsof -t -- "$publisher_lock" 2>/dev/null | sort -u || true)
    for holder in $lock_holders; do
      holder_pgid=$(ps -o pgid= -p "$holder" 2>/dev/null | tr -d ' ')
      case "$holder_pgid" in
        ''|*[!0-9]*) ;;
        *) kill -- "-$holder_pgid" 2>/dev/null || kill "$holder" 2>/dev/null || true ;;
      esac
    done
  fi
  publisher_lock_ready=0
  for _ in $(seq 1 30); do
    if flock -n "$publisher_lock" -c true; then
      publisher_lock_ready=1
      break
    fi
    sleep 1
  done
  if [ "$publisher_lock_ready" != 1 ]; then
    echo "ERROR: old durable batch publisher still owns $publisher_lock" >&2
    exit 1
  fi
  nohup setsid env SOURCE_RUN_DIR="/data/glm52-rl/runs/$RUN/$RUN_NAME" \
    DEST_HOST="$trainer_ssh_target" \
    "$deployment/training_ops/runtime/h200/sync_rollout_batches.sh" \
    >>"/data/glm52-rl/batch-sync-$RUN.log" 2>&1 </dev/null &
  publisher_pid=$!
  sleep 1
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    echo "ERROR: durable batch publisher exited during startup" >&2
    exit 1
  fi
  echo "durable batch publisher started (pid $publisher_pid)"
  exit
fi

for _ in $(seq 1 60); do
  [ -f "$ctl" ] && break
  sleep 2
done
if [ -f "$ctl" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$trainer_ssh_target" \
      "mkdir -p /data/glm52-rl/runs/$RUN/$RUN_NAME/control" \
    && scp -q -o BatchMode=yes "$ctl" "$trainer_ssh_target:$ctl" \
    && echo "control/orch.toml -> trainer ok" \
    || echo "WARNING: could not ship orch.toml to the trainer; it will not start" >&2
else
  echo "WARNING: $ctl never appeared; trainer will not start" >&2
fi
