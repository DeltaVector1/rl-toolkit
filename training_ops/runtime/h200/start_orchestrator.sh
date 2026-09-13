#!/usr/bin/env bash
# Run the orchestrator on h200a instead of n82.
# Engine traffic stays local. Only adapter and completed batch data cross the
# China link.
set -uo pipefail

tunnels_only=0
restart_tunnels=0
for argument in "$@"; do
  case "$argument" in
    --tunnels-only) tunnels_only=1 ;;
    --restart-tunnels) restart_tunnels=1 ;;
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

if ! secondary_settings=$(python3 - "$nodes_file" "$config_file" <<'PY'
import ipaddress
import re
import sys
import tomllib
from urllib.parse import urlsplit


def port_value(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise SystemExit(f"judges.secondary.{name} is invalid")
    return value


with open(sys.argv[1], "rb") as stream:
    nodes = tomllib.load(stream)
with open(sys.argv[2], "rb") as stream:
    active_config = tomllib.load(stream)

judges = nodes.get("judges")
external = judges.get("external") if isinstance(judges, dict) else None
secondary = judges.get("secondary") if isinstance(judges, dict) else None
if not isinstance(external, dict) or not isinstance(secondary, dict):
    raise SystemExit("nodes config has no complete judge inventory")

host = secondary.get("host")
try:
    ipaddress.ip_address(host)
except (TypeError, ValueError) as error:
    raise SystemExit("judges.secondary.host must be an IP address") from error
ssh_port = port_value(secondary.get("ssh_port"), "ssh_port")
local_api_port = port_value(secondary.get("local_api_port"), "local_api_port")
remote_api_port = port_value(secondary.get("remote_api_port"), "remote_api_port")
ssh_user = secondary.get("ssh_user")
if not isinstance(ssh_user, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", ssh_user) is None:
    raise SystemExit("judges.secondary.ssh_user is invalid")
identity = secondary.get("ssh_identity_file")
if (
    not isinstance(identity, str)
    or not identity.startswith("/")
    or re.fullmatch(r"/[A-Za-z0-9_./-]+", identity) is None
    or any(part in {"", ".", ".."} for part in identity.split("/")[1:])
):
    raise SystemExit("judges.secondary.ssh_identity_file is invalid")
api_url = secondary.get("api_url")
expected_api_url = f"http://127.0.0.1:{local_api_port}/v1"
parsed_api_url = urlsplit(api_url) if isinstance(api_url, str) else None
if (
    api_url != expected_api_url
    or parsed_api_url is None
    or parsed_api_url.query
    or parsed_api_url.fragment
):
    raise SystemExit("judges.secondary.api_url must be the configured local API URL")
model = secondary.get("model")
external_model = external.get("model")
if not isinstance(model, str) or model != external_model:
    raise SystemExit("judges.secondary.model must match judges.external.model")
data_parallel_size = secondary.get("data_parallel_size")
tensor_parallel_size = secondary.get("tensor_parallel_size")
if (
    not isinstance(data_parallel_size, int)
    or isinstance(data_parallel_size, bool)
    or data_parallel_size != 2
):
    raise SystemExit("judges.secondary.data_parallel_size must be 2")
if (
    not isinstance(tensor_parallel_size, int)
    or isinstance(tensor_parallel_size, bool)
    or tensor_parallel_size != 4
    or tensor_parallel_size * data_parallel_size != 8
):
    raise SystemExit("judges.secondary.tensor_parallel_size must be 4 for eight GPUs")


def values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from values(item)
    elif isinstance(value, list):
        for item in value:
            yield from values(item)
    else:
        yield value


active = int(any(value == api_url for value in values(active_config)))
print(host, ssh_port, ssh_user, identity, local_api_port, remote_api_port, api_url, active)
PY
); then
  echo "cannot read the configured secondary judge" >&2
  exit 2
fi
read -r secondary_host secondary_ssh_port secondary_ssh_user secondary_ssh_identity \
  secondary_local_api_port secondary_remote_api_port secondary_api_url secondary_enabled <<<"$secondary_settings"
secondary_ssh_target="$secondary_ssh_user@$secondary_host"
secondary_ssh_options=(
  -o IdentitiesOnly=yes
  -i "$secondary_ssh_identity"
  -p "$secondary_ssh_port"
)

if [ "$tunnels_only" = 0 ] && grep -q 'refusals-classifier-eval-v1' "$config_file"; then
  bash "$deployment/training_ops/runtime/h200/start_refusals_classifier.sh"
fi

generic_tunnel_ssh_match() {
  local command_line=$1 via=$2 forward=$3 comp=$4 index
  shift 4
  local -a ssh_options=("$@")
  local -a command_args expected_args
  expected_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=$comp
    "${ssh_options[@]}"
    -L "$forward"
  )
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  [ "${#command_args[@]}" -eq "$(( ${#expected_args[@]} + 2 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[-1]}" = "$via" ]
}

generic_tunnel_supervisor_match() {
  local command_line=$1 marker=$2 via=$3 forward=$4 comp=$5
  shift 5
  local -a ssh_options=("$@")
  local ssh_command
  case "$command_line" in *"while true; do"*"done # $marker") ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  generic_tunnel_ssh_match "$ssh_command" "$via" "$forward" "$comp" \
    "${ssh_options[@]}"
}

generic_tunnel_listener_ready() {
  local lp=$1 marker=$2 via=$3 forward=$4 comp=$5
  shift 5
  local -a ssh_options=("$@")
  local listener pid ppid owner_command supervisor_command
  listener=$(ss -H -ltnp 2>/dev/null | awk -v port=":$lp" '$4 ~ port "$" {print; exit}')
  [ -n "$listener" ] || return 1
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  owner_command=$(ps -o args= -p "$pid" 2>/dev/null || true)
  generic_tunnel_ssh_match "$owner_command" "$via" "$forward" "$comp" "${ssh_options[@]}" || return 1
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$ppid" != 1 ] || return 1
  supervisor_command=$(ps -o args= -p "$ppid" 2>/dev/null || true)
  generic_tunnel_supervisor_match "$supervisor_command" "$marker" "$via" "$forward" "$comp" \
    "${ssh_options[@]}" || return 1
  timeout --signal=TERM --kill-after=1s 5 bash -c ":</dev/tcp/127.0.0.1/$lp" 2>/dev/null
}

tunnel() { # local_port  target_host:port  ssh_target  [compress]
  local lp=$1 tgt=$2 via=$3 comp=${4:-no}
  shift $(( $# >= 4 ? 4 : $# ))
  local -a ssh_options=("$@")
  local ssh_text=""
  [ "${#ssh_options[@]}" -eq 0 ] || ssh_text=$(printf ' %q' "${ssh_options[@]}")
  local marker="zmq-tunnel-$lp" forward="127.0.0.1:$lp:$tgt"
  local pid command_line
  for pid in $(pgrep -f -- "$marker" || true); do
    command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
    if generic_tunnel_supervisor_match "$command_line" "$marker" "$via" "$forward" "$comp" \
      "${ssh_options[@]}" \
      && generic_tunnel_listener_ready "$lp" "$marker" "$via" "$forward" "$comp" \
        "${ssh_options[@]}"; then
      return 0
    fi
    generic_tunnel_stop_stale "$pid" "$command_line" "$marker" "$forward" || return 1
  done
  setsid nohup bash -c "while true; do
      ssh -NT -o BatchMode=yes -o ConnectTimeout=30 -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
          -o IPQoS=none -o Compression=$comp$ssh_text \
          -L 127.0.0.1:$lp:$tgt $via
      sleep 3
    done # $marker" >/dev/null 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    if generic_tunnel_listener_ready "$lp" "$marker" "$via" "$forward" "$comp" \
      "${ssh_options[@]}"; then
      return 0
    fi
    sleep 1
  done
  echo "generic tunnel :$lp is not owned by the configured supervisor or is not usable" >&2
  return 1
}

generic_tunnel_stale_supervisor_match() {
  local command_line=$1 marker=$2 forward=$3 ssh_command
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  [ "$(grep -o -- '-L ' <<<"$ssh_command" | wc -l)" -eq 1 ] || return 1
  tunnel_command_has "$ssh_command" "$forward"
}

generic_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 marker=$3 forward=$4
  local group_command expected_ssh group_listing group_seen=0
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        generic_tunnel_stale_supervisor_match "$group_command" "$marker" "$forward" || return 1
        [ "$(tunnel_supervisor_ssh_command "$group_command" "$marker")" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        [ "$group_command" = "$expected_ssh" ] || return 1
        ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ]
}

generic_tunnel_stop_stale() {
  local pid=$1 command_line=$2 marker=$3 forward=$4 pgid
  generic_tunnel_stale_supervisor_match "$command_line" "$marker" "$forward" || return 0
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$")
      echo "could not prove stale generic tunnel group for $marker" >&2
      return 1
      ;;
    *)
      if ! generic_tunnel_stale_group_safe "$pgid" "$command_line" "$marker" "$forward"; then
        echo "stale generic tunnel group $pgid failed ownership proof for $marker" >&2
        return 1
      fi
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop stale generic tunnel group $pgid for $marker" >&2
        return 1
      }
      ;;
  esac
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
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -i "$peer_ssh_identity"
    -p "$peer_ssh_port"
  )
  local forward_text ssh_text peer_target marker pid command_line
  local expected_shape=current
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  marker="peer-tunnel-$connection_name"
  [ "$connection_name" = api ] && expected_shape=api
  if [ "${restart_tunnels:-0}" = 0 ]; then
    for pid in $(pgrep -f -- "$signature" || true); do
      command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
      if peer_tunnel_supervisor_forwards_match "$command_line" "$expected_shape" "$marker" "${forwards[@]}"; then
        return 0
      fi
      peer_kill_owned "$pid" "$command_line" "$expected_shape" "$marker" "${forwards[@]}"
    done
  fi
  setsid nohup bash -c "while true; do
      ssh$ssh_text$forward_text $peer_target
      sleep 3
    done # $marker" >/dev/null 2>&1 < /dev/null &
}

peer_tunnel_forward_shape() {
  local command_line=$1 shape=$2
  local forward_total spec source_port destination_port
  local -a specs seen=()
  forward_total=$(grep -o -- '-[LR] ' <<<"$command_line" | wc -l)
  case "$shape:$forward_total" in
    cleanup:1|cleanup:2|cleanup:4|current:2|api:1) ;;
    *) return 1 ;;
  esac
  mapfile -t specs < <(grep -oE -- '-[LR] 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#specs[@]}" -eq "$forward_total" ] || return 1
  for spec in "${specs[@]}"; do
    case "$spec" in
      -R\ 127.0.0.1:*:127.0.0.1:*)
        source_port=${spec#-R 127.0.0.1:}
        source_port=${source_port%%:*}
        destination_port=${spec##*:}
        case "$destination_port" in
          ''|*[!0-9]*) return 1 ;;
        esac
        [ "$destination_port" -ge 1 ] && [ "$destination_port" -le 65535 ] || return 1
        if [ "$shape" != cleanup ] && [ "$destination_port" != "$relay_port" ]; then
          return 1
        fi
        case " ${lora_source_ports[*]} " in
          *" $source_port "*) ;;
          *) return 1 ;;
        esac
        case " ${seen[*]} " in
          *" $source_port "*) return 1 ;;
          *) seen+=("$source_port") ;;
        esac
        ;;
      -L\ 127.0.0.1:"$peer_api_port":127.0.0.1:8000)
        [ "$forward_total" -eq 1 ] || return 1
        ;;
      *) return 1 ;;
    esac
  done
}

peer_tunnel_marker_match() {
  local command_line=$1 marker=${2:-}
  if [ -n "$marker" ]; then
    case "$command_line" in *"# $marker") ;; *) return 1 ;; esac
  else
    [[ "$command_line" =~ \#\ peer-tunnel-(api|lora-[0-9]+)[[:space:]]*$ ]] || return 1
  fi
}

peer_tunnel_supervisor_match() {
  local command_line=$1 shape=$2 marker=${3:-}
  local ssh_command
  peer_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  if [ -z "$marker" ]; then
    marker=$(sed -n 's/.*# \(peer-tunnel-\(api\|lora-[0-9][0-9]*\)\)$/\1/p' <<<"$command_line")
  fi
  [ -n "$marker" ] || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  peer_tunnel_ssh_args_match "$ssh_command" "$shape"
}

peer_tunnel_supervisor_forwards_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  peer_tunnel_supervisor_match "$command_line" "$shape" "$marker" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

peer_tunnel_stale_supervisor_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  peer_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  peer_tunnel_forward_shape "$command_line" "$shape" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

tunnel_supervisor_ssh_command() {
  local command_line=$1 marker=$2 normalized prefix body suffix ssh_command
  normalized=$(tr '\n\t' '  ' <<<"$command_line" | sed -E 's/[[:space:]]+/ /g; s/^ +//; s/ +$//')
  case "$normalized" in *"while true; do "*) ;; *) return 1 ;; esac
  prefix=${normalized%%while true; do *}
  case "$prefix" in
    *';'*|*'|'*|*'&'*|*'`'*|*'$('*|*'<'*|*'>'*|*'('*|*')'*|*'{'*|*'}'*) return 1 ;;
    ""|"bash -c "|"nohup bash -c ") ;;
    *) return 1 ;;
  esac
  body=${normalized#*while true; do }
  case "$body" in
    "ssh "*"; sleep 3; done # $marker") suffix="; sleep 3; done # $marker" ;;
    "ssh "*" sleep 3; done # $marker") suffix=" sleep 3; done # $marker" ;;
    "ssh "*" sleep 3 done # $marker") suffix=" sleep 3 done # $marker" ;;
    *) return 1 ;;
  esac
  ssh_command=${body%"$suffix"}
  case "$ssh_command" in
    *';'*|*'|'*|*'&'*|*'`'*|*'$('*|*'<'*|*'>'*|*'('*|*')'*|*'{'*|*'}'*|*'#'*) return 1 ;;
  esac
  printf '%s\n' "$ssh_command"
}

tunnel_channel_timeout_normalize() {
  sed 's/ChannelTimeout=\\\*-tcpip=45s/ChannelTimeout=*-tcpip=45s/g' <<<"$1"
}

china_tunnel_proxy_ssh_command() {
  local route_target=${trainer_ssh_target:-} hostname= port= proxyjump=
  local key value
  if [ "${china_tunnel_proxy_target:-}" = "$route_target" ]; then
    [ -n "${china_tunnel_proxy_command:-}" ] || return 1
    printf '%s\n' "$china_tunnel_proxy_command"
    return 0
  fi
  china_tunnel_proxy_target=$route_target
  china_tunnel_proxy_command=
  [ -n "$route_target" ] || return 1
  while read -r key value _; do
    case "$key" in
      hostname) hostname=$value ;;
      port) port=$value ;;
      proxyjump) proxyjump=$value ;;
    esac
  done < <(ssh -G "$route_target" 2>/dev/null || true)
  case "$hostname" in ''|*[!A-Za-z0-9_.:-]*) return 1 ;; esac
  case "$port" in ''|*[!0-9]*) return 1 ;; esac
  [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || return 1
  case "$proxyjump" in
    ''|none|*,*|*@*|*:*|*[!A-Za-z0-9_.-]*) return 1 ;;
  esac
  china_tunnel_proxy_command="ssh -W [$hostname]:$port $proxyjump"
  printf '%s\n' "$china_tunnel_proxy_command"
}

china_tunnel_proxy_child_match() {
  local command_line=$1 expected normalized=$1
  expected=$(china_tunnel_proxy_ssh_command) || return 1
  case "$normalized" in
    /usr/bin/ssh\ *) normalized="ssh ${normalized#/usr/bin/ssh }" ;;
  esac
  [ "$normalized" = "$expected" ]
}

peer_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 shape=$3 marker=$4
  local group_command expected_ssh actual_ssh group_seen=0 normalized_command
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
  while IFS= read -r group_command; do
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        peer_tunnel_stale_supervisor_match "$group_command" "$shape" "$marker" || return 1
        actual_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$marker") || return 1
        actual_ssh=$(tunnel_channel_timeout_normalize "$actual_ssh") || return 1
        [ "$actual_ssh" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        [ "$normalized_command" = "$expected_ssh" ] || return 1
        ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done < <(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}')
  [ "$group_seen" = 1 ]
}

peer_tunnel_ssh_args_match() {
  local command_line=$1 shape=$2
  local index forward_count forward_start target_index remaining
  local -a command_args expected_args
  command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
  expected_args=(
    -NT
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -i "$peer_ssh_identity"
    -p "$peer_ssh_port"
  )
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  target_index=$((${#command_args[@]} - 1))
  [ "$target_index" -gt "$(( ${#expected_args[@]} + 1 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[target_index]}" = "$peer_ssh_user@$peer_host" ] || return 1
  forward_start=$((1 + ${#expected_args[@]}))
  remaining=$((target_index - forward_start))
  [ "$remaining" -gt 0 ] && [ "$((remaining % 2))" -eq 0 ] || return 1
  forward_count=$((remaining / 2))
  for ((index = forward_start; index < target_index; index += 2)); do
    case "${command_args[index]}" in -L|-R) ;; *) return 1 ;; esac
    case "${command_args[index + 1]}" in
      127.0.0.1:[0-9]*:127.0.0.1:[0-9]*) ;;
      *) return 1 ;;
    esac
  done
  peer_tunnel_forward_shape "$command_line" "$shape" || return 1
  [ "$forward_count" -eq "$(grep -o -- '-[LR] ' <<<"$command_line" | wc -l)" ]
}

peer_tunnel_group_safe() {
  local pgid=$1 group_command group_seen=0
  while IFS= read -r group_command; do
    group_seen=1
    case "$group_command" in
      *"while true; do"*) peer_tunnel_supervisor_match "$group_command" cleanup || return 1 ;;
      ssh\ *|/usr/bin/ssh\ *) peer_tunnel_ssh_args_match "$group_command" cleanup || return 1 ;;
      "sleep 3") ;;
      *) return 1 ;;
    esac
  done < <(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}')
  [ "$group_seen" = 1 ]
}

peer_tunnel_remove_legacy() {
  local keep_current=${1:-0}
  local pid command_line marker
  while read -r pid command_line; do
    marker=$(sed -n 's/.*# \(peer-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
    [ -n "$marker" ] || continue
    peer_tunnel_stale_supervisor_match "$command_line" cleanup "$marker" || continue
    if [ "$keep_current" = 1 ]; then
      if peer_tunnel_supervisor_match "$command_line" api \
          || peer_tunnel_supervisor_match "$command_line" current; then
        continue
      fi
    fi
    peer_kill_owned "$pid" "$command_line" cleanup "$marker"
  done < <(ps -eo pid=,args=)
}

peer_kill_owned() {
  local pid=$1 command_line=$2 shape=${3:-} marker=${4:-} pgid
  if [ -n "$shape" ] && [ -n "$marker" ]; then
    peer_tunnel_stale_supervisor_match "$command_line" "$shape" "$marker" || return 0
  else
    peer_tunnel_supervisor_match "$command_line" cleanup || return 0
  fi
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$") kill "$pid" 2>/dev/null || true ;;
    *)
      if [ -n "$shape" ] && [ -n "$marker" ]; then
        if ! peer_tunnel_stale_group_safe "$pgid" "$command_line" "$shape" "$marker"; then
          echo "peer tunnel group $pgid failed ownership proof for $marker" >&2
          return 1
        fi
        kill -- "-$pgid" 2>/dev/null || {
          echo "could not stop peer tunnel group $pgid for $marker" >&2
          return 1
        }
        return 0
      fi
      if ! peer_tunnel_group_safe "$pgid"; then
        echo "peer tunnel group $pgid failed ownership proof" >&2
        return 1
      fi
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop peer tunnel group $pgid" >&2
        return 1
      }
      ;;
  esac
}

peer_tunnel_remove_orphan_children() {
  local pid ppid command_line executable found=0
  while read -r pid ppid command_line; do
    [ "$ppid" = 1 ] || continue
    executable=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    [ "$executable" = /usr/bin/ssh ] || continue
    peer_tunnel_ssh_args_match "$command_line" cleanup || continue
    kill "$pid" 2>/dev/null || true
    found=1
  done < <(ps -eo pid=,ppid=,args=)
  [ "$found" = 0 ]
}

peer_remote_ports_ready() {
  local probe_seconds=${1:-10} peer_target verified_ports
  peer_target=$(printf '%q' "$peer_ssh_user@$peer_host")
  if ! verified_ports=$(timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout="$probe_seconds" \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
      -o IPQoS=none -i "$peer_ssh_identity" -p "$peer_ssh_port" \
      "$peer_target" "python3 - ${lora_source_ports[0]} ${lora_source_ports[-1]}" 2>/dev/null <<'PY'
import http.client
import sys
from concurrent.futures import ThreadPoolExecutor

first = int(sys.argv[1])
last = int(sys.argv[2])


def is_relay(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/adapter-relay-status")
        response = connection.getresponse()
        body = response.read()
        if response.status != 404 or body != b"source URL is not configured\n":
            return port
    except OSError:
        return port
    finally:
        connection.close()
    return None


ports = list(range(first, last + 1))
with ThreadPoolExecutor(max_workers=min(32, len(ports))) as executor:
    failed = [port for port in executor.map(is_relay, ports) if port is not None]
if failed:
    print("wrong adapter relay ports: " + ",".join(map(str, failed)), file=sys.stderr)
    raise SystemExit(1)
print(len(ports))
PY
  ); then
    return 1
  fi
  [ "$verified_ports" = "${#lora_source_ports[@]}" ]
}

china_source_ports_ready() {
  local probe_seconds=${1:-10} expected_lanes observed_lanes verified_ports
  expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  observed_lanes=$(ps -eo args= | sed -n 's/.*# china-tunnel-lora-\([0-9][0-9]*\)$/\1/p' | sort -n)
  [ "$observed_lanes" = "$(seq 0 $((expected_lanes - 1)))" ] || return 1
  if ! verified_ports=$(timeout --signal=TERM --kill-after=1s "$probe_seconds" \
      python3 - "${lora_source_ports[0]}" "${lora_source_ports[-1]}" <<'PY'
import http.client
import sys
from concurrent.futures import ThreadPoolExecutor

first = int(sys.argv[1])
last = int(sys.argv[2])


def is_adapter_server(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("HEAD", "/")
        response = connection.getresponse()
        response.read()
        server = response.getheader("Server") or ""
        if response.status != 200 or not server.startswith("SimpleHTTP/"):
            return port
    except OSError:
        return port
    finally:
        connection.close()
    return None


ports = list(range(first, last + 1))
with ThreadPoolExecutor(max_workers=min(32, len(ports))) as executor:
    failed = [port for port in executor.map(is_adapter_server, ports) if port is not None]
if failed:
    print("wrong adapter source ports: " + ",".join(map(str, failed)), file=sys.stderr)
    raise SystemExit(1)
print(len(ports))
PY
  ); then
    return 1
  fi
  [ "$verified_ports" = "${#lora_source_ports[@]}" ]
}

tunnel_command_has() {
  local command_line=$1 token=$2
  case " $command_line " in
    *" $token "*|*" $token;"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Validate source forwards from the configured inventory.  The caller chooses
# the accepted shape so cleanup can handle the one- and four-forward layouts,
# while readiness accepts only the current two-forward layout.
china_tunnel_forward_shape() {
  local command_line=$1 shape=$2
  local source_port forward_total spec
  local -a specs seen=()
  forward_total=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
  case "$shape:$forward_total" in
    cleanup:1|cleanup:2|cleanup:4|current:2) ;;
    *) return 1 ;;
  esac
  mapfile -t specs < <(grep -oE -- '-L 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#specs[@]}" -eq "$forward_total" ] || return 1
  for spec in "${specs[@]}"; do
    case "$spec" in
      -L\ 127.0.0.1:*:127.0.0.1:"$adapter_port") ;;
      *) return 1 ;;
    esac
    source_port=${spec#-L 127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${lora_source_ports[*]} " in
      *" $source_port "*) ;;
      *) return 1 ;;
    esac
    case " ${seen[*]} " in
      *" $source_port "*) return 1 ;;
      *) seen+=("$source_port") ;;
    esac
  done
}

china_tunnel_marker_match() {
  local command_line=$1 marker=${2:-}
  if [ -n "$marker" ]; then
    case "$command_line" in *"# $marker") ;; *) return 1 ;; esac
  else
    [[ "$command_line" =~ \#\ china-tunnel-lora-[0-9]+[[:space:]]*$ ]] || return 1
  fi
}

china_tunnel_supervisor_match() {
  local command_line=$1 shape=$2 marker=${3:-}
  local ssh_command
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  if [ -z "$marker" ]; then
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  fi
  [ -n "$marker" ] || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_ssh_args_match "$ssh_command" "$shape"
}

china_tunnel_supervisor_forwards_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  china_tunnel_supervisor_match "$command_line" "$shape" "$marker" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_known_old_supervisor_forwards_match() {
  local command_line=$1 marker=$2 forward ssh_command
  shift 2
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_known_old_ssh_args_match "$ssh_command" current || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_known_old_supervisor_match() {
  local command_line=$1 marker ssh_command
  marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  [ -n "$marker" ] || return 1
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  china_tunnel_forward_shape "$command_line" current || return 1
  ssh_command=$(tunnel_supervisor_ssh_command "$command_line" "$marker") || return 1
  china_tunnel_known_old_ssh_args_match "$ssh_command" current
}

china_tunnel_stale_supervisor_match() {
  local command_line=$1 shape=$2 marker=$3 forward
  shift 3
  china_tunnel_marker_match "$command_line" "$marker" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  china_tunnel_forward_shape "$command_line" "$shape" || return 1
  for forward in "$@"; do
    tunnel_command_has "$command_line" "$forward" || return 1
  done
}

china_tunnel_ssh_args_match_mode() {
  local command_line=$1 shape=$2 mode=${3:-current}
  local source_port forward_count index
  local -a command_args expected_args
  expected_args=(
    -NT
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
  )
  case "$mode" in
    current) expected_args+=( -o 'ChannelTimeout=*-tcpip=45s' ) ;;
    old) ;;
    *) return 1 ;;
  esac
  expected_args+=(
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=no
  )
  command_line=$(tunnel_channel_timeout_normalize "$command_line") || return 1
  read -r -a command_args <<<"$command_line"
  case "${command_args[0]:-}" in ssh|/usr/bin/ssh) ;; *) return 1 ;; esac
  [ "${#command_args[@]}" -ge "$(( ${#expected_args[@]} + 4 ))" ] || return 1
  for index in "${!expected_args[@]}"; do
    [ "${command_args[index + 1]}" = "${expected_args[index]}" ] || return 1
  done
  [ "${command_args[-1]}" = "$trainer_ssh_target" ] || return 1
  forward_count=$(( (${#command_args[@]} - ${#expected_args[@]} - 2) / 2 ))
  china_tunnel_forward_shape "$command_line" "$shape" || return 1
  [ "$forward_count" -eq "$(grep -o -- '-L ' <<<"$command_line" | wc -l)" ] || return 1
  for ((index = ${#expected_args[@]} + 1; index < ${#command_args[@]} - 1; index += 2)); do
    [ "${command_args[index]}" = "-L" ] || return 1
    source_port=${command_args[index + 1]#127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${lora_source_ports[*]} " in
      *" $source_port "*) ;;
      *) return 1 ;;
    esac
  done
}

china_tunnel_ssh_args_match() {
  china_tunnel_ssh_args_match_mode "$1" "$2" current
}

china_tunnel_known_old_ssh_args_match() {
  china_tunnel_ssh_args_match_mode "$1" "$2" old
}

china_tunnel_group_safe() {
  local pgid=$1 expected_marker=${2:-} group_command
  local group_seen=0 supervisor_seen=0 direct_seen=0 proxy_seen=0 sleep_seen=0
  local expected_ssh proxy_ssh normalized_command
  local group_listing
  group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  expected_ssh=
  proxy_ssh=$(china_tunnel_proxy_ssh_command || true)
  while IFS= read -r group_command; do
    case "$group_command" in
      *"while true; do"*)
        [ "$supervisor_seen" = 0 ] || return 1
        supervisor_seen=1
        china_tunnel_supervisor_match "$group_command" cleanup "$expected_marker" || return 1
        expected_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$expected_marker") || return 1
        expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
        ;;
    esac
  done <<<"$group_listing"
  [ "$supervisor_seen" = 1 ] || return 1
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        if [ "$normalized_command" = "$expected_ssh" ]; then
          [ "$direct_seen" = 0 ] || return 1
          direct_seen=1
        else
          [ -n "$proxy_ssh" ] && [ "$normalized_command" = "$proxy_ssh" ] || return 1
          [ "$proxy_seen" = 0 ] || return 1
          proxy_seen=1
        fi
        ;;
      "sleep 3")
        [ "$sleep_seen" = 0 ] || return 1
        sleep_seen=1
        ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ] && [ "$supervisor_seen" = 1 ]
}

china_tunnel_stale_group_safe() {
  local pgid=$1 supervisor_command=$2 marker=$3
  local group_command expected_ssh actual_ssh group_listing group_seen=0
  local supervisor_seen=0 direct_seen=0 proxy_seen=0 sleep_seen=0 normalized_command
  expected_ssh=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  expected_ssh=$(tunnel_channel_timeout_normalize "$expected_ssh") || return 1
  if ! china_tunnel_ssh_args_match "$expected_ssh" cleanup; then
    china_tunnel_known_old_ssh_args_match "$expected_ssh" cleanup || return 1
  fi
  group_listing=$(ps -eo pgid=,args= | awk -v group="$pgid" '$1 == group {sub(/^[^ ]+ +/, ""); print}' | sed -E 's/[[:space:]]+$//') || return 1
  while IFS= read -r group_command; do
    [ -n "$group_command" ] || continue
    group_seen=1
    case "$group_command" in
      *"while true; do"*)
        [ "$supervisor_seen" = 0 ] || return 1
        supervisor_seen=1
        china_tunnel_stale_supervisor_match "$group_command" cleanup "$marker" || return 1
        actual_ssh=$(tunnel_supervisor_ssh_command "$group_command" "$marker") || return 1
        actual_ssh=$(tunnel_channel_timeout_normalize "$actual_ssh") || return 1
        [ "$actual_ssh" = "$expected_ssh" ] || return 1
        ;;
      ssh\ *|/usr/bin/ssh\ *)
        normalized_command=$group_command
        case "$normalized_command" in
          /usr/bin/ssh\ *) normalized_command="ssh ${normalized_command#/usr/bin/ssh }" ;;
        esac
        normalized_command=$(tunnel_channel_timeout_normalize "$normalized_command") || return 1
        if [ "$normalized_command" != "$expected_ssh" ]; then
          china_tunnel_proxy_child_match "$normalized_command" || return 1
          [ "$proxy_seen" = 0 ] || return 1
          proxy_seen=1
        else
          [ "$direct_seen" = 0 ] || return 1
          direct_seen=1
        fi
        ;;
      "sleep 3")
        [ "$sleep_seen" = 0 ] || return 1
        sleep_seen=1
        ;;
      *) return 1 ;;
    esac
  done <<<"$group_listing"
  [ "$group_seen" = 1 ] && [ "$supervisor_seen" = 1 ]
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
  case "$marker" in
    china-tunnel-*) china_tunnel_ssh_args_match "$owner_command" current || return 1 ;;
    peer-tunnel-api) peer_tunnel_ssh_args_match "$owner_command" api || return 1 ;;
    peer-tunnel-*) peer_tunnel_ssh_args_match "$owner_command" current || return 1 ;;
  esac
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
  case "$marker" in
    china-tunnel-*) china_tunnel_supervisor_forwards_match "$supervisor_command" current "$marker" "${forwards[@]}" || return 1 ;;
    peer-tunnel-api) peer_tunnel_supervisor_forwards_match "$supervisor_command" api "$marker" "${forwards[@]}" || return 1 ;;
    peer-tunnel-*) peer_tunnel_supervisor_forwards_match "$supervisor_command" current "$marker" "${forwards[@]}" || return 1 ;;
  esac
}

china_tunnel_supervisor_owns_forwards() {
  local supervisor_pid=$1 supervisor_command=$2 mode=${3:-current}
  local listener forward source_port child_pid child_ppid owner_command index ssh_command marker owner_pid=
  local -a forward_specs=()
  shift 3
  while [ "$#" -gt 0 ]; do
    if [ "$1" = -L ]; then
      shift
      [ "$#" -gt 0 ] || return 1
      forward_specs+=("$1")
    else
      case "$1" in -L\ *) forward_specs+=("${1#-L }") ;; *) return 1 ;; esac
    fi
    shift
  done
  [ "${#forward_specs[@]}" -gt 0 ] || return 1
  marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$supervisor_command")
  ssh_command=$(tunnel_supervisor_ssh_command "$supervisor_command" "$marker") || return 1
  if [ "$mode" = current ]; then
    china_tunnel_ssh_args_match "$ssh_command" current || return 1
  else
    china_tunnel_known_old_ssh_args_match "$ssh_command" current || return 1
  fi
  for forward in "${forward_specs[@]}"; do
    tunnel_command_has "$supervisor_command" "$forward" || return 1
    source_port=${forward#127.0.0.1:}
    source_port=${source_port%%:*}
    listener=$(ss -H -ltnp 2>/dev/null | awk -v port=":$source_port" '$4 ~ port "$" {print; exit}')
    [ -n "$listener" ] || return 1
    child_pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
    case "$child_pid" in ''|*[!0-9]*) return 1 ;; esac
    if [ -z "$owner_pid" ]; then
      owner_pid=$child_pid
    else
      [ "$child_pid" = "$owner_pid" ] || return 1
    fi
    child_ppid=$(ps -o ppid= -p "$child_pid" 2>/dev/null | tr -d ' ')
    [ "$child_ppid" = "$supervisor_pid" ] || return 1
    owner_command=$(ps -o args= -p "$child_pid" 2>/dev/null || true)
    if [ "$mode" = current ]; then
      china_tunnel_ssh_args_match "$owner_command" current || return 1
    else
      china_tunnel_known_old_ssh_args_match "$owner_command" current || return 1
    fi
    tunnel_command_has "$owner_command" "$forward" || return 1
  done
}

china_tunnel_grouped_legacy_match() {
  local command_line=$1 forward_total
  china_tunnel_marker_match "$command_line" || return 1
  case "$command_line" in *"while true; do"*) ;; *) return 1 ;; esac
  tunnel_command_has "$command_line" "$trainer_ssh_target" || return 1
  china_tunnel_forward_shape "$command_line" cleanup || return 1
  case "$command_line" in
    *"while true; do"*"$trainer_ssh_target"*"# china-tunnel-lora-"*) ;;
    *) return 1 ;;
  esac
  forward_total=$(grep -o -- '-L ' <<<"$command_line" | wc -l)
  [ "$forward_total" -eq 1 ] || [ "$forward_total" -eq 4 ]
}

china_tunnel_ssh_child_match() {
  local pid=$1 shape=${2:-cleanup} command_line executable
  executable=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
  case "$executable" in
    ''|/usr/bin/ssh) ;;
    *) return 1 ;;
  esac
  command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | sed 's/[[:space:]]*$//') || return 1
  china_tunnel_ssh_args_match "$command_line" "$shape" ||
    china_tunnel_known_old_ssh_args_match "$command_line" "$shape" ||
    china_tunnel_proxy_child_match "$command_line"
}

china_tunnel_remove_orphan_children() {
  local requested source_port listener state receive_queue send_queue local_address remote_address details pid ppid seen_pid
  local -a requested_ports=() listener_pids=()
  local -A requested_port=()
  for requested in "$@"; do
    case "$requested" in
      127.0.0.1:*:127.0.0.1:*)
        requested=${requested#127.0.0.1:}
        requested_ports+=("${requested%%:*}")
        ;;
      [0-9]*) requested_ports+=("$requested") ;;
    esac
  done
  [ "${#requested_ports[@]}" -gt 0 ] || requested_ports=("${lora_source_ports[@]}")
  for source_port in "${requested_ports[@]}"; do
    requested_port[$source_port]=1
  done
  while IFS= read -r listener; do
    read -r state receive_queue send_queue local_address remote_address details <<<"$listener"
    [ "${local_address%:*}" = "127.0.0.1" ] || continue
    source_port=${local_address##*:}
    [ -n "${requested_port[$source_port]+set}" ] || continue
    pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener")
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    seen_pid=0
    for listener_pid in "${listener_pids[@]}"; do
      [ "$listener_pid" = "$pid" ] && seen_pid=1
    done
    [ "$seen_pid" = 0 ] || continue
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$ppid" = 1 ] || continue
    china_tunnel_ssh_child_match "$pid" || continue
    listener_pids+=("$pid")
    china_legacy_cleanup_required=1
    china_legacy_cleanup_pids+=("$pid")
    china_legacy_cleanup_orphan_pids+=("$pid")
    case " ${china_legacy_cleanup_ports[*]} " in
      *" $source_port "*) ;;
      *) china_legacy_cleanup_ports+=("$source_port") ;;
    esac
    kill "$pid" 2>/dev/null || true
  done < <(ss -H -ltnp 2>/dev/null)
}

china_tunnel_ports_clear() {
  local requested source_port
  local -a requested_ports=()
  for requested in "$@"; do
    case "$requested" in
      127.0.0.1:*:127.0.0.1:*)
        requested=${requested#127.0.0.1:}
        requested_ports+=("${requested%%:*}")
        ;;
    esac
  done
  [ "${#requested_ports[@]}" -gt 0 ] || return 1
  for source_port in "${requested_ports[@]}"; do
    if ss -H -ltn 2>/dev/null | awk -v local_port="127.0.0.1:$source_port" '$4 == local_port {found=1} END {exit !found}'; then
      return 1
    fi
  done
}

china_tunnel_stop_supervisor() {
  local pid=$1 command_line=$2 expected_marker=${3:-} stale=${4:-0} pgid forward source_port
  local -a owned_forwards=()
  mapfile -t owned_forwards < <(grep -oE -- '-L 127\.0\.0\.1:[0-9]+:127\.0\.0\.1:[0-9]+' <<<"$command_line")
  [ "${#owned_forwards[@]}" -gt 0 ] || return 1
  china_legacy_cleanup_required=1
  china_legacy_cleanup_pids+=("$pid")
  for forward in "${owned_forwards[@]}"; do
    source_port=${forward#-L 127.0.0.1:}
    source_port=${source_port%%:*}
    case " ${china_legacy_cleanup_ports[*]} " in
      *" $source_port "*) ;;
      *) china_legacy_cleanup_ports+=("$source_port") ;;
    esac
  done
  if [ -z "$expected_marker" ]; then
    expected_marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
  fi
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$")
      echo "could not prove China tunnel group for ${expected_marker:-unknown}" >&2
      return 1
      ;;
    *)
      case " ${china_legacy_cleanup_pgids[*]} " in
        *" $pgid "*) ;;
        *) china_legacy_cleanup_pgids+=("$pgid") ;;
      esac
      if [ "$stale" = 1 ]; then
        if ! china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker"; then
          echo "stale China tunnel group $pgid failed ownership proof for ${expected_marker:-unknown}" >&2
          return 1
        fi
      elif ! china_tunnel_group_safe "$pgid" "$expected_marker"; then
        echo "China tunnel group $pgid failed ownership proof for ${expected_marker:-unknown}" >&2
        return 1
      fi
      kill -- "-$pgid" 2>/dev/null || {
        echo "could not stop China tunnel group $pgid for ${expected_marker:-unknown}" >&2
        return 1
      }
      ;;
  esac
}

china_tunnel_preflight_group() {
  local pid=$1 command_line=$2 stale=${3:-0} expected_marker=${4:-} pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$pgid" in
    ''|*[!0-9]*|0|1|"$$") return 1 ;;
  esac
  case " ${china_tunnel_preflight_pgids[*]:-} " in
    *" $pgid "*) return 1 ;;
  esac
  if [ "$stale" = 1 ]; then
    china_tunnel_stale_group_safe "$pgid" "$command_line" "$expected_marker" || return 1
  else
    china_tunnel_group_safe "$pgid" "$expected_marker" || return 1
  fi
  china_tunnel_preflight_pgids+=("$pgid")
}

china_tunnel_remove_all_owned() {
  local pid command_line marker invalid=0 index
  local -a current_pids=() current_commands=() old_pids=() old_commands=()
  while read -r pid command_line; do
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"$command_line")
    [ -n "$marker" ] || continue
    if china_tunnel_supervisor_match "$command_line" cleanup "$marker"; then
      current_pids+=("$pid")
      current_commands+=("$command_line")
    elif china_tunnel_known_old_supervisor_match "$command_line"; then
      old_pids+=("$pid")
      old_commands+=("$command_line")
    else
      invalid=1
    fi
  done < <(ps -eo pid=,args=)
  [ "$invalid" -eq 0 ] || return 1
  china_tunnel_preflight_pgids=()
  for index in "${!current_pids[@]}"; do
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"${current_commands[index]}")
    china_tunnel_preflight_group "${current_pids[index]}" "${current_commands[index]}" 0 "$marker" || return 1
  done
  for index in "${!old_pids[@]}"; do
    marker=$(sed -n 's/.*# \(china-tunnel-lora-[0-9][0-9]*\)$/\1/p' <<<"${old_commands[index]}")
    china_tunnel_preflight_group "${old_pids[index]}" "${old_commands[index]}" 1 "$marker" || return 1
  done
  for index in "${!current_pids[@]}"; do
    china_tunnel_stop_supervisor "${current_pids[index]}" "${current_commands[index]}" || return 1
  done
  for index in "${!old_pids[@]}"; do
    china_tunnel_stop_supervisor "${old_pids[index]}" "${old_commands[index]}" "" 1 || return 1
  done
}

china_tunnel_remove_legacy() {
  local only_marker=${1:-}
  local lora_source_port pattern pid command_line cleanup_port
  local grouped_legacy stale
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
        china_tunnel_marker_match "$command_line" "$only_marker" || continue
        case "$command_line" in *"while true; do"*) ;; *) continue ;; esac
        tunnel_command_has "$command_line" "$trainer_ssh_target" || continue
        china_tunnel_forward_shape "$command_line" cleanup || continue
        tunnel_command_has "$command_line" "$pattern" || continue
        grouped_legacy=1
      else
        china_tunnel_grouped_legacy_match "$command_line" || continue
        grouped_legacy=1
      fi
      if [ "$grouped_legacy" = 1 ]; then
        stale=1
        if [ -n "$only_marker" ] && china_tunnel_supervisor_match "$command_line" cleanup "$only_marker"; then
          stale=0
        fi
        china_tunnel_stop_supervisor "$pid" "$command_line" "$only_marker" "$stale" || return 1
      fi
    done
  done
}

china_tunnel_wait_legacy_clear() {
  local pid pgid group_processes listener_remaining
  for wait_cycle in $(seq 1 30); do
    china_tunnel_remove_orphan_children "${china_legacy_cleanup_ports[@]}"
    if [ "$wait_cycle" -eq 15 ]; then
      for pid in "${china_legacy_cleanup_orphan_pids[@]}"; do
        china_tunnel_ssh_child_match "$pid" && kill -KILL "$pid" 2>/dev/null || true
      done
    fi
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
    [ "$listener_remaining" = 0 ] && return 0
    sleep 1
  done
  echo "stale China adapter tunnels did not stop before migration" >&2
  return 1
}

china_tunnel_wait_group_exit() {
  local pgid=$1
  case "$pgid" in ''|*[!0-9]*|0|1) return 1 ;; esac
  for _ in $(seq 1 30); do
    if ! ps -eo pgid= 2>/dev/null | awk -v group="$pgid" '$1 == group {found=1} END {exit !found}'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

china_tunnel_reconcile_connection() {
  local marker=$1
  shift
  local -a forwards=("$@")
  local pid command_line current_count=0 keeper= old_keeper= pgid invalid=0 index replace_all=0
  local -a current_pids=() current_commands=() old_pids=() old_commands=()
  for pid in $(pgrep -f -- "$marker" || true); do
    command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
    if china_tunnel_supervisor_forwards_match "$command_line" current "$marker" "${forwards[@]}"; then
      current_count=$((current_count + 1))
      current_pids+=("$pid")
      current_commands+=("$command_line")
    elif china_tunnel_known_old_supervisor_forwards_match "$command_line" "$marker" "${forwards[@]}"; then
      old_pids+=("$pid")
      old_commands+=("$command_line")
    elif [ -n "$command_line" ] && china_tunnel_marker_match "$command_line" "$marker"; then
      invalid=1
    fi
  done
  [ "$invalid" -eq 0 ] || return 1
  if [ "$current_count" -gt 0 ]; then
    for index in "${!current_pids[@]}"; do
      if china_tunnel_supervisor_owns_forwards "${current_pids[index]}" \
          "${current_commands[index]}" current "${forwards[@]}"; then
        keeper=${current_pids[index]}
        break
      fi
    done
    if [ -z "$keeper" ]; then
      for index in "${!old_pids[@]}"; do
        if china_tunnel_supervisor_owns_forwards "${old_pids[index]}" \
            "${old_commands[index]}" old "${forwards[@]}"; then
          old_keeper=${old_pids[index]}
          break
        fi
      done
      [ -n "$old_keeper" ] || return 1
      replace_all=1
    fi
  fi
  china_tunnel_preflight_pgids=()
  for index in "${!old_pids[@]}"; do
    china_tunnel_preflight_group "${old_pids[index]}" "${old_commands[index]}" 1 "$marker" || return 1
  done
  for index in "${!current_pids[@]}"; do
    china_tunnel_preflight_group "${current_pids[index]}" "${current_commands[index]}" 0 "$marker" || return 1
  done
  for index in "${!old_pids[@]}"; do
    pgid=$(ps -o pgid= -p "${old_pids[index]}" 2>/dev/null | tr -d ' ')
    china_tunnel_stop_supervisor "${old_pids[index]}" "${old_commands[index]}" "$marker" 1 || return 1
    china_tunnel_wait_group_exit "$pgid" || return 1
  done
  for index in "${!current_pids[@]}"; do
    [ "$replace_all" = 0 ] && [ "${current_pids[index]}" = "$keeper" ] && continue
    pgid=$(ps -o pgid= -p "${current_pids[index]}" 2>/dev/null | tr -d ' ')
    china_tunnel_stop_supervisor "${current_pids[index]}" "${current_commands[index]}" "$marker" 1 || return 1
    china_tunnel_wait_group_exit "$pgid" || return 1
  done
  if [ -n "$keeper" ] && [ "$replace_all" = 0 ]; then
    return 0
  fi
  return 2
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
    -o 'ChannelTimeout=*-tcpip=45s'
    -o TCPKeepAlive=yes
    -o IPQoS=none
    -o Compression=no
  )
  local forward_text ssh_text target marker pid command_line ports_clear reconcile_status
  forward_text=$(printf ' %q' "${forwards[@]}")
  ssh_text=$(printf ' %q' "${ssh_args[@]}")
  target=$(printf '%q' "$trainer_ssh_target")
  marker="china-tunnel-$connection_name"
  if [ "${restart_tunnels:-0}" = 0 ]; then
    if china_tunnel_reconcile_connection "$marker" "${forwards[@]}"; then
      reconcile_status=0
    else
      reconcile_status=$?
    fi
    case "$reconcile_status" in
      0) return 0 ;;
      2) ;;
      *) return 1 ;;
    esac
    china_tunnel_remove_legacy "$marker" "${forwards[@]}"
    ports_clear=0
    for _ in $(seq 1 10); do
      china_tunnel_remove_orphan_children "${forwards[@]}"
      if china_tunnel_ports_clear "${forwards[@]}"; then
        ports_clear=1
        break
      fi
      sleep 1
    done
    if [ "$ports_clear" != 1 ]; then
      echo "old adapter tunnel ports did not clear for $marker" >&2
      return 1
    fi
  fi
  setsid nohup bash -c "while true; do
      ssh$ssh_text$forward_text $target
      sleep 3
    done # $marker" >/dev/null 2>&1 < /dev/null &
}

peer_tunnel() {
  local peer_keep_current peer_orphans_clear
  peer_keep_current=$peer_enabled
  [ "${restart_tunnels:-0}" = 1 ] && peer_keep_current=0
  peer_tunnel_remove_legacy "$peer_keep_current"
  peer_orphans_clear=0
  for _ in $(seq 1 10); do
    if peer_tunnel_remove_orphan_children; then
      peer_orphans_clear=1
      break
    fi
    sleep 1
  done
  if [ "$peer_orphans_clear" != 1 ]; then
    echo "old peer tunnel connections did not stop" >&2
    return 1
  fi
  [ "$peer_enabled" = 1 ] || return 0
  peer_tunnel_connection "api" \
    "-L 127.0.0.1:$peer_api_port:127.0.0.1:8000" \
    -L "127.0.0.1:$peer_api_port:127.0.0.1:8000"

  # Keep each SSH connection to two source ports so all configured source listeners
  # have independent bounded workers.
  local peer_lane_size=2
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

supervisor_lane_inventory_ready() {
  local kind=$1 expected_lanes=$2 pid command_line marker lane prefix match_function
  local inventory_count=0
  local -a seen_lanes=()
  case "$kind" in
    china) prefix=china-tunnel-lora; match_function=china_tunnel_supervisor_match ;;
    peer) prefix=peer-tunnel-lora; match_function=peer_tunnel_supervisor_match ;;
    *) return 1 ;;
  esac
  while read -r pid command_line; do
    if [[ "$command_line" =~ \#\ ($prefix-[^[:space:]]+)[[:space:]]*$ ]]; then
      marker=${BASH_REMATCH[1]}
      case "$pid" in ''|*[!0-9]*) return 1 ;; esac
      "$match_function" "$command_line" current "$marker" || return 1
      lane=${marker#${prefix}-}
      case "$lane" in ''|*[!0-9]*) return 1 ;; esac
      [ "$lane" -lt "$expected_lanes" ] || return 1
      case " ${seen_lanes[*]} " in
        *" $lane "*) return 1 ;;
        *) seen_lanes+=("$lane") ;;
      esac
      inventory_count=$((inventory_count + 1))
    fi
  done < <(ps -eo pid=,args=)
  [ "$inventory_count" -eq "$expected_lanes" ]
}

china_tunnel_supervisors_ready() {
  local lane_start=0 lane_number=0 lane_ports=() lane_forwards=() pid command_line
  local marker ready lora_source_port matching_pid matching_command match_count invalid
  local expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  supervisor_lane_inventory_ready china "$expected_lanes" || return 1
  china_tunnel_preflight_pgids=()
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    lane_ports=("${lora_source_ports[@]:lane_start:2}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=("-L 127.0.0.1:$lora_source_port:127.0.0.1:$adapter_port")
    done
    marker="china-tunnel-lora-$lane_number"
    ready=0
    matching_pid=
    match_count=0
    invalid=0
    for pid in $(pgrep -f -- "$marker" || true); do
      command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
      if china_tunnel_supervisor_forwards_match "$command_line" current "$marker" "${lane_forwards[@]}"; then
        match_count=$((match_count + 1))
        matching_pid=$pid
        matching_command=$command_line
      elif [ -n "$command_line" ] && china_tunnel_marker_match "$command_line" "$marker"; then
        invalid=1
      fi
    done
    [ "$invalid" -eq 0 ] || return 1
    [ "$match_count" -eq 1 ] || return 1
    china_tunnel_supervisor_owns_forwards "$matching_pid" "$matching_command" current \
      "${lane_forwards[@]}" || return 1
    china_tunnel_preflight_group "$matching_pid" "$matching_command" 0 "$marker" || return 1
    lane_start=$((lane_start + 2))
    lane_number=$((lane_number + 1))
  done
  supervisor_lane_inventory_ready china "$expected_lanes"
}

peer_tunnel_supervisors_ready() {
  local lane_start=0 lane_number=0 lane_ports=() lane_forwards=() pid command_line
  local marker ready lora_source_port
  [ "$peer_enabled" = 1 ] || return 0
  local expected_lanes=$(( (${#lora_source_ports[@]} + 1) / 2 ))
  supervisor_lane_inventory_ready peer "$expected_lanes" || return 1
  while [ "$lane_start" -lt "${#lora_source_ports[@]}" ]; do
    lane_ports=("${lora_source_ports[@]:lane_start:2}")
    lane_forwards=()
    for lora_source_port in "${lane_ports[@]}"; do
      lane_forwards+=("-R 127.0.0.1:$lora_source_port:127.0.0.1:$relay_port")
    done
    marker="peer-tunnel-lora-$lane_number"
    ready=0
    for pid in $(pgrep -f -- "$marker" || true); do
      command_line=$(ps -o args= -p "$pid" 2>/dev/null || true)
      if peer_tunnel_supervisor_forwards_match "$command_line" current "$marker" "${lane_forwards[@]}"; then
        ready=1
        break
      fi
    done
    [ "$ready" = 1 ] || return 1
    lane_start=$((lane_start + 2))
    lane_number=$((lane_number + 1))
  done
  supervisor_lane_inventory_ready peer "$expected_lanes"
}

adapter_tunnel_readiness() {
  local tunnels_ready=0 readiness_port="" listener_status=0 remote_ready=1
  for _ in $(seq 1 30); do
    tunnels_ready=0
    china_tunnel_supervisors_ready && china_source_ports_ready 10 && tunnels_ready=1
    if [ "$tunnels_ready" = 1 ] && [ "$peer_enabled" = 1 ]; then
      peer_tunnel_supervisors_ready || {
        tunnels_ready=0
        readiness_port=peer
      }
      if [ "$tunnels_ready" = 1 ]; then
        listener_status=0
        tunnel_listener_ready "$peer_api_port" peer-tunnel-api \
          "$peer_ssh_user@$peer_host" 1 \
          -L "127.0.0.1:$peer_api_port:127.0.0.1:8000" || listener_status=$?
        if [ "$listener_status" -ne 0 ]; then
          tunnels_ready=0
          readiness_port=$peer_api_port
        fi
      fi
    fi
    if [ "$tunnels_ready" = 1 ] && [ "$secondary_enabled" = 1 ]; then
      listener_status=0
      generic_tunnel_listener_ready "$secondary_local_api_port" "zmq-tunnel-$secondary_local_api_port" \
        "$secondary_ssh_target" \
        "127.0.0.1:$secondary_local_api_port:127.0.0.1:$secondary_remote_api_port" \
        no "${secondary_ssh_options[@]}" || listener_status=$?
      if [ "$listener_status" -ne 0 ]; then
        tunnels_ready=0
        readiness_port=$secondary_local_api_port
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
    return 1
  fi
  if [ "$peer_enabled" = 1 ]; then
    remote_ready=0
    for _ in $(seq 1 10); do
      if peer_remote_ports_ready 10; then
        remote_ready=1
        break
      fi
      sleep 1
    done
  fi
  if [ "$remote_ready" != 1 ]; then
    echo "adapter tunnel reconciliation did not establish all remote reverse forwards" >&2
    return 1
  fi
}

china_legacy_cleanup_required=0
china_legacy_cleanup_pids=()
china_legacy_cleanup_pgids=()
china_legacy_cleanup_orphan_pids=()
china_legacy_cleanup_ports=()
if [ "$restart_tunnels" = 1 ]; then
  china_tunnel_remove_all_owned
else
  china_tunnel_remove_legacy
fi
china_tunnel_remove_orphan_children
if [ "$china_legacy_cleanup_required" = 1 ] && ! china_tunnel_wait_legacy_clear; then
  exit 1
fi
china_lane_size=2
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
if [ "$secondary_enabled" = 1 ]; then
  if [ "$tunnels_only" = 1 ]; then
    tunnel "$secondary_local_api_port" "127.0.0.1:$secondary_remote_api_port" \
      "$secondary_ssh_target" no "${secondary_ssh_options[@]}"
  elif ! tunnel "$secondary_local_api_port" "127.0.0.1:$secondary_remote_api_port" \
      "$secondary_ssh_target" no "${secondary_ssh_options[@]}"; then
    echo "secondary judge tunnel failed" >&2
    exit 1
  fi
fi
if ! adapter_tunnel_readiness; then
  exit 1
fi
if [ "$tunnels_only" = 1 ]; then
  echo "adapter tunnel reconciliation complete"
  exit 0
fi
ports=("${lora_source_ports[@]}")
if [ "$peer_enabled" = 1 ]; then
  ports+=("$peer_api_port")
fi
if [ "$secondary_enabled" = 1 ]; then
  ports+=("$secondary_local_api_port")
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
  -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py":/app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:ro \
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
