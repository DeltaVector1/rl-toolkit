"""Read and operate the configured training system."""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

repo_root = Path(__file__).resolve().parents[1]
nodes_path = repo_root / "configs" / "nodes.toml"
default_config_dir = "configs/general-agent"


def _read_nodes(path: Path = nodes_path) -> dict:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read nodes config {path}: {error}") from error
    return data


nodes = _read_nodes()
hosts = nodes["hosts"]
ports = nodes["ports"]
paths = nodes["paths"]
containers = nodes["containers"]
models = nodes["models"]
inference_settings = nodes["inference"]

trainer_head = hosts["trainer_head"]
trainer_hosts = list(hosts["trainers"])
inference_hosts = list(hosts["inference"])
orchestrator_host = hosts["orchestrator"]
metric_hosts = list(hosts["metrics"])
blocked_hosts = set(hosts["blocked"])
adapter_port = ports["adapter"]
judge_api_url = nodes["judges"]["external"]["api_url"].rstrip("/")
remote_repo = paths["remote_repository"]
storage_root = paths["storage_root"]
run_root = paths["runs"]
runtime_root = f"{remote_repo}/training_ops/runtime"
trainer_launcher = f"{runtime_root}/china/cluster.sh"
orchestrator_launcher = f"{runtime_root}/h200/start_orchestrator.sh"
trainer_container = containers["trainer"]
orchestrator_container = containers["orchestrator"]
adapter_container = containers["adapter"]
adapter_relay_container = containers["adapter_relay"]
glm_inference_container = containers["glm52_inference"]
glm_model = models["glm52"]
adapter_archive_bucket = paths.get("adapter_bucket")

minimum_trend_steps = 10
trend_spread_multiplier = 2.0
stall_seconds = 600
final_validation_steps = 50
deployment_file_workers = 4
deployment_host_workers = 4
deployment_bundle_workers = 2
deployment_mkdir_timeout_seconds = 30
deployment_scp_timeout_seconds = 300
deployment_bundle_scp_timeout_seconds = 60
large_file_transfer_timeout_seconds = 7200
adapter_update_engine_startup_timeout_seconds = 900
adapter_delta_stage_timeout_seconds = 300
adapter_stage_acceptance_target_seconds = 300
trainer_seconds_max = 600
rollout_seconds_max = 720
orchestrator_repair_staging_max_seconds = 7200
adapter_archive_files = ("adapter_config.json", "adapter_model.safetensors")
dependency_manifest_name = ".training-ops-manifest"
_dependency_remote_archive_pattern = re.compile(
    r"^/tmp/trainer-dependencies-[A-Za-z0-9._-]+\.tar(?:\.gz)?$"
)


def _adapter_stage_limit_seconds(policy_step: int) -> int:
    if type(policy_step) is not int or policy_step <= 0:
        raise Refused("adapter policy step must be a positive integer")
    if policy_step == 1:
        return 900
    if policy_step == 2:
        return 600
    return adapter_delta_stage_timeout_seconds

if set(inference_hosts) != set(inference_settings):
    raise RuntimeError("each inference host must have one [inference.<host>] entry")
for inference_host in inference_hosts:
    if "glm52" not in inference_settings[inference_host]:
        raise RuntimeError(
            f"inference settings for {inference_host} are missing glm52"
        )


def _inference_config(host: str) -> dict:
    settings = inference_settings.get(host, {}).get("glm52")
    if not isinstance(settings, dict):
        raise RuntimeError(f"inference settings for {host} and glm52 are missing")
    return settings


def _config_dir(config_dir: str = default_config_dir, *, require_orchestrator: bool = False) -> dict:
    """Resolve one explicit run directory without a profile registry."""
    if not isinstance(config_dir, str) or not config_dir:
        raise Refused("run config directory must be a non-empty path")
    path = Path(config_dir)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Refused(f"run config directory is unsafe: {config_dir!r}")
    resolved = (repo_root / path).resolve()
    configs_root = (repo_root / "configs").resolve()
    if not resolved.is_relative_to(configs_root):
        raise Refused(f"run config directory must be below configs/: {config_dir!r}")
    trainer = path / "trainer.toml"
    orchestrator = path / "orchestrator.toml"
    if not (repo_root / trainer).is_file():
        raise Refused(f"run config directory has no trainer.toml: {config_dir}")
    selected = {"kind": "glm52", "trainer": trainer.as_posix()}
    if (repo_root / orchestrator).is_file():
        selected["orchestrator"] = orchestrator.as_posix()
    elif require_orchestrator:
        raise Refused(f"run config directory has no orchestrator.toml: {config_dir}")
    return selected


def _config_for_run(
    run: str,
    required: bool = True,
    config_dir: str | None = None,
) -> dict | None:
    """Select a real run directory whose output matches the run."""
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused(f"active run name is invalid: {run!r}")
    if config_dir is not None:
        candidates = [_config_dir(config_dir)]
    else:
        configs_root = repo_root / "configs"
        candidates = [
            _config_dir(trainer.parent.relative_to(repo_root).as_posix())
            for trainer in sorted(configs_root.rglob("trainer.toml"))
        ]
    matches = []
    for selected in candidates:
        trainer_path = repo_root / selected["trainer"]
        try:
            with trainer_path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError(f"cannot read trainer config {trainer_path}: {error}") from error
        output_dir = config.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir:
            raise RuntimeError(f"trainer config {trainer_path} has no output_dir")
        if Path(output_dir).name == run:
            matches.append(selected)
    if not matches and not required:
        return None
    if not matches:
        raise Refused(f"no run config directory has output_dir ending in {run!r}")
    if len(matches) > 1:
        names = ", ".join(item["trainer"] for item in matches)
        raise Refused(f"multiple run config directories match {run!r}: {names}")
    return matches[0]


def _active_policy_probe_config(run: str) -> tuple[str, str] | None:
    """Return the durable receiver-ACK directory and configured LoRA name."""
    profile = _config_for_run(run, required=False)
    if profile is None or not isinstance(profile.get("orchestrator"), str):
        return None
    path = repo_root / profile["orchestrator"]
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read orchestrator config {path}: {error}") from error
    output_dir = config.get("output_dir")
    lora = config.get("model", {}).get("lora", {})
    lora_name = lora.get("name") if isinstance(lora, dict) else None
    if not isinstance(output_dir, str) or not output_dir.startswith("/"):
        raise RuntimeError(f"orchestrator config {path} has no absolute output_dir")
    if not isinstance(lora_name, str) or not lora_name:
        return None
    return f"{output_dir.rstrip('/')}/broadcasts", lora_name

ssh_command = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "NumberOfPasswordPrompts=0",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "RequestTTY=no",
    "-o",
    "ConnectTimeout=25",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "StrictHostKeyChecking=accept-new",
]
scp_command = ["scp", "-q", *ssh_command[1:]]


class Refused(RuntimeError):
    """Raised when a call would break one of the rules in SKILL.md."""


# ------------------------------------------------------------------ plumbing


def _check_host(host: str) -> str:
    for bad in blocked_hosts:
        if re.search(rf"\b{bad}\b", host):
            raise Refused(f"{bad} is off limits. The owner said so with no exception.")
    return host


def _ssh_destination(host: str) -> tuple[list[str], list[str], str]:
    """Return SSH options, SCP options, and the configured destination."""
    _check_host(host)
    settings = inference_settings.get(host, {})
    address = settings.get("ssh_host")
    if address is None:
        return [], [], host
    port = settings.get("ssh_port")
    user = settings.get("ssh_user")
    identity = settings.get("ssh_identity_file")
    if not isinstance(address, str) or not address:
        raise RuntimeError(f"inference host {host} has an invalid ssh_host")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise RuntimeError(f"inference host {host} has an invalid ssh_port")
    if not isinstance(user, str) or not user:
        raise RuntimeError(f"inference host {host} has an invalid ssh_user")
    if not isinstance(identity, str) or not identity:
        raise RuntimeError(f"inference host {host} has an invalid ssh_identity_file")
    return (
        ["-p", str(port), "-i", identity],
        ["-P", str(port), "-i", identity],
        f"{user}@{address}",
    )


def _run_process(
    command: list[str], *, stdin, timeout: int
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _ssh(host: str, script: str, timeout: int = 120) -> tuple[int, str]:
    ssh_options, _, destination = _ssh_destination(host)
    try:
        result = _run_process(
            ssh_command + ssh_options + [destination, script],
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    output = result.stdout if result.returncode == 0 else result.stdout + result.stderr
    return result.returncode, output


def _ssh_address(host: str) -> str:
    """Return the IP selected by the configured SSH alias."""
    ssh_options, _, destination = _ssh_destination(host)
    configured = inference_settings.get(host, {}).get("ssh_host")
    if configured is not None:
        return configured
    try:
        result = subprocess.run(
            ssh_command + ssh_options + ["-G", destination],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot resolve SSH alias {host}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"cannot resolve SSH alias {host}: {detail}")
    match = re.search(r"^hostname\s+(\S+)\s*$", result.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"SSH alias {host} has no hostname")
    address = match.group(1)
    try:
        ipaddress.ip_address(address)
    except ValueError as error:
        raise RuntimeError(
            f"SSH alias {host} resolves to {address!r}, not an IP address"
        ) from error
    return address


def _remote_ok(host: str, script: str, label: str, timeout: int = 120) -> str:
    code, output = _ssh(host, script, timeout)
    if code:
        raise RuntimeError(f"{label} failed on {host}: {output.strip()}")
    return output


def _fan(hosts, fn, workers: int | None = None) -> dict:
    hosts = [_check_host(h) for h in hosts]
    with ThreadPoolExecutor(max_workers=workers or max(1, len(hosts))) as ex:
        return dict(zip(hosts, ex.map(fn, hosts)))


def _no_self_match(pattern: str) -> str:
    """Change the last token so a process check cannot match itself."""
    head, sep, tail = pattern.rpartition(" ")
    for i, ch in enumerate(tail):
        if ch.isalnum():
            return f"{head}{sep}{tail[:i]}[{ch}]{tail[i + 1:]}"
    return pattern


def _kv(out: str) -> dict[str, str]:
    """Parse `KEY value` report lines into a dict. Later lines win."""
    d: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(" ", 1)
        if (
            len(parts) == 2
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", parts[0])
        ):
            d[parts[0]] = parts[1].strip()
        elif len(parts) == 1 and parts[0].isupper() and parts[0]:
            d[parts[0]] = ""
    return d


def _age(rfc3339: str) -> float | None:
    """Seconds since a docker `-t` timestamp, or None if it will not parse."""
    try:
        t = rfc3339.strip().replace("Z", "+00:00")
        t = re.sub(r"\.(\d{6})\d+", r".\1", t)  # docker prints 9 fractional digits
        return (
            _dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(t)
        ).total_seconds()
    except Exception:
        return None


def _hms(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = int(sec)
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _shown(value) -> str:
    return "?" if value is None else str(value)


def _run_dir(run: str) -> str:
    """Remote shell fragment that resolves the run directory."""
    return (
        f"d=$(ls -d {run_root}/{run}/*/ 2>/dev/null | while read x; do "
        f'[ -f "$x/rollout-metrics.jsonl" ] && echo "$x"; done | head -1)'
    )


def _select_active_metric_rows(
    rows: list[dict],
    requested_run: str,
    active_run: str | None,
    active_start_epoch: float | None,
) -> tuple[list[dict], int, int]:
    """Select rows written by the active container, when it is this run."""
    if (
        active_run != requested_run
        or active_start_epoch is None
        or not math.isfinite(active_start_epoch)
    ):
        return rows, 0, 0

    selected: list[dict] = []
    excluded = 0
    without_time = 0
    for row in rows:
        timestamp = row.get("time")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            without_time += 1
            continue
        if not math.isfinite(float(timestamp)):
            without_time += 1
            continue
        if float(timestamp) < active_start_epoch:
            excluded += 1
        else:
            selected.append(row)
    return selected, excluded, without_time


def _parse_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return None
    return epoch if math.isfinite(epoch) else None


def _parse_trainer_compute(value: str) -> tuple[int, float | None] | None:
    """Parse the current step and optional authoritative completion duration."""
    parts = value.split()
    if len(parts) not in (1, 2):
        return None
    try:
        step = int(parts[0])
    except (TypeError, ValueError):
        return None
    if step < 0:
        return None
    seconds = _parse_seconds(parts[1]) if len(parts) == 2 else None
    if len(parts) == 2 and seconds is None:
        return None
    return step, seconds


def _parse_seconds(value: str) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _select_active_reward_rows(
    rows: list[dict],
    requested_run: str,
    active_run: str | None,
    active_start_epoch: float | None,
) -> tuple[list[dict], int, int]:
    """Keep the public reward-row helper name for existing callers."""
    return _select_active_metric_rows(
        rows, requested_run, active_run, active_start_epoch
    )


# -------------------------------------------------------------------- probes

trainer_compute_probe = r"""
started_at=$(sudo docker inspect __TRAINER__ --format '{{.State.StartedAt}}' 2>/dev/null | head -1)
started_epoch=$(date -u -d "$started_at" +%s.%N 2>/dev/null || true)
if [ -z "$started_at" ] || [ -z "$started_epoch" ]; then
  echo "TRAINER_COMPUTE_STATE unknown"
else
  trainer_log=$(sudo docker logs -t --since "$started_at" __TRAINER__ 2>&1 | strip)
  compute=$(printf '%s\n' "$trainer_log" | awk '
/TRAINER_FORWARD_BACKWARD_START step=[0-9]+/ {
  line=$0
  sub(/^.*step=/, "", line)
  current=line
  sub(/ .*/, "", current)
  active=0
  completed=""
  pp=""
  started=$1
  active=1
}
/PP step: [0-9]+ micro-batches in [0-9]+([.][0-9]+)?s/ && active {
  pp=$0
  sub(/^.*PP step: [0-9]+ micro-batches in /, "", pp)
  sub(/s.*$/, "", pp)
}
/TRAINER_FORWARD_BACKWARD_COMPLETE step=[0-9]+ seconds=[0-9]+([.][0-9]+)?/ && current != "" {
  line=$0
  sub(/^.*step=/, "", line)
  step=line
  sub(/ .*/, "", step)
  if (step == current) {
    completed=$0
    sub(/^.*seconds=/, "", completed)
    sub(/[^0-9.].*$/, "", completed)
    active=0
  }
}
END {
  if (current != "") {
    printf "%s", current
    if (completed != "") printf " %s", completed
    else printf " -"
    if (pp != "") printf " %s", pp
    else printf " -"
    printf " %s %s", active, started
    print ""
  }
}')
  read -r compute_step compute_seconds compute_pp compute_active compute_start <<EOF
$compute
EOF
  compute_elapsed=""
  if [ "$compute_active" = 1 ] && [ "$compute_start" != "-" ]; then
    compute_start_epoch=$(date -u -d "$compute_start" +%s.%N 2>/dev/null || true)
    now_epoch=$(date -u +%s.%N)
    if [ -n "$compute_start_epoch" ]; then
      compute_elapsed=$(awk -v now="$now_epoch" -v start="$compute_start_epoch" \
        'BEGIN { if (now >= start) printf "%.6f", now - start }')
    fi
  fi
  if [ "$compute_step" = "" ]; then
    echo "TRAINER_COMPUTE_STATE pending"
  else
    if [ "$compute_seconds" = "-" ]; then compute_seconds=""; fi
    if [ "$compute_pp" = "-" ]; then compute_pp=""; fi
    echo "TRAINER_COMPUTE $compute_step${compute_seconds:+ $compute_seconds}"
    echo "TRAINER_COMPUTE_PP_SECONDS $compute_pp"
    echo "TRAINER_COMPUTE_ACTIVE $compute_active"
    echo "TRAINER_COMPUTE_ELAPSED_SECONDS $compute_elapsed"
    if [ "$compute_active" = 1 ]; then
      echo "TRAINER_COMPUTE_STATE active"
    else
      echo "TRAINER_COMPUTE_STATE complete"
    fi
  fi
fi
"""

trainer_probe = r"""
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "TRAINER_UP $(sudo docker ps --filter name=^__TRAINER__$ --format up 2>/dev/null | head -1)"
trainer_cfg=$(sudo docker inspect __TRAINER__ --format '{{range .Config.Cmd}}{{println .}}{{end}}' 2>/dev/null | grep -E '\.toml$' | tail -1)
if [ -n "$trainer_cfg" ] && grep -q '^\[data\.fake\]' "$trainer_cfg" 2>/dev/null; then
  echo "MODE trainer-only"
else
  echo "MODE rl"
fi
echo "ORCH_UP $(sudo docker ps --filter name=^__ORCH__$ --format up 2>/dev/null | head -1)"
echo "TRAINER_STEPS $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -ac 'SUCCESS Step')"
echo "TRAINER_LAST $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "TRAINER_CURRENT $(sudo docker logs --tail 1 __TRAINER__ 2>&1 | strip | tail -1)"
__TRAINER_COMPUTE__
echo "TRAINER_FINISHED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'RL trainer finished|Trainer finished')"
echo "TRAINER_FAILED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'prime_rl\.trainer\.rl\.train FAILED|CheckpointException')"
echo "ORCH_LAST $(sudo docker logs __ORCH__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "BATCH $(sudo docker logs --tail 40 __ORCH__ 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1)"
echo "FAILS5M $(sudo docker logs --since 5m __ORCH__ 2>&1 | grep -ac 'Rollout failed')"
echo "TRAINER_LOGAGE $(sudo docker logs -t --tail 1 __TRAINER__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "ORCH_LOGAGE $(sudo docker logs -t --tail 1 __ORCH__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "ADAPTER $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__ADAPTER__/)"
"""


trainer_only_probe = r"""
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "MODE trainer-only"
echo "TRAINER_UP $(sudo docker ps --filter name=^__TRAINER__$ --format up 2>/dev/null | head -1)"
echo "TRAINER_STEPS $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -ac 'SUCCESS Step')"
echo "TRAINER_LAST $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "TRAINER_CURRENT $(sudo docker logs --tail 1 __TRAINER__ 2>&1 | strip | tail -1)"
__TRAINER_COMPUTE__
echo "TRAINER_FINISHED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'RL trainer finished|Trainer finished')"
echo "TRAINER_FAILED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'prime_rl\.trainer\.rl\.train FAILED|CheckpointException')"
echo "TRAINER_LOGAGE $(sudo docker logs -t --tail 1 __TRAINER__ 2>&1 | tail -1 | cut -d' ' -f1)"
"""


orchestrator_probe = r"""
strip() { sed "s/\\x1b\\[[0-9;]*m//g"; }
echo "ORCH_UP $(sudo docker ps --filter name=^__ORCH__$ --format up 2>/dev/null | head -1)"
last=$(sudo docker logs --tail 1 __ORCH__ 2>&1 | strip | tail -1)
case "$last" in
  *'Failed to finish run'|*'No space left on device'*) echo "ORCH_FAILED 1" ;;
  *) echo "ORCH_FAILED 0" ;;
esac
echo "BATCH_SYNC $(pgrep -f 'bash .*[s]ync_rollout_batches.sh' | wc -l)"
echo "TRAINER_FAIL_WD $(pgrep -f 'bash .*[t]rainer_failure_watchdog.sh' | wc -l)"
echo "ORCH_LAST $(sudo docker logs __ORCH__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "BATCH $(sudo docker logs --tail 40 __ORCH__ 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1)"
echo "FAILS5M $(sudo docker logs --since 5m __ORCH__ 2>&1 | grep -ac 'Rollout failed')"
echo "ORCH_LOGAGE $(sudo docker logs -t --tail 1 __ORCH__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "JUDGE $(curl -s -o /dev/null -m 10 -w '%{http_code}' __JUDGE_URL__/models)"
"""

node_probe = r"""
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '{m+=$1; u+=$2; n++} END{printf "GPU %d %d %d\n", m/n, u/n, n}'
free -g | sed -n 2p | awk '{print "MEM " $7 " " $5}'
df -BG --output=avail,pcent /data | tail -1 | awk '{gsub(/G|%/, ""); print "DISK " $1 " " $2}'
echo "CT $(sudo docker ps --filter name=^__TRAINER__$ --format '{{.Status}}' 2>/dev/null | head -1)"
"""

inference_probe = r"""
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
  | awk '{u+=$1; n++} END{printf "GPU %d %d\n", u/n, n}'
ok=0
total=0
longest=0
policy_state=unknown
policy_step=
policy_slot=
policy_target_sha256=
policy_target_size=
policy_config_sha256=
policy_config_size=
policy_marker=
if [ -n "__POLICY_BROADCAST_DIR__" ] && [ -d "__POLICY_BROADCAST_DIR__" ]; then
  policy_marker=$(find "__POLICY_BROADCAST_DIR__" -mindepth 2 -maxdepth 2 \
    -type f -name receivers-ready.json -printf '%p\n' 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$policy_marker" ]; then
  policy_values=$(python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); print("\t".join(str(value.get(name, "")) for name in ("step","slot","target_sha256","target_size","config_sha256","config_size")))' "$policy_marker" 2>/dev/null || true)
  read -r policy_step policy_slot policy_target_sha256 policy_target_size policy_config_sha256 policy_config_size <<<"$policy_values"
  if [ -n "$policy_step" ] && [ -n "$policy_slot" ] && [ -n "$policy_target_sha256" ] \
    && [ -n "$policy_target_size" ] && [ -n "$policy_config_sha256" ] && [ -n "$policy_config_size" ]; then
    policy_state=ready
  fi
fi
for port in __PORTS__; do
  total=$((total+1))
  started=$(date +%s%N)
  stage_json=$(curl -fsS -m 10 "http://127.0.0.1:$port/staging_status" 2>/dev/null || true)
  stage_state=$(printf '%s' "$stage_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' 2>/dev/null || true)
  if [ "$stage_state" = staging ]; then
    policy_state=staging
    code=STAGING
  elif [ "$policy_state" != ready ]; then
    code=POLICY_UNKNOWN
  else
    active_model=$(printf '%s' "$stage_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); slot=sys.argv[1]; expected={"target_step":int(sys.argv[2]),"target_sha256":sys.argv[3],"target_size":int(sys.argv[4]),"config_sha256":sys.argv[5],"config_size":int(sys.argv[6])}; receipts=d.get("loaded_receipts", {}); receipt=receipts.get(slot) if isinstance(receipts, dict) else None; print(slot if isinstance(receipt, dict) and all(receipt.get(k)==v for k,v in expected.items()) else "")' "$policy_slot" "$policy_step" "$policy_target_sha256" "$policy_target_size" "$policy_config_sha256" "$policy_config_size" 2>/dev/null || true)
    if [ -z "$active_model" ]; then
      code=POLICY_MISMATCH
    else
      code=$(curl -s -o /dev/null -m 25 -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/completions" \
          -H 'Content-Type: application/json' \
          -d "{\"model\":\"$active_model\",\"prompt\":\"hi\",\"max_tokens\":3}")
    fi
  fi
  if [ "$code" = 200 ]; then
    :
  elif [ "$code" = STAGING ]; then
    policy_state=staging
  elif [ "$policy_state" = ready ]; then
    policy_state=mismatch
  fi
  if [ "$code" = 200 ]; then
    ok=$((ok+1))
  fi
  ended=$(date +%s%N)
  elapsed=$(( (ended-started)/1000000 ))
  [ "$elapsed" -gt "$longest" ] && longest=$elapsed
done
[ "$total" -gt 0 ] || exit 3
[ "$policy_state" = staging ] && code=STAGING
[ "$policy_state" = ready ] && [ "$ok" -eq "$total" ] && code=200
[ "$code" = "" ] && code=000
echo "GEN $code $longest"
echo "PROBES $ok $total"
echo "POLICY $policy_state ${policy_step:-unknown}"
echo "RUNNING $(sudo docker logs --since 60s __ENGINE__ 2>&1 | grep -aoE 'Running: [0-9]+' | grep -oE '[0-9]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}')"
echo "TPUT $(sudo docker logs --since 60s __ENGINE__ 2>&1 | grep -aoE 'generation throughput: [0-9.]+' | grep -oE '[0-9.]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}')"
echo "WD $(pgrep -f 'bash .*[e]ngine_watchdog.sh' | wc -l)"
"""


def _inference_script(
    host: str,
    model: str,
    container: str,
    policy_broadcast_dir: str = "",
) -> str:
    settings = _inference_config(host)
    gpu_count = settings.get("gpu_count")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise RuntimeError(f"GPU count for {host} is invalid")
    tensor_parallel_size = settings.get("tensor_parallel_size")
    data_parallel_size = settings.get("data_parallel_size", 1)
    if (
        not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size < 1
        or not isinstance(data_parallel_size, int)
        or data_parallel_size < 1
        or gpu_count % (tensor_parallel_size * data_parallel_size)
    ):
        raise RuntimeError(f"parallel sizes for {host} are invalid")
    ports = settings.get("ports")
    if not isinstance(ports, list) or not ports or not all(
        isinstance(port, int) and 1 <= port <= 65535 for port in ports
    ):
        raise RuntimeError(f"inference ports for {host} are invalid")
    if len(ports) != gpu_count // (tensor_parallel_size * data_parallel_size):
        raise RuntimeError(f"inference replica count for {host} is inconsistent")
    return (
        inference_probe.replace("__MODEL__", model)
        .replace("__ENGINE__", container)
        .replace("__PORTS__", " ".join(map(str, ports)))
        .replace("__POLICY_BROADCAST_DIR__", policy_broadcast_dir)
    )


def _head_script() -> str:
    return (
        trainer_probe.replace("__TRAINER__", trainer_container)
        .replace(
            "__TRAINER_COMPUTE__",
            trainer_compute_probe.replace("__TRAINER__", trainer_container),
        )
        .replace("__ORCH__", orchestrator_container)
        .replace("__ADAPTER__", str(adapter_port))
    )


def _trainer_mode() -> str:
    """Return `trainer-only` when the active trainer uses `[data.fake]`."""
    rc, out = _ssh(
        trainer_head,
        f"cfg=$(sudo docker inspect {trainer_container} --format "
        "'{{range .Config.Cmd}}{{println .}}{{end}}' 2>/dev/null | "
        "grep -E '\\.toml$' | tail -1); "
        "if [ -n \"$cfg\" ] && grep -q '^\\[data\\.fake\\]' \"$cfg\" 2>/dev/null; "
        "then echo trainer-only; else echo rl; fi",
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(f"cannot read trainer mode on {trainer_head}: {out.strip()}")
    mode = out.strip().splitlines()[-1]
    if mode not in {"trainer-only", "rl"}:
        raise RuntimeError(f"trainer returned an invalid mode: {mode!r}")
    return mode


# ------------------------------------------------------------------- results


@dataclass
class NodeState:
    host: str
    gpu_mem_mb: int = 0
    gpu_util: int = 0
    gpus: int = 0
    mem_free_gb: int = 0
    mem_shared_gb: int = 0
    data_free_gb: int = 0
    data_use_percent: int = 0
    container: str = ""
    reachable: bool = True


@dataclass
class EngineState:
    host: str
    gen_code: str = "000"
    gen_ms: int = 0
    probes_ok: int = 0
    probes_total: int = 0
    gpu_util: int = 0
    running: int = 0
    throughput: float = 0.0
    watchdog: bool = False
    policy_state: str = "ready"
    policy_step: int | None = None
    reachable: bool = True

    @property
    def healthy(self) -> bool:
        return self.gen_code == "200" and self.policy_state == "ready"


@dataclass
class AcceptanceReport:
    """Read-only TRAINING_EXPECTATIONS evidence and alarms."""

    run: str = ""
    profile: str = "glm52"
    trainer_config_path: str = ""
    orchestrator_config_path: str = ""
    batch_size: int | None = None
    group_size: int | None = None
    context_length: int | None = None
    max_output_lengths: list[int] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    max_norm: float | None = None
    weight_decay: float | None = None
    native_fp8: bool | None = None
    precision_ok: bool | None = None
    completed_steps: int = 0
    trainer_compute_step: int | None = None
    trainer_compute_seconds: float | None = None
    rollout_generation_step: int | None = None
    rollout_generation_seconds: float | None = None
    large_batch_evidence: bool = False
    reward_verdict: str = "no data"
    failures: dict[str, int] = field(default_factory=dict)
    alarms: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.alarms

    def report(self) -> str:
        values = (
            f"batch={_shown(self.batch_size)} group={_shown(self.group_size)} "
            f"context={_shown(self.context_length)} "
            f"max_output={','.join(map(str, self.max_output_lengths)) or '?'} "
            f"lr={','.join(f'{x:g}' for x in self.learning_rates) or '?'} "
            f"max_norm={_shown(self.max_norm)} weight_decay={_shown(self.weight_decay)}"
        )
        precision = (
            "yes"
            if self.precision_ok is True
            else "no"
            if self.precision_ok is False
            else "?"
        )
        trainer = (
            f"step {self.trainer_compute_step}: {_hms(self.trainer_compute_seconds)}"
            if self.trainer_compute_step is not None
            else "no trainer timing measurement"
        )
        generation = (
            f"step {self.rollout_generation_step}: {_hms(self.rollout_generation_seconds)}"
            if self.rollout_generation_step is not None
            else "no separate generation measurement"
        )
        L = [f"training requirements for {self.run}"]
        L.append(f"profile: {self.profile}")
        L.append(f"config: {values}")
        L.append(f"native FP8 trainer: {precision}")
        L.append(
            f"trainer time/forward_backward (includes optimizer): {trainer} "
            f"(upper bound {trainer_seconds_max:g}s)"
        )
        L.append(
            f"active rollout generation time: {generation} "
            f"(upper bound {rollout_seconds_max:g}s)"
        )
        L.append(
            "excluded from both times: policy/check waiting, judging/scoring, batch transfer, "
            "adapter transfer, and checkpoint save"
        )
        stability = (
            f"stability: {self.completed_steps}/{final_validation_steps} completed steps | "
            f"reward verdict={self.reward_verdict}"
        )
        if self.profile == "glm52":
            stability += (
                " | 2,048-batch validation="
                + ("yes" if self.large_batch_evidence else "no")
            )
        L.append(stability)
        if self.failures:
            L.append(
                "failure evidence: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.failures.items()))
            )
        L.append(f"result: {'pass' if self.passed else 'fail'}")
        for alarm in self.alarms:
            L.append(f"requirement failed: {alarm}")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


@dataclass
class Status:
    when: str = ""
    run: str = ""
    mode: str = "rl"
    trainer_up: bool = False
    trainer_finished: bool = False
    trainer_failed: bool = False
    orch_up: bool = False
    orch_failed: bool = False
    batch_sync: bool = False
    trainer_failure_watchdog: bool = False
    trainer_steps: int = 0
    trainer_compute_state: str = "pending"
    trainer_compute_step: int | None = None
    trainer_compute_seconds: float | None = None
    trainer_compute_pp_seconds: float | None = None
    trainer_compute_active: bool = False
    trainer_compute_elapsed_seconds: float | None = None
    trainer_last: str = ""
    trainer_current: str = ""
    orch_last: str = ""
    batch: str = ""
    fails_5m: int = 0
    trainer_log_age: float | None = None
    orch_log_age: float | None = None
    judge: str = "000"
    adapter: str = "000"
    nodes: list[NodeState] = field(default_factory=list)
    engines: list[EngineState] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    acceptance_alarms: list[str] = field(default_factory=list)

    def report(self) -> str:
        L = [f"{self.when} UTC | run {self.run}"]
        if self.mode == "trainer-only":
            trainer_state = (
                "failed"
                if self.trainer_failed
                else "finished"
                if self.trainer_finished
                else "up"
                if self.trainer_up
                else "down"
            )
            L.append(
                f"trainer {trainer_state} | "
                "mode trainer-only qualification | rollout components not required"
            )
        else:
            trainer_state = (
                "failed"
                if self.trainer_failed
                else "finished"
                if self.trainer_finished
                else "up"
                if self.trainer_up
                else "down"
            )
            L.append(
                f"trainer {trainer_state} | "
                f"orchestrator "
                f"{'failed' if self.orch_failed else 'up' if self.orch_up else 'finished' if self.trainer_finished else 'down'} | "
                f"judge {self.judge} | adapter {self.adapter}"
            )
        L.append(
            f"run steps {self.trainer_steps} | failures in 5m {self.fails_5m} | {self.batch or 'no batch line'}"
        )
        if self.trainer_compute_state == "unknown":
            compute = "unknown"
        elif (
            self.trainer_compute_state == "active"
            and self.trainer_compute_active
            and self.trainer_compute_step is not None
        ):
            compute = f"step {self.trainer_compute_step} active"
            if self.trainer_compute_elapsed_seconds is not None:
                compute += f" {self.trainer_compute_elapsed_seconds:.2f}s"
        elif (
            self.trainer_compute_state == "complete"
            and self.trainer_compute_step is not None
            and self.trainer_compute_seconds is not None
        ):
            compute = (
                f"step {self.trainer_compute_step} "
                f"{self.trainer_compute_seconds:.2f}s (forward/backward complete)"
            )
        else:
            compute = "pending"
        if self.trainer_compute_pp_seconds is not None:
            compute += f"; PP {self.trainer_compute_pp_seconds:.2f}s"
        L.append(f"  trainer compute: {compute}")
        if self.trainer_last:
            L.append(f"  trainer: {self.trainer_last[:150]}")
        elif self.trainer_current:
            L.append(f"  trainer now: {self.trainer_current[:150]}")
        if self.orch_last:
            L.append(f"  orch:    {self.orch_last[:150]}")
        if self.mode == "trainer-only":
            L.append(f"  last log line: trainer {_hms(self.trainer_log_age)} ago")
        else:
            L.append(
                f"  last log line: trainer {_hms(self.trainer_log_age)} ago, "
                f"orchestrator {_hms(self.orch_log_age)} ago"
            )
            L.append(
                f"  supervisors: local=disabled "
                f"batch-sync={'yes' if self.batch_sync else 'no'} "
                f"trainer-check={'yes' if self.trainer_failure_watchdog else 'no'}"
            )
        L.append("trainer nodes")
        for n in self.nodes:
            if not n.reachable:
                L.append(f"{n.host:<5} unreachable")
                continue
            L.append(
                f"{n.host:<5} {n.gpu_mem_mb / 1024:5.1f} GiB/gpu  {n.gpu_util:3d}% util  "
                f"{n.mem_free_gb:4d}G RAM  {n.data_free_gb:4d}G disk "
                f"({n.data_use_percent:2d}% used)  {n.container[:28]}"
            )
        if self.mode != "trainer-only":
            L.append("inference")
            for e in self.engines:
                if not e.reachable:
                    L.append(f"{e.host:<6} unreachable")
                    continue
                L.append(
                    f"{e.host:<6} probes={e.probes_ok}/{e.probes_total} in {e.gen_ms:>6d} ms  {e.gpu_util:3d}% util  "
                    f"running={e.running:<4d} {e.throughput:8.1f} tok/s  "
                    f"policy={e.policy_state}"
                    f"{f' v{e.policy_step}' if e.policy_step is not None else ''}  "
                    f"watchdog={'yes' if e.watchdog else 'no'}"
                )
        for w in self.warnings:
            L.append(f"!! {w}")
        for alarm in self.acceptance_alarms:
            L.append(f"requirement failed: {alarm}")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


# ------------------------------------------------------ acceptance reporting


def _strip_toml_comment(line: str) -> str:
    """Remove a TOML comment while preserving quoted # characters."""
    quote = ""
    escaped = False
    out = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote == '"':
            out.append(ch)
            escaped = True
            continue
        if ch in ("'", '"'):
            if not quote:
                quote = ch
            elif quote == ch:
                quote = ""
            out.append(ch)
            continue
        if ch == "#" and not quote:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_toml_value(raw: str):
    raw = _strip_toml_comment(raw).strip()
    if not raw:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    number = raw.replace("_", "")
    try:
        return (
            float(number)
            if any(c in number.lower() for c in (".", "e"))
            else int(number)
        )
    except ValueError:
        return raw


def _parse_toml_scalars(text: str) -> dict[str, list[object]]:
    """Read the scalar values used by the acceptance report."""
    section = ""
    values: dict[str, list[object]] = {}
    for raw_line in text.splitlines():
        line = _strip_toml_comment(raw_line)
        if not line:
            continue
        table = re.fullmatch(r"\[\[?([^\]]+?)\]?\]", line)
        if table:
            section = table.group(1).strip()
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)\s*=\s*(.+)$", line)
        if not match:
            continue
        key = f"{section}.{match.group(1)}" if section else match.group(1)
        value = _parse_toml_value(match.group(2))
        if (
            value is not None
            and not isinstance(value, str)
            or (isinstance(value, str) and not value.startswith(("[", "{")))
        ):
            values.setdefault(key, []).append(value)
    return values


def _last(values: dict[str, list[object]], key: str):
    found = values.get(key, [])
    return found[-1] if found else None


def _numbers(values: dict[str, list[object]], keys: tuple[str, ...]) -> list[float]:
    out = []
    for key in keys:
        for value in values.get(key, []):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
    return out


def _expected_value(values: dict[str, list[object]], keys: tuple[str, ...]):
    """Return one run setting, or None when it is absent or inconsistent."""
    found = []
    for key in keys:
        found.extend(values.get(key, []))
    if not found:
        return None
    first = found[0]
    if all(value == first for value in found[1:]):
        return first
    return None


def _parse_jsonl_metrics(text: str) -> list[dict]:
    """Merge valid JSONL monitor records by step."""
    by_step: dict[int, dict] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (TypeError, ValueError) as error:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise RuntimeError(f"invalid metric JSON on line {index + 1}") from error
        if not isinstance(row, dict) or not isinstance(row.get("step"), (int, float)):
            raise RuntimeError(f"metric line {index + 1} has no numeric step")
        step = int(row["step"])
        by_step.setdefault(step, {"step": step}).update(row)
    return [by_step[step] for step in sorted(by_step)]


def _near(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)


def _failure_evidence(
    trainer_rows: list[dict], rollout_rows: list[dict], logs: str
) -> dict[str, int]:
    """Count measured failure evidence. Counts are lines or affected metric steps."""
    failures: dict[str, int] = {}

    rollout_error_steps = sum(
        1
        for row in rollout_rows
        if float(row.get("train/agg/all/has_error/mean") or 0.0) > 0.0
    )
    environment_steps = sum(
        1
        for row in rollout_rows
        if any(
            isinstance(value, (int, float))
            and value > 0
            and (
                "user_sim_failed" in key
                or "environment_error" in key
                or "env_error" in key
            )
            for key, value in row.items()
        )
    )
    if rollout_error_steps:
        failures["transport/rollout"] = rollout_error_steps
    if environment_steps:
        failures["environment"] = environment_steps

    categories = {
        "transport": re.compile(
            r"rollout failed|readtimeout|connect(?:ion)?error|transport(?:error| failure)|"
            r"(?:batch|file).*(?:transfer|sync).*(?:failed|error|mismatch)|"
            r"checksum.*(?:failed|mismatch)|tunnel.*(?: down|failed)",
            re.I,
        ),
        "adapter": re.compile(
            r"(?:adapter|lora).*(?:failed|error|timeout|mismatch|not visible)|"
            r"(?:failed|error).*(?:adapter|lora)",
            re.I,
        ),
        "environment": re.compile(
            r"(?:environment|user.sim(?:ulation)?).*(?:failed|error|exception)|"
            r"(?:failed|error).*(?:environment|user.sim(?:ulation)?)",
            re.I,
        ),
        "stability": re.compile(
            r"out of memory|\boom\b|\b(?:nan|inf)\b|nonfinite", re.I
        ),
    }
    lines = list(
        dict.fromkeys(line.strip() for line in logs.splitlines() if line.strip())
    )
    for category, pattern in categories.items():
        count = sum(
            1
            for line in lines
            if pattern.search(line)
            and not re.search(
                r"Out of range float values are not JSON compliant:\s*nan",
                line,
                re.I,
            )
            and not re.search(
                r"(?:error|failed|nonfinite)[^0-9]{0,8}0(?:\.0+)?%?", line, re.I
            )
        )
        if count:
            failures[category] = failures.get(category, 0) + count

    nonfinite_steps = 0
    for row in trainer_rows:
        bad = float(row.get("diagnostic/nonfinite_gradient") or 0.0) > 0
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                bad = True
        nonfinite_steps += int(bad)
    if nonfinite_steps:
        failures["stability"] = failures.get("stability", 0) + nonfinite_steps
    return failures


def _large_batch_evidence(
    orchestrator_values: dict[str, list[object]],
    trainer_rows: list[dict],
    logs: str,
) -> bool:
    batch = _last(orchestrator_values, "batch_size")
    if batch == 2048 and any(row["step"] >= 1 for row in trainer_rows):
        return True
    explicit = re.search(r"\b(?:batch(?:_size)?\s*[=: ]\s*2048|b2048)\b", logs, re.I)
    completed = re.search(
        r"\b(?:SUCCESS Step|validated|acceptance|completed)\b", logs, re.I
    )
    return bool(explicit and completed)


def _acceptance_from_text(
    run: str,
    trainer_config: str,
    orchestrator_config: str,
    trainer_metrics: str,
    rollout_metrics: str,
    logs: str,
    reward_verdict: str,
    trainer_active_run: str | None = None,
    trainer_start_epoch: float | None = None,
    rollout_active_run: str | None = None,
    rollout_start_epoch: float | None = None,
) -> AcceptanceReport:
    """Build an acceptance report from text. This function never uses SSH."""
    trainer_values = _parse_toml_scalars(trainer_config)
    orch_values = _parse_toml_scalars(orchestrator_config)
    trainer_rows = _parse_jsonl_metrics(trainer_metrics)
    rollout_rows = _parse_jsonl_metrics(rollout_metrics)
    trainer_rows, _, _ = _select_active_metric_rows(
        trainer_rows, run, trainer_active_run, trainer_start_epoch
    )
    rollout_rows, _, _ = _select_active_metric_rows(
        rollout_rows, run, rollout_active_run, rollout_start_epoch
    )
    rep = AcceptanceReport(run=run, profile="glm52", reward_verdict=reward_verdict)
    expected_batch = _expected_value(orch_values, ("batch_size",))
    expected_group = _expected_value(orch_values, ("group_size",))
    expected_context = _expected_value(trainer_values, ("model.seq_len",))
    expected_output = _expected_value(
        orch_values,
        (
            "train.sampling.max_completion_tokens",
            "train.env.sampling.max_completion_tokens",
            "train.env.env.agent.max_output_tokens",
        ),
    )
    expected_learning_rate = _expected_value(
        trainer_values, ("optim.lr",)
    )
    orch_learning_rate = _expected_value(orch_values, ("optim.lr",))
    if expected_learning_rate is None or orch_learning_rate != expected_learning_rate:
        expected_learning_rate = None
    expected_max_norm = _expected_value(trainer_values, ("optim.max_norm",))
    expected_weight_decay = _expected_value(
        trainer_values, ("optim.weight_decay",)
    )

    rep.batch_size = _last(orch_values, "batch_size")
    rep.group_size = _last(orch_values, "group_size")
    rep.context_length = _last(trainer_values, "model.seq_len")
    output_values = _numbers(
        orch_values,
        (
            "train.sampling.max_completion_tokens",
            "train.env.sampling.max_completion_tokens",
            "train.env.env.agent.max_output_tokens",
        ),
    )
    rep.max_output_lengths = sorted(set(int(v) for v in output_values))
    rep.learning_rates = _numbers(trainer_values, ("optim.lr",)) + _numbers(
        orch_values, ("optim.lr",)
    )
    rep.max_norm = _last(trainer_values, "optim.max_norm")
    rep.weight_decay = _last(trainer_values, "optim.weight_decay")

    quant_type = _last(trainer_values, "model.quantization.type")
    dequantize_checkpoint = _last(
        trainer_values, "model.quantization.dequantize_checkpoint"
    )
    optional_dequant_flags = [
        _last(trainer_values, key)
        for key in (
            "model.quantization.dequantize_checkpoint_linears",
            "model.quantization.dequantize_checkpoint_moe",
        )
    ]
    if not trainer_config.strip():
        rep.native_fp8 = None
        rep.precision_ok = None
    else:
        rep.native_fp8 = (
            quant_type == "fp8"
            and dequantize_checkpoint is False
            and not any(flag is True for flag in optional_dequant_flags)
        )
        rep.precision_ok = rep.native_fp8

    rep.completed_steps = max((row["step"] for row in trainer_rows), default=0)
    compute = []
    invalid_compute = []
    for row in trainer_rows:
        if row["step"] < 1:
            continue
        value = row.get("time/forward_backward")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            invalid_compute.append(row["step"])
            continue
        compute.append((row["step"], float(value)))
    if compute:
        rep.trainer_compute_step, rep.trainer_compute_seconds = compute[-1]

    generation_keys = (
        "train/agg/all/timing/generation/max",
        "train/decensor/all/timing/generation/max",
    )
    generation = []
    invalid_generation = []
    for row in rollout_rows:
        present = next((key for key in generation_keys if key in row), None)
        value = row.get(present) if present is not None else None
        if (
            present is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            invalid_generation.append(row["step"])
            continue
        generation.append((row["step"], float(value)))
    if generation:
        rep.rollout_generation_step, rep.rollout_generation_seconds = generation[-1]

    rep.large_batch_evidence = _large_batch_evidence(orch_values, trainer_rows, logs)
    rep.failures = _failure_evidence(trainer_rows, rollout_rows, logs)

    def expected(name: str, actual, want) -> None:
        if actual is None or actual != want:
            rep.alarms.append(f"{name} is {_shown(actual)}; expected {want}")

    expected("batch size", rep.batch_size, expected_batch)
    expected("group size", rep.group_size, expected_group)
    expected("context length", rep.context_length, expected_context)
    if not rep.max_output_lengths or any(
        expected_output is None or v != expected_output for v in rep.max_output_lengths
    ):
        shown = ",".join(map(str, rep.max_output_lengths)) or "?"
        rep.alarms.append(
            f"maximum output length is {shown}; expected {expected_output}"
        )
    if len(rep.learning_rates) < 2 or any(
        expected_learning_rate is None
        or not _near(v, float(expected_learning_rate))
        for v in rep.learning_rates
    ):
        shown = ",".join(f"{v:g}" for v in rep.learning_rates) or "?"
        rep.alarms.append(
            f"trainer/orchestrator learning rates are {shown}; expected {expected_learning_rate}"
        )
    if expected_max_norm is None or rep.max_norm is None or not _near(rep.max_norm, float(expected_max_norm)):
        rep.alarms.append(
            f"maximum gradient norm is {_shown(rep.max_norm)}; expected {expected_max_norm}"
        )
    if expected_weight_decay is None or rep.weight_decay is None or not _near(rep.weight_decay, float(expected_weight_decay)):
        rep.alarms.append(
            f"weight decay is {_shown(rep.weight_decay)}; expected {expected_weight_decay}"
        )
    if rep.precision_ok is not True:
        rep.alarms.append("trainer is not confirmed on native FP8 checkpoint weights")

    bad_compute = [step for step, seconds in compute if seconds > trainer_seconds_max]
    if invalid_compute:
        rep.alarms.append(
            "trainer time/forward_backward is missing or invalid at steps "
            + ",".join(map(str, invalid_compute[-8:]))
        )
    if bad_compute:
        rep.alarms.append(
            f"trainer time/forward_backward exceeds configured upper bound "
            f"{trainer_seconds_max:g}s at steps "
            f"{','.join(map(str, bad_compute[-8:]))}"
        )
    elif rep.completed_steps >= 1 and not compute and not invalid_compute:
        rep.alarms.append(
            "no trainer time/forward_backward measurement"
        )

    bad_generation = [step for step, seconds in generation if seconds > rollout_seconds_max]
    if invalid_generation:
        rep.alarms.append(
            "active rollout generation time is missing or invalid at steps "
            + ",".join(map(str, invalid_generation[-8:]))
        )
    if bad_generation:
        rep.alarms.append(
            f"active rollout generation time exceeds configured upper bound "
            f"{rollout_seconds_max:g}s at steps "
            f"{','.join(map(str, bad_generation[-8:]))}"
        )
    elif rollout_rows and not generation and not invalid_generation:
        rep.alarms.append("no active rollout generation time measurement")

    for category, count in sorted(rep.failures.items()):
        rep.alarms.append(f"{category} failure evidence is present ({count})")
    if reward_verdict == "falling":
        rep.alarms.append("effective reward verdict is falling")
    if not rep.large_batch_evidence:
        rep.alarms.append("no measured batch-2048 validation evidence")
    if rep.completed_steps < final_validation_steps:
        rep.alarms.append(
            f"final stability validation has {rep.completed_steps}/{final_validation_steps} completed trainer steps"
        )
    return rep


acceptance_log_pattern = (
    r"SUCCESS Step|batch(?:_size)?[=: ]+2048|b2048|out of memory|\bOOM\b|"
    r"\bNaN\b|\bInf\b|nonfinite|adapter|LoRA|Rollout failed|ReadTimeout|"
    r"ConnectError|TransportError|environment|user.sim|checksum|tunnel.*DOWN"
)


def _acceptance_probe_script(run: str, role: str) -> str:
    root = shlex.quote(f"{run_root}/{run}")
    run_prefix = shlex.quote(f"{run_root}/")
    if role == "trainer":
        container = trainer_container
        metric_name = "trainer-metrics.jsonl"
        containers = trainer_container
    elif role == "edge":
        container = orchestrator_container
        metric_name = "rollout-metrics.jsonl"
        containers = f"{orchestrator_container} {glm_inference_container}"
    else:
        container = glm_inference_container
        metric_name = "__none__"
        containers = glm_inference_container
    return f"""
cfg=$(sudo docker inspect {container} --format '{{{{range .Config.Cmd}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null |
      grep -E '\\.toml$' | tail -1)
container_cmd=$(sudo docker inspect {container} --format '{{{{range .Config.Cmd}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null || true)
container_env=$(sudo docker inspect {container} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null || true)
active_run=$(printf '%s\\n' "$container_env" | sed -n 's/^RUN=//p' | head -1)
active_run_source=RUN
if [ -z "$active_run" ]; then
  active_run=$(printf '%s\\n' "$container_env" | sed -n 's#^HF_HOME=/data/glm52-rl/runs/\\([^/]*\\)/.*#\\1#p' | head -1)
  active_run_source=HF_HOME
fi
if [ -z "$active_run" ]; then
  active_run=$(printf '%s\\n' "$container_cmd" | awk -v root={run_prefix} '$0 ~ "^--log-dir=" root {{ value=substr($0, 11 + length(root)); sub("/.*$", "", value); print value; exit }}')
  active_run_source=log-dir
fi
started_at=$(sudo docker inspect {container} --format '{{{{.State.StartedAt}}}}' 2>/dev/null | head -1)
start_epoch=
if [ -n "$started_at" ]; then start_epoch=$(date -u -d "$started_at" +%s.%N 2>/dev/null || true); fi
printf 'ACTIVE_RUN %s\\n' "$active_run"
printf 'ACTIVE_RUN_SOURCE %s\\n' "$active_run_source"
printf 'START_EPOCH %s\\n' "$start_epoch"
metric=$(find {root} -maxdepth 3 -type f -name {shlex.quote(metric_name)} -print 2>/dev/null | head -1)
emit() {{ printf '%s_B64 ' "$1"; if [ -n "$2" ] && [ -r "$2" ]; then "$3" "$2" | base64 -w0; fi; printf '\\n'; }}
emit PATH "$cfg" cat
emit CONFIG "$cfg" cat
emit METRICS "$metric" tail
logs=""
for ct in {containers}; do
  if [ -n "$started_at" ]; then
    current_logs=$(sudo docker logs --since "$started_at" --tail 20000 "$ct" 2>&1 || true)
  else
    current_logs=$(sudo docker logs --tail 20000 "$ct" 2>&1 || true)
  fi
  logs="${{logs}}
$(printf '%s' "$current_logs" | grep -aiE {shlex.quote(acceptance_log_pattern)} | tail -200)"
done
printf 'LOGS_B64 '; printf '%s' "$logs" | base64 -w0; printf '\\n'
"""


def _decode_acceptance_probe(out: str) -> dict[str, str]:
    decoded = {}
    for line in out.splitlines():
        match = re.match(
            r"^(ACTIVE_RUN_SOURCE|ACTIVE_RUN|START_EPOCH)(?: +(.*))?$", line
        )
        if match:
            decoded[match.group(1).lower()] = (match.group(2) or "").strip()
            continue
        match = re.match(r"^(PATH|CONFIG|METRICS|LOGS)_B64(?: +(.*))?$", line)
        if not match:
            continue
        try:
            decoded[match.group(1).lower()] = base64.b64decode(
                match.group(2) or "", validate=True
            ).decode("utf-8", errors="replace")
        except (ValueError, TypeError) as error:
            raise RuntimeError("acceptance probe returned invalid base64") from error
    return decoded


def acceptance(run: str = "", quiet: bool = False) -> AcceptanceReport:
    """Read active configs and measured logs, then apply TRAINING_EXPECTATIONS.

    This is read-only. Trainer compute uses only ``time/forward_backward``.
    Rollout generation uses only the environment's ``timing/generation/max``.
    The report never adds policy waiting, judging, transfer, or checkpoint time
    to either measurement.
    """
    run = run or current_run()
    if _trainer_mode() == "trainer-only":
        raise Refused(
            "acceptance() is for production RL runs. Use status() for the "
            "20-step trainer-only qualification and run direct parity separately."
        )
    with ThreadPoolExecutor(max_workers=3 + len(inference_hosts)) as ex:
        f_trainer = ex.submit(_ssh, trainer_head, _acceptance_probe_script(run, "trainer"), 180)
        f_edge = ex.submit(_ssh, orchestrator_host, _acceptance_probe_script(run, "edge"), 180)
        engine_futures = {
            host: ex.submit(_ssh, host, _acceptance_probe_script(run, "engine"), 180)
            for host in inference_hosts
        }
        f_reward = ex.submit(reward, run, True)
        trainer_code, trainer_output = f_trainer.result()
        if trainer_code:
            raise RuntimeError(
                f"cannot read acceptance data on {trainer_head}: {trainer_output.strip()}"
            )
        edge_code, edge_output = f_edge.result()
        if edge_code:
            raise RuntimeError(
                f"cannot read acceptance data on {orchestrator_host}: {edge_output.strip()}"
            )
        trainer = _decode_acceptance_probe(trainer_output)
        edge = _decode_acceptance_probe(edge_output)
        engine_logs = []
        for host, future in engine_futures.items():
            code, output = future.result()
            if code:
                raise RuntimeError(f"cannot read inference logs on {host}: {output.strip()}")
            engine_logs.append(_decode_acceptance_probe(output).get("logs", ""))
        reward_report = f_reward.result()

    rep = _acceptance_from_text(
        run,
        trainer.get("config", ""),
        edge.get("config", ""),
        trainer.get("metrics", ""),
        edge.get("metrics", ""),
        "\n".join((trainer.get("logs", ""), edge.get("logs", ""), *engine_logs)),
        reward_report.verdict,
        trainer_active_run=trainer.get("active_run") or None,
        trainer_start_epoch=_parse_epoch(trainer.get("start_epoch")),
        rollout_active_run=edge.get("active_run") or None,
        rollout_start_epoch=_parse_epoch(edge.get("start_epoch")),
    )
    rep.trainer_config_path = trainer.get("path", "")
    rep.orchestrator_config_path = edge.get("path", "")
    if not quiet:
        print(rep.report())
    return rep


# --------------------------------------------------------------------- reads


def logs(role: str = "trainer", lines: int = 200) -> str:
    """Read the current container tail without a separate SSH command."""
    roles = {
        "trainer": (trainer_head, trainer_container),
        "orchestrator": (orchestrator_host, orchestrator_container),
        **{host: (host, glm_inference_container) for host in inference_hosts},
    }
    if role not in roles:
        raise Refused(f"log role must be one of: {', '.join(roles)}")
    try:
        lines = int(lines)
    except (TypeError, ValueError) as error:
        raise Refused("log line count must be an integer") from error
    if not 1 <= lines <= 5000:
        raise Refused("log line count must be between 1 and 5000")
    host, container = roles[role]
    rc, output = _ssh(
        host,
        f"sudo docker logs --tail {lines} {container} 2>&1 | "
        "sed 's/\\x1b\\[[0-9;]*m//g'",
        timeout=180,
    )
    if rc:
        raise RuntimeError(f"could not read {role} logs on {host}: {output.strip()}")
    print(output, end="" if output.endswith("\n") else "\n")
    return output


def _parity_reader_script(path: str) -> str:
    return """python3 - """ + shlex.quote(path) + r""" <<'PY'
import hashlib,json,math,sys

path=sys.argv[1]
try:
    with open(path,'rb') as stream:
        raw=stream.read()
except OSError as error:
    raise SystemExit(f"invalid parity record: cannot read file: {error}")
try:
    data=json.loads(raw)
except (TypeError,ValueError) as error:
    raise SystemExit(f"invalid parity record: invalid JSON: {error}")
if not isinstance(data,dict):
    raise SystemExit('invalid parity record: top-level JSON must be an object')

def fail(message):
    raise SystemExit(f"invalid parity record: {message}")

lengths=data.get('sequence_lengths')
if (not isinstance(lengths,list) or not lengths or
    any(type(length) is not int or length <= 0 for length in lengths)):
    fail('sequence_lengths must contain positive integers')
record_tokens=sum(lengths)

def array(name):
    values=data.get(name)
    if not isinstance(values,list) or len(values) != record_tokens:
        fail(f"{name} length must equal record token count {record_tokens}")
    return values

positions=array('position_ids')
if any(type(value) is not int for value in positions):
    fail('position_ids must contain integers')
mask=array('loss_mask')
if any(type(value) is not bool for value in mask):
    fail('loss_mask must contain booleans')

def finite_array(name):
    values=array(name)
    if any(
        isinstance(value,bool) or not isinstance(value,(int,float)) or
        not math.isfinite(float(value)) for value in values
    ):
        fail(f'{name} must contain finite numbers')
    return values

delta=finite_array('delta')
trainer=finite_array('trainer_logprobs')
reference=finite_array('reference_logprobs')
if any(
    not math.isclose(actual, predicted - baseline, rel_tol=1e-6, abs_tol=1e-6)
    for actual, predicted, baseline in zip(delta, trainer, reference)
):
    fail('delta must equal trainer_logprobs - reference_logprobs')
valid=[value for value,keep in zip(delta,mask) if keep]
valid_trainer=[value for value,keep in zip(trainer,mask) if keep]
valid_reference=[value for value,keep in zip(reference,mask) if keep]
if not valid:
    fail('loss_mask selects no tokens')
ordered=sorted(abs(value) for value in valid)
try:
    proxy=math.fsum(math.expm1(value)-value for value in valid)/len(valid)
except (OverflowError,ValueError) as error:
    fail(f'K3 proxy overflow: {error}')
if not math.isfinite(proxy):
    fail('K3 proxy is not finite')
expected=[]
for length in lengths:
    expected.extend(range(length))
out={
  'sha256':hashlib.sha256(raw).hexdigest(),
  'record_tokens':record_tokens,
  'sequences':len(lengths),
  'selected_tokens':len(valid),
  'loss_mask_excluded_fraction':1.0-len(valid)/record_tokens,
  'position_mismatches':sum(a!=b for a,b in zip(positions,expected)),
  'mean_abs_delta':sum(abs(value) for value in valid)/len(valid),
  'mean_delta':sum(valid)/len(valid),
  'mean_trainer_logprob':sum(valid_trainer)/len(valid_trainer),
  'mean_reference_logprob':sum(valid_reference)/len(valid_reference),
  'p95_abs_delta':ordered[min(int(0.95*len(ordered)),len(ordered)-1)],
  'max_abs_delta':max(ordered),
  'mismatch_kl_proxy':proxy,
}
print(json.dumps(out,sort_keys=True))
PY
"""


def parity(run: str = "") -> dict:
    """Summarize the newest token parity record on configured trainer hosts."""
    explicit_run = bool(run)
    run = run or current_run()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("run name contains invalid characters")
    profile = _config_for_run(run, required=not explicit_run)
    if profile is None:
        source = "archive"
        find_path = f"{run_root}/{run}"
        find_options = "-maxdepth 4 -type f -name 'parity-step*.json'"
    else:
        source = "configured"
        trainer_path = repo_root / profile["trainer"]
        try:
            with trainer_path.open("rb") as stream:
                trainer_config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError(f"cannot read trainer config {trainer_path}: {error}") from error
        output_dir = trainer_config.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir:
            raise RuntimeError(f"trainer config {trainer_path} has no output_dir")
        data_config = trainer_config.get("data", {})
        fake_config = data_config.get("fake", {}) if isinstance(data_config, dict) else {}
        trainer_only = (
            profile.get("orchestrator") is None
            and isinstance(fake_config, dict)
            and isinstance(fake_config.get("reference_path"), str)
        )
        if trainer_only:
            find_path = output_dir.rstrip("/") + "/direct-parity.json"
            find_options = "-maxdepth 0 -type f"
        else:
            find_path = output_dir
            find_options = "-maxdepth 4 -type f -name 'parity-step*.json'"
    find_script = f"""
candidate=$(find {shlex.quote(find_path)} {find_options} -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1)
if [ -n \"$candidate\" ]; then
  printf 'PARITY %s\\n' \"$candidate\"
else
  printf 'NO_PARITY\\n'
fi
"""
    candidates = []
    unavailable = []
    for host, (code, output) in _fan(
        trainer_hosts, lambda host: _ssh(host, find_script, timeout=180)
    ).items():
        if code:
            unavailable.append(host)
            continue
        match = re.search(r"^PARITY ([0-9]+(?:\.[0-9]+)?) (.+)$", output, re.M)
        if match:
            candidates.append((float(match.group(1)), host, match.group(2).strip()))
    if not candidates:
        detail = f"; unavailable hosts: {', '.join(unavailable)}" if unavailable else ""
        print(f"no parity record found for {run} on configured trainer hosts{detail}")
        return {
            "run": run,
            "available": False,
            "reason": "no parity record found",
            "unavailable_hosts": unavailable,
        }
    _, parity_host, path = max(candidates, key=lambda item: item[0])
    script = _parity_reader_script(path)
    code, output = _ssh(parity_host, script, timeout=180)
    if code:
        detail = output.strip().splitlines()[-1] if output.strip() else "reader failed"
        raise RuntimeError(f"cannot read parity record {path} on {parity_host}: {detail}")
    try:
        summary = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"parity record {path} is invalid") from error
    summary["run"] = run
    summary["available"] = True
    summary["source"] = source
    summary["path"] = path
    summary["host"] = parity_host
    summary["unavailable_hosts"] = unavailable
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def status(run: str = "", quiet: bool = False) -> Status:
    """Probe every part of the run once and return the whole picture."""
    run = run or current_run()
    s = Status(
        when=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), run=run
    )
    trainer_only = _trainer_mode() == "trainer-only"
    policy_probe = None if trainer_only else _active_policy_probe_config(run)
    policy_broadcast_dir = policy_probe[0] if policy_probe is not None else ""

    def node(h):
        return _ssh(h, node_probe.replace("__TRAINER__", trainer_container), timeout=60)

    def engine(h):
        return _ssh(
            h,
            _inference_script(
                h, glm_model, glm_inference_container, policy_broadcast_dir
            ),
            timeout=90,
        )

    with ThreadPoolExecutor(max_workers=12) as ex:
        if trainer_only:
            f_head = ex.submit(
                _ssh,
                trainer_head,
                trainer_only_probe.replace("__TRAINER__", trainer_container).replace(
                    "__TRAINER_COMPUTE__",
                    trainer_compute_probe.replace("__TRAINER__", trainer_container),
                ),
                120,
            )
            f_nodes = ex.submit(_fan, trainer_hosts, node)
            head_rc, head_out = f_head.result()
            nodes_out = f_nodes.result()
            edge_orch_rc, edge_orch_out = 0, ""
            engines_out = {}
        else:
            f_head = ex.submit(_ssh, trainer_head, _head_script(), 300)
            f_edge_orch = ex.submit(
                _ssh,
                orchestrator_host,
                orchestrator_probe.replace("__ORCH__", orchestrator_container).replace(
                    "__JUDGE_URL__", judge_api_url
                ),
                120,
            )
            f_nodes = ex.submit(_fan, trainer_hosts, node)
            f_engines = ex.submit(_fan, inference_hosts, engine)
            head_rc, head_out = f_head.result()
            edge_orch_rc, edge_orch_out = f_edge_orch.result()
            nodes_out = f_nodes.result()
            engines_out = f_engines.result()

    h = _kv(head_out)
    edge_orch = _kv(edge_orch_out) if edge_orch_rc == 0 else {}
    edge_keys = (
        "ORCH_UP",
        "ORCH_FAILED",
        "BATCH_SYNC",
        "TRAINER_FAIL_WD",
        "ORCH_LAST",
        "BATCH",
        "FAILS5M",
        "ORCH_LOGAGE",
        "JUDGE",
    )
    for key in edge_keys:
        h.pop(key, None)
    if edge_orch_rc == 0:
        h.update({key: edge_orch[key] for key in edge_keys if key in edge_orch})
    s.mode = h.get("MODE", "rl")
    head_trainer_up = h.get("TRAINER_UP", "") == "up"
    s.orch_failed = int(h.get("ORCH_FAILED", 0) or 0) > 0
    s.orch_up = h.get("ORCH_UP", "") == "up" and not s.orch_failed
    s.batch_sync = int(h.get("BATCH_SYNC", 0) or 0) > 0
    s.trainer_failure_watchdog = int(h.get("TRAINER_FAIL_WD", 0) or 0) > 0
    s.trainer_steps = int(h.get("TRAINER_STEPS", 0) or 0)
    state = h.get("TRAINER_COMPUTE_STATE", "pending")
    if state in {"pending", "active", "complete", "unknown"}:
        s.trainer_compute_state = state
    compute = _parse_trainer_compute(h.get("TRAINER_COMPUTE", ""))
    if compute is not None:
        s.trainer_compute_step, s.trainer_compute_seconds = compute
        s.trainer_compute_pp_seconds = _parse_seconds(
            h.get("TRAINER_COMPUTE_PP_SECONDS", "")
        )
        s.trainer_compute_active = h.get("TRAINER_COMPUTE_ACTIVE", "") == "1"
        s.trainer_compute_elapsed_seconds = _parse_seconds(
            h.get("TRAINER_COMPUTE_ELAPSED_SECONDS", "")
        )
    s.trainer_last = h.get("TRAINER_LAST", "")
    s.trainer_current = h.get("TRAINER_CURRENT", "")
    s.trainer_finished = int(h.get("TRAINER_FINISHED", 0) or 0) > 0
    s.trainer_failed = int(h.get("TRAINER_FAILED", 0) or 0) > 0
    s.orch_last = h.get("ORCH_LAST", "")
    s.batch = h.get("BATCH", "")
    s.fails_5m = int(h.get("FAILS5M", 0) or 0)
    s.trainer_log_age = _age(h.get("TRAINER_LOGAGE", ""))
    s.orch_log_age = _age(h.get("ORCH_LOGAGE", ""))
    s.judge = h.get("JUDGE", "000")
    s.adapter = h.get("ADAPTER", "000")

    for host, (rc, out) in nodes_out.items():
        n = NodeState(host=host, reachable=(rc == 0))
        for line in out.splitlines():
            p = line.split()
            if p[:1] == ["GPU"] and len(p) >= 4:
                n.gpu_mem_mb, n.gpu_util, n.gpus = int(p[1]), int(p[2]), int(p[3])
            elif p[:1] == ["MEM"] and len(p) >= 3:
                n.mem_free_gb, n.mem_shared_gb = int(p[1]), int(p[2])
            elif p[:1] == ["DISK"] and len(p) >= 3:
                n.data_free_gb, n.data_use_percent = int(p[1]), int(p[2])
            elif p[:1] == ["CT"]:
                n.container = line[3:].strip()
        s.nodes.append(n)

    cluster_trainers_up = len(s.nodes) == len(trainer_hosts) and all(
        n.reachable and n.container.startswith("Up") for n in s.nodes
    )
    s.trainer_up = head_trainer_up and cluster_trainers_up

    for host, (rc, out) in engines_out.items():
        e = EngineState(host=host, reachable=(rc == 0))
        for line in out.splitlines():
            p = line.split()
            if p[:1] == ["GPU"] and len(p) >= 2:
                e.gpu_util = int(p[1])
            elif p[:1] == ["GEN"] and len(p) >= 3:
                e.gen_code, e.gen_ms = p[1], int(p[2])
            elif p[:1] == ["PROBES"] and len(p) >= 3:
                e.probes_ok, e.probes_total = int(p[1]), int(p[2])
            elif p[:1] == ["RUNNING"] and len(p) >= 2:
                e.running = int(p[1])
            elif p[:1] == ["TPUT"] and len(p) >= 2:
                e.throughput = float(p[1])
            elif p[:1] == ["WD"] and len(p) >= 2:
                e.watchdog = int(p[1]) > 0
            elif p[:1] == ["POLICY"] and len(p) >= 2:
                e.policy_state = p[1]
                if len(p) >= 3 and p[2].isdigit():
                    e.policy_step = int(p[2])
        s.engines.append(e)

    s.warnings = _warnings(s)
    if not quiet:
        print(s.report())
    return s


def _warnings(s: Status) -> list[str]:
    w = []
    if (
        s.trainer_compute_seconds is not None
        and s.trainer_compute_seconds > trainer_seconds_max
    ):
        w.append(
            f"trainer compute step {s.trainer_compute_step} exceeds configured upper bound "
            f"{trainer_seconds_max:g}s; operator must stop"
        )
    if (
        s.trainer_compute_active
        and s.trainer_compute_elapsed_seconds is not None
        and s.trainer_compute_elapsed_seconds > trainer_seconds_max
    ):
        w.append(
            f"trainer compute step {s.trainer_compute_step} has been active for "
            f"more than {trainer_seconds_max:g}s; operator must stop"
        )
    if s.trainer_failed:
        w.append(f"trainer process failed after {s.trainer_steps} completed steps")
    elif not s.trainer_up and not s.trainer_finished:
        failed_nodes = [
            n.host
            for n in s.nodes
            if not n.reachable or not n.container.startswith("Up")
        ]
        detail = f" on {', '.join(failed_nodes)}" if failed_nodes else ""
        w.append(f"trainer cluster is not complete{detail}")
    trainer_only = s.mode == "trainer-only"
    if not trainer_only and s.orch_failed:
        w.append("orchestrator process failed")
    elif not trainer_only and not s.orch_up and not s.trainer_finished:
        w.append("orchestrator container is not up")
    inflight = re.search(r"(\d+) inflight", s.batch or "")
    live = [e for e in s.engines if e.reachable]
    if inflight and int(inflight.group(1)) > 40 and live:
        tput = sum(e.throughput for e in live)
        if tput < 5.0 and sum(e.running for e in live) == 0:
            w.append(
                f"{inflight.group(1)} rollouts in flight but the engines generate {tput:.1f} tok/s "
                "and run nothing. Sample again before acting. If it holds, the requests are queued "
                "before inference; lower max_inflight_rollouts, which is independent of batch_size."
            )
    for e in (() if trainer_only else s.engines):
        if e.reachable and e.policy_state == "staging":
            w.append(
                f"{e.host} is staging the active policy adapter; completion health is pending"
            )
        elif e.reachable and not e.healthy:
            if e.watchdog:
                w.append(
                    f"{e.host} failed a real completion probe (code {e.gen_code}), but its watchdog is active. "
                    "Sample again and inspect /root/engine_watchdog.log before a manual repair."
                )
            else:
                w.append(
                    f"{e.host} failed a real completion probe (code {e.gen_code}). Run repair('engine:{e.host}')."
                )
        if e.reachable and not e.watchdog:
            w.append(f"{e.host} has no engine watchdog running")
    if not trainer_only and s.judge != "200":
        w.append(
            f"judge returned {s.judge}; every reward multiplies by the judge score"
        )
    if not trainer_only and s.adapter != "200":
        w.append(
            f"adapter server returned {s.adapter} on port {adapter_port}. "
            "If the engines cannot fetch the LoRA, they keep serving the base model and nothing learns."
        )
    if s.orch_up and not s.batch_sync:
        w.append("durable batch publisher is not running; completed rollout batches cannot reach the trainer")
    if s.orch_up and not s.trainer_failure_watchdog:
        w.append("edge trainer-failure watchdog is not running; rollout production will not fail closed")
    trainer_busy = s.trainer_up and any(
        node.reachable and node.gpu_util >= 50 for node in s.nodes
    )
    ages = (("trainer", s.trainer_log_age),) if trainer_only else (
        ("trainer", s.trainer_log_age),
        ("orchestrator", s.orch_log_age),
    )
    for name, age in (() if s.trainer_finished else ages):
        if age is not None and age > stall_seconds:
            if name == "trainer" and trainer_busy:
                continue
            w.append(
                f"{name} has written nothing for {_hms(age)}; the limit is {stall_seconds // 60} minutes"
            )
    for n in s.nodes:
        if n.reachable and n.mem_free_gb < 60:
            w.append(
                f"{n.host} has only {n.mem_free_gb}G of host memory free. "
                'Check that ac_offloading = "None" is set; the default fills pinned host memory.'
            )
        if n.reachable and n.data_free_gb < 25:
            w.append(
                f"{n.host} has only {n.data_free_gb}G free on /data "
                f"({n.data_use_percent}% used). Do not start a save until space is available."
            )
    return w


def current_run() -> str:
    """The run name the trainer container was started with."""
    rc, out = _ssh(
        trainer_head,
        f"sudo docker inspect {trainer_container} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null | grep -E '^RUN='",
    )
    m = re.search(r"^RUN=(\S+)", out, re.M)
    if m:
        return m.group(1)
    rc, command = _ssh(
        trainer_head,
        f"sudo docker inspect {trainer_container} --format '{{{{range .Config.Cmd}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null",
    )
    m = re.search(rf"^--log-dir={re.escape(run_root)}/([^/]+)/", command, re.M)
    if m:
        return m.group(1)
    detail = command.strip() or out.strip() or f"container {trainer_container} has no run path"
    raise RuntimeError(f"cannot identify the active run on {trainer_head}: {detail}")


def engines(run: str = "") -> dict[str, EngineState]:
    """Probe every configured inference host with a real completion."""
    run = run or current_run()
    policy_probe = _active_policy_probe_config(run)
    policy_broadcast_dir = policy_probe[0] if policy_probe is not None else ""
    states = []
    for host, (rc, out) in _fan(
        inference_hosts,
        lambda h: _ssh(
            h,
            _inference_script(
                h, glm_model, glm_inference_container, policy_broadcast_dir
            ),
            timeout=90,
        ),
    ).items():
        e = EngineState(host=host, reachable=(rc == 0))
        for line in out.splitlines():
            p = line.split()
            if p[:1] == ["GPU"] and len(p) >= 2:
                e.gpu_util = int(p[1])
            elif p[:1] == ["GEN"] and len(p) >= 3:
                e.gen_code, e.gen_ms = p[1], int(p[2])
            elif p[:1] == ["PROBES"] and len(p) >= 3:
                e.probes_ok, e.probes_total = int(p[1]), int(p[2])
            elif p[:1] == ["RUNNING"] and len(p) >= 2:
                e.running = int(p[1])
            elif p[:1] == ["TPUT"] and len(p) >= 2:
                e.throughput = float(p[1])
            elif p[:1] == ["WD"] and len(p) >= 2:
                e.watchdog = int(p[1]) > 0
            elif p[:1] == ["POLICY"] and len(p) >= 2:
                e.policy_state = p[1]
                if len(p) >= 3 and p[2].isdigit():
                    e.policy_step = int(p[2])
        states.append(e)
    for e in states:
        print(
            f"{e.host}: {e.probes_ok}/{e.probes_total} probes in {e.gen_ms} ms, "
            f"running={e.running}, {e.throughput} tok/s, policy={e.policy_state}"
            f"{f' v{e.policy_step}' if e.policy_step is not None else ''}"
        )
    return {e.host: e for e in states}


def alive(pattern: str, host: str = trainer_head, log: str = "") -> bool:
    """Test that a process runs. Never use `pgrep -fc`; it counts the grep.

    A process that runs but writes no new log line does not work, so pass
    `log` when the process has one.
    """
    rc, out = _ssh(
        host,
        f"pgrep -f {json.dumps(_no_self_match(pattern))} | wc -l; "
        + (f"tail -1 {log} 2>/dev/null || true" if log else ""),
    )
    if rc:
        raise RuntimeError(f"cannot check {pattern!r} on {host}: {out.strip()}")
    lines = out.strip().splitlines()
    if not lines or not lines[0].strip().isdigit():
        raise RuntimeError(f"invalid process count for {pattern!r} on {host}")
    n = int(lines[0])
    running = n > 0
    print(f"{host}: {pattern!r} -> {n} process(es)")
    if log and len(lines) > 1:
        age = _age(lines[1].split(" ")[0])
        print(f"  last log line {_hms(age)} old")
        if age is not None and age > 300:
            print("  the process exists but its log is stale; treat it as dead")
            return False
    return running


def run_on_hosts(
    command: str,
    group: str = "trainer",
    timeout: int = 120,
) -> dict[str, tuple[int, str]]:
    """Run one explicit command on one configured host group."""
    groups = {
        "trainer": trainer_hosts,
        "inference": inference_hosts,
        "judge": [hosts["judge"]],
        "all": list(dict.fromkeys([*trainer_hosts, *inference_hosts])),
    }
    if group not in groups:
        raise Refused(f"host group must be one of: {', '.join(groups)}")
    if not command.strip():
        raise Refused("command is empty")
    results = _fan(groups[group], lambda host: _ssh(host, command, timeout))
    for host, (code, output) in results.items():
        state = "ok" if code == 0 else f"error {code}"
        print(f"{host}: {state}")
        if output.strip():
            print(output.strip())
    return results


@dataclass(frozen=True)
class AdapterBroadcast:
    path: str
    step: int


def _validated_run_base(run: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("run name contains invalid characters")
    base = f"{run_root.rstrip('/')}/{run}"
    if not base.startswith("/") or ".." in PurePosixPath(base).parts:
        raise Refused("run path is invalid")
    return base


def _parse_adapter_broadcasts(output: str, run_base: str) -> list[AdapterBroadcast]:
    broadcasts: list[AdapterBroadcast] = []
    prefix = run_base.rstrip("/") + "/"
    for line in output.splitlines():
        path = line.strip()
        if not path:
            continue
        if not path.startswith(prefix):
            raise RuntimeError(f"stable adapter path is outside the run: {path}")
        match = re.fullmatch(r"(.+)/broadcasts/step_(\d+)", path)
        if not match:
            raise RuntimeError(f"invalid stable adapter path: {path}")
        broadcasts.append(AdapterBroadcast(path=path, step=int(match.group(2))))
    unique = {(item.path, item.step): item for item in broadcasts}
    result = list(unique.values())
    roots = {item.path.rsplit("/broadcasts/", 1)[0] for item in result}
    if len(roots) > 1:
        raise RuntimeError("run has more than one adapter broadcast directory")
    return sorted(result, key=lambda item: item.step)


def _adapter_bucket_run_folder(run: str) -> str:
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("adapter archive run name is invalid")
    return f"{run}-run"


def _local_command(command: list[str], label: str, timeout: int) -> subprocess.CompletedProcess:
    try:
        result = _run_process(command, stdin=subprocess.DEVNULL, timeout=timeout)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} timed out") from error
    if result.returncode:
        detail = (result.stderr or result.stdout or "failed").strip().splitlines()[-1]
        raise RuntimeError(f"{label} failed: {detail[:160]}")
    return result


def copy_file(local_path: str | Path, host: str, remote_path: str) -> bool:
    """Copy one large local file to configured storage with resume support."""
    if not isinstance(host, str) or not host:
        raise Refused("copy destination host must be a non-empty string")
    if not isinstance(remote_path, str) or not remote_path:
        raise Refused("copy destination path must be a non-empty string")
    try:
        local = Path(local_path)
    except TypeError as error:
        raise Refused(f"local copy source is not a file: {local_path!r}") from error
    if not local.is_file():
        raise Refused(f"local copy source is not a file: {local}")

    remote = PurePosixPath(remote_path)
    root = PurePosixPath(storage_root.rstrip("/"))
    if (
        not remote.is_absolute()
        or remote.as_posix() != remote_path
        or any(part in {"", ".", ".."} for part in remote.parts)
        or not remote.is_relative_to(root)
        or remote == root
    ):
        raise Refused(
            f"remote copy path must be a canonical file below {storage_root}: {remote_path!r}"
        )

    allowed_hosts = {*trainer_hosts, *inference_hosts, hosts["judge"]}
    if host not in allowed_hosts:
        raise Refused(f"copy destination host is not configured: {host}")
    _check_host(host)
    ssh_options, _, destination = _ssh_destination(host)
    remote_text = remote.as_posix()
    remote_parent = remote.parent.as_posix()
    mkdir_code, mkdir_output = _ssh(
        host,
        f"mkdir -p {shlex.quote(remote_parent)}",
        timeout=deployment_mkdir_timeout_seconds,
    )
    if mkdir_code:
        detail = mkdir_output.strip().splitlines()[-1] if mkdir_output.strip() else "unknown error"
        raise RuntimeError(f"create copy directory on {host} failed: {detail[:160]}")

    try:
        size = local.stat().st_size
    except OSError as error:
        raise RuntimeError(f"cannot stat local copy source {local}: {error}") from error
    transport = shlex.join(ssh_command + ssh_options)
    command = [
        "rsync",
        "--partial",
        "--checksum",
        "--info=progress2,name0",
        "--human-readable",
        "--protect-args",
        "--timeout=120",
        "--rsh",
        transport,
        str(local),
        f"{destination}:{remote_text}",
    ]
    try:
        result = _run_process(
            command,
            stdin=subprocess.DEVNULL,
            timeout=large_file_transfer_timeout_seconds,
        )
    except FileNotFoundError as error:
        raise RuntimeError("rsync is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"copy {local} to {host}:{remote_text} timed out after "
            f"{large_file_transfer_timeout_seconds}s"
        ) from error
    if result.returncode:
        detail = ((result.stderr or "") + (result.stdout or "")).strip()
        detail = detail.splitlines()[-1] if detail else "unknown error"
        raise RuntimeError(f"copy {local} to {host}:{remote_text} failed: {detail[:160]}")

    stat_code, stat_output = _ssh(
        host,
        f"stat -c %s -- {shlex.quote(remote_text)}",
        timeout=deployment_mkdir_timeout_seconds,
    )
    stat_lines = stat_output.strip().splitlines()
    remote_size_text = stat_lines[-1].strip() if stat_lines else ""
    if stat_code or not re.fullmatch(r"[0-9]+", remote_size_text):
        raise RuntimeError(f"cannot verify byte size of {host}:{remote_text}")
    remote_size = int(remote_size_text)
    if remote_size != size:
        raise RuntimeError(
            f"copy size mismatch for {host}:{remote_text}: "
            f"local {size} bytes, remote {remote_size} bytes"
        )
    print(f"copied {size} bytes to {host}:{remote_text}")
    return True


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"cannot read {label}: {error}") from error
    return digest.hexdigest()


def _remote_sha256(path: str) -> str:
    code, output = _ssh(
        trainer_head,
        f"sha256sum -- {shlex.quote(path)} | cut -d' ' -f1",
        timeout=120,
    )
    digest = output.strip().splitlines()[-1] if output.strip() else ""
    if code or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"cannot read checksum for {path}")
    return digest


def _remote_broadcast_files(path: str) -> list[str]:
    code, output = _ssh(
        trainer_head,
        f"find {shlex.quote(path)} -maxdepth 1 -type f -printf '%f\\n' | sort",
        timeout=120,
    )
    if code:
        raise RuntimeError(f"cannot inspect adapter directory {path}")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _confirmed_active_policy_step(run: str) -> int:
    """Return the newest policy step confirmed loaded by the orchestrator."""
    code, output = _ssh(
        orchestrator_host,
        "sudo docker logs --tail 20000 "
        + shlex.quote(orchestrator_container)
        + " 2>&1 | grep -oE 'Updated weights to step [0-9]+' | tail -1",
        timeout=180,
    )
    if code and not output.strip():
        raise RuntimeError("cannot read the orchestrator policy-update log")
    match = re.search(r"Updated weights to step (\d+)", output)
    if match is None:
        raise Refused(f"no active policy update is confirmed for {run}")
    return int(match.group(1))


def _download_adapter_file(remote_path: str, local_path: Path) -> None:
    command = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "ConnectTimeout=25",
        f"{trainer_head}:{remote_path}",
        str(local_path),
    ]
    _local_command(command, f"download {remote_path}", timeout=1800)


def _archive_one_file(local_path: Path, remote_uri: str, expected_digest: str) -> None:
    _local_command(
        ["hf", "buckets", "cp", "--quiet", str(local_path), remote_uri],
        f"upload {remote_uri}",
        timeout=1800,
    )
    with tempfile.NamedTemporaryFile(prefix="adapter-verify-", delete=False) as stream:
        verify_path = Path(stream.name)
    try:
        _local_command(
            ["hf", "buckets", "cp", "--quiet", remote_uri, str(verify_path)],
            f"verify {remote_uri}",
            timeout=1800,
        )
        actual_digest = _sha256_file(verify_path, f"downloaded {remote_uri}")
        if actual_digest != expected_digest:
            raise RuntimeError(f"remote checksum differs for {remote_uri}")
    finally:
        verify_path.unlink(missing_ok=True)


def archive_adapters(
    run: str,
    steps_to_archive: list[int] | None = None,
    finished: bool = False,
) -> list[int]:
    """Upload stable old LoRA adapters, then remove verified remote copies."""
    if not isinstance(adapter_archive_bucket, str) or not adapter_archive_bucket:
        raise Refused("adapter archive is not configured")
    run_base = _validated_run_base(run)
    bucket_run_folder = _adapter_bucket_run_folder(run)
    if steps_to_archive is not None:
        if not steps_to_archive or any(
            not isinstance(step, int) or step < 0 for step in steps_to_archive
        ):
            raise Refused("adapter steps must be non-negative integers")
        requested_steps = set(steps_to_archive)
    else:
        requested_steps = None

    _local_command(["hf", "auth", "whoami"], "Hugging Face authentication", timeout=30)
    code, output = _ssh(
        trainer_head,
        "test -d " + shlex.quote(run_base)
        + " && find "
        + shlex.quote(run_base)
        + " -type f -path '*/broadcasts/step_*/STABLE' -printf '%h\\n' | sort -u",
        timeout=120,
    )
    if code:
        raise RuntimeError(f"run directory does not exist on {trainer_head}: {run_base}")
    broadcasts = _parse_adapter_broadcasts(output, run_base)
    if not broadcasts:
        print(f"{run}: no stable adapters")
        return []

    expected_files = sorted(("STABLE", *adapter_archive_files))
    available_broadcasts: list[AdapterBroadcast] = []
    for item in broadcasts:
        actual_files = _remote_broadcast_files(item.path)
        if actual_files == ["STABLE"]:
            continue
        if actual_files != expected_files:
            raise RuntimeError(
                f"{item.path} has unexpected files; expected {', '.join(expected_files)}"
            )
        available_broadcasts.append(item)

    if not available_broadcasts:
        print(f"{run}: no local adapters")
        return []

    latest_step = max(item.step for item in available_broadcasts)
    if finished:
        active_run = current_run()
        if active_run == run:
            raise Refused(
                "--finished is only allowed after a different run is active"
            )
    else:
        if requested_steps is not None and latest_step in requested_steps:
            raise Refused(f"step-{latest_step} is the latest live adapter and cannot be removed")
        active_step = _confirmed_active_policy_step(run)
        if active_step != latest_step:
            raise Refused(
                f"latest stable adapter step-{latest_step} is not the confirmed "
                f"active policy step (step-{active_step})"
            )
    selected = [
        item
        for item in available_broadcasts
        if (finished or item.step != latest_step)
        and (requested_steps is None or item.step in requested_steps)
    ]
    missing = (
        sorted(requested_steps - {item.step for item in broadcasts})
        if requested_steps is not None
        else []
    )
    if missing:
        raise RuntimeError(f"requested adapter steps are missing: {missing}")
    if not selected:
        if not finished:
            print(f"{run}: kept latest live adapter step-{latest_step}")
        return []

    archived: list[int] = []
    with tempfile.TemporaryDirectory(prefix=f"{run}-adapter-") as staging:
        stage_dir = Path(staging)
        for item in selected:
            remote_dir = item.path
            for filename in adapter_archive_files:
                remote_path = f"{remote_dir}/{filename}"
                local_path = stage_dir / f"step-{item.step}-{filename}"
                expected_digest = _remote_sha256(remote_path)
                _download_adapter_file(remote_path, local_path)
                actual_digest = _sha256_file(local_path, f"downloaded {remote_path}")
                if actual_digest != expected_digest:
                    raise RuntimeError(f"download checksum differs for {remote_path}")
                remote_uri = (
                    f"{adapter_archive_bucket}/{bucket_run_folder}/step-{item.step}/{filename}"
                )
                _archive_one_file(local_path, remote_uri, expected_digest)

            if _remote_broadcast_files(remote_dir) != expected_files:
                raise RuntimeError(f"adapter directory changed during archive: {remote_dir}")
            code, output = _ssh(
                trainer_head,
                "set -e; "+ "; ".join(
                    f"rm -f -- {shlex.quote(f'{remote_dir}/{filename}')}"
                    for filename in (*adapter_archive_files, "STABLE")
                )
                + f"; rmdir -- {shlex.quote(remote_dir)}",
                timeout=120,
            )
            if code:
                raise RuntimeError(
                    f"upload verified but source cleanup failed for {remote_dir}: {output.strip()[:160]}"
                )
            archived.append(item.step)
            print(f"archived {run} step-{item.step}")
    if not finished:
        print(f"{run}: kept latest live adapter step-{latest_step}")
    return archived


def steps(run: str = "", limit: int = 10) -> list[dict]:
    """Show recent trainer step times from the metric file."""
    run = run or current_run()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("run name contains invalid characters")
    if not 1 <= int(limit) <= 200:
        raise Refused("step limit must be between 1 and 200")
    profile = _config_for_run(run, required=False)
    source = "archive"
    configured_path = None
    lookup_path = f"{run_root}/{run}"
    if profile is not None:
        trainer_path = repo_root / profile["trainer"]
        try:
            with trainer_path.open("rb") as stream:
                trainer_config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError(f"cannot read trainer config {trainer_path}: {error}") from error
        output_dir = trainer_config.get("output_dir")
        file_monitor = trainer_config.get("file_monitor")
        filename = file_monitor.get("filename") if isinstance(file_monitor, dict) else None
        if not isinstance(output_dir, str) or not output_dir:
            raise RuntimeError(f"trainer config {trainer_path} has no output_dir")
        if not isinstance(filename, str) or not filename:
            raise RuntimeError(f"trainer config {trainer_path} has no file_monitor.filename")
        configured_path = str(Path(output_dir) / filename)
        lookup_path = configured_path
        source = "configured"
    container = shlex.quote(trainer_container)
    run_prefix = shlex.quote(f"{run_root}/")
    if configured_path is None:
        path_setup = f"""
path=$(find {shlex.quote(lookup_path)} -maxdepth 3 -type f -name trainer-metrics.jsonl -print 2>/dev/null | head -1)
if [ -z "$path" ]; then
  printf 'missing trainer metrics under %s\\n' {shlex.quote(lookup_path)} >&2
  exit 4
fi
"""
    else:
        path_setup = f"""
path={shlex.quote(configured_path)}
if [ ! -f "$path" ]; then
  printf 'missing trainer metrics: %s\\n' "$path" >&2
  exit 4
fi
"""
    script = f"""
{path_setup}
container_env=$(sudo docker inspect {container} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null || true)
active_run=$(printf '%s\\n' "$container_env" | sed -n 's/^RUN=//p' | head -1)
if [ -z "$active_run" ]; then
  container_cmd=$(sudo docker inspect {container} --format '{{{{range .Config.Cmd}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null || true)
  active_run=$(printf '%s\\n' "$container_cmd" | awk -v root={run_prefix} '$0 ~ "^--log-dir=" root {{ value=substr($0, 11 + length(root)); sub("/.*$", "", value); print value; exit }}')
fi
started_at=$(sudo docker inspect {container} --format '{{{{.State.StartedAt}}}}' 2>/dev/null | head -1)
start_epoch=
if [ -n "$started_at" ]; then start_epoch=$(date -u -d "$started_at" +%s.%N 2>/dev/null || true); fi
python3 - "$path" {int(limit)} {shlex.quote(run)} "$active_run" "$start_epoch" <<'PY'
import json,sys
rows_by_step={{}}
invalid=0
excluded=0
requested_run=sys.argv[3]
active_run=sys.argv[4] or None
try: active_start=float(sys.argv[5]) if sys.argv[5] else None
except ValueError: active_start=None
filter_rows=active_run == requested_run and active_start is not None
for line in open(sys.argv[1]):
    try:
        row=json.loads(line)
    except ValueError:
        invalid+=1
        continue
    if not (isinstance(row.get('step'), (int,float)) and not isinstance(row.get('step'), bool)):
        continue
    step=int(row['step'])
    if filter_rows:
        timestamp=row.get('time')
        if not isinstance(timestamp, (int,float)) or isinstance(timestamp, bool):
            excluded+=1
            continue
        try: timestamp=float(timestamp)
        except (TypeError,ValueError):
            excluded+=1
            continue
        if timestamp < active_start:
            excluded+=1
            continue
    item=rows_by_step.setdefault(step, {{'step':step}})
    compute=row.get('time/forward_backward')
    if isinstance(compute, (int,float)) and not isinstance(compute, bool):
        item['forward_backward_seconds']=float(compute)
    wait=row.get('time/wait_for_batch')
    if isinstance(wait, (int,float)) and not isinstance(wait, bool):
        item['wait_seconds']=float(wait)
    throughput=row.get('perf/throughput')
    if isinstance(throughput, (int,float)) and not isinstance(throughput, bool):
        item['tokens_per_second']=float(throughput)
rows=sorted(
    (row for row in rows_by_step.values() if 'forward_backward_seconds' in row),
    key=lambda row: row['step'],
)
for row in rows:
    row.setdefault('wait_seconds', 0.0)
    row.setdefault('tokens_per_second', 0.0)
print(json.dumps({{'metrics_path':sys.argv[1], 'rows':rows[-int(sys.argv[2]):], 'invalid_lines':invalid, 'excluded_lines':excluded}}))
PY
"""
    code, output = _ssh(trainer_head, script, timeout=180)
    if code:
        detail = output.strip()
        raise RuntimeError(
            f"cannot read trainer steps on {trainer_head}: source={source} "
            f"path={lookup_path} exit={code}"
            + (f": {detail}" if detail else "")
        )
    try:
        result = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("trainer step output is invalid") from error
    if not isinstance(result, dict):
        raise RuntimeError("trainer step output is not an object")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("trainer step output has no row list")
    metrics_path = result.get("metrics_path")
    if not isinstance(metrics_path, str) or not metrics_path:
        raise RuntimeError("trainer step output has no metrics path")
    print(f"step source={source} path={metrics_path}")
    print("step  forward/backward  wait  tokens/s")
    for row in rows:
        print(
            f"{row['step']:>4}  {row['forward_backward_seconds']:>16.1f}s  "
            f"{row['wait_seconds']:>8.1f}s  {row['tokens_per_second']:.1f}"
        )
    invalid = int(result.get("invalid_lines") or 0)
    if invalid:
        print(f"invalid metric lines: {invalid}")
    excluded = int(result.get("excluded_lines") or 0)
    if excluded:
        print(f"older metric lines excluded: {excluded}")
    return rows


def package_versions(*names: str) -> dict[str, dict[str, str]]:
    """Show versions from each configured live trainer and inference container."""
    packages = names or (
        "transformers",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
        "verifiers",
    )
    if any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        for name in packages
    ):
        raise Refused("package names must contain only letters, numbers, '.', '_' or '-'")
    package_args = " ".join(shlex.quote(name) for name in packages)
    probe = (
        "import importlib.metadata as m,json,sys; out={}; "
        "\nfor n in sys.argv[1:]:\n"
        " try: out[n]=m.version(n)\n"
        " except m.PackageNotFoundError: out[n]=None\n"
        " except Exception as e: out[n]='ERROR:'+type(e).__name__\n"
        "print(json.dumps(out,sort_keys=True))"
    )
    remote_code = shlex.quote(probe)
    targets = {}
    for host in trainer_hosts:
        targets.setdefault(host, []).append(trainer_container)
    for host in inference_hosts:
        targets.setdefault(host, []).append(glm_inference_container)
    reports: dict[str, dict[str, str]] = {}

    def read_host(host: str) -> tuple[int, str]:
        target_names = " ".join(shlex.quote(container) for container in targets[host])
        script = f"""
for container in {target_names}; do
  if ! sudo docker ps --format '{{{{.Names}}}}' | grep -Fxq \"$container\"; then
    continue
  fi
  python_path=$(sudo docker exec \"$container\" sh -lc 'if [ -x /app/.venv/bin/python ]; then printf /app/.venv/bin/python; else command -v python || command -v python3; fi' 2>/dev/null || true)
  if [ -z \"$python_path\" ]; then
    printf 'PACKAGE_ERROR %s no-python\\n' \"$container\"
    continue
  fi
  result=$(sudo docker exec \"$container\" \"$python_path\" -c {remote_code} {package_args} 2>&1) || {{
    printf 'PACKAGE_ERROR %s %s\\n' \"$container\" \"$result\"
    continue
  }}
  printf 'PACKAGE_JSON %s %s\\n' \"$container\" \"$python_path\"
  printf '%s\\n' \"$result\"
done
"""
        return _ssh(host, script, timeout=180)

    for host, (code, output) in _fan(targets, read_host).items():
        if code:
            reports[f"{host}/error"] = {"error": output.strip() or f"exit {code}"}
            continue
        lines = output.splitlines()
        container = None
        python_path = None
        found = False
        for line in lines:
            match = re.fullmatch(r"PACKAGE_JSON (\S+) (\S+)", line)
            if match:
                container, python_path = match.groups()
                continue
            error_match = re.fullmatch(r"PACKAGE_ERROR (\S+) (.*)", line)
            if error_match:
                reports[f"{host}/{error_match.group(1)}"] = {
                    "error": error_match.group(2) or "package query failed"
                }
                found = True
                container = None
                python_path = None
                continue
            if container and python_path and line.startswith("{"):
                try:
                    values = json.loads(line)
                except ValueError:
                    reports[f"{host}/{container}"] = {"error": "invalid package report"}
                else:
                    reports[f"{host}/{container}"] = {
                        **{name: ("MISSING" if value is None else str(value)) for name, value in values.items()},
                        "python": python_path,
                    }
                found = True
                container = None
                python_path = None
        if not found:
            reports[f"{host}/none"] = {"status": "no configured target container is running"}
    print("package versions (live containers):")
    for target, values in sorted(reports.items()):
        details = ", ".join(f"{name}={value}" for name, value in values.items())
        print(f"{target}: {details}")
    return reports


# ------------------------------------------------------------ reward reading


@dataclass
class RewardReport:
    run: str = ""
    steps: list[int] = field(default_factory=list)
    effective: list[float] = field(default_factory=list)
    all_scope: list[float] = field(default_factory=list)
    p10: list[float] = field(default_factory=list)
    p90: list[float] = field(default_factory=list)
    error_rate: list[float] = field(default_factory=list)
    step_seconds: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    slope: float = 0.0
    total_change: float = 0.0
    residual_std: float = 0.0
    verdict: str = "too few steps"
    note: str = ""
    source: str = ""

    def report(self) -> str:
        L = [f"reward for {self.run} ({len(self.steps)} steps)"]
        L.append(
            f"{'step':>5} {'effective':>10} {'all':>8} {'p10':>8} {'p90':>8} {'error':>7} {'time':>8}"
        )
        for i, st in enumerate(self.steps):
            L.append(
                f"{st:>5} {self.effective[i]:10.4f} {self.all_scope[i]:8.4f} "
                f"{self.p10[i]:8.4f} {self.p90[i]:8.4f} "
                f"{self.error_rate[i] * 100:6.1f}% {self.step_seconds[i] / 60:7.1f}m"
            )
        L.append("")
        L.append(
            f"effective reward: mean {self.mean:.4f}, standard deviation {self.std:.4f} across steps"
        )
        L.append(
            f"fit: {self.total_change:+.4f} over the window, residual spread {self.residual_std:.4f}, "
            f"threshold {trend_spread_multiplier:g} x residual = {trend_spread_multiplier * self.residual_std:.4f}"
        )
        L.append(f"result: {self.verdict}")
        if self.source:
            L.append(self.source)
        if self.note:
            L.append(self.note)
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


def reward(run: str = "", quiet: bool = False, host: str = "") -> RewardReport:
    """Read the reward series and say whether it moved or only wandered.

    The verdict is deliberately conservative. A window shorter than
    minimum_trend_steps has no verdict, and a change smaller than twice the
    residual spread is called noise. Report the verdict as it stands.
    """
    run = run or current_run()
    metric_host = _check_host(host or metric_hosts[0])
    if metric_host not in metric_hosts:
        raise Refused(
            f"metric host {metric_host!r} is not configured; choose from {', '.join(metric_hosts)}"
        )
    script = (
        _run_dir(run)
        + '\n[ -z "$d" ] && { echo NORUN; exit 0; }\n'
        + "container_env=$(sudo docker inspect "
        + shlex.quote(orchestrator_container)
        + " --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true)\n"
        + "active_run=$(printf '%s\\n' \"$container_env\" | sed -n 's/^RUN=//p' | head -1)\n"
        + "active_run_source=RUN\n"
        + "if [ -z \"$active_run\" ]; then\n"
        + "  active_run=$(printf '%s\\n' \"$container_env\" | sed -n 's#^HF_HOME=/data/glm52-rl/runs/\\([^/]*\\)/.*#\\1#p' | head -1)\n"
        + "  active_run_source=HF_HOME\n"
        + "fi\n"
        + "started_at=$(sudo docker inspect "
        + shlex.quote(orchestrator_container)
        + " --format '{{.State.StartedAt}}' 2>/dev/null | head -1)\n"
        + "start_epoch=\n"
        + "if [ -n \"$started_at\" ]; then start_epoch=$(date -u -d \"$started_at\" +%s.%N 2>/dev/null || true); fi\n"
        + "python3 - \"$d\" \"$active_run\" \"$start_epoch\" \"$active_run_source\" <<'PY'\n"
        "import json,sys\n"
        "active_run=sys.argv[2] or None\n"
        "try: active_start=float(sys.argv[3]) if sys.argv[3] else None\n"
        "except ValueError: active_start=None\n"
        "active_source=sys.argv[4] or None\n"
        "print('REWARD_META '+json.dumps({'active_run':active_run,'active_start_epoch':active_start,'active_run_source':active_source}))\n"
        "K={'eff':'train/agg/effective/reward/mean','all':'train/agg/all/reward/mean',\n"
        "   'p10':'train/agg/effective/reward/p10','p90':'train/agg/effective/reward/p90',\n"
        "   'err':'train/agg/all/has_error/mean','t':'time/step'}\n"
        "out=[]; invalid=0\n"
        "for line in open(sys.argv[1].rstrip('/')+'/rollout-metrics.jsonl'):\n"
        "    line=line.strip()\n"
        "    if not line: continue\n"
        "    try: r=json.loads(line)\n"
        "    except ValueError: invalid+=1; continue\n"
        "    if 'step' not in r: invalid+=1; continue\n"
        "    out.append({'step':r['step'],'time':r.get('time'),**{k:r.get(v) for k,v in K.items()}})\n"
        "print('REWARD_INVALID '+str(invalid))\n"
        "print('REWARD_JSON '+json.dumps(out))\n"
        "PY\n"
    )
    code, output = _ssh(metric_host, script, timeout=180)
    if code:
        raise RuntimeError(f"cannot read reward data on {metric_host}: {output.strip()}")
    rep = RewardReport(run=run)
    rows = None
    reward_meta: dict[str, object] = {}
    invalid_lines = 0
    for line in output.splitlines():
        if line.startswith("REWARD_INVALID "):
            try:
                invalid_lines = int(line.removeprefix("REWARD_INVALID "))
            except ValueError as error:
                raise RuntimeError("reward invalid-line count is not an integer") from error
        if line.startswith("REWARD_META "):
            try:
                reward_meta = json.loads(line.removeprefix("REWARD_META "))
            except (TypeError, ValueError) as error:
                raise RuntimeError("reward metadata is not valid JSON") from error
        if line.startswith("REWARD_JSON "):
            try:
                rows = json.loads(line.removeprefix("REWARD_JSON "))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"reward data on {metric_host} is not valid JSON"
                ) from error
    if invalid_lines:
        raise RuntimeError(
            f"reward data on {metric_host} has {invalid_lines} invalid line(s)"
        )
    if rows is None:
        rep.verdict = "no data"
        rep.note = (
            f"no rollout-metrics.jsonl found under {run_root}/{run}/ on {metric_host}; "
            "choose another configured metric host explicitly if needed"
        )
        if not quiet:
            print(rep.report())
        return rep
    if not isinstance(rows, list):
        raise RuntimeError(f"reward data on {metric_host} is not a list")

    rows = [r for r in rows if isinstance(r, dict) and r.get("eff") is not None]
    active_run = reward_meta.get("active_run")
    active_start_epoch = reward_meta.get("active_start_epoch")
    if not isinstance(active_run, str):
        active_run = None
    if not isinstance(active_start_epoch, (int, float)) or isinstance(
        active_start_epoch, bool
    ):
        active_start_epoch = None
    active_run_source = reward_meta.get("active_run_source")
    if not isinstance(active_run_source, str) or not active_run_source:
        active_run_source = "unknown"
    rows, excluded_rows, untimestamped_rows = _select_active_reward_rows(
        rows, run, active_run, float(active_start_epoch)
        if active_start_epoch is not None
        else None,
    )
    if active_run == run and active_start_epoch is not None:
        rep.source = (
            f"source: active orchestrator run {active_run} from {active_run_source} "
            f"started at Unix "
            f"{float(active_start_epoch):.3f}; selected {len(rows)} current rows, "
            f"excluded {excluded_rows} older rows"
        )
        if untimestamped_rows:
            rep.source += f"; skipped {untimestamped_rows} rows without valid timestamps"
    elif active_run:
        rep.source = (
            f"source: active orchestrator is {active_run}; no active-run filter "
            f"was applied to requested run {run}"
        )
    else:
        rep.source = "source: no active orchestrator run boundary was available; all rows used"
    rows.sort(key=lambda r: r["step"])
    rep.steps = [r["step"] for r in rows]
    rep.effective = [float(r["eff"]) for r in rows]
    rep.all_scope = [float(r.get("all") or 0.0) for r in rows]
    rep.p10 = [float(r.get("p10") or 0.0) for r in rows]
    rep.p90 = [float(r.get("p90") or 0.0) for r in rows]
    rep.error_rate = [float(r.get("err") or 0.0) for r in rows]
    rep.step_seconds = [float(r.get("t") or 0.0) for r in rows]

    y = rep.effective
    n = len(y)
    if n:
        rep.mean = statistics.fmean(y)
        rep.std = statistics.stdev(y) if n > 1 else 0.0
    if n >= 3:
        xs = list(range(n))
        xbar, ybar = statistics.fmean(xs), rep.mean
        denom = sum((x - xbar) ** 2 for x in xs)
        rep.slope = (
            sum((x - xbar) * (v - ybar) for x, v in zip(xs, y)) / denom
            if denom
            else 0.0
        )
        rep.total_change = rep.slope * (n - 1)
        resid = [v - (ybar + rep.slope * (x - xbar)) for x, v in zip(xs, y)]
        rep.residual_std = statistics.stdev(resid) if n > 2 else 0.0

    if n < minimum_trend_steps:
        rep.verdict = f"too few steps ({n} of {minimum_trend_steps} needed)"
    elif abs(rep.total_change) < trend_spread_multiplier * rep.residual_std:
        rep.verdict = "inside the noise"
    else:
        rep.verdict = "rising" if rep.total_change > 0 else "falling"

    rep.note = _reward_note(rep)
    if not quiet:
        print(rep.report())
    return rep


def _reward_note(r: RewardReport) -> str:
    notes = []
    if len(r.steps) >= 2:
        d_err = r.error_rate[0] - r.error_rate[-1]
        d_all = r.all_scope[-1] - r.all_scope[0]
        d_eff = r.effective[-1] - r.effective[0]
        if d_err > 0.05 and d_all > d_eff:
            notes.append(
                f"The error rate fell {d_err * 100:.1f} points over this window. A failed rollout scores zero, "
                f"so the `all` series rose {d_all:+.4f} while `effective` rose only {d_eff:+.4f}. "
                "The difference is transport recovery, not learning. Quote `effective`."
            )
        if r.p90[-1] > r.p90[0] * 1.5 and r.p10[-1] < 0.01:
            notes.append(
                f"The 90th percentile moved {r.p90[0]:.4f} -> {r.p90[-1]:.4f} while the 10th percentile stayed at "
                f"{r.p10[-1]:.4f}. The reward is a product of five factors, so the bottom stays pinned at zero. "
                "The distribution is widening at the top even when the mean is flat."
            )
    return "\n".join(notes)


# ----------------------------------------------------------- active watching


def watch(hours: float = 2.0, every: int = 300, run: str = "") -> list[Status]:
    """Print status in the foreground until the requested time ends."""
    if every > 600:
        raise Refused("watch interval cannot exceed 600 seconds")
    if every < 10:
        raise Refused("watch interval must be at least 10 seconds")
    deadline = time.time() + hours * 3600
    history: list[Status] = []
    last_change = time.time()
    last_key = None
    while time.time() < deadline:
        try:
            s = status(run=run, quiet=True)
        except RuntimeError as error:
            print(f"status check failed: {error}", flush=True)
            remaining = min(every, max(0, deadline - time.time()))
            if remaining <= 0:
                break
            print(f"--- sleeping {int(remaining)}s ---", flush=True)
            time.sleep(remaining)
            continue
        history.append(s)
        print(s.report(), flush=True)
        key = (
            s.trainer_steps,
            s.batch,
            s.trainer_compute_step,
            s.trainer_compute_seconds,
            s.trainer_compute_active,
        )
        if key != last_key:
            last_key, last_change = key, time.time()
        else:
            still = time.time() - last_change
            if still > stall_seconds:
                trainer_busy = s.trainer_up and any(
                    node.reachable and node.gpu_util >= 50 for node in s.nodes
                )
                if trainer_busy:
                    print(
                        f"-- trainer GPUs remain busy after {_hms(still)} without a counter change; "
                        "GPU use is high, but no step or batch progress is confirmed. "
                        "Check the compute timer; operator must stop if it exceeds the configured limit.",
                        flush=True,
                    )
                else:
                    print(
                        f"no step or batch counter changed for {_hms(still)}; "
                        "check adapter transfer progress and logs before deciding it is stalled or restarting.",
                        flush=True,
                    )
        remaining = min(every, max(0, deadline - time.time()))
        if remaining <= 0:
            break
        print(f"--- sleeping {int(remaining)}s ---", flush=True)
        time.sleep(remaining)
    return history


# -------------------------------------------------------------------- writes


def _microbatch_sync_config(run: str) -> tuple[str, list[str], str]:
    """Resolve the current trainer output and exact peer addresses."""
    selected = _config_for_run(run)
    trainer_path = repo_root / selected["trainer"]
    try:
        with trainer_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read current trainer config {trainer_path}: {error}") from error
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.startswith(f"{run_root}/"):
        raise Refused(
            f"current trainer output_dir {output_dir!r} is outside configured runs {run_root!r}"
        )
    if Path(output_dir).name != run:
        raise Refused(
            f"current trainer output_dir {output_dir!r} does not belong to current run {run!r}"
        )
    if any(part in {"", ".", ".."} for part in Path(output_dir).parts):
        raise Refused(f"current trainer output_dir is unsafe: {output_dir!r}")
    peers = [host for host in trainer_hosts if host != trainer_head and host not in blocked_hosts]
    if not peers:
        raise Refused("no configured trainer peers are available for microbatch sync")
    peer_addresses = [_ssh_address(host) for host in peers]
    return output_dir, peer_addresses, f"{runtime_root}/china/sync_microbatches.sh"


def _microbatch_sync_repair_script(
    run: str, source_dir: str, peer_addresses: list[str], sync_script: str
) -> str:
    """Build a bounded repair and post-launch verification on trainer head."""
    expected_nodes = " ".join(peer_addresses)
    log_file = f"{storage_root}/microbatch-sync-{run}.log"
    return f"""set -euo pipefail
run_dir={shlex.quote(source_dir)}
sync_script={shlex.quote(sync_script)}
expected_nodes={shlex.quote(expected_nodes)}
log_file={shlex.quote(log_file)}
pid_file="$run_dir/.microbatch-sync.pid"
marker_file="$run_dir/.microbatch-sync.marker"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

test -x "$sync_script" || {{ echo "microbatch sync refused: deployed script is missing or not executable" >&2; exit 20; }}
source=$(find "$run_dir/rollouts" -type f -name 'rank_0.bin' -print 2>/dev/null | sort -V | tail -1)
[ -n "$source" ] || {{ echo "microbatch sync refused: source rank_0 is missing under $run_dir/rollouts" >&2; exit 21; }}
source_generation=${{source#"$run_dir/"}}
source_generation=${{source_generation%/*}}
source_count=$(find "$(dirname "$source")" -maxdepth 1 -type f -name 'rank_*.bin' -printf '%f\\n' 2>/dev/null | wc -l)
source_digest=$(sha256sum "$source" | cut -d' ' -f1)
expected_marker="generation=$source_generation count=$source_count sha256=$source_digest"

sync_process_matches() {{
  local pid=$1 command_line
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ "$pid" -gt 1 ] || return 1
  [ -r "/proc/$pid/environ" ] && [ -r "/proc/$pid/cmdline" ] || return 1
  cat "/proc/$pid/environ" 2>/dev/null | tr '\\0' '\\n' | grep -Fxq "run_dir=$run_dir" || return 1
  command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\\0' ' ') || return 1
  case " $command_line " in *" $sync_script "*) return 0;; esac
  return 1
}}

sync_process_healthy() {{
  local pid=$1
  sync_process_matches "$pid" || return 1
  cat "/proc/$pid/environ" 2>/dev/null | tr '\\0' '\\n' | grep -Fxq "nodes=$expected_nodes"
}}

stop_sync_process() {{
  local pid=$1 pgid
  sync_process_matches "$pid" || return 0
  pgid=$(awk '{{print $5}}' "/proc/$pid/stat" 2>/dev/null || true)
  # The deployed launcher uses setsid, therefore the exact process is the
  # process-group leader. Do not signal a broad or inherited process group.
  if [ "$pgid" = "$pid" ]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      sync_process_matches "$pid" || break
      sleep 0.1
    done
    if sync_process_matches "$pid"; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  fi
  for _ in $(seq 1 20); do
    sync_process_matches "$pid" || break
    sleep 0.1
  done
  if sync_process_matches "$pid"; then
    echo "microbatch sync terminal error: process group $pgid did not exit" >&2
    return 1
  fi
  rm -f -- "$run_dir/.microbatch-sync.pid" "$run_dir/.microbatch-sync.lock"
}}

healthy=""
for proc_dir in /proc/[0-9]*; do
  pid=${{proc_dir##*/}}
  if sync_process_healthy "$pid"; then healthy="$pid"; break; fi
done
[ -z "$healthy" ] || {{ echo "microbatch sync refused: healthy current distributor pid=$healthy has resolved targets" >&2; exit 22; }}

peer_mismatch=0
for target in $expected_nodes; do
  state=$(ssh -n "${{ssh_opts[@]}}" "$target" \\
    "if [ -f '$source' ]; then sha256sum '$source' | cut -d' ' -f1; else echo MISSING; fi; if [ -f '$marker_file' ]; then cat '$marker_file'; else echo MISSING_MARKER; fi" 2>/dev/null || true)
  peer_digest=$(printf '%s\\n' "$state" | sed -n '1p')
  peer_marker=$(printf '%s\\n' "$state" | sed -n '2p')
  if [ "$peer_digest" != "$source_digest" ] || [ "$peer_marker" != "$expected_marker" ]; then
    peer_mismatch=1
  fi
done
[ "$peer_mismatch" = 1 ] || {{ echo "microbatch sync refused: peers already match source hash and generation/count marker" >&2; exit 23; }}

# Remove an old distributor only after exact /proc identity checks. This also
# handles a missing PID file; no name-based kill is used.
for proc_dir in /proc/[0-9]*; do
  pid=${{proc_dir##*/}}
  if sync_process_matches "$pid" && ! sync_process_healthy "$pid"; then
    stop_sync_process "$pid" || exit 24
  fi
done
mkdir -p "$run_dir"
nohup setsid env run_dir="$run_dir" nodes="$expected_nodes" interval=1 \\
  "$sync_script" >>"$log_file" 2>&1 </dev/null &
pid=$!
printf '%s\\n' "$pid" >"$pid_file"
matched=0
for _ in $(seq 1 50); do
  if sync_process_matches "$pid"; then matched=1; break; fi
  sleep 0.1
done
[ "$matched" = 1 ] || {{ echo "microbatch sync terminal error: launched process identity was not observed" >&2; exit 25; }}

for _ in $(seq 1 120); do
  source=$(find "$run_dir/rollouts" -type f -name 'rank_0.bin' -print 2>/dev/null | sort -V | tail -1)
  if [ -n "$source" ]; then
    source_generation=${{source#"$run_dir/"}}
    source_generation=${{source_generation%/*}}
    source_count=$(find "$(dirname "$source")" -maxdepth 1 -type f -name 'rank_*.bin' -printf '%f\\n' 2>/dev/null | wc -l)
    source_digest=$(sha256sum "$source" | cut -d' ' -f1)
    expected_marker="generation=$source_generation count=$source_count sha256=$source_digest"
    good=1
    for target in $expected_nodes; do
      state=$(ssh -n "${{ssh_opts[@]}}" "$target" \\
        "if [ -f '$source' ]; then sha256sum '$source' | cut -d' ' -f1; else echo MISSING; fi; if [ -f '$marker_file' ]; then cat '$marker_file'; else echo MISSING_MARKER; fi" 2>/dev/null || true)
      peer_digest=$(printf '%s\\n' "$state" | sed -n '1p')
      peer_marker=$(printf '%s\\n' "$state" | sed -n '2p')
      [ "$peer_digest" = "$source_digest" ] && [ "$peer_marker" = "$expected_marker" ] || good=0
    done
    if [ "$good" = 1 ] && [ -f "$marker_file" ] && [ "$(cat "$marker_file")" = "$expected_marker" ]; then
      echo "microbatch sync repaired pid=$pid generation=$source_generation count=$source_count sha256=$source_digest peers=$expected_nodes"
      exit 0
    fi
  fi
  sleep 0.5
done
echo "microbatch sync terminal error: peers did not reach source hash and generation/count marker" >&2
stop_sync_process "$pid" || true
exit 26
"""


def _adapter_update_config(run: str) -> dict:
    """Resolve adapter update paths from the selected profile."""
    selected = _config_for_run(run)
    orchestrator_path = selected.get("orchestrator")
    if not isinstance(orchestrator_path, str):
        raise Refused(f"profile for {run} has no orchestrator config")
    path = repo_root / orchestrator_path
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read orchestrator config {path}: {error}") from error

    run_base = _validated_run_base(run)
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.startswith(run_base + "/"):
        raise Refused(
            f"orchestrator output_dir {output_dir!r} is outside configured run {run!r}"
        )
    if any(part in {"", ".", ".."} for part in PurePosixPath(output_dir).parts):
        raise Refused(f"orchestrator output_dir is unsafe: {output_dir!r}")
    service_name = PurePosixPath(output_dir).name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", service_name):
        raise Refused(f"orchestrator run directory is invalid: {service_name!r}")

    model = config.get("model")
    client = model.get("client") if isinstance(model, dict) else None
    if not isinstance(client, dict):
        raise Refused("orchestrator model.client config is missing")
    admin_urls = client.get("admin_base_url") or client.get("base_url")
    if not isinstance(admin_urls, list) or not admin_urls or not all(
        isinstance(url, str) and url.startswith(("http://", "https://"))
        for url in admin_urls
    ):
        raise Refused("orchestrator model.client has no configured inference URLs")

    broadcast = config.get("weight_broadcast")
    source_url = broadcast.get("source_url") if isinstance(broadcast, dict) else None
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        raise Refused("orchestrator weight_broadcast.source_url is missing")
    if f"/{run}/" not in source_url.rstrip("/") + "/":
        raise Refused("adapter source URL is not bound to the active run")

    ports_by_host = {}
    for host in inference_hosts:
        configured_ports = _inference_config(host).get("ports")
        if not isinstance(configured_ports, list) or not configured_ports:
            raise Refused(f"inference ports for {host} are missing")
        if not all(isinstance(port, int) and 1 <= port <= 65535 for port in configured_ports):
            raise Refused(f"inference ports for {host} are invalid")
        ports_by_host[host] = list(configured_ports)
    return {
        "kind": "glm52",
        "engine_container": glm_inference_container,
        "output_dir": output_dir,
        "service_name": service_name,
        "run_marker": f"/{run}/",
        "service_marker": f"/{service_name}/",
        "source_url": source_url.rstrip("/"),
        "ports": ports_by_host[orchestrator_host],
        "ports_by_host": ports_by_host,
    }


def _adapter_server_launch_script(run: str, config: dict) -> str:
    """Build the configured adapter-http launch without touching the trainer."""
    source_url = config.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        raise Refused("active GLM profile has no valid adapter source URL")
    values = {
        "node_id": trainer_head.removeprefix("n"),
        "deployment_root": remote_repo,
        "storage_root": storage_root,
        "runs_path": run_root,
        "trainer_container_name": trainer_container,
        "master_node_id": trainer_head.removeprefix("n"),
        "adapter_node_id": trainer_head.removeprefix("n"),
        "adapter_service_port": adapter_port,
        "adapter_container_name": adapter_container,
        "RUN": run,
        "RUN_NAME": f"run_{run}",
        "BROADCAST_ROOT": run_root,
        "ADAPTER_PREDICTIVE_UPGRADE": 1,
        "ADAPTER_PREDICTIVE_RUN": run,
        "ADAPTER_PREDICTIVE_BROADCAST_PATH": f"{run}/run_{run}/broadcasts",
        "ADAPTER_PREDICTIVE_SOURCE_URL": source_url,
    }
    assignments = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items()
    )
    return f"set -euo pipefail\n{assignments} bash {shlex.quote(trainer_launcher)} adapter-http\n"


def _adapter_server_health_script() -> str:
    return f"""set -euo pipefail
code=$(curl -sS -o /dev/null -m 30 -w '%{{http_code}}' http://127.0.0.1:{adapter_port}/)
[ \"$code\" = 200 ] || {{ echo \"adapter server returned $code\" >&2; exit 71; }}
echo \"adapter server ready on port {adapter_port}\"
"""


def _validate_adapter_stage_log(log: str) -> str:
    """Require a current adapter transfer failure, not a generic error."""
    failures = list(
        re.finditer(
            r"(?:Failed to apply policy update \d+|Policy update \d+ failed on attempt (?P<attempt>\d+)/(?P=attempt); stopping)",
            log,
        )
    )
    if not failures:
        raise Refused("no failed adapter policy update is present in the orchestrator log")
    latest_failure = failures[-1]
    successes = list(re.finditer(r"Staged policy v\d+ on every inference host", log))
    if successes and successes[-1].start() > latest_failure.start():
        raise Refused("the latest adapter policy update already succeeded")
    window = log[max(0, latest_failure.start() - 4000) : latest_failure.end() + 20000]
    transfer_words = ("stage_lora_adapter", "adapter", "lora")
    transport_words = (
        "ReadTimeout",
        "TimeoutError",
        "timed out",
        "ConnectError",
        "TransportError",
        "incomplete adapter",
        "byte range",
    )
    if not any(word.lower() in window.lower() for word in transfer_words):
        raise Refused("latest policy failure has no adapter staging evidence")
    if not any(word.lower() in window.lower() for word in transport_words):
        raise Refused("latest adapter failure has no transfer timeout evidence")
    return latest_failure.group(0)


def _adapter_stage_status_script(status_path: str) -> str:
    return f"""python3 - <<'PY'
import json
import os
import stat

path = {json.dumps(status_path)}
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
if not getattr(os, "O_NOFOLLOW", 0):
    raise SystemExit("O_NOFOLLOW is unavailable")
info = os.lstat(path)
if not stat.S_ISREG(info.st_mode):
    raise SystemExit("adapter stage status is not a regular file")
descriptor = os.open(path, flags)
with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
    value = json.load(stream)
if not isinstance(value, dict):
    raise SystemExit("adapter stage status is not an object")
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
"""


def _latest_adapter_broadcast(run: str) -> AdapterBroadcast:
    run_base = _validated_run_base(run)
    code, output = _ssh(
        trainer_head,
        "test -d "
        + shlex.quote(run_base)
        + " && find "
        + shlex.quote(run_base)
        + " -type f -path '*/broadcasts/step_*/STABLE' -printf '%h\\n' | sort -u",
        timeout=120,
    )
    if code:
        raise RuntimeError(f"cannot inspect stable adapter broadcasts for {run}")
    broadcasts = _parse_adapter_broadcasts(output, run_base)
    if not broadcasts:
        raise Refused(f"no stable adapter broadcast exists for {run}")
    return broadcasts[-1]


def _adapter_stage_status_evidence(run: str, config: dict, log: str) -> str:
    latest = _latest_adapter_broadcast(run)
    run_base = PurePosixPath(_validated_run_base(run))
    try:
        relative_path = PurePosixPath(latest.path).relative_to(run_base)
    except ValueError as error:
        raise Refused("latest adapter broadcast is outside the active run") from error
    source_base = config.get("source_url")
    if not isinstance(source_base, str) or not source_base.startswith(("http://", "https://")):
        raise Refused("active adapter source URL is invalid")
    expected_source = f"{source_base.rstrip('/')}/{relative_path.as_posix()}"
    expected_path = (
        f"{paths['adapter_cache'].rstrip('/')}/"
        + hashlib.sha256(expected_source.encode()).hexdigest()
    )

    completed_steps = [
        int(value)
        for value in re.findall(r"Staged policy v(\d+) on every inference host", log)
    ]
    completed_steps.extend(
        int(value) for value in re.findall(r"Updated weights to step (\d+)", log)
    )
    if completed_steps and max(completed_steps) >= latest.step:
        raise Refused("the latest adapter policy update already succeeded")

    status_path = f"{paths['adapter_cache'].rstrip('/')}/.adapter-stage-status.json"
    script = _adapter_stage_status_script(status_path)
    results = _fan(
        inference_hosts,
        lambda host: _ssh(host, script, timeout=90),
    )
    records: dict[str, dict] = {}
    for host in inference_hosts:
        code, output = results.get(host, (1, "missing status result"))
        if code:
            raise Refused(f"cannot read exact adapter stage status on {host}")
        lines = [line for line in output.splitlines() if line.strip()]
        if len(lines) != 1:
            raise Refused(f"adapter stage status on {host} is not one JSON object")
        try:
            record = json.loads(lines[0])
        except (TypeError, ValueError) as error:
            raise Refused(f"adapter stage status on {host} is invalid") from error
        if not isinstance(record, dict):
            raise Refused(f"adapter stage status on {host} is not an object")
        records[host] = record

    failures = []
    if latest.step <= 0:
        raise Refused("latest adapter broadcast is not a positive policy update")
    stage_limit = _adapter_stage_limit_seconds(latest.step)
    target_exceeded = []
    for host, record in records.items():
        state = record.get("status")
        if state not in {"ready", "failed"}:
            raise Refused(f"adapter stage status on {host} is not terminal")
        if record.get("source_url") != expected_source or record.get("path") != expected_path:
            raise Refused(f"adapter stage status on {host} is stale or mismatched")
        elapsed = record.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or elapsed > stage_limit
        ):
            raise Refused(f"adapter stage duration on {host} is invalid")
        if record.get("policy_step") != latest.step:
            raise Refused(f"adapter stage status on {host} has the wrong policy step")
        if record.get("stage_timeout_seconds") != stage_limit:
            raise Refused(f"adapter stage status on {host} has the wrong operational limit")
        if record.get("acceptance_target_seconds") != adapter_stage_acceptance_target_seconds:
            raise Refused(f"adapter stage status on {host} has the wrong acceptance target")
        if elapsed > adapter_stage_acceptance_target_seconds:
            target_exceeded.append(host)
        if state == "failed":
            error = record.get("error")
            if not isinstance(error, str):
                raise Refused(f"adapter stage failure on {host} has no error reason")
            error = error.strip()
            if re.search(r"TimeoutError|timed out|time limit", error, re.IGNORECASE):
                reason = "timeout"
            elif re.fullmatch(
                r"(?:OSError: )?adapter manifest request failed with HTTP status 500",
                error,
            ):
                reason = "manifest HTTP 500"
            else:
                raise Refused(
                    f"adapter stage failure on {host} is not an allowed transfer failure"
                )
            failures.append((host, reason, error))
    if not failures:
        raise Refused("the latest adapter stage succeeded on every inference host")
    failure_details = ", ".join(
        f"{host} ({error})" for host, _reason, error in failures
    )
    return (
        f"policy update {latest.step} failed on {failure_details}; "
        f"operational limit {stage_limit:g}s; "
        f"300s acceptance target {'exceeded on ' + ', '.join(target_exceeded) if target_exceeded else 'not exceeded'}; "
        "exact terminal stage status matched every inference host"
    )


def _adapter_stage_evidence(run: str, config: dict) -> str:
    code, output = _ssh(
        orchestrator_host,
        f"sudo docker logs --tail 10000 {shlex.quote(orchestrator_container)} 2>&1",
        timeout=180,
    )
    if not code:
        try:
            return _validate_adapter_stage_log(output)
        except Refused:
            pass
    return _adapter_stage_status_evidence(run, config, output)


def _adapter_idle_script(ports: list[int]) -> str:
    if not ports or not all(isinstance(port, int) and 1 <= port <= 65535 for port in ports):
        raise Refused("adapter idle check requires configured inference ports")
    port_list = " ".join(str(port) for port in ports)
    return f"""set -euo pipefail
for port in {port_list}; do
  health=$(curl -sS -o /dev/null -m 30 -w '%{{http_code}}' "http://127.0.0.1:$port/health") || {{ echo "health request failed on $port" >&2; exit 31; }}
  [ "$health" = 200 ] || {{ echo "health returned $health on $port" >&2; exit 32; }}
  metrics=$(curl -sS -m 30 "http://127.0.0.1:$port/metrics") || {{ echo "metrics request failed on $port" >&2; exit 33; }}
  running=$(printf '%s\\n' "$metrics" | awk '
    $1 ~ /^vllm:num_requests_running(\\{{|$)/ {{
      if ($NF !~ /^[0-9]+(\\.[0-9]+)?$/) bad=1
      else {{ total += $NF; found=1 }}
    }}
    END {{ if (!found || bad || total != 0) exit 1; printf "%.0f", total }}
  ') || {{ echo "active or unreadable requests on $port" >&2; exit 34; }}
  echo "port=$port running=$running"
done
"""


def _orchestrator_resume_safety_script(
    ports: list[int], staging_max_seconds: int = orchestrator_repair_staging_max_seconds
) -> str:
    """Check inference safety without requiring a completion request."""
    if not ports or not all(isinstance(port, int) and 1 <= port <= 65535 for port in ports):
        raise Refused("orchestrator resume check requires configured inference ports")
    if (
        isinstance(staging_max_seconds, bool)
        or not isinstance(staging_max_seconds, int)
        or staging_max_seconds <= 0
    ):
        raise Refused("orchestrator staging limit must be a positive integer")
    port_list = " ".join(str(port) for port in ports)
    return f"""set -euo pipefail
for port in {port_list}; do
  health=$(curl -sS -o /dev/null -m 30 -w '%{{http_code}}' "http://127.0.0.1:$port/health") || {{ echo "health request failed on $port" >&2; exit 31; }}
  [ "$health" = 200 ] || {{ echo "health returned $health on $port" >&2; exit 32; }}
  liveness=$(curl -sS -o /dev/null -m 30 -w '%{{http_code}}' "http://127.0.0.1:$port/liveness") || {{ echo "liveness request failed on $port" >&2; exit 33; }}
  [ "$liveness" = 200 ] || {{ echo "liveness returned $liveness on $port" >&2; exit 34; }}
  metrics=$(curl -sS -m 30 "http://127.0.0.1:$port/metrics") || {{ echo "metrics request failed on $port" >&2; exit 35; }}
  running=$(printf '%s\\n' "$metrics" | awk '
    $1 ~ /^vllm:num_requests_running(\\{{|$)/ {{
      if ($NF !~ /^[0-9]+(\\.[0-9]+)?$/) bad=1
      else {{ total += $NF; found=1 }}
    }}
    END {{ if (!found || bad || total != 0) exit 1; printf "%.0f", total }}
  ') || {{ echo "active or unreadable requests on $port" >&2; exit 36; }}
  staging=$(curl -sS -m 30 "http://127.0.0.1:$port/staging_status") || {{ echo "staging status request failed on $port" >&2; exit 37; }}
  staging_state=$(printf '%s\\n' "$staging" | sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p')
  staging_age=$(printf '%s\\n' "$staging" | sed -n 's/.*"age_seconds"[[:space:]]*:[[:space:]]*\\([0-9][0-9.]*\\).*/\\1/p')
  case "$staging_state" in
    idle|ready|failed)
      ;;
    staging)
      [ -n "$staging_age" ] || {{ echo "staging age is missing on $port" >&2; exit 39; }}
      awk -v age="$staging_age" -v limit={staging_max_seconds} 'BEGIN {{ exit !(age >= 0 && age <= limit) }}' || {{ echo "staging exceeds the bounded limit on $port" >&2; exit 40; }}
      ;;
    *)
      echo "unknown staging status on $port" >&2
      exit 38
      ;;
  esac
  echo "port=$port health=$health liveness=$liveness running=$running staging=$staging_state age=$staging_age"
done
"""


def _adapter_prefetch_stop_script(run_marker: str, service_marker: str, output_dir: str) -> str:
    for value in (run_marker, service_marker, output_dir):
        if not value or any(ch in value for ch in "'\\\n"):
            raise Refused("adapter process identity marker is unsafe")
    return f"""set -euo pipefail
run_marker={shlex.quote(run_marker)}
service_marker={shlex.quote(service_marker)}
service_dir={shlex.quote(output_dir)}

proc_read() {{
  sudo cat "$1" 2>/dev/null || cat "$1" 2>/dev/null
}}

process_matches() {{
  local pid=$1 exe command_line environment
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ "$pid" -gt 1 ] || return 1
  exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
  case "${{exe##*/}}" in curl|timeout) ;; *) return 1;; esac
  command_line=$(proc_read "/proc/$pid/cmdline" | tr '\\0' ' ') || return 2
  [ -n "$command_line" ] || return 2
  case "$command_line" in *stage_lora_adapter*) ;; *) return 1;; esac
  case "$command_line" in *"$run_marker"*) ;; *) return 1;; esac
  case "$command_line" in *"$service_marker"*) ;; *) return 1;; esac
  environment=$(proc_read "/proc/$pid/environ" | tr '\\0' '\\n') || return 2
  case " $environment " in *" SOURCE_RUN_DIR=$service_dir "*) ;; *) return 1;; esac
  return 0
}}

pids=()
ambiguous=0
for proc_dir in /proc/[0-9]*; do
  pid=${{proc_dir##*/}}
  if [ -e "$proc_dir/exe" ]; then
    exe=$(readlink "$proc_dir/exe" 2>/dev/null || true)
    case "${{exe##*/}}" in
      curl|timeout)
        if process_matches "$pid"; then
          pids+=("$pid")
        elif [ "$?" = 2 ]; then
          ambiguous=1
        fi
        ;;
    esac
  fi
done
[ "$ambiguous" = 0 ] || {{ echo "adapter prefetch identity is unreadable" >&2; exit 41; }}

stop_one() {{
  local pid=$1 match_status
  if process_matches "$pid"; then :; else
    match_status=$?
    [ "$match_status" = 1 ] && return 0
    return 2
  fi
  kill -TERM "$pid" 2>/dev/null || sudo kill -TERM "$pid" 2>/dev/null || return 1
  for _ in $(seq 1 200); do
    if process_matches "$pid"; then :; else
      match_status=$?
      [ "$match_status" = 1 ] && return 0
      return 2
    fi
    sleep 0.1
  done
  if process_matches "$pid"; then :; else
    match_status=$?
    [ "$match_status" = 1 ] && return 0
    return 2
  fi
  kill -KILL "$pid" 2>/dev/null || sudo kill -KILL "$pid" 2>/dev/null || return 1
  for _ in $(seq 1 50); do
    if process_matches "$pid"; then :; else
      match_status=$?
      [ "$match_status" = 1 ] && return 0
      return 2
    fi
    sleep 0.1
  done
  return 1
}}

for pid in "${{pids[@]}}"; do
  stop_one "$pid" || {{ echo "exact adapter prefetch pid $pid did not stop" >&2; exit 42; }}
done
for proc_dir in /proc/[0-9]*; do
  pid=${{proc_dir##*/}}
  if process_matches "$pid"; then
    echo "adapter prefetch remains pid=$pid" >&2
    exit 43
  elif [ "$?" = 2 ]; then
    echo "adapter prefetch identity became unreadable" >&2
    exit 44
  fi
done
echo "stopped ${{#pids[@]}} exact adapter prefetch process(es)"
"""


def _adapter_engine_state_script(
    container: str,
    ports: list[int],
    run_marker: str,
    service_marker: str,
    output_dir: str,
) -> str:
    """Check exact engine identity, health, and stale adapter activity."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", container):
        raise Refused(f"inference container name is invalid: {container!r}")
    if not ports or not all(isinstance(port, int) and 1 <= port <= 65535 for port in ports):
        raise Refused("inference health ports are invalid")
    for value in (run_marker, service_marker, output_dir):
        if not value or any(ch in value for ch in "'\\\n"):
            raise Refused("adapter process identity marker is unsafe")
    quoted = shlex.quote(container)
    run_quoted = shlex.quote(run_marker)
    service_quoted = shlex.quote(service_marker)
    output_quoted = shlex.quote(output_dir)
    port_list = " ".join(str(port) for port in ports)
    return f"""set -euo pipefail
container={quoted}
run_marker={run_quoted}
service_marker={service_quoted}
service_dir={output_quoted}
name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
[ "$name" = "/$container" ] || {{ echo "configured inference container identity is not exact" >&2; exit 51; }}
state=$(sudo docker inspect --format '{{{{.State.Running}}}}' "$container" 2>/dev/null || true)
[ "$state" = true ] || {{ echo "configured inference container is not running" >&2; exit 52; }}

proc_read() {{
  sudo cat "$1" 2>/dev/null || cat "$1" 2>/dev/null
}}

staging_process_matches() {{
  local pid=$1 exe command_line environment
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ "$pid" -gt 1 ] || return 1
  exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
  case "${{exe##*/}}" in curl|timeout) ;; *) return 1;; esac
  command_line=$(proc_read "/proc/$pid/cmdline" | tr '\\0' ' ') || return 2
  [ -n "$command_line" ] || return 2
  case "$command_line" in *"$run_marker"*) ;; *) return 1;; esac
  case "$command_line" in *"$service_marker"*) ;; *) return 1;; esac
  case "$command_line" in *stage_lora_adapter*) ;; *) return 1;; esac
  environment=$(proc_read "/proc/$pid/environ" | tr '\\0' '\\n') || return 2
  case " $environment " in *" SOURCE_RUN_DIR=$service_dir "*) ;; *) return 1;; esac
  return 0
}}

staging_pids=()
ambiguous=0
for proc_dir in /proc/[0-9]*; do
  pid=${{proc_dir##*/}}
  if [ -e "$proc_dir/exe" ]; then
    exe=$(readlink "$proc_dir/exe" 2>/dev/null || true)
    case "${{exe##*/}}" in
      curl|timeout)
        if staging_process_matches "$pid"; then
          staging_pids+=("$pid")
        elif [ "$?" = 2 ]; then
          ambiguous=1
        fi
        ;;
    esac
  fi
done
[ "$ambiguous" = 0 ] || {{ echo "adapter staging process identity is unreadable" >&2; exit 54; }}
[ "${{#staging_pids[@]}}" = 0 ] || {{ echo "adapter staging remains: ${{staging_pids[*]}}" >&2; exit 55; }}

healthy=1
failed_ports=""
for port in {port_list}; do
  health=$(curl -sS -o /dev/null -m 30 -w '%{{http_code}}' "http://127.0.0.1:$port/health" 2>/dev/null || true)
  if [ "$health" != 200 ]; then
    healthy=0
    failed_ports="$failed_ports $port:$health"
  fi
done
if [ "$healthy" = 1 ]; then
  echo "adapter engine ready: exact container $container is healthy and has no staging activity"
  exit 0
fi
echo "adapter engine restart required: unhealthy ports:$failed_ports"
exit 10
"""


def _restart_adapter_engine_script(
    container: str,
    health_port: int,
    startup_timeout_seconds: int = adapter_update_engine_startup_timeout_seconds,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", container):
        raise Refused(f"inference container name is invalid: {container!r}")
    if not isinstance(health_port, int) or not 1 <= health_port <= 65535:
        raise Refused("inference health port is invalid")
    if (
        isinstance(startup_timeout_seconds, bool)
        or not isinstance(startup_timeout_seconds, int)
        or startup_timeout_seconds < 600
    ):
        raise Refused("inference startup timeout must be an integer >= 600 seconds")
    quoted = shlex.quote(container)
    return f"""set -euo pipefail
container={quoted}
name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
[ "$name" = "/$container" ] || {{ echo "configured inference container identity is not exact" >&2; exit 51; }}
state=$(sudo docker inspect --format '{{{{.State.Running}}}}' "$container" 2>/dev/null || true)
[ "$state" = true ] || {{ echo "configured inference container is not running" >&2; exit 52; }}
sudo docker restart --time 30 "$container" >/dev/null
for _ in $(seq 1 {startup_timeout_seconds}); do
  health=$(curl -sS -o /dev/null -m 10 -w '%{{http_code}}' http://127.0.0.1:{health_port}/health 2>/dev/null || true)
  if [ "$health" = 200 ]; then
    echo "restarted exact inference container $container"
    exit 0
  fi
  sleep 1
done
echo "inference container did not become healthy after restart" >&2
exit 53
"""


def _orchestrator_state_script(container: str) -> str:
    """Check the exact configured orchestrator container without changing it."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", container):
        raise Refused(f"orchestrator container name is invalid: {container!r}")
    quoted = shlex.quote(container)
    return f"""set -euo pipefail
container={quoted}
if name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null); then
  :
else
  listed=$(sudo docker ps -a --filter "name=^$container$" --format '{{{{.Names}}}}' 2>/dev/null) || {{
    echo "configured orchestrator container state is unknown" >&2
    exit 62
  }}
  [ -z "$listed" ] || {{
    echo "configured orchestrator container state is unknown" >&2
    exit 62
  }}
  echo absent
  exit 0
fi
[ "$name" = "/$container" ] || {{ echo "configured orchestrator container identity is not exact" >&2; exit 61; }}
state=$(sudo docker inspect --format '{{{{.State.Running}}}}' "$container" 2>/dev/null || true)
case "$state" in
  true) echo running ;;
  false) echo stopped ;;
  *) echo "configured orchestrator container state is unknown" >&2; exit 62 ;;
esac
"""


def _recreate_orchestrator_script(
    run: str, profile: dict, was_running: bool, observed_state: str | None = None
) -> str:
    """Recreate only the exact orchestrator with the active run settings."""
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused(f"orchestrator run name is invalid: {run!r}")
    if not isinstance(profile, dict):
        raise Refused("orchestrator profile must be a mapping")
    orchestrator_config = profile.get("orchestrator")
    if not isinstance(orchestrator_config, str):
        raise Refused(f"active profile for {run} has no orchestrator config")
    config_path = PurePosixPath(orchestrator_config)
    if (
        config_path.is_absolute()
        or config_path.as_posix() != orchestrator_config
        or ".." in config_path.parts
        or not (repo_root / config_path).is_file()
    ):
        raise Refused(f"active orchestrator config is unsafe: {orchestrator_config!r}")
    if not isinstance(was_running, bool):
        raise Refused("orchestrator running state must be a boolean")
    expected_state = observed_state or ("running" if was_running else "stopped")
    if expected_state not in {"running", "stopped", "absent"}:
        raise Refused(f"orchestrator state is invalid: {expected_state!r}")
    if expected_state == "running" and not was_running:
        raise Refused("running orchestrator state conflicts with status")
    if expected_state == "stopped" and was_running:
        raise Refused("stopped orchestrator state conflicts with status")
    if expected_state == "absent" and was_running:
        raise Refused("absent orchestrator state conflicts with status")
    for name, value in (
        ("trainer", trainer_container),
        ("orchestrator", orchestrator_container),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise Refused(f"{name} container name is invalid: {value!r}")
    trainer_nodes = [host for host in trainer_hosts if host not in blocked_hosts]
    if trainer_head not in trainer_nodes:
        raise Refused("configured trainer head is not in the active trainer hosts")
    trainer_addresses = " ".join(_ssh_address(host) for host in trainer_nodes)
    if not trainer_addresses:
        raise Refused("configured trainer addresses are empty")

    expected = "true" if expected_state == "running" else "false"
    quoted_orchestrator = shlex.quote(orchestrator_container)
    quoted_run = shlex.quote(run)
    quoted_run_name = shlex.quote(f"run_{run}")
    quoted_config = shlex.quote(f"{remote_repo}/{orchestrator_config}")
    quoted_addresses = shlex.quote(trainer_addresses)
    quoted_trainer = shlex.quote(trainer_container)
    quoted_launcher = shlex.quote(orchestrator_launcher)
    return f"""set -euo pipefail
container={quoted_orchestrator}
if [ {shlex.quote(expected_state)} = absent ]; then
  if name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null); then
    :
  else
    listed=$(sudo docker ps -a --filter "name=^$container$" --format '{{{{.Names}}}}' 2>/dev/null) || {{ echo "orchestrator state changed before recreation" >&2; exit 62; }}
    [ -z "$listed" ] || {{ echo "orchestrator state changed before recreation" >&2; exit 62; }}
    name=""
  fi
  [ -z "$name" ] || {{ echo "orchestrator state changed before recreation" >&2; exit 62; }}
else
  name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
  [ "$name" = "/$container" ] || {{ echo "configured orchestrator container identity is not exact" >&2; exit 61; }}
  state=$(sudo docker inspect --format '{{{{.State.Running}}}}' "$container" 2>/dev/null || true)
  [ "$state" = {expected} ] || {{ echo "orchestrator state changed before recreation" >&2; exit 62; }}
  sudo docker rm -f "$container" >/dev/null
fi
RUN={quoted_run} RUN_NAME={quoted_run_name} \\
  ORCH_CONFIG={quoted_config} \\
  TRAINER_ADDRESSES={quoted_addresses} \\
  TRAINER_CT={quoted_trainer} ORCH_CT={quoted_orchestrator} \\
  bash {quoted_launcher}
name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
[ "$name" = "/$container" ] || {{ echo "recreated orchestrator container identity is not exact" >&2; exit 63; }}
state=$(sudo docker inspect --format '{{{{.State.Running}}}}' "$container" 2>/dev/null || true)
[ "$state" = true ] || {{ echo "recreated orchestrator container is not running" >&2; exit 64; }}
echo "recreated exact orchestrator container $container"
"""


def repair(what: str) -> str:
    """Apply a named repair to the configured hosts."""
    if what.startswith("engine-watchdog:"):
        return _repair_engine_watchdog(what.split(":", 1)[1])
    if what.startswith("engine:"):
        return _repair_engine(what.split(":", 1)[1])
    if what == "supervisors":
        return _repair_supervisors()
    if what == "microbatch-sync":
        return _repair_microbatch_sync()
    if what == "adapter-update":
        return _repair_adapter_update()
    if what == "adapter-server":
        return _repair_adapter_server()
    if what == "adapter-server:reload":
        return _repair_adapter_server(force=True)
    if what == "orchestrator":
        return _repair_orchestrator()
    if what == "adapter-tunnels":
        return _repair_adapter_tunnels()
    if what.startswith("adapter-tunnels:"):
        config_dir = what.split(":", 1)[1]
        if not config_dir:
            raise Refused("adapter-tunnels requires a run config directory")
        return _repair_adapter_tunnels(config_dir)
    raise Refused(
        f"unknown repair {what!r}. Use engine-watchdog:<host>, engine:<host>, supervisors, microbatch-sync, adapter-update, adapter-server[:reload], orchestrator, or adapter-tunnels[:config-dir]."
    )


def _repair_adapter_server(force: bool = False) -> str:
    """Reload the active GLM adapter server through the configured launcher."""
    run = current_run()
    config = _adapter_update_config(run)
    if config["kind"] != "glm52":
        raise Refused("adapter-server repair is only supported for the active GLM profile")
    before = status(run, quiet=True)
    if not before.trainer_up or before.trainer_failed:
        raise Refused("adapter-server repair requires a healthy trainer cluster")
    if before.adapter == "200" and not force:
        raise Refused(
            "adapter server is healthy; use repair('adapter-server:reload') for an explicit reload"
        )
    inflight = re.search(r"(\d+) inflight", before.batch or "")
    if inflight and int(inflight.group(1)) > 0:
        raise Refused("adapter-server repair requires zero active rollout requests")

    for path in (
        "training_ops/runtime/china/cluster.sh",
        "training_ops/runtime/china/adapter_server.py",
        "prime-rl/src/prime_rl/utils/adapter_delta.py",
    ):
        if not deploy(path, [trainer_head]):
            raise RuntimeError(f"adapter-server deployment failed for {path}")
    code, output = _ssh(
        trainer_head,
        _adapter_server_launch_script(run, config),
        timeout=300,
    )
    if code:
        raise RuntimeError(f"adapter-server launcher failed on {trainer_head}: {output.strip()}")
    health_code, health_output = _ssh(
        trainer_head,
        _adapter_server_health_script(),
        timeout=60,
    )
    if health_code:
        raise RuntimeError(
            f"adapter-server health check failed on {trainer_head}: {health_output.strip()}"
        )
    result = (
        f"reloaded active GLM adapter server {adapter_container} on {trainer_head}; "
        f"adapter status was {before.adapter}, health passed on port {adapter_port}"
    )
    print(result)
    return result


def _repair_adapter_update() -> str:
    run = current_run()
    profile = _config_for_run(run)
    config = _adapter_update_config(run)
    before = status(run, quiet=True)
    if not before.trainer_up or before.trainer_failed:
        raise Refused("adapter repair requires a healthy trainer cluster")
    if any(
        node.reachable and node.gpu_util >= 50
        for node in before.nodes
    ):
        raise Refused("adapter repair requires the trainer to be waiting")
    orchestrator_was_running = before.orch_up or before.orch_failed
    if orchestrator_was_running:
        observed_state = "running"
    else:
        state_code, state_output = _ssh(
            orchestrator_host,
            _orchestrator_state_script(orchestrator_container),
            timeout=90,
        )
        if state_code:
            raise Refused(
                "adapter repair requires a stopped or absent exact orchestrator container: "
                f"{state_output.strip()}"
            )
        observed_states = state_output.strip().splitlines()
        if len(observed_states) != 1 or observed_states[0] not in {"stopped", "absent"}:
            raise Refused(
                "adapter repair requires a stopped or absent exact orchestrator container; "
                f"observed {observed_states[-1] if observed_states else 'no state'}"
            )
        observed_state = observed_states[0]
    if len(before.engines) != len(inference_hosts) or any(
        not engine.reachable
        or not engine.healthy
        or engine.probes_total <= 0
        or engine.probes_ok != engine.probes_total
        for engine in before.engines
    ):
        raise Refused("adapter repair requires every configured inference endpoint to be healthy")
    inflight = re.search(r"(\d+) inflight", before.batch or "")
    if inflight and int(inflight.group(1)) > 0:
        raise Refused("adapter repair requires zero active rollout requests")

    idle = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_idle_script(config["ports_by_host"][host]),
            timeout=90,
        ),
    )
    failed_idle = {host: output.strip() for host, (code, output) in idle.items() if code}
    if failed_idle:
        raise Refused(f"adapter repair requires idle healthy inference endpoints: {failed_idle}")
    evidence = _adapter_stage_evidence(run, config)

    stopped = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_prefetch_stop_script(
                config["run_marker"], config["service_marker"], config["output_dir"]
            ),
            timeout=90,
        ),
    )
    failed_stop = {host: output.strip() for host, (code, output) in stopped.items() if code}
    if failed_stop:
        raise RuntimeError(f"exact adapter prefetch stop failed: {failed_stop}")

    deployment_targets = (
        ("prime-rl/src/prime_rl/utils/client.py", None),
        ("prime-rl/src/prime_rl/inference/vllm/server.py", inference_hosts),
        (
            "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py",
            inference_hosts,
        ),
        ("prime-rl/src/prime_rl/orchestrator/orchestrator.py", [orchestrator_host]),
        ("prime-rl/src/prime_rl/orchestrator/watcher.py", [orchestrator_host]),
        ("prime-rl/src/prime_rl/utils/checkpoint_delta.py", inference_hosts),
        ("prime-rl/src/prime_rl/inference/relay.py", [orchestrator_host]),
        ("training_ops/runtime/h200/start_orchestrator.sh", [orchestrator_host]),
        ("training_ops/runtime/h200/sync_rollout_batches.sh", None),
    )
    for path, nodes in deployment_targets:
        deployed = deploy(path) if nodes is None else deploy(path, nodes=nodes)
        if not deployed:
            raise RuntimeError(f"adapter repair deployment failed for {path}")

    engine_state = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_engine_state_script(
                config["engine_container"],
                config["ports_by_host"][host],
                config["run_marker"],
                config["service_marker"],
                config["output_dir"],
            ),
            timeout=90,
        ),
    )
    failed_state = {
        host: output.strip()
        for host, (code, output) in engine_state.items()
        if code not in (0, 10)
    }
    if failed_state:
        raise RuntimeError(f"inference identity or staging check failed: {failed_state}")
    reload_hosts = list(inference_hosts)
    reloaded = _fan(
        reload_hosts,
        lambda host: _ssh(
            host,
            _restart_adapter_engine_script(
                config["engine_container"],
                config["ports_by_host"][host][0],
                adapter_update_engine_startup_timeout_seconds,
            ),
            timeout=adapter_update_engine_startup_timeout_seconds + 120,
        ),
    )
    failed_reload = {
        host: output.strip() for host, (code, output) in reloaded.items() if code
    }
    if failed_reload:
        raise RuntimeError(f"inference reload failed: {failed_reload}")

    post_engine_state = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_engine_state_script(
                config["engine_container"],
                config["ports_by_host"][host],
                config["run_marker"],
                config["service_marker"],
                config["output_dir"],
            ),
            timeout=90,
        ),
    )
    failed_post_state = {
        host: output.strip()
        for host, (code, output) in post_engine_state.items()
        if code
    }
    if failed_post_state:
        raise RuntimeError(f"inference identity or staging check failed after reload: {failed_post_state}")

    post_idle = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_idle_script(config["ports_by_host"][host]),
            timeout=90,
        ),
    )
    failed_post_idle = {
        host: output.strip() for host, (code, output) in post_idle.items() if code
    }
    if failed_post_idle:
        raise RuntimeError(f"inference was not idle after restart: {failed_post_idle}")

    resume_code, resume_output = _ssh(
        orchestrator_host,
        _recreate_orchestrator_script(
            run, profile, orchestrator_was_running, observed_state=observed_state
        ),
        timeout=180,
    )
    if resume_code:
        raise RuntimeError(f"orchestrator resume failed: {resume_output.strip()}")
    resume_action = "recreated"
    result = (
        f"adapter update repaired for {run}: {evidence}; stopped exact prefetch processes; "
        "deployed the transfer client, receiver, relay, and batch synchronizer; "
        f"reloaded exact GLM inference containers on {','.join(reload_hosts)} and "
        "verified health, identity, and staging state; "
        f"{resume_action} exact orchestrator and relay only after inference recovered"
    )
    print(result)
    return result


def _repair_microbatch_sync() -> str:
    """Restart the current run's exact rank-0 microbatch distributor."""
    run = current_run()
    source_dir, peer_addresses, sync_script = _microbatch_sync_config(run)
    script = _microbatch_sync_repair_script(run, source_dir, peer_addresses, sync_script)
    code, output = _ssh(trainer_head, script, timeout=300)
    detail = output.strip()
    if code:
        raise RuntimeError(
            f"microbatch sync repair failed on {trainer_head}: "
            f"{detail or f'exit {code}'}"
        )
    result = detail.splitlines()[-1] if detail else "microbatch sync repaired"
    print(result)
    return result


def _repair_orchestrator() -> str:
    """Resume the exact orchestrator after proving all peers are safe."""
    run = current_run()
    profile = _config_for_run(run)
    if not isinstance(profile.get("orchestrator"), str):
        raise Refused(f"active profile for {run} has no orchestrator config")

    before = status(run, quiet=True)
    if not before.trainer_up or before.trainer_failed:
        raise Refused("orchestrator repair requires a healthy trainer cluster")
    if before.orch_up:
        raise Refused(
            "orchestrator is healthy; repair requires status to report it down"
        )
    if any(not node.reachable for node in before.nodes):
        raise Refused("orchestrator repair requires every trainer host to be reachable")
    if any(node.gpu_util >= 50 for node in before.nodes):
        raise Refused("orchestrator repair requires the trainer to be idle or waiting")
    state_code, state_output = _ssh(
        orchestrator_host,
        _orchestrator_state_script(orchestrator_container),
        timeout=90,
    )
    if state_code:
        raise Refused(
            "orchestrator repair requires one exact configured container: "
            f"{state_output.strip()}"
        )
    observed_state = state_output.strip().splitlines()
    if len(observed_state) != 1 or observed_state[0] not in {
        "running",
        "stopped",
        "absent",
    }:
        raise Refused(
            "orchestrator repair found an ambiguous container state: "
            f"{state_output.strip() or 'no state'}"
        )
    was_running = observed_state[0] == "running"
    if was_running != (before.orch_up or before.orch_failed):
        raise Refused(
            "orchestrator container state changed during the safety check; "
            "refusing an ambiguous resume"
        )
    inflight = re.search(r"(\d+) inflight", before.batch or "")
    if was_running and inflight and int(inflight.group(1)) > 0:
        raise Refused("orchestrator repair requires zero active rollout requests")

    checks = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _orchestrator_resume_safety_script(
                _inference_config(host)["ports"],
            ),
            timeout=120,
        ),
    )
    failed = {host: output.strip() for host, (code, output) in checks.items() if code}
    if failed:
        raise Refused(
            "orchestrator repair requires healthy, live, idle inference endpoints: "
            f"{failed}"
        )

    client_path = "prime-rl/src/prime_rl/utils/client.py"
    if not deploy(client_path):
        raise RuntimeError(f"orchestrator repair deployment failed for {client_path}")
    resume_code, resume_output = _ssh(
        orchestrator_host,
        _recreate_orchestrator_script(
            run, profile, was_running, observed_state=observed_state[0]
        ),
        timeout=180,
    )
    if resume_code:
        raise RuntimeError(f"orchestrator resume failed: {resume_output.strip()}")
    action = "recreated"
    result = (
        f"{action} exact orchestrator {orchestrator_container} for {run}; "
        "trainer was idle, rollout requests were zero, inference health/liveness "
        "passed, staging was bounded, and the resume-timeout client was deployed"
    )
    print(result)
    return result


def _repair_adapter_tunnels(config_dir: str | None = None) -> str:
    """Reconcile the configured adapter tunnels without starting services."""
    if config_dir:
        profile = _config_dir(config_dir, require_orchestrator=True)
    else:
        profile = _config_for_run(current_run())
    orchestrator_config = profile.get("orchestrator")
    if not isinstance(orchestrator_config, str):
        raise Refused("selected profile has no orchestrator config for adapter tunnels")
    try:
        with (repo_root / orchestrator_config).open("rb") as stream:
            edge = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read orchestrator config {orchestrator_config}: {error}") from error
    output_dir = edge.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.startswith(run_root.rstrip("/") + "/"):
        raise Refused("selected orchestrator config has no configured run output directory")
    output_path = PurePosixPath(output_dir.rstrip("/"))
    run = output_path.parent.name
    run_name = output_path.name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run) or run_name != f"run_{run}":
        raise Refused("selected orchestrator output directory is not an exact run path")
    trainer_nodes = [host for host in trainer_hosts if host not in blocked_hosts]
    trainer_addresses = " ".join(_ssh_address(host) for host in trainer_nodes)
    if not trainer_addresses:
        raise Refused("configured trainer addresses are empty")
    if not deploy(
        "training_ops/runtime/h200/start_orchestrator.sh",
        nodes=[orchestrator_host],
    ):
        raise RuntimeError("adapter tunnel launcher deployment failed")
    command = (
        f"RUN={shlex.quote(run)} RUN_NAME={shlex.quote(run_name)} "
        f"ORCH_CONFIG={shlex.quote(f'{remote_repo}/{orchestrator_config}')} "
        f"TRAINER_ADDRESSES={shlex.quote(trainer_addresses)} "
        f"TRAINER_CT={shlex.quote(trainer_container)} ORCH_CT={shlex.quote(orchestrator_container)} "
        f"bash {shlex.quote(orchestrator_launcher)} --tunnels-only"
    )
    code, output = _ssh(orchestrator_host, command, timeout=180)
    if code:
        raise RuntimeError(f"adapter tunnel reconciliation failed: {output.strip() or f'exit {code}'}")
    result = output.strip().splitlines()[-1] if output.strip() else "adapter tunnel reconciliation complete"
    print(result)
    return result


def _engine_watchdog_identity_script(
    host: str,
    engine_container: str,
    watchdog_script: str,
    start_script: str,
    model: str,
    startup_grace: int,
    *,
    ensure_running: bool,
) -> str:
    """Stop stale watchdogs for one configured engine identity."""
    values = {
        "ENGINE_HOST": host,
        "ENGINE_CONTAINER": engine_container,
    }
    lines = [
        "set -euo pipefail",
        f"watchdog_script={shlex.quote(watchdog_script)}",
        f"start_script={shlex.quote(start_script)}",
        f"model={shlex.quote(model)}",
        f"startup_grace={shlex.quote(str(startup_grace))}",
        "watchdog_command_matches() {",
        "  local pid=$1 command_line",
        "  case \"$pid\" in ''|*[!0-9]*) return 1;; esac",
        "  [ \"$pid\" -gt 1 ] || return 1",
        "  [ -r \"/proc/$pid/cmdline\" ] || return 1",
        "  command_line=$(cat \"/proc/$pid/cmdline\" 2>/dev/null | tr '\\0' ' ' || true)",
        "  command_line=${command_line% }",
        "  case \"$command_line\" in",
        r"    bash\ */engine_watchdog.sh|/bin/bash\ */engine_watchdog.sh|/usr/bin/bash\ */engine_watchdog.sh) return 0;;",
        "  esac",
        "  return 1",
        "}",
        "watchdog_environment() {",
        "  local pid=$1",
        "  [ -r \"/proc/$pid/environ\" ] || return 1",
        "  cat \"/proc/$pid/environ\" 2>/dev/null | tr '\\0' '\\n'",
        "}",
        "watchdog_matches() {",
        "  local pid=$1 environment",
        "  watchdog_command_matches \"$pid\" || return 1",
        "  environment=$(watchdog_environment \"$pid\" || true)",
    ]
    for key, value in values.items():
        lines.append(
            f"  printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'{key}={value}')} || return 1"
        )
    lines.extend(
        [
            "  return 0",
            "}",
            "canonical_watchdog_matches() {",
            "  local pid=$1 command_line environment",
            "  watchdog_matches \"$pid\" || return 1",
            "  command_line=$(cat \"/proc/$pid/cmdline\" 2>/dev/null | tr '\\0' ' ' || true)",
            "  command_line=${command_line% }",
            "  case \"$command_line\" in",
            '    "bash $watchdog_script"|"/bin/bash $watchdog_script"|"/usr/bin/bash $watchdog_script") ;;',
            "    *) return 1;;",
            "  esac",
            "  environment=$(watchdog_environment \"$pid\" || true)",
            f"  printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'START={start_script}')} || return 1",
            f"  printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'MODEL={model}')} || return 1",
            f"  printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'STARTUP_GRACE={startup_grace}')} || return 1",
            "  return 0",
            "}",
            "stop_watchdog() {",
            "  local pid=$1 pgid",
            "  watchdog_matches \"$pid\" || return 0",
            "  pgid=$(awk '{print $5}' \"/proc/$pid/stat\" 2>/dev/null || true)",
            '  [ "$pgid" = "$pid" ] || { echo "watchdog identity is not its own process group: pid=$pid pgid=$pgid" >&2; return 1; }',
            '  kill -TERM -- "-$pid" 2>/dev/null || true',
            '  for _ in $(seq 1 50); do watchdog_matches "$pid" || break; sleep 0.1; done',
            '  if watchdog_matches "$pid"; then kill -KILL -- "-$pid" 2>/dev/null || true; fi',
            '  for _ in $(seq 1 20); do watchdog_matches "$pid" || break; sleep 0.1; done',
            '  watchdog_matches "$pid" && { echo "watchdog did not exit: pid=$pid" >&2; return 1; }',
            "}",
            "matches=()",
            "canonical_matches=()",
            "for proc_dir in /proc/[0-9]*; do",
            "  pid=${proc_dir##*/}",
            "  watchdog_matches \"$pid\" && matches+=(\"$pid\")",
            "  canonical_watchdog_matches \"$pid\" && canonical_matches+=(\"$pid\")",
            "done",
        ]
    )
    if ensure_running:
        lines.extend(
            [
                'if [ "${#matches[@]}" = 1 ] && [ "${#canonical_matches[@]}" = 1 ]; then',
                '  echo "canonical engine watchdog already running"',
                "  exit 0",
                "fi",
            ]
        )
    lines.extend(
        [
            'for pid in "${matches[@]}"; do stop_watchdog "$pid"; done',
            "matches=()",
            "for proc_dir in /proc/[0-9]*; do",
            "  pid=${proc_dir##*/}",
            "  watchdog_matches \"$pid\" && matches+=(\"$pid\")",
            "done",
            '[ "${#matches[@]}" = 0 ] || { echo "target engine watchdog did not stop: ${matches[*]}" >&2; exit 71; }',
        ]
    )
    if ensure_running:
        canonical_environment = {
            **values,
            "STARTUP_GRACE": str(startup_grace),
            "MODEL": model,
            "START": start_script,
        }
        environment = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in canonical_environment.items()
        )
        lines.extend(
            [
                f"env {environment} nohup setsid bash \"$watchdog_script\" >>/root/engine_watchdog.log 2>&1 </dev/null &",
                "new_pid=$!",
                "matched=0",
                "for _ in $(seq 1 50); do",
                "  if canonical_watchdog_matches \"$new_pid\"; then matched=1; break; fi",
                "  sleep 0.1",
                "done",
                '[ "$matched" = 1 ] || { echo "canonical engine watchdog was not observed" >&2; exit 72; }',
                "matches=()",
                "canonical_matches=()",
                "for proc_dir in /proc/[0-9]*; do",
                "  pid=${proc_dir##*/}",
                "  watchdog_matches \"$pid\" && matches+=(\"$pid\")",
                "  canonical_watchdog_matches \"$pid\" && canonical_matches+=(\"$pid\")",
                "done",
                '[ "${#matches[@]}" = 1 ] && [ "${#canonical_matches[@]}" = 1 ] || { echo "target engine watchdog identity is ambiguous" >&2; exit 73; }',
                'echo "reconciled one canonical engine watchdog"',
            ]
        )
    else:
        lines.append('echo "stopped target engine watchdogs"')
    return "\n".join(lines) + "\n"


def _repair_engine(host: str) -> str:
    _check_host(host)
    if host not in inference_hosts:
        raise Refused(f"{host} is not an engine host. Use one of {inference_hosts}.")
    engine_container = glm_inference_container
    engine_model = glm_model
    start_script = f"{runtime_root}/h200/start_inference.sh"
    startup_grace = 600
    print(
        f"{host}: removing and restarting {engine_container}. The model load needs 5 to 10 minutes."
    )
    watchdog_script = f"{runtime_root}/h200/engine_watchdog.sh"
    stop_code, stop_output = _ssh(
        host,
        _engine_watchdog_identity_script(
            host,
            engine_container,
            watchdog_script,
            start_script,
            engine_model,
            startup_grace,
            ensure_running=False,
        ),
    )
    if stop_code:
        raise RuntimeError(f"cannot stop the watchdog on {host}: {stop_output.strip()}")
    rc, out = _ssh(
        host,
        f"sudo docker rm -f {engine_container} >/dev/null 2>&1; "
        f"nohup setsid env ENGINE_HOST={shlex.quote(host)} bash {start_script} "
        ">/root/start_inference.log 2>&1 </dev/null & "
        "sleep 3; echo issued",
        timeout=180,
    )
    if rc:
        raise RuntimeError(f"inference restart failed on {host}: {out.strip()}")
    watchdog_code, watchdog_output = _ssh(
        host,
        _engine_watchdog_identity_script(
            host,
            engine_container,
            watchdog_script,
            start_script,
            engine_model,
            startup_grace,
            ensure_running=True,
        ),
    )
    if watchdog_code:
        raise RuntimeError(
            f"watchdog restart failed on {host}: {watchdog_output.strip()}"
        )
    print("Do not test with /v1/models. Test with engines() when the load finishes.")
    return out.strip()


def _engine_watchdog_repair_script(watchdog_script: str, specs: list[dict]) -> str:
    """Build an exact watchdog replacement without touching its engine."""
    if not specs:
        raise Refused("no configured watchdogs are available")
    if not watchdog_script.startswith("/"):
        raise Refused("watchdog script path must be absolute")

    lines = [
        "set -euo pipefail",
        f"watchdog_script={shlex.quote(watchdog_script)}",
        "test -x \"$watchdog_script\" || { echo 'watchdog script is missing or not executable' >&2; exit 70; }",
        "watchdog_matches() {",
        "  local pid=$1 command_line environment key value",
        "  case \"$pid\" in ''|*[!0-9]*) return 1;; esac",
        "  [ \"$pid\" -gt 1 ] || return 1",
        "  [ -r \"/proc/$pid/cmdline\" ] && [ -r \"/proc/$pid/environ\" ] || return 1",
        "  command_line=$(cat \"/proc/$pid/cmdline\" 2>/dev/null | tr '\\0' ' ' || true)",
        "  case \"$command_line\" in",
        "    \"bash $watchdog_script\"|\"/bin/bash $watchdog_script\"|\"/usr/bin/bash $watchdog_script\") ;;",
        "    *) return 1;;",
        "  esac",
        "  environment=$(cat \"/proc/$pid/environ\" 2>/dev/null | tr '\\0' '\\n' || true)",
    ]
    for index, spec in enumerate(specs):
        lines.append(f"  if [ \"${{watchdog_identity:-}}\" = {index!r} ]; then")
        for key, value in spec["environment"].items():
            lines.append(
                f"    printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'{key}={value}')} || return 1"
            )
        lines.append("  fi")
    lines.extend(
        [
        "  return 0",
        "}",
        "watchdog_command_matches() {",
        "  local pid=$1 command_line",
        "  case \"$pid\" in ''|*[!0-9]*) return 1;; esac",
        "  [ -r \"/proc/$pid/cmdline\" ] || return 1",
        "  command_line=$(cat \"/proc/$pid/cmdline\" 2>/dev/null | tr '\\0' ' ' || true)",
        "  case \"$command_line\" in \"bash $watchdog_script\"|\"/bin/bash $watchdog_script\"|\"/usr/bin/bash $watchdog_script\") return 0;; esac",
        "  return 1",
        "}",
        "watchdog_legacy_matches() {",
        "  local pid=$1 environment",
        "  watchdog_command_matches \"$pid\" || return 1",
        "  environment=$(cat \"/proc/$pid/environ\" 2>/dev/null | tr '\\0' '\\n' || true)",
        "  printf '%s\\n' \"$environment\" | grep -Eq '^WATCHDOG_IDENTITY=' && return 1",
        ]
    )
    for index, spec in enumerate(specs):
        port = str(spec["environment"]["PORT"])
        lines.append(f"  if [ \"${{watchdog_identity:-}}\" = {index!r} ]; then")
        lines.append(
            f"    if printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote('PORT=' + port)}; then"
        )
        for key, value in spec["environment"].items():
            if key != "WATCHDOG_IDENTITY":
                lines.append(
                    f"      printf '%s\\n' \"$environment\" | grep -Fxq {shlex.quote(f'{key}={value}')} || return 1"
                )
        lines.append("      return 0")
        lines.append("    fi")
        lines.append(
            "    if ! printf '%s\\n' \"$environment\" | grep -Eq '^(PORT|MODEL|START|ENGINE_CONTAINER|WATCHDOG_IDENTITY)'; then return 0; fi"
        )
        lines.append("  fi")
    lines.extend(
        [
            "  return 1",
            "}",
        "stop_watchdog() {",
            "  local pid=$1 pgid",
            "  (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") || return 0",
            "  pgid=$(awk '{print $5}' \"/proc/$pid/stat\" 2>/dev/null || true)",
            "  [ \"$pgid\" = \"$pid\" ] || { echo \"watchdog identity is not its own process group: pid=$pid pgid=$pgid\" >&2; return 1; }",
            "  kill -TERM -- \"-$pid\" 2>/dev/null || true",
            "  for _ in $(seq 1 50); do (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") || break; sleep 0.1; done",
            "  if (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\"); then kill -KILL -- \"-$pid\" 2>/dev/null || true; fi",
            "  for _ in $(seq 1 20); do (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") || break; sleep 0.1; done",
            "  (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") && { echo \"watchdog did not exit: pid=$pid\" >&2; return 1; }",
            "}",
        ]
    )
    for index, spec in enumerate(specs):
        port = str(spec["environment"]["PORT"])
        lines.extend(
            [
                f"watchdog_identity={index}",
                "matches=()",
                "for proc_dir in /proc/[0-9]*; do",
                "  pid=${proc_dir##*/}",
                "  (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") && matches+=(\"$pid\")",
                "done",
                "candidates=()",
                "for proc_dir in /proc/[0-9]*; do",
                "  pid=${proc_dir##*/}",
                "  if watchdog_command_matches \"$pid\"; then",
                f"    if tr '\\0' '\\n' <\"/proc/$pid/environ\" 2>/dev/null | grep -Fxq {shlex.quote(f'PORT={spec["environment"]["PORT"]}')}; then",
                f"      (watchdog_matches \"$pid\" || watchdog_legacy_matches \"$pid\") || {{ echo {shlex.quote(f'watchdog settings mismatch for port {spec["environment"]["PORT"]}')} >&2; exit 74; }}",
                "      candidates+=(\"$pid\")",
                "    elif ! tr '\\0' '\\n' <\"/proc/$pid/environ\" 2>/dev/null | grep -Eq '^PORT='; then",
                f"      if tr '\\0' '\\n' <\"/proc/$pid/environ\" 2>/dev/null | grep -Eq '^(PORT|MODEL|START|ENGINE_CONTAINER|WATCHDOG_IDENTITY)'; then echo {shlex.quote(f'watchdog settings mismatch for port {spec["environment"]["PORT"]}')} >&2; exit 74; fi",
                "    fi",
                "  fi",
                "done",
                f"if [ \"${{#candidates[@]}}\" = 1 ] && [ \"${{#matches[@]}}\" = 0 ]; then echo {shlex.quote(f'watchdog settings mismatch for port {spec["environment"]["PORT"]}')} >&2; exit 74; fi",
                "for pid in \"${matches[@]}\"; do stop_watchdog \"$pid\"; done",
            ]
        )
        env_parts = [f"{key}={shlex.quote(str(value))}" for key, value in spec["environment"].items()]
        lines.append("env " + " ".join(env_parts) + " \\")
        lines.extend(
            [
                "  nohup setsid bash \"$watchdog_script\" >>" + shlex.quote(spec["log"]) + " 2>&1 </dev/null &",
                "new_pid=$!",
                "matched=0",
                "for _ in $(seq 1 50); do",
                "  if watchdog_matches \"$new_pid\"; then matched=1; break; fi",
                "  sleep 0.1",
                "done",
                f"[ \"$matched\" = 1 ] || {{ echo {shlex.quote(f'new watchdog identity was not observed for port {spec["environment"]["PORT"]}')} >&2; exit 72; }}",
                "matches=()",
                "for proc_dir in /proc/[0-9]*; do",
                "  pid=${proc_dir##*/}",
                "  watchdog_matches \"$pid\" && matches+=(\"$pid\")",
                "done",
                f"[ \"${{#matches[@]}}\" = 1 ] || {{ echo {shlex.quote(f'watchdog identity is ambiguous after restart for port {spec["environment"]["PORT"]}:')} \"${{matches[*]}}\" >&2; exit 73; }}",
            ]
        )
    lines.append('echo "reloaded exact engine watchdog(s) without restarting an engine"')
    return "\n".join(lines) + "\n"


def _repair_engine_watchdog(host: str) -> str:
    """Deploy and reload only the active profile's exact watchdog process."""
    _check_host(host)
    if host not in inference_hosts:
        raise Refused(f"{host} is not an engine host. Use one of {inference_hosts}.")
    run = current_run()
    _config_for_run(run)
    settings = _inference_config(host)
    configured_ports = settings.get("ports")
    if not isinstance(configured_ports, list) or not configured_ports:
        raise Refused(f"inference ports for {host} are missing")
    if not all(isinstance(port, int) and 1 <= port <= 65535 for port in configured_ports):
        raise Refused(f"inference ports for {host} are invalid")
    model = glm_model
    container = glm_inference_container
    start_script = f"{runtime_root}/h200/start_inference.sh"
    startup_grace = 600
    specs = []
    for replica, port in enumerate(configured_ports):
        environment = {
            "ENGINE_HOST": host,
            "STARTUP_GRACE": str(startup_grace),
            "MODEL": model,
            "START": start_script,
            "ENGINE_CONTAINER": container,
            "PORT": str(port),
            "WATCHDOG_IDENTITY": f"engine-watchdog:{host}:{run}:{port}",
        }
        specs.append(
            {
                "environment": environment,
                "log": "/root/engine_watchdog.log",
            }
        )
    watchdog_path = f"{runtime_root}/h200/engine_watchdog.sh"
    if not deploy("training_ops/runtime/h200/engine_watchdog.sh", [host]):
        raise RuntimeError(f"watchdog deployment failed on {host}")
    code, output = _ssh(
        host,
        _engine_watchdog_repair_script(watchdog_path, specs),
        timeout=180,
    )
    if code:
        raise RuntimeError(f"watchdog repair failed on {host}: {output.strip()}")
    result = output.strip().splitlines()[-1] if output.strip() else "engine watchdog reloaded"
    print(f"{host}: {result}")
    return result


def _repair_supervisors(config_dir: str | None = None) -> str:
    out = []
    if config_dir is None:
        _config_for_run(current_run())
    else:
        _config_dir(config_dir, require_orchestrator=True)
    for h in inference_hosts:
        start_script = f"{runtime_root}/h200/start_inference.sh"
        code, output = _ssh(
            h,
            _engine_watchdog_identity_script(
                h,
                glm_inference_container,
                f"{runtime_root}/h200/engine_watchdog.sh",
                start_script,
                glm_model,
                600,
                ensure_running=True,
            ),
        )
        if code:
            raise RuntimeError(f"cannot reconcile watchdogs on {h}: {output.strip()}")
        result = output.strip().splitlines()[-1] if output.strip() else "reconciled one canonical engine watchdog"
        out.append(f"{h}: {result}")
    msg = "; ".join(out)
    print(msg)
    print("Use start(run, mode='restart') to replace a run supervisor.")
    return msg


def _mounts_in(script: Path) -> set[str]:
    """Repository paths mounted by one container launcher."""
    return set(re.findall(r'-v "\$deployment/(\S+?)":', script.read_text()))


def mounted_files() -> list[str]:
    """The source paths that the trainer and H200 launchers mount.

    A change to any other file has no effect: the container keeps using the
    copy baked into the image.
    """
    scripts = (
        repo_root / "training_ops" / "runtime" / "china" / "cluster.sh",
        repo_root / "training_ops" / "runtime" / "h200" / "start_orchestrator.sh",
        repo_root / "training_ops" / "runtime" / "h200" / "start_inference.sh",
    )
    paths = sorted(set().union(*(_mounts_in(script) for script in scripts)))
    print("mounted source paths:")
    if paths:
        print("\n".join(paths))
    else:
        print("none")
    return paths


def _mounted_on(path: str, host: str) -> bool:
    if host in inference_hosts:
        scripts = (
            repo_root / "training_ops" / "runtime" / "h200" / "start_orchestrator.sh",
            repo_root / "training_ops" / "runtime" / "h200" / "start_inference.sh",
        )
    else:
        scripts = (repo_root / "training_ops" / "runtime" / "china" / "cluster.sh",)
    return any(path in _mounts_in(script) for script in scripts)


def _repository_root(name: str = "default") -> str:
    """Resolve the configured repository root."""
    if name != "default":
        raise Refused(f"unknown repository {name!r}; choose default")
    root = paths.get("remote_repository")
    if not isinstance(root, str) or not root.startswith("/"):
        raise Refused(f"repository {name!r} is missing from nodes config")
    return root.rstrip("/")


_QUALIFICATION_LAUNCH_ENV_PATTERNS = {
    "PRIME_NSYS_PROFILE": re.compile(r"[01]"),
    "PRIME_CUDA_PROFILE": re.compile(r"[01]"),
    "PRIME_CUDA_PROFILE_RANK": re.compile(r"[0-9]+"),
    "PRIME_CUDA_PROFILE_STEPS": re.compile(r"(?:[2-9]|[1-9][0-9]+)"),
    "PRIME_CUDA_PROFILE_SECONDS": re.compile(
        r"(?:[3-9][0-9]|1[01][0-9]|120)"
    ),
}


def _qualification_launch_env(profile: dict) -> dict[str, str]:
    """Validate the optional, allowlisted trainer launch environment."""
    launch_env = profile.get("launch_env", {})
    if not isinstance(launch_env, dict):
        raise Refused("qualification launch_env must be a mapping")
    validated = {}
    for key, value in launch_env.items():
        if not isinstance(key, str) or key not in _QUALIFICATION_LAUNCH_ENV_PATTERNS:
            raise Refused(f"qualification launch_env key {key!r} is not allowed")
        if not isinstance(value, str) or not value:
            raise Refused(f"qualification launch_env {key} must be a non-empty string")
        if _QUALIFICATION_LAUNCH_ENV_PATTERNS[key].fullmatch(value) is None:
            raise Refused(f"qualification launch_env {key} has an invalid value")
        validated[key] = value
    return validated


def deploy(
    path: str,
    nodes: list[str] | None = None,
    repository: str = "default",
) -> bool:
    """Push one repository file to its hosts and verify the checksum."""
    repository_root = _repository_root(repository)
    if nodes is None:
        if path.startswith("prime-rl/"):
            nodes = [
                host
                for host in dict.fromkeys([*trainer_hosts, *inference_hosts])
                if _mounted_on(path, host)
            ]
        elif path.startswith(("environments/", "evals/")):
            nodes = [orchestrator_host]
        elif path.startswith("training_ops/runtime/h200/") or path == "configs/general-agent/inference.toml":
            nodes = inference_hosts
        elif path.startswith("training_ops/runtime/china/"):
            nodes = trainer_hosts
        elif path.endswith("/trainer.toml"):
            nodes = trainer_hosts
        elif path.endswith("/orchestrator.toml"):
            nodes = [orchestrator_host]
        else:
            raise Refused("destination is not clear; pass nodes explicitly")
        if not nodes:
            raise Refused(f"no configured host mounts {path}")
    nodes = [_check_host(h) for h in nodes]
    local = repo_root / path
    if not local.is_file():
        raise Refused(f"not in the repository: {local}")
    if path.startswith("prime-rl/"):
        unmounted = [host for host in nodes if not _mounted_on(path, host)]
        if unmounted:
            raise Refused(
                f"{path} is not mounted by the launcher for {unmounted}, so the "
                "container would ignore it. Add a -v line to the applicable "
                "launcher first, or pick a mounted path. Call mounted_files() "
                "to list the union."
            )
    want = __import__("hashlib").md5(local.read_bytes()).hexdigest()
    remote = f"{repository_root}/{path}"
    remote_mode = "0755" if local.stat().st_mode & 0o111 else "0644"

    def one(h):
        ssh_options, scp_options, destination = _ssh_destination(h)
        try:
            mkdir = _run_process(
                ssh_command
                + ssh_options
                + [
                    destination,
                    f"mkdir -p {shlex.quote(str(Path(remote).parent))}",
                ],
                stdin=subprocess.DEVNULL,
                timeout=deployment_mkdir_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "mkdir TIMEOUT"
        if mkdir.returncode:
            detail = ((mkdir.stdout or "") + (mkdir.stderr or "")).strip()
            return f"mkdir failed: {detail[:80]}"
        try:
            p = _run_process(
                scp_command
                + scp_options
                + [str(local), f"{destination}:{remote}"],
                stdin=subprocess.DEVNULL,
                timeout=deployment_scp_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "scp TIMEOUT"
        if p.returncode:
            detail = ((p.stdout or "") + (p.stderr or "")).strip()
            return f"scp failed: {detail[:80]}"
        rc, out = _ssh(h, f"chmod {remote_mode} {shlex.quote(remote)}")
        if rc:
            return f"chmod failed: {out.strip()[:80]}"
        rc, got = _ssh(h, f"md5sum {remote} | cut -d' ' -f1")
        return "ok" if got.strip() == want else f"MISMATCH {got.strip()[:8]}"

    res = _fan(nodes, one, workers=deployment_host_workers)
    bad = [h for h, v in res.items() if v != "ok"]
    print(
        f"{path} -> {len(nodes) - len(bad)}/{len(nodes)} nodes"
        + (f", FAILED {bad}" if bad else "")
    )
    if not bad:
        if path.startswith(("prime-rl/", "environments/", "evals/")) or path.endswith("cluster.sh"):
            print(
                "The running container still holds the old file. Recreate that container for this to take effect."
            )
        else:
            print(
                "This runs on the host. Restart the process only if it is a long-running loop."
            )
    return not bad


def _dependency_inventory_for_bundle(
    required_files: list[tuple[str, list[str]]],
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Validate dependency paths and return files plus their host selections."""
    files: dict[str, Path] = {}
    destinations: dict[str, list[str]] = {}
    root = repo_root.resolve()
    for item in required_files:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise Refused("invalid trainer dependency entry")
        path, hosts_for_path = item
        if not isinstance(path, str) or not path:
            raise Refused("trainer dependency path must be a non-empty string")
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or relative.as_posix() != path
            or ".." in relative.parts
            or any(ord(char) < 0x20 for char in path)
            or path == dependency_manifest_name
        ):
            raise Refused(f"unsafe trainer dependency path: {path!r}")
        if isinstance(hosts_for_path, str) or not hosts_for_path:
            raise Refused(f"no configured host for trainer dependency {path}")
        selected_hosts = []
        for host in hosts_for_path:
            if not isinstance(host, str):
                raise Refused(f"invalid host for trainer dependency {path}")
            host = _check_host(host)
            if host not in selected_hosts:
                selected_hosts.append(host)
        local = repo_root / path
        try:
            resolved = local.resolve(strict=True)
        except OSError as error:
            raise Refused(f"cannot resolve trainer dependency {path}: {error}") from error
        if not resolved.is_relative_to(root) or not local.is_file() or local.is_symlink():
            raise Refused(f"trainer dependency is not a regular repository file: {path}")
        files[path] = local
        current = destinations.setdefault(path, [])
        for host in selected_hosts:
            if host not in current:
                current.append(host)
    return files, destinations


def _create_dependency_bundle(
    required_files: list[tuple[str, list[str]]], directory: Path
) -> tuple[Path, dict[str, list[str]]]:
    """Create one tar bundle and its checksum manifest for trainer files."""
    files, destinations = _dependency_inventory_for_bundle(required_files)
    if not files:
        raise Refused("no trainer dependency files")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / dependency_manifest_name
    manifest.write_text(
        "".join(
            f"{_sha256_file(files[path], path)}\t{path}\n"
            for path in sorted(files)
        ),
        encoding="utf-8",
    )
    bundle = directory / f"trainer-dependencies-{directory.name}.tar.gz"
    with tarfile.open(bundle, mode="w:gz") as archive:
        for path in sorted(files):
            archive.add(files[path], arcname=path, recursive=False)
        archive.add(manifest, arcname=dependency_manifest_name, recursive=False)
    return bundle, destinations


def _dependency_remote_script(
    bundle_path: str, repository_root: str, install_paths: list[str]
) -> str:
    """Build the bounded remote extraction, verification, and install script."""
    install_array = " ".join(shlex.quote(path) for path in install_paths)
    return f"""set -euo pipefail
archive={shlex.quote(bundle_path)}
repository={shlex.quote(repository_root)}
manifest_name={shlex.quote(dependency_manifest_name)}
install_paths=({install_array})
temp=""
cleanup() {{
  status=$?
  set +e
  [ -z "$temp" ] || rm -rf -- "$temp"
  rm -f -- "$archive"
  exit "$status"
}}
trap cleanup EXIT
die() {{ echo "trainer dependency bundle: $1" >&2; exit 1; }}
temp=$(mktemp -d /tmp/training-ops-dependencies.XXXXXX)
archive_list="$temp/archive.list"
manifest="$temp/$manifest_name"
expected_paths="$temp/expected.paths"
tar -tf "$archive" > "$archive_list" || die "cannot list archive"

# Reject traversal, absolute names, and directory entries before extraction.
while IFS= read -r member; do
  case "$member" in
    ""|/*|..|../*|*/..|*/../*|*/)
      die "unsafe archive member: $member"
      ;;
  esac
done < "$archive_list"
manifest_count=$(grep -Fxc -- "$manifest_name" "$archive_list" || true)
[ "$manifest_count" -eq 1 ] || die "archive must contain one checksum manifest"
tar -xOf "$archive" "$manifest_name" > "$manifest" || die "cannot extract checksum manifest"
[ -f "$manifest" ] && [ ! -L "$manifest" ] || die "checksum manifest is not a regular file"

: > "$expected_paths"
expected_count=0
while IFS="$(printf '\\t')" read -r digest member extra; do
  [ -n "$digest" ] && [ -n "$member" ] && [ -z "${{extra:-}}" ] || die "invalid checksum manifest line"
  if ! printf '%s\\n' "$digest" | grep -Eq '^[0-9a-f]{{64}}$'; then
    die "invalid checksum for $member"
  fi
  case "$member" in
    ""|/*|..|../*|*/..|*/../*|*/|"$manifest_name")
      die "unsafe manifest path: $member"
      ;;
  esac
  if grep -Fqx -- "$member" "$expected_paths"; then
    die "duplicate manifest path: $member"
  fi
  printf '%s\\n' "$member" >> "$expected_paths"
  expected_count=$((expected_count + 1))
done < "$manifest"
[ "$expected_count" -gt 0 ] || die "empty checksum manifest"
archive_count=$(wc -l < "$archive_list")
[ "$archive_count" -eq $((expected_count + 1)) ] || die "archive member count mismatch"

while IFS= read -r member; do
  [ "$member" = "$manifest_name" ] && continue
  grep -Fqx -- "$member" "$expected_paths" || die "unlisted archive member: $member"
done < "$archive_list"
while IFS= read -r member; do
  member_count=$(grep -Fxc -- "$member" "$archive_list" || true)
  [ "$member_count" -eq 1 ] || die "missing or duplicate archive path: $member"
done < "$expected_paths"

tar -xpf "$archive" -C "$temp" --no-same-owner --no-overwrite-dir || die "cannot extract archive"
while IFS="$(printf '\\t')" read -r digest member extra; do
  source="$temp/$member"
  [ -f "$source" ] && [ ! -L "$source" ] || die "missing extracted path: $member"
  got=$(sha256sum -- "$source" | cut -d' ' -f1) || die "cannot checksum $member"
  [ "$got" = "$digest" ] || die "checksum mismatch: $member"
done < "$manifest"

for member in "${{install_paths[@]}}"; do
  grep -Fqx -- "$member" "$expected_paths" || die "install path is not listed: $member"
  source="$temp/$member"
  destination="$repository/$member"
  mkdir -p -- "$(dirname -- "$destination")" || die "cannot create destination for $member"
  [ ! -L "$destination" ] || die "destination is a symlink: $member"
  cp -p -- "$source" "$destination" || die "cannot install $member"
  if [ -x "$source" ]; then
    chmod 0755 -- "$destination" || die "cannot set executable mode: $member"
  else
    chmod 0644 -- "$destination" || die "cannot set file mode: $member"
  fi
  [ -f "$destination" ] && [ ! -L "$destination" ] || die "invalid installed path: $member"
  source_digest=$(sha256sum -- "$source" | cut -d' ' -f1) || die "cannot checksum $member"
  got=$(sha256sum -- "$destination" | cut -d' ' -f1) || die "cannot verify $member"
  [ "$got" = "$source_digest" ] || die "installed checksum mismatch: $member"
done
echo "installed ${{#install_paths[@]}} trainer dependency files"
"""


def _dependency_checksum_script(
    repository_root: str, expected: list[tuple[str, str, str]]
) -> str:
    """Build one read-only remote check for the expected dependency files."""
    lines = [
        "set -euo pipefail",
        f"repository={shlex.quote(repository_root)}",
    ]
    for path, digest, mode in expected:
        lines.extend(
            [
                f"file=\"$repository\"/{shlex.quote(path)}",
                '[ -f "$file" ] && [ ! -L "$file" ]',
                'got=$(sha256sum -- "$file" | cut -d\' \' -f1)',
                f'[ "$got" = {shlex.quote(digest)} ]',
                'got_mode=$(stat -c %a -- "$file")',
                f'[ "$got_mode" = {shlex.quote(mode)} ]',
            ]
        )
    lines.append("printf '%s\\n' 'checksums match'")
    return "\n".join(lines)


def _validated_dependency_remote_archive(bundle: Path) -> str:
    """Return the exact safe temporary archive path used on a trainer host."""
    remote_archive = f"/tmp/{bundle.name}"
    if _dependency_remote_archive_pattern.fullmatch(remote_archive) is None:
        raise Refused(f"invalid trainer dependency temporary archive: {remote_archive}")
    return remote_archive


def _retry_dependency_copy(label: str, copy, cleanup) -> None:
    """Run one dependency copy at most three times."""
    last_detail = "failed"
    for attempt in range(1, 4):
        try:
            code, detail = copy()
        except subprocess.TimeoutExpired as error:
            code = 1
            detail = error.stderr or "timed out"
        except OSError as error:
            code = 1
            detail = str(error)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        last_detail = str(detail or "failed").strip()
        if code == 0:
            return
        cleanup()
        print(
            f"trainer dependency bundle: {label} attempt {attempt}/3 failed: "
            f"{last_detail[:160]}",
            file=sys.stderr,
        )
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError(f"trainer dependency bundle: {label} failed: {last_detail[:160]}")


def _deploy_dependencies(required_files: list[tuple[str, list[str]]]) -> None:
    """Deploy trainer dependencies as one verified archive per host."""
    if not required_files:
        print("trainer dependency deployment: no files")
        return
    repository_root = _repository_root()
    with tempfile.TemporaryDirectory(prefix="training-ops-dependencies-") as work:
        bundle, destinations = _create_dependency_bundle(required_files, Path(work))
        manifest_path = Path(work) / dependency_manifest_name
        manifest_entries = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, path = line.split("\t", 1)
            manifest_entries[path] = digest
        hosts_for_bundle = sorted(
            {host for selected_hosts in destinations.values() for host in selected_hosts}
        )
        if not hosts_for_bundle:
            raise Refused("no configured trainer hosts for dependency bundle")
        print(
            f"trainer dependency bundle: created {len(destinations)} files "
            f"({bundle.stat().st_size} bytes)"
        )
        workers = min(deployment_bundle_workers, len(hosts_for_bundle))
        print(
            f"trainer dependency bundle: uploading to {len(hosts_for_bundle)} hosts "
            f"with {workers} parallel workers"
        )
        remote_archive = _validated_dependency_remote_archive(bundle)
        trainer_targets = [
            host for host in hosts_for_bundle if host in trainer_hosts
        ]
        relay_base = remote_archive.removesuffix(".tar.gz")
        relay_archive = relay_base + "-relay.tar.gz"
        edge_relay_archive = relay_base + "-edge-relay.tar.gz"
        if any(
            _dependency_remote_archive_pattern.fullmatch(path) is None
            for path in (relay_archive, edge_relay_archive)
        ):
            raise Refused("invalid trainer dependency relay path")

        if trainer_targets:
            _, scp_options, destination = _ssh_destination(orchestrator_host)
            upload_command = scp_command + scp_options + [
                str(bundle),
                f"{destination}:{edge_relay_archive}",
            ]

            def upload_to_edge() -> tuple[int, str]:
                result = _run_process(
                    upload_command,
                    stdin=subprocess.DEVNULL,
                    timeout=deployment_bundle_scp_timeout_seconds,
                )
                return result.returncode, result.stderr or result.stdout

            _retry_dependency_copy(
                f"upload to {orchestrator_host}",
                upload_to_edge,
                lambda: _ssh(
                    orchestrator_host,
                    f"rm -f -- {shlex.quote(edge_relay_archive)}",
                    timeout=deployment_mkdir_timeout_seconds,
                ),
            )

            def relay_to_trainer() -> tuple[int, str]:
                return _ssh(
                    orchestrator_host,
                    "scp -q -o BatchMode=yes -o ConnectTimeout=25 "
                    "-o StrictHostKeyChecking=accept-new -- "
                    f"{shlex.quote(edge_relay_archive)} "
                    f"{shlex.quote('china-trainer:' + relay_archive)}",
                    timeout=deployment_bundle_scp_timeout_seconds,
                )

            _retry_dependency_copy(
                f"relay to {trainer_head}",
                relay_to_trainer,
                lambda: _ssh(
                    trainer_head,
                    f"rm -f -- {shlex.quote(relay_archive)}",
                    timeout=deployment_mkdir_timeout_seconds,
                ),
            )
            relay_code, relay_output = _ssh(
                trainer_head,
                f"test $(stat -c %s -- {shlex.quote(relay_archive)}) -eq {bundle.stat().st_size}",
                timeout=deployment_mkdir_timeout_seconds,
            )
            if relay_code:
                raise RuntimeError(
                    "trainer dependency relay size check failed: "
                    + relay_output.strip()[:160]
                )
            print(
                f"trainer dependency bundle: relayed through {orchestrator_host} "
                f"and staged once on {trainer_head} "
                f"for {len(trainer_targets)} trainer hosts"
            )

        def one(host: str) -> str:
            selected = [
                path for path, selected_hosts in destinations.items() if host in selected_hosts
            ]
            expected = [
                (
                    path,
                    manifest_entries[path],
                    "0755" if (repo_root / path).stat().st_mode & 0o111 else "0644",
                )
                for path in selected
            ]
            try:
                code, output = _ssh(
                    host,
                    _dependency_checksum_script(repository_root, expected),
                    timeout=deployment_bundle_scp_timeout_seconds,
                )
            except Exception:
                code, output = 1, ""
            if code == 0 and output.strip() == "checksums match":
                return "skipped"

            def remove_partial_archive() -> None:
                try:
                    _ssh(
                        host,
                        f"rm -f -- {shlex.quote(remote_archive)}",
                        timeout=deployment_mkdir_timeout_seconds,
                    )
                except Exception:
                    pass

            for attempt in range(1, 4):
                result = "failed"
                stderr = ""
                ssh_options, scp_options, destination = _ssh_destination(host)
                try:
                    if host in trainer_hosts:
                        if host == trainer_head:
                            relay_command = (
                                f"cp -- {shlex.quote(relay_archive)} "
                                f"{shlex.quote(remote_archive)}"
                            )
                        else:
                            peer = hosts["trainer_lan"].get(host)
                            if not isinstance(peer, str) or not peer:
                                raise RuntimeError(
                                    f"trainer LAN address is missing for {host}"
                                )
                            relay_command = (
                                "scp -q -o BatchMode=yes -o ConnectTimeout=10 "
                                "-o StrictHostKeyChecking=accept-new -- "
                                f"{shlex.quote(relay_archive)} "
                                f"{shlex.quote(peer + ':' + remote_archive)}"
                            )
                        relay_code, relay_output = _ssh(
                            trainer_head,
                            relay_command,
                            timeout=deployment_bundle_scp_timeout_seconds,
                        )
                        uploaded = subprocess.CompletedProcess(
                            args=relay_command,
                            returncode=relay_code,
                            stdout="",
                            stderr=relay_output,
                        )
                    else:
                        if attempt == 1:
                            upload_command = scp_command + scp_options + [
                                str(bundle),
                                f"{destination}:{remote_archive}",
                            ]
                            upload_stdin = subprocess.DEVNULL
                        else:
                            upload_command = ssh_command + ssh_options + [
                                destination,
                                f"umask 077; cat > {shlex.quote(remote_archive)}",
                            ]
                            upload_stdin = bundle.open("rb")
                        try:
                            uploaded = _run_process(
                                upload_command,
                                stdin=upload_stdin,
                                timeout=deployment_bundle_scp_timeout_seconds,
                            )
                        finally:
                            if upload_stdin is not subprocess.DEVNULL:
                                upload_stdin.close()
                except subprocess.TimeoutExpired as error:
                    stderr = error.stderr or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    result = "scp TIMEOUT" if attempt == 1 else "ssh stream TIMEOUT"
                    remove_partial_archive()
                except OSError as error:
                    stderr = str(error)
                    result = "scp failed" if attempt == 1 else "ssh stream failed"
                    remove_partial_archive()
                else:
                    if uploaded.returncode:
                        stderr = uploaded.stderr or ""
                        if isinstance(stderr, bytes):
                            stderr = stderr.decode(errors="replace")
                        result = "scp failed" if attempt == 1 else "ssh stream failed"
                        remove_partial_archive()
                    else:
                        try:
                            code, output = _ssh(
                                host,
                                _dependency_remote_script(
                                    remote_archive, repository_root, selected
                                ),
                                timeout=deployment_scp_timeout_seconds,
                            )
                        except Exception as error:
                            stderr = str(error)
                            result = "remote failed"
                            remove_partial_archive()
                        else:
                            if code:
                                stderr = output
                                result = f"remote failed ({code})"
                                remove_partial_archive()
                            else:
                                return "ok"

                print(
                    f"trainer dependency bundle: {host} attempt {attempt}/3 failed "
                    f"({result}); stderr:",
                    file=sys.stderr,
                )
                if stderr:
                    print(
                        stderr,
                        file=sys.stderr,
                        end="" if stderr.endswith("\n") else "\n",
                    )
                else:
                    print("<empty>", file=sys.stderr)
                if attempt < 3:
                    time.sleep(attempt)
            return result

        failures = []
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(one, host): host for host in hosts_for_bundle}
                for future in as_completed(futures):
                    host = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = f"failed: {error}"
                    completed += 1
                    if result not in {"ok", "skipped"}:
                        failures.append(f"{host}: {result}")
                    print(
                        f"trainer dependency bundle: {host} "
                        f"[{result}] ({completed}/{len(hosts_for_bundle)})"
                    )
        finally:
            if trainer_targets:
                _ssh(
                    trainer_head,
                    f"rm -f -- {shlex.quote(relay_archive)}",
                    timeout=deployment_mkdir_timeout_seconds,
                )
                _ssh(
                    orchestrator_host,
                    f"rm -f -- {shlex.quote(edge_relay_archive)}",
                    timeout=deployment_mkdir_timeout_seconds,
                )
        if failures:
            raise RuntimeError("failed to deploy dependency bundle: " + "; ".join(failures))
        print("trainer dependency bundle: complete")


def _trainer_dependency_inventory(
    relative_config: str, nodes: list[str]
) -> list[tuple[str, list[str]]]:
    """Return files required by the trainer launcher on the selected nodes."""
    inventory: dict[str, list[str]] = {}

    def add(path: str, destinations: list[str]) -> None:
        destinations = list(dict.fromkeys(_check_host(host) for host in destinations))
        if not destinations:
            raise Refused(f"no configured host for trainer dependency {path}")
        current = inventory.setdefault(path, [])
        for host in destinations:
            if host not in current:
                current.append(host)

    def add_files(root: str, destinations: list[str]) -> None:
        base = repo_root / root
        if not base.is_dir():
            raise Refused(f"missing trainer dependency directory: {base}")
        for local in sorted(base.rglob("*")):
            if "__pycache__" in local.parts or local.suffix in {".pyc", ".pyo"}:
                continue
            if local.is_file():
                add(local.relative_to(repo_root).as_posix(), destinations)

    add_files(
        "training_ops/runtime/china",
        nodes,
    )
    for path in mounted_files():
        if not path.startswith("prime-rl/"):
            continue
        destinations = [host for host in nodes if _mounted_on(path, host)]
        if destinations:
            add(path, destinations)
    add(relative_config, nodes)

    missing = [path for path in inventory if not (repo_root / path).is_file()]
    if missing:
        raise Refused("missing trainer dependencies: " + ", ".join(missing))
    return list(inventory.items())


def _start_dependency_inventory(
    selected: dict, nodes: list[str]
) -> list[tuple[str, list[str]]]:
    """Return and validate the files needed before a run's containers start."""
    inventory: dict[str, list[str]] = {}

    def add(path: str, destinations: list[str]) -> None:
        destinations = list(dict.fromkeys(_check_host(host) for host in destinations))
        if not destinations:
            raise Refused(f"no configured host for start dependency {path}")
        current = inventory.setdefault(path, [])
        for host in destinations:
            if host not in current:
                current.append(host)

    def add_files(root: str, destinations: list[str], suffix: str | None = None) -> None:
        base = repo_root / root
        if not base.is_dir():
            raise Refused(f"missing start dependency directory: {base}")
        for local in sorted(base.rglob("*")):
            if "__pycache__" in local.parts or local.suffix in {".pyc", ".pyo"}:
                continue
            if local.is_file() and (suffix is None or local.suffix == suffix):
                add(local.relative_to(repo_root).as_posix(), destinations)

    trainer_destinations = list(dict.fromkeys([*nodes, trainer_head]))
    add_files(
        "training_ops/runtime/china",
        trainer_destinations,
    )
    add_files(
        "training_ops/runtime/h200",
        inference_hosts,
    )

    configured_hosts = list(dict.fromkeys([*trainer_hosts, *inference_hosts]))
    add("configs/nodes.toml", configured_hosts)
    add("configs/general-agent/inference.toml", inference_hosts)
    for path in mounted_files():
        if not path.startswith("prime-rl/"):
            continue
        destinations = [host for host in configured_hosts if _mounted_on(path, host)]
        add(path, destinations)

    add(selected["trainer"], nodes)
    add(selected["orchestrator"], [orchestrator_host])
    with (repo_root / selected["orchestrator"]).open("rb") as stream:
        train_envs = tomllib.load(stream).get("train", {}).get("env", [])
    environment_sources = {
        "general-agent": ("environments/general-agent/general_agent",),
        "decensor-v1": ("environments/decensor/src/decensor_v1",),
    }
    if not isinstance(train_envs, list):
        raise Refused("orchestrator train.env must be a list")
    for environment in train_envs:
        if not isinstance(environment, dict):
            raise Refused("orchestrator train.env entries must be mappings")
        environment_id = environment.get("id")
        if environment_id is None:
            nested_environment = environment.get("env")
            if isinstance(nested_environment, dict):
                environment_id = nested_environment.get("id")
                if environment_id is None:
                    taskset = nested_environment.get("taskset")
                    if isinstance(taskset, dict):
                        environment_id = taskset.get("id")
        if not isinstance(environment_id, str) or not environment_id:
            raise Refused("orchestrator train.env entry has no id")
        source = environment_sources.get(environment_id)
        if source is None:
            raise Refused(f"unsupported orchestrator environment id: {environment_id}")
        for source_root in source:
            add_files(source_root, [orchestrator_host], suffix=".py")

    missing = [path for path in inventory if not (repo_root / path).is_file()]
    if missing:
        raise Refused("missing start dependencies: " + ", ".join(missing))
    return list(inventory.items())


def _copy_resume_adapter(path: str, nodes: list[str]) -> None:
    """Copy one adapter from the trainer head to missing trainer nodes."""
    if not path.startswith(f"{run_root}/"):
        raise Refused(f"resume adapter must be below {run_root}")
    targets = [host for host in nodes if host != trainer_head]
    if not targets:
        return
    target_addresses = " ".join(f"10.92.20.{host.removeprefix('n')}" for host in targets)
    quoted_path = shlex.quote(path)
    script = f"""
set -euo pipefail
source_file={quoted_path}
test -f "$source_file"
digest=$(sha256sum "$source_file" | cut -d' ' -f1)
copy_one() {{
  target=$1
  final=$source_file
  directory=$(dirname "$final")
  partial="$directory/.adapter_model.$digest.partial"
  remote_digest=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target" \
    "sha256sum '$final' 2>/dev/null | cut -d' ' -f1" || true)
  [ "$remote_digest" = "$digest" ] && return 0
  [ -z "$remote_digest" ] || {{
    echo "$target has a different adapter at $final" >&2
    return 1
  }}
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target" \
    "mkdir -p '$directory'"
  transfer=rsync
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target" \
    "command -v rsync >/dev/null" || transfer=scp
  for attempt in 1 2 3; do
    if [ "$transfer" = rsync ]; then
      rsync -a --partial --append-verify --timeout=120 \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
        "$source_file" "$target:$partial" && copied=1 || copied=0
    else
      scp -p -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
        "$source_file" "$target:$partial" && copied=1 || copied=0
    fi
    if [ "$copied" = 1 ]; then
      remote_digest=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target" \
        "sha256sum '$partial' | cut -d' ' -f1")
      if [ "$remote_digest" = "$digest" ]; then
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target" \
          "mv '$partial' '$final'"
        return 0
      fi
    fi
    echo "$target adapter copy attempt $attempt failed" >&2
    sleep 5
  done
  return 1
}}
fail=0
for target in {target_addresses}; do
  copy_one "$target" || fail=1
done
exit "$fail"
"""
    code, output = _ssh(trainer_head, script, timeout=1800)
    if code:
        detail = " | ".join(output.strip().splitlines()[-5:]) if output.strip() else "no output"
        raise RuntimeError(f"resume adapter copy failed: {detail}")


def _cold_run_paths(run: str) -> tuple[str, str, str]:
    """Return the validated run root, run directory, and service directory."""
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("run name can contain only letters, digits, '_' and '-'")
    if not isinstance(run_root, str):
        raise Refused(f"configured runs path is unsafe: {run_root!r}")
    root = Path(run_root)
    if not root.is_absolute() or root == Path("/"):
        raise Refused(f"configured runs path is unsafe: {run_root!r}")
    # Keep this lexical check separate from Path.resolve(): the path is checked
    # again on each remote host, where a symlink could otherwise escape runs/.
    if any(part in {"", ".", ".."} for part in root.parts):
        raise Refused(f"configured runs path is unsafe: {run_root!r}")
    run_parent = root / run
    run_dir = run_parent / f"run_{run}"
    if run_parent.parent != root or run_dir.parent != run_parent or run_dir == root:
        raise Refused(f"run path is unsafe: {run!r}")
    return str(root), str(run_parent), str(run_dir)


def _cold_cleanup_script(run: str, *, stop_only: bool = False) -> str:
    """Build a bounded remote cleanup for one configured run only."""
    runs_path, run_parent, run_dir = _cold_run_paths(run)
    sync_script = f"{runtime_root}/china/sync_microbatches.sh"
    stop_only_value = 1 if stop_only else 0
    return f"""set -euo pipefail
runs_path={shlex.quote(runs_path)}
run_parent={shlex.quote(run_parent)}
run_dir={shlex.quote(run_dir)}
sync_script={shlex.quote(sync_script)}
stop_only={stop_only_value}

# A PID file alone is not enough: the PID can be stale or reused. Require the
# exact publisher output directory and sync command before sending a signal.
sync_process_matches() {{
  local pid=$1 expected_run_dir=$2 command_line
  case "$pid" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$pid" -gt 1 ] || return 1
  [ -r "/proc/$pid/environ" ] || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  cat "/proc/$pid/environ" 2>/dev/null | tr '\\0' '\\n' |
    grep -Fxq "run_dir=$expected_run_dir" || return 1
  command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\\0' ' ') || return 1
  case " $command_line " in
    *" $sync_script "*) return 0 ;;
    *) return 1 ;;
  esac
}}

stop_sync_processes() {{
  local pid_file pid stale_run_dir command_line proc_dir
  stop_one_sync_process() {{
    local pid=$1 expected_run_dir=$2
    # Re-read the process identity immediately before signalling. This also
    # protects against PID reuse after the /proc scan.
    sync_process_matches "$pid" "$expected_run_dir" || return 0
    command_line=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\\0' ' ' || true)
    echo "cold cleanup: stopping microbatch sync pid=$pid run=$expected_run_dir command=$command_line"
    # sync_microbatches.sh is started under setsid, so its PID is also the
    # process-group ID. The single-process fallback remains PID exact and is
    # only reached after the same exact identity check.
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      sync_process_matches "$pid" "$expected_run_dir" || break
      sleep 0.1
    done
    if sync_process_matches "$pid" "$expected_run_dir"; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    # Remove only this run's control files after the process has exited.
    rm -f -- "$expected_run_dir/.microbatch-sync.pid" "$expected_run_dir/.microbatch-sync.lock"
  }}

  # The PID file is an optimisation only. A process can outlive an unlink or
  # lose the file during a failed startup, so inspect every configured publisher
  # directory and every /proc entry. The matcher above is exact; this is not a
  # name-based kill.
  scan_run_dir() {{
    local expected_run_dir=$1 proc_dir pid
    [ -d "$expected_run_dir" ] || return 0
    [ -L "$expected_run_dir" ] && return 0
    for proc_dir in /proc/[0-9]*; do
      pid=${{proc_dir##*/}}
      sync_process_matches "$pid" "$expected_run_dir" &&
        stop_one_sync_process "$pid" "$expected_run_dir"
    done
    return 0
  }}
  scan_run_dir "$run_parent"
  scan_run_dir "$run_dir"
  for service_dir in "$runs_path"/*/run_*; do
    [ "$service_dir" = "$run_dir" ] && continue
    case "$service_dir" in
      "$runs_path"/*/run_*) scan_run_dir "$service_dir" ;;
    esac
  done
  for pid_file in "$run_parent/.microbatch-sync.pid" "$runs_path"/*/run_*/.microbatch-sync.pid; do
    [ -f "$pid_file" ] || continue
    stale_run_dir=${{pid_file%/.microbatch-sync.pid}}
    case "$stale_run_dir" in
      "$runs_path"/*/run_*) ;;
      *) continue ;;
    esac
    # Do not follow a symlink while deriving a process identity.
    [ -L "$stale_run_dir" ] && continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    sync_process_matches "$pid" "$stale_run_dir" && stop_one_sync_process "$pid" "$stale_run_dir"
  done
  return 0
}}

# Refuse symlinked path components before removing anything.  A missing run is
# a valid cold-start state and must not cause the parent directory to be made.
case "$run_parent" in
  "$runs_path"/*) ;;
  *) echo "cold cleanup refused: run is outside configured runs path" >&2; exit 1 ;;
esac
case "$run_dir" in
  "$run_parent"/run_*) ;;
  *) echo "cold cleanup refused: service directory is outside run" >&2; exit 1 ;;
esac
if [ -L "$runs_path" ] || [ -L "$run_parent" ] || [ -L "$run_dir" ]; then
  echo "cold cleanup refused: configured run path is a symlink" >&2
  exit 1
fi
stop_sync_processes
remaining_sync=""
for expected_run_dir in "$run_parent" "$run_dir"; do
  [ -d "$expected_run_dir" ] || continue
  [ -L "$expected_run_dir" ] && continue
  for proc_dir in /proc/[0-9]*; do
    pid=${{proc_dir##*/}}
    if sync_process_matches "$pid" "$expected_run_dir"; then
      remaining_sync="$remaining_sync $pid:$expected_run_dir"
    fi
  done
done
if [ -n "$remaining_sync" ]; then
  echo "cold cleanup: microbatch sync remains:$remaining_sync" >&2
  exit 1
fi
if [ "$stop_only" = 1 ]; then
  echo "cold cleanup: microbatch publishers stopped for $run_parent"
  exit 0
fi
if [ ! -e "$run_parent" ]; then
  echo "cold cleanup: no state at $run_parent"
  exit 0
fi
[ -d "$run_parent" ] || {{ echo "cold cleanup refused: run path is not a directory" >&2; exit 1; }}
# Single-run trainers keep checkpoints at output_dir/checkpoints. Remove this
# exact run's copy even when the service subdirectory was never created.
remove_dir() {{
  local path=$1 label=$2
  if [ -d "$path" ] || [ -L "$path" ]; then
    rm -rf -- "$path"
    echo "cold cleanup: removed $label ($path)"
    removed=$((removed + 1))
  fi
}}
removed=0
remove_dir "$run_parent/checkpoints" "trainer checkpoints and optimizer state"
if [ ! -e "$run_dir" ]; then
  echo "cold cleanup: no service state at $run_dir"
  exit 0
fi
[ -d "$run_dir" ] || {{ echo "cold cleanup refused: service path is not a directory" >&2; exit 1; }}

remove_file() {{
  local path=$1
  if [ -f "$path" ] || [ -L "$path" ]; then
    rm -f -- "$path"
    echo "cold cleanup: removed $path"
    removed=$((removed + 1))
  fi
}}
# These directories contain only generated run state.  In particular,
# rollouts includes train_rollouts.bin and its .delivered/partial state;
# broadcasts contains old adapters and STABLE markers; checkpoints contains
# optimizer/progress state.  All paths remain below this exact run.
remove_dir "$run_parent/rollouts" "trainer rollout batches and micro-batches"
remove_dir "$run_dir/rollouts" "rollout train_rollouts.bin batches and delivery markers"
remove_dir "$run_dir/broadcasts" "broadcast adapters and STABLE markers"
remove_dir "$run_dir/checkpoints" "trainer checkpoints and optimizer state"
remove_dir "$run_dir/control" "orchestrator control state"
remove_dir "$run_dir/.incoming" "in-flight delivery state"
remove_dir "$run_dir/hf-cache" "run-local model/data cache"
for tracking_dir in "$run_dir"/run-20*; do
  [ -e "$tracking_dir" ] || continue
  [ -d "$tracking_dir" ] && [ ! -L "$tracking_dir" ] || {{
    echo "cold cleanup refused invalid tracking path: $tracking_dir" >&2
    exit 1
  }}
  remove_dir "$tracking_dir" "old tracking data"
done
remove_file "$run_dir/final_summary.json"
remove_file "$run_dir/rollout-metrics.jsonl"
remove_file "$run_dir/.batch-sync.pid"
remove_file "$run_dir/.batch-sync.lock"
remove_file "$run_dir/.microbatch-sync.pid"
remove_file "$run_dir/.microbatch-sync.lock"
remove_file "$run_dir/.microbatch-sync.marker"
remove_file "$run_dir/.microbatch-sync.marker.tmp"
remove_file "$run_parent/parity-step1.json"
echo "cold cleanup: complete ($removed entries) target=$run_dir"
"""


def _cold_start_cleanup(run: str, nodes: list[str]) -> None:
    """Stop old microbatch publishers, then remove stale state for one run."""
    _cold_run_paths(run)  # Validate before any host action.
    targets = list(dict.fromkeys([*nodes, orchestrator_host]))
    targets = [_check_host(host) for host in targets]

    def phase(script: str, label: str, error_label: str) -> None:
        results = _fan(
            targets,
            lambda host: _ssh(host, script, timeout=300),
        )
        failures = []
        for host, (code, output) in results.items():
            detail = " | ".join(line.strip() for line in output.splitlines() if line.strip())
            if code:
                failures.append(f"{host}: {detail or f'exit {code}'}")
            else:
                print(f"cold cleanup {run} {label} on {host}: {detail or 'complete'}")
        if failures:
            raise RuntimeError(f"{error_label}: " + "; ".join(failures))

    phase(
        _cold_cleanup_script(run, stop_only=True),
        "microbatch publishers stopped",
        "cold publisher stop failed",
    )
    phase(
        _cold_cleanup_script(run),
        "state removed",
        "cold cleanup failed",
    )


def _initial_adapter_cache_identity(relative_config: str, host: str) -> dict[str, int | str] | None:
    """Read the configured initial adapter identity without hashing its model."""
    config_path = repo_root / relative_config
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Refused(f"cannot read trainer config {config_path}: {error}") from error
    initial = config.get("model", {}).get("lora", {}).get("initial_adapter_path")
    if initial is None:
        return None
    if not isinstance(initial, str) or not initial.startswith(f"{run_root}/"):
        raise Refused(
            "initial adapter path must be below the configured runs path for cache reuse"
        )
    quoted = shlex.quote(initial)
    script = f"""set -euo pipefail
configured={quoted}
if [ -d \"$configured\" ]; then
  adapter=\"$configured/adapter_model.safetensors\"
  adapter_dir=\"$configured\"
else
  adapter=\"$configured\"
  adapter_dir=$(dirname \"$configured\")
fi
config_file=\"$adapter_dir/adapter_config.json\"
test -f \"$adapter\"
test -f \"$adapter.sha256\"
test -f \"$config_file\"
target_sha=$(sed -n '1p' \"$adapter.sha256\" | tr -d '[:space:]')
target_size=$(stat -c '%s' \"$adapter\")
config_sha=$(sha256sum \"$config_file\" | cut -d' ' -f1)
config_size=$(stat -c '%s' \"$config_file\")
printf '%s\\n%s\\n%s\\n%s\\n' \"$target_sha\" \"$target_size\" \"$config_sha\" \"$config_size\"
"""
    code, output = _ssh(host, script, timeout=30)
    if code:
        raise Refused(
            f"initial adapter identity is unavailable on {host}: "
            + (output.strip() or f"exit {code}")
        )
    fields = output.strip().splitlines()
    if len(fields) != 4 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise Refused(f"initial adapter identity is invalid on {host}")
    try:
        target_size = int(fields[1])
        config_size = int(fields[3])
    except ValueError as error:
        raise Refused(f"initial adapter sizes are invalid on {host}") from error
    if target_size <= 0 or config_size < 0 or re.fullmatch(r"[0-9a-f]{64}", fields[2]) is None:
        raise Refused(f"initial adapter identity is invalid on {host}")
    return {
        "target_sha256": fields[0],
        "target_size": target_size,
        "config_sha256": fields[2],
        "config_size": config_size,
    }


def _adapter_cache_cleanup_script(
    preserve_identity: dict[str, int | str] | None = None,
) -> str:
    """Remove unused adapter data before a cold run."""
    cache = paths["adapter_cache"].rstrip("/")
    if not cache.startswith("/") or cache in {"", "/"}:
        raise Refused("adapter cache path is unsafe")
    identity = preserve_identity or {}
    target_sha = identity.get("target_sha256", "")
    target_size = identity.get("target_size", "")
    config_sha = identity.get("config_sha256", "")
    config_size = identity.get("config_size", "")
    if identity and (
        not isinstance(target_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", target_sha) is None
        or type(target_size) is not int
        or target_size <= 0
        or not isinstance(config_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha) is None
        or type(config_size) is not int
        or config_size < 0
    ):
        raise Refused("initial adapter cache identity is invalid")
    return f"""set -euo pipefail
cache={shlex.quote(cache)}
preserve_target_sha={shlex.quote(str(target_sha))}
preserve_target_size={shlex.quote(str(target_size))}
preserve_config_sha={shlex.quote(str(config_sha))}
preserve_config_size={shlex.quote(str(config_size))}
[ -d "$cache" ] || {{ echo "adapter cache does not exist: $cache" >&2; exit 31; }}
[ ! -L "$cache" ] || {{ echo "adapter cache is a symbolic link" >&2; exit 32; }}
removed=0
preserved=0
for path in "$cache"/*; do
  [ -e "$path" ] || continue
  name=${{path##*/}}
  case "$name" in
    adapter-staging.lock) continue ;;
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*|\
    *.partial|*.parts|*.assembling|*.retired-*) ;;
    *) echo "unknown adapter cache entry: $path" >&2; exit 33 ;;
  esac
  base=${{name%%.*}}
  base=${{base%%.retired-*}}
  case "$base" in
    *[!0-9a-f]*|'') echo "invalid adapter cache entry: $path" >&2; exit 34 ;;
  esac
  [ "${{#base}}" -eq 64 ] || {{ echo "invalid adapter cache entry: $path" >&2; exit 34; }}
  if [ "$name" = "$base" ] && [ -n "$preserve_target_sha" ] && [ "$preserved" -eq 0 ]; then
    [ ! -L "$path" ] || {{ echo "adapter cache entry is a symbolic link: $path" >&2; exit 36; }}
    [ -d "$path" ] || {{ echo "adapter cache entry is not a directory: $path" >&2; exit 37; }}
    if python3 - "$path" "$preserve_target_sha" "$preserve_target_size" "$preserve_config_sha" "$preserve_config_size" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target_sha, target_size, config_sha, config_size = sys.argv[2:]
try:
    ready = root / "READY"
    if ready.is_symlink() or ready.read_text(encoding="ascii") != "ready\\n":
        raise ValueError
    manifest_file = root / "manifest.json"
    if manifest_file.is_symlink():
        raise ValueError
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("source_url"), str):
        raise ValueError
    files = manifest["files"]
    if set(files) != {{"adapter_model.safetensors", "adapter_config.json"}}:
        raise ValueError
    target = files["adapter_model.safetensors"]
    config = files["adapter_config.json"]
    sidecar = root / "adapter_model.safetensors.sha256"
    if sidecar.is_symlink() or sidecar.read_text(encoding="ascii").strip() != target_sha:
        raise ValueError
    if target["sha256"] != target_sha or int(target["size"]) != int(target_size):
        raise ValueError
    if config["sha256"] != config_sha or int(config["size"]) != int(config_size):
        raise ValueError
    for name, size in (("adapter_model.safetensors", target_size), ("adapter_config.json", config_size)):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(size):
            raise ValueError
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
      echo "preserving matching adapter cache: $path"
      preserved=1
      continue
    fi
  fi
  lock="$cache/.$base.readers.lock"
  if ! sudo flock -n "$lock" -c "sudo find '$path' -xdev -depth -delete"; then
    echo "adapter cache entry is in use: $path" >&2
    exit 35
  fi
  removed=$((removed + 1))
done
sudo find "$cache" -maxdepth 1 -type f \
  \\( -name '.adapter-stage-status.json' -o -name 'adapter-staging.lock' \\) -delete
free=$(df -BG --output=avail "$cache" | tail -1 | tr -dc '0-9')
echo "removed=$removed free_gib=$free"
"""


def _cold_adapter_cache_cleanup(
    preserve_identity: dict[str, int | str] | None = None,
) -> None:
    """Require idle inference, then clear old adapter data on every inference host."""
    idle = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_idle_script(_inference_config(host)["ports"]),
            timeout=90,
        ),
    )
    busy = {host: output.strip() for host, (code, output) in idle.items() if code}
    if busy:
        raise Refused(f"cold adapter cleanup requires idle inference: {busy}")
    results = _fan(
        inference_hosts,
        lambda host: _ssh(
            host,
            _adapter_cache_cleanup_script(preserve_identity),
            timeout=600,
        ),
    )
    failures = {host: output.strip() for host, (code, output) in results.items() if code}
    if failures:
        raise RuntimeError(f"cold adapter cleanup failed: {failures}")
    for host, (_code, output) in results.items():
        print(f"cold adapter cleanup on {host}: {output.strip()}")


def start(
    run: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
    supervisors: bool = True,
    resume_adapter: str | None = None,
    resume_step: int | None = None,
    config_dir: str = default_config_dir,
) -> None:
    """Start a run from one explicit run config directory."""
    if mode not in ("cold", "restart"):
        raise Refused('mode must be "cold" or "restart"')
    _runs_path, _run_parent, service_run_dir = _cold_run_paths(run)
    if (resume_adapter is None) != (resume_step is None):
        raise Refused("resume_adapter and resume_step must be set together")
    selected = _config_dir(config_dir, require_orchestrator=True)
    if resume_adapter is not None:
        if not resume_adapter.startswith(f"{run_root}/"):
            raise Refused(f"resume adapter must be below {run_root}")
        if resume_step is None or resume_step < 0:
            raise Refused("resume_step must be zero or greater")
    nodes = [_check_host(h) for h in (nodes or trainer_hosts)]
    if trainer_head in nodes:  # rank 0 is the rendezvous master and carries the readable log
        nodes = [trainer_head] + [n for n in nodes if n != trainer_head]
    required_files = _start_dependency_inventory(selected, nodes)
    _deploy_dependencies(required_files)
    name = f"run_{run}"
    trainer_config = f"{remote_repo}/{selected['trainer']}"
    edge_config = f"{remote_repo}/{selected['orchestrator']}"
    if _ssh(orchestrator_host, f"test -f {edge_config}")[0] != 0:
        raise Refused(
            f"missing orchestrator config {edge_config} on {orchestrator_host}"
        )
    missing_trainer_config = [
        host
        for host, (code, _output) in _fan(
            nodes, lambda host: _ssh(host, f"test -f {trainer_config}")
        ).items()
        if code
    ]
    if missing_trainer_config:
        raise Refused(f"missing trainer config {trainer_config} on {missing_trainer_config}")
    initial_adapter_identity = (
        _initial_adapter_cache_identity(selected["trainer"], trainer_head)
        if mode == "cold"
        else None
    )
    trainer_nodes = " ".join(n.lstrip("n") for n in nodes)
    trainer_addresses = " ".join(_ssh_address(node) for node in nodes)
    runtime_env = (
        f"deployment_root={shlex.quote(remote_repo)} "
        f"storage_root={shlex.quote(storage_root)} "
        f"runs_path={shlex.quote(run_root)} "
        f"trainer_container_name={shlex.quote(trainer_container)} "
        f"master_node_id={shlex.quote(trainer_head.removeprefix('n'))} "
        f"trainer_sync_targets={shlex.quote(trainer_addresses)} "
        f"adapter_node_id={shlex.quote(trainer_head.removeprefix('n'))} "
        f"adapter_service_port={adapter_port} "
        f"adapter_container_name={shlex.quote(adapter_container)}"
    )
    predictive_adapter_env = ""
    local_edge_config = repo_root / selected["orchestrator"]
    try:
        with local_edge_config.open("rb") as stream:
            edge_settings = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Refused(f"cannot read GLM orchestrator config {edge_config}: {error}") from error
    broadcast_settings = edge_settings.get("weight_broadcast")
    source_url = (
        broadcast_settings.get("source_url")
        if isinstance(broadcast_settings, dict)
        else None
    )
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        raise Refused("GLM orchestrator config has no adapter source URL")
    if edge_settings.get("output_dir") != service_run_dir:
        raise Refused("GLM orchestrator output directory does not match the active run")
    parsed_source_url = urlsplit(source_url.rstrip("/"))
    if (
        parsed_source_url.hostname not in {"127.0.0.1", "localhost"}
        or parsed_source_url.username
        or parsed_source_url.password
        or parsed_source_url.query
        or parsed_source_url.fragment
        or parsed_source_url.path.rstrip("/") != f"/{run}"
    ):
        raise Refused("GLM adapter source URL does not match the active run")
    broadcast_path = (
        PurePosixPath(service_run_dir) / "broadcasts"
    ).relative_to(PurePosixPath(run_root)).as_posix()
    if broadcast_path != f"{run}/run_{run}/broadcasts":
        raise Refused("GLM adapter broadcast path does not match the active run")
    predictive_adapter_env = (
        " ADAPTER_PREDICTIVE_UPGRADE=1"
        f" ADAPTER_PREDICTIVE_RUN={shlex.quote(run)}"
        f" ADAPTER_PREDICTIVE_BROADCAST_PATH={shlex.quote(broadcast_path)}"
        f" ADAPTER_PREDICTIVE_SOURCE_URL={shlex.quote(source_url.rstrip('/'))}"
    )
    env = (
        f"{runtime_env} trainer_nodes='{trainer_nodes}' RUN={run} RUN_NAME={name} "
        f"TRAINER_CONFIG={trainer_config} PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 "
        "PRIME_INT4_COMBINE_ROUNDS=8 PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=1"
    )
    if resume_adapter is not None:
        missing = [
            host
            for host, (rc, _out) in _fan(
                nodes, lambda host: _ssh(host, f"test -f {shlex.quote(resume_adapter)}")
            ).items()
            if rc
        ]
        if missing:
            print(f"copying the resume adapter to {len(missing)} trainer nodes")
            _copy_resume_adapter(resume_adapter, nodes)
            still_missing = [
                host
                for host, (rc, _out) in _fan(
                    nodes,
                    lambda host: _ssh(
                        host, f"test -f {shlex.quote(resume_adapter)}"
                    ),
                ).items()
                if rc
            ]
            if still_missing:
                raise RuntimeError(
                    f"resume adapter copy is incomplete on {still_missing}"
                )
        env += (
            f" PRIME_RL_RESUME_ADAPTER={shlex.quote(resume_adapter)}"
            f" PRIME_RL_RESUME_ADAPTER_STEP={resume_step}"
        )

    def launch_trainer():
        print(f"starting trainer on {len(nodes)} nodes")
        results = _fan(
            nodes,
            lambda h: _ssh(
                h,
                f"node_id={shlex.quote(h.removeprefix('n'))} {env} "
                f"bash {trainer_launcher} trainer",
                300,
            ),
        )
        failed = {
            host: " | ".join(output.strip().splitlines()[-5:])
            if output.strip()
            else "no output"
            for host, (code, output) in results.items()
            if code
        }
        if failed:
            raise RuntimeError(f"trainer start failed: {failed}")

    def launch_adapter():
        print(f"starting adapter server on {trainer_head}")
        node_id = trainer_head.removeprefix("n")
        rc, out = _ssh(
            trainer_head,
            f"node_id={shlex.quote(node_id)} {runtime_env} RUN={shlex.quote(run)}"
            f"{predictive_adapter_env} "
            f"bash {trainer_launcher} adapter-http",
            300,
        )
        if rc:
            detail = " | ".join(out.strip().splitlines()[-5:]) if out.strip() else "no output"
            raise RuntimeError(f"adapter server start failed on {trainer_head}: {detail}")

    def launch_orch():
        print(f"starting edge orchestrator on {orchestrator_host}")
        rc, out = _ssh(
            orchestrator_host,
            f"RUN={shlex.quote(run)} RUN_NAME={shlex.quote(name)} "
            f"ORCH_CONFIG={shlex.quote(edge_config)} "
            f"TRAINER_ADDRESSES={shlex.quote(trainer_addresses)} "
            f"TRAINER_CT={shlex.quote(trainer_container)} "
            f"ORCH_CT={shlex.quote(orchestrator_container)} "
            f"bash {orchestrator_launcher}",
            300,
        )
        if rc:
            raise RuntimeError(f"edge orchestrator start failed on {orchestrator_host}: {out}")

    def wait_for_orchestrator_control() -> None:
        control_file = f"{service_run_dir}/control/orch.toml"
        script = f"""set -eu
for _ in $(seq 1 60); do
  if [ -s {shlex.quote(control_file)} ]; then
    echo ready
    exit 0
  fi
  sleep 1
done
echo "orchestrator control file did not arrive" >&2
exit 1
"""
        code, output = _ssh(trainer_head, script, timeout=70)
        if code:
            detail = output.strip() or "no output"
            raise RuntimeError(
                f"orchestrator control file did not reach {trainer_head}: {detail}"
            )

    restart_marker = f"{service_run_dir}/.trainer-restart-started"

    def mark_restart_start() -> None:
        _remote_ok(
            trainer_head,
            f"test -d {shlex.quote(service_run_dir)} && "
            f"touch {shlex.quote(restart_marker)}",
            "trainer restart marker",
        )

    def wait_for_fresh_broadcast() -> None:
        broadcast_dir = f"{service_run_dir}/broadcasts"
        script = f"""set -eu
marker={shlex.quote(restart_marker)}
broadcast_dir={shlex.quote(broadcast_dir)}
trap 'rm -f -- "$marker"' EXIT
for _ in $(seq 1 {adapter_update_engine_startup_timeout_seconds}); do
  stable=$(find "$broadcast_dir" -mindepth 2 -maxdepth 2 -type f -name STABLE -newer "$marker" -print -quit 2>/dev/null || true)
  if [ -n "$stable" ]; then
    echo "fresh adapter ready: ${{stable%/STABLE}}"
    exit 0
  fi
  running=$(sudo docker ps --filter name=^{trainer_container}$ --format '{{{{.Names}}}}')
  if [ "$running" != {shlex.quote(trainer_container)} ]; then
    echo "trainer exited before publishing a fresh adapter" >&2
    exit 1
  fi
  sleep 1
done
echo "trainer did not publish a fresh adapter within {adapter_update_engine_startup_timeout_seconds}s" >&2
exit 1
"""
        code, output = _ssh(
            trainer_head,
            script,
            timeout=adapter_update_engine_startup_timeout_seconds + 30,
        )
        if code:
            detail = " | ".join(output.strip().splitlines()[-5:]) or "no output"
            raise RuntimeError(f"fresh adapter wait failed on {trainer_head}: {detail}")
        print(output.strip())

    print(f"stopping any previous containers for {run}")
    _remote_ok(
        trainer_head,
        "pkill -f 'bash .*[r]un_supervisor.sh' 2>/dev/null || true; echo done",
        "old local supervisor stop",
    )
    _remote_ok(
        orchestrator_host,
        f"pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null; "
        f"sudo docker rm -f {orchestrator_container} {adapter_relay_container} >/dev/null 2>&1; echo done",
        "old orchestrator stop",
    )
    _remote_ok(
        trainer_head,
        f"sudo docker rm -f {orchestrator_container} >/dev/null 2>&1; echo done",
        "old trainer-side orchestrator stop",
    )
    stop_results = _fan(
        nodes,
        lambda h: _ssh(
            h,
            f"sudo pkill -9 -f '[P]RIME-RL' 2>/dev/null; sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
        ),
    )
    stop_failures = [host for host, (code, _output) in stop_results.items() if code]
    if stop_failures:
        raise RuntimeError(f"old trainer stop failed on {stop_failures}")
    time.sleep(20)

    if mode == "cold":
        _cold_start_cleanup(run, nodes)
        _cold_adapter_cache_cleanup(initial_adapter_identity)
    launch_adapter()
    if mode == "cold":
        launch_orch()
        wait_for_orchestrator_control()
        launch_trainer()
    else:
        mark_restart_start()
        launch_trainer()
        wait_for_fresh_broadcast()
        launch_orch()

    if supervisors:
        _repair_supervisors(config_dir)
    print(f"started {run} in {mode} mode. Call status() in a few minutes.")


def qualify(
    name: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
    config_dir: str = default_config_dir,
) -> None:
    """Start one configured trainer-only qualification."""
    if mode not in ("cold", "restart"):
        raise Refused('qualification mode must be "cold" or "restart"')
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("qualification name can contain only letters, digits, '_' and '-'")
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    if trainer_head in nodes:
        nodes = [trainer_head] + [host for host in nodes if host != trainer_head]

    selected = _config_dir(config_dir)
    launch_env = _qualification_launch_env(selected)
    if "orchestrator" in selected:
        raise Refused(f"run config directory {config_dir!r} is a full run; use start()")
    relative_config = selected["trainer"]
    local_config = repo_root / relative_config
    if not local_config.is_file():
        raise Refused(f"missing qualification config {relative_config}")
    with local_config.open("rb") as stream:
        config = tomllib.load(stream)
    if "fake" not in config.get("data", {}):
        raise Refused(
            f"{relative_config} has no [data.fake]; use start() for an RL run"
        )
    rc, out = _ssh(
        orchestrator_host,
        f"sudo docker ps --filter name=^{orchestrator_container}$ --format '{{{{.Names}}}}'",
    )
    if rc or out.strip():
        raise Refused(
            "an orchestrator is active; a trainer-only qualification must not replace an RL run"
        )

    def trainer_running(host: str) -> tuple[int, str]:
        return _ssh(
            host,
            f"sudo docker ps --filter name=^{trainer_container}$ --format '{{{{.Names}}}}'",
        )

    active = [
        host
        for host, (rc, out) in _fan(nodes, trainer_running).items()
        if rc == 0 and out.strip()
    ]
    if active and mode == "cold":
        raise Refused(
            "trainer qualification is already active on "
            + ", ".join(active)
            + "; use mode='restart' only when replacing it is intended"
        )

    repository_root = _repository_root()
    required_files = _trainer_dependency_inventory(relative_config, nodes)
    _deploy_dependencies(required_files)

    trainer_nodes = " ".join(host.removeprefix("n") for host in nodes)
    runtime_env = (
        f"deployment_root={shlex.quote(repository_root)} "
        f"storage_root={shlex.quote(storage_root)} "
        f"runs_path={shlex.quote(run_root)} "
        f"trainer_container_name={shlex.quote(trainer_container)} "
        f"master_node_id={shlex.quote(trainer_head.removeprefix('n'))}"
    )
    env = (
        f"{runtime_env} trainer_nodes='{trainer_nodes}' RUN={shlex.quote(name)} "
        f"RUN_NAME=run_{shlex.quote(name)} "
        f"TRAINER_CONFIG={repository_root}/{relative_config} "
        "PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 PRIME_INT4_COMBINE_ROUNDS=1 "
        "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=0"
    )
    for key, value in launch_env.items():
        env += f" {key}={shlex.quote(value)}"

    if active:
        _fan(
            nodes,
            lambda host: _ssh(
                host,
                f"sudo pkill -9 -f '[P]RIME-RL' 2>/dev/null; "
                f"sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
            ),
        )
        time.sleep(20)

    results = _fan(
        nodes,
        lambda host: _ssh(
            host,
            f"node_id={shlex.quote(host.removeprefix('n'))} {env} "
            f"bash {repository_root}/training_ops/runtime/china/cluster.sh "
            "trainer",
            300,
        ),
    )
    failed = {
        host: " | ".join(output.strip().splitlines()[-5:])
        if output.strip()
        else "no output"
        for host, (rc, output) in results.items()
        if rc
    }
    if failed:
        raise RuntimeError(f"qualification trainer launch failed: {failed}")
    print(
        f"started trainer-only qualification {name} on {len(nodes)} nodes; "
        "call status() or watch()"
    )


def stop() -> None:
    """Stop the edge orchestrator, then the trainer. Supervisors stop first."""
    _remote_ok(
        trainer_head,
        "pkill -f 'bash .*[r]un_supervisor.sh' 2>/dev/null || true; "
        "pkill -f 'bash .*[t]unnel_watchdog.sh' 2>/dev/null || true; echo done",
        "supervisor stop",
    )
    _remote_ok(
        orchestrator_host,
        "pkill -f 'bash .*[t]rainer_failure_watchdog.sh' 2>/dev/null || true; echo done",
        "edge trainer watchdog stop",
    )
    _remote_ok(
        orchestrator_host,
        f"pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null; "
        f"sudo docker rm -f {orchestrator_container} {adapter_relay_container} >/dev/null 2>&1; echo done",
        "orchestrator stop",
    )
    _remote_ok(
        trainer_head,
        f"sudo docker rm -f {orchestrator_container} >/dev/null 2>&1; echo done",
        "trainer-side orchestrator stop",
    )
    results = _fan(
        trainer_hosts,
        lambda h: _ssh(
            h,
            f"sudo pkill -9 -f '[P]RIME-RL' 2>/dev/null; sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
        ),
    )
    failed = [host for host, (code, _output) in results.items() if code]
    if failed:
        raise RuntimeError(f"trainer stop failed on {failed}")
    print(
        "stopped supervisors, edge orchestrator, and trainer. The engines are still running."
    )


# ------------------------------------------------------------------ entry


def _remote_action(arguments: list[str]):
    if len(arguments) < 2:
        raise Refused("remote needs a host group and a command")
    return run_on_hosts(" ".join(arguments[1:]), arguments[0])


def _start_action(arguments: list[str]):
    if not arguments:
        raise Refused("start needs a run name")
    return start(
        arguments[0],
        arguments[1] if len(arguments) > 1 else "cold",
        config_dir=arguments[2] if len(arguments) > 2 else default_config_dir,
    )


def _qualify_action(arguments: list[str]):
    if not arguments:
        raise Refused("qualify needs a run name")
    return qualify(
        arguments[0],
        arguments[1] if len(arguments) > 1 else "cold",
        config_dir=arguments[2] if len(arguments) > 2 else default_config_dir,
    )


def _archive_action(arguments: list[str]):
    if not arguments:
        raise Refused("archive needs a run name")
    finished = False
    step_args = []
    for argument in arguments[1:]:
        if argument == "--finished":
            finished = True
        elif argument.startswith("--"):
            raise Refused(f"unknown archive option {argument}")
        else:
            step_args.append(argument)
    try:
        steps_to_archive = [int(value) for value in step_args]
    except ValueError as error:
        raise Refused("archive steps must be non-negative integers") from error
    return archive_adapters(arguments[0], steps_to_archive or None, finished=finished)


actions = {
    "status": lambda a: status(*a),
    "acceptance": lambda a: acceptance(*a),
    "reward": lambda a: reward(a[0] if a else "", host=a[1] if len(a) > 1 else ""),
    "engines": lambda a: engines(*a),
    "logs": lambda a: logs(a[0] if a else "trainer", int(a[1]) if len(a) > 1 else 200),
    "parity": lambda a: parity(*a),
    "steps": lambda a: steps(a[0] if a else "", int(a[1]) if len(a) > 1 else 10),
    "remote": _remote_action,
    "versions": lambda a: package_versions(*a),
    "watch": lambda a: watch(*[float(x) for x in a]),
    "repair": lambda a: repair(*a),
    "deploy": lambda a: deploy(*a),
    "start": _start_action,
    "archive": _archive_action,
    "qualify": _qualify_action,
    "stop": lambda a: stop(),
    "mounts": lambda a: print("\n".join(mounted_files())),
}


def run(argstr: str = "status"):
    """Run one command from a notebook or Python shell."""
    parts = shlex.split(argstr)
    if not parts or parts[0] not in actions:
        raise Refused(f"use one of: {', '.join(sorted(actions))}")
    return actions[parts[0]](parts[1:])


usage = f"""training-ops <action> [args]

  status [run]                  show the complete run state
  acceptance [run]              check training requirements
  reward [run] [host]           show reward and its measured trend
  steps [run] [count]           show recent trainer times
  engines                       test every configured inference endpoint
  logs [role] [lines]           read one service log
  parity [run]                  read the trainer parity record
  remote <group> <command>      run on trainer, inference, or all hosts
  versions [package ...]        show installed package versions
  watch [hours] [seconds]       monitor in the foreground
  repair <name>                 repair an engine, adapter server, orchestrator, supervisors, or microbatch sync
  deploy <path>                 copy one repository file
  start <run> [mode] [config-dir]  start one run config directory
  archive <run> [step ...]      upload stable old LoRA adapters
                                add --finished after another run is active
  qualify <name> [mode] [config-dir]  start a trainer-only config directory
  mounts                        list live-mounted source files
  stop                          stop the run but keep inference loaded

Read {repo_root / '.agents/skills/training-ops/SKILL.md'} first.
"""


def cli() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(usage)
        sys.exit(0)
    try:
        run(" ".join(args))
    except (Refused, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
