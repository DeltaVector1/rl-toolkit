"""Read and operate the configured training system."""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import ipaddress
import json
import math
import re
import shlex
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

repo_root = Path(__file__).resolve().parents[5]
operations_path = repo_root / "configs" / "operations.toml"


def _read_operations(path: Path = operations_path) -> dict:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read operations config {path}: {error}") from error
    return data


operations = _read_operations()
hosts = operations["hosts"]
ports = operations["ports"]
paths = operations["paths"]
containers = operations["containers"]
models = operations["models"]
inference_settings = operations["inference"]
reward_settings = operations["reward"]
limits = operations["limits"]
shared_requirements = operations["requirements"]["shared"]
muse_requirements = operations["requirements"]["muse"]
glm_requirements = operations["requirements"]["glm52"]

trainer_head = hosts["trainer_head"]
trainer_hosts = list(hosts["trainers"])
inference_hosts = list(hosts["inference"])
orchestrator_host = hosts["orchestrator"]
metric_hosts = list(hosts["metrics"])
blocked_hosts = set(hosts["blocked"])
inference_tunnel_ports = list(
    range(ports["inference_tunnel_first"], ports["inference_tunnel_last"] + 1)
)
judge_port = ports["judge"]
adapter_port = ports["adapter"]
batch_port = ports["batch"]
remote_repo = paths["remote_repository"]
storage_root = paths["storage_root"]
run_root = paths["runs"]
runtime_root = f"{remote_repo}/.agents/skills/training-ops/runtime"
trainer_launcher = f"{runtime_root}/china/cluster.sh"
orchestrator_launcher = f"{runtime_root}/h200/start_orchestrator.sh"
trainer_container = containers["trainer"]
orchestrator_container = containers["orchestrator"]
adapter_container = containers["adapter"]
glm_inference_container = containers["glm52_inference"]
muse_inference_container = containers["muse_inference"]
glm_model = models["glm52"]
muse_model = models["muse"]
baseline_std = reward_settings["reference_spread"]
baseline_run = reward_settings["reference_run"]
minimum_trend_steps = reward_settings["minimum_steps"]
trend_spread_multiplier = reward_settings["spread_multiplier"]
stall_seconds = limits["stall_seconds"]
final_validation_steps = limits["final_steps"]
deployment_file_workers = limits["deployment_file_workers"]
deployment_host_workers = limits["deployment_host_workers"]
deployment_bundle_workers = limits.get("deployment_bundle_workers", 2)
deployment_mkdir_timeout_seconds = limits["deployment_mkdir_timeout_seconds"]
deployment_scp_timeout_seconds = limits["deployment_scp_timeout_seconds"]
deployment_bundle_scp_timeout_seconds = limits.get(
    "deployment_bundle_scp_timeout_seconds", 60
)
group_size = shared_requirements["group_size"]
max_output_length = shared_requirements["max_output_length"]
learning_rate = shared_requirements["learning_rate"]
max_gradient_norm = shared_requirements["max_gradient_norm"]
weight_decay = shared_requirements["weight_decay"]
muse_batch_size = muse_requirements["batch_size"]
muse_context_length = muse_requirements["context_length"]
muse_compute_max_seconds = muse_requirements["trainer_seconds_max"]
muse_rollout_max_seconds = muse_requirements["rollout_seconds_max"]
glm_batch_size = glm_requirements["batch_size"]
glm_context_length = glm_requirements["context_length"]
glm_compute_min_seconds = glm_requirements["trainer_seconds_min"]
glm_compute_max_seconds = glm_requirements["trainer_seconds_max"]
glm_rollout_min_seconds = glm_requirements["rollout_seconds_min"]
glm_rollout_max_seconds = glm_requirements["rollout_seconds_max"]

adapter_archive_bucket = paths["adapter_bucket"]
adapter_archive_files = ("adapter_config.json", "adapter_model.safetensors")
dependency_manifest_name = ".training-ops-manifest"
_dependency_remote_archive_pattern = re.compile(
    r"^/tmp/trainer-dependencies-[A-Za-z0-9._-]+\.tar$"
)

if set(inference_hosts) != set(inference_settings):
    raise RuntimeError("each inference host must have one [inference.<host>] entry")
profile_kinds = {
    profile.get("kind") for profile in operations.get("profiles", {}).values()
}
for inference_host in inference_hosts:
    missing_kinds = profile_kinds - set(inference_settings[inference_host])
    if missing_kinds:
        raise RuntimeError(
            f"inference settings for {inference_host} are missing {sorted(missing_kinds)}"
        )


def _inference_config(host: str, kind: str | None = None) -> dict:
    kind = kind or _profile()["kind"]
    settings = inference_settings.get(host, {}).get(kind)
    if not isinstance(settings, dict):
        raise RuntimeError(f"inference settings for {host} and {kind} are missing")
    return settings


def _profile(name: str = "current") -> dict:
    profiles = operations.get("profiles", {})
    if name not in profiles:
        raise Refused(f"unknown profile {name!r}; choose from {', '.join(sorted(profiles))}")
    profile = dict(profiles[name])
    if profile.get("kind") not in {"muse", "glm52"}:
        raise RuntimeError(f"profile {name!r} has an invalid kind")
    trainer = profile.get("trainer")
    if not isinstance(trainer, str) or not (repo_root / trainer).is_file():
        raise RuntimeError(f"profile {name!r} has no readable trainer config")
    orchestrator = profile.get("orchestrator")
    if orchestrator is not None and (
        not isinstance(orchestrator, str) or not (repo_root / orchestrator).is_file()
    ):
        raise RuntimeError(f"profile {name!r} has no readable orchestrator config")
    return profile


def _profile_for_run(run: str) -> dict:
    """Select the configured trainer profile whose output matches the run."""
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused(f"active run name is invalid: {run!r}")
    matches = []
    for name in sorted(operations.get("profiles", {})):
        profile = _profile(name)
        trainer_path = repo_root / profile["trainer"]
        try:
            with trainer_path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError(f"cannot read trainer config {trainer_path}: {error}") from error
        output_dir = config.get("output_dir")
        if isinstance(output_dir, str) and Path(output_dir).name == run:
            matches.append((name, profile))
    if not matches:
        raise Refused(
            f"no configured trainer profile has output_dir ending in {run!r}"
        )
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches)
        raise Refused(
            f"multiple configured trainer profiles have output_dir ending in {run!r}: {names}"
        )
    return matches[0][1]

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


def _ssh(host: str, script: str, timeout: int = 120) -> tuple[int, str]:
    _check_host(host)
    try:
        p = subprocess.run(
            ssh_command + [host, script],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = p.stdout if p.returncode == 0 else p.stdout + p.stderr
        return p.returncode, output
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _ssh_address(host: str) -> str:
    """Return the IP selected by the configured SSH alias."""
    _check_host(host)
    try:
        result = subprocess.run(
            ssh_command + ["-G", host],
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


# -------------------------------------------------------------------- probes

trainer_probe = r"""
ok=0; bad=""
for p in $(seq __TUNNEL_FIRST__ __TUNNEL_LAST__); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
  if [ "$c" = "200" ]; then ok=$((ok+1)); else bad="$bad $p"; fi
done
echo "TUNNELS_OK $ok"
echo "TUNNELS_BAD $bad"
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "TRAINER_UP $(sudo docker ps --filter name=__TRAINER__ --format up 2>/dev/null | head -1)"
trainer_cfg=$(sudo docker inspect __TRAINER__ --format '{{range .Config.Cmd}}{{println .}}{{end}}' 2>/dev/null | grep -E '\.toml$' | tail -1)
if [ -n "$trainer_cfg" ] && grep -q '^\[data\.fake\]' "$trainer_cfg" 2>/dev/null; then
  echo "MODE trainer-only"
else
  echo "MODE rl"
fi
echo "ORCH_UP $(sudo docker ps --filter name=__ORCH__ --format up 2>/dev/null | head -1)"
echo "TRAINER_STEPS $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -ac 'SUCCESS Step')"
echo "TRAINER_LAST $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "TRAINER_CURRENT $(sudo docker logs --tail 1 __TRAINER__ 2>&1 | strip | tail -1)"
echo "TRAINER_FINISHED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'RL trainer finished|Trainer finished')"
echo "TRAINER_FAILED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'prime_rl\.trainer\.rl\.train FAILED|CheckpointException')"
echo "ORCH_LAST $(sudo docker logs __ORCH__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "BATCH $(sudo docker logs --tail 40 __ORCH__ 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1)"
echo "FAILS5M $(sudo docker logs --since 5m __ORCH__ 2>&1 | grep -ac 'Rollout failed')"
echo "TRAINER_LOGAGE $(sudo docker logs -t --tail 1 __TRAINER__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "ORCH_LOGAGE $(sudo docker logs -t --tail 1 __ORCH__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "JUDGE $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__JUDGE__/v1/models)"
echo "ADAPTER $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__ADAPTER__/)"
echo "SUP_TUN $(pgrep -f 'bash .*[t]unnel_watchdog.sh' | wc -l)"
echo "SUP_TUN_LOG $(tail -1 /data/glm52-rl/tunnel_watchdog.log 2>/dev/null | cut -d' ' -f1)"
"""


trainer_only_probe = r"""
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "MODE trainer-only"
echo "TRAINER_UP $(sudo docker ps --filter name=__TRAINER__ --format up 2>/dev/null | head -1)"
echo "TRAINER_STEPS $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -ac 'SUCCESS Step')"
echo "TRAINER_LAST $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "TRAINER_CURRENT $(sudo docker logs --tail 1 __TRAINER__ 2>&1 | strip | tail -1)"
echo "TRAINER_FINISHED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'RL trainer finished|Trainer finished')"
echo "TRAINER_FAILED $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -aEc 'prime_rl\.trainer\.rl\.train FAILED|CheckpointException')"
echo "TRAINER_LOGAGE $(sudo docker logs -t --tail 1 __TRAINER__ 2>&1 | tail -1 | cut -d' ' -f1)"
"""


orchestrator_probe = r"""
strip() { sed "s/\\x1b\\[[0-9;]*m//g"; }
echo "ORCH_UP $(sudo docker ps --filter name=__ORCH__ --format up 2>/dev/null | head -1)"
echo "BATCH_SYNC $(pgrep -f 'bash .*[s]ync_rollout_batches.sh' | wc -l)"
echo "TRAINER_FAIL_WD $(pgrep -f 'bash .*[t]rainer_failure_watchdog.sh' | wc -l)"
echo "ORCH_LAST $(sudo docker logs __ORCH__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "BATCH $(sudo docker logs --tail 40 __ORCH__ 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1)"
echo "FAILS5M $(sudo docker logs --since 5m __ORCH__ 2>&1 | grep -ac 'Rollout failed')"
echo "ORCH_LOGAGE $(sudo docker logs -t --tail 1 __ORCH__ 2>&1 | tail -1 | cut -d' ' -f1)"
"""

node_probe = r"""
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '{m+=$1; u+=$2; n++} END{printf "GPU %d %d %d\n", m/n, u/n, n}'
free -g | sed -n 2p | awk '{print "MEM " $7 " " $5}'
df -BG --output=avail,pcent /data | tail -1 | awk '{gsub(/G|%/, ""); print "DISK " $1 " " $2}'
echo "CT $(sudo docker ps --filter name=__TRAINER__ --format '{{.Status}}' 2>/dev/null | head -1)"
"""

inference_probe = r"""
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
  | awk '{u+=$1; n++} END{printf "GPU %d %d\n", u/n, n}'
ok=0
total=0
longest=0
for port in __PORTS__; do
  total=$((total+1))
  started=$(date +%s%N)
  code=$(curl -s -o /dev/null -m 25 -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"__MODEL__","prompt":"hi","max_tokens":3}')
  ended=$(date +%s%N)
  elapsed=$(( (ended-started)/1000000 ))
  [ "$elapsed" -gt "$longest" ] && longest=$elapsed
  [ "$code" = 200 ] && ok=$((ok+1))
done
[ "$total" -gt 0 ] || exit 3
[ "$ok" -eq "$total" ] && code=200 || code=000
echo "GEN $code $longest"
echo "PROBES $ok $total"
if sudo docker ps -a --filter name=^__MUSE__-0$ --format '{{.Names}}' | grep -q .; then
  replicas=$(sudo docker ps -a --filter name=__MUSE__- --format '{{.Names}}' | grep -E '^__MUSE__-[0-9]+$' | sort -V)
  echo "RUNNING $(for c in $replicas; do sudo docker logs --since 60s $c 2>&1 | grep -aoE 'Running: [0-9]+' | grep -oE '[0-9]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}'; done | awk '{s+=$1} END{print s+0}')"
  echo "TPUT $(for c in $replicas; do sudo docker logs --since 60s $c 2>&1 | grep -aoE 'generation throughput: [0-9.]+' | grep -oE '[0-9.]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}'; done | awk '{s+=$1} END{print s+0}')"
else
  echo "RUNNING $(sudo docker logs --since 60s __ENGINE__ 2>&1 | grep -aoE 'Running: [0-9]+' | grep -oE '[0-9]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}')"
  echo "TPUT $(sudo docker logs --since 60s __ENGINE__ 2>&1 | grep -aoE 'generation throughput: [0-9.]+' | grep -oE '[0-9.]+' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m}')"
fi
echo "WD $(pgrep -f 'bash .*[e]ngine_watchdog.sh' | wc -l)"
"""


def _inference_script(
    host: str, model: str, container: str, kind: str | None = None
) -> str:
    settings = _inference_config(host, kind)
    gpu_count = settings.get("gpu_count")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise RuntimeError(f"GPU count for {host} is invalid")
    tensor_parallel_size = settings.get("tensor_parallel_size")
    if (
        not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size < 1
        or gpu_count % tensor_parallel_size
    ):
        raise RuntimeError(f"tensor parallel size for {host} is invalid")
    ports = settings.get("ports")
    if not isinstance(ports, list) or not ports or not all(
        isinstance(port, int) and 1 <= port <= 65535 for port in ports
    ):
        raise RuntimeError(f"inference ports for {host} are invalid")
    if len(ports) != gpu_count // tensor_parallel_size:
        raise RuntimeError(f"inference replica count for {host} is inconsistent")
    return (
        inference_probe.replace("__MODEL__", model)
        .replace("__ENGINE__", container)
        .replace("__MUSE__", muse_inference_container)
        .replace("__PORTS__", " ".join(map(str, ports)))
    )


def _head_script() -> str:
    return (
        trainer_probe.replace("__TRAINER__", trainer_container)
        .replace("__ORCH__", orchestrator_container)
        .replace("__JUDGE__", str(judge_port))
        .replace("__ADAPTER__", str(adapter_port))
        .replace("__TUNNEL_FIRST__", str(inference_tunnel_ports[0]))
        .replace("__TUNNEL_LAST__", str(inference_tunnel_ports[-1]))
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
    reachable: bool = True

    @property
    def healthy(self) -> bool:
        return self.gen_code == "200"


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
            else "no post-step-1 measurement"
        )
        generation = (
            f"step {self.rollout_generation_step}: {_hms(self.rollout_generation_seconds)}"
            if self.rollout_generation_step is not None
            else "no separate generation measurement"
        )
        L = [f"training requirements for {self.run}"]
        L.append(f"profile: {self.profile}")
        L.append(f"config: {values}")
        if self.profile == "muse":
            L.append(f"BF16 trainer: {precision}")
            L.append(
                f"trainer forward/backward only: {trainer} (alarm above 8m after step 1)"
            )
            L.append(
                f"active rollout generation only: {generation} (alarm above 20m)"
            )
        else:
            L.append(f"native FP8 trainer: {precision}")
            L.append(
                f"trainer forward/backward only: {trainer} (check 10-15m after step 1)"
            )
            L.append(
                f"active rollout generation only: {generation} (approximate check 8-12m)"
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
    tunnels_ok: int = 0
    tunnels_bad: list[int] = field(default_factory=list)
    trainer_up: bool = False
    trainer_finished: bool = False
    trainer_failed: bool = False
    orch_up: bool = False
    batch_sync: bool = False
    trainer_failure_watchdog: bool = False
    trainer_steps: int = 0
    trainer_last: str = ""
    trainer_current: str = ""
    orch_last: str = ""
    batch: str = ""
    fails_5m: int = 0
    trainer_log_age: float | None = None
    orch_log_age: float | None = None
    judge: str = "000"
    adapter: str = "000"
    supervisor_tunnel: bool = False
    supervisor_tunnel_log_age: float | None = None
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
                f"orchestrator {'up' if self.orch_up else 'down'} | "
                f"tunnels {self.tunnels_ok}/{len(inference_tunnel_ports)} | judge {self.judge} | adapter {self.adapter}"
            )
        L.append(
            f"run steps {self.trainer_steps} | failures in 5m {self.fails_5m} | {self.batch or 'no batch line'}"
        )
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
                f"tunnel={'yes' if self.supervisor_tunnel else 'no'} "
                f"batch-sync={'yes' if self.batch_sync else 'no'} "
                f"trainer-check={'yes' if self.trainer_failure_watchdog else 'no'} "
                f"(tunnel log {_hms(self.supervisor_tunnel_log_age)} old)"
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
) -> AcceptanceReport:
    """Build an acceptance report from text. This function never uses SSH."""
    trainer_values = _parse_toml_scalars(trainer_config)
    orch_values = _parse_toml_scalars(orchestrator_config)
    trainer_rows = _parse_jsonl_metrics(trainer_metrics)
    rollout_rows = _parse_jsonl_metrics(rollout_metrics)
    is_muse = run.lower().startswith("muse") or "muse-glimmer" in trainer_config.lower()
    profile = "muse" if is_muse else "glm52"
    rep = AcceptanceReport(
        run=run, profile=profile, reward_verdict=reward_verdict
    )
    expected_batch = muse_batch_size if is_muse else glm_batch_size
    expected_context = (
        muse_context_length if is_muse else glm_context_length
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
    dequant_flags = [
        _last(trainer_values, key)
        for key in (
            "model.quantization.dequantize_checkpoint",
            "model.quantization.dequantize_checkpoint_linears",
            "model.quantization.dequantize_checkpoint_moe",
        )
    ]
    if not trainer_config.strip():
        rep.native_fp8 = None
        rep.precision_ok = None
    elif is_muse:
        optimization_dtype = _last(trainer_values, "model.optimization_dtype")
        reduce_dtype = _last(trainer_values, "model.reduce_dtype")
        rep.native_fp8 = False
        rep.precision_ok = (
            optimization_dtype == "bfloat16"
            and reduce_dtype == "bfloat16"
            and quant_type is None
        )
    else:
        rep.native_fp8 = quant_type == "fp8" and all(
            flag is False for flag in dequant_flags
        )
        rep.precision_ok = rep.native_fp8

    rep.completed_steps = max((row["step"] for row in trainer_rows), default=0)
    compute = [
        (row["step"], float(row["time/forward_backward"]))
        for row in trainer_rows
        if row["step"] > 1
        and isinstance(row.get("time/forward_backward"), (int, float))
    ]
    if compute:
        rep.trainer_compute_step, rep.trainer_compute_seconds = compute[-1]

    generation_keys = (
        "train/agg/all/timing/generation/max",
        "train/decensor/all/timing/generation/max",
    )
    generation = []
    for row in rollout_rows:
        value = next(
            (
                row[key]
                for key in generation_keys
                if isinstance(row.get(key), (int, float))
            ),
            None,
        )
        if value is not None:
            generation.append((row["step"], float(value)))
    if generation:
        rep.rollout_generation_step, rep.rollout_generation_seconds = generation[-1]

    rep.large_batch_evidence = _large_batch_evidence(orch_values, trainer_rows, logs)
    rep.failures = _failure_evidence(trainer_rows, rollout_rows, logs)

    def expected(name: str, actual, want) -> None:
        if actual is None or actual != want:
            rep.alarms.append(f"{name} is {_shown(actual)}; expected {want}")

    expected("batch size", rep.batch_size, expected_batch)
    expected("group size", rep.group_size, group_size)
    expected("context length", rep.context_length, expected_context)
    if not rep.max_output_lengths or any(
        v != max_output_length for v in rep.max_output_lengths
    ):
        shown = ",".join(map(str, rep.max_output_lengths)) or "?"
        rep.alarms.append(
            f"maximum output length is {shown}; expected {max_output_length}"
        )
    if len(rep.learning_rates) < 2 or any(
        not _near(v, learning_rate) for v in rep.learning_rates
    ):
        shown = ",".join(f"{v:g}" for v in rep.learning_rates) or "?"
        rep.alarms.append(
            f"trainer/orchestrator learning rates are {shown}; expected {learning_rate:g}"
        )
    if rep.max_norm is None or not _near(rep.max_norm, max_gradient_norm):
        rep.alarms.append(
            f"maximum gradient norm is {_shown(rep.max_norm)}; expected {max_gradient_norm:g}"
        )
    if rep.weight_decay is None or not _near(rep.weight_decay, weight_decay):
        rep.alarms.append(
            f"weight decay is {_shown(rep.weight_decay)}; expected {weight_decay:g}"
        )
    if rep.precision_ok is not True:
        if is_muse:
            rep.alarms.append("trainer is not confirmed on unquantized BF16 weights")
        else:
            rep.alarms.append("trainer is not confirmed on native FP8 checkpoint weights")

    if is_muse:
        bad_compute = [
            step
            for step, seconds in compute
            if seconds > muse_compute_max_seconds
        ]
    else:
        bad_compute = [
            step
            for step, seconds in compute
            if seconds < glm_compute_min_seconds
            or seconds > glm_compute_max_seconds
        ]
    if bad_compute:
        if is_muse:
            rep.alarms.append(
                f"trainer forward/backward exceeds 8m after step 1 at steps "
                f"{','.join(map(str, bad_compute[-8:]))}"
            )
        else:
            rep.alarms.append(
                f"trainer forward/backward is outside 10-15m after step 1 at steps "
                f"{','.join(map(str, bad_compute[-8:]))}"
            )
    elif rep.completed_steps > 1 and not compute:
        rep.alarms.append(
            "no separate trainer forward/backward measurement after step 1"
        )

    if is_muse:
        bad_generation = [
            step
            for step, seconds in generation
            if seconds > muse_rollout_max_seconds
        ]
    else:
        bad_generation = [
            step
            for step, seconds in generation
            if seconds < glm_rollout_min_seconds
            or seconds > glm_rollout_max_seconds
        ]
    if bad_generation:
        if is_muse:
            rep.alarms.append(
                f"active rollout generation exceeds 20m at steps "
                f"{','.join(map(str, bad_generation[-8:]))}"
            )
        else:
            rep.alarms.append(
                f"active rollout generation is outside approximately 10m (8-12m alarm band) "
                f"at steps {','.join(map(str, bad_generation[-8:]))}"
            )
    elif rollout_rows and not generation:
        rep.alarms.append("no separate active rollout generation measurement")

    for category, count in sorted(rep.failures.items()):
        rep.alarms.append(f"{category} failure evidence is present ({count})")
    if reward_verdict == "falling":
        rep.alarms.append("effective reward verdict is falling")
    if not is_muse and not rep.large_batch_evidence:
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
metric=$(find {root} -maxdepth 3 -type f -name {shlex.quote(metric_name)} -print 2>/dev/null | head -1)
emit() {{ printf '%s_B64 ' "$1"; if [ -n "$2" ] && [ -r "$2" ]; then "$3" "$2" | base64 -w0; fi; printf '\\n'; }}
emit PATH "$cfg" cat
emit CONFIG "$cfg" cat
emit METRICS "$metric" tail
logs=""
for ct in {containers}; do
  logs="${{logs}}
$(sudo docker logs --tail 20000 "$ct" 2>&1 |
    grep -aiE {shlex.quote(acceptance_log_pattern)} | tail -200)"
done
printf 'LOGS_B64 '; printf '%s' "$logs" | base64 -w0; printf '\\n'
"""


def _decode_acceptance_probe(out: str) -> dict[str, str]:
    decoded = {}
    for line in out.splitlines():
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
    )
    rep.trainer_config_path = trainer.get("path", "")
    rep.orchestrator_config_path = edge.get("path", "")
    if not quiet:
        print(rep.report())
    return rep


# --------------------------------------------------------------------- reads


def logs(role: str = "trainer", lines: int = 200) -> str:
    """Read the current container tail without a separate SSH command."""
    kind = _profile_for_run(current_run())["kind"]
    inference_container = (
        muse_inference_container if kind == "muse" else glm_inference_container
    )
    roles = {
        "trainer": (trainer_head, trainer_container),
        "orchestrator": (orchestrator_host, orchestrator_container),
        **{host: (host, inference_container) for host in inference_hosts},
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


def parity(run: str = "") -> dict:
    """Summarize the step-1 token parity record on the owning trainer host."""
    run = run or current_run()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise Refused("run name contains invalid characters")
    path = f"{run_root}/{run}/parity-step1.json"
    script = f"""
python3 - {shlex.quote(path)} <<'PY'
import hashlib,json,math,sys
p=sys.argv[1]
try:
    raw=open(p,'rb').read()
except FileNotFoundError:
    raise SystemExit(3)
d=json.loads(raw)
lengths=d['sequence_lengths']
expected=[]
for length in lengths:
    expected.extend(range(length))
positions=d['position_ids']
mask=d['loss_mask']
delta=d['delta']
trainer=d['trainer_logprobs']
reference=d['reference_logprobs']
valid=[x for x,keep in zip(delta,mask) if keep]
valid_trainer=[x for x,keep in zip(trainer,mask) if keep]
valid_reference=[x for x,keep in zip(reference,mask) if keep]
ordered=sorted(abs(x) for x in valid)
proxy=sum(math.exp(min(x,80.0))-x-1.0 for x in valid)/max(len(valid),1)
out={{
  'sha256':hashlib.sha256(raw).hexdigest(),
  'tokens':len(delta),
  'sequences':len(lengths),
  'valid_tokens':len(valid),
  'masked_fraction':1.0-len(valid)/max(len(delta),1),
  'position_mismatches':sum(a!=b for a,b in zip(positions,expected)) + abs(len(positions)-len(expected)),
  'mean_abs_delta':sum(abs(x) for x in valid)/max(len(valid),1),
  'mean_delta':sum(valid)/max(len(valid),1),
  'mean_trainer_logprob':sum(valid_trainer)/max(len(valid_trainer),1),
  'mean_reference_logprob':sum(valid_reference)/max(len(valid_reference),1),
  'p95_abs_delta':ordered[min(int(0.95*len(ordered)),len(ordered)-1)] if ordered else 0.0,
  'max_abs_delta':max(ordered,default=0.0),
  'mismatch_kl_proxy':proxy,
}}
print(json.dumps(out,sort_keys=True))
PY
"""
    code, output = _ssh(trainer_head, script, timeout=180)
    if code:
        raise RuntimeError(f"cannot read parity record {path} on {trainer_head}")
    try:
        summary = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"parity record {path} is invalid") from error
    summary["host"] = trainer_head
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def status(run: str = "", quiet: bool = False) -> Status:
    """Probe every part of the run once and return the whole picture."""
    run = run or current_run()
    kind = _profile_for_run(run)["kind"]
    muse_run = kind == "muse"
    engine_model = muse_model if muse_run else glm_model
    engine_container = muse_inference_container if muse_run else glm_inference_container
    s = Status(
        when=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), run=run
    )
    trainer_only = _trainer_mode() == "trainer-only"

    def node(h):
        return _ssh(h, node_probe.replace("__TRAINER__", trainer_container), timeout=60)

    def engine(h):
        return _ssh(
            h,
            _inference_script(h, engine_model, engine_container, kind),
            timeout=90,
        )

    with ThreadPoolExecutor(max_workers=12) as ex:
        if trainer_only:
            f_head = ex.submit(
                _ssh,
                trainer_head,
                trainer_only_probe.replace("__TRAINER__", trainer_container),
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
                orchestrator_probe.replace("__ORCH__", orchestrator_container),
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
        "BATCH_SYNC",
        "TRAINER_FAIL_WD",
        "ORCH_LAST",
        "BATCH",
        "FAILS5M",
        "ORCH_LOGAGE",
    )
    for key in edge_keys:
        h.pop(key, None)
    if edge_orch_rc == 0:
        h.update({key: edge_orch[key] for key in edge_keys if key in edge_orch})
    s.tunnels_ok = int(h.get("TUNNELS_OK", 0) or 0)
    s.mode = h.get("MODE", "rl")
    s.tunnels_bad = [int(p) for p in h.get("TUNNELS_BAD", "").split()]
    head_trainer_up = h.get("TRAINER_UP", "") == "up"
    s.orch_up = h.get("ORCH_UP", "") == "up"
    s.batch_sync = int(h.get("BATCH_SYNC", 0) or 0) > 0
    s.trainer_failure_watchdog = int(h.get("TRAINER_FAIL_WD", 0) or 0) > 0
    s.trainer_steps = int(h.get("TRAINER_STEPS", 0) or 0)
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
    s.supervisor_tunnel = int(h.get("SUP_TUN", 0) or 0) > 0
    s.supervisor_tunnel_log_age = _age(h.get("SUP_TUN_LOG", ""))

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
        s.engines.append(e)

    s.warnings = _warnings(s)
    if not quiet:
        print(s.report())
    return s


def _warnings(s: Status) -> list[str]:
    w = []
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
    if not trainer_only and not s.orch_up:
        w.append("orchestrator container is not up")
    if not trainer_only and s.tunnels_ok <= 20:
        w.append(
            f"only {s.tunnels_ok}/{len(inference_tunnel_ports)} tunnels answer. Check the engines before you believe this. "
            "If they are generating, the probe is starved by load, not by dead tunnels."
        )
    elif not trainer_only and s.tunnels_ok < 36:
        w.append(
            f"{s.tunnels_ok}/{len(inference_tunnel_ports)} tunnels answer. "
            "Compare against the engine concurrency below."
        )
    inflight = re.search(r"(\d+) inflight", s.batch or "")
    live = [e for e in s.engines if e.reachable]
    if inflight and int(inflight.group(1)) > 40 and live:
        tput = sum(e.throughput for e in live)
        if tput < 5.0 and sum(e.running for e in live) == 0:
            w.append(
                f"{inflight.group(1)} rollouts in flight but the engines generate {tput:.1f} tok/s "
                "and run nothing. Sample again before acting. If it holds, the requests are queued "
                "in the tunnels; lower max_inflight_rollouts, which is independent of batch_size."
            )
    for e in (() if trainer_only else s.engines):
        if e.reachable and not e.healthy:
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
    if not trainer_only and not s.supervisor_tunnel:
        w.append("tunnel watchdog is not running")
    elif not trainer_only and s.supervisor_tunnel_log_age and s.supervisor_tunnel_log_age > 300:
        w.append(
            f"tunnel watchdog process exists but its log is {_hms(s.supervisor_tunnel_log_age)} old; "
            "a process that writes no log line does not work"
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
    for name, age in ages:
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


def tunnels() -> dict[int, str]:
    """HTTP code for each rollout tunnel port."""
    rc, out = _ssh(
        trainer_head,
        f'for p in $(seq {inference_tunnel_ports[0]} {inference_tunnel_ports[-1]}); do printf "%s " $p; '
        'curl -s -o /dev/null -m 6 -w "%{http_code}\\n" http://127.0.0.1:$p/v1/models 2>/dev/null || echo 000; done',
        timeout=240,
    )
    if rc:
        raise RuntimeError(f"cannot check tunnels on {trainer_head}: {out.strip()}")
    res = {port: "000" for port in inference_tunnel_ports}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            res[int(p[0])] = p[1]
    ok = sum(1 for v in res.values() if v == "200")
    print(
        f"tunnels {ok}/{len(res)} answering; bad: {[p for p, v in res.items() if v != '200']}"
    )
    return res


def engines(run: str = "") -> dict[str, EngineState]:
    """Probe every configured inference host with a real completion."""
    run = run or current_run()
    kind = _profile_for_run(run)["kind"]
    muse_run = kind == "muse"
    engine_model = muse_model if muse_run else glm_model
    engine_container = muse_inference_container if muse_run else glm_inference_container
    states = []
    for host, (rc, out) in _fan(
        inference_hosts,
        lambda h: _ssh(
            h, _inference_script(h, engine_model, engine_container, kind), timeout=90
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
        states.append(e)
    for e in states:
        print(
            f"{e.host}: {e.probes_ok}/{e.probes_total} probes in {e.gen_ms} ms, "
            f"running={e.running}, {e.throughput} tok/s"
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
    match = re.fullmatch(r"muse-?(\d+)", run)
    if not match:
        raise Refused("adapter archive requires a Muse run such as muse19")
    return f"muse-{int(match.group(1))}-run"


def _local_command(command: list[str], label: str, timeout: int) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} timed out") from error
    if result.returncode:
        detail = (result.stderr or result.stdout or "failed").strip().splitlines()[-1]
        raise RuntimeError(f"{label} failed: {detail[:160]}")
    return result


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
    script = f"""
path=$(find {shlex.quote(f'{run_root}/{run}')} -maxdepth 3 -type f -name trainer-metrics.jsonl -print 2>/dev/null | head -1)
[ -n "$path" ] || exit 4
python3 - "$path" {int(limit)} <<'PY'
import json,sys
rows=[]
invalid=0
for line in open(sys.argv[1]):
    try:
        row=json.loads(line)
    except ValueError:
        invalid+=1
        continue
    if isinstance(row.get('step'), (int,float)) and isinstance(row.get('time/forward_backward'), (int,float)):
        rows.append({{
            'step':int(row['step']),
            'forward_backward_seconds':float(row['time/forward_backward']),
            'wait_seconds':float(row.get('time/wait_for_batch') or 0),
            'tokens_per_second':float(row.get('perf/throughput') or 0),
        }})
print(json.dumps({{'rows':rows[-int(sys.argv[2]):], 'invalid_lines':invalid}}))
PY
"""
    code, output = _ssh(trainer_head, script, timeout=180)
    if code:
        raise RuntimeError(f"cannot read trainer steps on {trainer_head}: {output.strip()}")
    try:
        result = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("trainer step output is invalid") from error
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("trainer step output has no row list")
    print("step  forward/backward  wait  tokens/s")
    for row in rows:
        print(
            f"{row['step']:>4}  {_hms(row['forward_backward_seconds']):>16}  "
            f"{_hms(row['wait_seconds']):>4}  {row['tokens_per_second']:.1f}"
        )
    invalid = int(result.get("invalid_lines") or 0)
    if invalid:
        print(f"invalid metric lines: {invalid}")
    return rows


def package_versions(*names: str) -> dict[str, str]:
    """Show installed versions without importing the packages."""
    import importlib.metadata

    packages = names or (
        "transformers",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
        "verifiers",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"package is not installed: {name}") from error
    for name, version in versions.items():
        print(f"{name} {version}")
    return versions


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
            f"baseline {baseline_run}: standard deviation {baseline_std:.4f} at batch 256 over 36 steps"
        )
        L.append(
            f"fit: {self.total_change:+.4f} over the window, residual spread {self.residual_std:.4f}, "
            f"threshold {trend_spread_multiplier:g} x residual = {trend_spread_multiplier * self.residual_std:.4f}"
        )
        L.append(f"result: {self.verdict}")
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
        + "python3 - \"$d\" <<'PY'\n"
        "import json,sys\n"
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
        "    out.append({'step':r['step'],**{k:r.get(v) for k,v in K.items()}})\n"
        "print('REWARD_INVALID '+str(invalid))\n"
        "print('REWARD_JSON '+json.dumps(out))\n"
        "PY\n"
    )
    code, output = _ssh(metric_host, script, timeout=180)
    if code:
        raise RuntimeError(f"cannot read reward data on {metric_host}: {output.strip()}")
    rep = RewardReport(run=run)
    rows = None
    invalid_lines = 0
    for line in output.splitlines():
        if line.startswith("REWARD_INVALID "):
            try:
                invalid_lines = int(line.removeprefix("REWARD_INVALID "))
            except ValueError as error:
                raise RuntimeError("reward invalid-line count is not an integer") from error
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
    if len(r.steps) >= minimum_trend_steps and r.std:
        cmp = "below" if r.std < baseline_std else "above"
        notes.append(
            f"Step-to-step spread is {r.std:.4f}, {cmp} the {baseline_run} baseline of {baseline_std:.4f}. "
            "That comparison is the reason the batch size was raised."
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
        key = (s.trainer_steps, s.batch)
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
                        "forward/backward or off-policy backpressure is active.",
                        flush=True,
                    )
                else:
                    print(
                        f"stalled: step count and batch progress have not changed for {_hms(still)}. "
                        "Read references/errors.md and repair before waiting further.",
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
    selected = _profile_for_run(run)
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


def repair(what: str) -> str:
    """Apply a named repair to the configured hosts."""
    if what == "tunnels":
        return _repair_tunnels()
    if what.startswith("engine:"):
        return _repair_engine(what.split(":", 1)[1])
    if what == "supervisors":
        return _repair_supervisors()
    if what == "microbatch-sync":
        return _repair_microbatch_sync()
    raise Refused(
        f"unknown repair {what!r}. Use tunnels, engine:<host>, supervisors, or microbatch-sync."
    )


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


def _repair_tunnels() -> str:
    rc, out = _ssh(
        trainer_head,
        f"k=0; for p in $(seq {inference_tunnel_ports[0]} {inference_tunnel_ports[-1]}); do "
        'c=$(curl -s -o /dev/null -m 6 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null); '
        'if [ "$c" != "200" ]; then '
        'pid=$(sudo ss -H -lntp "sport = :$p" | sed -n "s/.*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1); '
        '[ -n "$pid" ] && sudo kill $pid && k=$((k+1)); fi; done; echo "killed $k"',
        timeout=300,
    )
    if rc:
        raise RuntimeError(f"tunnel repair failed on {trainer_head}: {out.strip()}")
    msg = out.strip().splitlines()[-1] if out.strip() else "no output"
    print(f"tunnels: {msg}. The H200 supervisors rebuild within about 5 seconds.")
    print(f"The batch tunnel on port {batch_port} was not touched.")
    return msg


def _repair_engine(host: str) -> str:
    _check_host(host)
    if host not in inference_hosts:
        raise Refused(f"{host} is not an engine host. Use one of {inference_hosts}.")
    muse_engine = _profile()["kind"] == "muse"
    engine_container = muse_inference_container if muse_engine else glm_inference_container
    engine_model = muse_model if muse_engine else glm_model
    start_script = (
        f"{runtime_root}/h200/start_muse_inference.sh"
        if muse_engine
        else f"{runtime_root}/h200/start_inference.sh"
    )
    startup_grace = 600 if muse_engine else 180
    print(
        f"{host}: removing and restarting {engine_container}. The model load needs 5 to 10 minutes."
    )
    stop_code, stop_output = _ssh(
        host,
        f"pkill -f '^bash {runtime_root}/h200/engine_watchdog.sh$' 2>/dev/null; echo stopped",
    )
    if stop_code:
        raise RuntimeError(f"cannot stop the watchdog on {host}: {stop_output.strip()}")
    rc, out = _ssh(
        host,
        f"sudo docker rm -f {engine_container} >/dev/null 2>&1; "
        f"nohup setsid bash {start_script} >/root/start_inference.log 2>&1 </dev/null & "
        "sleep 3; echo issued",
        timeout=180,
    )
    if rc:
        raise RuntimeError(f"inference restart failed on {host}: {out.strip()}")
    if muse_engine:
        settings = _inference_config(host, "muse")
        watchdogs = []
        for replica, port in enumerate(settings["ports"]):
            watchdogs.append(
                f"nohup setsid env STARTUP_GRACE={startup_grace} MODEL={engine_model} "
                f"MUSE_GPU_COUNT={settings['gpu_count']} "
                f"MUSE_TP_SIZE={settings['tensor_parallel_size']} "
                f"PORT={port} MUSE_REPLICA_INDEX={replica} START={start_script} "
                f"ENGINE_CONTAINER={engine_container}-{replica} "
                f"bash {runtime_root}/h200/engine_watchdog.sh "
                f">> /root/engine_watchdog-{replica}.log 2>&1 </dev/null &"
            )
        watchdog_code, watchdog_output = _ssh(
            host, " ".join(watchdogs) + " disown"
        )
    else:
        watchdog_code, watchdog_output = _ssh(
            host,
            f"nohup setsid env STARTUP_GRACE={startup_grace} MODEL={engine_model} "
            f"START={start_script} ENGINE_CONTAINER={engine_container} "
            f"bash {runtime_root}/h200/engine_watchdog.sh "
            ">> /root/engine_watchdog.log 2>&1 </dev/null & disown",
        )
    if watchdog_code:
        raise RuntimeError(
            f"watchdog restart failed on {host}: {watchdog_output.strip()}"
        )
    print("Do not test with /v1/models. Test with engines() when the load finishes.")
    return out.strip()


def _repair_supervisors(profile: str = "current") -> str:
    out = []
    for pat, cmd, log in (
        (
            "bash .*tunnel_watchdog.sh",
            f"nohup setsid env ports={shlex.quote(' '.join(map(str, inference_tunnel_ports)))} "
            f"timeout=20 interval=45 bash {runtime_root}/china/tunnel_watchdog.sh "
            ">> /data/glm52-rl/tunnel_watchdog.log 2>&1 </dev/null & disown",
            "/data/glm52-rl/tunnel_watchdog.log",
        ),
    ):
        if not alive(pat, trainer_head, log):
            code, restart_output = _ssh(trainer_head, cmd)
            if code:
                raise RuntimeError(
                    f"cannot restart {pat} on {trainer_head}: {restart_output.strip()}"
                )
            out.append(f"restarted {pat}")
        else:
            out.append(f"{pat} already runs")
    for h in inference_hosts:
        muse_engine = _profile(profile)["kind"] == "muse"
        rc, count_text = _ssh(
            h,
            f"pgrep -f '^bash {runtime_root}/h200/engine_watchdog.sh$' | wc -l",
        )
        if rc or not count_text.strip().isdigit():
            raise RuntimeError(f"cannot read watchdog count on {h}: {count_text.strip()}")
        count = int(count_text.strip())
        if muse_engine:
            settings = _inference_config(h, "muse")
            expected = len(settings["ports"])
        else:
            expected = 1
        if count == expected:
            out.append(f"{expected} engine watchdog(s) already run on {h}")
            continue
        stop_code, stop_output = _ssh(
            h,
            f"pkill -f '^bash {runtime_root}/h200/engine_watchdog.sh$' 2>/dev/null; true",
        )
        if stop_code:
            raise RuntimeError(f"cannot stop watchdogs on {h}: {stop_output.strip()}")
        if muse_engine:
            start_script = f"{runtime_root}/h200/start_muse_inference.sh"
            watchdogs = []
            for replica, port in enumerate(settings["ports"]):
                watchdogs.append(
                    f"nohup setsid env STARTUP_GRACE=600 MODEL={muse_model} "
                    f"MUSE_GPU_COUNT={settings['gpu_count']} "
                    f"MUSE_TP_SIZE={settings['tensor_parallel_size']} "
                    f"PORT={port} MUSE_REPLICA_INDEX={replica} START={start_script} "
                    f"ENGINE_CONTAINER={muse_inference_container}-{replica} "
                    f"bash {runtime_root}/h200/engine_watchdog.sh "
                    f">> /root/engine_watchdog-{replica}.log 2>&1 </dev/null &"
                )
            start_code, start_output = _ssh(h, " ".join(watchdogs) + " disown")
        else:
            start_code, start_output = _ssh(
                h,
                f"nohup setsid bash {runtime_root}/h200/engine_watchdog.sh >> /root/engine_watchdog.log 2>&1 </dev/null & disown",
            )
        if start_code:
            raise RuntimeError(f"cannot start watchdogs on {h}: {start_output.strip()}")
        out.append(f"restarted {expected} engine watchdog(s) on {h}")
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
        repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "china" / "cluster.sh",
        repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_orchestrator.sh",
        repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_inference.sh",
        repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_muse_inference.sh",
    )
    return sorted(set().union(*(_mounts_in(script) for script in scripts)))


def _mounted_on(path: str, host: str) -> bool:
    if host in inference_hosts:
        scripts = (
            repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_orchestrator.sh",
            repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_inference.sh",
            repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "h200" / "start_muse_inference.sh",
        )
    else:
        scripts = (repo_root / ".agents" / "skills" / "training-ops" / "runtime" / "china" / "cluster.sh",)
    return any(path in _mounts_in(script) for script in scripts)


def _repository_root(name: str = "default") -> str:
    """Resolve the configured repository root."""
    if name != "default":
        raise Refused(f"unknown repository {name!r}; choose default")
    root = paths.get("remote_repository")
    if not isinstance(root, str) or not root.startswith("/"):
        raise Refused(f"repository {name!r} is missing from operations config")
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
        elif path.startswith((".agents/skills/training-ops/runtime/h200/", "configs/runtime/")):
            nodes = inference_hosts
        elif path.startswith(".agents/skills/training-ops/runtime/china/"):
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
        try:
            mkdir = subprocess.run(
                ssh_command
                + [
                    h,
                    f"mkdir -p {shlex.quote(str(Path(remote).parent))}",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=deployment_mkdir_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "mkdir TIMEOUT"
        if mkdir.returncode:
            detail = ((mkdir.stdout or "") + (mkdir.stderr or "")).strip()
            return f"mkdir failed: {detail[:80]}"
        try:
            p = subprocess.run(
                scp_command + [str(local), f"{h}:{remote}"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
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
    bundle = directory / f"trainer-dependencies-{directory.name}.tar"
    with tarfile.open(bundle, mode="w") as archive:
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
  [ -f "$destination" ] && [ ! -L "$destination" ] || die "invalid installed path: $member"
  source_digest=$(sha256sum -- "$source" | cut -d' ' -f1) || die "cannot checksum $member"
  got=$(sha256sum -- "$destination" | cut -d' ' -f1) || die "cannot verify $member"
  [ "$got" = "$source_digest" ] || die "installed checksum mismatch: $member"
done
echo "installed ${{#install_paths[@]}} trainer dependency files"
"""


def _dependency_checksum_script(
    repository_root: str, expected: list[tuple[str, str]]
) -> str:
    """Build one read-only remote check for the expected dependency files."""
    lines = [
        "set -euo pipefail",
        f"repository={shlex.quote(repository_root)}",
    ]
    for path, digest in expected:
        lines.extend(
            [
                f"file=\"$repository\"/{shlex.quote(path)}",
                '[ -f "$file" ] && [ ! -L "$file" ]',
                'got=$(sha256sum -- "$file" | cut -d\' \' -f1)',
                f'[ "$got" = {shlex.quote(digest)} ]',
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

        def one(host: str) -> str:
            selected = [
                path for path, selected_hosts in destinations.items() if host in selected_hosts
            ]
            expected = [(path, manifest_entries[path]) for path in selected]
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
                try:
                    if attempt == 1:
                        upload_command = scp_command + [
                            str(bundle),
                            f"{host}:{remote_archive}",
                        ]
                        upload_stdin = subprocess.DEVNULL
                    else:
                        upload_command = ssh_command + [
                            host,
                            f"umask 077; cat > {shlex.quote(remote_archive)}",
                        ]
                        upload_stdin = bundle.open("rb")
                    try:
                        uploaded = subprocess.run(
                            upload_command,
                            stdin=upload_stdin,
                            capture_output=True,
                            text=True,
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
            if local.is_file():
                add(local.relative_to(repo_root).as_posix(), destinations)

    add_files(
        ".agents/skills/training-ops/runtime/china",
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
            if local.is_file() and (suffix is None or local.suffix == suffix):
                add(local.relative_to(repo_root).as_posix(), destinations)

    trainer_destinations = list(dict.fromkeys([*nodes, trainer_head]))
    add_files(
        ".agents/skills/training-ops/runtime/china",
        trainer_destinations,
    )
    add_files(
        ".agents/skills/training-ops/runtime/h200",
        inference_hosts,
    )

    configured_hosts = list(dict.fromkeys([*trainer_hosts, *inference_hosts]))
    for path in mounted_files():
        if not path.startswith("prime-rl/"):
            continue
        destinations = [host for host in configured_hosts if _mounted_on(path, host)]
        add(path, destinations)

    add(selected["trainer"], nodes)
    add(selected["orchestrator"], [orchestrator_host])
    for root in (
        "environments/decensor/src",
        "environments/ornith/src",
        "evals/refusals-classifier",
    ):
        add_files(root, [orchestrator_host], suffix=".py")

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


def _cold_cleanup_script(run: str) -> str:
    """Build a bounded remote cleanup for one configured run only."""
    runs_path, run_parent, run_dir = _cold_run_paths(run)
    sync_script = f"{runtime_root}/china/sync_microbatches.sh"
    return f"""set -euo pipefail
runs_path={shlex.quote(runs_path)}
run_parent={shlex.quote(run_parent)}
run_dir={shlex.quote(run_dir)}
sync_script={shlex.quote(sync_script)}

# A PID file alone is not enough: the PID can be stale or reused. Require the
# exact run_dir environment and sync command before sending a signal.
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
  # lose the file during a failed startup, so inspect every configured service
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
  scan_run_dir "$run_dir"
  for service_dir in "$runs_path"/*/run_*; do
    [ "$service_dir" = "$run_dir" ] && continue
    case "$service_dir" in
      "$runs_path"/*/run_*) scan_run_dir "$service_dir" ;;
    esac
  done
  for pid_file in "$runs_path"/*/run_*/.microbatch-sync.pid; do
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
    """Remove stale generated state for one run before a cold start."""
    _cold_run_paths(run)  # Validate before any host action.
    targets = list(dict.fromkeys([*nodes, orchestrator_host]))
    targets = [_check_host(host) for host in targets]
    script = _cold_cleanup_script(run)

    def one(host: str) -> tuple[int, str]:
        return _ssh(host, script, timeout=300)

    results = _fan(targets, one)
    failures = []
    for host, (code, output) in results.items():
        detail = " | ".join(line.strip() for line in output.splitlines() if line.strip())
        if code:
            failures.append(f"{host}: {detail or f'exit {code}'}")
        else:
            print(f"cold cleanup {run} on {host}: {detail or 'complete'}")
    if failures:
        raise RuntimeError("cold cleanup failed: " + "; ".join(failures))


def start(
    run: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
    supervisors: bool = True,
    resume_adapter: str | None = None,
    resume_step: int | None = None,
    profile: str = "current",
) -> None:
    """Start a run from one configured profile."""
    if mode not in ("cold", "restart"):
        raise Refused('mode must be "cold" or "restart"')
    _cold_run_paths(run)
    if (resume_adapter is None) != (resume_step is None):
        raise Refused("resume_adapter and resume_step must be set together")
    selected = _profile(profile)
    if "orchestrator" not in selected:
        raise Refused(f"profile {profile!r} is trainer-only; use qualify()")
    muse_run = selected["kind"] == "muse"
    if resume_adapter is not None:
        if not muse_run:
            raise Refused("adapter-only resume is supported only for Muse runs")
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
    if muse_run:
        env = (
            f"{runtime_env} trainer_nodes='{trainer_nodes}' RUN={run} RUN_NAME={name} "
            f"TRAINER_CONFIG={trainer_config} PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 "
            "PRIME_INT4_COMBINE_ROUNDS=1 PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=0 "
            "TRANSFORMERS_OVERRIDE_DIR=/data/muse-glimmer/transformers "
            "MUSE_PYTHON_OVERRIDE_DIR=/data/muse-glimmer/python-overrides"
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
    else:
        env = (
            f"{runtime_env} trainer_nodes='{trainer_nodes}' RUN={run} RUN_NAME={name} "
            f"TRAINER_CONFIG={trainer_config} PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 "
            "PRIME_INT4_COMBINE_ROUNDS=8 PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=1"
        )

    def launch_trainer():
        print(f"starting trainer on {len(nodes)} nodes")
        results = _fan(
            nodes,
            lambda h: _ssh(
                h,
                f"node_id={shlex.quote(h.removeprefix('n'))} {env} "
                f"bash {trainer_launcher} trainer >/dev/null 2>&1; echo done",
                300,
            ),
        )
        failed = [host for host, (code, _output) in results.items() if code]
        if failed:
            raise RuntimeError(f"trainer start failed on {failed}")

    def launch_adapter():
        print(f"starting adapter server on {trainer_head}")
        node_id = trainer_head.removeprefix("n")
        rc, out = _ssh(
            trainer_head,
            f"node_id={shlex.quote(node_id)} {runtime_env} RUN={shlex.quote(run)} "
            f"bash {trainer_launcher} adapter-http >/dev/null 2>&1; echo done",
            300,
        )
        if rc:
            raise RuntimeError(f"adapter server start failed on {trainer_head}: {out}")

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

    print(f"stopping any previous containers for {run}")
    _remote_ok(
        trainer_head,
        "pkill -f 'bash .*[r]un_supervisor.sh' 2>/dev/null || true; echo done",
        "old local supervisor stop",
    )
    _remote_ok(
        orchestrator_host,
        f"pkill -f '[s]ync_rollout_batches.sh' 2>/dev/null; "
        f"sudo docker rm -f {orchestrator_container} >/dev/null 2>&1; echo done",
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
    launch_adapter()
    if mode == "cold":
        launch_orch()
        launch_trainer()
    else:
        launch_trainer()
        time.sleep(60)
        launch_orch()

    if supervisors:
        _repair_supervisors(profile)
    print(f"started {run} in {mode} mode. Call status() in a few minutes.")


def qualify(
    name: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
) -> None:
    """Start one configured trainer-only qualification."""
    if mode not in ("cold", "restart"):
        raise Refused('qualification mode must be "cold" or "restart"')
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("qualification name can contain only letters, digits, '_' and '-'")
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    if trainer_head in nodes:
        nodes = [trainer_head] + [host for host in nodes if host != trainer_head]

    selected = _profile(name)
    launch_env = _qualification_launch_env(selected)
    if "orchestrator" in selected:
        raise Refused(f"profile {name!r} is a full run; use start()")
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
    muse = selected["kind"] == "muse"

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
    if muse:
        env += (
            " TRANSFORMERS_OVERRIDE_DIR=/data/muse-glimmer/transformers"
            " MUSE_PYTHON_OVERRIDE_DIR=/data/muse-glimmer/python-overrides"
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
            f"bash {repository_root}/.agents/skills/training-ops/runtime/china/cluster.sh "
            "trainer >/dev/null 2>&1; echo done",
            300,
        ),
    )
    failed = [host for host, (rc, _out) in results.items() if rc]
    if failed:
        raise RuntimeError(f"qualification trainer launch failed on {failed}")
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
        f"sudo docker rm -f {orchestrator_container} >/dev/null 2>&1; echo done",
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
        profile=arguments[2] if len(arguments) > 2 else "current",
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
    "tunnels": lambda a: tunnels(),
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
    "qualify": lambda a: qualify(*a),
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
  tunnels                       check {len(inference_tunnel_ports)} inference tunnels
  engines                       test every configured inference endpoint
  logs [role] [lines]           read one service log
  parity [run]                  read the trainer parity record
  remote <group> <command>      run on trainer, inference, or all hosts
  versions [package ...]        show installed package versions
  watch [hours] [seconds]       monitor in the foreground
  repair <name>                 repair tunnels, an engine, supervisors, or microbatch sync
  deploy <path>                 copy one repository file
  start <run> [mode] [profile]  start one configured profile
  archive <run> [step ...]      upload stable old LoRA adapters
                                add --finished after another run is active
  qualify <profile>             start a trainer-only profile
  mounts                        list live-mounted source files
  stop                          stop the run but keep inference loaded

Read {Path(__file__).resolve().parents[2] / 'SKILL.md'} first.
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
