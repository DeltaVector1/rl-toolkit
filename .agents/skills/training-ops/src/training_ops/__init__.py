"""Read and operate the configured training system."""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
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
isolation = operations["isolation"]
models = operations["models"]
inference_settings = operations["inference"]
reward_settings = operations["reward"]
limits = operations["limits"]
shared_requirements = operations["requirements"]["shared"]
muse_requirements = operations["requirements"]["muse"]
glm_requirements = operations["requirements"]["glm52"]
glm_parity_paths = operations["parity"]["glm52"]
parity_trace_root = "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces"
parity_token_root = "/data/glm52-rl/qualifications/glm52-policy-parity/trainer/token_exports"

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
h200_parity_backend_port = ports["h200_parity_backend"]
remote_repo = paths["remote_repository"]
glm_parity_inference_config = (
    f"{remote_repo}/configs/qualification/glm52-policy-parity-h200.toml"
)
glm_parity_vllm_extra = (
    '{"kv_cache_dtype":"fp8","max_num_seqs":128,'
    '"max_num_batched_tokens":32769,'
    '"compilation_config":{"max_cudagraph_capture_size":256}}'
)
run_root = paths["runs"]
trainer_launcher = f"{remote_repo}/scripts/china/cluster.sh"
orchestrator_launcher = f"{remote_repo}/scripts/h200/start_orchestrator.sh"
trainer_container = containers["trainer"]
trainer_image = containers["trainer_image"]
inference_image = containers["inference_image"]
orchestrator_container = containers["orchestrator"]
environment_container = containers["environment"]
glm_inference_container = containers["glm52_inference"]
muse_inference_container = containers["muse_inference"]
glm_model_volume = containers["glm52_model_volume"]
glm_model = models["glm52"]
muse_model = models["muse"]
protected_repository = isolation["protected_repository"]
protected_trainer = isolation["protected_trainer"]
protected_orchestrator = isolation["protected_orchestrator"]
protected_inference = list(isolation["protected_inference"])

if remote_repo == protected_repository:
    raise RuntimeError("the GLM optimization deployment must not be the live deployment")
if trainer_container == protected_trainer or orchestrator_container == protected_orchestrator:
    raise RuntimeError("the GLM optimization containers must not use live container names")
baseline_std = reward_settings["reference_spread"]
baseline_run = reward_settings["reference_run"]
minimum_trend_steps = reward_settings["minimum_steps"]
trend_spread_multiplier = reward_settings["spread_multiplier"]
stall_seconds = limits["stall_seconds"]
final_validation_steps = limits["final_steps"]
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


def _configured_h200_engine_startup_grace() -> int:
    value = limits.get("h200_engine_startup_grace_seconds")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Refused(
            "limits.h200_engine_startup_grace_seconds must be a positive integer"
        )
    return value


def _configured_live_service_check() -> tuple[int, int, int, int, int, int, int]:
    settings = limits.get("live_service_check")
    if not isinstance(settings, dict):
        raise Refused("limits.live_service_check must be configured")
    names = (
        "attempts",
        "connect_timeout_seconds",
        "command_timeout_seconds",
        "retry_delay_seconds",
        "workers",
        "trainer_budget_seconds",
        "inference_budget_seconds",
    )
    values = tuple(settings.get(name) for name in names)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Refused("live service check limits must be integers")
    (
        attempts,
        connect_timeout,
        command_timeout,
        retry_delay,
        workers,
        trainer_budget,
        inference_budget,
    ) = values
    if not 1 <= attempts <= 3:
        raise Refused("live service check attempts must be from 1 to 3")
    if not 1 <= connect_timeout <= 120:
        raise Refused("live service check connect timeout must be from 1 to 120 seconds")
    if not connect_timeout < command_timeout <= 120:
        raise Refused(
            "live service check command timeout must exceed its connect timeout and be at most 120 seconds"
        )
    if not 0 <= retry_delay <= 10:
        raise Refused("live service check retry delay must be from 0 to 10 seconds")
    if not 1 <= workers <= len(trainer_hosts):
        raise Refused("live service check workers must fit the trainer inventory")
    if not 1 <= trainer_budget <= 300:
        raise Refused("live trainer service check budget must be from 1 to 300 seconds")
    if not 1 <= inference_budget <= 180:
        raise Refused("live inference service check budget must be from 1 to 180 seconds")
    if trainer_head not in trainer_hosts:
        raise Refused("live service check trainer head is outside the trainer inventory")
    per_host = attempts * command_timeout + (attempts - 1) * retry_delay
    remaining_trainers = len(trainer_hosts) - 1
    trainer_stages = 1 + math.ceil(remaining_trainers / workers)
    trainer_worst_case = trainer_stages * per_host
    if trainer_worst_case > trainer_budget:
        raise Refused(
            f"live trainer service check worst case is {trainer_worst_case} seconds; "
            f"budget is {trainer_budget} seconds"
        )
    if per_host > inference_budget:
        raise Refused(
            f"live inference service check worst case is {per_host} seconds; "
            f"budget is {inference_budget} seconds"
        )
    return (
        attempts,
        connect_timeout,
        command_timeout,
        retry_delay,
        workers,
        trainer_budget,
        inference_budget,
    )

adapter_archive_bucket = paths["adapter_bucket"]
adapter_archive_files = ("adapter_config.json", "adapter_model.safetensors")

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
    environment = profile.get("environment")
    if orchestrator is not None and (
        not isinstance(environment, str) or not (repo_root / environment).is_file()
    ):
        raise RuntimeError(f"profile {name!r} has no readable environment config")
    return profile

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
    "StrictHostKeyChecking=accept-new",
]


class Refused(RuntimeError):
    """Raised when a call would break one of the rules in SKILL.md."""


# ------------------------------------------------------------------ plumbing


def _check_host(host: str) -> str:
    for bad in blocked_hosts:
        if re.search(rf"\b{bad}\b", host):
            raise Refused(f"{bad} is off limits. The owner said so with no exception.")
    return host


def _check_parity_host(host: str) -> str:
    _check_host(host)
    if host not in trainer_hosts:
        raise Refused(f"parity host is not in the configured trainer list: {host}")
    return host


def _ssh(
    host: str,
    script: str,
    timeout: int = 120,
    connect_timeout: int | None = None,
) -> tuple[int, str]:
    _check_host(host)
    command = list(ssh_command)
    if connect_timeout is not None:
        if (
            isinstance(connect_timeout, bool)
            or not isinstance(connect_timeout, int)
            or not 1 <= connect_timeout <= 120
        ):
            raise Refused("SSH connect timeout must be an integer from 1 to 120 seconds")
        timeout_options = [
            index
            for index, value in enumerate(command)
            if value.startswith("ConnectTimeout=")
        ]
        if len(timeout_options) != 1:
            raise Refused("SSH connect timeout inventory is invalid")
        command[timeout_options[0]] = f"ConnectTimeout={connect_timeout}"
    try:
        p = subprocess.run(
            command + [host, script],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = p.stdout if p.returncode == 0 else p.stdout + p.stderr
        return p.returncode, output
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _remote_ok(
    host: str,
    script: str,
    label: str,
    timeout: int = 120,
    connect_timeout: int | None = None,
) -> str:
    if connect_timeout is None:
        code, output = _ssh(host, script, timeout)
    else:
        code, output = _ssh(
            host, script, timeout, connect_timeout=connect_timeout
        )
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


def _assert_live_services_are_down(nodes: list[str], include_inference: bool) -> None:
    """Refuse an isolated launch while the live deployment uses the same GPUs."""

    active: list[str] = []
    (
        attempts,
        connect_timeout,
        command_timeout,
        retry_delay,
        workers,
        trainer_budget,
        inference_budget,
    ) = _configured_live_service_check()

    def check(host: str, script: str, deadline: float) -> tuple[int, str]:
        result = (1, "live service check did not run")
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 124, "live service check aggregate deadline exceeded"
            result = _ssh(
                host,
                script,
                timeout=min(command_timeout, max(1, math.ceil(remaining))),
                connect_timeout=connect_timeout,
            )
            code, output = result
            if code == 0 or not (
                code == 255 and "banner exchange" in output.lower()
            ):
                break
            if attempt + 1 < attempts:
                remaining = deadline - time.monotonic()
                if retry_delay >= remaining:
                    return 124, "live service check aggregate deadline exceeded"
                time.sleep(retry_delay)
        return result

    trainer_deadline = time.monotonic() + trainer_budget

    def trainer_active(host: str) -> tuple[int, str]:
        return check(
            host,
            f"sudo docker ps --filter name=^{protected_trainer}$ --format '{{{{.Names}}}}'",
            trainer_deadline,
        )

    remaining_nodes = list(nodes)
    if trainer_head in remaining_nodes:
        code, output = trainer_active(trainer_head)
        if code:
            raise RuntimeError(
                f"cannot check the live trainer on {trainer_head}: {output.strip()}"
            )
        if output.strip():
            raise Refused(
                "the live deployment is active on the requested GPUs: "
                f"{trainer_head}:{output.strip()}"
            )
        remaining_nodes = [host for host in remaining_nodes if host != trainer_head]

    trainer_results = _fan(remaining_nodes, trainer_active, workers=workers)
    for host, (code, output) in trainer_results.items():
        if code:
            raise RuntimeError(f"cannot check the live trainer on {host}: {output.strip()}")
        if output.strip():
            active.append(f"{host}:{output.strip()}")
    if active:
        raise Refused(
            "the live deployment is active on the requested GPUs: "
            + ", ".join(active)
        )
    if time.monotonic() > trainer_deadline:
        raise RuntimeError("live trainer service check exceeded its aggregate deadline")

    if include_inference:
        inference_deadline = time.monotonic() + inference_budget
        names = [protected_orchestrator, *protected_inference]
        expression = "|".join(re.escape(name) for name in names)
        code, output = check(
            orchestrator_host,
            "sudo docker ps --format '{{.Names}}' | "
            f"grep -E '^({expression})(-[0-9]+)?$' || true",
            inference_deadline,
        )
        if code:
            raise RuntimeError(
                f"cannot check live inference services on {orchestrator_host}: {output.strip()}"
            )
        active.extend(f"{orchestrator_host}:{name}" for name in output.split())
        if time.monotonic() > inference_deadline:
            raise RuntimeError("live inference service check exceeded its deadline")

    if active:
        raise Refused(
            "the live deployment is active on the requested GPUs: "
            + ", ".join(active)
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
echo "SUP_RUN $(pgrep -f 'bash .*[r]un_supervisor.sh' | wc -l)"
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


def _inference_script(host: str, model: str, container: str) -> str:
    settings = _inference_config(host)
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
    supervisor_run: bool = False
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
                f"  supervisors: run={'yes' if self.supervisor_run else 'no'} "
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
        rep.native_fp8 = (
            quant_type == "fp8"
            and dequantize_checkpoint is False
            and not any(flag is True for flag in optional_dequant_flags)
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
    kind = _profile()["kind"]
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
    muse_run = _profile()["kind"] == "muse"
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
            _inference_script(h, engine_model, engine_container),
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
    s.supervisor_run = int(h.get("SUP_RUN", 0) or 0) > 0
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
    if not trainer_only and not s.supervisor_run:
        w.append("fail-closed run supervisor is not running")
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
    muse_run = _profile()["kind"] == "muse"
    engine_model = muse_model if muse_run else glm_model
    engine_container = muse_inference_container if muse_run else glm_inference_container
    states = []
    for host, (rc, out) in _fan(
        inference_hosts,
        lambda h: _ssh(h, _inference_script(h, engine_model, engine_container), timeout=90),
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
        + (f"tail -1 {log} 2>/dev/null" if log else ""),
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


def repair(what: str) -> str:
    """Apply a named repair to the configured hosts."""
    if what == "tunnels":
        return _repair_tunnels()
    if what.startswith("engine:"):
        return _repair_engine(what.split(":", 1)[1])
    if what == "supervisors":
        return _repair_supervisors()
    raise Refused(
        f"unknown repair {what!r}. Use tunnels, engine:<host>, or supervisors."
    )


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


def _h200_parity_container_preflight(host: str, expected_trace_dir: str) -> dict[str, object]:
    """Verify the isolated GLM command and TOML before model-load waiting."""
    _check_host(host)
    inspect_command = (
        f"sudo docker inspect {shlex.quote(glm_inference_container)} "
        "--format '{{json .State.Running}}\\n{{json .Config.Image}}\\n"
        "{{json .Config.Cmd}}\\n{{json .Config.Env}}\\n{{json .Mounts}}\\n"
        "{{json .HostConfig.RestartPolicy.Name}}'"
    )
    rc, output = _ssh(host, inspect_command, timeout=45)
    if rc:
        raise Refused(f"cannot inspect {glm_inference_container} on {host}: {output.strip()}")
    # Docker's Go template emits literal backslash-n separators on this host.
    lines = output.replace("\\n", "\n").splitlines()
    if len(lines) != 6:
        raise Refused("H200 parity container inspection returned an invalid record")
    try:
        running = json.loads(lines[0])
        image = json.loads(lines[1])
        command = json.loads(lines[2])
        environment = json.loads(lines[3])
        mounts = json.loads(lines[4])
        restart_policy = json.loads(lines[5])
    except json.JSONDecodeError as error:
        raise Refused(f"H200 parity container inspection is not valid JSON: {error}") from error
    expected_command = ["@", glm_parity_inference_config, "--vllm-extra", glm_parity_vllm_extra]
    if running is not True:
        raise Refused("H200 parity container is not running")
    if image != inference_image:
        raise Refused(f"H200 parity image is {image!r}, expected {inference_image!r}")
    if command != expected_command:
        raise Refused(f"H200 parity command is {command!r}, expected {expected_command!r}")
    if restart_policy != "no":
        raise Refused("H200 parity container restart policy is not isolated")
    if not isinstance(environment, list) or any(
        not isinstance(value, str) for value in environment
    ):
        raise Refused("H200 parity container environment is invalid")
    if any(value.startswith("PRIME_PARITY_BOOTSTRAP_TOKEN=") for value in environment):
        raise Refused("H200 parity container metadata contains a bootstrap token")
    if environment.count(
        "PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap"
    ) != 1:
        raise Refused("H200 parity container has no exact bootstrap FIFO path")
    if re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/"
        r"\.diagnostic-work-[0-9a-f]{16}/h200-router-trace",
        expected_trace_dir,
    ) is None or environment.count(
        f"PRIME_PARITY_ROUTER_TRACE_DIR={expected_trace_dir}"
    ) != 1:
        raise Refused("H200 parity container has no exact router trace path")
    bootstrap_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict)
        and mount.get("Destination") == "/run/prime-parity-bootstrap"
    ] if isinstance(mounts, list) else []
    if len(bootstrap_mounts) != 1:
        raise Refused("H200 parity container has no exact bootstrap FIFO mount")
    bootstrap_mount = bootstrap_mounts[0]
    if (
        bootstrap_mount.get("Type") != "bind"
        or bootstrap_mount.get("RW") is not True
        or not re.fullmatch(
            r"/data/glm52-rl/qualifications/glm52-policy-parity/"
            r"\.diagnostic-run-[0-9a-f]{16}/bootstrap\.fifo",
            str(bootstrap_mount.get("Source", "")),
        )
    ):
        raise Refused("H200 parity bootstrap FIFO mount is invalid")

    config_command = (
        "python3 -c 'import json,sys,tomllib; "
        "c=tomllib.load(open(sys.argv[1],\"rb\")); "
        "print(json.dumps({\"max_model_len\":c[\"model\"][\"max_model_len\"],"
        "\"tp\":c[\"parallel\"][\"tp\"],\"dp\":c[\"parallel\"][\"dp\"],"
        "\"api_servers\":c[\"api_server_count\"],"
        "\"lora_rank\":c[\"max_lora_rank\"],"
        "\"router_trace\":c[\"enable_return_routed_experts\"]}))' "
        f"{shlex.quote(glm_parity_inference_config)}"
    )
    rc, config_output = _ssh(host, config_command, timeout=45)
    if rc:
        raise Refused(f"cannot parse H200 parity config: {config_output.strip()}")
    try:
        config = json.loads(config_output.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise Refused("H200 parity config inspection returned invalid JSON") from error
    expected_config = {
        "max_model_len": 32769,
        "tp": 8,
        "dp": 1,
        "api_servers": 1,
        "lora_rank": 32,
        "router_trace": True,
    }
    if config != expected_config:
        raise Refused(f"H200 parity config is {config!r}, expected {expected_config!r}")
    result = {
        "host": host,
        "image": image,
        "command": command,
        "config": config,
        "bootstrap_fifo": "/run/prime-parity-bootstrap",
    }
    print("H200 parity container preflight passed: " + json.dumps(result, sort_keys=True))
    return result


def _h200_parity_port_preflight_command() -> str:
    protected = [protected_orchestrator, *protected_inference]
    expression = "|".join(re.escape(name) for name in protected)
    return (
        "set -euo pipefail; "
        "inventory=$(sudo docker ps --format '{{.Names}}'); "
        "set +e; "
        "active=$(printf '%s\\n' \"$inventory\" | "
        f"grep -E '^({expression})(-[0-9]+)?$'); "
        "match_status=$?; set -e; "
        "test \"$match_status\" -le 1 || exit \"$match_status\"; "
        "test -z \"$active\" || { echo protected-inference-active; exit 73; }; "
        "router_listener=$(sudo ss -H -lntp 'sport = :8000'); "
        "test -z \"$router_listener\" || { echo port-8000-in-use; exit 74; }; "
        f"backend_listener=$(sudo ss -H -lntp 'sport = :{h200_parity_backend_port}'); "
        f"test -z \"$backend_listener\" || {{ echo port-{h200_parity_backend_port}-in-use; exit 75; }}; "
        "echo parity-ports-free"
    )


def _assert_h200_parity_port_free(host: str) -> None:
    _check_host(host)
    command = _h200_parity_port_preflight_command()
    code, output = _ssh(host, command, timeout=60)
    if code or output.strip() != "parity-ports-free":
        raise Refused(
            f"isolated parity port preflight failed on {host}: {output.strip()}"
        )


def _h200_parity_listener_identity_command(timeout: int, port: int = 8000) -> str:
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0:
        raise Refused("H200 parity listener timeout is invalid")
    if port not in {8000, h200_parity_backend_port}:
        raise Refused("H200 parity listener port is invalid")
    error_prefix = "" if port == 8000 else "api-"
    loopback_check = ""
    if port == h200_parity_backend_port:
        loopback_check = rf'''
  listener_local=$(awk 'NR==1 {{print $4}}' <<<"${{listeners[0]}}") || fail 91 "${{phase}}listener-address-unreadable"
  case "$listener_local" in
    127.0.0.1:{port}|\[::1\]:{port}) ;;
    *) fail 91 "${{phase}}listener-not-loopback" ;;
  esac
'''
    expected_command = json.dumps(
        ["@", glm_parity_inference_config, "--vllm-extra", glm_parity_vllm_extra],
        separators=(",", ":"),
    )
    return rf"""
set -euo pipefail
name={shlex.quote(glm_inference_container)}
deadline=$((SECONDS+{timeout}))
error_prefix={shlex.quote(error_prefix)}
fail() {{
  code=$1
  detail=$2
  printf 'parity-listener-error:%s%s\n' "$error_prefix" "$detail" >&2
  exit "$code"
}}
read_container() {{
  phase=$1
  inspect=$(sudo docker inspect "$name" --format '{{{{.State.Running}}}}
{{{{.State.Pid}}}}
{{{{.Id}}}}
{{{{json .Config.Image}}}}
{{{{json .Config.Cmd}}}}
{{{{.RestartCount}}}}
{{{{json .HostConfig.RestartPolicy.Name}}}}') || fail 81 "${{phase}}container-inspect-failed"
  mapfile -t record <<<"$inspect" || fail 82 "${{phase}}container-inspect-record-unreadable"
  test "${{#record[@]}}" = 7 || fail 82 "${{phase}}container-inspect-record-invalid"
  running=${{record[0]}}
  container_pid=${{record[1]}}
  container_id=${{record[2]}}
  image=${{record[3]}}
  command=${{record[4]}}
  restart_count=${{record[5]}}
  restart_policy=${{record[6]}}
  test "$running" = true || fail 83 "${{phase}}container-not-running"
  [[ "$container_pid" =~ ^[0-9]+$ ]] || fail 84 "${{phase}}container-pid-invalid"
  test "$container_pid" -gt 1 || fail 84 "${{phase}}container-pid-invalid"
  [[ "$container_id" =~ ^[0-9a-f]{{64}}$ ]] || fail 85 "${{phase}}container-id-invalid"
  test "$image" = {shlex.quote(json.dumps(inference_image))} || fail 86 "${{phase}}container-image-mismatch"
  test "$command" = {shlex.quote(expected_command)} || fail 87 "${{phase}}container-command-mismatch"
  test "$restart_count" = 0 || fail 88 "${{phase}}container-restarted"
  test "$restart_policy" = {shlex.quote(json.dumps('no'))} || fail 89 "${{phase}}container-restart-policy-invalid"
}}
read_listener() {{
  phase=$1
  listener_output=$(sudo ss -H -lntpe 'sport = :{port}') || fail 90 "${{phase}}listener-probe-failed"
  test -n "$listener_output" || return 10
  mapfile -t listeners <<<"$listener_output" || fail 91 "${{phase}}listener-record-unreadable"
  test "${{#listeners[@]}}" = 1 || fail 91 "${{phase}}multiple-listeners"
{loopback_check}
  remaining=${{listeners[0]}}
  listener_pids=()
  while [[ "$remaining" =~ pid=([0-9]+) ]]; do
    listener_pids+=("${{BASH_REMATCH[1]}}")
    remaining=${{remaining#*"${{BASH_REMATCH[0]}}"}}
  done
  test "${{#listener_pids[@]}}" -gt 0 || fail 92 "${{phase}}listener-pid-missing"
  declare -A unique_listener_pids=()
  for pid in "${{listener_pids[@]}}"; do unique_listener_pids["$pid"]=1; done
  test "${{#unique_listener_pids[@]}}" = 1 || fail 93 "${{phase}}multiple-listener-processes"
  for listener_pid in "${{!unique_listener_pids[@]}}"; do :; done
  [[ "$listener_pid" =~ ^[0-9]+$ ]] || fail 94 "${{phase}}listener-pid-invalid"
  test "$listener_pid" -gt 1 || fail 94 "${{phase}}listener-pid-invalid"
  remaining=${{listeners[0]}}
  listener_inodes=()
  while [[ "$remaining" =~ ino:([0-9]+) ]]; do
    listener_inodes+=("${{BASH_REMATCH[1]}}")
    remaining=${{remaining#*"${{BASH_REMATCH[0]}}"}}
  done
  test "${{#listener_inodes[@]}}" = 1 || fail 94 "${{phase}}listener-socket-inode-invalid"
  listener_inode=${{listener_inodes[0]}}
}}
process_starttime() {{
  pid=$1
  stat_record=$(sudo cat -- "/proc/$pid/stat") || return 1
  stat_tail=${{stat_record##*) }}
  test "$stat_tail" != "$stat_record" || return 1
  read -r -a stat_fields <<<"$stat_tail" || return 1
  test "${{#stat_fields[@]}}" -ge 20 || return 1
  starttime=${{stat_fields[19]}}
  [[ "$starttime" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$starttime"
}}
process_namespace_pid() {{
  pid=$1
  status_record=$(sudo cat -- "/proc/$pid/status") || return 1
  namespace_lines=()
  while IFS= read -r line; do
    if [[ "$line" == NSpid:* ]]; then namespace_lines+=("$line"); fi
  done <<<"$status_record"
  test "${{#namespace_lines[@]}}" = 1 || return 1
  read -r -a namespace_pids <<<"${{namespace_lines[0]#NSpid:}}" || return 1
  test "${{#namespace_pids[@]}}" -gt 0 || return 1
  for namespace_pid in "${{namespace_pids[@]}}"; do
    [[ "$namespace_pid" =~ ^[0-9]+$ ]] || return 1
    test "$namespace_pid" -gt 0 || return 1
  done
  printf '%s\n' "${{namespace_pids[-1]}}"
}}
read_namespaces() {{
  pid=$1
  pid_namespace=$(sudo stat -Lc %i -- "/proc/$pid/ns/pid") || return 1
  mount_namespace=$(sudo stat -Lc %i -- "/proc/$pid/ns/mnt") || return 1
  [[ "$pid_namespace" =~ ^[0-9]+$ ]] || return 1
  [[ "$mount_namespace" =~ ^[0-9]+$ ]] || return 1
}}
cgroup_is_equal_or_below() {{
  parent=$1
  child=$2
  declare -A parent_paths=()
  declare -A child_paths=()
  while IFS=: read -r hierarchy controllers path extra; do
    test -z "$extra" || return 1
    test -n "$hierarchy" || return 1
    [[ "$path" == /* ]] || return 1
    key="$hierarchy:$controllers"
    [[ ! -v 'parent_paths[$key]' ]] || return 1
    parent_paths["$key"]=$path
  done <<<"$parent"
  while IFS=: read -r hierarchy controllers path extra; do
    test -z "$extra" || return 1
    test -n "$hierarchy" || return 1
    [[ "$path" == /* ]] || return 1
    key="$hierarchy:$controllers"
    [[ ! -v 'child_paths[$key]' ]] || return 1
    child_paths["$key"]=$path
  done <<<"$child"
  test "${{#parent_paths[@]}}" -gt 0 || return 1
  test "${{#child_paths[@]}}" = "${{#parent_paths[@]}}" || return 1
  for key in "${{!parent_paths[@]}}"; do
    [[ -v 'child_paths[$key]' ]] || return 1
    parent_path=${{parent_paths[$key]}}
    child_path=${{child_paths[$key]}}
    if test "$parent_path" = /; then
      [[ "$child_path" == /* ]] || return 1
    else
      [[ "$child_path" == "$parent_path" || "$child_path" == "$parent_path/"* ]] || return 1
    fi
  done
}}
read_membership() {{
  phase=$1
  member_pid=$2
  top_output=$(sudo docker top "$name" -eo pid) || fail 99 "${{phase}}docker-top-failed"
  member_count=0
  while read -r top_pid extra; do
    test -z "${{extra:-}}" || fail 99 "${{phase}}docker-top-record-invalid"
    if test "$top_pid" = "$member_pid"; then member_count=$((member_count+1)); fi
  done <<<"$top_output"
  test "$member_count" = 1 || fail 99 "${{phase}}listener-docker-membership-invalid"
}}
while :; do
  read_container ""
  if read_listener ""; then
    break
  else
    listener_status=$?
  fi
  test "$listener_status" = 10 || fail 99 listener-read-failed
  if test $SECONDS -ge $deadline; then
    fail 124 listener-timeout
  else
    sleep 2 || fail 99 listener-wait-failed
  fi
done
first_inspect=$inspect
first_container_pid=$container_pid
first_container_id=$container_id
first_listener_output=$listener_output
first_listener_pid=$listener_pid
first_listener_inode=$listener_inode
read_membership "" "$first_listener_pid"
first_init_starttime=$(process_starttime "$first_container_pid") || fail 95 container-init-stat-unreadable
first_listener_starttime=$(process_starttime "$first_listener_pid") || fail 95 listener-stat-unreadable
first_listener_namespace_pid=$(process_namespace_pid "$first_listener_pid") || fail 95 listener-namespace-pid-unreadable
read_namespaces "$first_container_pid" || fail 95 container-init-namespace-unreadable
first_init_pid_namespace=$pid_namespace
first_init_mount_namespace=$mount_namespace
read_namespaces "$first_listener_pid" || fail 95 listener-namespace-unreadable
first_listener_pid_namespace=$pid_namespace
first_listener_mount_namespace=$mount_namespace
test "$first_listener_pid_namespace" = "$first_init_pid_namespace" || fail 96 listener-pid-namespace-mismatch
test "$first_listener_mount_namespace" = "$first_init_mount_namespace" || fail 96 listener-mount-namespace-mismatch
first_init_cgroup=$(sudo cat -- "/proc/$first_container_pid/cgroup") || fail 95 container-init-cgroup-unreadable
first_listener_cgroup=$(sudo cat -- "/proc/$first_listener_pid/cgroup") || fail 95 listener-cgroup-unreadable
cgroup_is_equal_or_below "$first_init_cgroup" "$first_listener_cgroup" || fail 96 listener-cgroup-mismatch
first_boot_id=$(sudo cat -- /proc/sys/kernel/random/boot_id) || fail 97 host-boot-id-unreadable
[[ "$first_boot_id" =~ ^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$ ]] || fail 98 host-boot-id-invalid

read_container second-
second_inspect=$inspect
second_container_pid=$container_pid
second_container_id=$container_id
test "$second_container_id" = "$first_container_id" || fail 100 second-container-replaced
test "$second_container_pid" = "$first_container_pid" || fail 100 second-container-init-pid-changed
test "$second_inspect" = "$first_inspect" || fail 100 second-container-inspect-changed
if read_listener second-; then
  :
else
  listener_status=$?
  test "$listener_status" = 10 || fail 100 second-listener-read-failed
  fail 100 second-listener-missing
fi
second_listener_output=$listener_output
second_listener_pid=$listener_pid
second_listener_inode=$listener_inode
test "$second_listener_pid" = "$first_listener_pid" || fail 100 listener-pid-rebound
test "$second_listener_inode" = "$first_listener_inode" || fail 100 listener-socket-rebound
test "$second_listener_output" = "$first_listener_output" || fail 100 listener-record-changed
read_membership second- "$second_listener_pid"
second_init_starttime=$(process_starttime "$second_container_pid") || fail 100 second-container-init-stat-unreadable
second_listener_starttime=$(process_starttime "$second_listener_pid") || fail 100 second-listener-stat-unreadable
second_listener_namespace_pid=$(process_namespace_pid "$second_listener_pid") || fail 100 second-listener-namespace-pid-unreadable
test "$second_init_starttime" = "$first_init_starttime" || fail 100 container-init-pid-reused
test "$second_listener_starttime" = "$first_listener_starttime" || fail 100 listener-pid-reused
test "$second_listener_namespace_pid" = "$first_listener_namespace_pid" || fail 100 listener-namespace-pid-changed
read_namespaces "$second_container_pid" || fail 100 second-container-init-namespace-unreadable
second_init_pid_namespace=$pid_namespace
second_init_mount_namespace=$mount_namespace
read_namespaces "$second_listener_pid" || fail 100 second-listener-namespace-unreadable
second_listener_pid_namespace=$pid_namespace
second_listener_mount_namespace=$mount_namespace
test "$second_init_pid_namespace" = "$first_init_pid_namespace" || fail 100 container-init-pid-namespace-changed
test "$second_init_mount_namespace" = "$first_init_mount_namespace" || fail 100 container-init-mount-namespace-changed
test "$second_listener_pid_namespace" = "$first_listener_pid_namespace" || fail 100 listener-pid-namespace-changed
test "$second_listener_mount_namespace" = "$first_listener_mount_namespace" || fail 100 listener-mount-namespace-changed
test "$second_listener_pid_namespace" = "$second_init_pid_namespace" || fail 100 second-listener-pid-namespace-mismatch
test "$second_listener_mount_namespace" = "$second_init_mount_namespace" || fail 100 second-listener-mount-namespace-mismatch
second_init_cgroup=$(sudo cat -- "/proc/$second_container_pid/cgroup") || fail 100 second-container-init-cgroup-unreadable
second_listener_cgroup=$(sudo cat -- "/proc/$second_listener_pid/cgroup") || fail 100 second-listener-cgroup-unreadable
test "$second_init_cgroup" = "$first_init_cgroup" || fail 100 container-init-cgroup-changed
test "$second_listener_cgroup" = "$first_listener_cgroup" || fail 100 listener-cgroup-changed
cgroup_is_equal_or_below "$second_init_cgroup" "$second_listener_cgroup" || fail 100 second-listener-cgroup-mismatch
second_boot_id=$(sudo cat -- /proc/sys/kernel/random/boot_id) || fail 100 second-host-boot-id-unreadable
test "$second_boot_id" = "$first_boot_id" || fail 100 host-boot-id-changed
relation=child
if test "$first_listener_pid" = "$first_container_pid"; then relation=init; fi
printf '{{"listener_port":%s,"listener_pid":%s,"listener_namespace_pid":%s,"container_init_pid":%s,"listener_relation":"%s","listener_socket_inode":%s,"listener_starttime":%s,"container_id":"%s","host_boot_id":"%s"}}\n' \
  {port} "$first_listener_pid" "$first_listener_namespace_pid" "$first_container_pid" "$relation" "$first_listener_inode" "$first_listener_starttime" "$first_container_id" "$first_boot_id" || fail 100 identity-write-failed
"""


def _h200_parity_listener_identity(
    host: str,
    sensitive: tuple[str, ...],
    timeout: int,
    port: int = 8000,
) -> dict[str, object]:
    _check_host(host)
    command = _h200_parity_listener_identity_command(timeout, port)
    output = _sensitive_ssh(
        host,
        command,
        "isolated parity listener identity",
        sensitive,
        timeout + 30,
    ).strip()
    try:
        identity = json.loads(output)
    except json.JSONDecodeError as error:
        raise Refused("isolated parity listener identity is not valid JSON") from error
    if (
        not isinstance(identity, dict)
        or identity.get("listener_port") != port
        or not isinstance(identity.get("listener_pid"), int)
        or identity["listener_pid"] <= 1
        or not isinstance(identity.get("listener_namespace_pid"), int)
        or identity["listener_namespace_pid"] <= 0
        or not isinstance(identity.get("container_init_pid"), int)
        or identity["container_init_pid"] <= 1
        or not isinstance(identity.get("listener_socket_inode"), int)
        or identity["listener_socket_inode"] <= 0
        or not isinstance(identity.get("listener_starttime"), int)
        or identity["listener_starttime"] <= 0
        or identity.get("listener_relation") not in {"init", "child"}
        or (
            identity["listener_relation"] == "init"
            and identity["listener_pid"] != identity["container_init_pid"]
        )
        or (
            identity["listener_relation"] == "child"
            and identity["listener_pid"] == identity["container_init_pid"]
        )
        or re.fullmatch(r"[0-9a-f]{64}", str(identity.get("container_id", ""))) is None
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(identity.get("host_boot_id", "")),
        )
        is None
    ):
        raise Refused("isolated parity listener identity is invalid")
    return identity


def _h200_capture_bootstrap_identity(
    host: str,
    capture_directory: str,
    bootstrap_token: str,
    expected_process_id: int,
) -> dict[str, object]:
    if re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/"
        r"\.diagnostic-work-[0-9a-f]{16}/capture",
        capture_directory,
    ) is None:
        raise Refused("H200 parity capture identity path is invalid")
    command = (
        "python3 -c 'import json,pathlib,sys; "
        "p=pathlib.Path(sys.argv[1]); "
        "assert p.is_dir() and not p.is_symlink(); "
        "m=p/\"manifest.json\"; assert m.is_file() and not m.is_symlink(); "
        "print(json.dumps(json.loads(m.read_text())[\"bootstrap_identity\"],"
        "sort_keys=True,separators=(\",\",\":\")))' "
        f"{shlex.quote(capture_directory)}"
    )
    output = _sensitive_ssh(
        host,
        command,
        "H200 capture bootstrap identity",
        (bootstrap_token,),
        60,
    ).strip()
    try:
        identity = json.loads(output)
    except json.JSONDecodeError as error:
        raise Refused("H200 capture bootstrap identity is not valid JSON") from error
    expected_sha256 = hashlib.sha256(bootstrap_token.encode("ascii")).hexdigest()
    if (
        isinstance(expected_process_id, bool)
        or not isinstance(expected_process_id, int)
        or expected_process_id <= 0
        or not isinstance(identity, dict)
        or identity.get("status") != "ready"
        or identity.get("process_id") != expected_process_id
        or re.fullmatch(r"[0-9a-f]{32}", str(identity.get("boot_nonce", ""))) is None
        or identity.get("token_sha256") != expected_sha256
    ):
        raise Refused("H200 capture bootstrap identity is invalid")
    return identity


def _diagnostic_capture_preflight(config_path: Path, side: str) -> dict[str, int]:
    if side not in {"trainer", "h200"}:
        raise Refused("diagnostic capture side is invalid")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    env = config.get("env_vars", {})
    if env.get("PRIME_PARITY_DIAGNOSTICS") != "1":
        return {"files": 0, "estimated_bytes": 0}
    names = ["embedding"]
    for layer in range(3):
        names.extend(
            (
                f"layer{layer}_input",
                f"layer{layer}_attention_raw",
                f"layer{layer}_mlp_raw",
                f"layer{layer}_output",
            )
        )
    names.append("layer3_router_input")
    widths = [6144] * len(names)
    if side == "h200":
        extra = {
            "layer3_h200_input": 6144,
            "layer3_h200_logits": 256,
            "layer3_h200_scores": 256,
            "layer3_h200_selection_scores": 256,
            "layer3_h200_route_ids": 8,
            "layer3_h200_route_weights": 8,
        }
        names.extend(extra)
        widths.extend(extra.values())
    if len(names) != len(set(names)):
        raise Refused("diagnostic capture plan contains duplicate semantic files")
    try:
        tokens = int(env["PRIME_PARITY_DIAGNOSTIC_TOKENS"])
        max_files = int(env["PRIME_PARITY_DIAGNOSTIC_MAX_FILES"])
        max_bytes = int(env["PRIME_PARITY_DIAGNOSTIC_MAX_BYTES"])
    except (KeyError, TypeError, ValueError) as error:
        raise Refused("diagnostic capture limits are invalid") from error
    estimated_bytes = sum(tokens * width * 4 + 64 * 1024 for width in widths)
    if len(names) > max_files or estimated_bytes > max_bytes:
        raise Refused("diagnostic capture plan exceeds its configured cap")
    return {"files": len(names), "estimated_bytes": estimated_bytes}


def _redact_sensitive(value: str, sensitive: tuple[str, ...]) -> str:
    redacted = value
    for secret in sensitive:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _validate_parity_transaction_path(path: str) -> str:
    match = re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/" r"\.diagnostic-run-([0-9a-f]{16})/transaction\.json",
        path,
    )
    if match is None:
        raise Refused("parity transaction path is not exact")
    return match.group(1)


def _parity_container_write_paths(nonce: str) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{16}", nonce) is None:
        raise Refused("parity container write nonce is invalid")
    root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    work = f"{root}/.diagnostic-work-{nonce}"
    run = f"{root}/.diagnostic-run-{nonce}"
    return {
        "work": work,
        "fixture": f"{work}/fixture",
        "diagnostic": f"{work}/diagnostic.npz",
        "capture": f"{work}/capture",
        "router_trace": f"{work}/h200-router-trace",
        "bootstrap_fifo": f"{run}/bootstrap.fifo",
        "lora_cache": "/data/glm52-rl/lora-cache",
        "vllm_cache": f"{remote_repo}/cache/vllm",
    }


def _validate_parity_transaction_state(path: str, state: dict[str, object]) -> None:
    nonce = _validate_parity_transaction_path(path)
    root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    writes = _parity_container_write_paths(nonce)
    if (
        state.get("schema_version") != 1
        or state.get("nonce") != nonce
        or state.get("phase")
        not in {
            "fixture_ready",
            "fixture_published",
            "capture_ready",
            "capture_published",
            "committed",
            "rolled_back",
        }
    ):
        raise RuntimeError("parity transaction identity is invalid")
    expected_paths = {
        "fixture": {
            "live": glm_parity_paths["fixture"],
            "previous": f"{root}/.fixture-previous-{nonce}",
            "candidate": writes["fixture"],
            "failed": f"{root}/.fixture-failed-{nonce}",
        },
        "capture": {
            "live": glm_parity_paths["h200_capture"],
            "previous": f"{root}/.h200-capture-previous-{nonce}",
            "candidate": writes["capture"],
            "failed": f"{root}/.h200-capture-failed-{nonce}",
        },
    }
    for kind, paths in expected_paths.items():
        artifact = state.get(kind)
        if not isinstance(artifact, dict) or artifact.get("paths") != paths:
            raise RuntimeError(f"parity transaction {kind} paths are invalid")
        action = artifact.get("action")
        if action not in {None, "created", "replaced"}:
            raise RuntimeError(f"parity transaction {kind} action is invalid")
        new_hash = artifact.get("new_manifest_sha256")
        old_hash = artifact.get("old_manifest_sha256")
        if new_hash is not None and re.fullmatch(r"[0-9a-f]{64}", str(new_hash)) is None:
            raise RuntimeError(f"parity transaction {kind} new hash is invalid")
        if old_hash is not None and re.fullmatch(r"[0-9a-f]{64}", str(old_hash)) is None:
            raise RuntimeError(f"parity transaction {kind} old hash is invalid")
        if action is not None and new_hash is None:
            raise RuntimeError(f"parity transaction {kind} published hash is missing")


def _prepare_parity_container_work(
    host: str,
    paths: dict[str, str],
    sensitive: tuple[str, ...],
) -> None:
    work = paths.get("work", "")
    match = re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/"
        r"\.diagnostic-work-([0-9a-f]{16})",
        work,
    )
    if match is None or paths != _parity_container_write_paths(match.group(1)):
        raise Refused("parity container write inventory is not exact")
    command = f"""
set -euo pipefail
work={shlex.quote(paths["work"])}
trace={shlex.quote(paths["router_trace"])}
fixture={shlex.quote(paths["fixture"])}
diagnostic={shlex.quote(paths["diagnostic"])}
capture={shlex.quote(paths["capture"])}
test ! -e "$work"
test ! -L "$work"
sudo install -d -o 1000 -g 1000 -m 0750 "$work"
sudo install -d -o 1000 -g 1000 -m 0750 "$trace"
for path in "$work" "$trace"; do
  test -d "$path"
  test ! -L "$path"
  test "$(realpath -e "$path")" = "$path"
  test "$(stat -c '%u:%g:%a' "$path")" = "1000:1000:750"
  sudo -u '#1000' test -r "$path"
  sudo -u '#1000' test -w "$path"
  sudo -u '#1000' test -x "$path"
done
test "$(find "$work" -mindepth 1 -maxdepth 1 -printf '%f\n')" = "h200-router-trace"
test -z "$(find "$trace" -mindepth 1 -maxdepth 1 -print -quit)"
for path in "$fixture" "$diagnostic" "$capture"; do
  test ! -e "$path"
  test ! -L "$path"
done
printf 'work-ready\n'
"""
    output = _sensitive_ssh(
        host,
        command,
        "parity container work preparation",
        sensitive,
        60,
    ).strip()
    if output != "work-ready":
        raise RuntimeError("parity container work preparation returned an invalid state")


def _verify_parity_container_outputs(
    host: str,
    paths: dict[str, str],
    stage: str,
    sensitive: tuple[str, ...],
) -> None:
    work = paths.get("work", "")
    match = re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/"
        r"\.diagnostic-work-([0-9a-f]{16})",
        work,
    )
    if match is None or paths != _parity_container_write_paths(match.group(1)):
        raise Refused("parity container write inventory is not exact")
    if stage == "refresh":
        directories = (paths["fixture"], paths["router_trace"])
        files = (paths["diagnostic"],)
        names = "diagnostic.npz\nfixture\nh200-router-trace"
    elif stage == "capture":
        directories = (paths["capture"], paths["router_trace"])
        files = (paths["diagnostic"],)
        names = "capture\ndiagnostic.npz\nh200-router-trace"
    else:
        raise Refused("parity container output stage is invalid")
    directory_values = " ".join(shlex.quote(path) for path in directories)
    file_values = " ".join(shlex.quote(path) for path in files)
    command = f"""
set -euo pipefail
work={shlex.quote(paths["work"])}
test -d "$work"
test ! -L "$work"
test "$(realpath -e "$work")" = "$work"
test "$(stat -c '%u:%g:%a' "$work")" = "1000:1000:750"
test "$(find "$work" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" = {shlex.quote(names)}
for path in {directory_values}; do
  test -d "$path"
  test ! -L "$path"
  test "$(realpath -e "$path")" = "$path"
  test "$(stat -c '%u:%g' "$path")" = "1000:1000"
  sudo -u '#1000' test -r "$path"
  sudo -u '#1000' test -w "$path"
  sudo -u '#1000' test -x "$path"
  test -z "$(find "$path" -xdev \\( -type l -o ! -uid 1000 -o ! -gid 1000 -o -perm /022 \\) -print -quit)"
done
for path in {file_values}; do
  test -f "$path"
  test ! -L "$path"
  test "$(realpath -e "$path")" = "$path"
  test "$(stat -c '%u:%g' "$path")" = "1000:1000"
  test $((8#$(stat -c '%a' "$path") & 8#022)) -eq 0
  sudo -u '#1000' test -r "$path"
done
for directory in {directory_values}; do
  test -z "$(sudo -u '#1000' find "$directory" -xdev \
    \\( ! -readable -o \\( -type d ! -executable \\) \\) -print -quit)"
done
printf 'outputs-ready\n'
"""
    output = _sensitive_ssh(
        host,
        command,
        f"parity container {stage} output verification",
        sensitive,
        60,
    ).strip()
    if output != "outputs-ready":
        raise RuntimeError(
            f"parity container {stage} output verification returned an invalid state"
        )


def _write_parity_transaction_state(
    host: str,
    path: str,
    state: dict[str, object],
    sensitive: tuple[str, ...],
) -> None:
    _validate_parity_transaction_state(path, state)
    encoded = base64.b64encode((json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()).decode()
    command = (
        "sudo python3 -c 'import base64,os,pathlib,sys; "
        'p=pathlib.Path(sys.argv[1]); t=p.with_name(".transaction.json.tmp"); '
        "t.write_bytes(base64.b64decode(sys.argv[2],validate=True)); "
        "os.chmod(t,0o600); os.replace(t,p)' "
        f"{shlex.quote(path)} {shlex.quote(encoded)}"
    )
    _sensitive_ssh(host, command, "parity transaction state write", sensitive, 60)


def _parity_transaction_snapshot(
    host: str,
    path: str,
    sensitive: tuple[str, ...],
) -> dict[str, object]:
    _validate_parity_transaction_path(path)
    command = (
        "sudo python3 -c 'import hashlib,json,os,pathlib,stat,sys; "
        "p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); observed={}; "
        '[(lambda q,k: observed.__setitem__(k,{"exists":q.exists(),'
        '"symlink":q.is_symlink(),"manifest_sha256":'
        'hashlib.sha256((q/"manifest.json").read_bytes()).hexdigest() '
        'if q.is_dir() and not q.is_symlink() and (q/"manifest.json").is_file() else None}))'
        '(pathlib.Path(v),f"{kind}.{key}") for kind in ("fixture","capture") '
        'for key,v in s[kind]["paths"].items()]; '
        'print(json.dumps({"state":s,"observed":observed},sort_keys=True))\' '
        f"{shlex.quote(path)}"
    )
    output = _sensitive_ssh(host, command, "parity transaction reconciliation", sensitive, 60)
    try:
        snapshot = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("parity transaction reconciliation returned invalid JSON") from error
    if not isinstance(snapshot, dict):
        raise RuntimeError("parity transaction reconciliation returned no object")
    state = snapshot.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("parity transaction reconciliation omitted state")
    _validate_parity_transaction_state(path, state)
    return snapshot


def _validate_published_transaction_artifact(snapshot: dict[str, object], kind: str) -> bool:
    state = snapshot.get("state")
    observed = snapshot.get("observed")
    if not isinstance(state, dict) or not isinstance(observed, dict):
        raise RuntimeError("parity transaction snapshot is incomplete")
    artifact = state.get(kind)
    if not isinstance(artifact, dict):
        raise RuntimeError(f"parity transaction {kind} state is missing")
    action = artifact.get("action")
    if action not in {"created", "replaced"}:
        raise RuntimeError(f"parity transaction {kind} publish is not recorded")
    live = observed.get(f"{kind}.live")
    previous = observed.get(f"{kind}.previous")
    candidate = observed.get(f"{kind}.candidate")
    failed = observed.get(f"{kind}.failed")
    if (
        not isinstance(candidate, dict)
        or candidate.get("exists") is not False
        or candidate.get("symlink") is not False
    ):
        raise RuntimeError(f"parity transaction {kind} candidate remains")
    if (
        not isinstance(failed, dict)
        or failed.get("exists") is not False
        or failed.get("symlink") is not False
    ):
        raise RuntimeError(f"parity transaction {kind} failed path is occupied")
    if (
        not isinstance(live, dict)
        or live.get("symlink") is not False
        or (live.get("manifest_sha256") != artifact.get("new_manifest_sha256"))
    ):
        raise RuntimeError(f"parity transaction {kind} live artifact differs")
    if action == "replaced":
        if (
            not isinstance(previous, dict)
            or previous.get("symlink") is not False
            or (previous.get("manifest_sha256") != artifact.get("old_manifest_sha256"))
        ):
            raise RuntimeError(f"parity transaction {kind} backup differs")
    elif (
        not isinstance(previous, dict)
        or previous.get("exists") is not False
        or previous.get("symlink") is not False
    ):
        raise RuntimeError(f"parity transaction {kind} has an unexpected backup")
    return action == "replaced"


def _parity_transaction_update_command(path: str, phase: str, kind: str, action_shell_value: str) -> str:
    _validate_parity_transaction_path(path)
    if phase not in {
        "fixture_published",
        "capture_published",
        "committed",
        "rolled_back",
    }:
        raise Refused("parity transaction phase is invalid")
    if kind not in {"fixture", "capture", "none"}:
        raise Refused("parity transaction artifact kind is invalid")
    return (
        "sudo python3 -c 'import json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        's=json.loads(p.read_text()); s["phase"]=sys.argv[2]; '
        'None if sys.argv[3]=="none" else s[sys.argv[3]].__setitem__("action",sys.argv[4]); '
        't=p.with_name(".transaction.json.tmp"); '
        't.write_text(json.dumps(s,sort_keys=True,separators=(",",":"))+"\\n"); '
        "os.chmod(t,0o600); os.replace(t,p)' "
        f"{shlex.quote(path)} {shlex.quote(phase)} {shlex.quote(kind)} "
        f"{action_shell_value}"
    )


def _remove_completed_parity_transaction(
    host: str,
    path: str,
    run_directory: str,
    sensitive: tuple[str, ...],
) -> None:
    nonce = _validate_parity_transaction_path(path)
    expected_directory = (
        "/data/glm52-rl/qualifications/glm52-policy-parity/"
        f".diagnostic-run-{nonce}"
    )
    if run_directory != expected_directory or str(PurePosixPath(path).parent) != run_directory:
        raise Refused("completed parity transaction directory is not exact")
    command = f"""
set -euo pipefail
run={shlex.quote(run_directory)}
state={shlex.quote(path)}
test -d "$run"
test ! -L "$run"
test "$(realpath -e "$run")" = "$run"
test -f "$state"
test ! -L "$state"
test "$(realpath -e "$state")" = "$state"
test "$(find "$run" -mindepth 1 -maxdepth 1 -printf '%f\n')" = "transaction.json"
sudo rm -f -- "$state"
sudo rmdir -- "$run"
test ! -e "$state"
test ! -e "$run"
"""
    _sensitive_ssh(
        host,
        command,
        "completed parity transaction cleanup",
        sensitive,
        60,
    )


def _remote_manifest_sha256(host: str, path: str, label: str, sensitive: tuple[str, ...]) -> str | None:
    value = _sensitive_ssh(
        host,
        f"if test -f {shlex.quote(path + '/manifest.json')}; then "
        f"sha256sum {shlex.quote(path + '/manifest.json')} | cut -d' ' -f1; "
        "else echo absent; fi",
        label,
        sensitive,
        60,
    ).strip()
    if value == "absent":
        return None
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} returned an invalid hash")
    return value


def _parity_pair_rollback_command(artifacts: list[dict[str, object]], state_update_command: str) -> str:
    if not artifacts or len(artifacts) > 2 or not state_update_command:
        raise Refused("parity rollback plan is invalid")
    values = []
    for artifact in artifacts:
        live = str(artifact.get("live", ""))
        previous = str(artifact.get("previous", ""))
        failed = str(artifact.get("failed", ""))
        replaced = artifact.get("replaced")
        if not all(path.startswith("/") for path in (live, previous, failed)) or not isinstance(replaced, bool):
            raise Refused("parity rollback artifact is invalid")
        values.append((live, previous, failed, replaced))
    lines = ["set -euo pipefail"]
    for index, (live, previous, failed, replaced) in enumerate(values):
        lines.extend(
            (
                f"live_{index}={shlex.quote(live)}",
                f"previous_{index}={shlex.quote(previous)}",
                f"failed_{index}={shlex.quote(failed)}",
                f"replaced_{index}={int(replaced)}",
                f"moved_{index}=0",
                f'test -d "$live_{index}"',
                f'test ! -L "$live_{index}"',
                f'test ! -e "$failed_{index}"',
            )
        )
        if replaced:
            lines.extend(
                (
                    f'test -d "$previous_{index}"',
                    f'test ! -L "$previous_{index}"',
                )
            )
    lines.extend(("recover_pair() {", "  set +e"))
    if len(values) == 2:
        lines.extend(
            (
                '  if test "$moved_0" != "$moved_1"; then',
                '    if test "$moved_0" = 1 && ! test -e "$live_0"; then mv "$failed_0" "$live_0"; fi',
                '    if test "$moved_1" = 1 && ! test -e "$live_1"; then mv "$failed_1" "$live_1"; fi',
                "    return",
                "  fi",
            )
        )
    for index, (_live, _previous, _failed, replaced) in enumerate(values):
        if replaced:
            lines.append(
                f'  if ! test -e "$live_{index}" && test -e "$previous_{index}"; then mv "$previous_{index}" "$live_{index}"; fi'
            )
    lines.append("}")
    lines.append("trap recover_pair EXIT")
    for index in range(len(values)):
        lines.extend(
            (
                f'mv "$live_{index}" "$failed_{index}"',
                f"moved_{index}=1",
            )
        )
    for index, (_live, _previous, _failed, replaced) in enumerate(values):
        if replaced:
            lines.append(f'mv "$previous_{index}" "$live_{index}"')
    lines.append(state_update_command)
    lines.append("trap - EXIT")
    for index in range(len(values)):
        lines.append(f'rm -rf -- "$failed_{index}"')
    return "\n".join(lines)


def _sensitive_ssh(
    host: str,
    command: str,
    label: str,
    sensitive: tuple[str, ...],
    timeout: int,
) -> str:
    code, output = _ssh(host, command, timeout=timeout)
    if code:
        clean = _redact_sensitive(output.strip(), sensitive)
        raise RuntimeError(f"{label} failed on {host}: {clean}")
    return _redact_sensitive(output, sensitive)


def _sensitive_ssh_input(
    host: str,
    command: str,
    secret: str,
    label: str,
    timeout: int,
) -> str:
    _check_host(host)
    try:
        process = subprocess.run(
            ssh_command + [host, command],
            input=secret + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} timed out on {host}") from error
    output = process.stdout if process.returncode == 0 else process.stdout + process.stderr
    clean = _redact_sensitive(output, (secret,))
    if process.returncode:
        raise RuntimeError(f"{label} failed on {host}: {clean.strip()}")
    return clean


def _receive_parity_bootstrap(host: str, command: str, timeout: int) -> str:
    _check_host(host)
    try:
        process = subprocess.run(
            ssh_command + [host, command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"parity bootstrap receive timed out on {host}") from error
    if process.returncode:
        raise RuntimeError(f"parity bootstrap receive failed on {host}")
    value = process.stdout
    if len(value) != 65 or not value.endswith("\n"):
        raise RuntimeError("parity bootstrap receive returned invalid framing")
    token = value[:-1]
    if re.fullmatch(r"[A-Za-z0-9_-]{64}", token) is None:
        raise RuntimeError("parity bootstrap receive returned invalid capacity")
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as error:
        raise RuntimeError("parity bootstrap receive returned invalid encoding") from error
    if len(decoded) != 48:
        raise RuntimeError("parity bootstrap receive returned invalid capacity")
    return token


def _parity_bootstrap_receive_command(path: str, timeout: int) -> str:
    return (
        "set -euo pipefail; status=0; "
        f"sudo timeout --foreground {timeout} cat -- {shlex.quote(path)} || status=$?; "
        f"sudo rm -f -- {shlex.quote(path)}; "
        f"test ! -e {shlex.quote(path)}; "
        f"test ! -L {shlex.quote(path)}; "
        'exit "$status"'
    )


def _parity_temporary_cleanup_command(
    *,
    bootstrap_fifo: str,
    transaction_path: str,
    run_directory: str,
    source_diagnostic_input: str,
    source_session: str,
    source_compact: str,
    source_compact_manifest: str,
    container_work_directory: str,
    retain_container_work: bool,
) -> str:
    fifo = shlex.quote(bootstrap_fifo)
    transaction = shlex.quote(transaction_path)
    run = shlex.quote(run_directory)
    diagnostic_input = shlex.quote(source_diagnostic_input)
    session = shlex.quote(source_session)
    compact = shlex.quote(source_compact)
    compact_manifest = shlex.quote(source_compact_manifest)
    work = shlex.quote(container_work_directory)
    if retain_container_work:
        work_cleanup = f"""
test -d {work}
test ! -L {work}
test "$(realpath -e {work})" = {work}
"""
    else:
        work_cleanup = f"""
if test -e {work}; then
  test -d {work}
  test ! -L {work}
  test "$(realpath -e {work})" = {work}
  sudo rm -rf -- {work}
elif test -L {work}; then
  exit 1
fi
test ! -e {work}
test ! -L {work}
"""
    return f"""
set -euo pipefail
sudo rm -f -- {fifo}
if test -e {transaction}; then
  test -f {transaction}
  test ! -L {transaction}
else
  sudo rmdir -- {run}
fi
rm -f -- {diagnostic_input}
rm -f -- {session}
rm -f -- {compact}
rm -f -- {compact_manifest}
{work_cleanup}
test ! -e {fifo}
test ! -L {fifo}
if test -e {transaction}; then
  test -f {transaction}
  test ! -L {transaction}
  test -d {run}
  test ! -L {run}
else
  test ! -e {run}
  test ! -L {run}
fi
test ! -e {diagnostic_input}
test ! -L {diagnostic_input}
test ! -e {session}
test ! -L {session}
test ! -e {compact}
test ! -L {compact}
test ! -e {compact_manifest}
test ! -L {compact_manifest}
"""


def _isolated_container_status(
    host: str,
    name: str,
    sensitive: tuple[str, ...],
) -> str:
    allowed = {
        glm_inference_container,
        f"{glm_inference_container}-0",
        f"{glm_inference_container}-1",
    }
    if name not in allowed or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise Refused("isolated container name is not exact")
    command = f"""
set -euo pipefail
name={shlex.quote(name)}
found=$(sudo docker ps -a --filter "name=^/${{name}}$" --format '{{{{.Names}}}}')
if test -z "$found"; then echo absent; exit 0; fi
test "$found" = "$name"
sudo docker inspect "$name" --format '{{{{.State.Status}}}}'
"""
    status = _sensitive_ssh(
        host, command, "isolated container inspection", sensitive, 60
    ).strip()
    if status not in {
        "absent",
        "created",
        "restarting",
        "running",
        "removing",
        "paused",
        "exited",
        "dead",
    }:
        raise Refused(f"isolated container {name} returned invalid status {status!r}")
    return status


def _require_remote_source_matches(
    host: str,
    relative: str,
    sensitive: tuple[str, ...],
) -> None:
    local = repo_root / relative
    remote = f"{remote_repo}/{relative}"
    expected = _sha256_file(local, relative)
    actual = _sensitive_ssh(
        host,
        f"test -f {shlex.quote(remote)} && test ! -L {shlex.quote(remote)} && "
        f"sha256sum -- {shlex.quote(remote)} | cut -d' ' -f1",
        f"isolated parity source {relative}",
        sensitive,
        60,
    ).strip()
    if actual != expected:
        raise Refused(f"remote isolated parity source differs: {relative}")


def _copy_parity_diagnostic_input(
    source_host: str,
    source_path: str,
    destination_host: str,
    destination_path: str,
) -> None:
    _check_parity_host(source_host)
    _check_host(destination_host)
    source = PurePosixPath(source_path)
    destination = PurePosixPath(destination_path)
    parity_root = PurePosixPath("/data/glm52-rl/qualifications/glm52-policy-parity")
    if source.parent.parent != PurePosixPath(parity_trace_root) or not re.fullmatch(
        r"trainer-rank-0-pp-0-cp-0-diagnostic-layer3_router_input\.npz",
        source.name,
    ):
        raise Refused("trainer diagnostic input path is outside the exact trace root")
    if destination.parent != parity_root or re.fullmatch(
        r"\.diagnostic-(?:input|source)-[0-9a-f]{16}\.npz", destination.name
    ) is None:
        raise Refused("H200 diagnostic input path is outside the exact parity root")
    with tempfile.TemporaryDirectory(prefix="glm52-parity-diagnostic-") as temporary:
        local_input = Path(temporary) / "layer3-router-input.npz"
        download = subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                f"{source_host}:{source_path}",
                str(local_input),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if download.returncode or not local_input.is_file() or local_input.is_symlink():
            raise RuntimeError("trainer CP0 diagnostic input copy failed")
        upload = subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                str(local_input),
                f"{destination_host}:{destination_path}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if upload.returncode:
            raise RuntimeError("H200 diagnostic input copy failed")


def _resolve_glm52_parity_token_export(host: str, rank: int) -> str:
    _check_parity_host(host)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise Refused("final trainer token export rank is invalid")
    command = f"""
set -euo pipefail
root={shlex.quote(parity_token_root)}
test -d "$root"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
mapfile -t files < <(find "$root" -mindepth 2 -maxdepth 2 -type f -name {shlex.quote(f'rank_{rank}.jsonl')} -print)
test "${{#files[@]}}" -eq 1
file=${{files[0]}}
test ! -L "$file"
test -f "$(dirname "$file")/STABLE"
test ! -L "$(dirname "$file")/STABLE"
printf '%s\n' "$file"
"""
    output = _remote_ok(host, command, "final PP token export", timeout=60).strip()
    expected = re.fullmatch(
        re.escape(parity_token_root) + rf"/step_[0-9]+/rank_{rank}\.jsonl", output
    )
    if expected is None:
        raise Refused("final PP token export path is not unique and exact")
    return output


def _copy_parity_refresh_file(
    source_host: str,
    source_path: str,
    destination_host: str,
    destination_path: str,
    kind: str,
) -> None:
    _check_parity_host(source_host)
    _check_host(destination_host)
    source = PurePosixPath(source_path)
    destination = PurePosixPath(destination_path)
    root = PurePosixPath("/data/glm52-rl/qualifications/glm52-policy-parity")
    destination_patterns = {
        "session": r"\.diagnostic-session-[0-9a-f]{16}\.json",
        "token": r"\.diagnostic-token-[0-9a-f]{16}\.jsonl",
        "compact": r"\.diagnostic-compact-[0-9a-f]{16}\.npz",
        "compact_manifest": r"\.diagnostic-compact-[0-9a-f]{16}\.json",
    }
    if destination.parent != root or re.fullmatch(
        destination_patterns.get(kind, r"(?!)"), destination.name
    ) is None:
        raise Refused("parity refresh destination is outside the exact parity root")
    if kind == "session":
        valid_source = source.parent.parent == PurePosixPath(parity_trace_root) and (
            source.name == "session.json"
        )
    elif kind == "token":
        valid_source = (
            source.parent.parent == PurePosixPath(parity_token_root)
            and re.fullmatch(r"step_[0-9]+", source.parent.name) is not None
            and re.fullmatch(r"rank_[0-9]+\.jsonl", source.name) is not None
        )
    elif kind == "compact":
        valid_source = (
            source.parent.parent == PurePosixPath(parity_trace_root)
            and source.name == "trainer-fixture-source.npz"
        )
    elif kind == "compact_manifest":
        valid_source = (
            source.parent.parent == PurePosixPath(parity_trace_root)
            and source.name == "trainer-fixture-source.json"
        )
    else:
        raise Refused("parity refresh source kind is invalid")
    if not valid_source:
        raise Refused("parity refresh source is outside its exact root")
    with tempfile.TemporaryDirectory(prefix="glm52-parity-refresh-") as temporary:
        local = Path(temporary) / source.name
        for arguments, label in (
            (
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    f"{source_host}:{source_path}",
                    str(local),
                ],
                "trainer parity refresh download",
            ),
            (
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    str(local),
                    f"{destination_host}:{destination_path}",
                ],
                "H200 parity refresh upload",
            ),
        ):
            result = subprocess.run(
                arguments, capture_output=True, text=True, timeout=600
            )
            if result.returncode:
                raise RuntimeError(f"{label} failed")
        if not local.is_file() or local.is_symlink():
            raise RuntimeError("parity refresh transfer did not produce a regular file")


def _resolve_glm52_parity_diagnostic_source(
    name: str,
) -> tuple[str, str, int, str, dict[str, object]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("parity diagnostic name contains unsafe characters")
    trainer_config = repo_root / "configs/qualification/glm52-policy-parity-trainer.toml"
    with trainer_config.open("rb") as stream:
        trainer_values = tomllib.load(stream)
    model_values = trainer_values.get("model", {})
    pp_size = model_values.get("pp")
    cp_size = model_values.get("cp")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (pp_size, cp_size)
    ):
        raise Refused("parity trainer TOML must define positive model.pp and model.cp")
    if len(trainer_hosts) != pp_size or trainer_head not in trainer_hosts:
        raise Refused("configured trainer hosts do not match the parity PP topology")
    source_root = f"{parity_trace_root}/{name}"
    command = f"""
set -euo pipefail
root={shlex.quote(source_root)}
test -d "$root"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
for file in session.json trainer-fixture-source.npz trainer-fixture-source.json trainer-rank-0-pp-0-cp-0-diagnostic-layer3_router_input.npz; do
  test -f "$root/$file"
  test ! -L "$root/$file"
done
python3 - "$root" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
names={{
 "stable":"STABLE",
 "union":"trainer-provenance-union.json",
 "manifest":"trainer-fixture-source.json",
 "source":"trainer-fixture-source.npz",
}}
paths={{name:root/value for name,value in names.items()}}
for path in paths.values():
 assert path.is_file() and not path.is_symlink()
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
 return h.hexdigest()
def canonical(value):
 return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n").encode()
stable=json.loads(paths["stable"].read_text())
union=json.loads(paths["union"].read_text())
manifest=json.loads(paths["manifest"].read_text())
for value,kind in ((stable,"glm52_policy_parity_trainer_bundle_stable"),(union,"glm52_policy_parity_trainer_provenance_union"),(manifest,"glm52_policy_parity_trainer_fixture_source")):
 assert value["kind"]==kind
 claimed=value.pop("content_sha256")
 assert claimed==hashlib.sha256(canonical(value)).hexdigest()
 value["content_sha256"]=claimed
assert stable["provenance_union_sha256"]==digest(paths["union"])
assert stable["fixture_source_manifest_sha256"]==digest(paths["manifest"])
assert stable["fixture_source_sha256"]==digest(paths["source"])
assert union["fixture_source_manifest_sha256"]==stable["fixture_source_manifest_sha256"]
assert union["fixture_source_sha256"]==stable["fixture_source_sha256"]
assert union["binding_sha256"]==manifest["binding_sha256"]==stable["binding_sha256"]
assert union["session"]==json.loads((root/"session.json").read_text())
binding=union["binding"]
assert union["binding_sha256"]==hashlib.sha256(canonical(binding)).hexdigest()
assert binding["session_identity"]==union["session_identity"]
assert binding["model_revision"]==union["session"]["model_revision"]
assert binding["adapter_sha256"]==union["session"]["adapter_sha256"]
assert binding["fixture_sha256"]==union["session"]["fixture_sha256"]
assert binding["token_ids_sha256"]==union["session"]["token_ids_sha256"]
assert binding["records_sha256"]==hashlib.sha256(canonical(union["records"])).hexdigest()
assert binding["modules_sha256"]==hashlib.sha256(canonical(union["records"][0]["modules"])).hexdigest()
assert paths["source"].stat().st_size<=18*1024*1024
print(json.dumps({{"union":union,"manifest":manifest}},sort_keys=True))
PY
"""
    output = _remote_ok(
        trainer_head,
        command,
        "trainer parity provenance bundle",
        timeout=60,
        connect_timeout=45,
    )
    try:
        bundle = json.loads(output.strip().splitlines()[-1])
        union = bundle["union"]
        compact_manifest = bundle["manifest"]
        session_value = union["session"]
        record_values = union["records"]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
        raise RuntimeError("trainer parity provenance bundle returned invalid JSON") from error
    if (
        not isinstance(union, dict)
        or union.get("kind") != "glm52_policy_parity_trainer_provenance_union"
        or not isinstance(compact_manifest, dict)
        or compact_manifest.get("kind")
        != "glm52_policy_parity_trainer_fixture_source"
        or not isinstance(record_values, list)
        or len(record_values) != pp_size
    ):
        raise Refused("trainer parity provenance bundle is invalid")
    evidence = {
        str(record.get("host")): {
            "session": session_value,
            "provenance": record,
            "provenance_file": (
                f"trainer-runtime-pp-{record.get('pp_rank')}-host-"
                f"{record.get('host')}.json"
            ),
        }
        for record in record_values
        if isinstance(record, dict)
    }
    records: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    module_sets: list[object] = []
    for host in trainer_hosts:
        value = evidence.get(host)
        if not isinstance(value, dict):
            raise Refused(f"trainer parity evidence is missing from {host}")
        session = value.get("session")
        record = value.get("provenance")
        filename = value.get("provenance_file")
        if not isinstance(session, dict) or not isinstance(record, dict):
            raise Refused(f"trainer parity evidence is invalid on {host}")
        match = re.fullmatch(
            r"trainer-runtime-pp-(\d+)-host-(n\d+)\.json", str(filename)
        )
        if match is None:
            raise Refused(f"trainer parity provenance filename is invalid on {host}")
        pp_rank = record.get("pp_rank")
        cp_rank = record.get("cp_rank")
        if (
            record.get("schema_version") != 1
            or record.get("batch_size") != 8
            or
            record.get("kind") != "glm52_policy_parity_trainer_runtime"
            or record.get("host") != host
            or match.group(2) != host
            or not isinstance(pp_rank, int)
            or isinstance(pp_rank, bool)
            or int(match.group(1)) != pp_rank
            or cp_rank != 0
            or record.get("global_rank") != pp_rank * cp_size
            or record.get("pp_size") != pp_size
            or record.get("cp_size") != cp_size
        ):
            raise Refused(f"trainer parity rank provenance is invalid on {host}")
        image_digest = record.get("image_digest")
        if not isinstance(image_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_digest
        ):
            raise Refused(f"trainer parity image digest is invalid on {host}")
        modules = record.get("modules")
        if not isinstance(modules, dict) or not modules:
            raise Refused(f"trainer parity module provenance is invalid on {host}")
        for module in modules.values():
            if (
                not isinstance(module, dict)
                or module.get("location") not in {"mounted", "unmounted"}
                or not isinstance(module.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", module["sha256"]) is None
            ):
                raise Refused(f"trainer parity module provenance is invalid on {host}")
        records.append(record)
        sessions.append(session)
        module_sets.append(modules)
    pp0 = [
        record
        for record in records
        if record.get("pp_rank") == 0 and record.get("cp_rank") == 0
    ]
    if len(pp0) != 1:
        raise Refused("trainer parity provenance must contain one PP0/CP0 host")
    if {record["pp_rank"] for record in records} != set(range(pp_size)):
        raise Refused("trainer parity provenance does not uniquely cover every PP rank")
    dense_layers = session_value.get("dense_layers")
    hidden_layers = session_value.get("hidden_layers")
    if (
        isinstance(dense_layers, bool)
        or not isinstance(dense_layers, int)
        or isinstance(hidden_layers, bool)
        or not isinstance(hidden_layers, int)
        or not 0 <= dense_layers < hidden_layers
    ):
        raise Refused("trainer parity session layer counts are invalid")
    layer_owners: dict[int, int] = {}
    for record in records:
        owned = record.get("owned_layers")
        if not isinstance(owned, list) or not owned:
            raise Refused("trainer parity layer ownership is missing")
        for layer in owned:
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < dense_layers
                or layer >= hidden_layers
                or layer in layer_owners
            ):
                raise Refused("trainer parity layer ownership is invalid")
            layer_owners[layer] = int(record["pp_rank"])
    if set(layer_owners) != set(range(dense_layers, hidden_layers)):
        raise Refused("trainer parity layer ownership is incomplete")
    source_host = str(pp0[0]["host"])
    if source_host != trainer_head:
        raise Refused(
            "trainer parity PP0/CP0 host differs from the configured trainer head"
        )
    if any(session != sessions[0] for session in sessions[1:]):
        raise Refused("trainer parity sessions differ between hosts")
    session = sessions[0]
    if (
        session.get("side") != "trainer"
        or not isinstance(session.get("model_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", session["model_revision"]) is None
        or not isinstance(session.get("adapter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", session["adapter_sha256"]) is None
    ):
        raise Refused("trainer parity session identity is invalid")
    if any(modules != module_sets[0] for modules in module_sets[1:]):
        raise Refused("trainer parity module hashes differ between hosts")
    final = [record for record in records if record.get("pp_rank") == pp_size - 1]
    if len(final) != 1:
        raise Refused("trainer parity provenance must contain one final PP host")
    final_host = str(final[0]["host"])
    final_rank = int(final[0]["global_rank"])
    if (
        compact_manifest.get("rank") != final_rank
        or compact_manifest.get("file") != "trainer-fixture-source.npz"
        or compact_manifest.get("binding_sha256") != union.get("binding_sha256")
        or compact_manifest.get("file_sha256")
        != union.get("fixture_source_sha256")
    ):
        raise Refused("trainer compact fixture source differs from provenance")
    return source_host, final_host, final_rank, source_root, session


def diagnose_glm52_policy_parity(name: str, keep_engine: bool = False) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("parity diagnostic name contains unsafe characters")
    host = orchestrator_host
    if host != inference_hosts[0] or len(inference_hosts) != 1:
        raise Refused("parity diagnostics require the one configured H200 host")
    config_path = repo_root / "configs/qualification/glm52-policy-parity-h200.toml"
    plan = _diagnostic_capture_preflight(config_path, "h200")
    if plan["files"] == 0:
        raise Refused("H200 parity diagnostics are not enabled")
    root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    capture = glm_parity_paths["h200_capture"]
    fixture = glm_parity_paths["fixture"]
    if PurePosixPath(capture).parent != PurePosixPath(root):
        raise Refused("H200 capture path is outside the exact parity root")
    source_host, _final_host, final_rank, source_root, metadata = _resolve_glm52_parity_diagnostic_source(name)
    source_input = f"{source_root}/trainer-rank-0-pp-0-cp-0-" "diagnostic-layer3_router_input.npz"
    session_path = f"{source_root}/session.json"
    compact_source = f"{source_root}/trainer-fixture-source.npz"
    compact_manifest = f"{source_root}/trainer-fixture-source.json"
    nonce = secrets.token_hex(8)
    bootstrap_token = ""
    sensitive: tuple[str, ...] = ()
    container_writes = _parity_container_write_paths(nonce)
    run_directory = f"{root}/.diagnostic-run-{nonce}"
    bootstrap_fifo = container_writes["bootstrap_fifo"]
    source_diagnostic_input = f"{root}/.diagnostic-source-{nonce}.npz"
    source_session = f"{root}/.diagnostic-session-{nonce}.json"
    source_compact = f"{root}/.diagnostic-compact-{nonce}.npz"
    source_compact_manifest = f"{root}/.diagnostic-compact-{nonce}.json"
    transaction_path = f"{run_directory}/transaction.json"
    container_work_directory = container_writes["work"]
    remote_input = container_writes["diagnostic"]
    fixture_candidate = container_writes["fixture"]
    fixture_previous = f"{root}/.fixture-previous-{nonce}"
    temporary_capture = container_writes["capture"]
    h200_trace_directory = container_writes["router_trace"]
    capture_previous = f"{root}/.h200-capture-previous-{nonce}"
    engine_launch_attempted = False
    fixture_published = False
    fixture_replaced = False
    capture_published = False
    capture_replaced = False
    succeeded = False
    committed = False
    rollback_completed = False
    transaction_present = False
    listener_identity: dict[str, object] | None = None
    api_listener_identity: dict[str, object] | None = None
    bootstrap_identity: dict[str, object] | None = None
    cleanup_warnings: list[str] = []
    operation_error: BaseException | None = None
    try:
        root_command = (
            f"test -d {shlex.quote(root)} && test ! -L {shlex.quote(root)} && "
            f'test "$(realpath -e {shlex.quote(root)})" = {shlex.quote(root)}'
        )
        _sensitive_ssh(host, root_command, "isolated parity root and config", sensitive, 60)
        for relative in (
            "configs/qualification/glm52-policy-parity-h200.toml",
            "scripts/h200/start_inference.sh",
            "scripts/glm52_policy_parity.py",
            "source/src/prime_rl/inference/vllm/server.py",
            "source/src/prime_rl/parity_router_trace.py",
        ):
            _require_remote_source_matches(host, relative, sensitive)
        _assert_h200_parity_port_free(host)

        for related_name in (
            f"{glm_inference_container}-0",
            f"{glm_inference_container}-1",
        ):
            related_status = _isolated_container_status(host, related_name, sensitive)
            if related_status != "absent":
                raise Refused(f"related isolated container {related_name} is {related_status!r}; expected absent")
        status = _isolated_container_status(host, glm_inference_container, sensitive)
        if status == "exited":
            _sensitive_ssh(
                host,
                f"sudo docker rm {shlex.quote(glm_inference_container)} >/dev/null",
                "exited isolated container removal",
                sensitive,
                60,
            )
        elif status != "absent":
            raise Refused(f"isolated parity container status is {status!r}, expected absent or exited")

        fifo_setup = f"""
set -euo pipefail
root={shlex.quote(root)}
run_directory={shlex.quote(run_directory)}
fifo={shlex.quote(bootstrap_fifo)}
test ! -e "$run_directory"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
sudo install -d -o 0 -g 0 -m 0700 "$run_directory"
sudo mkfifo -m 0600 "$fifo"
sudo chown 1000:1000 "$fifo"
test ! -L "$fifo"
test -p "$fifo"
test "$(stat -c '%u:%g:%a' "$fifo")" = "1000:1000:600"
echo fifo-ready
"""
        fifo_state = _sensitive_ssh(host, fifo_setup, "parity bootstrap FIFO setup", sensitive, 60).strip()
        if fifo_state != "fifo-ready":
            raise RuntimeError("parity bootstrap FIFO setup returned an invalid state")

        _prepare_parity_container_work(host, container_writes, sensitive)
        _copy_parity_diagnostic_input(source_host, source_input, host, source_diagnostic_input)
        _copy_parity_refresh_file(source_host, session_path, host, source_session, "session")
        _copy_parity_refresh_file(source_host, compact_source, host, source_compact, "compact")
        _copy_parity_refresh_file(
            source_host,
            compact_manifest,
            host,
            source_compact_manifest,
            "compact_manifest",
        )
        refresh_command = (
            "sudo docker run --rm -v /data:/data "
            f"-v {shlex.quote(remote_repo)}:/workspace:ro -w /workspace "
            f"--entrypoint /app/.venv/bin/python {shlex.quote(inference_image)} "
            f"scripts/glm52_policy_parity.py --profile {shlex.quote(production_parity_profile)} "
            f"refresh-fixture --source-fixture {shlex.quote(fixture)} "
            f"--source-session {shlex.quote(source_session)} "
            f"--compact-source {shlex.quote(source_compact)} "
            f"--compact-manifest {shlex.quote(source_compact_manifest)} "
            f"--diagnostic-input {shlex.quote(source_diagnostic_input)} "
            f"--diagnostic-output {shlex.quote(remote_input)} "
            f"--expected-rank {final_rank} {shlex.quote(fixture_candidate)}"
        )
        refresh_output = _sensitive_ssh(host, refresh_command, "current parity fixture rebuild", sensitive, 600)
        _verify_parity_container_outputs(host, container_writes, "refresh", sensitive)
        try:
            refresh_result = json.loads(refresh_output)
        except json.JSONDecodeError as error:
            raise RuntimeError("current parity fixture rebuild returned invalid JSON") from error
        for field in (
            "fixture_sha256",
            "source_fixture_sha256",
            "token_ids_sha256",
            "positions_sha256",
            "advantages_sha256",
            "token_export_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(refresh_result.get(field, ""))) is None:
                raise RuntimeError(f"current parity fixture rebuild omitted {field}")
        if refresh_result.get("rank") != final_rank:
            raise RuntimeError("current parity fixture rebuild used another trainer rank")
        transaction_state: dict[str, object] = {
            "schema_version": 1,
            "nonce": nonce,
            "phase": "fixture_ready",
            "fixture": {
                "action": None,
                "new_manifest_sha256": refresh_result["fixture_sha256"],
                "old_manifest_sha256": refresh_result["source_fixture_sha256"],
                "paths": {
                    "live": fixture,
                    "previous": fixture_previous,
                    "candidate": fixture_candidate,
                    "failed": f"{root}/.fixture-failed-{nonce}",
                },
            },
            "capture": {
                "action": None,
                "new_manifest_sha256": None,
                "old_manifest_sha256": None,
                "paths": {
                    "live": capture,
                    "previous": capture_previous,
                    "candidate": temporary_capture,
                    "failed": f"{root}/.h200-capture-failed-{nonce}",
                },
            },
        }
        _write_parity_transaction_state(host, transaction_path, transaction_state, sensitive)
        transaction_present = True
        fixture_publish = f"""
set -euo pipefail
root={shlex.quote(root)}
fixture={shlex.quote(fixture)}
candidate={shlex.quote(fixture_candidate)}
previous={shlex.quote(fixture_previous)}
failed={shlex.quote(root + '/.fixture-publish-failed-' + nonce)}
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
test -d "$candidate"
test ! -L "$candidate"
test -f "$candidate/manifest.json"
test ! -e "$previous"
test ! -e "$failed"
changed=0
had_previous=0
restore_fixture() {{
  if test "$changed" = 1; then
    if test -e "$fixture"; then mv "$fixture" "$failed"; fi
    if test "$had_previous" = 1 && test -e "$previous"; then mv "$previous" "$fixture"; fi
    if test -e "$failed"; then rm -rf -- "$failed"; fi
  fi
}}
trap restore_fixture EXIT
state=created
if test -e "$fixture"; then
  test -d "$fixture"
  test ! -L "$fixture"
  mv "$fixture" "$previous"
  changed=1
  had_previous=1
  state=replaced
fi
mv "$candidate" "$fixture"
changed=1
test -f "$fixture/manifest.json"
{_parity_transaction_update_command(transaction_path, "fixture_published", "fixture", '"$state"')}
trap - EXIT
printf '%s\n' "$state"
"""
        fixture_publish_error: BaseException | None = None
        try:
            fixture_ack = _sensitive_ssh(
                host, fixture_publish, "current parity fixture publish", sensitive, 120
            ).strip()
            if fixture_ack not in {"created", "replaced"}:
                raise RuntimeError("current parity fixture publish returned an invalid state")
        except BaseException as error:
            fixture_publish_error = error
        fixture_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
        try:
            fixture_replaced = _validate_published_transaction_artifact(fixture_snapshot, "fixture")
        except RuntimeError:
            if fixture_publish_error is not None:
                raise fixture_publish_error
            raise
        fixture_published = True

        start_command = (
            "env "
            f"prime_deployment={shlex.quote(remote_repo)} "
            f"prime_image={shlex.quote(inference_image)} "
            f"prime_model_volume={shlex.quote(glm_model_volume)} "
            f"prime_inference_container={shlex.quote(glm_inference_container)} "
            f"prime_parity_bootstrap_fifo={shlex.quote(bootstrap_fifo)} "
            f"prime_parity_trace_dir={shlex.quote(h200_trace_directory)} "
            "prime_require_absent=1 "
            f"bash {shlex.quote(remote_repo + '/scripts/h200/start_inference.sh')} "
            f"{shlex.quote(glm_parity_inference_config)}"
        )
        engine_launch_attempted = True
        _sensitive_ssh(host, start_command, "isolated parity engine start", sensitive, 180)
        startup_grace = _configured_h200_engine_startup_grace()
        _h200_parity_container_preflight(host, h200_trace_directory)
        fifo_read = _parity_bootstrap_receive_command(
            bootstrap_fifo,
            startup_grace,
        )
        bootstrap_token = _receive_parity_bootstrap(
            host,
            fifo_read,
            startup_grace + 30,
        )
        sensitive = (bootstrap_token,)
        listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            startup_grace,
        )
        api_listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            startup_grace,
            h200_parity_backend_port,
        )
        if (
            api_listener_identity["container_id"] != listener_identity["container_id"]
            or api_listener_identity["host_boot_id"] != listener_identity["host_boot_id"]
        ):
            raise Refused("H200 parity router and API listeners differ")
        expected_api_process_id = api_listener_identity["listener_namespace_pid"]
        capture_command = (
            "sudo docker run --rm -i --network host -v /data:/data "
            f"-v {shlex.quote(remote_repo)}:/workspace:ro -w /workspace "
            f"--entrypoint /app/.venv/bin/python {shlex.quote(inference_image)} "
            f"scripts/glm52_policy_parity.py --profile {shlex.quote(production_parity_profile)} "
            f"capture-h200 --fixture {shlex.quote(fixture)} "
            "--base-url http://127.0.0.1:8000 "
            f"--control-url http://127.0.0.1:{h200_parity_backend_port} "
            f"--expected-api-process-id {expected_api_process_id} "
            f"--model {shlex.quote(glm_model)} "
            f"--model-revision {shlex.quote(metadata['model_revision'])} "
            f"--adapter-sha256 {shlex.quote(metadata['adapter_sha256'])} "
            f"--image {shlex.quote(inference_image)} "
            f"--diagnostic-input {shlex.quote(remote_input)} "
            f"{shlex.quote(temporary_capture)}"
        )
        _sensitive_ssh_input(
            host,
            capture_command,
            bootstrap_token,
            "H200 parity capture and replay",
            3600,
        )
        _verify_parity_container_outputs(host, container_writes, "capture", sensitive)
        final_api_listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            60,
            h200_parity_backend_port,
        )
        if final_api_listener_identity != api_listener_identity:
            raise Refused("H200 parity API listener changed during capture")
        final_listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            60,
        )
        if final_listener_identity != listener_identity:
            raise Refused("H200 parity router listener changed during capture")
        bootstrap_identity = _h200_capture_bootstrap_identity(
            host,
            temporary_capture,
            bootstrap_token,
            expected_api_process_id,
        )
        capture_new_sha256 = _remote_manifest_sha256(
            host, temporary_capture, "new H200 parity capture manifest", sensitive
        )
        if capture_new_sha256 is None:
            raise RuntimeError("new H200 parity capture manifest is missing")
        capture_old_sha256 = _remote_manifest_sha256(host, capture, "existing H200 parity capture manifest", sensitive)
        current_state = fixture_snapshot.get("state")
        if not isinstance(current_state, dict):
            raise RuntimeError("fixture transaction state is missing before capture publish")
        transaction_state = current_state
        transaction_state["phase"] = "capture_ready"
        capture_state_value = transaction_state.get("capture")
        if not isinstance(capture_state_value, dict):
            raise RuntimeError("capture transaction state is missing")
        capture_state_value["new_manifest_sha256"] = capture_new_sha256
        capture_state_value["old_manifest_sha256"] = capture_old_sha256
        _write_parity_transaction_state(host, transaction_path, transaction_state, sensitive)
        publish_command = f"""
set -euo pipefail
root={shlex.quote(root)}
capture={shlex.quote(capture)}
candidate={shlex.quote(temporary_capture)}
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
test -f "$candidate/manifest.json"
python3 -c 'import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text())
assert m.get("kind")=="glm52_policy_parity_h200_capture"
assert m.get("direct_router_diagnostic") is not None
for e in m.get("arrays",{{}}).values():
 f=p/e["file"]; assert f.is_file() and not f.is_symlink()
 assert hashlib.sha256(f.read_bytes()).hexdigest()==e["sha256"]' "$candidate"
previous={shlex.quote(capture_previous)}
failed="$root/.h200-capture-publish-failed-{nonce}"
test ! -e "$previous"
test ! -e "$failed"
changed=0
had_previous=0
restore_previous() {{
  if test "$changed" = 1; then
    if test -e "$capture"; then mv "$capture" "$failed"; fi
    if test "$had_previous" = 1 && test -e "$previous"; then mv "$previous" "$capture"; fi
    if test -e "$failed"; then rm -rf -- "$failed"; fi
  fi
}}
trap restore_previous EXIT
state=created
if test -e "$capture"; then
  test -d "$capture"
  test ! -L "$capture"
  mv "$capture" "$previous"
  changed=1
  had_previous=1
  state=replaced
fi
mv "$candidate" "$capture"
changed=1
test -f "$capture/manifest.json"
{_parity_transaction_update_command(transaction_path, "capture_published", "capture", '"$state"')}
trap - EXIT
printf '%s\n' "$state"
"""
        capture_publish_error: BaseException | None = None
        try:
            capture_ack = _sensitive_ssh(host, publish_command, "H200 parity capture publish", sensitive, 300).strip()
            if capture_ack not in {"created", "replaced"}:
                raise RuntimeError("H200 parity capture publish returned an invalid state")
        except BaseException as error:
            capture_publish_error = error
        capture_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
        try:
            fixture_replaced = _validate_published_transaction_artifact(capture_snapshot, "fixture")
            capture_replaced = _validate_published_transaction_artifact(capture_snapshot, "capture")
        except RuntimeError:
            if capture_publish_error is not None:
                raise capture_publish_error
            raise
        capture_published = True
        succeeded = True
        return {
            "capture": capture,
            "listener_identity": listener_identity,
            "api_listener_identity": api_listener_identity,
            "bootstrap_identity": bootstrap_identity,
            "diagnostic_plan": plan,
            "engine_kept": keep_engine,
            "container_work": container_work_directory if keep_engine else None,
            "cleanup_warnings": cleanup_warnings,
        }
    except BaseException as error:
        operation_error = error
        raise
    finally:
        cleanup_failures = []
        if engine_launch_attempted and (not keep_engine or not succeeded):
            container_cleanup = f"""
set -euo pipefail
name={shlex.quote(glm_inference_container)}
found=$(sudo docker ps -a --filter "name=^/${{name}}$" --format '{{{{.Names}}}}')
if test -n "$found"; then
  test "$found" = "$name"
  sudo docker rm -f "$name" >/dev/null
fi
test -z "$(sudo docker ps -a --filter "name=^/${{name}}$" --format '{{{{.Names}}}}')"
"""
            container_code, container_output = _ssh(
                host,
                container_cleanup,
                timeout=180,
            )
            if container_code:
                cleanup_failures.append(
                    "isolated parity container cleanup failed: "
                    + _redact_sensitive(container_output.strip(), sensitive)
                )
        retain_container_work = keep_engine and succeeded
        cleanup = _parity_temporary_cleanup_command(
            bootstrap_fifo=bootstrap_fifo,
            transaction_path=transaction_path,
            run_directory=run_directory,
            source_diagnostic_input=source_diagnostic_input,
            source_session=source_session,
            source_compact=source_compact,
            source_compact_manifest=source_compact_manifest,
            container_work_directory=container_work_directory,
            retain_container_work=retain_container_work,
        )
        cleanup_code, cleanup_output = _ssh(host, cleanup, timeout=120)
        if cleanup_code:
            cleanup_failures.append(
                "temporary parity artifact cleanup failed: " + _redact_sensitive(cleanup_output.strip(), sensitive)
            )
        if operation_error is None and not cleanup_failures and succeeded:
            try:
                commit_command = _parity_transaction_update_command(transaction_path, "committed", "none", "committed")
                _sensitive_ssh(
                    host,
                    commit_command,
                    "parity transaction commit record",
                    sensitive,
                    60,
                )
                commit_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
                commit_state = commit_snapshot.get("state")
                if not isinstance(commit_state, dict) or commit_state.get("phase") != ("committed"):
                    raise RuntimeError("parity transaction commit state differs")
                fixture_replaced = _validate_published_transaction_artifact(commit_snapshot, "fixture")
                capture_replaced = _validate_published_transaction_artifact(commit_snapshot, "capture")
                committed = True
            except BaseException as error:
                cleanup_failures.append(f"parity transaction commit failed: {error}")
        if operation_error is not None or cleanup_failures:
            state_code, _state_output = _ssh(host, f"test -f {shlex.quote(transaction_path)}", timeout=60)
            rollback_artifacts: list[dict[str, object]] = []
            reconciliation_ok = False
            if state_code == 0:
                transaction_present = True
                try:
                    rollback_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
                    rollback_state = rollback_snapshot.get("state")
                    if not isinstance(rollback_state, dict):
                        raise RuntimeError("remote transaction state is missing")
                    for kind in ("fixture", "capture"):
                        artifact = rollback_state.get(kind)
                        if not isinstance(artifact, dict) or artifact.get("action") is None:
                            continue
                        replaced = _validate_published_transaction_artifact(rollback_snapshot, kind)
                        paths = artifact.get("paths")
                        if not isinstance(paths, dict):
                            raise RuntimeError(f"remote {kind} recovery paths are missing")
                        rollback_artifacts.append(
                            {
                                "live": paths.get("live"),
                                "previous": paths.get("previous"),
                                "failed": paths.get("failed"),
                                "replaced": replaced,
                            }
                        )
                    reconciliation_ok = True
                except BaseException as error:
                    cleanup_failures.append(
                        "parity transaction reconciliation failed; recovery paths retained: " + str(error)
                    )
            elif fixture_published or capture_published:
                cleanup_failures.append("parity transaction state is missing; recovery paths retained")
            if rollback_artifacts:
                rollback = _parity_pair_rollback_command(
                    rollback_artifacts,
                    _parity_transaction_update_command(transaction_path, "rolled_back", "none", "rolled_back"),
                )
                rollback_code, rollback_output = _ssh(host, rollback, timeout=300)
                if rollback_code:
                    cleanup_failures.append(
                        "parity pair rollback did not complete; recovery paths retained: "
                        + _redact_sensitive(rollback_output.strip(), sensitive)
                    )
                else:
                    fixture_published = False
                    capture_published = False
                    rollback_completed = True
            elif reconciliation_ok:
                rollback_completed = True
        elif committed:
            for replaced, previous, label in (
                (fixture_replaced, fixture_previous, "fixture"),
                (capture_replaced, capture_previous, "capture"),
            ):
                if not replaced:
                    continue
                backup_cleanup = (
                    f"test -d {shlex.quote(previous)} && "
                    f"test ! -L {shlex.quote(previous)} && "
                    f"rm -rf -- {shlex.quote(previous)}"
                )
                backup_code, backup_output = _ssh(host, backup_cleanup, timeout=180)
                if backup_code:
                    warning = (
                        f"committed parity {label} backup cleanup failed; "
                        f"backup remains at {previous}: " + _redact_sensitive(backup_output.strip(), sensitive)
                    )
                    cleanup_warnings.append(warning)
                    print("WARNING: " + warning)
        remove_transaction = (
            (committed and not cleanup_failures and not cleanup_warnings)
            or (rollback_completed and not cleanup_failures)
        )
        if transaction_present and remove_transaction:
            try:
                _remove_completed_parity_transaction(
                    host,
                    transaction_path,
                    run_directory,
                    sensitive,
                )
                transaction_present = False
            except BaseException as error:
                message = (
                    "completed parity transaction cleanup failed; "
                    f"state retained at {run_directory}: {error}"
                )
                if operation_error is None:
                    cleanup_warnings.append(message)
                    print("WARNING: " + message)
                else:
                    cleanup_failures.append(message)
        if transaction_present:
            retained = f"parity transaction state retained at {run_directory}"
            reported = any(
                run_directory in message
                for message in (*cleanup_failures, *cleanup_warnings)
            )
            if not reported:
                if operation_error is not None or cleanup_failures:
                    cleanup_failures.append(retained)
                else:
                    cleanup_warnings.append(retained)
                    print("WARNING: " + retained)
        bootstrap_token = ""
        sensitive = ()
        if cleanup_failures:
            detail = "; ".join(cleanup_failures)
            if operation_error is not None:
                operation_error.add_note(detail)
            else:
                raise RuntimeError(detail)
        if succeeded and not committed:
            raise RuntimeError("parity fixture and capture transaction did not commit")


def cleanup_glm52_policy_parity_engine() -> str:
    host = orchestrator_host
    if host != inference_hosts[0] or len(inference_hosts) != 1:
        raise Refused("parity cleanup requires the one configured H200 host")
    if glm_inference_container in set(isolation["protected_inference"]):
        raise Refused("parity cleanup container overlaps protected inference")
    for related in (
        f"{glm_inference_container}-0",
        f"{glm_inference_container}-1",
    ):
        status = _isolated_container_status(host, related, ())
        if status != "absent":
            raise Refused(f"related isolated container {related} is {status!r}")
    status = _isolated_container_status(host, glm_inference_container, ())
    if status == "absent":
        return "absent"
    if status != "exited":
        raise Refused(
            f"isolated parity container is {status!r}; only an exited container can be removed"
        )
    inspection = _sensitive_ssh(
        host,
        f"sudo docker inspect {shlex.quote(glm_inference_container)} "
        "--format '{{json .Config.Image}}\\n{{json .Config.Cmd}}\\n"
        "{{json .HostConfig.RestartPolicy.Name}}'",
        "exited parity container inspection",
        (),
        60,
    ).replace("\\n", "\n").splitlines()
    if len(inspection) != 3:
        raise Refused("exited parity container inspection returned an invalid record")
    try:
        image, command, restart_policy = (json.loads(value) for value in inspection)
    except json.JSONDecodeError as error:
        raise Refused("exited parity container inspection is not valid JSON") from error
    expected_command = [
        "@",
        glm_parity_inference_config,
        "--vllm-extra",
        glm_parity_vllm_extra,
    ]
    if image != inference_image or command != expected_command or restart_policy != "no":
        raise Refused("exited container does not match the isolated parity engine")
    command_output = _remote_ok(
        host,
        f"set -euo pipefail; name={shlex.quote(glm_inference_container)}; "
        "test \"$(sudo docker inspect \"$name\" --format '{{.State.Status}}')\" = exited; "
        "sudo docker rm \"$name\" >/dev/null; "
        "test -z \"$(sudo docker ps -a --filter \"name=^/${name}$\" "
        "--format '{{.Names}}')\"; echo removed",
        "exited isolated parity container cleanup",
    ).strip()
    if command_output != "removed":
        raise RuntimeError("exited isolated parity container cleanup returned invalid output")
    return "removed"


def _repair_engine(host: str) -> str:
    _check_host(host)
    if host not in inference_hosts:
        raise Refused(f"{host} is not an engine host. Use one of {inference_hosts}.")
    muse_engine = _profile()["kind"] == "muse"
    if not muse_engine:
        raise Refused(
            "start the isolated GLM parity engine with diagnose-glm52-parity and a trace run name"
        )
    engine_container = muse_inference_container
    engine_model = muse_model
    start_script = f"{remote_repo}/scripts/h200/start_muse_inference.sh"
    startup_grace = 600
    print(
        f"{host}: removing and restarting {engine_container}. The model load needs 5 to 10 minutes."
    )
    stop_code, stop_output = _ssh(
        host,
        f"pkill -f '^bash {remote_repo}/scripts/h200/engine_watchdog.sh$' 2>/dev/null; echo stopped",
    )
    if stop_code:
        raise RuntimeError(f"cannot stop the watchdog on {host}: {stop_output.strip()}")
    rc, out = _ssh(
        host,
        f"sudo docker rm -f {engine_container} >/dev/null 2>&1; "
        f"nohup setsid env prime_deployment={shlex.quote(remote_repo)} "
        f"prime_image={shlex.quote(inference_image)} "
        f"prime_model_volume={shlex.quote(glm_model_volume)} "
        f"prime_inference_container={shlex.quote(engine_container)} "
        f"bash {start_script} >/root/start_inference.log 2>&1 </dev/null & "
        "sleep 3; echo issued",
        timeout=180,
    )
    if rc:
        raise RuntimeError(f"inference restart failed on {host}: {out.strip()}")
    settings = _inference_config(host, "muse")
    watchdogs = []
    for replica, port in enumerate(settings["ports"]):
        watchdogs.append(
            f"nohup setsid env STARTUP_GRACE={startup_grace} MODEL={engine_model} "
            f"MUSE_GPU_COUNT={settings['gpu_count']} "
            f"MUSE_TP_SIZE={settings['tensor_parallel_size']} "
            f"PORT={port} MUSE_REPLICA_INDEX={replica} START={start_script} "
            f"ENGINE_CONTAINER={engine_container}-{replica} "
            f"bash {remote_repo}/scripts/h200/engine_watchdog.sh "
            f">> /root/engine_watchdog-{replica}.log 2>&1 </dev/null &"
        )
    watchdog_code, watchdog_output = _ssh(host, " ".join(watchdogs) + " disown")
    if watchdog_code:
        raise RuntimeError(
            f"watchdog restart failed on {host}: {watchdog_output.strip()}"
        )
    print("Do not test with /v1/models. Test with engines() when the load finishes.")
    return out.strip()


def _repair_supervisors() -> str:
    if _profile()["kind"] != "muse":
        raise Refused(
            "isolated GLM parity engines do not use a reusable engine supervisor"
        )
    out = []
    for pat, cmd, log in (
        (
            "bash .*tunnel_watchdog.sh",
            f"nohup setsid env TIMEOUT=20 INTERVAL=45 bash {remote_repo}/scripts/china/tunnel_watchdog.sh "
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
        muse_engine = _profile()["kind"] == "muse"
        rc, count_text = _ssh(
            h,
            f"pgrep -f '^bash {remote_repo}/scripts/h200/engine_watchdog.sh$' | wc -l",
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
            f"pkill -f '^bash {remote_repo}/scripts/h200/engine_watchdog.sh$' 2>/dev/null; true",
        )
        if stop_code:
            raise RuntimeError(f"cannot stop watchdogs on {h}: {stop_output.strip()}")
        if muse_engine:
            start_script = f"{remote_repo}/scripts/h200/start_muse_inference.sh"
            watchdogs = []
            for replica, port in enumerate(settings["ports"]):
                watchdogs.append(
                    f"nohup setsid env STARTUP_GRACE=600 MODEL={muse_model} "
                    f"MUSE_GPU_COUNT={settings['gpu_count']} "
                    f"MUSE_TP_SIZE={settings['tensor_parallel_size']} "
                    f"PORT={port} MUSE_REPLICA_INDEX={replica} START={start_script} "
                    f"ENGINE_CONTAINER={muse_inference_container}-{replica} "
                    f"bash {remote_repo}/scripts/h200/engine_watchdog.sh "
                    f">> /root/engine_watchdog-{replica}.log 2>&1 </dev/null &"
                )
            start_code, start_output = _ssh(h, " ".join(watchdogs) + " disown")
        else:
            start_code, start_output = _ssh(
                h,
                f"nohup setsid bash {remote_repo}/scripts/h200/engine_watchdog.sh >> /root/engine_watchdog.log 2>&1 </dev/null & disown",
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
        repo_root / "scripts" / "china" / "cluster.sh",
        repo_root / "scripts" / "h200" / "start_orchestrator.sh",
        repo_root / "scripts" / "h200" / "start_inference.sh",
        repo_root / "scripts" / "h200" / "start_muse_inference.sh",
    )
    return sorted(set().union(*(_mounts_in(script) for script in scripts)))


def trainer_mount_preflight(nodes: list[str] | None = None) -> dict[str, str]:
    """Require every configured trainer bind source to match this worktree."""
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    manifest = sorted(_mounts_in(repo_root / "scripts" / "china" / "cluster.sh"))
    missing_local = [path for path in manifest if not (repo_root / path).is_file()]
    if missing_local:
        raise Refused(
            "trainer mount source is missing locally: " + ", ".join(missing_local)
        )

    expected = {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in manifest
    }
    checks = []
    for path in manifest:
        remote = f"{remote_repo}/{path}"
        checks.append(
            f"if test -f {shlex.quote(remote)}; then "
            f"printf '%s\\t' {shlex.quote(path)}; "
            f"sha256sum {shlex.quote(remote)} | cut -d' ' -f1; "
            f"else printf '%s\\tMISSING\\n' {shlex.quote(path)}; fi"
        )
    command = "; ".join(checks)
    results = _fan(nodes, lambda host: _ssh(host, command, timeout=180))
    failures: list[str] = []
    for host, (code, output) in results.items():
        if code:
            failures.append(f"{host}: manifest command failed: {output.strip()[:120]}")
            continue
        observed: dict[str, str] = {}
        for line in output.splitlines():
            path, separator, digest = line.partition("\t")
            if separator:
                observed[path] = digest.strip()
        for path, digest in expected.items():
            actual = observed.get(path, "MISSING")
            if actual != digest:
                failures.append(
                    f"{host}: {path} is {actual[:12]}, expected {digest[:12]}"
                )
    if failures:
        raise Refused("trainer mount preflight failed: " + "; ".join(failures))
    print(
        f"trainer mount preflight: {len(manifest)}/{len(manifest)} files match "
        f"on {len(nodes)}/{len(nodes)} hosts"
    )
    return expected


production_parity_report = "/data/glm52-rl/qualifications/glm52-policy-parity/report.json"
production_parity_fixture = "/data/glm52-rl/qualifications/glm52-policy-parity/fixture"
production_parity_profile = "configs/qualification/glm52-policy-parity.toml"
production_batch2048_profile = "configs/qualification/glm52-native-b2048.toml"
production_stability20_profile = "configs/qualification/glm52-native-stability20.toml"


def _remote_json(host: str, path: str) -> dict:
    """Read one JSON file through the configured operations transport."""
    code, output = _ssh(host, f"base64 -w0 {shlex.quote(path)}", timeout=60)
    if code:
        raise Refused(f"required evidence file is unreadable on {host}: {path}")
    try:
        value = json.loads(base64.b64decode(output.strip()).decode())
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise Refused(f"required evidence file is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise Refused(f"required evidence file is not a JSON object: {path}")
    return value


def _canonical_json_sha256(value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_remote_fixture(host: str, manifest: dict) -> list[str]:
    """Check the fixed fixture files before trusting its manifest hash."""
    failures: list[str] = []
    arrays = manifest.get("arrays")
    expected_names = {"token_ids", "positions", "selected_indices", "advantages"}
    if not isinstance(arrays, dict) or set(arrays) != expected_names:
        return ["parity fixture arrays are incomplete or have unexpected names"]
    checks: list[tuple[str, str, str]] = []
    for name in sorted(expected_names):
        entry = arrays.get(name)
        if not isinstance(entry, dict):
            failures.append(f"parity fixture array entry is invalid: {name}")
            continue
        filename = entry.get("file")
        digest = entry.get("sha256")
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            failures.append(f"parity fixture array hash entry is invalid: {name}")
            continue
        checks.append((name, filename, digest))
    if failures:
        return failures
    command = "sha256sum " + " ".join(
        shlex.quote(f"{production_parity_fixture}/{filename}")
        for _name, filename, _digest in checks
    )
    code, output = _ssh(host, command, timeout=60)
    if code:
        return ["parity fixture array files are missing or unreadable"]
    observed = {}
    for line in output.splitlines():
        digest, separator, filename = line.strip().partition("  ")
        if separator:
            observed[PurePosixPath(filename).name] = digest
    for _name, filename, expected in checks:
        if observed.get(filename) != expected:
            failures.append(f"parity fixture checksum differs for {filename}")
    return failures


def _validate_live_parity_report(
    report: dict,
    fixture_manifest: dict,
    profile: dict,
    *,
    require_passed: bool = True,
) -> list[str]:
    """Return all parity evidence failures; never accept missing fields."""
    failures: list[str] = []
    if report.get("schema_version") != 1:
        failures.append("parity report schema_version is not 1")
    if report.get("kind") != "glm52_policy_parity_report":
        failures.append("parity report kind is invalid")
    if not isinstance(report.get("passed"), bool):
        failures.append("parity report passed field is invalid")
    if require_passed and report.get("passed") is not True:
        failures.append("parity report passed is not true")
    if fixture_manifest.get("schema_version") != 1:
        failures.append("parity fixture schema_version is not 1")
    if fixture_manifest.get("kind") != "glm52_policy_parity_fixture":
        failures.append("parity fixture kind is invalid")
    expected_fixture = _canonical_json_sha256(fixture_manifest)
    expected_profile = hashlib.sha256(
        (repo_root / production_parity_profile).read_bytes()
    ).hexdigest()
    if report.get("fixture_sha256") != expected_fixture:
        failures.append("parity fixture hash does not match its manifest")
    if report.get("profile_sha256") != expected_profile:
        failures.append("parity profile hash does not match the pinned profile")

    hex_fields = {"model_revision": 40, "adapter_sha256": 64}
    for name, length in hex_fields.items():
        value = report.get(name)
        if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            failures.append(f"parity report {name} is not a lowercase SHA-256/revision")
    trainer_image_digest = report.get("trainer_image_digest")
    if not isinstance(trainer_image_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trainer_image_digest
    ):
        failures.append("parity report trainer image digest is missing or invalid")
    expected_image_digest = operations.get("containers", {}).get("trainer_image_digest")
    if expected_image_digest is None:
        expected_image_digest = operations.get("profiles", {}).get(
            "glm52-policy-parity", {}
        ).get("trainer_image_digest")
    trainer_image_digests = report.get("trainer_image_digests")
    if not isinstance(trainer_image_digests, dict) or not trainer_image_digests:
        failures.append("parity report trainer host image digests are missing")
    elif any(
        not isinstance(host, str)
        or not isinstance(digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        for host, digest in trainer_image_digests.items()
    ):
        failures.append("parity report trainer host image digest is invalid")
    elif trainer_image_digest not in trainer_image_digests.values():
        failures.append("parity report trainer image digest is not one of the host digests")
    trainer_modules = report.get("trainer_model_modules")
    if not isinstance(trainer_modules, dict) or not trainer_modules:
        failures.append("parity report trainer model module hashes are missing")
    else:
        locations = set()
        for name, record in trainer_modules.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                failures.append("parity report trainer model module hash record is invalid")
                continue
            location = record.get("location")
            digest = record.get("sha256")
            if location not in {"mounted", "unmounted"} or not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", digest
            ):
                failures.append(f"parity report trainer model module hash is invalid: {name}")
            else:
                locations.add(location)
        if locations != {"mounted", "unmounted"}:
            failures.append("parity report trainer mounted/unmounted module hashes are incomplete")
    host_records = report.get("trainer_host_provenance")
    if not isinstance(host_records, list) or len(host_records) != 8:
        failures.append("parity report must contain eight trainer host provenance records")
    else:
        pp_ranks = {record.get("pp_rank") for record in host_records if isinstance(record, dict)}
        host_names = {
            record.get("host")
            for record in host_records
            if isinstance(record, dict) and isinstance(record.get("host"), str)
        }
        if pp_ranks != set(range(8)):
            failures.append("parity report trainer PP host mapping is incomplete")
        if len(host_names) != len(host_records) or not all(
            re.fullmatch(r"n[0-9]+", host) for host in host_names
        ):
            failures.append("parity report trainer host names are invalid or duplicated")
        if not isinstance(trainer_image_digests, dict) or set(trainer_image_digests) != host_names:
            failures.append("parity report trainer host image digest keys do not match hosts")
        module_records = []
        for record in host_records:
            if not isinstance(record, dict) or record.get("cp_rank") != 0:
                failures.append("parity report trainer host mapping has a nonzero CP rank")
                continue
            if record.get("global_rank") != record.get("pp_rank", -1) * 8:
                failures.append("parity report trainer host global rank mapping is invalid")
            image_digest = record.get("image_digest")
            if not isinstance(image_digest, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", image_digest
            ):
                failures.append("parity report trainer host image digest is invalid")
            elif not isinstance(trainer_image_digests, dict) or trainer_image_digests.get(
                record.get("host")
            ) != image_digest:
                failures.append("parity report trainer host image digest record does not match")
            modules = record.get("modules")
            if not isinstance(modules, dict):
                failures.append("parity report trainer host module hashes are missing")
            else:
                module_records.append(modules)
        if module_records and any(value != module_records[0] for value in module_records[1:]):
            failures.append("parity report trainer host module hashes differ")
        if module_records and trainer_modules != module_records[0]:
            failures.append("parity report trainer module hashes differ from host records")
        if expected_image_digest is not None and any(
            record.get("image_digest") != expected_image_digest
            for record in host_records
            if isinstance(record, dict)
        ):
            failures.append("parity report trainer host image digest differs from operations config")
    selected_tokens = report.get("selected_tokens")
    if isinstance(selected_tokens, bool) or not isinstance(selected_tokens, int) or selected_tokens <= 0:
        failures.append("parity report selected_tokens is invalid")

    numeric_limits = (
        "unmasked_mismatch_kl",
        "masked_mismatch_kl",
        "masked_fraction",
        "trainer_router_weight_sum_max_error",
        "h200_router_weight_sum_max_error",
    )
    thresholds = {
        "unmasked_mismatch_kl": profile.get("max_unmasked_mismatch_kl"),
        "masked_mismatch_kl": profile.get("max_masked_mismatch_kl"),
        "masked_fraction": profile.get("max_masked_fraction"),
        "trainer_router_weight_sum_max_error": profile.get("max_router_weight_sum_error"),
        "h200_router_weight_sum_max_error": profile.get("max_router_weight_sum_error"),
    }
    for name in numeric_limits:
        value = report.get(name)
        limit = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"parity report {name} is missing or non-finite")
        elif not isinstance(limit, (int, float)) or (require_passed and value > limit):
            failures.append(f"parity report {name} exceeds its pinned limit")
    report_thresholds = report.get("thresholds")
    if not isinstance(report_thresholds, dict):
        failures.append("parity report thresholds are missing")
    else:
        for name, expected in (
            ("max_unmasked_mismatch_kl", profile.get("max_unmasked_mismatch_kl")),
            ("max_masked_mismatch_kl", profile.get("max_masked_mismatch_kl")),
            ("max_masked_fraction", profile.get("max_masked_fraction")),
            ("max_router_weight_sum_error", profile.get("max_router_weight_sum_error")),
        ):
            if report_thresholds.get(name) != expected:
                failures.append(f"parity report threshold {name} differs from the pinned profile")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or (
        require_passed and not all(value is True for value in checks.values())
    ):
        failures.append("parity report checks are not all true")
    provisional = report.get("provisional_checks")
    if not isinstance(provisional, dict) or (
        require_passed and provisional.get("router_weight_delta") is not True
    ):
        failures.append("parity report provisional router check is not true")
    if report.get("router_weight_delta_status") != profile.get("router_weight_delta_status"):
        failures.append("parity report router delta status differs from the pinned profile")
    return failures


def _qualification_metric_path(config_name: str) -> str:
    with (repo_root / config_name).open("rb") as stream:
        values = tomllib.load(stream)
    output_dir = values.get("output_dir")
    filename = values.get("file_monitor", {}).get("filename")
    if not isinstance(output_dir, str) or not isinstance(filename, str) or not filename:
        raise Refused(f"qualification profile has no metric file: {config_name}")
    return f"{output_dir.rstrip('/')}/{filename.lstrip('/')}"


def _successful_metric_steps(host: str, path: str) -> list[int]:
    code, output = _ssh(host, f"base64 -w0 {shlex.quote(path)}", timeout=60)
    if code:
        raise Refused(f"qualification result is unreadable on {host}: {path}")
    try:
        raw = base64.b64decode(output.strip()).decode()
    except (ValueError, UnicodeError) as error:
        raise Refused(f"qualification result is not valid UTF-8: {path}") from error
    steps: set[int] = set()
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = record.get("step") if isinstance(record, dict) else None
        duration = record.get("time/step") if isinstance(record, dict) else None
        if (
            isinstance(step, int)
            and not isinstance(step, bool)
            and step > 0
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and duration > 0
        ):
            steps.add(step)
    return sorted(steps)


def _check_image_prime08(image: str, group: str) -> list[str]:
    command = (
        f"sudo docker run --rm --network none --entrypoint /app/.venv/bin/python "
        f"{shlex.quote(image)} -c "
        "'import importlib.metadata; print(importlib.metadata.version(\"prime-rl\"))'"
    )
    results = run_on_hosts(command, group=group, timeout=180)
    failures: list[str] = []
    for host, (code, output) in results.items():
        version = output.strip().splitlines()[-1] if output.strip() else ""
        if code or not re.fullmatch(r"0\.8(?:\.[0-9]+)?", version):
            failures.append(f"{host}: {image} does not report Prime-RL 0.8")
    return failures


def production_start_preflight() -> dict[str, object]:
    """Refuse a GLM production start until all pinned evidence is complete."""
    failures: list[str] = []
    inference_host = inference_hosts[0]
    try:
        report = _remote_json(inference_host, production_parity_report)
        fixture_manifest = _remote_json(
            inference_host, f"{production_parity_fixture}/manifest.json"
        )
        with (repo_root / production_parity_profile).open("rb") as stream:
            parity_profile = tomllib.load(stream)
        failures.extend(_validate_remote_fixture(inference_host, fixture_manifest))
        failures.extend(_validate_live_parity_report(report, fixture_manifest, parity_profile))
    except (OSError, Refused, tomllib.TOMLDecodeError) as error:
        failures.append(str(error))

    try:
        batch_steps = _successful_metric_steps(
            trainer_head, _qualification_metric_path(production_batch2048_profile)
        )
        if not set(range(1, 3)).issubset(batch_steps):
            failures.append(f"batch2048 qualification has steps {batch_steps}, needs 1-2")
    except (OSError, Refused, tomllib.TOMLDecodeError) as error:
        failures.append(str(error))
    try:
        stability_steps = _successful_metric_steps(
            trainer_head, _qualification_metric_path(production_stability20_profile)
        )
        if not set(range(1, 21)).issubset(stability_steps):
            failures.append(
                f"Stability20 qualification has steps {stability_steps}, needs 1-20"
            )
    except (OSError, Refused, tomllib.TOMLDecodeError) as error:
        failures.append(str(error))

    try:
        if len(trainer_mount_preflight()) != 45:
            failures.append("trainer mount manifest is not exactly 45 files")
    except (OSError, Refused) as error:
        failures.append(str(error))
    failures.extend(_check_image_prime08(inference_image, "inference"))
    failures.extend(_check_image_prime08(trainer_image, "trainer"))
    if failures:
        raise Refused("GLM production start preflight failed: " + "; ".join(failures))
    result = {
        "parity_report": production_parity_report,
        "batch2048_steps": [1, 2],
        "stability20_steps": list(range(1, 21)),
        "trainer_mounts": 45,
        "inference_image": inference_image,
        "trainer_image": trainer_image,
        "prime_rl": "0.8",
    }
    print("GLM production start preflight passed: " + json.dumps(result, sort_keys=True))
    return result


def _mounted_on(path: str, host: str) -> bool:
    if host in inference_hosts:
        scripts = (
            repo_root / "scripts" / "h200" / "start_orchestrator.sh",
            repo_root / "scripts" / "h200" / "start_inference.sh",
            repo_root / "scripts" / "h200" / "start_muse_inference.sh",
        )
    else:
        scripts = (repo_root / "scripts" / "china" / "cluster.sh",)
    return any(path in _mounts_in(script) for script in scripts)


def deploy(path: str, nodes: list[str] | None = None) -> bool:
    """Push one repository file to its hosts and verify the checksum."""
    if nodes is None:
        if path.startswith("source/"):
            nodes = [
                host
                for host in dict.fromkeys([*trainer_hosts, *inference_hosts])
                if _mounted_on(path, host)
            ]
        elif path.startswith("environments/"):
            nodes = [orchestrator_host]
        elif path.startswith(("scripts/h200/", "configs/runtime/")):
            nodes = inference_hosts
        elif path.startswith("scripts/china/"):
            nodes = trainer_hosts
        elif path.endswith(("/trainer.toml", "muse-trainer.toml", "muse-parity.toml")):
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
    if path.startswith("source/"):
        unmounted = [host for host in nodes if not _mounted_on(path, host)]
        if unmounted:
            raise Refused(
                f"{path} is not mounted by the launcher for {unmounted}, so the "
                "container would ignore it. Add a -v line to the applicable "
                "launcher first, or pick a mounted path. Call mounted_files() "
                "to list the union."
            )
    want = hashlib.sha256(local.read_bytes()).hexdigest()
    remote = f"{remote_repo}/{path}"
    remote_mode = "0755" if local.stat().st_mode & 0o111 else "0644"

    def one(h):
        staging = f"/tmp/training-ops-{want}-{Path(path).name}"
        p = subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", str(local), f"{h}:{staging}"],
            capture_output=True,
            text=True,
        )
        if p.returncode:
            detail = p.stderr.strip().replace("\n", " ")
            return f"scp failed: {detail[:160]}"
        rc, out = _ssh(
            h,
            f"if test -d {shlex.quote(remote)}; then "
            f"test -z \"$(find {shlex.quote(remote)} -mindepth 1 -maxdepth 1 -print -quit)\" "
            f"|| {{ echo 'destination directory is not empty'; exit 44; }}; "
            f"sudo rmdir {shlex.quote(remote)}; fi; "
            f"sudo install -D -m {remote_mode} {shlex.quote(staging)} "
            f"{shlex.quote(remote)} && rm -f {shlex.quote(staging)}",
        )
        if rc:
            return f"install failed: {out.strip()[:80]}"
        rc, got = _ssh(
            h,
            f"test -f {shlex.quote(remote)} && "
            f"sha256sum {shlex.quote(remote)} | cut -d' ' -f1",
        )
        if rc:
            return f"verify failed: {got.strip()[:80]}"
        return "ok" if got.strip() == want else f"MISMATCH {got.strip()[:12]}"

    res = _fan(nodes, one)
    bad = [h for h, v in res.items() if v != "ok"]
    print(
        f"{path} -> {len(nodes) - len(bad)}/{len(nodes)} nodes"
        + (f", FAILED {bad}" if bad else "")
    )
    for host in bad:
        print(f"{host}: {res[host]}")
    if not bad:
        if path.startswith(("source/", "environments/")) or path.endswith("cluster.sh"):
            print(
                "The running container still holds the old file. Recreate that container for this to take effect."
            )
        else:
            print(
                "This runs on the host. Restart the process only if it is a long-running loop."
            )
    return not bad


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


def start(
    run: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
    supervisors: bool = False,
    resume_adapter: str | None = None,
    resume_step: int | None = None,
    profile: str = "current",
) -> None:
    """Start a run from one configured profile."""
    if mode not in ("cold", "restart"):
        raise Refused('mode must be "cold" or "restart"')
    if supervisors:
        raise Refused(
            "the shared supervisor scripts are not isolated from the live deployment"
        )
    if (resume_adapter is None) != (resume_step is None):
        raise Refused("resume_adapter and resume_step must be set together")
    selected = _profile(profile)
    if "orchestrator" not in selected:
        raise Refused(f"profile {profile!r} is trainer-only; use qualify()")
    with (repo_root / selected["trainer"]).open("rb") as stream:
        trainer_values = tomllib.load(stream)
    with (repo_root / selected["orchestrator"]).open("rb") as stream:
        orchestrator_values = tomllib.load(stream)
    with (repo_root / selected["environment"]).open("rb") as stream:
        environment_values = tomllib.load(stream)
    expected_run_dir = f"{run_root}/{run}"
    expected_orchestrator_dir = f"{expected_run_dir}/run_{run}"
    configured_paths = {
        "trainer": trainer_values.get("output_dir"),
        "orchestrator": orchestrator_values.get("output_dir"),
        "environment": environment_values.get("output_dir"),
    }
    expected_paths = {
        "trainer": expected_run_dir,
        "orchestrator": expected_orchestrator_dir,
        "environment": f"{expected_orchestrator_dir}/env",
    }
    if configured_paths != expected_paths:
        raise Refused(
            f"profile {profile!r} is configured for {configured_paths}; "
            f"the requested run {run!r} requires {expected_paths}"
        )
    if selected["kind"] == "glm52":
        production_start_preflight()
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
    _assert_live_services_are_down(nodes, include_inference=True)
    required_files = [
        ("scripts/china/cluster.sh", nodes),
        (selected["trainer"], nodes),
        (selected["orchestrator"], [orchestrator_host]),
        (selected["environment"], [orchestrator_host]),
        ("scripts/h200/start_orchestrator.sh", [orchestrator_host]),
        ("scripts/h200/trainer_failure_watchdog.sh", [orchestrator_host]),
    ]
    required_files.extend(
        (path, nodes)
        for path in sorted(
            _mounts_in(repo_root / "scripts" / "china" / "cluster.sh")
        )
    )
    required_files.extend(
        (path, [orchestrator_host])
        for path in sorted(
            _mounts_in(repo_root / "scripts" / "h200" / "start_orchestrator.sh")
        )
    )
    for path, destinations in required_files:
        if not deploy(path, nodes=destinations):
            raise RuntimeError(f"failed to deploy {path}")
    trainer_mount_preflight(nodes)
    name = f"run_{run}"
    trainer_config = f"{remote_repo}/{selected['trainer']}"
    edge_config = f"{remote_repo}/{selected['orchestrator']}"
    environment_config = f"{remote_repo}/{selected['environment']}"
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
    isolated_env = (
        f"prime_deployment={shlex.quote(remote_repo)} "
        f"prime_image={shlex.quote(trainer_image)} "
        f"prime_trainer_container={shlex.quote(trainer_container)} "
    )
    if muse_run:
        env = (
            f"{isolated_env}trainer_nodes='{trainer_nodes}' RUN={run} RUN_NAME={name} "
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
            f"{isolated_env}trainer_nodes='{trainer_nodes}' RUN={run} RUN_NAME={name} "
            f"TRAINER_CONFIG={trainer_config} PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 "
            "PRIME_INT4_COMBINE_ROUNDS=8 PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=1"
        )

    def launch_trainer():
        print(f"starting trainer on {len(nodes)} nodes")
        results = _fan(
            nodes,
            lambda h: _ssh(
                h, f"{env} bash {trainer_launcher} trainer >/dev/null 2>&1; echo done", 300
            ),
        )
        failed = [host for host, (code, _output) in results.items() if code]
        if failed:
            raise RuntimeError(f"trainer start failed on {failed}")

    def launch_orch():
        print(f"starting edge orchestrator on {orchestrator_host}")
        rc, out = _ssh(
            orchestrator_host,
            f"prime_deployment={shlex.quote(remote_repo)} "
            f"prime_image={shlex.quote(inference_image)} "
            f"prime_model_volume={shlex.quote(glm_model_volume)} "
            f"prime_trainer_container={shlex.quote(trainer_container)} "
            f"prime_orchestrator_container={shlex.quote(orchestrator_container)} "
            f"prime_env_container={shlex.quote(environment_container)} "
            f"RUN={run} RUN_NAME={name} ORCH_CONFIG={edge_config} "
            f"ENV_CONFIG={environment_config} "
            f"bash {orchestrator_launcher}",
            300,
        )
        if rc:
            raise RuntimeError(f"edge orchestrator start failed on {orchestrator_host}: {out}")

    print(f"stopping any previous containers for {run}")
    _remote_ok(
        orchestrator_host,
        f"sudo docker rm -f {orchestrator_container} {environment_container} "
        ">/dev/null 2>&1; echo done",
        "old edge services stop",
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
            f"sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
        ),
    )
    stop_failures = [host for host, (code, _output) in stop_results.items() if code]
    if stop_failures:
        raise RuntimeError(f"old trainer stop failed on {stop_failures}")
    time.sleep(20)

    if mode == "cold":
        launch_orch()
        launch_trainer()
    else:
        launch_trainer()
        time.sleep(60)
        launch_orch()

    if supervisors:
        _remote_ok(
            trainer_head,
            "pkill -f '[r]un_supervisor.sh' 2>/dev/null; echo done",
            "old run supervisor stop",
        )
        time.sleep(3)
        _remote_ok(
            trainer_head,
            f"nohup setsid env RUN={run} RUN_NAME={name} INTERVAL=120 "
            f"bash {remote_repo}/scripts/china/run_supervisor.sh >> /data/glm52-rl/run_supervisor.log 2>&1 </dev/null & disown",
            "run supervisor start",
        )
        time.sleep(3)
        n = _remote_ok(
            trainer_head,
            "pgrep -f 'bash .*[r]un_supervisor.sh' | wc -l",
            "run supervisor check",
        ).strip()
        if not n.isdigit() or int(n) < 1:
            raise RuntimeError(f"run supervisor did not start on {trainer_head}")
        print(f"run supervisor: {n} process(es) for {run}")
        _repair_supervisors()
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
    _assert_live_services_are_down(nodes, include_inference=False)

    selected = _profile(name)
    if "orchestrator" in selected:
        raise Refused(f"profile {name!r} is a full run; use start()")
    relative_config = selected["trainer"]
    local_config = repo_root / relative_config
    if not local_config.is_file():
        raise Refused(f"missing qualification config {relative_config}")
    with local_config.open("rb") as stream:
        config = tomllib.load(stream)
    if relative_config == "configs/qualification/glm52-policy-parity-trainer.toml":
        _diagnostic_capture_preflight(local_config, "trainer")
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

    trainer_mounts = sorted(
        _mounts_in(repo_root / "scripts" / "china" / "cluster.sh")
    )
    for path in ("scripts/china/cluster.sh", *trainer_mounts, relative_config):
        if not deploy(path, nodes=nodes):
            raise RuntimeError(f"failed to deploy {path}")
    trainer_mount_preflight(nodes)

    trainer_nodes = " ".join(host.removeprefix("n") for host in nodes)
    env = (
        f"prime_deployment={shlex.quote(remote_repo)} "
        f"prime_image={shlex.quote(trainer_image)} "
        f"prime_trainer_container={shlex.quote(trainer_container)} "
        f"trainer_nodes='{trainer_nodes}' RUN={shlex.quote(name)} "
        f"RUN_NAME=run_{shlex.quote(name)} "
        f"TRAINER_CONFIG={remote_repo}/{relative_config} "
        "PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 PRIME_INT4_COMBINE_ROUNDS=1 "
        "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=0"
    )
    if muse:
        env += (
            " TRANSFORMERS_OVERRIDE_DIR=/data/muse-glimmer/transformers"
            " MUSE_PYTHON_OVERRIDE_DIR=/data/muse-glimmer/python-overrides"
        )

    if active:
        _fan(
            nodes,
            lambda host: _ssh(
                host,
                f"sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
            ),
        )
        time.sleep(20)

    results = _fan(
        nodes,
        lambda host: _ssh(
            host,
            f"{env} bash {trainer_launcher} trainer >/dev/null 2>&1; echo done",
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


def _validate_glm_parity_stage(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("GLM parity stage is not a regular directory")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("GLM parity stage manifest is not a regular file")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read GLM parity stage manifest: {error}") from error
    if manifest.get("schema_version") != 1 or manifest.get("kind") != (
        "glm52_policy_parity_trainer_stage"
    ):
        raise RuntimeError("GLM parity stage manifest kind is invalid")
    profile_path = repo_root / "configs/qualification/glm52-policy-parity.toml"
    if manifest.get("profile_sha256") != hashlib.sha256(
        profile_path.read_bytes()
    ).hexdigest():
        raise RuntimeError("GLM parity stage profile checksum differs")
    if manifest.get("route_output_scale") != 2.5:
        raise RuntimeError("GLM parity stage route output scale differs")
    for name in ("fixture_sha256", "h200_capture_manifest_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"GLM parity stage {name} is invalid")
    if manifest.get("router_weight_provenance") != (
        "selected_weights_before_route_output_scale"
    ):
        raise RuntimeError("GLM parity stage router weight provenance differs")
    expected = {
        "trainer-reference.json",
        "trainer-router-trace/session.json",
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise RuntimeError("GLM parity stage file list differs")
    for relative, wanted in files.items():
        artifact = path / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"GLM parity stage file is missing: {relative}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != wanted:
            raise RuntimeError(f"GLM parity stage checksum differs: {relative}")
    actual_files = {
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_file()
    }
    if actual_files != {*expected, "manifest.json"}:
        raise RuntimeError("GLM parity stage contains unexpected files")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _deploy_glm_parity_staging_sources() -> None:
    for relative in (
        "configs/operations.toml",
        "scripts/glm52_policy_parity.py",
    ):
        if not deploy(relative, nodes=[orchestrator_host]):
            raise RuntimeError(f"failed to deploy GLM parity source: {relative}")
        local_sha256 = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        remote = f"{remote_repo}/{relative}"
        code, remote_sha256 = _ssh(
            orchestrator_host,
            f"test -f {shlex.quote(remote)} && "
            f"sha256sum {shlex.quote(remote)} | cut -d' ' -f1",
        )
        if code:
            raise RuntimeError(f"GLM parity remote source is missing: {relative}")
        if remote_sha256.strip() != local_sha256:
            raise RuntimeError(f"GLM parity remote source checksum differs: {relative}")


def _prepare_glm_parity_stage_directory() -> str:
    stage = glm_parity_paths["trainer_stage"]
    root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    stage_path = PurePosixPath(stage)
    if stage_path.parent != PurePosixPath(root) or stage_path.name != (
        "trainer-stage-final"
    ):
        raise Refused("GLM parity stage path is not the exact pinned directory")
    quoted_stage = shlex.quote(stage)
    quoted_root = shlex.quote(root)
    command = f"""
set -euo pipefail
stage={quoted_stage}
root={quoted_root}
test -d "$root"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
if test -L "$stage"; then
  echo "stage path is a symlink" >&2
  exit 41
fi
if ! test -e "$stage"; then
  sudo install -d -o 1000 -g 1000 -m 0750 "$stage"
  echo created
  exit 0
fi
test -d "$stage" || {{ echo "stage path is not a directory" >&2; exit 42; }}
identity=$(stat -c '%u:%g:%a' "$stage")
test "$identity" = "1000:1000:750" || {{
  echo "stage owner or mode differs: $identity" >&2
  exit 43
}}
if test -n "$(find "$stage" -mindepth 1 -maxdepth 1 -print -quit)" && \
   ! test -f "$stage/manifest.json"; then
  echo "stage has content but no manifest" >&2
  exit 44
fi
echo existing
"""
    code, output = _ssh(orchestrator_host, command)
    if code:
        raise RuntimeError(f"GLM parity stage directory refused: {output.strip()}")
    state = output.strip()
    if state not in {"created", "existing"}:
        raise RuntimeError("GLM parity stage directory returned an invalid state")
    return state


def _prepare_glm_parity_capture_directory() -> str:
    """Prepare the exact H200 capture directory for the capture container."""
    capture = glm_parity_paths["h200_capture"]
    root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    capture_path = PurePosixPath(capture)
    if capture_path.parent != PurePosixPath(root) or capture_path.name != (
        "h200-capture-final"
    ):
        raise Refused("GLM parity capture path is not the exact pinned directory")
    quoted_capture = shlex.quote(capture)
    quoted_root = shlex.quote(root)
    command = f"""
set -euo pipefail
capture={quoted_capture}
root={quoted_root}
test -d "$root"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
if test -L "$capture"; then
  echo "capture path is a symlink" >&2
  exit 41
fi
if ! test -e "$capture"; then
  sudo install -d -o 1000 -g 1000 -m 0750 "$capture"
  echo created
  exit 0
fi
test -d "$capture" || {{ echo "capture path is not a directory" >&2; exit 42; }}
identity=$(stat -c '%u:%g:%a' "$capture")
test "$identity" = "1000:1000:750" || {{
  echo "capture owner or mode differs: $identity" >&2
  exit 43
}}
if test -n "$(find "$capture" -mindepth 1 -maxdepth 1 -print -quit)" && \
   ! test -f "$capture/manifest.json"; then
  echo "capture has content but no manifest" >&2
  exit 44
fi
echo existing
"""
    code, output = _ssh(orchestrator_host, command)
    if code:
        raise RuntimeError(f"GLM parity capture directory refused: {output.strip()}")
    state = output.strip()
    if state not in {"created", "existing"}:
        raise RuntimeError("GLM parity capture directory returned an invalid state")
    return state


def stage_glm52_policy_parity(nodes: list[str] | None = None) -> None:
    """Build the exact H200 parity reference and copy it to trainer nodes."""
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    fixture = glm_parity_paths["fixture"]
    capture = glm_parity_paths["h200_capture"]
    stage = glm_parity_paths["trainer_stage"]
    qualification_root = "/data/glm52-rl/qualifications/glm52-policy-parity/"
    if not all(path.startswith(qualification_root) for path in (fixture, capture, stage)):
        raise Refused("GLM parity paths must stay under the qualification directory")
    _prepare_glm_parity_capture_directory()
    _deploy_glm_parity_staging_sources()
    _prepare_glm_parity_stage_directory()
    quoted_stage = shlex.quote(stage)
    code, output = _ssh(orchestrator_host, f"test -f {quoted_stage}/manifest.json")
    if code:
        command = (
            f"sudo docker run --rm -v /data:/data "
            f"-v {shlex.quote(remote_repo)}:/workspace:ro -w /workspace "
            f"--entrypoint /app/.venv/bin/python {shlex.quote(inference_image)} "
            "scripts/glm52_policy_parity.py stage-trainer "
            f"--fixture {shlex.quote(fixture)} "
            f"--h200-capture {shlex.quote(capture)} {quoted_stage}"
        )
        _remote_ok(
            orchestrator_host,
            command,
            "GLM parity trainer staging",
            timeout=1800,
        )
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-stage-") as temporary:
        local_parent = Path(temporary)
        copied = subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-r",
                f"{orchestrator_host}:{stage}",
                str(local_parent),
            ],
            capture_output=True,
            text=True,
        )
        if copied.returncode:
            raise RuntimeError(
                f"cannot copy GLM parity stage: {copied.stderr.strip()[:160]}"
            )
        local_stage = local_parent / Path(stage).name
        manifest_sha256 = _validate_glm_parity_stage(local_stage)
        live_capture_sha256 = _remote_ok(
            orchestrator_host,
            f"sha256sum {shlex.quote(capture + '/manifest.json')} | cut -d' ' -f1",
            "GLM parity H200 capture checksum",
        ).strip()
        stage_manifest = json.loads((local_stage / "manifest.json").read_text())
        if stage_manifest.get("h200_capture_manifest_sha256") != (
            live_capture_sha256
        ):
            raise RuntimeError("GLM parity stage does not match the live H200 capture")

        def install(host: str) -> str:
            temporary_remote = f"/tmp/glm52-parity-stage-{manifest_sha256}"
            copied_to_host = subprocess.run(
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    "-r",
                    str(local_stage),
                    f"{host}:{temporary_remote}",
                ],
                capture_output=True,
                text=True,
            )
            if copied_to_host.returncode:
                return f"scp failed: {copied_to_host.stderr.strip()[:120]}"
            command = (
                f"if test -e {quoted_stage}; then "
                f"test -f {quoted_stage}/manifest.json && "
                f"test \"$(sha256sum {quoted_stage}/manifest.json | cut -d' ' -f1)\" "
                f"= {shlex.quote(manifest_sha256)} || exit 44; "
                f"rm -rf {shlex.quote(temporary_remote)}; "
                "else "
                f"sudo install -d -m 0755 {shlex.quote(str(PurePosixPath(stage).parent))} && "
                f"sudo mv {shlex.quote(temporary_remote)} {quoted_stage}; fi"
            )
            rc, detail = _ssh(host, command)
            return "ok" if rc == 0 else f"install failed: {detail.strip()[:120]}"

        results = _fan(nodes, install)
    failed = [host for host, result in results.items() if result != "ok"]
    if failed:
        raise RuntimeError(f"GLM parity stage failed on {failed}")
    print(f"GLM parity trainer stage verified on {len(nodes)} nodes")


def _validate_parity_remote_source(remote: str) -> None:
    if remote == parity_token_root:
        return
    if re.fullmatch(
        rf"{re.escape(parity_trace_root)}/[A-Za-z0-9_-]+", remote
    ):
        return
    raise Refused("parity evidence source is not an approved exact directory")


def _parity_python() -> str:
    configured = operations.get("paths", {}).get("parity_python")
    candidates = [configured, "/usr/bin/python3", sys.executable]
    checked: set[str] = set()
    failures: list[str] = []
    dependency_check = (
        "import numpy, tomllib, urllib.request; "
        "assert numpy.__version__"
    )
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            failures.append(f"{candidate}: interpreter path is not absolute")
            continue
        try:
            resolved = str(path.resolve(strict=True))
        except OSError:
            failures.append(f"{candidate}: interpreter does not exist")
            continue
        if resolved in checked:
            continue
        checked.add(resolved)
        if not os.access(resolved, os.X_OK):
            failures.append(f"{resolved}: interpreter is not executable")
            continue
        try:
            result = subprocess.run(
                [resolved, "-c", dependency_check],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{resolved}: dependency check failed: {error}")
            continue
        if result.returncode == 0:
            return resolved
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:160]
        failures.append(f"{resolved}: missing parity dependency: {detail}")
    raise RuntimeError(
        "no usable parity Python interpreter was found before evidence transfer: "
        + "; ".join(failures)
    )


def _run_parity_command(
    python: str,
    script: Path,
    profile: Path,
    action: str,
    arguments: list[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python, str(script), "--profile", str(profile), action, *arguments],
        check=check,
    )


def _validate_parity_extraction(destination: Path) -> None:
    root = destination.resolve()
    for current, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(current) / name
            if path.is_symlink():
                raise RuntimeError(f"parity evidence contains a symlink: {path}")
            if not path.resolve().is_relative_to(root):
                raise RuntimeError(f"parity evidence escapes its destination: {path}")


def _copy_parity_tree_stream(host: str, remote: str, local: Path) -> None:
    """Extract one remote tar stream into a controlled local directory."""
    _check_parity_host(host)
    _validate_parity_remote_source(remote)
    if local.exists() and (local.is_symlink() or not local.is_dir()):
        raise RuntimeError(f"parity evidence destination is not a directory: {local}")
    local.mkdir(parents=True, exist_ok=True)
    remote_process = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "tar",
            "-C",
            remote,
            "-cf",
            "-",
            ".",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert remote_process.stdout is not None
    local_process = subprocess.Popen(
        [
            "tar",
            "--extract",
            "--file=-",
            "--directory",
            str(local),
            "--no-same-owner",
            "--no-same-permissions",
            "--no-overwrite-dir",
        ],
        stdin=remote_process.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    remote_process.stdout.close()
    timeout = 1800
    try:
        _, local_error = local_process.communicate(timeout=timeout)
        _, remote_error = remote_process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        for process in (local_process, remote_process):
            process.kill()
        for process in (local_process, remote_process):
            process.communicate()
        raise RuntimeError(f"parity evidence tar transfer timed out after {timeout}s") from error
    if remote_process.returncode:
        detail = (remote_error or b"").decode(errors="replace").strip()[:200]
        raise RuntimeError(f"parity evidence remote tar failed: {detail}")
    if local_process.returncode:
        detail = (local_error or b"").decode(errors="replace").strip()[:200]
        raise RuntimeError(f"parity evidence local tar failed: {detail}")
    _validate_parity_extraction(local)


def _copy_parity_host_evidence(
    nodes: list[str],
    name: str,
    root: Path,
    trace_root: str,
    token_root: str,
    token_host: str,
    copy_tree=None,
) -> list[tuple[str, Path, Path | None]]:
    """Copy each trainer host's trace and token evidence in parallel."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("parity trace run name contains unsafe characters")
    if trace_root != parity_trace_root or token_root != parity_token_root:
        raise Refused("parity evidence roots are not the configured exact directories")
    if len(set(nodes)) != len(nodes):
        raise Refused("parity trainer host list contains duplicates")
    for host in nodes:
        _check_parity_host(host)
    copy_tree = copy_tree or _copy_parity_tree_stream

    def copy_host(host: str) -> tuple[str, Path, Path | None]:
        trace = root / "sources" / host
        trace.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(host, f"{trace_root}/{name}", trace)
        if not (trace / "session.json").is_file():
            raise RuntimeError(f"parity trace session is missing from {host}")
        if not list(trace.glob("trainer-runtime-pp-*-host-*.json")):
            raise RuntimeError(f"parity trace provenance is missing from {host}")
        tokens = None
        if host == token_host:
            tokens = root / "tokens" / host
            tokens.parent.mkdir(parents=True, exist_ok=True)
            copy_tree(host, token_root, tokens)
            if not any(tokens.rglob("rank_*.jsonl")):
                raise RuntimeError(f"parity token exports are missing from {host}")
        return host, trace, tokens

    workers = min(8, len(nodes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(copy_host, nodes))


def collect_glm52_policy_parity(name: str, nodes: list[str] | None = None) -> dict:
    """Collect every trainer host trace and write the canonical parity report."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("parity trace run name contains unsafe characters")
    parity_python = _parity_python()
    nodes = [_check_parity_host(host) for host in (nodes or trainer_hosts)]
    profile = repo_root / production_parity_profile
    trainer_config = repo_root / "configs/qualification/glm52-policy-parity-trainer.toml"
    with trainer_config.open("rb") as stream:
        trainer_values = tomllib.load(stream)
    pp_size = trainer_values.get("model", {}).get("pp")
    if isinstance(pp_size, bool) or not isinstance(pp_size, int) or pp_size < 1:
        raise Refused("parity trainer TOML must define a positive model.pp")
    if len(nodes) != pp_size:
        raise Refused(
            f"parity collection requires {pp_size} configured PP hosts, found {len(nodes)}"
        )
    trace_root = parity_trace_root
    token_root = parity_token_root
    remote_root = "/data/glm52-rl/qualifications/glm52-policy-parity"
    script = repo_root / "scripts/glm52_policy_parity.py"

    def copy_tree(host: str, remote: str, local: Path) -> None:
        _check_host(host)
        result = subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-r", f"{host}:{remote}", str(local)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"parity evidence copy failed from {host}:{remote}: {result.stderr.strip()[:200]}")

    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-collect-") as temporary:
        root = Path(temporary)
        copied_hosts = _copy_parity_host_evidence(
            nodes, name, root, trace_root, token_root, nodes[pp_size - 1],
            _copy_parity_tree_stream,
        )
        sources = [trace for _, trace, _ in copied_hosts]
        fixture = root / "fixture"
        capture = root / "h200-capture-final"
        copy_tree(orchestrator_host, f"{remote_root}/fixture", fixture)
        copy_tree(orchestrator_host, f"{remote_root}/h200-capture-final", capture)
        merged = root / "trainer-router-trace"
        _run_parity_command(
            parity_python,
            script,
            profile,
            "collect-trainer",
            [
                "--trainer-config",
                str(trainer_config),
                *sum((["--source", str(source)] for source in sources), []),
                str(merged),
            ],
            check=True,
        )
        trainer = root / "trainer"
        model_revision = json.loads((capture / "manifest.json").read_text())["model_revision"]
        adapter_sha256 = json.loads((capture / "manifest.json").read_text())["adapter_sha256"]
        _run_parity_command(
            parity_python,
            script,
            profile,
            "capture-trainer",
            [
                "--fixture", str(fixture),
                "--token-exports", str(root / "tokens"),
                "--router-trace", str(merged),
                "--model-revision", model_revision,
                "--adapter-sha256", adapter_sha256,
                "--image", trainer_image,
                str(trainer),
            ],
            check=True,
        )
        report = root / "report.json"
        _run_parity_command(
            parity_python,
            script,
            profile,
            "compare",
            [
                "--fixture", str(fixture),
                "--trainer", str(trainer),
                "--h200", str(capture),
                "--operations", str(repo_root / "configs/operations.toml"),
                "--output", str(report),
            ],
            check=False,
        )
        if not report.is_file():
            raise RuntimeError("parity comparison did not produce a report")
        report_value = json.loads(report.read_text())
        fixture_manifest = json.loads((fixture / "manifest.json").read_text())
        with profile.open("rb") as stream:
            profile_value = tomllib.load(stream)
        validation_failures = _validate_live_parity_report(
            report_value, fixture_manifest, profile_value, require_passed=False
        )
        if validation_failures:
            raise RuntimeError("parity report validation failed: " + "; ".join(validation_failures))
        local_digest = hashlib.sha256(report.read_bytes()).hexdigest()
        temporary_report = f"{remote_root}/.report-{name}-{local_digest}.json"
        copy_result = subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", str(report),
             f"{orchestrator_host}:{temporary_report}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if copy_result.returncode:
            raise RuntimeError(f"canonical parity report upload failed: {copy_result.stderr.strip()[:200]}")
        remote_check = _ssh(
            orchestrator_host,
            f"test \"$(sha256sum {shlex.quote(temporary_report)} | cut -d' ' -f1)\" = {local_digest} "
            f"&& mv -f {shlex.quote(temporary_report)} {remote_root}/report.json "
            f"&& test \"$(sha256sum {remote_root}/report.json | cut -d' ' -f1)\" = {local_digest}",
            timeout=300,
        )
        if remote_check[0]:
            raise RuntimeError(f"canonical parity report checksum or publish failed: {remote_check[1].strip()[:200]}")
        return report_value


def stop_legacy_qualification(expected_output_path: str) -> None:
    """Stop the old trainer name only when every worker is one qualification."""
    expected = PurePosixPath(expected_output_path)
    qualification_root = PurePosixPath("/data/glm52-rl/qualifications")
    if expected.parent == expected or qualification_root not in expected.parents:
        raise Refused(
            "expected_output_path must name one directory under "
            f"{qualification_root}"
        )

    service_hosts = list(dict.fromkeys([trainer_head, orchestrator_host]))
    for host in service_hosts:
        rc, output = _ssh(
            host,
            "sudo docker ps --format '{{.Names}}' | "
            "grep -Ei 'orchestrator|(^|[-_])muse.*trainer|trainer.*muse' || true",
        )
        if rc:
            raise RuntimeError(f"cannot inspect active services on {host}: {output.strip()}")
        if output.strip():
            raise Refused(
                f"active trainer or orchestrator service on {host}: "
                f"{', '.join(output.splitlines())}"
            )

    def identify(host: str) -> tuple[bool, str]:
        rc, output = _ssh(
            host,
            f"sudo docker inspect {shlex.quote(protected_trainer)} "
            "--format '{{json .State.Running}}\n{{json .Config.Image}}\n{{json .Config.Cmd}}'",
        )
        if rc:
            return False, f"container is absent: {output.strip()}"
        lines = output.splitlines()
        if len(lines) != 3:
            return False, "container inspection returned an invalid record"
        try:
            running = json.loads(lines[0])
            image = json.loads(lines[1])
            command = json.loads(lines[2])
        except json.JSONDecodeError as error:
            return False, f"container inspection is not valid JSON: {error}"
        if running is not True:
            return False, "container is not running"
        if image != trainer_image:
            return False, f"image is {image!r}, expected {trainer_image!r}"
        if "@" not in command:
            return False, "trainer command has no configuration path"
        config_index = command.index("@") + 1
        if config_index >= len(command):
            return False, "trainer command has no configuration after @"
        config_path = PurePosixPath(command[config_index])
        repository = PurePosixPath(remote_repo)
        if repository not in config_path.parents:
            return False, f"configuration is outside {repository}: {config_path}"
        rc, configured_output = _ssh(
            host,
            "python3 -c 'import sys,tomllib; "
            "print(tomllib.load(open(sys.argv[1],\"rb\"))[\"output_dir\"])' "
            f"{shlex.quote(str(config_path))}",
        )
        if rc:
            return False, f"cannot read {config_path}: {configured_output.strip()}"
        actual = configured_output.strip()
        if actual != str(expected):
            return False, f"output is {actual!r}, expected {str(expected)!r}"
        return True, str(config_path)

    identified = _fan(trainer_hosts, identify)
    refused = {
        host: detail for host, (matches, detail) in identified.items() if not matches
    }
    if refused:
        detail = "; ".join(f"{host}: {reason}" for host, reason in refused.items())
        raise Refused(f"legacy qualification identity check failed: {detail}")

    results = _fan(
        trainer_hosts,
        lambda host: _ssh(
            host,
            f"sudo docker rm -f {shlex.quote(protected_trainer)} >/dev/null && echo done",
        ),
    )
    failed = [host for host, (code, output) in results.items() if code or output.strip() != "done"]
    if failed:
        raise RuntimeError(f"legacy qualification stop failed on {failed}")
    print(
        f"stopped {protected_trainer} on {len(trainer_hosts)} hosts after all "
        f"workers matched {expected}. No orchestrator was active."
    )


def stop() -> None:
    """Stop only the isolated containers. Live deployment processes are untouched."""
    _remote_ok(
        orchestrator_host,
        f"sudo docker rm -f {orchestrator_container} {environment_container} "
        ">/dev/null 2>&1; echo done",
        "edge services stop",
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
            f"sudo docker rm -f {trainer_container} >/dev/null 2>&1; echo done",
        ),
    )
    failed = [host for host, (code, _output) in results.items() if code]
    if failed:
        raise RuntimeError(f"trainer stop failed on {failed}")
    print(
        "stopped the isolated orchestrator and trainer containers. Inference is still running."
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


def _diagnose_glm52_parity_action(arguments: list[str]):
    keep_engine = False
    names = []
    for argument in arguments:
        if argument == "--keep-engine":
            keep_engine = True
        elif argument.startswith("--"):
            raise Refused(f"unknown GLM parity diagnostic option {argument}")
        else:
            names.append(argument)
    if len(names) != 1:
        raise Refused("diagnose-glm52-parity needs exactly one trace run name")
    return diagnose_glm52_policy_parity(names[0], keep_engine=keep_engine)


def _cleanup_glm52_parity_engine_action(arguments: list[str]):
    if arguments:
        raise Refused("cleanup-glm52-parity-engine takes no arguments")
    return cleanup_glm52_policy_parity_engine()


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
    "stage-glm52-parity": lambda a: stage_glm52_policy_parity(a or None),
    "collect-glm52-parity": lambda a: collect_glm52_policy_parity(a[0] if a else ""),
    "diagnose-glm52-parity": _diagnose_glm52_parity_action,
    "cleanup-glm52-parity-engine": _cleanup_glm52_parity_engine_action,
    "stop-legacy-qualification": lambda a: stop_legacy_qualification(*a),
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
  repair <name>                 repair tunnels, an engine, or supervisors
  deploy <path>                 copy one repository file
  start <run> [mode] [profile]  start one configured profile
  archive <run> [step ...]      upload stable old LoRA adapters
                                add --finished after another run is active
  qualify <profile>             start a trainer-only profile
  stage-glm52-parity [node ...] validate and stage the exact parity reference
  diagnose-glm52-parity <run>  capture H200 parity and replay trainer layer 3
                                add --keep-engine to retain a successful engine
  cleanup-glm52-parity-engine  remove only the exited isolated parity engine
  stop-legacy-qualification <output-path>
                                stop a verified legacy qualification
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
