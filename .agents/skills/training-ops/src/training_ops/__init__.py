"""Read and operate the configured training system."""

from __future__ import annotations

import ast
import base64
import binascii
import datetime as _dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock

import tomllib

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
glm_step20_h200_gate = operations["gate"]["glm52_step20_h200"]
glm_parity_root = "/data/glm52-rl/qualifications/glm52-policy-parity"
glm_parity_stage_candidate_root = f"{glm_parity_root}/.stage-candidates"
glm_parity_stage_docker_timeout = 1800
glm_parity_stage_inspect_timeout = 30
glm_parity_stage_remove_timeout = 120
glm_parity_stage_outer_timeout = (
    glm_parity_stage_docker_timeout
    + 3 * glm_parity_stage_inspect_timeout
    + glm_parity_stage_remove_timeout
    + 120
)
parity_trace_root = "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces"
parity_token_root = "/data/glm52-rl/qualifications/glm52-policy-parity/trainer/token_exports"

trainer_head = hosts["trainer_head"]
trainer_ssh = hosts["trainer_ssh"]
trainer_hosts = list(hosts["trainers"])
trainer_internal_addresses = dict(hosts["trainer_internal_addresses"])
inference_hosts = list(hosts["inference"])
orchestrator_host = hosts["orchestrator"]
metric_hosts = list(hosts["metrics"])
blocked_hosts = set(hosts["blocked"])
if (
    not isinstance(trainer_ssh, str)
    or re.fullmatch(r"[A-Za-z0-9_.-]+", trainer_ssh) is None
    or set(trainer_internal_addresses) != set(trainer_hosts)
    or len(set(trainer_internal_addresses.values())) != len(trainer_hosts)
    or any(
        not isinstance(address, str)
        or re.fullmatch(r"10\.92\.20\.(?:[1-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])", address)
        is None
        for address in trainer_internal_addresses.values()
    )
):
    raise RuntimeError("trainer internal address inventory is invalid")
inference_tunnel_ports = list(
    range(ports["inference_tunnel_first"], ports["inference_tunnel_last"] + 1)
)
judge_port = ports["judge"]
adapter_port = ports["adapter"]
glm_adapter_port = ports["glm_adapter"]
glm_adapter_tunnel_ports = range(
    ports["glm_adapter_tunnel_first"],
    ports["glm_adapter_tunnel_last"] + 1,
)
if len(glm_adapter_tunnel_ports) != 8:
    raise RuntimeError("GLM adapter tunnel inventory is invalid")
step20_tunnel_attempts = 3
step20_tunnel_start_seconds = 45
step20_tunnel_cleanup_seconds = 5
step20_tunnel_launch_stagger_seconds = 0.25
step20_tunnel_probe_seconds = 30
step20_tunnel_outer_timeout_seconds = 270
batch_port = ports["batch"]
h200_parity_backend_port = ports["h200_parity_backend"]
remote_repo = paths["remote_repository"]
glm_adapter_root = PurePosixPath(paths["glm_adapter_root"])
glm_adapter_tunnel_state = paths["glm_adapter_tunnel_state"]
glm_parity_inference_config = (
    f"{remote_repo}/configs/qualification/glm52-policy-parity-h200.toml"
)
glm_parity_vllm_extra = (
    '{"host":"127.0.0.1","kv_cache_dtype":"auto","max_num_seqs":128,'
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
glm_adapter_server_container = containers["glm52_adapter_server"]
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


def _configured_h200_parity_listener_stabilization() -> int:
    value = limits.get("h200_parity_listener_stabilization_seconds")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 60:
        raise Refused(
            "limits.h200_parity_listener_stabilization_seconds must be between 1 and 60"
        )
    return value


def _configured_h200_parity_cleanup() -> tuple[int, int]:
    settings = limits.get("h200_parity_cleanup")
    if not isinstance(settings, dict):
        raise Refused("limits.h200_parity_cleanup must be configured")
    timeout = settings.get("settle_timeout_seconds")
    poll = settings.get("poll_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 10 <= timeout <= 180
        or isinstance(poll, bool)
        or not isinstance(poll, int)
        or not 1 <= poll <= 10
        or poll > timeout
    ):
        raise Refused("H200 parity cleanup limits are invalid")
    return timeout, poll


def _configured_h200_parity_failure_evidence() -> tuple[int, int, int, int]:
    settings = limits.get("h200_parity_failure_evidence")
    if not isinstance(settings, dict):
        raise Refused("limits.h200_parity_failure_evidence must be configured")
    poll = settings.get("state_poll_seconds")
    inspect_bytes = settings.get("inspect_max_bytes")
    logs_bytes = settings.get("logs_max_bytes")
    events_bytes = settings.get("events_max_bytes")
    values = (poll, inspect_bytes, logs_bytes, events_bytes)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or not 1 <= poll <= 10
        or not 4096 <= inspect_bytes <= 262144
        or not 4096 <= logs_bytes <= 262144
        or not 4096 <= events_bytes <= 262144
    ):
        raise Refused("H200 parity failure evidence limits are invalid")
    return poll, inspect_bytes, logs_bytes, events_bytes


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


def _repository_root(name: str = "default") -> str:
    """Resolve a configured repository name to its remote root."""
    keys = {
        "default": "remote_repository",
        "glm52-qualification": "glm52_qualification_repository",
    }
    key = keys.get(name)
    if key is None:
        raise Refused(f"unknown repository {name!r}")
    root = paths.get(key)
    if not isinstance(root, str) or not root.startswith("/"):
        raise Refused(f"repository {name!r} is missing from operations config")
    return root.rstrip("/")


def _profile_trainer_container(profile: dict) -> str:
    """Resolve the configured trainer container for one profile."""
    key = profile.get("trainer_container", "trainer")
    if not isinstance(key, str) or key not in containers:
        raise Refused(
            f"profile trainer_container {key!r} is not a configured container"
        )
    container = containers[key]
    if (
        not isinstance(container, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container)
    ):
        raise Refused(f"configured trainer container {key!r} is invalid")
    return container


def _profile_trainer_repository(profile: dict) -> str:
    """Resolve the configured repository used by a profile's trainer."""
    qualification = profile.get("kind") == "glm52" and "orchestrator" not in profile
    return _repository_root("glm52-qualification" if qualification else "default")


_QUALIFICATION_LAUNCH_ENV_PATTERNS = {
    "PRIME_NSYS_PROFILE": re.compile(r"[01]"),
    "PRIME_CUDA_PROFILE": re.compile(r"[01]"),
    "PRIME_CUDA_PROFILE_RANK": re.compile(r"[0-9]+"),
    "PRIME_CUDA_PROFILE_STEPS": re.compile(r"(?:[2-9]|[1-9][0-9]+)"),
    "PRIME_CUDA_MEMORY_CAPTURE": re.compile(r"1"),
    "PRIME_CUDA_MEMORY_CAPTURE_RANK": re.compile(r"0"),
    "PRIME_CUDA_MEMORY_CAPTURE_STEP": re.compile(r"1"),
    "PRIME_CUDA_MEMORY_CAPTURE_PATH": re.compile(
        r"/data/glm52-rl/qualifications/glm52-native-b2048-memory/"
        r"cuda-memory-capture\.json"
    ),
    "PRIME_BF16_COMBINE_ROUNDS": re.compile(r"8"),
}

_CUDA_MEMORY_CAPTURE_PROFILE = "glm52-native-b2048-memory"
_CUDA_MEMORY_CAPTURE_KEYS = frozenset(
    {
        "PRIME_CUDA_MEMORY_CAPTURE",
        "PRIME_CUDA_MEMORY_CAPTURE_RANK",
        "PRIME_CUDA_MEMORY_CAPTURE_STEP",
        "PRIME_CUDA_MEMORY_CAPTURE_PATH",
    }
)
_CUDA_MEMORY_CAPTURE_ENV = {
    "PRIME_CUDA_MEMORY_CAPTURE": "1",
    "PRIME_CUDA_MEMORY_CAPTURE_RANK": "0",
    "PRIME_CUDA_MEMORY_CAPTURE_STEP": "1",
    "PRIME_CUDA_MEMORY_CAPTURE_PATH": (
        "/data/glm52-rl/qualifications/glm52-native-b2048-memory/"
        "cuda-memory-capture.json"
    ),
}

_BF16_COMBINE_ROUNDS_PROFILE = "glm52-native-b2048-memory"
_BF16_COMBINE_ROUNDS_KEYS = frozenset({"PRIME_BF16_COMBINE_ROUNDS"})
_BF16_COMBINE_ROUNDS_ENV = {"PRIME_BF16_COMBINE_ROUNDS": "8"}
_MEMORY_PROFILE_LAUNCH_ENV = {
    **_CUDA_MEMORY_CAPTURE_ENV,
    **_BF16_COMBINE_ROUNDS_ENV,
}

_NSYS_QUALIFICATION_PROFILE = "glm52-native-b2048-nsys"
_NSYS_TRAINER_MODULE = "prime_rl.trainer.nsys_worker"
_NSYS_ENV_KEYS = (
    "PRIME_NSYS_BIN",
    "PRIME_NSYS_VERSION",
    "PRIME_NSYS_OUTPUT",
    "PRIME_NSYS_SESSION",
)
_NSYS_BIN_PATTERN = re.compile(
    r"/opt/nvidia/nsight-systems/[A-Za-z0-9][A-Za-z0-9._+-]*/"
    r"target-linux[A-Za-z0-9._+-]*/nsys"
)


@dataclass(frozen=True)
class _QualificationNsightCapture:
    rank: int
    host: str
    output: str


def _qualification_nsys_capture(
    profile: str,
    launch_env: dict[str, str],
    nodes: list[str],
    trainer_output: str,
) -> _QualificationNsightCapture | None:
    """Resolve the one allowed Nsight worker for a strict qualification."""
    if profile != _NSYS_QUALIFICATION_PROFILE:
        return None
    if launch_env.get("PRIME_NSYS_PROFILE") != "1":
        raise Refused("Nsight qualification profile is not enabled")
    if launch_env.get("PRIME_CUDA_PROFILE") != "1":
        raise Refused("Nsight qualification requires CUDA profiler capture")
    if launch_env.get("PRIME_CUDA_PROFILE_RANK") != "24":
        raise Refused("Nsight qualification capture rank is not 24")
    if launch_env.get("PRIME_CUDA_PROFILE_STEPS") != "2":
        raise Refused("Nsight qualification capture step is not 2")
    try:
        rank = int(launch_env["PRIME_CUDA_PROFILE_RANK"])
    except (KeyError, TypeError, ValueError) as error:
        raise Refused("Nsight qualification capture rank is invalid") from error
    node_index, local_rank = divmod(rank, 8)
    if local_rank != 0 or node_index >= len(nodes):
        raise Refused("Nsight qualification capture rank is outside the topology")
    host = nodes[node_index]
    if host != "n78":
        raise Refused("Nsight qualification rank 24 must run on n78")
    output = PurePosixPath(trainer_output)
    if output.is_absolute() is False or ".." in output.parts:
        raise Refused("Nsight qualification output directory is invalid")
    return _QualificationNsightCapture(
        rank=rank,
        host=host,
        output=str(output / f"nsys-rank-{rank}"),
    )


def _qualification_nsys_environment_mismatches(
    observed: object,
    capture: _QualificationNsightCapture | None,
    host: str,
) -> list[str]:
    """Validate the dynamic Nsight environment and reject stray copies."""
    if not isinstance(observed, dict):
        return ["Nsight environment"]
    expected_keys = set(_NSYS_ENV_KEYS) if capture is not None and host == capture.host else set()
    mismatches = []
    if set(observed) != expected_keys:
        mismatches.append("Nsight environment keys")
    if capture is None or host != capture.host:
        return mismatches
    binary = observed.get("PRIME_NSYS_BIN")
    if (
        not isinstance(binary, str)
        or _NSYS_BIN_PATTERN.fullmatch(binary) is None
        or ".." in PurePosixPath(binary).parts
    ):
        mismatches.append("Nsight binary")
    version = observed.get("PRIME_NSYS_VERSION")
    if (
        not isinstance(version, str)
        or len(version) > 256
        or any(char in version for char in "\0\r\n")
        or re.search(r"\b[0-9]+\.[0-9]+\.[0-9]+\b", version) is None
    ):
        mismatches.append("Nsight version")
    if observed.get("PRIME_NSYS_OUTPUT") != capture.output:
        mismatches.append("Nsight output")
    if (
        not isinstance(observed.get("PRIME_NSYS_SESSION"), str)
        or re.fullmatch(r"glmkernel-r24-[0-9]+", observed["PRIME_NSYS_SESSION"]) is None
    ):
        mismatches.append("Nsight session")
    return mismatches


def _qualification_launch_env(profile: dict) -> dict[str, str]:
    """Validate the allowlisted, application-controlled capture environment."""
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
    memory_keys = set(validated) & _CUDA_MEMORY_CAPTURE_KEYS
    if memory_keys and memory_keys != _CUDA_MEMORY_CAPTURE_KEYS:
        raise Refused(
            "CUDA memory capture launch_env must define all pinned capture values"
        )
    if memory_keys and {
        key: validated[key] for key in _CUDA_MEMORY_CAPTURE_KEYS
    } != _CUDA_MEMORY_CAPTURE_ENV:
        raise Refused("CUDA memory capture launch_env differs from the pinned profile")
    return validated

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
trainer_ssh_command = [
    "StrictHostKeyChecking=yes" if value == "StrictHostKeyChecking=accept-new" else value
    for value in ssh_command
] + [
    "-o", "UpdateHostKeys=no",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
]
ssh_master_attempts = 3
ssh_master_start_timeout_seconds = 35
ssh_master_cleanup_attempts = 3
ssh_master_cleanup_timeout_seconds = 10
ssh_master_retry_delay_seconds = 1
ssh_master_persist_seconds = 300
ssh_config_resolve_timeout_seconds = 10
_ssh_control_paths: dict[str, str] = {}
_ssh_control_lock = Lock()


class Refused(RuntimeError):
    """Raised when a call would break one of the rules in SKILL.md."""


class _TrainerContainerAbsent(RuntimeError):
    """Raised when the configured trainer container does not exist."""


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


def _active_ssh_control_options(host: str) -> list[str]:
    with _ssh_control_lock:
        path = _ssh_control_paths.get(host)
    if path is None:
        return []
    return ["-o", "ControlMaster=no", "-o", f"ControlPath={path}"]


def _configured_ssh_control_path(host: str) -> str | None:
    """Return a live configured control path, or fail closed if it is stale."""
    _check_host(host)
    try:
        resolved = subprocess.run(
            [*ssh_command, "-G", host],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=ssh_config_resolve_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise Refused(f"SSH configuration resolution timed out on {host}") from error
    except OSError as error:
        raise Refused(f"SSH configuration resolution failed on {host}") from error
    if resolved.returncode:
        detail = (resolved.stderr or resolved.stdout or "configuration failed").strip()
        raise Refused(
            f"SSH configuration resolution failed on {host}: {detail[:160]}"
        )
    configured_path = None
    for line in (resolved.stdout or "").splitlines():
        key, separator, value = line.partition(" ")
        if separator and key == "controlpath":
            configured_path = value.strip()
            break
    if not configured_path or configured_path == "none":
        return None
    path = Path(configured_path)
    if not path.is_absolute():
        raise Refused(
            f"SSH configuration has a non-absolute control path on {host}"
        )
    # Do not follow a configured path or any parent component. An existing
    # path must be a Unix socket owned by a live master. Never silently fall
    # back to a new credentialed connection when configured state is stale.
    for parent in reversed(path.parents):
        try:
            parent_stat = os.lstat(parent)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise Refused(
                f"cannot inspect configured SSH control path on {host}: "
                f"{configured_path}"
            ) from error
        if stat.S_ISLNK(parent_stat.st_mode):
            raise Refused(
                f"configured SSH control path has a symlinked parent on {host}: "
                f"{configured_path}"
            )
    try:
        control_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise Refused(
            f"cannot inspect configured SSH control path on {host}: "
            f"{configured_path}"
        ) from error
    if stat.S_ISLNK(control_stat.st_mode):
        raise Refused(
            f"configured SSH control path is a symlink on {host}: "
            f"{configured_path}"
        )
    if not stat.S_ISSOCK(control_stat.st_mode):
        raise Refused(
            f"configured SSH control path is not a Unix socket on {host}: "
            f"{configured_path}"
        )
    try:
        checked = subprocess.run(
            [
                *ssh_command,
                "-S",
                configured_path,
                "-O",
                "check",
                host,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=ssh_master_cleanup_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise Refused(
            f"configured SSH control master check timed out on {host}: "
            f"{configured_path}"
        ) from error
    except OSError as error:
        raise Refused(
            f"configured SSH control master check failed on {host}: "
            f"{configured_path}"
        ) from error
    if checked.returncode:
        detail = (checked.stderr or checked.stdout or "control check failed").strip()
        raise Refused(
            f"configured SSH control master is stale on {host}: "
            f"{configured_path} ({detail[:120]})"
        )
    return configured_path


def _close_ssh_control_master(host: str, path: str) -> bool:
    for attempt in range(ssh_master_cleanup_attempts):
        try:
            result = subprocess.run(
                ["ssh", "-S", path, "-O", "exit", host],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=ssh_master_cleanup_timeout_seconds,
            )
            if result.returncode == 0:
                deadline = time.monotonic() + ssh_master_cleanup_timeout_seconds
                while Path(path).exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                return not Path(path).exists()
            if not Path(path).exists():
                return True
        except subprocess.TimeoutExpired:
            pass
        if attempt + 1 < ssh_master_cleanup_attempts:
            time.sleep(ssh_master_retry_delay_seconds)
    return False


def _start_ssh_control_master(host: str, path: str) -> None:
    detail = "connection failed"
    for attempt in range(ssh_master_attempts):
        try:
            result = subprocess.run(
                [
                    *ssh_command,
                    "-S",
                    path,
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    f"ControlPersist={ssh_master_persist_seconds}",
                    "-MNf",
                    host,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=ssh_master_start_timeout_seconds,
            )
            if result.returncode == 0:
                return
            detail = (result.stderr or result.stdout or detail).strip()
        except subprocess.TimeoutExpired:
            detail = "connection timed out"
        if not _close_ssh_control_master(host, path):
            raise RuntimeError(
                f"SSH connection setup cleanup failed on {host}; "
                f"control socket retained: {path}"
            )
        if attempt + 1 < ssh_master_attempts:
            time.sleep(ssh_master_retry_delay_seconds)
    raise RuntimeError(
        f"SSH connection setup failed on {host} after {ssh_master_attempts} attempts: "
        f"{detail[:160]}"
    )


@contextmanager
def _ssh_control_masters(hosts_to_open):
    checked = list(dict.fromkeys(_check_host(host) for host in hosts_to_open))
    if trainer_head in checked:
        checked = [trainer_head, *[host for host in checked if host != trainer_head]]
    with _ssh_control_lock:
        missing = [host for host in checked if host not in _ssh_control_paths]
    if not missing:
        yield
        return
    directory = Path(tempfile.mkdtemp(prefix="training-ops-ssh-"))
    adopted: list[tuple[str, str]] = []
    opened: list[tuple[str, str]] = []
    try:
        for host in missing:
            configured_path = _configured_ssh_control_path(host)
            if configured_path is not None:
                with _ssh_control_lock:
                    _ssh_control_paths[host] = configured_path
                adopted.append((host, configured_path))
                continue
            name = hashlib.sha256(host.encode()).hexdigest()[:16]
            path = str(directory / name)
            _start_ssh_control_master(host, path)
            with _ssh_control_lock:
                _ssh_control_paths[host] = path
            opened.append((host, path))
        yield
    finally:
        with _ssh_control_lock:
            for host, path in [*adopted, *opened]:
                if _ssh_control_paths.get(host) == path:
                    del _ssh_control_paths[host]
        failed = [
            f"{host}:{path}"
            for host, path in reversed(opened)
            if not _close_ssh_control_master(host, path)
        ]
        if failed:
            raise RuntimeError(
                "SSH connection cleanup failed; control sockets retained: "
                + ", ".join(failed)
            )
        try:
            directory.rmdir()
        except OSError as error:
            raise RuntimeError(
                f"SSH connection cleanup retained files in {directory}"
            ) from error


def _ssh(
    host: str,
    script: str,
    timeout: int = 120,
    connect_timeout: int | None = None,
) -> tuple[int, str]:
    _check_host(host)
    command = [*ssh_command, *_active_ssh_control_options(host)]
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
    except subprocess.TimeoutExpired as error:
        parts = []
        for value in (error.stdout, error.stderr):
            if isinstance(value, bytes):
                parts.append(value.decode("utf-8", errors="replace"))
            elif isinstance(value, str):
                parts.append(value)
        partial = "".join(parts)
        if len(partial) > 65536:
            partial = partial[-65536:]
        return 124, "TIMEOUT" + ("\n" + partial if partial else "")


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
echo "MUSE_ADAPTER $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__ADAPTER__/)"
echo "GLM_ADAPTER $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__GLM_ADAPTER__/)"
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
if test -r __GLM_TUNNEL_MANAGER__ && test -r __OPERATIONS__; then
  echo "GLM_TUNNELS $(python3 __GLM_TUNNEL_MANAGER__ --operations __OPERATIONS__ status 2>/dev/null || echo error)"
else
  echo "GLM_TUNNELS error"
fi
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


def _head_script(container: str = trainer_container) -> str:
    return (
        trainer_probe.replace("__TRAINER__", container)
        .replace("__ORCH__", orchestrator_container)
        .replace("__JUDGE__", str(judge_port))
        .replace("__ADAPTER__", str(adapter_port))
        .replace("__GLM_ADAPTER__", str(glm_adapter_port))
        .replace("__TUNNEL_FIRST__", str(inference_tunnel_ports[0]))
        .replace("__TUNNEL_LAST__", str(inference_tunnel_ports[-1]))
    )


def _orchestrator_script() -> str:
    manager = f"{remote_repo}/scripts/h200/glm_adapter_tunnels.py"
    deployed_operations = f"{remote_repo}/configs/operations.toml"
    return (
        orchestrator_probe.replace("__ORCH__", orchestrator_container)
        .replace("__GLM_TUNNEL_MANAGER__", shlex.quote(manager))
        .replace("__OPERATIONS__", shlex.quote(deployed_operations))
    )


def _trainer_container_config(
    container: str = trainer_container,
) -> tuple[list[str], list[str]]:
    inspect_filter = (
        "import json,sys; records=json.load(sys.stdin); "
        "valid=isinstance(records,list) and len(records)==1 and "
        "isinstance(records[0],dict) and isinstance(records[0].get('Config'),dict); "
        "config=records[0]['Config'] if valid else {}; "
        "environment=config.get('Env'); command=config.get('Cmd'); "
        "valid=valid and isinstance(environment,list) and "
        "all(isinstance(value,str) for value in environment) and "
        "isinstance(command,list) and all(isinstance(value,str) for value in command); "
        "valid or sys.exit('trainer inspection is invalid'); "
        "print(json.dumps({'run_records':[value for value in environment "
        "if value.startswith('RUN=')],'command':command},separators=(',',':')))"
    )
    inspect_command = (
        "inspection=$(sudo docker inspect --type container "
        f"{shlex.quote(container)} 2>&1); status=$?; "
        "if [ \"$status\" -ne 0 ]; then printf '%s\\n' \"$inspection\"; "
        "exit \"$status\"; fi; printf '%s\\n' \"$inspection\""
    )
    rc, output = _ssh(
        trainer_head,
        f"set -o pipefail; {inspect_command} | "
        f"python3 -c {shlex.quote(inspect_filter)}",
    )
    if rc:
        detail = output.strip() or f"docker inspect returned status {rc}"
        absent_details = {
            f"Error: No such object: {container}",
            "[]\n"
            f"Error response from daemon: No such container: {container}",
        }
        if rc == 1 and detail in absent_details:
            raise _TrainerContainerAbsent(
                f"no active trainer container {container} on {trainer_head}"
            )
        raise RuntimeError(
            f"cannot inspect trainer container {container} on "
            f"{trainer_head}: {detail}"
        )
    try:
        record = json.loads(output)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"trainer container {container} returned invalid configuration"
        ) from error
    if (
        not isinstance(record, dict)
        or set(record) != {"run_records", "command"}
        or not isinstance(record["run_records"], list)
        or not all(isinstance(value, str) for value in record["run_records"])
        or not isinstance(record["command"], list)
        or not all(isinstance(value, str) for value in record["command"])
    ):
        raise RuntimeError(
            f"trainer container {container} returned invalid configuration"
        )
    return record["run_records"], record["command"]


def _trainer_mode(
    container: str = trainer_container,
    repository_root: str = remote_repo,
) -> str:
    """Return `trainer-only` when the active trainer uses `[data.fake]`."""
    _, command = _trainer_container_config(container)
    markers = [index for index, value in enumerate(command) if value == "@"]
    config_paths = [value for value in command if value.endswith(".toml")]
    if (
        len(markers) != 1
        or markers[0] + 1 >= len(command)
        or len(config_paths) != 1
        or command[markers[0] + 1] != config_paths[0]
    ):
        raise RuntimeError(
            f"trainer container {container} must have one TOML config"
        )
    config = PurePosixPath(config_paths[0])
    config_root = PurePosixPath(repository_root) / "configs"
    if (
        not config.is_absolute()
        or ".." in config.parts
        or config_root not in config.parents
        or config.suffix != ".toml"
        or str(config) != config_paths[0]
    ):
        raise RuntimeError(
            f"trainer container {container} has an unapproved TOML config"
        )
    rc, out = _ssh(
        trainer_head,
        f"if grep -q '^\\[data\\.fake\\]' -- {shlex.quote(str(config))}; then "
        "printf '%s\\n' trainer-only; "
        "else status=$?; if [ \"$status\" -eq 1 ]; then printf '%s\\n' rl; "
        "else exit \"$status\"; fi; fi",
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(f"cannot read trainer mode on {trainer_head}: {out.strip()}")
    mode = out.strip()
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
    kind: str = ""
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
    glm_adapter: str = "000"
    glm_tunnels_recorded: int = 0
    glm_tunnels_healthy: int = 0
    glm_tunnels_committed: bool = False
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
                f"tunnels {self.tunnels_ok}/{len(inference_tunnel_ports)} | judge {self.judge}"
            )
            L.append(
                f"adapter servers: Muse/raw {adapter_port} {self.adapter} | "
                f"GLM/gzip {glm_adapter_port} {self.glm_adapter} | "
                f"GLM tunnels {self.glm_tunnels_healthy}/{self.glm_tunnels_recorded} "
                f"recorded, committed={'yes' if self.glm_tunnels_committed else 'no'}"
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


def status(
    run: str = "",
    quiet: bool = False,
    profile: str = "current",
) -> Status:
    """Probe every part of the run once and return the whole picture."""
    selected = _profile(profile)
    selected_trainer_container = _profile_trainer_container(selected)
    selected_trainer_repository = _profile_trainer_repository(selected)
    trainer_absent = False
    if not run:
        try:
            run = current_run(selected_trainer_container)
        except _TrainerContainerAbsent:
            run = "none"
            trainer_absent = True
    muse_run = selected["kind"] == "muse"
    engine_model = muse_model if muse_run else glm_model
    engine_container = muse_inference_container if muse_run else glm_inference_container
    s = Status(
        when=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        run=run,
        kind="muse" if muse_run else "glm52",
    )
    try:
        trainer_only = (
            not trainer_absent
            and _trainer_mode(
                selected_trainer_container,
                selected_trainer_repository,
            )
            == "trainer-only"
        )
    except _TrainerContainerAbsent:
        trainer_only = False

    def node(h):
        return _ssh(
            h,
            node_probe.replace("__TRAINER__", selected_trainer_container),
            timeout=60,
        )

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
                trainer_only_probe.replace(
                    "__TRAINER__", selected_trainer_container
                ),
                120,
            )
            f_nodes = ex.submit(_fan, trainer_hosts, node)
            head_rc, head_out = f_head.result()
            nodes_out = f_nodes.result()
            edge_orch_rc, edge_orch_out = 0, ""
            engines_out = {}
        else:
            f_head = ex.submit(
                _ssh, trainer_head, _head_script(selected_trainer_container), 300
            )
            f_edge_orch = ex.submit(
                _ssh,
                orchestrator_host,
                _orchestrator_script(),
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
        "GLM_TUNNELS",
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
    s.adapter = h.get("MUSE_ADAPTER", "000")
    s.glm_adapter = h.get("GLM_ADAPTER", "000")
    try:
        tunnel_status = json.loads(h.get("GLM_TUNNELS", "error"))
        if (
            not isinstance(tunnel_status, dict)
            or set(tunnel_status) != {"recorded", "healthy", "committed"}
            or isinstance(tunnel_status["recorded"], bool)
            or not isinstance(tunnel_status["recorded"], int)
            or isinstance(tunnel_status["healthy"], bool)
            or not isinstance(tunnel_status["healthy"], int)
            or not isinstance(tunnel_status["committed"], bool)
        ):
            raise ValueError
        s.glm_tunnels_recorded = tunnel_status["recorded"]
        s.glm_tunnels_healthy = tunnel_status["healthy"]
        s.glm_tunnels_committed = tunnel_status["committed"]
    except (TypeError, ValueError):
        pass
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
    if not trainer_only and s.kind == "muse" and s.adapter != "200":
        w.append(f"Muse adapter server returned {s.adapter} on port {adapter_port}")
    if not trainer_only and s.kind == "glm52":
        if s.glm_adapter != "200":
            w.append(
                f"GLM gzip adapter server returned {s.glm_adapter} on port {glm_adapter_port}"
            )
        expected_tunnels = len(glm_adapter_tunnel_ports)
        if (
            s.glm_tunnels_recorded != expected_tunnels
            or s.glm_tunnels_healthy != expected_tunnels
            or not s.glm_tunnels_committed
        ):
            w.append(
                f"GLM adapter tunnels are {s.glm_tunnels_healthy}/{s.glm_tunnels_recorded} "
                f"healthy/recorded; expected {expected_tunnels} committed"
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


def current_run(container: str = trainer_container) -> str:
    """The run name the trainer container was started with."""
    run_records, command = _trainer_container_config(container)
    run_names = [value.removeprefix("RUN=") for value in run_records]
    if len(run_names) > 1:
        raise RuntimeError(f"trainer container {container} has conflicting RUN records")
    for value in run_names:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise RuntimeError(f"trainer container {container} has an invalid RUN record")

    log_prefix = f"--log-dir={run_root}/"
    log_arguments = [value for value in command if value.startswith("--log-dir=")]
    if len(log_arguments) > 1:
        raise RuntimeError(
            f"trainer container {container} has conflicting log directories"
        )
    command_runs: list[str] = []
    for value in log_arguments:
        if not value.startswith(log_prefix):
            raise RuntimeError(
                f"trainer container {container} has an invalid log directory"
            )
        relative_text = value.removeprefix(log_prefix)
        relative = PurePosixPath(relative_text)
        if (
            str(relative) != relative_text
            or ".." in relative.parts
            or len(relative.parts) < 2
            or not re.fullmatch(r"[A-Za-z0-9_-]+", relative.parts[0])
        ):
            raise RuntimeError(
                f"trainer container {container} has an invalid log directory"
            )
        command_runs.append(relative.parts[0])

    recorded = run_names + command_runs
    if not recorded:
        raise RuntimeError(
            f"trainer container {container} has no valid run metadata"
        )
    if len(set(recorded)) != 1:
        raise RuntimeError(f"trainer container {container} has conflicting run records")
    return recorded[0]


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


def _glm_adapter_range_server(
    source_path: str, *, replace_if_needed: bool
) -> dict[str, object]:
    source = PurePosixPath(source_path)
    qualification_root = glm_adapter_root / "qualifications"
    relative_script = "scripts/ops/glm_adapter_range_server.py"
    local_script = repo_root / relative_script
    remote_script = f"{remote_repo}/{relative_script}"
    if (
        qualification_root not in source.parents
        or ".." in source.parts
        or re.fullmatch(r"[A-Za-z0-9_./-]+", str(source)) is None
        or not local_script.is_file()
    ):
        raise Refused("GLM adapter server source is invalid")
    script_sha256 = hashlib.sha256(local_script.read_bytes()).hexdigest()
    if replace_if_needed and not deploy(relative_script, nodes=[trainer_head]):
        raise RuntimeError("failed to deploy the GLM adapter range server")
    expected = {
        "name": glm_adapter_server_container,
        "image": trainer_image,
        "port": glm_adapter_port,
        "root": str(glm_adapter_root),
        "script": remote_script,
        "script_sha256": script_sha256,
        "source": str(source.relative_to(glm_adapter_root)),
        "labels": {
            "prime.glm-adapter-server": "1",
            "prime.repository": remote_repo,
        },
    }
    encoded = base64.b64encode(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    mode = "ensure" if replace_if_needed else "check"
    script = r'''import base64,gzip,json,os,re,socket,subprocess,sys,time,urllib.parse,urllib.request
mode,encoded=sys.argv[1:]
expected=json.loads(base64.b64decode(encoded,validate=True))
name=expected["name"]
def docker(*arguments,check=True):
 result=subprocess.run(["sudo","docker",*arguments],capture_output=True,text=True)
 if check and result.returncode: raise RuntimeError("docker command failed: "+result.stderr.strip()[:300])
 return result
def inspect(reference):
 result=docker("inspect",reference,check=False)
 if result.returncode:
  error=result.stderr.casefold()
  if "no such object:" in error or "no such container:" in error: return None
  raise RuntimeError("cannot inspect GLM adapter server: "+result.stderr.strip()[:300])
 records=json.loads(result.stdout)
 if len(records)!=1: raise RuntimeError("GLM adapter server inspect count differs")
 return records[0]
def fixed_errors(record,require_current):
 config=record.get("Config") or {}; host=record.get("HostConfig") or {}; state=record.get("State") or {}
 labels=config.get("Labels") or {}; mounts=record.get("Mounts") or []
 by_destination={mount.get("Destination"):mount for mount in mounts}
 errors=[]
 if record.get("Name")!="/"+name: errors.append("name")
 if config.get("Image")!=expected["image"]: errors.append("image")
 if config.get("User")!="1000:1000": errors.append("user")
 if config.get("Entrypoint")!=["/app/.venv/bin/python"]: errors.append("entrypoint")
 if config.get("Cmd")!=["/ops/glm_adapter_range_server.py","--port",str(expected["port"]),"--directory",expected["root"]]: errors.append("command")
 if any(labels.get(key)!=value for key,value in expected["labels"].items()): errors.append("labels")
 digest=labels.get("prime.script-sha256","")
 if re.fullmatch(r"[0-9a-f]{64}",digest) is None: errors.append("script label")
 if require_current and digest!=expected["script_sha256"]: errors.append("script version")
 if host.get("NetworkMode")!="host": errors.append("network")
 if host.get("ReadonlyRootfs") is not True: errors.append("read-only root")
 if (host.get("RestartPolicy") or {}).get("Name")!="unless-stopped": errors.append("restart policy")
 if host.get("PidsLimit")!=64: errors.append("PID limit")
 if host.get("Memory")!=536870912: errors.append("memory limit")
 if host.get("Init") is not True: errors.append("init")
 if "ALL" not in (host.get("CapDrop") or []): errors.append("capabilities")
 if not any(value.startswith("no-new-privileges") for value in (host.get("SecurityOpt") or [])): errors.append("security")
 wanted={"/ops/glm_adapter_range_server.py":expected["script"],expected["root"]:expected["root"]}
 for destination,source in wanted.items():
  mount=by_destination.get(destination) or {}
  if mount.get("Type")!="bind" or mount.get("Source")!=source or mount.get("RW") is not False: errors.append("mount "+destination)
 if require_current and state.get("Running") is not True: errors.append("running state")
 return errors
def listening():
 with socket.socket() as client:
  client.settimeout(1); return client.connect_ex(("127.0.0.1",expected["port"]))==0
def probe():
 path=urllib.parse.quote(expected["source"],safe="/")
 request=urllib.request.Request(f'http://127.0.0.1:{expected["port"]}/{path}',headers={"Range":"bytes=0-1023","Accept-Encoding":"gzip"})
 with urllib.request.urlopen(request,timeout=30) as response:
  raw_length=response.headers.get("X-Prime-Raw-Length","")
  if response.status!=206 or response.headers.get("Content-Encoding","").lower()!="gzip" or not raw_length.isdigit(): raise RuntimeError("GLM adapter server did not return gzip range evidence")
  with gzip.GzipFile(fileobj=response,mode="rb") as stream: data=stream.read()
  if len(data)!=int(raw_length) or not 1<=len(data)<=1024: raise RuntimeError("GLM adapter server raw range length differs")
 return len(data)
def remove_exact(record):
 identity=record.get("Id","")
 if re.fullmatch(r"[0-9a-f]{64}",identity) is None: raise RuntimeError("GLM adapter server container ID is invalid")
 current=inspect(identity)
 if current is None or fixed_errors(current,False): raise RuntimeError("GLM adapter server changed identity before replacement")
 docker("rm","-f",identity)
def created_by_action(record,identity):
 if record.get("Id")!=identity or record.get("Name")!="/"+name: return False
 labels=(record.get("Config") or {}).get("Labels") or {}
 return all(labels.get(key)==value for key,value in {**expected["labels"],"prime.script-sha256":expected["script_sha256"]}.items())
if mode not in {"check","ensure"}: raise RuntimeError("GLM adapter server mode is invalid")
if not os.path.isfile(expected["script"]): raise RuntimeError("GLM adapter server script is missing")
digest=subprocess.run(["sha256sum",expected["script"]],capture_output=True,text=True,check=True).stdout.split()[0]
if digest!=expected["script_sha256"]: raise RuntimeError("GLM adapter server script checksum differs")
existing=inspect(name)
if mode=="check":
 errors=["absent"] if existing is None else fixed_errors(existing,True)
 if errors: raise RuntimeError("GLM adapter server identity differs: "+", ".join(errors))
 measured=probe()
 print(json.dumps({"state":"checked","raw_bytes":measured,"script_sha256":digest},separators=(",",":")))
 raise SystemExit
if existing is not None:
 errors=fixed_errors(existing,False)
 if errors: raise RuntimeError("refusing wrong GLM adapter server identity: "+", ".join(errors))
 if not fixed_errors(existing,True):
  try:
   measured=probe()
   print(json.dumps({"state":"reused","raw_bytes":measured,"script_sha256":digest},separators=(",",":")))
   raise SystemExit
  except (OSError,RuntimeError): pass
 remove_exact(existing)
if listening(): raise RuntimeError("GLM adapter server port is owned by another process")
arguments=["run","-d","--name",name,"--restart","unless-stopped","--network","host","--read-only","--init","--user","1000:1000","--cap-drop","ALL","--security-opt","no-new-privileges:true","--pids-limit","64","--memory","512m","--label","prime.glm-adapter-server=1","--label","prime.repository="+expected["labels"]["prime.repository"],"--label","prime.script-sha256="+digest,"--mount","type=bind,src="+expected["script"]+",dst=/ops/glm_adapter_range_server.py,readonly","--mount","type=bind,src="+expected["root"]+",dst="+expected["root"]+",readonly","--entrypoint","/app/.venv/bin/python",expected["image"],"/ops/glm_adapter_range_server.py","--port",str(expected["port"]),"--directory",expected["root"]]
created=docker(*arguments).stdout.strip()
try:
 if re.fullmatch(r"[0-9a-f]{64}",created) is None: raise RuntimeError("GLM adapter server start returned invalid ID")
 record=inspect(created); errors=[] if record is not None else ["absent after start"]
 if record is not None: errors.extend(fixed_errors(record,True))
 if errors: raise RuntimeError("new GLM adapter server identity differs: "+", ".join(errors))
 deadline=time.monotonic()+30
 while True:
  try: measured=probe(); break
  except (OSError,RuntimeError):
   if time.monotonic()>=deadline: raise
   time.sleep(0.25)
 print(json.dumps({"state":"started","raw_bytes":measured,"script_sha256":digest},separators=(",",":")))
except BaseException:
 record=inspect(created)
 if record is not None and created_by_action(record,created): docker("rm","-f",created)
 raise
'''
    output = _remote_ok(
        trainer_head,
        shlex.join(["python3", "-c", script, mode, encoded]),
        "GLM adapter range server",
        timeout=180,
    ).strip()
    try:
        result = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("GLM adapter server report is invalid") from error
    allowed_states = {"started", "reused"} if replace_if_needed else {"checked"}
    if (
        result.get("state") not in allowed_states
        or result.get("script_sha256") != script_sha256
        or isinstance(result.get("raw_bytes"), bool)
        or not isinstance(result.get("raw_bytes"), int)
        or not 1 <= result["raw_bytes"] <= 1024
    ):
        raise RuntimeError("GLM adapter server report differs")
    return result


def _ensure_glm_adapter_range_server(source_path: str) -> dict[str, object]:
    return _glm_adapter_range_server(source_path, replace_if_needed=True)


def _check_glm_adapter_range_server(source_path: str) -> dict[str, object]:
    return _glm_adapter_range_server(source_path, replace_if_needed=False)


def _step20_adapter_tunnel_command(port: int) -> list[str]:
    if port not in glm_adapter_tunnel_ports:
        raise Refused("temporary adapter tunnel port is invalid")
    return [
        "ssh",
        "-NT",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "IPQoS=none",
        "-o",
        "Compression=no",
        "-L",
        f"127.0.0.1:{port}:127.0.0.1:{glm_adapter_port}",
        trainer_ssh,
    ]


def _start_step20_adapter_tunnels(host: str, source_url: str) -> list[dict]:
    if (
        host != inference_hosts[0]
        or re.fullmatch(
            rf"http://127\.0\.0\.1:{glm_adapter_tunnel_ports.start}/qualifications/[A-Za-z0-9_./-]+",
            source_url,
        )
        is None
        or ".." in source_url
    ):
        raise Refused("temporary adapter tunnel source is invalid")
    commands = [
        _step20_adapter_tunnel_command(port)
        for port in glm_adapter_tunnel_ports
    ]
    encoded = base64.b64encode(
        json.dumps(commands, separators=(",", ":")).encode()
    ).decode()
    script = """import base64,gzip,json,os,signal,socket,subprocess,sys,time,urllib.request
source_url,encoded=sys.argv[1:]
commands=json.loads(base64.b64decode(encoded,validate=True))
processes=[]
started={}
def listening(port):
 with socket.socket() as client:
  client.settimeout(1); return client.connect_ex(("127.0.0.1",port))==0
def cmdline(pid):
 with open(f"/proc/{pid}/cmdline","rb") as source: return [value.decode() for value in source.read().split(b"\\0") if value]
def exact(process,command):
 try: return process.poll() is None and cmdline(process.pid)==command
 except (FileNotFoundError,ProcessLookupError): return False
def cleanup_processes(items):
 remaining=[]
 for process,command in items:
  if not exact(process,command): continue
  try: os.killpg(process.pid,signal.SIGTERM)
  except ProcessLookupError: continue
  remaining.append((process,command))
 deadline=time.monotonic()+__CLEANUP_SECONDS__
 while remaining and time.monotonic()<deadline:
  remaining=[item for item in remaining if exact(*item)]
  if not remaining: break
  time.sleep(0.1)
 for process,command in remaining:
  if not exact(process,command): continue
  try: os.killpg(process.pid,signal.SIGKILL)
  except ProcessLookupError: pass
def cleanup():
 cleanup_processes(reversed(processes))
try:
 for command in commands:
  port=int(command[command.index("-L")+1].split(":")[1])
  if listening(port): raise RuntimeError(f"temporary adapter tunnel port {port} is already in use")
 for attempt in range(__ATTEMPTS__):
  launched=[]
  for command in commands:
   port=int(command[command.index("-L")+1].split(":")[1])
   if port in started: continue
   if listening(port): raise RuntimeError(f"temporary adapter tunnel port {port} changed owner")
   process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
   processes.append((process,command))
   launched.append((process,command,port))
   time.sleep(__STAGGER_SECONDS__)
  deadline=time.monotonic()+__START_SECONDS__
  pending=launched
  while pending and time.monotonic()<deadline:
   waiting=[]
   for process,command,port in pending:
    if not exact(process,command): continue
    if listening(port): started[port]=(process,command); continue
    waiting.append((process,command,port))
   pending=waiting
   if pending:
    time.sleep(0.1)
  failed=[(process,command) for process,command,port in launched if port not in started]
  cleanup_processes(failed)
  for _process,command in failed:
   port=int(command[command.index("-L")+1].split(":")[1])
   if listening(port): raise RuntimeError(f"temporary adapter tunnel port {port} changed owner")
  if len(started)==len(commands): break
  if attempt==__LAST_ATTEMPT__:
   missing=sorted(int(command[command.index("-L")+1].split(":")[1]) for command in commands if int(command[command.index("-L")+1].split(":")[1]) not in started)
   raise RuntimeError(f"temporary adapter tunnels {missing} failed after __ATTEMPTS__ exact attempts")
 records=[]
 for command in commands:
  port=int(command[command.index("-L")+1].split(":")[1])
  process,command=started[port]
  if process.poll() is not None or not listening(port) or cmdline(process.pid)!=command: raise RuntimeError(f"temporary adapter tunnel {port} did not start with its exact command")
  records.append({"pid":process.pid,"cmdline":command,"port":port})
 request=urllib.request.Request(source_url,headers={"Range":"bytes=0-1023","Accept-Encoding":"gzip"})
 try:
  with urllib.request.urlopen(request,timeout=__PROBE_SECONDS__) as response:
   raw_length=response.headers.get("X-Prime-Raw-Length","")
   if response.status!=206 or response.headers.get("Content-Encoding","").lower()!="gzip" or not raw_length.isdigit(): raise RuntimeError
   with gzip.GzipFile(fileobj=response,mode="rb") as stream: data=stream.read()
   if len(data)!=int(raw_length) or not 1<=len(data)<=1024: raise RuntimeError
 except BaseException as error: raise RuntimeError("required GLM gzip adapter server is not ready through temporary staging tunnels") from error
 print(json.dumps(records,separators=(",",":")))
except BaseException:
 cleanup(); raise
"""
    script = (
        script.replace("__ATTEMPTS__", str(step20_tunnel_attempts))
        .replace("__LAST_ATTEMPT__", str(step20_tunnel_attempts - 1))
        .replace("__START_SECONDS__", str(step20_tunnel_start_seconds))
        .replace("__CLEANUP_SECONDS__", str(step20_tunnel_cleanup_seconds))
        .replace(
            "__STAGGER_SECONDS__", str(step20_tunnel_launch_stagger_seconds)
        )
        .replace("__PROBE_SECONDS__", str(step20_tunnel_probe_seconds))
    )
    output = _remote_ok(
        host,
        shlex.join(["python3", "-c", script, source_url, encoded]),
        "temporary step-20 adapter tunnels",
        timeout=step20_tunnel_outer_timeout_seconds,
    ).strip()
    try:
        records = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("temporary adapter tunnel record is invalid") from error
    expected = {
        port: _step20_adapter_tunnel_command(port)
        for port in glm_adapter_tunnel_ports
    }
    if (
        not isinstance(records, list)
        or len(records) != len(glm_adapter_tunnel_ports)
        or {
            record.get("port")
            for record in records
            if isinstance(record, dict)
        }
        != set(expected)
        or any(
            not isinstance(record, dict)
            or isinstance(record.get("pid"), bool)
            or not isinstance(record.get("pid"), int)
            or record["pid"] <= 1
            or record.get("cmdline") != expected.get(record.get("port"))
            for record in records
        )
    ):
        raise RuntimeError("temporary adapter tunnel identity differs")
    return records


def _stop_step20_adapter_tunnels(host: str, records: list[dict]) -> None:
    expected = {
        port: _step20_adapter_tunnel_command(port)
        for port in glm_adapter_tunnel_ports
    }
    if (
        host != inference_hosts[0]
        or len(records) != len(glm_adapter_tunnel_ports)
        or any(
            not isinstance(record, dict)
            or record.get("cmdline") != expected.get(record.get("port"))
            or isinstance(record.get("pid"), bool)
            or not isinstance(record.get("pid"), int)
            or record["pid"] <= 1
            for record in records
        )
    ):
        raise Refused("temporary adapter tunnel cleanup identity is invalid")
    encoded = base64.b64encode(
        json.dumps(records, separators=(",", ":")).encode()
    ).decode()
    script = """import base64,json,os,signal,sys,time
records=json.loads(base64.b64decode(sys.argv[1],validate=True))
def cmdline(pid):
 with open(f"/proc/{pid}/cmdline","rb") as source: return [value.decode() for value in source.read().split(b"\\0") if value]
errors=[]
exact=[]
for record in records:
 pid=record["pid"]
 try: current=cmdline(pid)
 except FileNotFoundError: continue
 if current!=record["cmdline"]:
  errors.append(f"temporary adapter tunnel PID {pid} changed identity"); continue
 os.killpg(pid,signal.SIGTERM); exact.append(record)
deadline=time.monotonic()+10
alive=[]
while time.monotonic()<deadline:
 alive=[]
 for record in exact:
  try: alive.append((record,cmdline(record["pid"])))
  except FileNotFoundError: pass
 if not alive: break
 time.sleep(0.1)
for record,current in alive:
 if current!=record["cmdline"]: errors.append(f"temporary adapter tunnel PID {record['pid']} changed identity")
 else: os.killpg(record["pid"],signal.SIGKILL)
if errors: raise RuntimeError("; ".join(errors))
print("cleaned")
"""
    result = _remote_ok(
        host,
        shlex.join(["python3", "-c", script, encoded]),
        "temporary step-20 adapter tunnel cleanup",
        timeout=30,
    ).strip()
    if result != "cleaned":
        raise RuntimeError("temporary adapter tunnel cleanup returned invalid state")


def _download_glm_adapter_direct(
    host: str,
    source_path: str,
    destination_path: str,
    expected_digest: str,
    expected_size: int,
) -> dict[str, object]:
    source = PurePosixPath(source_path)
    destination = PurePosixPath(destination_path)
    qualification_root = glm_adapter_root / "qualifications"
    destination_root = PurePosixPath(glm_step20_h200_gate["adapter"]).parent
    if (
        host != inference_hosts[0]
        or qualification_root not in source.parents
        or ".." in source.parts
        or not destination.is_absolute()
        or ".." in destination.parts
        or destination.parent.parent != destination_root
        or re.fullmatch(
            r"\.adapter-candidate-[0-9a-f]{16}", destination.parent.name
        )
        is None
        or destination.name != "adapter_model.safetensors"
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise Refused("direct GLM adapter identity is invalid")
    downloader = (
        f"{remote_repo}/source/src/prime_rl/inference/vllm/worker/filesystem.py"
    )
    served_path = source.relative_to(glm_adapter_root)
    source_url = (
        f"http://127.0.0.1:{glm_adapter_tunnel_ports.start}/"
        f"{served_path}"
    )
    adapter_ports = ",".join(
        str(port) for port in glm_adapter_tunnel_ports
    )
    command = shlex.join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=10s",
            "3300s",
            "env",
            "PRIME_GLM_ADAPTER_GZIP=1",
            f"PRIME_ADAPTER_PORTS={adapter_ports}",
            "python3",
            downloader,
            source_url,
            destination_path,
            expected_digest,
            str(expected_size),
        ]
    )
    output = _remote_ok(
        host,
        command,
        "direct step-20 adapter download",
        timeout=3360,
    ).strip()
    try:
        result = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("direct adapter download report is invalid") from error
    if (
        result.get("bytes") != expected_size
        or result.get("sha256") != expected_digest
        or isinstance(result.get("seconds"), bool)
        or not isinstance(result.get("seconds"), (int, float))
        or result["seconds"] <= 0
        or not isinstance(result.get("chunk_attempts"), list)
        or not result["chunk_attempts"]
        or any(
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= 3
            for attempt in result["chunk_attempts"]
        )
    ):
        raise RuntimeError("direct adapter download identity differs")
    return result


def _cleanup_step20_adapter_candidate(host: str, candidate: PurePosixPath) -> None:
    final = PurePosixPath(glm_step20_h200_gate["adapter"])
    if (
        host != inference_hosts[0]
        or candidate.parent != final.parent
        or re.fullmatch(r"\.adapter-candidate-[0-9a-f]{16}", candidate.name)
        is None
    ):
        raise Refused("step-20 candidate cleanup identity is invalid")
    script = """import os,re,stat,sys
root,candidate_name=sys.argv[1:]
directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
allowed={"adapter_model.safetensors","adapter_config.json","STABLE","adapter_model.safetensors.sha256",".qualification-result.json","adapter_model.safetensors.assembling","adapter_model.safetensors.parts"}
try:
 try: candidate=os.open(candidate_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
 except FileNotFoundError: print("absent"); raise SystemExit
 try:
  info=os.fstat(candidate)
  if info.st_uid not in {0,1000} or info.st_gid not in {0,1000} or stat.S_IMODE(info.st_mode) not in {0o750,0o755}: raise RuntimeError("candidate directory identity differs")
  names=set(os.listdir(candidate))
  if not names<=allowed: raise RuntimeError("candidate cleanup file set differs")
  parts_name="adapter_model.safetensors.parts"
  if parts_name in names:
   parts=os.open(parts_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=candidate)
   try:
    part_names=os.listdir(parts)
    if any(re.fullmatch(r"part-[0-9]{3}",name) is None for name in part_names): raise RuntimeError("candidate part file set differs")
    for name in part_names:
     fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parts)
     try:
      item=os.fstat(fd)
      if not stat.S_ISREG(item.st_mode) or item.st_nlink!=1: raise RuntimeError("candidate part identity differs")
     finally: os.close(fd)
     os.unlink(name,dir_fd=parts)
   finally: os.close(parts)
   os.rmdir(parts_name,dir_fd=candidate)
   names.remove(parts_name)
  for name in names:
   fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=candidate)
   try:
    item=os.fstat(fd)
    if not stat.S_ISREG(item.st_mode) or item.st_nlink!=1: raise RuntimeError("candidate file identity differs")
   finally: os.close(fd)
   os.unlink(name,dir_fd=candidate)
 finally: os.close(candidate)
 os.rmdir(candidate_name,dir_fd=directory)
 os.fsync(directory)
 print("removed")
finally: os.close(directory)
"""
    state = _remote_ok(
        host,
        shlex.join(["python3", "-c", script, str(candidate.parent), candidate.name]),
        "step-20 adapter candidate cleanup",
        timeout=120,
    ).strip()
    if state not in {"absent", "removed"}:
        raise RuntimeError("step-20 adapter candidate cleanup returned invalid state")


def _cleanup_failed_step20_adapter_stage(
    error: BaseException,
    host: str,
    candidate: PurePosixPath,
    tunnel_records: list[dict],
) -> None:
    for cleanup in (
        lambda: _cleanup_step20_adapter_candidate(host, candidate),
        lambda: _stop_step20_adapter_tunnels(host, tunnel_records),
    ):
        try:
            cleanup()
        except (OSError, Refused, RuntimeError) as cleanup_error:
            error.add_note(f"step-20 failure cleanup also failed: {cleanup_error}")


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
        "{{json .HostConfig.RestartPolicy.Name}}\\n{{json .HostConfig.Init}}'"
    )
    rc, output = _ssh(host, inspect_command, timeout=45)
    if rc:
        raise Refused(f"cannot inspect {glm_inference_container} on {host}: {output.strip()}")
    # Docker's Go template emits literal backslash-n separators on this host.
    lines = output.replace("\\n", "\n").splitlines()
    if len(lines) != 7:
        raise Refused("H200 parity container inspection returned an invalid record")
    try:
        running = json.loads(lines[0])
        image = json.loads(lines[1])
        command = json.loads(lines[2])
        environment = json.loads(lines[3])
        mounts = json.loads(lines[4])
        restart_policy = json.loads(lines[5])
        init = json.loads(lines[6])
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
    if init is not True:
        raise Refused("H200 parity container has no init process")
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


def _h200_parity_listener_identity_command(
    timeout: int,
    expected_container_id: str,
    expected_host_boot_id: str,
    port: int = 8000,
) -> str:
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0:
        raise Refused("H200 parity listener timeout is invalid")
    if port not in {8000, h200_parity_backend_port}:
        raise Refused("H200 parity listener port is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", expected_container_id) is None:
        raise Refused("H200 parity listener container ID is invalid")
    if (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            expected_host_boot_id,
        )
        is None
    ):
        raise Refused("H200 parity listener host boot ID is invalid")
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
expected_id={shlex.quote(expected_container_id)}
expected_boot={shlex.quote(expected_host_boot_id)}
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
{{{{json .HostConfig.RestartPolicy.Name}}}}
{{{{json .HostConfig.Init}}}}') || fail 81 "${{phase}}container-inspect-failed"
  mapfile -t record <<<"$inspect" || fail 82 "${{phase}}container-inspect-record-unreadable"
  test "${{#record[@]}}" = 8 || fail 82 "${{phase}}container-inspect-record-invalid"
  running=${{record[0]}}
  container_pid=${{record[1]}}
  container_id=${{record[2]}}
  image=${{record[3]}}
  command=${{record[4]}}
  restart_count=${{record[5]}}
  restart_policy=${{record[6]}}
  init=${{record[7]}}
  test "$running" = true || fail 83 "${{phase}}container-not-running"
  [[ "$container_pid" =~ ^[0-9]+$ ]] || fail 84 "${{phase}}container-pid-invalid"
  test "$container_pid" -gt 1 || fail 84 "${{phase}}container-pid-invalid"
  [[ "$container_id" =~ ^[0-9a-f]{{64}}$ ]] || fail 85 "${{phase}}container-id-invalid"
  test "$container_id" = "$expected_id" || fail 85 "${{phase}}container-id-mismatch"
  test "$image" = {shlex.quote(json.dumps(inference_image))} || fail 86 "${{phase}}container-image-mismatch"
  test "$command" = {shlex.quote(expected_command)} || fail 87 "${{phase}}container-command-mismatch"
  test "$restart_count" = 0 || fail 88 "${{phase}}container-restarted"
  test "$restart_policy" = {shlex.quote(json.dumps('no'))} || fail 89 "${{phase}}container-restart-policy-invalid"
  test "$init" = true || fail 89 "${{phase}}container-init-disabled"
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
  test "${{#listener_pids[@]}}" -gt 0 || return 11
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
  test "${{#listener_inodes[@]}}" -gt 0 || return 12
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
  allow_wait=$3
  top_output=$(sudo docker top "$expected_id" -eo pid) || {{
    test "$allow_wait" = 1 && return 10
    fail 99 "${{phase}}docker-top-failed"
  }}
  mapfile -t top_lines <<<"$top_output" || fail 99 "${{phase}}docker-top-record-unreadable"
  test "${{#top_lines[@]}}" -gt 0 || fail 99 "${{phase}}docker-top-record-invalid"
  test "${{top_lines[0]}}" = PID || fail 99 "${{phase}}docker-top-header-invalid"
  member_count=0
  for top_line in "${{top_lines[@]:1}}"; do
    read -r top_pid extra <<<"$top_line" || fail 99 "${{phase}}docker-top-record-invalid"
    test -z "${{extra:-}}" || fail 99 "${{phase}}docker-top-record-invalid"
    [[ "$top_pid" =~ ^[0-9]+$ ]] || fail 99 "${{phase}}docker-top-record-invalid"
    if test "$top_pid" = "$member_pid"; then member_count=$((member_count+1)); fi
  done
  if test "$member_count" = 0 && test "$allow_wait" = 1; then return 13; fi
  test "$member_count" = 1 || fail 99 "${{phase}}listener-docker-membership-invalid"
}}
startup_inspect=""
while :; do
  current_boot=$(sudo cat -- /proc/sys/kernel/random/boot_id) || fail 97 host-boot-id-unreadable
  test "$current_boot" = "$expected_boot" || fail 98 host-boot-id-changed
  read_container ""
  if test -z "$startup_inspect"; then
    startup_inspect=$inspect
    startup_container_id=$container_id
    startup_container_pid=$container_pid
  else
    test "$container_id" = "$startup_container_id" || fail 100 container-changed-during-listener-wait
    test "$container_pid" = "$startup_container_pid" || fail 100 container-changed-during-listener-wait
    test "$inspect" = "$startup_inspect" || fail 100 container-changed-during-listener-wait
  fi
  if read_listener ""; then
    if read_membership "" "$listener_pid" 1; then
      test "$SECONDS" -le "$deadline" || fail 124 listener-timeout
      break
    else
      listener_status=$?
    fi
  else
    listener_status=$?
  fi
  case "$listener_status" in
    10) wait_detail=listener ;;
    11) wait_detail=listener-pid-missing ;;
    12) wait_detail=listener-socket-inode-missing ;;
    13) wait_detail=listener-docker-membership-missing ;;
    *) fail 99 listener-read-failed ;;
  esac
  if test $SECONDS -ge $deadline; then
    fail 124 "$wait_detail-timeout"
  else
    listener_remaining=$((deadline - SECONDS))
    listener_delay=2
    if test "$listener_delay" -ge "$listener_remaining"; then
      if test "$listener_remaining" -gt 1; then
        listener_delay=$((listener_remaining - 1))
      else
        listener_delay=0.1
      fi
      sleep "$listener_delay" || fail 99 listener-wait-failed
    else
      timeout --foreground "$listener_remaining" sleep "$listener_delay" || fail 99 listener-wait-failed
    fi
    test "$SECONDS" -le "$deadline" || fail 124 "$wait_detail-timeout"
  fi
done
first_inspect=$inspect
first_container_pid=$container_pid
first_container_id=$container_id
first_listener_output=$listener_output
first_listener_pid=$listener_pid
first_listener_inode=$listener_inode
read_membership "" "$first_listener_pid" 0
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
test "$first_boot_id" = "$expected_boot" || fail 98 host-boot-id-changed

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
read_membership second- "$second_listener_pid" 0
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
    expected_container_id: str,
    expected_host_boot_id: str,
    port: int = 8000,
) -> dict[str, object]:
    _check_host(host)
    command = _h200_parity_listener_identity_command(
        timeout, expected_container_id, expected_host_boot_id, port
    )
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


def _h200_parity_launch_identity(output: str) -> dict[str, str]:
    matches = re.findall(
        r"^parity-engine-identity:([0-9a-f]{64}):"
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
        output,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise Refused("isolated parity launch identity is missing or duplicated")
    container_id, host_boot_id = matches[0]
    return {"container_id": container_id, "host_boot_id": host_boot_id}


def _recover_h200_parity_launch_identity(
    host: str,
    paths: dict[str, str],
    launch_output: str,
    sensitive: tuple[str, ...],
) -> dict[str, str]:
    work = paths.get("work", "")
    match = re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/"
        r"\.diagnostic-work-([0-9a-f]{16})",
        work,
    )
    if match is None or paths != _parity_container_write_paths(match.group(1)):
        raise Refused("parity launch identity inventory is not exact")
    stdout_identity = None
    try:
        stdout_identity = _h200_parity_launch_identity(launch_output)
    except Refused:
        pass
    expected_command = json.dumps(
        ["@", glm_parity_inference_config, "--vllm-extra", glm_parity_vllm_extra],
        separators=(",", ":"),
    )
    command = f"""
set -euo pipefail
work={shlex.quote(work)}
cidfile={shlex.quote(paths["engine_cid"])}
identity={shlex.quote(paths["engine_identity"])}
name={shlex.quote(glm_inference_container)}
test -d "$work"
test ! -L "$work"
test "$(realpath -e "$work")" = "$work"
test "$(stat -c '%u:%g:%a' "$work")" = "1000:1000:750"
mapfile -t recovered < <(python3 - "$work" "$(basename "$cidfile")" "$(basename "$identity")" <<'PY'
import json,os,re,stat,sys
parent_path,cid_name,identity_name=sys.argv[1:]
parent=os.open(parent_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 st=os.fstat(parent)
 assert stat.S_ISDIR(st.st_mode) and st.st_uid==1000 and st.st_gid==1000
 assert stat.S_IMODE(st.st_mode)==0o750
 values=[]
 for name,owner in ((cid_name,0),(identity_name,0)):
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
  try:
   st=os.fstat(fd)
   assert stat.S_ISREG(st.st_mode) and st.st_uid==owner and st.st_gid==owner
   assert stat.S_IMODE(st.st_mode)==0o600
   data=os.read(fd,4097)
   assert os.read(fd,1)==b""
  finally:
   os.close(fd)
  values.append(data)
 cid_data,identity_data=values
 assert re.fullmatch(rb"[0-9a-f]{{64}}",cid_data)
 cid=cid_data.decode()
 value=json.loads(identity_data)
 assert set(value)=={{"container_id","host_boot_id"}}
 assert value["container_id"]==cid
 assert re.fullmatch(r"[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}",value["host_boot_id"])
 print(cid)
 print(value["host_boot_id"])
finally:
 os.close(parent)
PY
)
test "${{#recovered[@]}}" = 2
cid=${{recovered[0]}}
boot=${{recovered[1]}}
test "$(cat /proc/sys/kernel/random/boot_id)" = "$boot"
record=""
if record=$(sudo docker inspect "$cid" --format '{{{{.Id}}}}\n{{{{.Name}}}}\n{{{{.Config.Image}}}}\n{{{{json .Config.Cmd}}}}\n{{{{.RestartCount}}}}\n{{{{.HostConfig.RestartPolicy.Name}}}}\n{{{{json .HostConfig.Init}}}}' 2>/dev/null); then
  mapfile -t fields <<<"$record"
  test "${{#fields[@]}}" = 7
  test "${{fields[0]}}" = "$cid"
  test "${{fields[1]}}" = "/$name"
  test "${{fields[2]}}" = {shlex.quote(inference_image)}
  test "${{fields[3]}}" = {shlex.quote(expected_command)}
  test "${{fields[4]}}" = 0
  test "${{fields[5]}}" = no
  test "${{fields[6]}}" = true
else
  by_name=$(sudo docker ps -a --no-trunc --filter "name=^/${{name}}$" --format '{{{{.ID}}}} {{{{.Names}}}}')
  by_id=$(sudo docker ps -a --no-trunc --filter "id=$cid" --format '{{{{.ID}}}} {{{{.Names}}}}')
  test -z "$by_name"
  test -z "$by_id"
fi
printf 'parity-engine-identity:%s:%s\n' "$cid" "$boot"
"""
    recovered_output = _sensitive_ssh(
        host,
        command,
        "isolated parity launch identity recovery",
        sensitive,
        60,
    )
    recovered = _h200_parity_launch_identity(recovered_output)
    if stdout_identity is not None and stdout_identity != recovered:
        raise Refused("isolated parity launch output differs from its identity file")
    return recovered


def _h200_parity_container_cleanup_command(
    expected_container_id: str,
    expected_host_boot_id: str,
    settle_timeout: int,
    poll_seconds: int,
) -> str:
    if not isinstance(expected_container_id, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_container_id
    ) is None:
        raise Refused("H200 parity cleanup container ID is invalid")
    if not isinstance(expected_host_boot_id, str) or re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        expected_host_boot_id,
    ) is None:
        raise Refused("H200 parity cleanup host boot ID is invalid")
    if (
        isinstance(settle_timeout, bool)
        or not isinstance(settle_timeout, int)
        or not 10 <= settle_timeout <= 180
        or isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not 1 <= poll_seconds <= 10
        or poll_seconds > settle_timeout
    ):
        raise Refused("H200 parity cleanup limits are invalid")
    expected_command = json.dumps(
        ["@", glm_parity_inference_config, "--vllm-extra", glm_parity_vllm_extra],
        separators=(",", ":"),
    )
    return f"""
set -euo pipefail
name={shlex.quote(glm_inference_container)}
expected_id={shlex.quote(expected_container_id)}
expected_boot={shlex.quote(expected_host_boot_id)}
expected_command={shlex.quote(expected_command)}
settle_timeout={settle_timeout}
poll_seconds={poll_seconds}
deadline=$((SECONDS + settle_timeout))
container_state=unknown
fail() {{
  reported_state=${{2:-$container_state}}
  printf 'parity-cleanup-state:%s\nparity-cleanup-error:%s\n' "$reported_state" "$1" >&2
  exit 1
}}
remaining() {{
  remaining_seconds=$((deadline - SECONDS))
  test "$remaining_seconds" -gt 0
}}
bounded() {{
  remaining || return 124
  timeout --foreground "$remaining_seconds" "$@"
  code=$?
  test "$code" = 0 || return "$code"
  test "$SECONDS" -le "$deadline" || return 124
}}
bounded_sleep() {{
  remaining || return 124
  delay=$poll_seconds
  test "$delay" -le "$remaining_seconds" || delay=$remaining_seconds
  timeout --foreground "$remaining_seconds" sleep "$delay"
  code=$?
  test "$code" = 0 || return "$code"
  test "$SECONDS" -le "$deadline" || return 124
}}
boot=$(bounded sudo cat /proc/sys/kernel/random/boot_id) || fail host-boot-read
test "$boot" = "$expected_boot" || fail host-boot-changed
inspect_name_with() {{
  runner=$1
  "$runner" sudo docker inspect "$name" --format '{{{{.Id}}}}\n{{{{.State.Status}}}}\n{{{{.State.Running}}}}\n{{{{.State.Pid}}}}\n{{{{json .Config.Image}}}}\n{{{{json .Config.Cmd}}}}\n{{{{.RestartCount}}}}\n{{{{json .HostConfig.RestartPolicy.Name}}}}\n{{{{json .HostConfig.Init}}}}'
}}
inspect_name() {{
  inspect_name_with bounded
}}
inventory_name() {{
  bounded sudo docker ps -a --no-trunc --filter "name=^/${{name}}$" --format '{{{{.ID}}}} {{{{.Names}}}}'
}}
inventory_id() {{
  bounded sudo docker ps -a --no-trunc --filter "id=$expected_id" --format '{{{{.ID}}}} {{{{.Names}}}}'
}}
verify_record() {{
  local record="$1" phase="$2" id status running pid image command restart policy init
  local -a fields=()
  mapfile -t fields <<<"$record"
  test "${{#fields[@]}}" = 9 || fail "$phase-record-count"
  id=${{fields[0]}}
  status=${{fields[1]}}
  running=${{fields[2]}}
  pid=${{fields[3]}}
  image=${{fields[4]}}
  command=${{fields[5]}}
  restart=${{fields[6]}}
  policy=${{fields[7]}}
  init=${{fields[8]}}
  test "$id" = "$expected_id" || fail "$phase-container-replaced" unknown
  test "$image" = {shlex.quote(json.dumps(inference_image))} || fail "$phase-image-changed"
  test "$command" = {shlex.quote(expected_command)} || fail "$phase-command-changed"
  test "$restart" = 0 || fail "$phase-container-restarted"
  test "$policy" = '"no"' || fail "$phase-restart-policy-changed"
  test "$init" = true || fail "$phase-init-disabled"
  printf '%s\n%s\n%s\n' "$status" "$running" "$pid"
}}
first=""
if first=$(inspect_name 2>/dev/null); then
  first_id=${{first%%$'\n'*}}
  test "$first_id" = "$expected_id" || fail initial-container-replaced unknown
  container_state=retained
else
  listing=$(inventory_name) || fail initial-container-inventory
  test -z "$listing" || fail initial-container-inspect
  id_listing=$(inventory_id) || fail initial-id-inventory
  test -z "$id_listing" || fail initial-name-rebound unknown
  container_state=absent
fi
tracked=""
if test -n "$first"; then
  state=$(verify_record "$first" initial)
  status=$(sed -n '1p' <<<"$state")
  running=$(sed -n '2p' <<<"$state")
  pid=$(sed -n '3p' <<<"$state")
  case "$status:$running" in
    running:true|created:false|exited:false|dead:false|removing:false) ;;
    *) fail initial-state-invalid ;;
  esac
  top_deadline=$((SECONDS + 10))
  test "$top_deadline" -le "$deadline" || top_deadline=$deadline
  bounded_top() {{
    top_remaining=$((top_deadline - SECONDS))
    test "$top_remaining" -gt 0 || return 124
    timeout --foreground "$top_remaining" "$@"
    code=$?
    test "$code" = 0 || return "$code"
    test "$SECONDS" -le "$top_deadline" || return 124
  }}
  inspect_name_top() {{
    inspect_name_with bounded_top
  }}
  snapshot_identity() {{
    snapshot_boot=$(bounded_top sudo cat /proc/sys/kernel/random/boot_id) || return 10
    test "$snapshot_boot" = "$expected_boot" || fail initial-top-host-boot-changed
    snapshot_listing=$(bounded_top sudo docker ps -a --no-trunc --filter "name=^/${{name}}$" --format '{{{{.ID}}}} {{{{.Names}}}}') || return 10
    test "$snapshot_listing" = "$expected_id $name" || fail initial-top-container-name-mismatch unknown
    snapshot_inspect=$(inspect_name_top 2>/dev/null) || return 10
    snapshot_state=$(verify_record "$snapshot_inspect" initial-top)
    status=$(sed -n '1p' <<<"$snapshot_state")
    running=$(sed -n '2p' <<<"$snapshot_state")
    pid=$(sed -n '3p' <<<"$snapshot_state")
    case "$status:$running" in
      running:true|created:false|exited:false|dead:false|removing:false) ;;
      *) fail initial-top-state-invalid ;;
    esac
  }}
  retry_top_snapshot() {{
    remaining || fail initial-top-timeout
    test "$SECONDS" -lt "$top_deadline" || fail initial-top-timeout
    if snapshot_identity; then :; else
      test "$?" = 10 || fail initial-top-identity-probe
    fi
    test "$status:$running" = running:true || return 20
    top_remaining=$((top_deadline - SECONDS))
    test "$top_remaining" -gt 0 || fail initial-top-timeout
    top_delay=$poll_seconds
    test "$top_delay" -le "$top_remaining" || top_delay=$top_remaining
    timeout --foreground "$top_remaining" sleep "$top_delay" || fail initial-top-timeout
    test "$SECONDS" -le "$top_deadline" || fail initial-top-timeout
  }}
  declare -A snapshot_pids=()
  while :; do
    top=""
    top_lines=()
    snapshot_retry=0
    if top=$(bounded_top sudo docker top "$expected_id" -eo pid 2>/dev/null); then
      mapfile -t top_lines <<<"$top" || fail initial-top-record-unreadable
      test "${{#top_lines[@]}}" -gt 0 || fail initial-top-record-invalid
      test "${{top_lines[0]}}" = PID || fail initial-top-header-invalid
      if test "$status" = running && test "${{#top_lines[@]}}" = 1; then
        snapshot_retry=1
      fi
    elif test "$status" != running; then
      break
    else
      snapshot_retry=1
    fi
    candidate_tracked=""
    snapshot_pids=()
    if test "$snapshot_retry" = 0; then
      for top_line in "${{top_lines[@]:1}}"; do
        read -r process extra <<<"$top_line" || fail initial-top-record-invalid
        test -z "${{extra:-}}" || fail initial-top-record-invalid
        [[ "$process" =~ ^[0-9]+$ ]] || fail initial-pid-invalid
        [[ ! -v 'snapshot_pids[$process]' ]] || fail initial-pid-duplicate
        snapshot_pids["$process"]=1
        if ! stat=$(bounded_top sudo cat "/proc/$process/stat" 2>/dev/null); then
          snapshot_retry=1
          break
        fi
        start=${{stat##*) }}
        start=$(awk '{{print $20}}' <<<"$start")
        if [[ ! "$start" =~ ^[0-9]+$ ]]; then
          snapshot_retry=1
          break
        fi
        candidate_tracked+="$process:$start "
      done
      test "$status" != running || test "${{#snapshot_pids[@]}}" -gt 0 || snapshot_retry=1
    fi
    if test "$snapshot_retry" = 0; then
      if snapshot_identity; then
        if test "$status" = running; then
          tracked=$candidate_tracked
          test -n "$tracked" || fail initial-top-empty-running-snapshot
        fi
        break
      else
        identity_status=$?
        test "$identity_status" = 10 || fail initial-top-identity-probe
      fi
    fi
    if retry_top_snapshot; then :; else
      retry_status=$?
      test "$retry_status" = 20 && break
      fail initial-top-identity-probe
    fi
  done
  if test "$status" = running; then
    bounded sudo docker kill "$expected_id" >/dev/null || {{
      check=$(inspect_name 2>/dev/null) || fail kill-failed
      check_state=$(verify_record "$check" kill)
      test "$(sed -n '2p' <<<"$check_state")" = false || fail kill-failed
    }}
  fi
  current="$first"
  remove_ready=0
  while :; do
    boot_now=$(bounded sudo cat /proc/sys/kernel/random/boot_id) || fail settle-host-boot-read
    test "$boot_now" = "$expected_boot" || fail settle-host-boot-changed
    listing=$(inventory_name) || fail settle-name-inventory
    id_listing=$(inventory_id) || fail settle-id-inventory
    if test -n "$listing"; then
      test "$listing" = "$expected_id $name" || fail settle-container-replaced unknown
    fi
    if test -n "$id_listing"; then
      test "$id_listing" = "$expected_id $name" || fail settle-id-rebound unknown
    fi
    current=$(inspect_name 2>/dev/null) || current=""
    if test -z "$current"; then
      test -z "$listing" || fail settle-container-inspect
      test -z "$id_listing" || fail settle-id-inspect
      container_state=absent
      break
    fi
    current_state=$(verify_record "$current" settle)
    status=$(sed -n '1p' <<<"$current_state")
    running=$(sed -n '2p' <<<"$current_state")
    case "$status:$running" in
      exited:false|dead:false|removing:false) ;;
      running:true|created:false) ;;
      *) fail settle-state-invalid ;;
    esac
    live=0
    for item in $tracked; do
      process=${{item%%:*}}
      start=${{item#*:}}
      stat=$(bounded sudo cat "/proc/$process/stat" 2>/dev/null) || continue
      current_start=${{stat##*) }}
      current_start=$(awk '{{print $20}}' <<<"$current_start")
      if test "$current_start" = "$start"; then live=1; fi
    done
    if test "$status" != removing && test "$running" = false && test "$live" = 0; then
      remove_ready=1
      break
    fi
    bounded_sleep || fail container-settle-timeout
  done
  if test "$remove_ready" = 1; then
    container_state=unknown
    bounded sudo docker rm "$expected_id" >/dev/null || fail container-remove-failed
    container_state=removed_cleanup_incomplete
  fi
fi
boot_now=$(bounded sudo cat /proc/sys/kernel/random/boot_id) || fail final-host-boot-read
test "$boot_now" = "$expected_boot" || fail final-host-boot-changed
listing=$(inventory_name) || fail final-name-inventory
test -z "$listing" || fail final-name-present unknown
id_listing=$(inventory_id) || fail final-id-inventory
test -z "$id_listing" || fail final-id-present unknown
listeners=$(bounded sudo ss -H -ltnp) || fail final-listener-probe
test -z "$(awk '$4 ~ /(^|:)(8000|8100)$/ {{print}}' <<<"$listeners")" || fail final-listener-present
gpu_processes=$(bounded nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || fail final-gpu-probe
for item in $tracked; do
  process=${{item%%:*}}
  test "$(printf '%s\n' "$gpu_processes" | grep -cx "$process")" = 0 || fail final-gpu-process-present
done
test "$container_state" = removed_cleanup_incomplete && container_state=removed
printf '%s\n' "$container_state"
"""


def _diagnostic_capture_preflight(
    config_path: Path, side: str
) -> dict[str, int | str]:
    if side not in {"trainer", "h200"}:
        raise Refused("diagnostic capture side is invalid")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    env = config.get("env_vars", {})
    if env.get("PRIME_PARITY_DIAGNOSTICS") != "1":
        return {"files": 0, "estimated_bytes": 0}
    limit_names = (
        "PRIME_PARITY_DIAGNOSTIC_TOKENS",
        "PRIME_PARITY_DIAGNOSTIC_MAX_FILES",
        "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES",
    )
    limit_values = [env.get(name) for name in limit_names]
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[1-9][0-9]*", value) is None
        for value in limit_values
    ):
        raise Refused("diagnostic capture limits are invalid")
    tokens, max_files, max_bytes = (int(value) for value in limit_values)
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
    capture_shapes = [(tokens, 6144)] * len(names)
    for name, width in (
        ("layer0_normalized_input", 6144),
        ("layer0_kv_before_norm", 512),
        ("layer0_normalized_kv", 512),
        ("layer0_o_proj_input_tp0", 2048),
    ):
        names.append(name)
        capture_shapes.append((1, width))
    internal_tokens = 32
    internal = [
        ("layer0_dense_input", 6144),
        ("layer0_dense_gate_up_raw_tp0", 3072),
        ("layer0_dense_post_silu_tp0", 1536),
        (
            "layer3_shared_output_full"
            if side == "trainer"
            else "layer3_shared_output_local_tp0",
            6144,
        ),
        ("layer3_moe_combined", 6144),
    ]
    names.extend(name for name, _width in internal)
    capture_shapes.extend((internal_tokens, width) for _name, width in internal)
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
        capture_shapes.extend((tokens, width) for width in extra.values())
    if len(names) != len(set(names)):
        raise Refused("diagnostic capture plan contains duplicate semantic files")
    qualification_root = env.get("PRIME_PARITY_QUALIFICATION_ROOT")
    if (
        not isinstance(qualification_root, str)
        or not qualification_root.startswith("/")
        or PurePosixPath(qualification_root).as_posix() != qualification_root
    ):
        raise Refused("diagnostic qualification root is invalid")
    internal_names = {name for name, _width in internal}
    estimated_bytes = sum(
        entry_tokens * width * (2 if name in internal_names else 4) + 64 * 1024
        for name, (entry_tokens, width) in zip(names, capture_shapes, strict=True)
    )
    if len(names) > max_files or estimated_bytes > max_bytes:
        raise Refused("diagnostic capture plan exceeds its configured cap")
    return {
        "files": len(names),
        "estimated_bytes": estimated_bytes,
        "max_files": max_files,
        "max_bytes": max_bytes,
        "qualification_root": qualification_root,
    }


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
        "engine_cid": f"{work}/engine.cid",
        "engine_identity": f"{work}/engine-identity.json",
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
    failure_evidence = state.get("failure_evidence")
    if failure_evidence is not None:
        if (
            not isinstance(failure_evidence, dict)
            or failure_evidence.get("path") != f"{root}/.diagnostic-run-{nonce}/failure-evidence.json"
            or re.fullmatch(r"[0-9a-f]{64}", str(failure_evidence.get("sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(failure_evidence.get("container_id", ""))) is None
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                str(failure_evidence.get("host_boot_id", "")),
            )
            is None
        ):
            raise RuntimeError("parity transaction failure evidence is invalid")


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
cid={shlex.quote(paths["engine_cid"])}
identity={shlex.quote(paths["engine_identity"])}
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
for path in "$fixture" "$diagnostic" "$capture" "$cid" "$identity"; do
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
        root_files: tuple[str, ...] = ()
        names = "diagnostic.npz\nfixture\nh200-router-trace"
    elif stage == "capture":
        directories = (paths["capture"], paths["router_trace"])
        files = (
            paths["diagnostic"],
        )
        root_files = (paths["engine_cid"], paths["engine_identity"])
        names = (
            "capture\ndiagnostic.npz\nengine-identity.json\nengine.cid\n"
            "h200-router-trace"
        )
    else:
        raise Refused("parity container output stage is invalid")
    directory_values = " ".join(shlex.quote(path) for path in directories)
    file_values = " ".join(shlex.quote(path) for path in files)
    root_file_values = " ".join(shlex.quote(path) for path in root_files)
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
for path in {root_file_values}; do
  test -f "$path"
  test ! -L "$path"
  test "$(realpath -e "$path")" = "$path"
  test "$(stat -c '%u:%g:%a' "$path")" = "0:0:600"
  sudo test -r "$path"
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


def _h200_capture_publish_command(
    root: str,
    capture: str,
    candidate: str,
    previous: str,
    nonce: str,
    transaction_path: str,
) -> str:
    if root != glm_parity_root:
        raise Refused("H200 capture publish root is invalid")
    if re.fullmatch(r"[0-9a-f]{16}", nonce) is None:
        raise Refused("H200 capture publish nonce is invalid")
    failed = f"{root}/.h200-capture-publish-failed-{nonce}"
    validator = (
        "python3 scripts/glm52_policy_parity.py "
        f"--profile {shlex.quote(production_parity_profile)} "
        "validate-h200-capture"
    )
    return f"""
set -euo pipefail
root={shlex.quote(root)}
capture={shlex.quote(capture)}
candidate={shlex.quote(candidate)}
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
test -f "$candidate/manifest.json"
cd {shlex.quote(remote_repo)}
{validator} "$candidate" >/dev/null
previous={shlex.quote(previous)}
failed={shlex.quote(failed)}
test ! -e "$previous"
test ! -L "$previous"
test ! -e "$failed"
test ! -L "$failed"
directory_identity() {{
  local path=$1
  test -d "$path"
  test ! -L "$path"
  test "$(realpath -e "$path")" = "$path"
  stat -c '%d:%i:%u:%g:%a' -- "$path"
}}
require_directory_identity() {{
  local path=$1
  local expected=$2
  local actual
  actual=$(directory_identity "$path") || return 1
  test "$actual" = "$expected"
}}
require_absent() {{
  test ! -e "$1"
  test ! -L "$1"
}}
changed=0
had_previous=0
new_live=0
previous_identity=
capture_identity=
restore_previous() {{
  status=$?
  trap - EXIT
  set +e
  rollback_failed=0
  if test "$changed" = 1; then
    failed_identity=
    if test "$new_live" = 1; then
      if require_directory_identity "$capture" "$capture_identity" &&
        require_absent "$failed" && mv "$capture" "$failed"; then
        failed_identity=$capture_identity
        require_directory_identity "$failed" "$failed_identity" || rollback_failed=1
      else
        rollback_failed=1
      fi
    else
      require_absent "$capture" || rollback_failed=1
    fi
    if test "$had_previous" = 1; then
      if test "$rollback_failed" = 0 &&
        require_directory_identity "$previous" "$previous_identity" &&
        require_absent "$capture" && mv "$previous" "$capture"; then
        require_directory_identity "$capture" "$previous_identity" || rollback_failed=1
      else
        rollback_failed=1
      fi
    fi
    if test "$rollback_failed" = 0 && test -n "$failed_identity"; then
      if require_directory_identity "$failed" "$failed_identity" &&
        rm -rf -- "$failed" && require_absent "$failed"; then :; else
        rollback_failed=1
      fi
    fi
    if test "$rollback_failed" != 0; then
      printf '%s\n' 'capture-publish-rollback-failed; recovery artifacts retained' >&2
      exit 78
    fi
  fi
  exit "$status"
}}
trap restore_previous EXIT
state=created
test ! -L "$capture"
if test -e "$capture"; then
  capture_identity=$(directory_identity "$capture")
  require_absent "$previous"
  mv "$capture" "$previous"
  previous_identity=$capture_identity
  require_directory_identity "$previous" "$previous_identity"
  changed=1
  had_previous=1
  state=replaced
fi
candidate_identity=$(directory_identity "$candidate")
require_absent "$capture"
mv "$candidate" "$capture"
capture_identity=$candidate_identity
require_directory_identity "$capture" "$capture_identity"
changed=1
new_live=1
test -f "$capture/manifest.json"
{validator} "$capture" >/dev/null
require_directory_identity "$capture" "$capture_identity"
{_parity_transaction_update_command(transaction_path, "capture_published", "capture", '"$state"')}
trap - EXIT
printf '%s\n' "$state"
"""


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
        candidates = re.findall(
            r"parity-bootstrap-error:"
            r"(?:container-(?:exited|exited-after-token):status=[a-z]+,exit=[0-9]+,oom=(?:true|false)"
            r"|(?:host-boot-changed|container-missing|container-inspect-invalid|"
            r"container-inspect-count|container-identity-changed|fifo-identity-changed|"
            r"fifo-metadata-invalid|startup-grace-expired|bootstrap-framing-invalid|"
            r"bootstrap-encoding-invalid|bootstrap-capacity-invalid|token-missing|"
            r"fifo-cleanup-failed:[A-Za-z0-9_-]+))",
            process.stderr,
        )
        detail = candidates[-1] if candidates else ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"parity bootstrap receive failed on {host}{suffix}")
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


def _parity_bootstrap_receive_command(
    path: str,
    timeout: int,
    poll_seconds: int,
    container_id: str,
    host_boot_id: str,
) -> str:
    if (
        re.fullmatch(
            r"/data/glm52-rl/qualifications/glm52-policy-parity/" r"\.diagnostic-run-[0-9a-f]{16}/bootstrap\.fifo",
            path,
        )
        is None
    ):
        raise Refused("parity bootstrap FIFO path is not exact")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or timeout < 1
        or isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not 1 <= poll_seconds <= 2
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            host_boot_id,
        )
        is None
    ):
        raise Refused("parity bootstrap wait identity or limit is invalid")
    expected_command = [
        "@",
        glm_parity_inference_config,
        "--vllm-extra",
        glm_parity_vllm_extra,
    ]
    program = r"""
import base64
import json
import os
import select
import stat
import subprocess
import sys
import time

fifo, timeout_text, poll_text, expected_id, expected_boot, expected_name, expected_image, expected_command_text = sys.argv[1:]
timeout_seconds = int(timeout_text)
poll_seconds = int(poll_text)
expected_command = json.loads(expected_command_text)
deadline = time.monotonic() + timeout_seconds
fifo_fd = None
fifo_stat = None
token = None
failure = None

def inspect_container():
    if open("/proc/sys/kernel/random/boot_id", encoding="ascii").read().strip() != expected_boot:
        raise RuntimeError("host-boot-changed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("startup-grace-expired")
    result = subprocess.run(
        ["docker", "inspect", expected_id],
        capture_output=True,
        text=True,
        timeout=min(10.0, remaining),
        check=False,
    )
    if result.returncode:
        raise RuntimeError("container-missing")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("container-inspect-invalid") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RuntimeError("container-inspect-count")
    value = values[0]
    config = value.get("Config")
    host_config = value.get("HostConfig")
    state = value.get("State")
    restart = host_config.get("RestartPolicy") if isinstance(host_config, dict) else None
    if (
        value.get("Id") != expected_id
        or value.get("Name") != "/" + expected_name
        or not isinstance(config, dict)
        or config.get("Image") != expected_image
        or config.get("Cmd") != expected_command
        or not isinstance(restart, dict)
        or restart.get("Name") != "no"
        or host_config.get("Init") is not True
        or value.get("RestartCount") != 0
        or not isinstance(state, dict)
    ):
        raise RuntimeError("container-identity-changed")
    return state

def check_fifo(expected_stat):
    current = os.lstat(fifo)
    if (
        not stat.S_ISFIFO(current.st_mode)
        or current.st_uid != 1000
        or current.st_gid != 1000
        or stat.S_IMODE(current.st_mode) != 0o600
        or (current.st_dev, current.st_ino) != (expected_stat.st_dev, expected_stat.st_ino)
    ):
        raise RuntimeError("fifo-identity-changed")

try:
    fifo_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    fifo_stat = os.fstat(fifo_fd)
    if (
        not stat.S_ISFIFO(fifo_stat.st_mode)
        or fifo_stat.st_uid != 1000
        or fifo_stat.st_gid != 1000
        or stat.S_IMODE(fifo_stat.st_mode) != 0o600
    ):
        raise RuntimeError("fifo-metadata-invalid")
    data = b""
    while True:
        check_fifo(fifo_stat)
        state = inspect_container()
        if not state.get("Running"):
            raise RuntimeError(
                "container-exited:status=%s,exit=%s,oom=%s"
                % (state.get("Status"), state.get("ExitCode"), str(state.get("OOMKilled")).lower())
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("startup-grace-expired")
        ready, _, _ = select.select([fifo_fd], [], [], min(float(poll_seconds), remaining))
        if not ready:
            continue
        chunk = os.read(fifo_fd, max(1, 66 - len(data)))
        if not chunk:
            if data:
                if len(data) != 65 or not data.endswith(b"\n") or data.count(b"\n") != 1:
                    raise RuntimeError("bootstrap-framing-invalid")
                candidate = data[:-1]
                try:
                    decoded = base64.urlsafe_b64decode(candidate)
                except Exception as error:
                    raise RuntimeError("bootstrap-encoding-invalid") from error
                if len(candidate) != 64 or len(decoded) != 48:
                    raise RuntimeError("bootstrap-capacity-invalid")
                state = inspect_container()
                if not state.get("Running"):
                    raise RuntimeError(
                        "container-exited-after-token:status=%s,exit=%s,oom=%s"
                        % (state.get("Status"), state.get("ExitCode"), str(state.get("OOMKilled")).lower())
                    )
                token = candidate.decode("ascii")
                break
            time.sleep(min(float(poll_seconds), max(0.0, deadline - time.monotonic())))
            continue
        data += chunk
        if (
            len(data) > 65
            or data.count(b"\n") > 1
            or (b"\n" in data and not data.endswith(b"\n"))
        ):
            raise RuntimeError("bootstrap-framing-invalid")
except BaseException as error:
    failure = str(error) or type(error).__name__
finally:
    if fifo_fd is not None:
        os.close(fifo_fd)
    try:
        current = os.lstat(fifo)
        if (
            fifo_stat is None
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISFIFO(current.st_mode)
            or current.st_uid != 1000
            or current.st_gid != 1000
            or stat.S_IMODE(current.st_mode) != 0o600
            or (current.st_dev, current.st_ino) != (fifo_stat.st_dev, fifo_stat.st_ino)
        ):
            raise RuntimeError("fifo-cleanup-identity-changed")
        os.unlink(fifo)
        try:
            os.lstat(fifo)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("fifo-cleanup-retained")
    except FileNotFoundError:
        pass
    except BaseException as error:
        failure = "fifo-cleanup-failed:" + (str(error) or type(error).__name__)

if failure is not None:
    print("parity-bootstrap-error:" + failure, file=sys.stderr)
    raise SystemExit(70)
if token is None:
    print("parity-bootstrap-error:token-missing", file=sys.stderr)
    raise SystemExit(71)
print(token)
""".strip()
    return " ".join(
        (
            "sudo python3 -c",
            shlex.quote(program),
            shlex.quote(path),
            str(timeout),
            str(poll_seconds),
            container_id,
            host_boot_id,
            shlex.quote(glm_inference_container),
            shlex.quote(inference_image),
            shlex.quote(json.dumps(expected_command, separators=(",", ":"))),
        )
    )


def _write_h200_parity_failure_evidence(
    host: str,
    transaction_path: str,
    evidence: dict[str, object],
    sensitive: tuple[str, ...],
) -> str:
    nonce = _validate_parity_transaction_path(transaction_path)
    evidence_path = f"{glm_parity_root}/.diagnostic-run-{nonce}/failure-evidence.json"
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    for secret in sensitive:
        if secret and secret in payload:
            raise RuntimeError("parity failure evidence contains a secret")
    _poll, inspect_max, logs_max, events_max = _configured_h200_parity_failure_evidence()
    payload_limit = inspect_max + logs_max + events_max + 65536
    wire_payload = payload + "\n"
    if len(wire_payload.encode("utf-8")) > payload_limit:
        raise RuntimeError("H200 parity failure evidence exceeds its configured bound")
    command = (
        "sudo python3 -c 'import hashlib,os,pathlib,stat,sys; "
        "p=pathlib.Path(sys.argv[1]); root=p.parent; "
        "st=os.lstat(root); "
        "assert stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode); "
        'assert p.name=="failure-evidence.json" and not p.exists() and not p.is_symlink(); '
        't=root/".failure-evidence.json.tmp"; '
        "limit=int(sys.argv[2]); data=sys.stdin.buffer.read(limit+1); "
        "assert 0<len(data)<=limit and data.endswith(b\"\\n\"); "
        "fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); "
        "os.write(fd,data); os.fsync(fd); os.close(fd); os.replace(t,p); "
        "d=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); os.fsync(d); os.close(d); "
        "q=os.lstat(p); assert stat.S_ISREG(q.st_mode) and not stat.S_ISLNK(q.st_mode) "
        "and stat.S_IMODE(q.st_mode)==0o600; "
        "print(hashlib.sha256(p.read_bytes()).hexdigest())' "
        f"{shlex.quote(evidence_path)} {payload_limit}"
    )
    remote_hash = _sensitive_ssh_input(
        host,
        command,
        payload,
        "H200 parity failure evidence write",
        60,
    ).strip()
    expected_hash = hashlib.sha256(wire_payload.encode("utf-8")).hexdigest()
    if remote_hash != expected_hash:
        raise RuntimeError("H200 parity failure evidence hash differs after write")
    snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
    state = snapshot.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("parity transaction state is missing for failure evidence")
    state["failure_evidence"] = {
        "path": evidence_path,
        "sha256": expected_hash,
        "container_id": evidence.get("container_id"),
        "host_boot_id": evidence.get("host_boot_id"),
    }
    _write_parity_transaction_state(host, transaction_path, state, sensitive)
    return evidence_path


def _capture_h200_parity_failure_evidence(
    host: str,
    transaction_path: str,
    launch_identity: dict[str, str],
    reason: BaseException,
    sensitive: tuple[str, ...],
) -> dict[str, object]:
    poll_seconds, inspect_max, logs_max, events_max = _configured_h200_parity_failure_evidence()
    del poll_seconds
    container_id = launch_identity.get("container_id", "")
    boot_id = launch_identity.get("host_boot_id", "")
    if (
        re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
    ):
        raise Refused("H200 parity failure evidence identity is invalid")
    inspect_command = (
        "set -euo pipefail; "
        f'test "$(cat /proc/sys/kernel/random/boot_id)" = {shlex.quote(boot_id)}; '
        f"sudo docker inspect {container_id} | "
        "python3 -c 'import sys; limit=int(sys.argv[1]); "
        "data=sys.stdin.buffer.read(limit+1); "
        "assert len(data)<=limit; sys.stdout.buffer.write(data)' "
        f"{inspect_max}"
    )
    inspect_text = _sensitive_ssh(
        host,
        inspect_command,
        "H200 parity failure inspect",
        sensitive,
        30,
    )
    if len(inspect_text.encode("utf-8")) > inspect_max:
        raise RuntimeError("H200 parity failure inspect exceeds its configured bound")
    try:
        inspected = json.loads(inspect_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("H200 parity failure inspect is invalid JSON") from error
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
        or inspected[0].get("Id") != container_id
    ):
        raise RuntimeError("H200 parity failure inspect identity differs")
    state = inspected[0].get("State")
    if not isinstance(state, dict):
        raise RuntimeError("H200 parity failure inspect has no state")
    logs_command = "set -euo pipefail; " f"sudo docker logs --tail 200 {container_id} 2>&1 | " f"tail -c {logs_max}"
    logs_text = _sensitive_ssh(
        host,
        logs_command,
        "H200 parity failure logs",
        sensitive,
        30,
    )
    if len(logs_text.encode("utf-8")) > logs_max:
        raise RuntimeError("H200 parity failure logs exceed their configured bound")
    events_command = (
        "set -euo pipefail; until=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
        f'sudo docker events --since 30m --until "$until" --filter container={container_id} '
        f"--format '{{{{json .}}}}' | tail -c {events_max}"
    )
    events_text = _sensitive_ssh(
        host,
        events_command,
        "H200 parity failure events",
        sensitive,
        30,
    )
    if len(events_text.encode("utf-8")) > events_max:
        raise RuntimeError("H200 parity failure events exceed their configured bound")
    clean_reason = _redact_sensitive(str(reason), sensitive)
    records = {}
    for key, text_value in (
        ("inspect", inspect_text),
        ("logs", logs_text),
        ("events", events_text),
    ):
        for secret in sensitive:
            if secret and secret in text_value:
                raise RuntimeError(f"H200 parity failure {key} contains a secret")
        data = text_value.encode("utf-8")
        records[key] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "text": text_value,
        }
    evidence: dict[str, object] = {
        "schema_version": 1,
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "container_id": container_id,
        "host_boot_id": boot_id,
        "reason": clean_reason,
        "state": {
            "status": state.get("Status"),
            "running": state.get("Running"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled"),
        },
        "bounds": {
            "inspect_bytes": inspect_max,
            "logs_lines": 200,
            "logs_bytes": logs_max,
            "events_bytes": events_max,
        },
        "records": records,
    }
    evidence_path = _write_h200_parity_failure_evidence(
        host,
        transaction_path,
        evidence,
        sensitive,
    )
    log_tail = logs_text[-4096:].strip()
    return {
        "path": evidence_path,
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
        "log_tail": log_tail,
    }


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
    capture = glm_parity_paths["h200_capture"]
    fixture = glm_parity_paths["fixture"]
    stage = glm_parity_paths["trainer_stage"]
    roots = {PurePosixPath(path).parent for path in (capture, fixture, stage)}
    expected_root = PurePosixPath(
        "/data/glm52-rl/qualifications/glm52-policy-parity"
    )
    if roots != {expected_root}:
        raise Refused("GLM parity paths do not share the exact parity root")
    root = expected_root.as_posix()
    if plan.get("qualification_root") != root:
        raise Refused("H200 diagnostic qualification root differs from operations")
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
    launch_identity: dict[str, str] | None = None
    failure_evidence: dict[str, object] | None = None
    failure_evidence_attempted = False
    failure_evidence_failed = False
    cleanup_warnings: list[str] = []
    operation_error: BaseException | None = None
    result: dict[str, object] | None = None
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
            "input_fixture_sha256",
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
                "old_manifest_sha256": refresh_result["input_fixture_sha256"],
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
test "$(stat -c '%u:%g:%a' "$candidate")" = "1000:1000:750"
test -f "$candidate/manifest.json"
test ! -e "$previous"
test ! -L "$previous"
test ! -e "$failed"
test ! -L "$failed"
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
            f"prime_parity_engine_cidfile={shlex.quote(container_writes['engine_cid'])} "
            f"prime_parity_engine_identity_file={shlex.quote(container_writes['engine_identity'])} "
            "prime_require_absent=1 "
            f"bash {shlex.quote(remote_repo + '/scripts/h200/start_inference.sh')} "
            f"{shlex.quote(glm_parity_inference_config)}"
        )
        engine_launch_attempted = True
        start_code, start_output = _ssh(host, start_command, timeout=180)
        try:
            launch_identity = _recover_h200_parity_launch_identity(
                host,
                container_writes,
                start_output,
                sensitive,
            )
        except BaseException as recovery_error:
            if start_code:
                raise RuntimeError(
                    "isolated parity engine start failed and its exact identity "
                    f"could not be recovered: {_redact_sensitive(start_output.strip(), sensitive)}; "
                    f"{recovery_error}"
                ) from recovery_error
            raise
        if start_code:
            raise RuntimeError(
                "isolated parity engine start failed: "
                + _redact_sensitive(start_output.strip(), sensitive)
            )
        if launch_identity is None:
            raise RuntimeError("isolated parity engine start returned no identity")
        startup_grace = _configured_h200_engine_startup_grace()
        state_poll_seconds, _inspect_max, _logs_max, _events_max = (
            _configured_h200_parity_failure_evidence()
        )
        _h200_parity_container_preflight(host, h200_trace_directory)
        fifo_read = _parity_bootstrap_receive_command(
            bootstrap_fifo,
            startup_grace,
            state_poll_seconds,
            launch_identity["container_id"],
            launch_identity["host_boot_id"],
        )
        try:
            bootstrap_token = _receive_parity_bootstrap(
                host,
                fifo_read,
                startup_grace + 30,
            )
        except BaseException as error:
            failure_evidence_attempted = True
            try:
                failure_evidence = _capture_h200_parity_failure_evidence(
                    host,
                    transaction_path,
                    launch_identity,
                    error,
                    sensitive,
                )
                error.add_note(
                    "H200 parity engine failure: "
                    f"status={failure_evidence['status']}, "
                    f"exit={failure_evidence['exit_code']}, "
                    f"oom={failure_evidence['oom_killed']}; "
                    f"evidence={failure_evidence['path']}; "
                    f"log tail={failure_evidence['log_tail']}"
                )
            except BaseException as evidence_error:
                failure_evidence_failed = True
                error.add_note(
                    "H200 parity failure evidence could not be saved; "
                    f"work retained at {container_work_directory}: "
                    + _redact_sensitive(str(evidence_error), sensitive)
                )
            raise
        sensitive = (bootstrap_token,)
        listener_stabilization = _configured_h200_parity_listener_stabilization()
        api_listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            listener_stabilization,
            launch_identity["container_id"],
            launch_identity["host_boot_id"],
            h200_parity_backend_port,
        )
        listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            listener_stabilization,
            launch_identity["container_id"],
            launch_identity["host_boot_id"],
        )
        if (
            api_listener_identity["container_id"] != listener_identity["container_id"]
            or api_listener_identity["host_boot_id"] != listener_identity["host_boot_id"]
            or listener_identity["container_id"] != launch_identity["container_id"]
            or listener_identity["host_boot_id"] != launch_identity["host_boot_id"]
        ):
            raise Refused("H200 parity launch, router, and API identities differ")
        expected_api_process_id = api_listener_identity["listener_namespace_pid"]
        capture_environment = (
            f"--env PRIME_PARITY_ROUTER_TRACE_DIR={shlex.quote(h200_trace_directory)} "
            f"--env PRIME_PARITY_QUALIFICATION_ROOT={shlex.quote(root)} "
            f"--env PRIME_PARITY_DIAGNOSTIC_MAX_FILES={plan['max_files']} "
            f"--env PRIME_PARITY_DIAGNOSTIC_MAX_BYTES={plan['max_bytes']} "
        )
        capture_command = (
            "sudo docker run --rm -i --network host -v /data:/data "
            f"-v {shlex.quote(remote_repo)}:/workspace:ro -w /workspace "
            f"{capture_environment}"
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
            listener_stabilization,
            launch_identity["container_id"],
            launch_identity["host_boot_id"],
            h200_parity_backend_port,
        )
        if final_api_listener_identity != api_listener_identity:
            raise Refused("H200 parity API listener changed during capture")
        final_listener_identity = _h200_parity_listener_identity(
            host,
            sensitive,
            listener_stabilization,
            launch_identity["container_id"],
            launch_identity["host_boot_id"],
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
        publish_command = _h200_capture_publish_command(
            root,
            capture,
            temporary_capture,
            capture_previous,
            nonce,
            transaction_path,
        )
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
        commit_command = _parity_transaction_update_command(
            transaction_path, "committed", "none", "committed"
        )
        _sensitive_ssh(
            host,
            commit_command,
            "parity transaction commit record",
            sensitive,
            60,
        )
        commit_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
        commit_state = commit_snapshot.get("state")
        if not isinstance(commit_state, dict) or commit_state.get("phase") != "committed":
            raise RuntimeError("parity transaction commit state differs")
        fixture_replaced = _validate_published_transaction_artifact(
            commit_snapshot, "fixture"
        )
        capture_replaced = _validate_published_transaction_artifact(
            commit_snapshot, "capture"
        )
        committed = True
        succeeded = True
        result = {
            "capture": capture,
            "listener_identity": listener_identity,
            "api_listener_identity": api_listener_identity,
            "bootstrap_identity": bootstrap_identity,
            "launch_identity": launch_identity,
            "diagnostic_plan": plan,
            "engine_kept": keep_engine,
            "engine_cleanup": "retained" if keep_engine else "pending",
            "container_work": container_work_directory if keep_engine else None,
            "cleanup_warnings": cleanup_warnings,
        }
        return result
    except BaseException as error:
        operation_error = error
        raise
    finally:
        cleanup_failures: list[str] = []
        if (
            operation_error is not None
            and engine_launch_attempted
            and launch_identity is not None
            and not failure_evidence_attempted
        ):
            failure_evidence_attempted = True
            try:
                failure_evidence = _capture_h200_parity_failure_evidence(
                    host,
                    transaction_path,
                    launch_identity,
                    operation_error,
                    sensitive,
                )
                operation_error.add_note("H200 parity failure evidence saved at " + str(failure_evidence["path"]))
            except BaseException as evidence_error:
                failure_evidence_failed = True
                cleanup_failures.append(
                    "H200 parity failure evidence could not be saved; "
                    f"work retained at {container_work_directory}: " + _redact_sensitive(str(evidence_error), sensitive)
                )
        engine_cleanup = "retained" if engine_launch_attempted else "absent"
        if engine_launch_attempted and (not keep_engine or not succeeded):
            try:
                settle_timeout, poll_seconds = _configured_h200_parity_cleanup()
                if launch_identity is None:
                    raise Refused(
                        "H200 parity launch cleanup identity is missing; "
                        f"work retained at {container_work_directory}"
                    )
                expected_id = launch_identity["container_id"]
                expected_boot = launch_identity["host_boot_id"]
                container_cleanup = _h200_parity_container_cleanup_command(
                    expected_id,
                    expected_boot,
                    settle_timeout,
                    poll_seconds,
                )
                container_code, container_output = _ssh(
                    host,
                    container_cleanup,
                    timeout=settle_timeout + 15,
                )
                cleaned = container_output.strip()
                if container_code:
                    state_match = re.search(
                        r"^parity-cleanup-state:"
                        r"(retained|removed|absent|removed_cleanup_incomplete|unknown)$",
                        cleaned,
                        re.MULTILINE,
                    )
                    engine_cleanup = state_match.group(1) if state_match else "unknown"
                    raise RuntimeError(
                        _redact_sensitive(cleaned, sensitive)
                        or f"cleanup command returned {container_code}"
                    )
                if cleaned not in {"removed", "absent"}:
                    engine_cleanup = "unknown"
                    raise RuntimeError("cleanup command returned an invalid state")
                engine_cleanup = cleaned
            except BaseException as error:
                cleanup_failures.append(
                    "isolated parity container cleanup failed: "
                    + _redact_sensitive(str(error), sensitive)
                )
        if result is not None:
            result["engine_cleanup"] = engine_cleanup
            result["engine_kept"] = engine_cleanup == "retained"
            result["container_work"] = (
                container_work_directory if engine_cleanup == "retained" else None
            )
        retain_container_work = (
            engine_cleanup in {"retained", "unknown"} or failure_evidence_failed
        )
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
        if result is not None:
            result["container_work"] = (
                None
                if cleanup_code == 0 and not retain_container_work
                else container_work_directory
            )
        if not committed and (operation_error is not None or cleanup_failures):
            state_code, _state_output = _ssh(host, f"test -f {shlex.quote(transaction_path)}", timeout=60)
            rollback_artifacts: list[dict[str, object]] = []
            reconciliation_ok = False
            if state_code == 0:
                transaction_present = True
                try:
                    rollback_snapshot = _parity_transaction_snapshot(host, transaction_path, sensitive)
                    rollback_state = rollback_snapshot.get("state")
                    rollback_observed = rollback_snapshot.get("observed")
                    if not isinstance(rollback_state, dict) or not isinstance(
                        rollback_observed, dict
                    ):
                        raise RuntimeError("remote transaction state is missing")
                    unrecorded_recovery = False
                    for kind in ("fixture", "capture"):
                        artifact = rollback_state.get(kind)
                        if not isinstance(artifact, dict) or artifact.get("action") is None:
                            if not isinstance(artifact, dict):
                                raise RuntimeError(
                                    f"remote {kind} recovery state is missing"
                                )
                            live = rollback_observed.get(f"{kind}.live")
                            previous = rollback_observed.get(f"{kind}.previous")
                            failed = rollback_observed.get(f"{kind}.failed")
                            old_hash = artifact.get("old_manifest_sha256")
                            clean_live = (
                                isinstance(live, dict)
                                and live.get("symlink") is False
                                and (
                                    live.get("manifest_sha256") == old_hash
                                    if old_hash is not None
                                    else live.get("exists") is False
                                )
                            )
                            clean_previous = (
                                isinstance(previous, dict)
                                and previous.get("exists") is False
                                and previous.get("symlink") is False
                            )
                            clean_failed = (
                                isinstance(failed, dict)
                                and failed.get("exists") is False
                                and failed.get("symlink") is False
                            )
                            if not (
                                clean_live and clean_previous and clean_failed
                            ):
                                unrecorded_recovery = True
                                cleanup_failures.append(
                                    f"parity {kind} publish is incomplete; "
                                    "recovery paths retained"
                                )
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
                    reconciliation_ok = not unrecorded_recovery
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
        if committed:
            for failure in cleanup_failures:
                warning = f"committed parity cleanup incomplete: {failure}"
                cleanup_warnings.append(warning)
                print("WARNING: " + warning)
            cleanup_failures.clear()
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
        ) and failure_evidence is None
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
        if result is not None:
            result["cleanup_warnings"] = cleanup_warnings
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
        "{{json .HostConfig.RestartPolicy.Name}}\\n{{json .HostConfig.Init}}'",
        "exited parity container inspection",
        (),
        60,
    ).replace("\\n", "\n").splitlines()
    if len(inspection) != 4:
        raise Refused("exited parity container inspection returned an invalid record")
    try:
        image, command, restart_policy, init = (
            json.loads(value) for value in inspection
        )
    except json.JSONDecodeError as error:
        raise Refused("exited parity container inspection is not valid JSON") from error
    expected_command = [
        "@",
        glm_parity_inference_config,
        "--vllm-extra",
        glm_parity_vllm_extra,
    ]
    if (
        image != inference_image
        or command != expected_command
        or restart_policy != "no"
        or init is not True
    ):
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


_QUALIFICATION_ONLY_MOUNTS = {
    _CUDA_MEMORY_CAPTURE_PROFILE: frozenset({"source/tests/test_final_pp16_cuda.py"}),
}


def _mounts_in(script: Path) -> set[str]:
    """Repository paths mounted by the normal container launcher."""
    mounted = set(re.findall(r'-v "\$deployment/(\S+?)":', script.read_text()))
    for paths in _QUALIFICATION_ONLY_MOUNTS.values():
        mounted.difference_update(paths)
    return mounted


def _qualification_mounts(profile_name: str | None = None) -> set[str]:
    """Return normal mounts plus mounts selected by one qualification profile."""
    mounted = _mounts_in(repo_root / "scripts" / "china" / "cluster.sh")
    if profile_name is not None:
        mounted.update(_QUALIFICATION_ONLY_MOUNTS.get(profile_name, ()))
    return mounted


def _qualification_source_paths(profile_name: str | None = None) -> set[str]:
    """Return the exact source inventory used by one trainer qualification."""
    launcher = "scripts/china/cluster.sh"
    mounted = _qualification_mounts(profile_name)
    if not mounted:
        raise Refused("trainer launcher has no repository source mounts")
    return {launcher, *mounted}


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


def trainer_mount_preflight(
    nodes: list[str] | None = None,
    profile_name: str | None = None,
    repository_root: str | None = None,
) -> dict[str, str]:
    """Require every configured trainer bind source to match this worktree."""
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    manifest = sorted(_qualification_mounts(profile_name))
    remote_root = repository_root or remote_repo
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
        remote = f"{remote_root}/{path}"
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
        duplicates: set[str] = set()
        for line in output.splitlines():
            path, separator, digest = line.partition("\t")
            if separator:
                if path in observed:
                    duplicates.add(path)
                observed[path] = digest.strip()
        if duplicates:
            failures.append(
                f"{host}: duplicate trainer mount records: {', '.join(sorted(duplicates))}"
            )
        unexpected = sorted(set(observed) - set(expected))
        if unexpected:
            failures.append(
                f"{host}: unexpected trainer mount records: {', '.join(unexpected)}"
            )
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


def _trainer_qualification_source_files(
    nodes: list[str],
    profile_name: str | None = None,
    repository_root: str | None = None,
) -> dict[str, str]:
    source_files = trainer_mount_preflight(
        nodes, profile_name=profile_name, repository_root=repository_root
    )
    launcher = "scripts/china/cluster.sh"
    expected_paths = _qualification_source_paths(profile_name)
    if set(source_files) != expected_paths - {launcher}:
        raise Refused("trainer source manifest differs from launcher mounts")
    digest = hashlib.sha256((repo_root / launcher).read_bytes()).hexdigest()
    remote_root = repository_root or remote_repo
    remote = f"{remote_root}/{launcher}"
    results = _fan(
        nodes,
        lambda host: _ssh(
            host,
            f"test -f {shlex.quote(remote)} && sha256sum {shlex.quote(remote)} | cut -d' ' -f1",
            timeout=60,
        ),
        workers=min(2, len(nodes)),
    )
    failures = []
    for host in nodes:
        code, output = results.get(host, (1, "missing result"))
        if code or output.strip() != digest:
            failures.append(host)
    if failures:
        raise Refused(
            "qualification launcher checksum differs on " + ", ".join(failures)
        )
    manifest = {**source_files, launcher: digest}
    if set(manifest) != expected_paths:
        raise Refused("trainer qualification source manifest is incomplete")
    return manifest


production_parity_report = "/data/glm52-rl/qualifications/glm52-policy-parity/report.json"
production_parity_fixture = "/data/glm52-rl/qualifications/glm52-policy-parity/fixture"
production_parity_profile = "configs/qualification/glm52-policy-parity.toml"
production_batch2048_profile = "configs/qualification/glm52-native-b2048.toml"
production_stability20_profile = "configs/qualification/glm52-native-stability20.toml"
production_step20_h200_profile = glm_step20_h200_gate["profile"]
production_step20_h200_result = glm_step20_h200_gate["result"]
qualification_train_source = "source/src/prime_rl/trainer/rl/train.py"
legacy_stability20_train_sha256 = (
    "88bb92799b5133ee820b212bc38aa0818e9a87d8dc81ba439c6bec6780655ac0"
)
approved_recovery_train_sha256 = (
    "e4e6a1ba16ed10a1c46e1405e6d135d7735f2a84ff02749e0cadbefc79d75a93"
)
qualification_source_targets = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}
qualification_adapter_targets = qualification_source_targets - {"shared_expert"}


def _glm52_adapter_inventory(
    header: dict[str, object],
) -> dict[str, object]:
    train_path = repo_root / qualification_train_source
    tree = ast.parse(train_path.read_text())
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "_qualification_source_targets",
                    "_qualification_adapter_targets",
                }
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_adapter_inventory"
        )
    ]
    namespace = {"hashlib": hashlib, "re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(train_path), "exec"), namespace)
    shapes = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if (
            not isinstance(name, str)
            or not isinstance(value, dict)
            or value.get("dtype") != "BF16"
            or not isinstance(value.get("shape"), list)
            or any(isinstance(size, bool) or not isinstance(size, int) for size in value["shape"])
        ):
            raise Refused(f"qualification adapter tensor header is invalid: {name}")
        shapes[name] = tuple(value["shape"])
    return namespace["_qualification_adapter_inventory"](shapes)


def _safetensors_layout(
    header: dict[str, object], header_size: int
) -> dict[str, int]:
    offsets = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        shape = value.get("shape") if isinstance(value, dict) else None
        data_offsets = value.get("data_offsets") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("dtype") != "BF16"
            or not isinstance(shape, list)
            or any(isinstance(size, bool) or not isinstance(size, int) for size in shape)
            or not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in data_offsets
            )
        ):
            raise Refused(f"qualification adapter tensor layout is invalid: {name}")
        start, end = data_offsets
        if start < 0 or end - start != 2 * math.prod(shape):
            raise Refused(f"qualification adapter tensor layout is invalid: {name}")
        offsets.append((start, end))
    cursor = 0
    for start, end in sorted(offsets):
        if start != cursor:
            raise Refused("qualification adapter tensor data is not contiguous")
        cursor = end
    return {
        "header_bytes": header_size,
        "tensor_bytes": cursor,
        "file_bytes": 8 + header_size + cursor,
    }


def _remote_safetensors_header_and_size(
    host: str, path: str
) -> tuple[dict[str, object], int]:
    code, encoded = _ssh(
        host,
        f"head -c 8 -- {shlex.quote(path)} | base64 -w0",
        timeout=60,
    )
    try:
        raw = base64.b64decode(encoded.strip(), validate=True)
        header_size = struct.unpack("<Q", raw)[0]
    except (ValueError, struct.error) as error:
        raise Refused("qualification adapter header length is invalid") from error
    if code or not 2 <= header_size <= 64 * 1024 * 1024:
        raise Refused("qualification adapter header length is invalid")
    code, encoded = _ssh(
        host,
        "set -o pipefail; "
        f"dd if={shlex.quote(path)} iflag=skip_bytes,count_bytes skip=8 "
        f"count={header_size} status=none | gzip -1 | base64 -w0",
        timeout=180,
    )
    if code:
        raise Refused("qualification adapter header is unreadable")
    try:
        header = json.loads(gzip.decompress(base64.b64decode(encoded.strip(), validate=True)))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise Refused("qualification adapter header is invalid") from error
    if not isinstance(header, dict):
        raise Refused("qualification adapter header is not an object")
    return header, header_size


def _remote_safetensors_header(host: str, path: str) -> dict[str, object]:
    return _remote_safetensors_header_and_size(host, path)[0]


def _remote_safetensors_header_and_layout(
    host: str, path: str
) -> tuple[dict[str, object], dict[str, int]]:
    header, header_size = _remote_safetensors_header_and_size(host, path)
    layout = _safetensors_layout(header, header_size)
    code, output = _ssh(host, f"stat -c %s -- {shlex.quote(path)}", timeout=60)
    try:
        observed_size = int(output.strip())
    except ValueError as error:
        raise Refused("qualification adapter file size is invalid") from error
    if code or observed_size != layout["file_bytes"]:
        raise Refused("qualification adapter file size differs from its header")
    return header, layout


def _remote_file_bytes(host: str, path: str, maximum: int) -> bytes:
    code, output = _ssh(
        host,
        f"size=$(stat -c %s -- {shlex.quote(path)}) && "
        f"test \"$size\" -le {maximum} && gzip -1 -c -- {shlex.quote(path)} | base64 -w0",
        timeout=180,
    )
    if code:
        raise Refused(f"required evidence file is unreadable on {host}: {path}")
    try:
        return gzip.decompress(base64.b64decode(output.strip(), validate=True))
    except (OSError, ValueError) as error:
        raise Refused(f"required evidence file is invalid on {host}: {path}") from error


def _qualification_metrics_restore_script(
    directory: str,
    name: str,
    archive_name: str,
    archive_sha256: str,
    archive_size: int,
    metrics_sha256: str,
    metrics_size: int,
) -> str:
    return f"""set -euo pipefail
python3 - {shlex.quote(directory)} {shlex.quote(name)} {shlex.quote(archive_name)} {archive_sha256} {archive_size} {metrics_sha256} {metrics_size} <<'PY'
import gzip
import hashlib
import os
import stat
import sys

root, name, archive_name, archive_sha256, archive_size, metrics_sha256, metrics_size = sys.argv[1:]
archive_size = int(archive_size)
metrics_size = int(metrics_size)
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
temporary = f".{{name}}-recovering-{{os.getpid()}}"
try:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("qualification metric file already exists")
    archive = os.open(archive_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        archive_stat = os.fstat(archive)
        if not stat.S_ISREG(archive_stat.st_mode) or archive_stat.st_size != archive_size:
            raise SystemExit("qualification metric archive identity differs")
        compressed = bytearray()
        while len(compressed) <= archive_size:
            block = os.read(archive, min(1024 * 1024, archive_size + 1 - len(compressed)))
            if not block:
                break
            compressed.extend(block)
    finally:
        os.close(archive)
    if len(compressed) != archive_size or hashlib.sha256(compressed).hexdigest() != archive_sha256:
        raise SystemExit("qualification metric archive checksum differs")
    try:
        payload = gzip.decompress(compressed)
    except (EOFError, OSError) as error:
        raise SystemExit("qualification metric archive is invalid") from error
    if len(payload) != metrics_size or hashlib.sha256(payload).hexdigest() != metrics_sha256:
        raise SystemExit("qualification metric payload differs")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.utime(
            descriptor,
            ns=(archive_stat.st_atime_ns, archive_stat.st_mtime_ns),
        )
        os.fsync(descriptor)
        restored = os.fstat(descriptor)
        if not stat.S_ISREG(restored.st_mode):
            raise SystemExit("qualification metric output is not a file")
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
    os.fsync(directory)
finally:
    os.close(directory)
PY
"""


def _restore_remote_qualification_metrics(
    path: str,
    archive_path: str,
    archive_sha256: str,
    archive_size: int,
    metrics_sha256: str,
    metrics_size: int,
) -> None:
    destination = PurePosixPath(path)
    archive = PurePosixPath(archive_path)
    expected = (
        PurePosixPath("/data/glm52-rl/qualifications/glm52-native-stability20")
        / "trainer-metrics.jsonl"
    )
    if destination != expected or archive != PurePosixPath(f"{expected}.gz"):
        raise Refused("qualification metric recovery destination differs")
    script = _qualification_metrics_restore_script(
        str(destination.parent),
        destination.name,
        archive.name,
        archive_sha256,
        archive_size,
        metrics_sha256,
        metrics_size,
    )
    _remote_ok(trainer_head, script, "qualification metric recovery", timeout=60)


def _write_remote_qualification_result(path: str, result: dict) -> None:
    destination = PurePosixPath(path)
    if (
        destination
        != PurePosixPath("/data/glm52-rl/qualifications/glm52-native-stability20")
        / "qualification-result.json"
    ):
        raise Refused("qualification recovery destination differs")
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    encoded = base64.b64encode(payload).decode()
    expected = hashlib.sha256(payload).hexdigest()
    script = f"""set -euo pipefail
python3 - {shlex.quote(str(destination.parent))} {shlex.quote(destination.name)} {expected} {shlex.quote(encoded)} <<'PY'
import base64
import hashlib
import os
import stat
import sys

root, name, expected, encoded = sys.argv[1:]
payload = base64.b64decode(encoded, validate=True)
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("qualification recovery payload differs")
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
temporary = f".{{name}}-recovering"
try:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("qualification result already exists")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode):
            raise SystemExit("qualification recovery output is not a file")
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
    os.fsync(directory)
finally:
    os.close(directory)
PY
"""
    _remote_ok(trainer_head, script, "qualification recovery write", timeout=60)


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


def _glm_qualification_limits() -> tuple[int, int, float, float]:
    values = limits.get("glm52_qualification")
    if not isinstance(values, dict):
        raise Refused("GLM qualification limits are missing")
    age = values.get("evidence_max_age_seconds")
    session_duration = values.get("session_duration_seconds_max")
    duration = values.get("forward_backward_seconds_max")
    memory = values.get("peak_memory_gib_max")
    if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 604800:
        raise Refused("GLM qualification evidence age limit is invalid")
    if (
        isinstance(session_duration, bool)
        or not isinstance(session_duration, int)
        or not 1 <= session_duration <= 21600
    ):
        raise Refused("GLM qualification session duration limit is invalid")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration != 900
    ):
        raise Refused("GLM qualification trainer time limit is invalid")
    if (
        isinstance(memory, bool)
        or not isinstance(memory, (int, float))
        or memory != 46.0
    ):
        raise Refused("GLM qualification memory limit is invalid")
    return age, session_duration, float(duration), float(memory)


def _glm_b2048_deadlines() -> tuple[int, int, int, int]:
    """Return the measured fail-fast limits for the two-step b2048 check."""
    values = limits.get("glm52_qualification")
    if not isinstance(values, dict):
        raise Refused("GLM qualification limits are missing")
    session = values.get("b2048_session_duration_seconds_max")
    startup = values.get("b2048_startup_seconds_max")
    step = values.get("forward_backward_seconds_max")
    profiler = values.get("b2048_profiler_finalize_seconds_max")
    for name, value in (
        ("b2048 session duration", session),
        ("b2048 startup", startup),
        ("b2048 profiler finalization", profiler),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Refused(f"GLM qualification {name} limit is invalid")
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or step != 900
        or (session, startup, profiler) != (3600, 900, 120)
        or startup >= session
        or profiler >= session
    ):
        raise Refused("GLM b2048 qualification deadline limits are invalid")
    return int(session), int(startup), int(step), int(profiler)


def _exact_stability_recovery(result: dict, profile: dict) -> bool:
    recovery = result.get("recovery")
    expected_metrics_archive = str(
        PurePosixPath(profile["output_dir"])
        / f"{profile['file_monitor']['filename']}.gz"
    )
    return (
        isinstance(recovery, dict)
        and recovery.get("reason") == "adapter_leaf_target_metadata"
        and recovery.get("original_train_sha256")
        == legacy_stability20_train_sha256
        and recovery.get("input_checks") == "executed_not_persisted"
        and recovery.get("runtime_records") == "derived_from_session"
        and re.fullmatch(r"[0-9a-f]{64}", str(recovery.get("finalizer_sha256")))
        is not None
        and recovery.get("metrics_archive_path") == expected_metrics_archive
        and re.fullmatch(
            r"[0-9a-f]{64}", str(recovery.get("metrics_archive_sha256"))
        )
        is not None
        and recovery.get("forward_backward_memory")
        == "steps_1_19_total_without_adapter_export"
        and recovery.get("step20_forward_backward_peak") == "not_persisted"
        and isinstance(recovery.get("step20_total_export_peak_gib"), (int, float))
        and not isinstance(recovery.get("step20_total_export_peak_gib"), bool)
        and math.isfinite(recovery["step20_total_export_peak_gib"])
        and set(recovery)
        == {
            "reason",
            "original_train_sha256",
            "input_checks",
            "runtime_records",
            "finalizer_sha256",
            "metrics_archive_path",
            "metrics_archive_sha256",
            "forward_backward_memory",
            "step20_forward_backward_peak",
            "step20_total_export_peak_gib",
        }
    )


def _qualification_sources_match(
    session: dict,
    result: dict,
    profile: dict,
    profile_name: str,
    current_sources: dict[str, str],
) -> bool:
    source_files = session.get("source_files")
    try:
        expected_paths = _qualification_source_paths(profile_name)
    except Refused:
        return False
    if (
        not isinstance(current_sources, dict)
        or set(current_sources) != expected_paths
    ):
        return False
    if source_files == current_sources:
        return True
    if (
        profile_name != "glm52-native-stability20"
        or not _exact_stability_recovery(result, profile)
        or not isinstance(source_files, dict)
        or set(source_files) != expected_paths
        or set(source_files) != set(current_sources)
        or source_files.get(qualification_train_source)
        != legacy_stability20_train_sha256
        or current_sources.get(qualification_train_source)
        != approved_recovery_train_sha256
    ):
        return False
    return all(
        source_files[path] == current_sources[path]
        for path in source_files
        if path != qualification_train_source
    )


def _validate_trainer_qualification(
    session: dict,
    result: dict,
    profile: dict,
    profile_name: str,
    *,
    now: _dt.datetime | None = None,
) -> list[str]:
    failures = []
    age_limit, session_duration_limit, duration_limit, memory_limit = (
        _glm_qualification_limits()
    )
    expected_step_counts = {
        "glm52-native-b2048": 2,
        "glm52-native-b2048-memory": 2,
        "glm52-native-b2048-nsys": 2,
        "glm52-native-delta3": 3,
        "glm52-native-stability20": 20,
    }
    if profile_name not in expected_step_counts:
        raise Refused("GLM qualification profile is invalid")
    expected_steps = expected_step_counts[profile_name]
    expected_adapter = profile_name in {
        "glm52-native-delta3",
        "glm52-native-stability20",
    }
    profile_sha = hashlib.sha256(
        (repo_root / f"configs/qualification/{profile_name}.toml").read_bytes()
    ).hexdigest()
    expected_topology = {
        "world_size": 64,
        "pp": 8,
        "cp": 8,
        "ep": 8,
        "dp_replicate": 1,
        "dp_shard": 1,
    }
    expected_targets = qualification_source_targets
    recovery = result.get("recovery")
    recovered = _exact_stability_recovery(result, profile)
    if recovery is not None and not recovered:
        failures.append("qualification recovery evidence is invalid")
    if recovered and (
        profile_name != "glm52-native-stability20"
        or session.get("source_files", {}).get(qualification_train_source)
        != legacy_stability20_train_sha256
    ):
        failures.append("qualification recovery source differs")
    if session.get("schema_version") != 1 or session.get("kind") != (
        "glm52_trainer_qualification_session"
    ):
        failures.append("qualification session schema is invalid")
    if profile_name in {
        "glm52-native-b2048",
        "glm52-native-b2048-memory",
        "glm52-native-b2048-nsys",
    }:
        session_deadline, startup_deadline, step_deadline, profiler_deadline = (
            _glm_b2048_deadlines()
        )
        if session.get("deadlines") != {
            "session_seconds": session_deadline,
            "startup_seconds": startup_deadline,
            "forward_backward_seconds": step_deadline,
            "profiler_finalize_seconds": profiler_deadline,
        }:
            failures.append("qualification deadline evidence differs")
    if result.get("schema_version") != 1 or result.get("kind") != (
        "glm52_trainer_qualification"
    ):
        failures.append("qualification result schema is invalid")
    session_sha = _canonical_json_sha256(session)
    for name in ("session_id", "profile", "config_sha256", "source_sha256"):
        if result.get(name) != session.get(name):
            failures.append(f"qualification {name} differs")
    if result.get("session_sha256") != session_sha:
        failures.append("qualification session checksum differs")
    if session.get("profile") != profile_name or session.get("config_sha256") != profile_sha:
        failures.append("qualification profile checksum differs")
    if session.get("topology") != expected_topology or result.get("topology") != expected_topology:
        failures.append("qualification topology differs")
    ranked_hosts = [trainer_head] + [host for host in trainer_hosts if host != trainer_head]
    if session.get("trainer_hosts") != ranked_hosts or session.get("trainer_head") != trainer_head:
        failures.append("qualification trainer host mapping differs")
    if result.get("trainer_head") != session.get("trainer_head"):
        failures.append("qualification result trainer head differs")
    image_digests = session.get("image_digests")
    if (
        not isinstance(image_digests, dict)
        or set(image_digests) != set(ranked_hosts)
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)) is None
            for value in image_digests.values()
        )
        or result.get("trainer_head_image_digest") != image_digests.get(trainer_head)
    ):
        failures.append("qualification trainer image evidence differs")
    source_files = session.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != _qualification_source_paths(profile_name)
    ):
        failures.append("qualification source file evidence differs from launcher mounts")
    elif hashlib.sha256(
        (json.dumps(source_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest() != session.get("source_sha256"):
        failures.append("qualification source file checksum differs")
    fake = profile.get("data", {}).get("fake", {})
    expected_fake = {
        "batch_size": 64,
        "samples_per_micro_batch": 32,
        "group_size": 16,
        "sequence_length": 32768,
        "rollout_cache": False,
    }
    if result.get("fake_data") != expected_fake or (
        fake.get("batch_size"),
        fake.get("samples_per_micro_batch"),
        fake.get("group_size"),
    ) != (64, 32, 16):
        failures.append("qualification fake batch shape differs")
    if "rollout_cache" in profile.get("data", {}):
        failures.append("qualification uses a rollout cache")
    lora = result.get("lora", {})
    if (
        lora.get("rank") != 32
        or lora.get("alpha") != 64
        or set(lora.get("targets", [])) != expected_targets
    ):
        failures.append("qualification LoRA coverage differs")
    if profile.get("optim", {}).get("lr") != 8e-6 or result.get("max_steps") != expected_steps:
        failures.append("qualification learning rate or step count differs")
    expected_metrics_path = str(
        PurePosixPath(profile["output_dir"])
        / profile["file_monitor"]["filename"]
    )
    if (
        result.get("metrics_path") != expected_metrics_path
        or re.fullmatch(r"[0-9a-f]{64}", str(result.get("metrics_sha256")))
        is None
    ):
        failures.append("qualification metric file evidence differs")
    records = result.get("steps")
    if not isinstance(records, list) or [row.get("step") for row in records if isinstance(row, dict)] != list(
        range(1, expected_steps + 1)
    ):
        failures.append("qualification completed steps differ")
    else:
        previous_input = None
        for row in records:
            numeric = (
                "data/rollout_rows",
                "loss/mean",
                "optim/grad_norm",
                "optim/lr",
                "perf/peak_memory",
                "time/forward_backward",
            )
            if not recovered:
                numeric = (*numeric, "perf/forward_backward_peak_memory")
            if any(
                isinstance(row.get(key), bool)
                or not isinstance(row.get(key), (int, float))
                or not math.isfinite(row[key])
                for key in numeric
            ):
                failures.append(f"qualification step {row.get('step')} has invalid metrics")
                continue
            if row["data/rollout_rows"] != 2048:
                failures.append(f"qualification step {row['step']} does not contain 2048 samples")
            if not math.isclose(row["optim/lr"], 8e-6, rel_tol=0, abs_tol=1e-12):
                failures.append(f"qualification step {row['step']} learning rate differs")
            if row["time/forward_backward"] <= 0 or row["time/forward_backward"] > duration_limit:
                failures.append(f"qualification step {row['step']} trainer time exceeds the limit")
            if row["perf/peak_memory"] <= 0 or row["perf/peak_memory"] > memory_limit:
                failures.append(f"qualification step {row['step']} memory exceeds the limit")
            if not recovered and (
                row["perf/forward_backward_peak_memory"] <= 0
                or row["perf/forward_backward_peak_memory"] > row["perf/peak_memory"]
            ):
                failures.append(
                    f"qualification step {row['step']} forward/backward memory is invalid"
                )
            input_sha256 = row.get("input_sha256")
            if recovered and input_sha256 is None:
                continue
            if re.fullmatch(r"[0-9a-f]{64}", str(input_sha256)) is None:
                failures.append(f"qualification step {row['step']} input checksum is invalid")
            elif input_sha256 == previous_input:
                failures.append(f"qualification step {row['step']} reused the previous input")
            previous_input = input_sha256
        if profile_name == "glm52-native-stability20":
            steady_times = sorted(
                float(row["time/forward_backward"])
                for row in records[1:]
                if isinstance(row, dict)
                and isinstance(row.get("time/forward_backward"), (int, float))
                and not isinstance(row.get("time/forward_backward"), bool)
                and math.isfinite(row["time/forward_backward"])
            )
            if len(steady_times) != 19:
                failures.append("qualification steady trainer times are incomplete")
            else:
                median_time = statistics.median(steady_times)
                p90_time = steady_times[math.ceil(0.9 * len(steady_times)) - 1]
                if median_time > 750:
                    failures.append("qualification median trainer time exceeds the limit")
                if p90_time > 1.2 * median_time:
                    failures.append("qualification trainer time is not stable")
            memory_name = (
                "perf/peak_memory"
                if recovered
                else "perf/forward_backward_peak_memory"
            )
            first_memory = [
                float(row[memory_name])
                for row in records[:5]
                if isinstance(row, dict)
                and isinstance(row.get(memory_name), (int, float))
                and not isinstance(row.get(memory_name), bool)
                and math.isfinite(row[memory_name])
            ]
            late_records = records[14:19] if recovered else records[-5:]
            last_memory = [
                float(row[memory_name])
                for row in late_records
                if isinstance(row, dict)
                and isinstance(row.get(memory_name), (int, float))
                and not isinstance(row.get(memory_name), bool)
                and math.isfinite(row[memory_name])
            ]
            if len(first_memory) != 5 or len(last_memory) != 5:
                failures.append("qualification memory trend is incomplete")
            elif max(last_memory) > statistics.median(first_memory) + 0.5:
                failures.append("qualification memory use grew across the run")
            if recovered and recovery.get("step20_total_export_peak_gib") != records[
                -1
            ].get("perf/peak_memory"):
                failures.append("qualification recovery export memory differs")
    runtime_ranks = result.get("runtime_ranks")
    expected_runtime = []
    if isinstance(image_digests, dict) and isinstance(source_files, dict):
        launcher_sha256 = source_files.get("scripts/china/cluster.sh")
        for rank in range(64):
            host = ranked_hosts[rank // 8]
            expected_runtime.append(
                {
                    "global_rank": rank,
                    "local_rank": rank % 8,
                    "host": host,
                    "image_digest": image_digests.get(host),
                    "source_sha256": session.get("source_sha256"),
                    "launcher_sha256": launcher_sha256,
                }
            )
    expected_runtime_sha256 = hashlib.sha256(
        (
            json.dumps(expected_runtime, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()
    if runtime_ranks != expected_runtime or result.get(
        "runtime_sha256"
    ) != expected_runtime_sha256:
        failures.append("qualification runtime rank evidence differs")
    adapter = result.get("adapter")
    if expected_adapter:
        adapter_root = (
            PurePosixPath(profile["output_dir"])
            / "run_trainer_only"
            / "broadcasts"
            / f"step_{expected_steps}"
        )
        expected_paths = {
            "path": str(adapter_root / "adapter_model.safetensors"),
            "config_path": str(adapter_root / "adapter_config.json"),
            "stable_path": str(adapter_root / "STABLE"),
            "checksum_path": str(adapter_root / "adapter_model.safetensors.sha256"),
        }
        layout = adapter.get("layout") if isinstance(adapter, dict) else None
        if (
            not isinstance(adapter, dict)
            or adapter.get("step") != expected_steps
            or re.fullmatch(r"[0-9a-f]{64}", str(adapter.get("sha256"))) is None
            or not isinstance(adapter.get("size"), int)
            or not isinstance(layout, dict)
            or set(layout) != {"header_bytes", "tensor_bytes", "file_bytes"}
            or any(
                isinstance(layout.get(name), bool)
                or not isinstance(layout.get(name), int)
                or layout[name] <= 0
                for name in layout
            )
            or layout["file_bytes"]
            != 8 + layout["header_bytes"] + layout["tensor_bytes"]
            or adapter["size"] != layout["file_bytes"]
        ):
            failures.append("qualification final adapter evidence is invalid")
        elif any(adapter.get(name) != path for name, path in expected_paths.items()):
            failures.append("qualification final adapter path differs")
        elif any(
            not isinstance(adapter.get(path_name), str)
            or not isinstance(adapter.get(digest_name), str)
            or re.fullmatch(r"[0-9a-f]{64}", adapter[digest_name]) is None
            for path_name, digest_name in (
                ("config_path", "config_sha256"),
                ("stable_path", "stable_sha256"),
                ("checksum_path", "checksum_sha256"),
            )
        ):
            failures.append("qualification adapter manifest is incomplete")
        adapter_config = adapter.get("config") if isinstance(adapter, dict) else None
        adapter_targets = (
            adapter_config.get("target_modules")
            if isinstance(adapter_config, dict)
            else None
        )
        if (
            not isinstance(adapter_config, dict)
            or isinstance(adapter_config.get("r"), bool)
            or adapter_config.get("r") != 32
            or isinstance(adapter_config.get("lora_alpha"), bool)
            or adapter_config.get("lora_alpha") != 64
            or isinstance(adapter_config.get("lora_dropout"), bool)
            or adapter_config.get("lora_dropout") != 0.0
            or not isinstance(adapter_targets, list)
            or any(not isinstance(target, str) for target in adapter_targets)
            or len(adapter_targets) != len(qualification_adapter_targets)
            or set(adapter_targets) != qualification_adapter_targets
        ):
            failures.append("qualification adapter configuration differs")
        inventory = adapter.get("inventory") if isinstance(adapter, dict) else None
        if (
            not isinstance(inventory, dict)
            or inventory.get("leaf_targets") != sorted(qualification_adapter_targets)
            or inventory.get("tensor_count") != 116_448
            or inventory.get("attention_tensors") != 780
            or inventory.get("dense_mlp_tensors") != 18
            or inventory.get("shared_expert_tensors") != 450
            or inventory.get("routed_expert_tensors") != 115_200
            or re.fullmatch(r"[0-9a-f]{64}", str(inventory.get("key_sha256")))
            is None
        ):
            failures.append("qualification adapter tensor coverage differs")
        if isinstance(adapter, dict) and adapter.get("checksum_value") != adapter.get(
            "sha256"
        ):
            failures.append("qualification adapter checksum content differs")
    elif adapter is not None:
        failures.append("batch qualification contains an unexpected adapter")
    current = now or _dt.datetime.now(_dt.timezone.utc)
    try:
        created = _dt.datetime.fromisoformat(session["created_utc"])
        completed = _dt.datetime.fromisoformat(result["completed_utc"])
        if created.tzinfo is None or completed.tzinfo is None:
            raise ValueError("timezone is missing")
        if (
            not created <= completed <= current
            or (current - created).total_seconds() > age_limit
            or (current - completed).total_seconds() > age_limit
            or (completed - created).total_seconds() > session_duration_limit
        ):
            failures.append("qualification evidence is stale")
    except (KeyError, TypeError, ValueError):
        failures.append("qualification evidence time is invalid")
    return failures


def _parse_qualification_file_stats(
    output: str, expected: set[str]
) -> dict[str, tuple[int, int]]:
    stats = {}
    for line in output.splitlines():
        fields = line.rsplit("|", 2)
        if len(fields) != 3 or fields[0] not in expected or fields[0] in stats:
            raise Refused("qualification recovery file identity is invalid")
        try:
            stats[fields[0]] = (int(fields[1]), int(fields[2]))
        except ValueError as error:
            raise Refused("qualification recovery file identity is invalid") from error
    if set(stats) != expected:
        raise Refused("qualification recovery file identity is incomplete")
    return stats


def _qualification_recovery_completed_epoch(
    stats: dict[str, tuple[int, int]], paths: dict[str, str], created_epoch: int
) -> int:
    if any(mtime < created_epoch for _size, mtime in stats.values()):
        raise Refused("qualification recovery file predates its session")
    adapter_epoch = stats[paths["adapter"]][1]
    config_epoch = stats[paths["config"]][1]
    checksum_epoch = stats[paths["checksum"]][1]
    stable_epoch = stats[paths["stable"]][1]
    metrics_epoch = stats[paths["metrics"]][1]
    if checksum_epoch > stable_epoch or stable_epoch - checksum_epoch > 1:
        raise Refused("qualification recovery completion markers differ")
    if adapter_epoch > checksum_epoch or config_epoch > checksum_epoch:
        raise Refused("qualification recovery adapter is newer than its checksum")
    if metrics_epoch > stable_epoch + 2:
        raise Refused("qualification recovery monitor flush is too late")
    return max(mtime for _size, mtime in stats.values())


def _qualification_metrics_from_gzip(archive: bytes, maximum: int) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive)) as stream:
            metrics = stream.read(maximum + 1)
    except (EOFError, OSError) as error:
        raise Refused("qualification metric archive is invalid") from error
    if len(metrics) > maximum:
        raise Refused("qualification metric archive is too large")
    return metrics


def _qualification_recovery_steps(
    metrics: str,
    created_time: float,
    completed_epoch: int,
    expected_steps: int = 20,
) -> list[dict[str, float | int]]:
    required = (
        "data/rollout_rows",
        "loss/mean",
        "optim/grad_norm",
        "optim/lr",
        "perf/peak_memory",
        "time/forward_backward",
    )
    merged = {step: {"step": step} for step in range(1, expected_steps + 1)}
    selected_steps = set()
    for line_number, line in enumerate(metrics.splitlines(), 1):
        try:
            row = json.loads(line)
        except (TypeError, ValueError) as error:
            raise Refused(
                f"qualification metric JSON is invalid on line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise Refused(f"qualification metric line {line_number} is not an object")
        timestamp = row.get("time")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
        ):
            raise Refused(f"qualification metric line {line_number} has no valid time")
        if not created_time <= timestamp < completed_epoch + 1:
            continue
        step = row.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or int(step) != step
            or int(step) not in merged
        ):
            raise Refused("qualification recovery contains an unexpected step")
        step = int(step)
        selected_steps.add(step)
        for name in required:
            if name not in row:
                continue
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise Refused(
                    f"qualification recovery step {step} has invalid {name}"
                )
            if name in merged[step]:
                raise Refused(
                    f"qualification recovery step {step} repeats {name}"
                )
            merged[step][name] = value
    if selected_steps != set(merged):
        raise Refused("qualification recovery does not contain exact steps 1-20")
    for step, row in merged.items():
        missing = set(required) - set(row)
        if missing:
            raise Refused(
                f"qualification recovery step {step} is missing {sorted(missing)[0]}"
            )
    return list(merged.values())


def finalize_glm52_stability20(source_sha256: str) -> dict:
    """Recover the result file after the pinned leaf-target metadata failure."""
    with (repo_root / production_stability20_profile).open("rb") as stream:
        profile = tomllib.load(stream)
    output_dir = profile["output_dir"].rstrip("/")
    result_path = f"{output_dir}/qualification-result.json"
    session = _remote_json(
        trainer_head, f"{output_dir}/qualification-session.json"
    )
    if source_sha256 != session.get("source_sha256"):
        raise Refused("qualification recovery source confirmation differs")
    mounted = sorted(_mounts_in(repo_root / "scripts/china/cluster.sh"))
    expected_sources = {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in mounted
    }
    expected_sources[qualification_train_source] = legacy_stability20_train_sha256
    expected_sources["scripts/china/cluster.sh"] = hashlib.sha256(
        (repo_root / "scripts/china/cluster.sh").read_bytes()
    ).hexdigest()
    if session.get("source_files") != expected_sources:
        raise Refused("qualification recovery is not from the exact legacy source")
    code, state = _ssh(
        trainer_head,
        f"if test -e {shlex.quote(result_path)}; then echo EXISTS; else echo MISSING; fi",
        timeout=60,
    )
    if code or state.strip() != "MISSING":
        raise Refused("qualification result already exists or cannot be checked")

    adapter_root = PurePosixPath(output_dir) / "run_trainer_only/broadcasts/step_20"
    paths = {
        "metrics": str(PurePosixPath(output_dir) / profile["file_monitor"]["filename"]),
        "metrics_archive": str(
            PurePosixPath(output_dir) / f"{profile['file_monitor']['filename']}.gz"
        ),
        "adapter": str(adapter_root / "adapter_model.safetensors"),
        "config": str(adapter_root / "adapter_config.json"),
        "stable": str(adapter_root / "STABLE"),
        "checksum": str(adapter_root / "adapter_model.safetensors.sha256"),
    }
    code, raw_state = _ssh(
        trainer_head,
        f"if test -L {shlex.quote(paths['metrics'])}; then echo INVALID; "
        f"elif test -f {shlex.quote(paths['metrics'])}; then echo FILE; "
        f"elif test -e {shlex.quote(paths['metrics'])}; then echo INVALID; "
        "else echo MISSING; fi",
        timeout=60,
    )
    if code or raw_state.strip() not in {"FILE", "MISSING"}:
        raise Refused("qualification recovery metric file identity is invalid")
    restore_metrics = raw_state.strip() == "MISSING"
    timed_paths = dict(paths)
    timed_paths["metrics"] = paths["metrics_archive"]
    stat_paths = {path for name, path in timed_paths.items() if name != "metrics_archive"}
    stat_command = "stat -c '%n|%s|%Y' -- " + " ".join(
        shlex.quote(path) for path in sorted(stat_paths)
    )
    code, stat_output = _ssh(trainer_head, stat_command, timeout=60)
    if code:
        raise Refused("qualification recovery files are incomplete")
    stats = _parse_qualification_file_stats(stat_output, stat_paths)
    try:
        created = _dt.datetime.fromisoformat(session["created_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise Refused("qualification recovery session time is invalid") from error
    if created.tzinfo is None:
        raise Refused("qualification recovery session time is invalid")
    created_epoch = int(created.timestamp())
    completed_epoch = _qualification_recovery_completed_epoch(
        stats, timed_paths, created_epoch
    )

    archive_bytes = _remote_file_bytes(
        trainer_head, paths["metrics_archive"], 4 * 1024 * 1024
    )
    if len(archive_bytes) != stats[paths["metrics_archive"]][0]:
        raise Refused("qualification metric archive size differs")
    metrics_bytes = _qualification_metrics_from_gzip(
        archive_bytes, 16 * 1024 * 1024
    )
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    metrics_sha256 = hashlib.sha256(metrics_bytes).hexdigest()
    if not restore_metrics:
        current_metrics = _remote_file_bytes(
            trainer_head, paths["metrics"], 16 * 1024 * 1024
        )
        if current_metrics != metrics_bytes:
            raise Refused("qualification metric file differs from its archive")
    try:
        metrics_text = metrics_bytes.decode()
    except UnicodeError as error:
        raise Refused("qualification metric file is not UTF-8") from error
    steps = _qualification_recovery_steps(
        metrics_text,
        created.timestamp(),
        completed_epoch,
    )

    adapter_config = _remote_json(trainer_head, paths["config"])
    adapter_header, adapter_header_size = _remote_safetensors_header_and_size(
        trainer_head, paths["adapter"]
    )
    inventory = _glm52_adapter_inventory(adapter_header)
    adapter_layout = _safetensors_layout(adapter_header, adapter_header_size)
    if stats[paths["adapter"]][0] != adapter_layout["file_bytes"]:
        raise Refused("qualification adapter file size differs from its header")
    adapter_sha256 = _remote_sha256(paths["adapter"])
    checksum = _remote_file_bytes(trainer_head, paths["checksum"], 1024)
    if checksum != f"{adapter_sha256}\n".encode():
        raise Refused("qualification recovery adapter checksum differs")
    runtime_ranks = [
        {
            "global_rank": rank,
            "local_rank": rank % 8,
            "host": session["trainer_hosts"][rank // 8],
            "image_digest": session["image_digests"][
                session["trainer_hosts"][rank // 8]
            ],
            "source_sha256": session["source_sha256"],
            "launcher_sha256": session["source_files"]["scripts/china/cluster.sh"],
        }
        for rank in range(64)
    ]
    result = {
        "schema_version": 1,
        "kind": "glm52_trainer_qualification",
        "session_id": session["session_id"],
        "profile": session["profile"],
        "session_sha256": _canonical_json_sha256(session),
        "config_sha256": session["config_sha256"],
        "source_sha256": session["source_sha256"],
        "trainer_head": session["trainer_head"],
        "trainer_head_image_digest": session["image_digests"][trainer_head],
        "completed_utc": _dt.datetime.fromtimestamp(
            completed_epoch, _dt.timezone.utc
        ).isoformat(),
        "topology": session["topology"],
        "fake_data": {
            "batch_size": 64,
            "samples_per_micro_batch": 32,
            "group_size": 16,
            "sequence_length": 32768,
            "rollout_cache": False,
        },
        "lora": {
            "rank": 32,
            "alpha": 64,
            "targets": sorted(profile["model"]["lora"]["target_modules"]),
        },
        "max_steps": 20,
        "steps": steps,
        "runtime_ranks": runtime_ranks,
        "runtime_sha256": hashlib.sha256(
            (json.dumps(runtime_ranks, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "metrics_path": paths["metrics"],
        "metrics_sha256": metrics_sha256,
        "adapter": {
            "path": paths["adapter"],
            "sha256": adapter_sha256,
            "size": stats[paths["adapter"]][0],
            "step": 20,
            "config_path": paths["config"],
            "config_sha256": _remote_sha256(paths["config"]),
            "stable_path": paths["stable"],
            "stable_sha256": _remote_sha256(paths["stable"]),
            "checksum_path": paths["checksum"],
            "checksum_sha256": _remote_sha256(paths["checksum"]),
            "checksum_value": adapter_sha256,
            "config": adapter_config,
            "inventory": inventory,
            "layout": adapter_layout,
        },
        "recovery": {
            "reason": "adapter_leaf_target_metadata",
            "original_train_sha256": legacy_stability20_train_sha256,
            "input_checks": "executed_not_persisted",
            "runtime_records": "derived_from_session",
            "finalizer_sha256": _sha256_file(Path(__file__), "training operations source"),
            "metrics_archive_path": paths["metrics_archive"],
            "metrics_archive_sha256": archive_sha256,
            "forward_backward_memory": "steps_1_19_total_without_adapter_export",
            "step20_forward_backward_peak": "not_persisted",
            "step20_total_export_peak_gib": steps[-1]["perf/peak_memory"],
        },
    }
    failures = _validate_trainer_qualification(
        session,
        result,
        profile,
        "glm52-native-stability20",
    )
    if failures:
        raise Refused("qualification recovery failed: " + "; ".join(failures))
    if restore_metrics:
        _restore_remote_qualification_metrics(
            paths["metrics"],
            paths["metrics_archive"],
            archive_sha256,
            len(archive_bytes),
            metrics_sha256,
            len(metrics_bytes),
        )
    if _remote_sha256(paths["metrics"]) != metrics_sha256:
        raise RuntimeError("qualification recovery metric file differs after restore")
    _write_remote_qualification_result(result_path, result)
    if _remote_json(trainer_head, result_path) != result:
        raise RuntimeError("qualification recovery result differs after write")
    return result


def _validate_step20_h200_gate(
    result: dict,
    profile: dict,
    stability: dict,
    parity_report: dict,
    capture: dict,
    *,
    qualification_result_sha256: str,
    capture_manifest_sha256: str,
    expected_engine_digest: str,
    now: _dt.datetime | None = None,
) -> list[str]:
    failures: list[str] = []
    pinned_profile = {
        "schema_version": 1,
        "name": "glm52-step20-h200",
        "output_path": production_step20_h200_result,
        "model": "glm52-step20-r32",
        "engine_config": "configs/qualification/glm52-policy-parity-h200.toml",
        "adapter_path": glm_step20_h200_gate["adapter"],
        "requests": 1024,
        "group_size": 16,
        "max_inflight": 128,
        "context_length": 32768,
        "max_output_tokens": 16384,
        "min_output_tokens": 16384,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 20260824,
        "request_timeout_seconds": 700,
        "max_retries": 2,
        "retry_count_max": 32,
        "qualification_deadline_seconds": 720,
        "tensor_parallel_size": 8,
        "cuda_graph": True,
        "cuda_graph_capture_size": 256,
        "prefix_cache": True,
        "kv_cache_dtype": "auto",
        "max_num_seqs": 128,
        "max_num_batched_tokens": 32769,
        "batch_seconds_min": 480,
        "batch_seconds_max": 720,
        "peak_vram_mib_max": 143000,
        "evidence_max_age_seconds": 86400,
    }
    if profile != pinned_profile:
        failures.append("step-20 H200 gate profile differs from production shape")
    expected_profile_sha256 = hashlib.sha256(
        (repo_root / production_step20_h200_profile).read_bytes()
    ).hexdigest()
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "glm52_step20_h200_gate"
        or result.get("profile_sha256") != expected_profile_sha256
    ):
        failures.append("step-20 H200 gate schema or profile differs")
    expected_settings = {
        name: profile[name]
        for name in (
            "requests",
            "group_size",
            "max_inflight",
            "context_length",
            "max_output_tokens",
            "min_output_tokens",
            "temperature",
            "top_p",
            "seed",
            "request_timeout_seconds",
            "max_retries",
            "retry_count_max",
            "qualification_deadline_seconds",
            "tensor_parallel_size",
            "cuda_graph",
            "cuda_graph_capture_size",
            "prefix_cache",
            "kv_cache_dtype",
            "max_num_seqs",
            "max_num_batched_tokens",
            "batch_seconds_min",
            "batch_seconds_max",
            "peak_vram_mib_max",
            "evidence_max_age_seconds",
        )
    }
    if result.get("settings") != expected_settings:
        failures.append("step-20 H200 gate settings differ")

    stability_adapter = stability.get("adapter")
    adapter = result.get("adapter")
    adapter_fields = {
        "trainer_path": "path",
        "sha256": "sha256",
        "size": "size",
        "config_sha256": "config_sha256",
        "stable_sha256": "stable_sha256",
        "checksum_sha256": "checksum_sha256",
    }
    if not isinstance(adapter, dict) or not isinstance(stability_adapter, dict):
        failures.append("step-20 H200 gate adapter evidence is missing")
        adapter = {}
        stability_adapter = {}
    elif (
        adapter.get("name") != profile.get("model")
        or adapter.get("h200_path") != glm_step20_h200_gate["adapter"]
        or adapter.get("qualification_result_sha256")
        != qualification_result_sha256
        or adapter.get("qualification_session_sha256")
        != stability.get("session_sha256")
        or any(
            adapter.get(gate_name) != stability_adapter.get(stability_name)
            for gate_name, stability_name in adapter_fields.items()
        )
    ):
        failures.append("step-20 H200 gate adapter identity differs")

    engine = result.get("engine")
    if (
        not isinstance(engine, dict)
        or engine.get("config") != profile.get("engine_config")
        or engine.get("config_sha256")
        != hashlib.sha256(
            (repo_root / str(profile.get("engine_config"))).read_bytes()
        ).hexdigest()
        or engine.get("image_digest") != expected_engine_digest
        or any(
            engine.get(name) != profile.get(name)
            for name in (
                "tensor_parallel_size",
                "cuda_graph",
                "cuda_graph_capture_size",
                "prefix_cache",
                "kv_cache_dtype",
                "max_num_seqs",
                "max_num_batched_tokens",
            )
        )
    ):
        failures.append("step-20 H200 gate engine identity differs")

    measurements = result.get("measurements")
    if not isinstance(measurements, dict):
        failures.append("step-20 H200 gate measurements are missing")
    else:
        batch_seconds = measurements.get("batch_seconds")
        output_rate = measurements.get("output_tokens_per_second")
        latency = measurements.get("latency_seconds")
        peak = measurements.get("peak_vram_mib")
        retries = measurements.get("retries")
        output_tokens = measurements.get("output_tokens")
        max_prompt_tokens = measurements.get("max_prompt_tokens")
        if (
            isinstance(batch_seconds, bool)
            or not isinstance(batch_seconds, (int, float))
            or not math.isfinite(batch_seconds)
            or not profile["batch_seconds_min"]
            <= batch_seconds
            <= profile["batch_seconds_max"]
        ):
            failures.append("step-20 H200 gate batch time is outside 480-720 seconds")
        if (
            measurements.get("completed_requests") != profile["requests"]
            or measurements.get("failed_requests") != 0
            or measurements.get("first_errors") != []
        ):
            failures.append("step-20 H200 gate request failures differ")
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= profile["retry_count_max"]
        ):
            failures.append("step-20 H200 gate retry count is invalid")
        retry_errors = measurements.get("retry_errors")
        if (
            not isinstance(retry_errors, list)
            or len(retry_errors) != retries
            or any(not isinstance(value, str) or len(value) > 1000 for value in retry_errors)
            or any(
                marker in value.lower()
                for value in retry_errors
                for marker in ("out of memory", "oom", "cuda error")
            )
        ):
            failures.append("step-20 H200 gate retry or OOM evidence is invalid")
        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens != profile["requests"] * profile["max_output_tokens"]
            or isinstance(output_rate, bool)
            or not isinstance(output_rate, (int, float))
            or not math.isfinite(output_rate)
            or output_rate <= 0
        ):
            failures.append("step-20 H200 gate output measurements are invalid")
        elif (
            not isinstance(batch_seconds, bool)
            and isinstance(batch_seconds, (int, float))
            and math.isfinite(batch_seconds)
            and batch_seconds > 0
            and not math.isclose(
                output_rate,
                output_tokens / batch_seconds,
                rel_tol=1e-9,
            )
        ):
            failures.append("step-20 H200 gate throughput arithmetic differs")
        if (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 1
            or max_prompt_tokens + profile["max_output_tokens"]
            > profile["context_length"]
        ):
            failures.append("step-20 H200 gate request context is invalid")
        if (
            not isinstance(latency, dict)
            or any(
                isinstance(latency.get(name), bool)
                or not isinstance(latency.get(name), (int, float))
                or not math.isfinite(latency[name])
                or latency[name] <= 0
                for name in ("mean", "p50", "p95", "p99", "max")
            )
            or not isinstance(peak, list)
            or len(peak) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > profile["peak_vram_mib_max"]
                for value in peak
            )
        ):
            failures.append("step-20 H200 gate latency or VRAM measurements are invalid")
        elif not (
            latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["max"]
            and latency["mean"] <= latency["max"]
        ):
            failures.append("step-20 H200 gate latency order is invalid")

    parity = result.get("parity")
    arrays = capture.get("arrays") if isinstance(capture, dict) else None
    expected_array_hashes = (
        {
            name: arrays.get(name, {}).get("sha256")
            for name in ("full_logprobs", "router_ids", "router_weights")
        }
        if isinstance(arrays, dict)
        else None
    )
    if (
        not isinstance(parity, dict)
        or parity.get("adapter_sha256") != adapter.get("sha256")
        or parity.get("capture_manifest_sha256") != capture_manifest_sha256
        or parity.get("array_sha256") != expected_array_hashes
        or parity.get("graph_eager_guard") != capture.get("graph_eager_guard")
        or capture.get("adapter_sha256") != adapter.get("sha256")
        or capture.get("adapter_loaded") is not True
        or capture.get("adapter_path") != glm_step20_h200_gate["adapter"]
        or parity_report.get("adapter_sha256") != adapter.get("sha256")
        or parity_report.get("h200_capture_manifest_sha256")
        != capture_manifest_sha256
        or parity_report.get("graph_eager_guard")
        != capture.get("graph_eager_guard")
    ):
        failures.append("step-20 H200 gate parity binding differs")

    current = now or _dt.datetime.now(_dt.timezone.utc)
    try:
        timestamps = [
            _dt.datetime.fromisoformat(value["created_utc"])
            for value in (result, parity_report, capture)
        ]
        if any(
            created.tzinfo is None
            or not created <= current
            or (current - created).total_seconds() > profile["evidence_max_age_seconds"]
            for created in timestamps
        ):
            failures.append("step-20 H200 gate or parity evidence is stale")
    except (KeyError, TypeError, ValueError):
        failures.append("step-20 H200 gate or parity evidence time is invalid")
    return failures


def _step20_stability_evidence() -> tuple[dict, str]:
    with (repo_root / production_stability20_profile).open("rb") as stream:
        profile = tomllib.load(stream)
    output_dir = profile["output_dir"].rstrip("/")
    session = _remote_json(
        trainer_head, f"{output_dir}/qualification-session.json"
    )
    result_path = f"{output_dir}/qualification-result.json"
    result = _remote_json(trainer_head, result_path)
    failures = _validate_trainer_qualification(
        session, result, profile, "glm52-native-stability20"
    )
    ranked_hosts = [trainer_head] + [
        host for host in trainer_hosts if host != trainer_head
    ]
    current_sources = _trainer_qualification_source_files(ranked_hosts)
    if not _qualification_sources_match(
        session,
        result,
        profile,
        "glm52-native-stability20",
        current_sources,
    ):
        failures.append("stability20 source differs from current trainer mounts")
    if session.get("image_digests") != _trainer_image_digest_map(ranked_hosts):
        failures.append("stability20 images differ from current trainer images")
    adapter = result.get("adapter")
    if isinstance(adapter, dict) and isinstance(adapter.get("path"), str):
        adapter_header, observed_layout = _remote_safetensors_header_and_layout(
            trainer_head, adapter["path"]
        )
        observed_inventory = _glm52_adapter_inventory(adapter_header)
        if adapter.get("inventory") != observed_inventory:
            failures.append("stability20 adapter tensor coverage differs")
        if adapter.get("layout") != observed_layout:
            failures.append("stability20 adapter file layout differs")
    code, result_sha256 = _ssh(
        trainer_head,
        f"sha256sum {shlex.quote(result_path)} | cut -d' ' -f1",
        timeout=60,
    )
    if code or re.fullmatch(r"[0-9a-f]{64}", result_sha256.strip()) is None:
        failures.append("stability20 result checksum is invalid")
    if failures:
        raise Refused("step-20 adapter evidence failed: " + "; ".join(failures))
    return result, result_sha256.strip()


def _stage_step20_adapter_on_h200(adapter: dict, result_sha256: str) -> str:
    qualification_result = glm_step20_h200_gate["qualification_result"]
    if qualification_result != (
        glm_step20_h200_gate["adapter"] + "/.qualification-result.json"
    ) or re.fullmatch(r"[0-9a-f]{64}", result_sha256) is None:
        raise Refused("step-20 qualification result destination is not exact")
    source_paths = {
        "adapter_model.safetensors": (adapter.get("path"), adapter.get("sha256")),
        "adapter_config.json": (adapter.get("config_path"), adapter.get("config_sha256")),
        "STABLE": (adapter.get("stable_path"), adapter.get("stable_sha256")),
        "adapter_model.safetensors.sha256": (
            adapter.get("checksum_path"),
            adapter.get("checksum_sha256"),
        ),
        ".qualification-result.json": (
            str(
                PurePosixPath(
                    tomllib.loads(
                        (repo_root / production_stability20_profile).read_text()
                    )["output_dir"]
                )
                / "qualification-result.json"
            ),
            result_sha256,
        ),
    }
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for path, digest in source_paths.values()
    ):
        raise Refused("step-20 adapter source identity is invalid")
    adapter_size = adapter.get("size")
    if (
        isinstance(adapter_size, bool)
        or not isinstance(adapter_size, int)
        or adapter_size <= 0
    ):
        raise Refused("step-20 adapter source size is invalid")
    final = PurePosixPath(glm_step20_h200_gate["adapter"])
    root = final.parent
    if not final.is_absolute() or ".." in final.parts or final.name != "adapter-step20":
        raise Refused("step-20 H200 adapter destination is not exact")
    nonce = secrets.token_hex(8)
    candidate = root / f".adapter-candidate-{nonce}"
    host = inference_hosts[0]
    served_path = PurePosixPath(
        str(source_paths["adapter_model.safetensors"][0])
    ).relative_to(glm_adapter_root)
    source_url = (
        f"http://127.0.0.1:{glm_adapter_tunnel_ports.start}/{served_path}"
    )
    _ensure_glm_adapter_range_server(
        str(source_paths["adapter_model.safetensors"][0])
    )
    tunnel_records = _start_step20_adapter_tunnels(host, source_url)
    setup = f"""
set -euo pipefail
root={shlex.quote(str(root))}
candidate={shlex.quote(str(candidate))}
if test -e "$root" || test -L "$root"; then
  test -d "$root"
  test ! -L "$root"
  test "$(realpath -e "$root")" = "$root"
  test "$(stat -c '%u:%g:%a' "$root")" = "1000:1000:755"
else
  sudo install -d -o 1000 -g 1000 -m 0755 "$root"
fi
test ! -e "$candidate"
test ! -L "$candidate"
install -d -m 0750 "$candidate"
"""
    try:
        _remote_ok(host, setup, "step-20 H200 adapter candidate", timeout=60)
    except BaseException as error:
        _cleanup_failed_step20_adapter_stage(
            error, host, candidate, tunnel_records
        )
        raise
    transfer_started = time.monotonic()
    try:
        transfer = _download_glm_adapter_direct(
            host,
            str(source_paths["adapter_model.safetensors"][0]),
            str(candidate / "adapter_model.safetensors"),
            str(source_paths["adapter_model.safetensors"][1]),
            adapter_size,
        )
    except BaseException as error:
        _cleanup_failed_step20_adapter_stage(
            error, host, candidate, tunnel_records
        )
        raise
    print(
        "Staged adapter weights directly: "
        f"{transfer['bytes'] / (1024**3):.2f} GiB in "
        f"{transfer['seconds']:.1f} seconds "
        f"({transfer['bytes'] / (1024**3) / transfer['seconds']:.3f} GiB/s)."
    )
    metadata_started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="glm52-step20-adapter-") as temporary:
            local_root = Path(temporary)
            for name, (source, digest) in source_paths.items():
                if name == "adapter_model.safetensors":
                    continue
                local = local_root / name
                _download_adapter_file(str(source), local)
                if _sha256_file(local, name) != digest:
                    raise RuntimeError(
                        f"step-20 adapter download checksum differs: {name}"
                    )
                _local_command(
                    [
                        "scp",
                        "-q",
                        "-o",
                        "BatchMode=yes",
                        str(local),
                        f"{host}:{candidate}/{name}",
                    ],
                    f"stage step-20 adapter {name}",
                    timeout=1800,
                )
    except BaseException as error:
        _cleanup_failed_step20_adapter_stage(
            error, host, candidate, tunnel_records
        )
        raise
    print(
        f"Staged {len(source_paths) - 1} metadata files in "
        f"{time.monotonic() - metadata_started:.1f} seconds; "
        f"total staging time {time.monotonic() - transfer_started:.1f} seconds."
    )
    encoded = base64.b64encode(
        json.dumps(
            {name: digest for name, (_path, digest) in source_paths.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).decode()
    previous = f".adapter-previous-{nonce}"
    install = f"""sudo python3 - {shlex.quote(str(root))} {shlex.quote(candidate.name)} {shlex.quote(final.name)} {shlex.quote(previous)} {shlex.quote(encoded)} <<'PY'
import base64,hashlib,json,os,stat,sys
root,candidate_name,final_name,previous_name,encoded=sys.argv[1:]
expected=json.loads(base64.b64decode(encoded,validate=True))
directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
root_info=os.fstat(directory)
if (root_info.st_uid,root_info.st_gid,stat.S_IMODE(root_info.st_mode))!=(1000,1000,0o755): raise RuntimeError("adapter root identity differs")
def exists(name):
 try: os.stat(name,dir_fd=directory,follow_symlinks=False); return True
 except FileNotFoundError: return False
def validate(name,installed=False,check_hashes=True):
 child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
 try:
  info=os.fstat(child)
  if installed and (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode))!=(0,0,0o755): raise RuntimeError("adapter directory identity differs")
  if set(os.listdir(child))!=set(expected): raise RuntimeError("adapter file set differs")
  for filename,wanted in expected.items():
   fd=os.open(filename,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=child)
   try:
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1: raise RuntimeError("adapter file identity differs")
    if installed and (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode))!=(0,0,0o644): raise RuntimeError("adapter file owner or mode differs")
    digest=hashlib.sha256()
    while block:=os.read(fd,1048576): digest.update(block)
    if check_hashes and digest.hexdigest()!=wanted: raise RuntimeError("adapter file checksum differs: "+filename)
   finally: os.close(fd)
 finally: os.close(child)
def prepare_candidate():
 child=os.open(candidate_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
 try:
  for filename in expected:
   fd=os.open(filename,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=child)
   try: os.fchown(fd,0,0); os.fchmod(fd,0o644)
   finally: os.close(fd)
  os.fchown(child,0,0); os.fchmod(child,0o755)
 finally: os.close(child)
def remove_exact(name):
 child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
 try:
  if set(os.listdir(child))!=set(expected): raise RuntimeError("adapter cleanup file set differs")
  for filename in expected: os.unlink(filename,dir_fd=child)
 finally: os.close(child)
 os.rmdir(name,dir_fd=directory)
try:
 validate(candidate_name)
 if exists(previous_name): raise RuntimeError("adapter rollback path already exists")
 try: validate(final_name,True)
 except FileNotFoundError:
  prepare_candidate()
  os.rename(candidate_name,final_name,src_dir_fd=directory,dst_dir_fd=directory)
  os.fsync(directory)
  try:
   validate(final_name,True)
  except BaseException:
   os.rename(final_name,candidate_name,src_dir_fd=directory,dst_dir_fd=directory)
   os.fsync(directory)
   raise
  print("installed")
 except RuntimeError:
  validate(final_name,True,False)
  prepare_candidate()
  os.rename(final_name,previous_name,src_dir_fd=directory,dst_dir_fd=directory)
  try:
   os.rename(candidate_name,final_name,src_dir_fd=directory,dst_dir_fd=directory)
   os.fsync(directory)
   validate(final_name,True)
  except BaseException:
   if exists(final_name): os.rename(final_name,candidate_name,src_dir_fd=directory,dst_dir_fd=directory)
   os.rename(previous_name,final_name,src_dir_fd=directory,dst_dir_fd=directory)
   os.fsync(directory)
   raise
  remove_exact(previous_name)
  os.fsync(directory)
  print("replaced")
 else:
  remove_exact(candidate_name)
  os.fsync(directory)
  print("reused")
finally: os.close(directory)
PY"""
    try:
        state = _remote_ok(
            host, install, "step-20 H200 adapter install", timeout=600
        ).strip()
        if state not in {"installed", "replaced", "reused"}:
            raise RuntimeError("step-20 H200 adapter install returned invalid state")
    except BaseException as error:
        _cleanup_failed_step20_adapter_stage(
            error, host, candidate, tunnel_records
        )
        raise
    _stop_step20_adapter_tunnels(host, tunnel_records)
    return str(final)


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
    current_sources = None
    current_images = None
    parity_report = None
    parity_capture = None
    stability_result = None
    stability_result_sha256 = None
    try:
        ranked_hosts = [trainer_head] + [
            host for host in trainer_hosts if host != trainer_head
        ]
        current_sources = _trainer_qualification_source_files(ranked_hosts)
        if set(current_sources) != _qualification_source_paths():
            failures.append("qualification source manifest differs from launcher mounts")
        current_images = _trainer_image_digest_map(
            ranked_hosts
        )
    except (OSError, Refused, RuntimeError) as error:
        failures.append(str(error))
    try:
        report = _remote_json(inference_host, production_parity_report)
        parity_report = report
        parity_capture = _remote_json(
            inference_host, f"{glm_parity_paths['h200_capture']}/manifest.json"
        )
        fixture_manifest = _remote_json(
            inference_host, f"{production_parity_fixture}/manifest.json"
        )
        with (repo_root / production_parity_profile).open("rb") as stream:
            parity_profile = tomllib.load(stream)
        failures.extend(_validate_remote_fixture(inference_host, fixture_manifest))
        failures.extend(_validate_live_parity_report(report, fixture_manifest, parity_profile))
    except (OSError, Refused, tomllib.TOMLDecodeError) as error:
        failures.append(str(error))

    for profile_name, relative in (
        ("glm52-native-b2048", production_batch2048_profile),
        ("glm52-native-stability20", production_stability20_profile),
    ):
        try:
            with (repo_root / relative).open("rb") as stream:
                profile = tomllib.load(stream)
            output_dir = profile["output_dir"].rstrip("/")
            session = _remote_json(
                trainer_head, f"{output_dir}/qualification-session.json"
            )
            result = _remote_json(
                trainer_head, f"{output_dir}/qualification-result.json"
            )
            failures.extend(
                f"{profile_name}: {failure}"
                for failure in _validate_trainer_qualification(
                    session, result, profile, profile_name
                )
            )
            if current_sources is not None and not _qualification_sources_match(
                session, result, profile, profile_name, current_sources
            ):
                failures.append(f"{profile_name}: qualification source differs from current mounts")
            if current_images is not None and session.get("image_digests") != current_images:
                failures.append(f"{profile_name}: qualification images differ from current images")
            metrics_path = str(
                PurePosixPath(output_dir) / profile["file_monitor"]["filename"]
            )
            files = [(metrics_path, result.get("metrics_sha256"))]
            recovery = result.get("recovery")
            if isinstance(recovery, dict):
                files.append(
                    (
                        f"{metrics_path}.gz",
                        recovery.get("metrics_archive_sha256"),
                    )
                )
            adapter = result.get("adapter")
            if isinstance(adapter, dict):
                adapter_dir = (
                    PurePosixPath(output_dir)
                    / "run_trainer_only"
                    / "broadcasts"
                    / "step_20"
                )
                files.append(
                    (
                        str(adapter_dir / "adapter_model.safetensors"),
                        adapter.get("sha256"),
                    )
                )
                files.extend(
                    (str(adapter_dir / filename), adapter.get(digest_name))
                    for filename, digest_name in (
                        ("adapter_config.json", "config_sha256"),
                        ("STABLE", "stable_sha256"),
                        (
                            "adapter_model.safetensors.sha256",
                            "checksum_sha256",
                        ),
                    )
                )
            for path, expected in files:
                if (
                    not isinstance(path, str)
                    or not isinstance(expected, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                ):
                    failures.append(f"{profile_name}: evidence file identity is invalid")
                    continue
                code, observed = _ssh(
                    trainer_head,
                    f"test -f {shlex.quote(path)} && sha256sum {shlex.quote(path)} | cut -d' ' -f1",
                    timeout=60,
                )
                if code or observed.strip() != expected:
                    failures.append(f"{profile_name}: evidence file checksum differs: {path}")
            if profile_name == "glm52-native-stability20":
                stability_result = result
                code, observed_result_sha256 = _ssh(
                    trainer_head,
                    f"sha256sum {shlex.quote(f'{output_dir}/qualification-result.json')} | cut -d' ' -f1",
                    timeout=60,
                )
                if code or re.fullmatch(
                    r"[0-9a-f]{64}", observed_result_sha256.strip()
                ) is None:
                    failures.append(
                        f"{profile_name}: qualification result checksum is invalid"
                    )
                else:
                    stability_result_sha256 = observed_result_sha256.strip()
                adapter_dir = (
                    PurePosixPath(output_dir)
                    / "run_trainer_only"
                    / "broadcasts"
                    / "step_20"
                )
                adapter_config = _remote_json(
                    trainer_head, str(adapter_dir / "adapter_config.json")
                )
                targets = adapter_config.get("target_modules")
                if (
                    isinstance(adapter_config.get("r"), bool)
                    or isinstance(adapter_config.get("lora_alpha"), bool)
                    or isinstance(adapter_config.get("lora_dropout"), bool)
                    or adapter_config.get("r") != 32
                    or adapter_config.get("lora_alpha") != 64
                    or adapter_config.get("lora_dropout") != 0.0
                    or not isinstance(targets, list)
                    or any(not isinstance(target, str) for target in targets)
                    or len(targets) != len(qualification_adapter_targets)
                    or set(targets) != qualification_adapter_targets
                ):
                    failures.append(
                        f"{profile_name}: adapter configuration differs"
                    )
                adapter_header, observed_layout = (
                    _remote_safetensors_header_and_layout(
                        trainer_head,
                        str(adapter_dir / "adapter_model.safetensors"),
                    )
                )
                observed_inventory = _glm52_adapter_inventory(adapter_header)
                if (
                    not isinstance(adapter, dict)
                    or adapter.get("inventory") != observed_inventory
                    or adapter.get("layout") != observed_layout
                ):
                    failures.append(
                        f"{profile_name}: adapter tensor coverage or layout differs"
                    )
                checksum_path = str(
                    adapter_dir / "adapter_model.safetensors.sha256"
                )
                code, encoded = _ssh(
                    trainer_head,
                    f"base64 -w0 {shlex.quote(checksum_path)}",
                    timeout=60,
                )
                try:
                    checksum_content = base64.b64decode(
                        encoded.strip(), validate=True
                    ).decode()
                except (ValueError, UnicodeError):
                    checksum_content = ""
                adapter_sha256 = (
                    adapter.get("sha256") if isinstance(adapter, dict) else None
                )
                if code or checksum_content != f"{adapter_sha256}\n":
                    failures.append(
                        f"{profile_name}: adapter checksum content differs"
                    )
        except (KeyError, OSError, Refused, RuntimeError, tomllib.TOMLDecodeError) as error:
            failures.append(f"{profile_name}: {error}")
    try:
        adapter = (
            stability_result.get("adapter")
            if isinstance(stability_result, dict)
            else None
        )
        if not isinstance(adapter, dict) or not isinstance(adapter.get("path"), str):
            raise Refused("step-20 adapter server source is missing")
        _check_glm_adapter_range_server(adapter["path"])
    except (OSError, Refused, RuntimeError) as error:
        failures.append(f"GLM adapter server: {error}")
    try:
        if (
            not isinstance(parity_report, dict)
            or not isinstance(parity_capture, dict)
            or not isinstance(stability_result, dict)
            or not isinstance(stability_result_sha256, str)
        ):
            raise Refused("step-20 H200 gate prerequisites are incomplete")
        with (repo_root / production_step20_h200_profile).open("rb") as stream:
            gate_profile = tomllib.load(stream)
        gate_result = _remote_json(inference_host, production_step20_h200_result)
        capture_manifest_path = f"{glm_parity_paths['h200_capture']}/manifest.json"
        code, capture_sha256 = _ssh(
            inference_host,
            f"sha256sum {shlex.quote(capture_manifest_path)} | cut -d' ' -f1",
            timeout=60,
        )
        if code or re.fullmatch(r"[0-9a-f]{64}", capture_sha256.strip()) is None:
            raise Refused("H200 parity capture manifest checksum is invalid")
        code, engine_digest = _ssh(
            inference_host,
            f"sudo docker image inspect {shlex.quote(inference_image)} --format '{{{{.Id}}}}'",
            timeout=60,
        )
        if code or re.fullmatch(r"sha256:[0-9a-f]{64}", engine_digest.strip()) is None:
            raise Refused("H200 gate engine image digest is invalid")
        failures.extend(
            _validate_step20_h200_gate(
                gate_result,
                gate_profile,
                stability_result,
                parity_report,
                parity_capture,
                qualification_result_sha256=stability_result_sha256,
                capture_manifest_sha256=capture_sha256.strip(),
                expected_engine_digest=engine_digest.strip(),
            )
        )
    except (KeyError, OSError, Refused, RuntimeError, tomllib.TOMLDecodeError) as error:
        failures.append(str(error))
    failures.extend(_check_image_prime08(inference_image, "inference"))
    failures.extend(_check_image_prime08(trainer_image, "trainer"))
    if failures:
        raise Refused("GLM production start preflight failed: " + "; ".join(failures))
    result = {
        "parity_report": production_parity_report,
        "batch2048_steps": [1, 2],
        "stability20_steps": list(range(1, 21)),
        "step20_h200_gate": production_step20_h200_result,
        "trainer_mounts": 45,
        "inference_image": inference_image,
        "trainer_image": trainer_image,
        "prime_rl": "0.8",
    }
    print("GLM production start preflight passed: " + json.dumps(result, sort_keys=True))
    return result


def _mounted_on(path: str, host: str, profile_name: str | None = None) -> bool:
    if host in inference_hosts:
        scripts = (
            repo_root / "scripts" / "h200" / "start_orchestrator.sh",
            repo_root / "scripts" / "h200" / "start_inference.sh",
            repo_root / "scripts" / "h200" / "start_muse_inference.sh",
        )
    else:
        scripts = (repo_root / "scripts" / "china" / "cluster.sh",)
    if any(path in _mounts_in(script) for script in scripts):
        return True
    return host in trainer_hosts and path in _QUALIFICATION_ONLY_MOUNTS.get(
        profile_name, ()
    )


def deploy(
    path: str,
    nodes: list[str] | None = None,
    repository: str = "default",
    profile_name: str | None = None,
) -> bool:
    """Push one repository file to its hosts and verify the checksum."""
    repository_root = _repository_root(repository)
    if nodes is None:
        if path.startswith("source/"):
            nodes = [
                host
                for host in dict.fromkeys([*trainer_hosts, *inference_hosts])
                if _mounted_on(path, host, profile_name)
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
        unmounted = [
            host for host in nodes if not _mounted_on(path, host, profile_name)
        ]
        if unmounted:
            raise Refused(
                f"{path} is not mounted by the launcher for {unmounted}, so the "
                "container would ignore it. Add a -v line to the applicable "
                "launcher first, or pick a mounted path. Call mounted_files() "
                "to list the union."
            )
    want = hashlib.sha256(local.read_bytes()).hexdigest()
    remote = f"{repository_root}/{path}"
    remote_mode = "0755" if local.stat().st_mode & 0o111 else "0644"

    def one(h):
        staging = f"/tmp/training-ops-{want}-{Path(path).name}"
        check = (
            f"if test -f {shlex.quote(remote)}; then "
            f"sha256sum {shlex.quote(remote)} | cut -d' ' -f1; "
            f"else printf MISSING; fi"
        )
        rc, observed = _ssh(h, check, timeout=60)
        if rc:
            return f"remote preflight failed: {observed.strip()[:120]}"
        if observed.strip() == want:
            return "ok"
        try:
            p = subprocess.run(
                [
                    "scp",
                    "-q",
                    *ssh_command[1:],
                    *_active_ssh_control_options(h),
                    str(local),
                    f"{h}:{staging}",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "scp failed: timed out after 120 seconds"
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

    with _ssh_control_masters(nodes):
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
    exact_step20 = glm_step20_h200_gate["trainer_adapter"]
    if not path.startswith(f"{run_root}/") and path != exact_step20:
        raise Refused("resume adapter is outside an allowed exact root")
    targets = [host for host in nodes if host != trainer_head]
    if not targets:
        return
    target_addresses = " ".join(trainer_internal_addresses[host] for host in targets)
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
        ("scripts/h200/glm_adapter_tunnels.py", [orchestrator_host]),
        ("configs/operations.toml", [orchestrator_host]),
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
            "PRIME_INT4_COMBINE_ROUNDS=8 PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=1 "
            "PRIME_NCCL_CHANNELS=16"
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


def _trainer_image_digest_map(nodes: list[str]) -> dict[str, str]:
    def read(host: str) -> tuple[int, str]:
        return _ssh(
            host,
            f"sudo docker image inspect {shlex.quote(trainer_image)} --format '{{{{.Id}}}}'",
            timeout=60,
        )

    results = _fan(nodes, read, workers=min(8, len(nodes)))
    digests = {}
    for host in nodes:
        code, text = results.get(host, (1, "missing result"))
        digest = text.strip()
        if code or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise Refused(f"qualification trainer image digest is invalid on {host}")
        digests[host] = digest
    return digests


@dataclass(frozen=True)
class TrainerQualificationIdentity:
    environment: str
    session_id: str
    profile: str
    config_sha256: str
    source_sha256: str
    session_sha256: str
    launcher_sha256: str
    image_digests: tuple[tuple[str, str], ...]


def _prepare_trainer_qualification_session(
    profile: str,
    config: dict,
    config_path: Path,
    nodes: list[str],
    source_files: dict[str, str],
) -> TrainerQualificationIdentity:
    output_dir = config.get("output_dir")
    qualification_root = PurePosixPath("/data/glm52-rl/qualifications")
    output = PurePosixPath(output_dir) if isinstance(output_dir, str) else None
    if output is None or output.parent != qualification_root:
        raise Refused("qualification output directory is not an exact child of its root")
    if (
        set(source_files) != _qualification_source_paths(profile)
        or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for path, digest in source_files.items()
        )
    ):
        raise Refused("qualification source manifest is invalid")

    image_digests = _trainer_image_digest_map(nodes)

    deadline_values = None
    if profile in {
        "glm52-native-b2048",
        "glm52-native-b2048-memory",
        "glm52-native-b2048-nsys",
    }:
        session_deadline, startup_deadline, step_deadline, profiler_deadline = (
            _glm_b2048_deadlines()
        )
        deadline_values = {
            "session_seconds": session_deadline,
            "startup_seconds": startup_deadline,
            "forward_backward_seconds": step_deadline,
            "profiler_finalize_seconds": profiler_deadline,
        }

    source_sha256 = hashlib.sha256(
        (json.dumps(source_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    session_id = secrets.token_hex(16)
    manifest = {
        "schema_version": 1,
        "kind": "glm52_trainer_qualification_session",
        "session_id": session_id,
        "profile": profile,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "source_files": source_files,
        "trainer_hosts": nodes,
        "trainer_head": trainer_head,
        "image_digests": image_digests,
        "topology": {
            "world_size": 64,
            "pp": 8,
            "cp": 8,
            "ep": 8,
            "dp_replicate": 1,
            "dp_shard": 1,
        },
    }
    if deadline_values is not None:
        manifest["deadlines"] = deadline_values
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    session_sha256 = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode()
    session_path = f"{output_dir}/qualification-session.json"
    temporary = f"{output_dir}/.qualification-session-{session_id}.tmp"
    command = f"""set -euo pipefail
sudo install -d -o 1000 -g 1000 -m 0755 {shlex.quote(output_dir)}
sudo python3 - {shlex.quote(output_dir)} {shlex.quote(PurePosixPath(temporary).name)} {shlex.quote(PurePosixPath(session_path).name)} {session_sha256} {shlex.quote(encoded)} <<'PY'
import base64
import hashlib
import os
import stat
import sys

root, temporary, destination, expected, encoded = sys.argv[1:]
root_stat = os.lstat(root)
if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
    raise SystemExit("qualification output root is not a directory")
if os.path.realpath(root) != root:
    raise SystemExit("qualification output root is not canonical")
payload = base64.b64decode(encoded, validate=True)
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("qualification session payload checksum differs")
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or stat.S_IMODE(result.st_mode) != 0o644:
            raise SystemExit("qualification session temporary file is invalid")
    finally:
        os.close(descriptor)
    os.replace(temporary, destination, src_dir_fd=directory, dst_dir_fd=directory)
    os.fsync(directory)
    descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
            raise SystemExit("qualification session file is invalid")
        observed = hashlib.sha256()
        while chunk := os.read(descriptor, 65536):
            observed.update(chunk)
        if observed.hexdigest() != expected:
            raise SystemExit("qualification session file checksum differs")
    finally:
        os.close(descriptor)
finally:
    os.close(directory)
PY
"""
    results = _fan(
        nodes,
        lambda host: _ssh(host, command, timeout=60),
        workers=min(2, len(nodes)),
    )
    failures = []
    for host in nodes:
        code, detail = results.get(host, (1, "missing result"))
        if code:
            failures.append(f"{host}: {detail.strip()[:120]}")
    if failures:
        raise RuntimeError(
            "qualification session write failed: " + "; ".join(failures)
        )
    environment = (
        f"PRIME_QUALIFICATION_SESSION_ID={session_id} "
        f"PRIME_QUALIFICATION_PROFILE={profile} "
        f"PRIME_QUALIFICATION_CONFIG_SHA256={manifest['config_sha256']} "
        f"PRIME_QUALIFICATION_SOURCE_SHA256={source_sha256} "
        f"PRIME_QUALIFICATION_SESSION_SHA256={session_sha256} "
    )
    if deadline_values is not None:
        environment += (
            f"PRIME_QUALIFICATION_SESSION_DEADLINE_SECONDS={deadline_values['session_seconds']} "
            f"PRIME_QUALIFICATION_STARTUP_DEADLINE_SECONDS={deadline_values['startup_seconds']} "
            f"PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS={deadline_values['forward_backward_seconds']} "
            f"PRIME_QUALIFICATION_PROFILER_FINALIZE_DEADLINE_SECONDS={deadline_values['profiler_finalize_seconds']} "
        )
    return TrainerQualificationIdentity(
        environment=environment,
        session_id=session_id,
        profile=profile,
        config_sha256=manifest["config_sha256"],
        source_sha256=source_sha256,
        session_sha256=session_sha256,
        launcher_sha256=source_files["scripts/china/cluster.sh"],
        image_digests=tuple((host, image_digests[host]) for host in nodes),
    )


def qualify(
    name: str,
    mode: str = "cold",
    nodes: list[str] | None = None,
    *,
    resume_adapter: str | None = None,
    resume_step: int | None = None,
) -> None:
    """Start one configured trainer-only qualification."""
    if mode not in ("cold", "restart"):
        raise Refused('qualification mode must be "cold" or "restart"')
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise Refused("qualification name can contain only letters, digits, '_' and '-'")
    if (resume_adapter is None) != (resume_step is None):
        raise Refused("qualification resume adapter and step must be set together")
    nodes = [_check_host(host) for host in (nodes or trainer_hosts)]
    if trainer_head in nodes:
        nodes = [trainer_head] + [host for host in nodes if host != trainer_head]
    with _ssh_control_masters(nodes):
        _qualify_connected(
            name,
            mode,
            nodes,
            resume_adapter=resume_adapter,
            resume_step=resume_step,
        )


def _qualify_connected(
    name: str,
    mode: str,
    nodes: list[str],
    *,
    resume_adapter: str | None,
    resume_step: int | None,
) -> None:
    _assert_live_services_are_down(nodes, include_inference=False)

    selected = _profile(name)
    launch_env = _qualification_launch_env(selected)
    memory_keys = set(launch_env) & _CUDA_MEMORY_CAPTURE_KEYS
    bf16_combine_keys = set(launch_env) & _BF16_COMBINE_ROUNDS_KEYS
    if name == _CUDA_MEMORY_CAPTURE_PROFILE and launch_env != _MEMORY_PROFILE_LAUNCH_ENV:
        raise Refused(
            "CUDA memory profile is not pinned to rank 0 step 1 and BF16 rounds 8"
        )
    if memory_keys and name != _CUDA_MEMORY_CAPTURE_PROFILE:
        raise Refused(
            "CUDA memory capture is only allowed by the b2048 memory profile"
        )
    if name == _BF16_COMBINE_ROUNDS_PROFILE:
        if launch_env.get("PRIME_BF16_COMBINE_ROUNDS") != "8":
            raise Refused(
                "BF16 combine rounds must be pinned to 8 in the b2048 memory profile"
            )
    elif bf16_combine_keys:
        raise Refused(
            "BF16 combine rounds are only allowed by the b2048 memory profile"
        )
    selected_trainer_container = _profile_trainer_container(selected)
    if "orchestrator" in selected:
        raise Refused(f"profile {name!r} is a full run; use start()")
    relative_config = selected["trainer"]
    expected_step20_adapter = glm_step20_h200_gate["trainer_adapter"]
    if resume_adapter is not None and (
        relative_config != "configs/qualification/glm52-policy-parity-trainer.toml"
        or resume_adapter != expected_step20_adapter
        or resume_step != 0
    ):
        raise Refused("only the exact step-20 adapter can seed trainer parity")
    local_config = repo_root / relative_config
    if not local_config.is_file():
        raise Refused(f"missing qualification config {relative_config}")
    repository_root = _profile_trainer_repository(selected)
    with local_config.open("rb") as stream:
        config = tomllib.load(stream)
    nsys_capture = _qualification_nsys_capture(
        name,
        launch_env,
        nodes,
        str(config.get("output_dir", "")),
    )
    strict_qualification = name in {
        "glm52-native-b2048",
        "glm52-native-b2048-memory",
        "glm52-native-b2048-nsys",
        "glm52-native-delta3",
        "glm52-native-stability20",
    }
    if strict_qualification and nodes != [trainer_head] + [
        host for host in trainer_hosts if host != trainer_head
    ]:
        raise Refused("GLM batch and stability qualifications require all trainer hosts")
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

    def trainer_container_state(host: str) -> tuple[int, str]:
        return _ssh(
            host,
            f"sudo docker ps -a --filter name=^{selected_trainer_container}$ "
            "--format '{{.Names}} {{.State}}'",
        )

    container_states = _fan(nodes, trainer_container_state)
    active = []
    stopped = []
    state_errors = []
    for host, (rc, output) in container_states.items():
        if rc:
            state_errors.append(f"{host}: {output.strip()[:240]}")
            continue
        observed = output.strip()
        if not observed:
            continue
        fields = observed.split()
        if len(fields) != 2 or fields[0] != selected_trainer_container:
            state_errors.append(f"{host}: unexpected trainer container record {observed!r}")
        elif fields[1] == "running":
            active.append(host)
        else:
            stopped.append(host)
    if state_errors:
        raise RuntimeError(
            "cannot determine trainer container state: " + "; ".join(state_errors)
        )
    if active and mode == "cold":
        raise Refused(
            "trainer qualification is already active on "
            + ", ".join(active)
            + "; use mode='restart' only when replacing it is intended"
        )

    trainer_mounts = sorted(
        _qualification_mounts(name)
        if strict_qualification
        else _mounts_in(repo_root / "scripts" / "china" / "cluster.sh")
    )
    qualification_identity = None
    qualification_env = ""
    deploy_options = {}
    if repository_root != remote_repo:
        deploy_options["repository"] = "glm52-qualification"
    if name in _QUALIFICATION_ONLY_MOUNTS:
        deploy_options["profile_name"] = name
    for path in ("scripts/china/cluster.sh", *trainer_mounts, relative_config):
        if not deploy(path, nodes=nodes, **deploy_options):
            raise RuntimeError(f"failed to deploy {path}")
    source_files = (
        _trainer_qualification_source_files(
            nodes, profile_name=name, repository_root=repository_root
        )
        if strict_qualification
        else trainer_mount_preflight(
            nodes, profile_name=name, repository_root=repository_root
        )
    )
    if strict_qualification:
        qualification_identity = _prepare_trainer_qualification_session(
            name, config, local_config, nodes, source_files
        )
        qualification_env = qualification_identity.environment

    if resume_adapter is not None:
        stability, _result_sha256 = _step20_stability_evidence()
        expected_digest = _remote_sha256(resume_adapter)
        if expected_digest != stability.get("adapter", {}).get("sha256"):
            raise Refused("trainer parity adapter differs from stability20 evidence")
        observed = _fan(
            nodes,
            lambda host: _ssh(
                host,
                f"sha256sum {shlex.quote(resume_adapter)} 2>/dev/null | cut -d' ' -f1",
                timeout=60,
            ),
        )
        missing = [
            host
            for host, (code, value) in observed.items()
            if code or not value.strip()
        ]
        differing = [
            host
            for host, (code, value) in observed.items()
            if not code and value.strip() and value.strip() != expected_digest
        ]
        if differing:
            raise Refused(f"trainer parity adapter differs on {differing}")
        if missing:
            _copy_resume_adapter(resume_adapter, nodes)
            observed = _fan(
                nodes,
                lambda host: _ssh(
                    host,
                    f"sha256sum {shlex.quote(resume_adapter)} 2>/dev/null | cut -d' ' -f1",
                    timeout=60,
                ),
            )
            incomplete = [
                host
                for host, (code, value) in observed.items()
                if code or value.strip() != expected_digest
            ]
            if incomplete:
                raise Refused(f"trainer parity adapter copy differs on {incomplete}")

    trainer_nodes = " ".join(host.removeprefix("n") for host in nodes)
    env = (
        f"prime_deployment={shlex.quote(repository_root)} "
        f"prime_image={shlex.quote(trainer_image)} "
        f"prime_trainer_container={shlex.quote(selected_trainer_container)} "
        f"trainer_nodes='{trainer_nodes}' RUN={shlex.quote(name)} "
        f"RUN_NAME=run_{shlex.quote(name)} "
        f"TRAINER_CONFIG={repository_root}/{relative_config} "
        f"{qualification_env}"
        "PRIME_EP_PROFILE=0 PRIME_EP_AC_CACHE=0 PRIME_INT4_COMBINE_ROUNDS=1 "
        "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=0 PRIME_NCCL_CHANNELS=16"
    )
    if resume_adapter is not None:
        env += (
            f" PRIME_RL_RESUME_ADAPTER={shlex.quote(resume_adapter)}"
            f" PRIME_RL_RESUME_ADAPTER_STEP={resume_step}"
        )
    if muse:
        env += (
            " TRANSFORMERS_OVERRIDE_DIR=/data/muse-glimmer/transformers"
            " MUSE_PYTHON_OVERRIDE_DIR=/data/muse-glimmer/python-overrides"
        )
    for key, value in launch_env.items():
        env += f" {key}={shlex.quote(value)}"

    containers_to_remove = sorted(set(active + stopped))
    if containers_to_remove:
        _fan(
            containers_to_remove,
            lambda host: _ssh(
                host, f"sudo docker rm -f {selected_trainer_container}", 60
            ),
        )
        time.sleep(20)

    results = _fan(
        nodes,
        lambda host: _ssh(
            host,
            f"{env} bash {repository_root}/scripts/china/cluster.sh trainer",
            300,
        ),
    )
    failed = [host for host, (rc, _out) in results.items() if rc]
    if failed:
        details = "; ".join(
            f"{host}: {results[host][1].strip()[:240]}"
            for host in failed
        )
        raise RuntimeError(f"qualification trainer launch failed on {failed}: {details}")
    if strict_qualification:
        if qualification_identity is None:
            raise RuntimeError("strict qualification launch identity is missing")
        image_digests = dict(qualification_identity.image_digests)
        environment_keys = (
            "PRIME_QUALIFICATION_SESSION_ID",
            "PRIME_QUALIFICATION_PROFILE",
            "PRIME_QUALIFICATION_CONFIG_SHA256",
            "PRIME_QUALIFICATION_SOURCE_SHA256",
            "PRIME_QUALIFICATION_SESSION_SHA256",
            "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256",
            "PRIME_QUALIFICATION_LAUNCHER_SHA256",
            "PRIME_TRAINER_IMAGE_DIGEST",
            "PRIME_TRAINER_HOST",
            "PRIME_TRAINER_HEAD",
            "PRIME_QUALIFICATION_SESSION_DEADLINE_SECONDS",
            "PRIME_QUALIFICATION_STARTUP_DEADLINE_SECONDS",
            "PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS",
            "PRIME_QUALIFICATION_PROFILER_FINALIZE_DEADLINE_SECONDS",
            "PRIME_CUDA_PROFILE",
            "PRIME_CUDA_PROFILE_RANK",
            "PRIME_CUDA_PROFILE_STEPS",
            *launch_env,
        )
        probe_source = """import json
import subprocess
import sys

container, encoded_keys = sys.argv[1:]
keys = json.loads(encoded_keys)
record = json.loads(subprocess.check_output(["sudo", "docker", "inspect", container]))[0]
environment = {}
nsys_environment = {}
for assignment in record["Config"]["Env"]:
    key, separator, value = assignment.partition("=")
    if separator and key in keys:
        environment[key] = value
    if separator and key.startswith("PRIME_NSYS_"):
        nsys_environment[key] = value
print(json.dumps({
    "name": record["Name"],
    "status": record["State"]["Status"],
    "image": record["Config"]["Image"],
    "image_digest": record["Image"],
    "path": record["Path"],
    "command": record["Config"]["Cmd"],
    "environment": environment,
    "nsys_environment": nsys_environment,
}, sort_keys=True, separators=(",", ":")))
"""
        encoded_keys = json.dumps(environment_keys, separators=(",", ":"))

        def trainer_qualification_state(host: str) -> tuple[int, str]:
            return _ssh(
                host,
                f"python3 -c {shlex.quote(probe_source)} "
                f"{shlex.quote(selected_trainer_container)} "
                f"{shlex.quote(encoded_keys)}",
                60,
            )

        observed_states = _fan(nodes, trainer_qualification_state)
        verification_failures = []
        observed_nsys_hosts = []
        for rank, host in enumerate(nodes):
            rc, output = observed_states.get(host, (1, "missing result"))
            try:
                observed = json.loads(output) if rc == 0 else None
            except json.JSONDecodeError:
                observed = None
            if not isinstance(observed, dict):
                verification_failures.append(
                    f"{host}: trainer identity is unreadable"
                )
                continue
            trainer_module = (
                _NSYS_TRAINER_MODULE
                if nsys_capture is not None and host == nsys_capture.host
                else "prime_rl.trainer.rl.train"
            )
            expected_command = [
                "--role=trainer",
                f"--nnodes={len(nodes)}",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                f"--local-addr={trainer_internal_addresses[host]}",
                f"--master-addr={trainer_internal_addresses[trainer_head]}",
                "--master-port=29500",
                f"--log-dir={run_root}/{name}/run_{name}/logs/trainer/torchrun",
                "--tee=3",
                "--redirects=3",
                "--local-ranks-filter=0",
                "-m",
                trainer_module,
                "@",
                f"{repository_root}/{relative_config}",
            ]
            expected_environment = {
                "PRIME_QUALIFICATION_SESSION_ID": qualification_identity.session_id,
                "PRIME_QUALIFICATION_PROFILE": qualification_identity.profile,
                "PRIME_QUALIFICATION_CONFIG_SHA256": qualification_identity.config_sha256,
                "PRIME_QUALIFICATION_SOURCE_SHA256": qualification_identity.source_sha256,
                "PRIME_QUALIFICATION_SESSION_SHA256": qualification_identity.session_sha256,
                "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256": qualification_identity.source_sha256,
                "PRIME_QUALIFICATION_LAUNCHER_SHA256": qualification_identity.launcher_sha256,
                "PRIME_TRAINER_IMAGE_DIGEST": image_digests.get(host),
                "PRIME_TRAINER_HOST": host,
                "PRIME_TRAINER_HEAD": trainer_head,
            }
            if name in {
                "glm52-native-b2048",
                "glm52-native-b2048-memory",
                "glm52-native-b2048-nsys",
            }:
                session_deadline, startup_deadline, step_deadline, profiler_deadline = (
                    _glm_b2048_deadlines()
                )
                expected_environment.update(
                    {
                        "PRIME_QUALIFICATION_SESSION_DEADLINE_SECONDS": str(session_deadline),
                        "PRIME_QUALIFICATION_STARTUP_DEADLINE_SECONDS": str(startup_deadline),
                        "PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS": str(step_deadline),
                        "PRIME_QUALIFICATION_PROFILER_FINALIZE_DEADLINE_SECONDS": str(profiler_deadline),
                    }
                )
            if launch_env and (
                memory_keys
                or (nsys_capture is not None and host == nsys_capture.host)
            ):
                expected_environment.update(launch_env)
            mismatches = []
            if observed.get("name") != f"/{selected_trainer_container}":
                mismatches.append("container name")
            if observed.get("status") != "running":
                mismatches.append("running state")
            if observed.get("image") != trainer_image:
                mismatches.append("container image")
            if observed.get("image_digest") != image_digests.get(host):
                mismatches.append("container image digest")
            if observed.get("path") != "/app/.venv/bin/torchrun":
                mismatches.append("container entrypoint")
            if observed.get("command") != expected_command:
                mismatches.append("trainer command")
            observed_command = observed.get("command")
            if isinstance(observed_command, list):
                try:
                    module_index = observed_command.index("-m") + 1
                    if observed_command[module_index] == _NSYS_TRAINER_MODULE:
                        observed_nsys_hosts.append(host)
                except (ValueError, IndexError):
                    pass
            observed_environment = observed.get("environment")
            if not isinstance(observed_environment, dict):
                mismatches.append("qualification environment")
            else:
                mismatches.extend(
                    key
                    for key, expected in expected_environment.items()
                    if observed_environment.get(key) != expected
                )
            mismatches.extend(
                _qualification_nsys_environment_mismatches(
                    observed.get("nsys_environment"), nsys_capture, host
                )
            )
            if mismatches:
                verification_failures.append(
                    f"{host}: qualification identity mismatch for {', '.join(mismatches)}"
                )
        expected_nsys_hosts = [nsys_capture.host] if nsys_capture is not None else []
        if observed_nsys_hosts != expected_nsys_hosts:
            verification_failures.append(
                "Nsight worker host/count differs: "
                f"expected={expected_nsys_hosts!r} observed={observed_nsys_hosts!r}"
            )
        if verification_failures:
            raise RuntimeError(
                "trainer qualification launch verification failed: "
                + "; ".join(verification_failures)
            )
    print(
        f"started trainer-only qualification {name} on {len(nodes)} nodes; "
        "call status() or watch()"
    )


def _validate_glm_parity_stage(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("GLM parity stage is not a regular directory")
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o750:
        raise RuntimeError("GLM parity stage directory mode differs")
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
    with profile_path.open("rb") as stream:
        profile = tomllib.load(stream)
    source_path = repo_root / "scripts/glm52_policy_parity.py"
    if manifest.get("source_sha256") != hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest():
        raise RuntimeError("GLM parity stage source checksum differs")
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
    try:
        reference = json.loads((path / "trainer-reference.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read GLM parity trainer reference: {error}") from error
    routed = reference.get("routed_experts")
    routed_fields = {
        "schema_version",
        "encoding",
        "dtype",
        "shape",
        "dense_layers",
        "experts",
        "sha256",
        "data",
    }
    expected_shape = [
        profile["context_length"],
        profile["hidden_layers"],
        profile["experts_per_token"],
    ]
    if (
        not isinstance(routed, dict)
        or set(routed) != routed_fields
        or routed.get("schema_version") != 1
        or routed.get("encoding") != "base64"
        or routed.get("dtype") != "uint8"
        or routed.get("shape") != expected_shape
        or routed.get("dense_layers") != profile["dense_layers"]
        or routed.get("experts") != profile["experts"]
        or not isinstance(routed.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", routed["sha256"]) is None
        or not isinstance(routed.get("data"), str)
    ):
        raise RuntimeError("GLM parity trainer reference route schema differs")
    reference_schema = {
        name: routed[name]
        for name in routed_fields - {"data", "sha256"}
    }
    reference_schema["routed_experts_sha256"] = routed["sha256"]
    for reference_name, schema_name in (
        ("capture_router_ids_sha256", "sha256"),
        ("capture_full_logprobs_sha256", "full_logprobs_sha256"),
    ):
        value = reference.get(reference_name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError("GLM parity trainer reference capture schema differs")
        reference_schema[schema_name] = value
    if (
        reference.get("h200_capture_manifest_sha256")
        != manifest["h200_capture_manifest_sha256"]
        or manifest.get("reference_schema") != reference_schema
    ):
        raise RuntimeError("GLM parity stage reference schema differs")
    try:
        session = json.loads(
            (path / "trainer-router-trace/session.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read GLM parity trainer session: {error}") from error
    identity_fields = {
        "fixture_sha256",
        "profile_sha256",
        "model_revision",
        "adapter_sha256",
        "h200_image",
        "trainer_image",
        "token_ids_sha256",
    }
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise RuntimeError("GLM parity stage identity schema differs")
    expected_common = {
        "fixture_sha256": manifest["fixture_sha256"],
        "profile_sha256": manifest["profile_sha256"],
        "model_revision": identity["model_revision"],
        "adapter_sha256": identity["adapter_sha256"],
    }
    if (
        identity["fixture_sha256"] != manifest["fixture_sha256"]
        or identity["profile_sha256"] != manifest["profile_sha256"]
        or reference.get("fixture_sha256") != expected_common["fixture_sha256"]
        or reference.get("profile_sha256") != expected_common["profile_sha256"]
        or reference.get("model_revision") != expected_common["model_revision"]
        or reference.get("adapter_sha256") != expected_common["adapter_sha256"]
        or reference.get("image") != identity["h200_image"]
    ):
        raise RuntimeError("GLM parity trainer reference identity differs")
    expected_session_fields = {
        "schema_version",
        "side",
        "fixture_sha256",
        "token_ids_sha256",
        "model_revision",
        "adapter_sha256",
        "image",
        "expected_tokens",
        "hidden_layers",
        "dense_layers",
        "topk",
        "experts",
        "route_output_scale",
        "weight_sum_tolerance",
    }
    if not isinstance(session, dict) or set(session) != expected_session_fields:
        raise RuntimeError("GLM parity trainer session schema differs")
    if (
        session["schema_version"] != 1
        or session["side"] != "trainer"
        or session["fixture_sha256"] != expected_common["fixture_sha256"]
        or session["model_revision"] != expected_common["model_revision"]
        or session["adapter_sha256"] != expected_common["adapter_sha256"]
        or session["image"] != identity["trainer_image"]
        or session["token_ids_sha256"] != identity["token_ids_sha256"]
        or session["expected_tokens"] != profile["context_length"]
        or session["hidden_layers"] != profile["hidden_layers"]
        or session["dense_layers"] != profile["dense_layers"]
        or session["topk"] != profile["experts_per_token"]
        or session["experts"] != profile["experts"]
        or session["route_output_scale"] != profile["router_scale"]
        or session["weight_sum_tolerance"]
        != profile["max_router_weight_sum_error"]
    ):
        raise RuntimeError("GLM parity trainer session identity differs")
    input_ids = reference.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or len(input_ids) != profile["context_length"]
        or any(isinstance(value, bool) or not isinstance(value, int) for value in input_ids)
    ):
        raise RuntimeError("GLM parity trainer reference token schema differs")
    try:
        token_bytes = struct.pack(f"<{len(input_ids)}q", *input_ids)
    except struct.error as error:
        raise RuntimeError("GLM parity trainer reference token value differs") from error
    if hashlib.sha256(token_bytes).hexdigest() != identity["token_ids_sha256"]:
        raise RuntimeError("GLM parity trainer token identity differs")
    expected_route_bytes = math.prod(expected_shape)
    if len(routed["data"]) != 4 * ((expected_route_bytes + 2) // 3):
        raise RuntimeError("GLM parity trainer reference route size differs")
    try:
        route_bytes = base64.b64decode(routed["data"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("GLM parity trainer reference routes are invalid") from error
    if (
        len(route_bytes) != expected_route_bytes
        or hashlib.sha256(route_bytes).hexdigest() != routed["sha256"]
    ):
        raise RuntimeError("GLM parity trainer reference route checksum differs")
    context_length, hidden_layers, experts_per_token = expected_shape
    dense_layers = profile["dense_layers"]
    experts = profile["experts"]
    for token in range(context_length):
        token_offset = token * hidden_layers * experts_per_token
        for layer in range(hidden_layers):
            offset = token_offset + layer * experts_per_token
            selected = route_bytes[offset : offset + experts_per_token]
            if layer < dense_layers:
                if any(selected):
                    raise RuntimeError(
                        "GLM parity trainer reference dense routes are not zero"
                    )
            elif any(expert >= experts for expert in selected):
                raise RuntimeError(
                    "GLM parity trainer reference expert ID is out of range"
                )
            elif len(set(selected)) != experts_per_token:
                raise RuntimeError(
                    "GLM parity trainer reference has duplicate expert IDs"
                )
    entries = list(path.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise RuntimeError("GLM parity stage contains a symlink")
    actual_entries = {str(item.relative_to(path)) for item in entries}
    expected_entries = {
        "manifest.json",
        "trainer-reference.json",
        "trainer-router-trace",
        "trainer-router-trace/session.json",
    }
    if actual_entries != expected_entries:
        raise RuntimeError("GLM parity stage contains unexpected entries")
    trace = path / "trainer-router-trace"
    if stat.S_IMODE(trace.stat(follow_symlinks=False).st_mode) != 0o755:
        raise RuntimeError("GLM parity stage trace mode differs")
    for artifact in (
        manifest_path,
        path / "trainer-reference.json",
        trace / "session.json",
    ):
        details = artifact.stat(follow_symlinks=False)
        if stat.S_IMODE(details.st_mode) != 0o644 or details.st_nlink != 1:
            raise RuntimeError("GLM parity stage file mode differs")
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
    quoted_candidates = shlex.quote(glm_parity_stage_candidate_root)
    command = f"""
set -euo pipefail
stage={quoted_stage}
root={quoted_root}
candidates={quoted_candidates}
test -d "$root"
test ! -L "$root"
test "$(realpath -e "$root")" = "$root"
if test -L "$candidates"; then
  echo "candidate root is a symlink" >&2
  exit 40
fi
if ! test -e "$candidates"; then
  sudo install -d -o 0 -g 0 -m 0755 "$candidates"
fi
test -d "$candidates" || {{ echo "candidate root is not a directory" >&2; exit 45; }}
candidate_identity=$(stat -c '%u:%g:%a' "$candidates")
test "$candidate_identity" = "0:0:755" || {{
  echo "candidate root owner or mode differs: $candidate_identity" >&2
  exit 46
}}
if test -L "$stage"; then
  echo "stage path is a symlink" >&2
  exit 41
fi
if ! test -e "$stage"; then
  echo missing
  exit 0
fi
test -d "$stage" || {{ echo "stage path is not a directory" >&2; exit 42; }}
if test "$(stat -c '%u:%g:%a' "$stage")" != "1000:1000:750" || \
   test -L "$stage/manifest.json" || ! test -f "$stage/manifest.json" || \
   test -L "$stage/trainer-reference.json" || ! test -f "$stage/trainer-reference.json" || \
   test "$(stat -c '%u:%g:%a' "$stage/manifest.json")" != "1000:1000:644" || \
   test "$(stat -c '%h' "$stage/manifest.json")" != "1" || \
   test "$(stat -c '%u:%g:%a' "$stage/trainer-reference.json")" != "1000:1000:644" || \
   test "$(stat -c '%h' "$stage/trainer-reference.json")" != "1" || \
   test -L "$stage/trainer-router-trace" || ! test -d "$stage/trainer-router-trace" || \
   test "$(stat -c '%u:%g:%a' "$stage/trainer-router-trace")" != "1000:1000:755" || \
   test -L "$stage/trainer-router-trace/session.json" || \
   ! test -f "$stage/trainer-router-trace/session.json" || \
   test "$(stat -c '%u:%g:%a' "$stage/trainer-router-trace/session.json")" != "1000:1000:644" || \
   test "$(stat -c '%h' "$stage/trainer-router-trace/session.json")" != "1" || \
   test "$(find "$stage/trainer-router-trace" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" != \
     "session.json " || \
   test "$(find "$stage" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" != \
     "manifest.json trainer-reference.json trainer-router-trace "; then
  echo stale
else
  echo existing
fi
"""
    code, output = _ssh(orchestrator_host, command)
    if code:
        raise RuntimeError(f"GLM parity stage directory refused: {output.strip()}")
    state = output.strip()
    if state not in {"missing", "stale", "existing"}:
        raise RuntimeError("GLM parity stage directory returned an invalid state")
    return state


def _glm_parity_stage_candidate_root_prepare_command() -> str:
    root = PurePosixPath(glm_parity_root)
    candidate_root = PurePosixPath(glm_parity_stage_candidate_root)
    if candidate_root != root / ".stage-candidates":
        raise Refused("GLM parity candidate root is not the exact pinned directory")
    return f"""python3 - {shlex.quote(str(root))} {shlex.quote(str(candidate_root))} <<'PY'
import os,stat,subprocess,sys
root_path,candidate_root_path=sys.argv[1:]
root=os.open(root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 opened=os.fstat(root)
 current=os.stat(root_path,follow_symlinks=False)
 if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
  raise RuntimeError("qualification root identity changed")
 try:
  candidate=os.open(candidate_root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 except FileNotFoundError:
  subprocess.run(["sudo","install","-d","-o","0","-g","0","-m","0755",candidate_root_path],check=True)
  candidate=os.open(candidate_root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  candidate_stat=os.fstat(candidate)
  candidate_current=os.stat(candidate_root_path,follow_symlinks=False)
  if (candidate_stat.st_dev,candidate_stat.st_ino)!=(candidate_current.st_dev,candidate_current.st_ino):
   raise RuntimeError("candidate root identity changed")
  if candidate_stat.st_uid!=0 or candidate_stat.st_gid!=0 or stat.S_IMODE(candidate_stat.st_mode)!=0o755:
   raise RuntimeError("candidate root owner or mode differs")
  if candidate_stat.st_dev!=opened.st_dev:
   raise RuntimeError("candidate root is on another filesystem")
 finally:
  os.close(candidate)
finally:
 os.close(root)
print("ready")
PY"""


def _glm_parity_stage_build_command(
    candidate: str,
    candidate_identity: str,
    fixture: str,
    capture: str,
) -> str:
    candidate_path = PurePosixPath(candidate)
    candidate_root = PurePosixPath(glm_parity_stage_candidate_root)
    match = re.fullmatch(
        r"trainer-stage-candidate-([0-9a-f]{16,64})", candidate_path.name
    )
    if candidate_path.parent != candidate_root or match is None:
        raise Refused("GLM parity build target is not an exact stage candidate")
    if re.fullmatch(r"[0-9]+:[0-9]+", candidate_identity) is None:
        raise Refused("GLM parity build candidate identity is invalid")
    container = f"glm52-policy-parity-stage-{match.group(1)}"
    return f"""sudo python3 - {shlex.quote(str(candidate_root))} {shlex.quote(candidate_path.name)} {shlex.quote(candidate_identity)} {shlex.quote(container)} {shlex.quote(remote_repo)} {shlex.quote(inference_image)} {shlex.quote(fixture)} {shlex.quote(capture)} {shlex.quote(candidate)} <<'PY'
import ctypes,errno,os,re,stat,subprocess,sys
candidate_root_path,candidate_name,wanted,container,repo,image,fixture,capture,host_output=sys.argv[1:]
wanted_dev,wanted_ino=(int(value) for value in wanted.split(":"))
allowed={{"manifest.json","trainer-reference.json","trainer-router-trace"}}
cidfile=f"/run/{{container}}.cid"
mount_name=container+".mount"
mount_path=f"/run/{{mount_name}}"
libc=ctypes.CDLL(None,use_errno=True)
MS_BIND=4096
os.umask(0o077)

def identity(descriptor):
 details=os.fstat(descriptor)
 return details.st_dev,details.st_ino

def cleanup_container():
 try:
  with open(cidfile,"r",encoding="ascii") as stream:
   value=stream.read().strip()
 except FileNotFoundError:
  value=container
 else:
  if re.fullmatch(r"[0-9a-f]{{64}}",value) is None:
   raise RuntimeError("stage container ID file is invalid")
 identity=inspect_container(value)
 if identity is not None:
  subprocess.run(["docker","rm","-f",identity],timeout={glm_parity_stage_remove_timeout},check=True)

def inspect_container(reference):
 inspected=subprocess.run(
  ["docker","container","inspect","--format","{{{{.Id}}}} {{{{.Name}}}}",reference],
  capture_output=True,text=True,timeout={glm_parity_stage_inspect_timeout},check=False,
 )
 if inspected.returncode==0:
  fields=inspected.stdout.strip().split()
  if len(fields)!=2 or re.fullmatch(r"[0-9a-f]{{64}}",fields[0]) is None or fields[1]!=f"/{{container}}":
   raise RuntimeError("stage container identity differs")
  return fields[0]
 if inspected.returncode==1 and inspected.stderr.strip() in {{
 f"Error: No such container: {{reference}}",
 f"Error: No such object: {{reference}}",
  f"Error response from daemon: No such container: {{reference}}",
  f"Error response from daemon: No such object: {{reference}}",
 }}:
  return None
 raise RuntimeError("stage container inspect failed: "+inspected.stderr.strip()[:160])

def remove_cidfile():
 try:
  details=os.lstat(cidfile)
 except FileNotFoundError:
  return
 if not stat.S_ISREG(details.st_mode) or details.st_uid!=0 or details.st_nlink!=1:
  raise RuntimeError("stage container ID file identity differs")
 os.unlink(cidfile)

def prepare_cidfile():
 try:
  descriptor=os.open(cidfile,os.O_RDONLY|os.O_NOFOLLOW)
 except FileNotFoundError:
  return
 try:
  opened=os.fstat(descriptor)
  current=os.stat(cidfile,follow_symlinks=False)
  if not stat.S_ISREG(opened.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino) or opened.st_uid!=0 or opened.st_gid!=0 or stat.S_IMODE(opened.st_mode)!=0o600 or opened.st_nlink!=1 or opened.st_size!=64:
   raise RuntimeError("pre-existing stage container ID file identity differs")
  data=os.read(descriptor,65)
  if re.fullmatch(rb"[0-9a-f]{{64}}",data) is None:
   raise RuntimeError("pre-existing stage container ID file is invalid")
  value=data.decode("ascii")
  if inspect_container(value) is not None or inspect_container(container) is not None:
   raise RuntimeError("pre-existing stage container still exists")
  current=os.stat(cidfile,follow_symlinks=False)
  if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
   raise RuntimeError("pre-existing stage container ID file changed")
 finally:
  os.close(descriptor)
 os.unlink(cidfile)

def prepare_mount_bridge(runtime_descriptor,candidate_descriptor):
 os.mkdir(mount_name,0o700,dir_fd=runtime_descriptor)
 bridge=os.open(mount_name,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=runtime_descriptor)
 try:
  opened=os.fstat(bridge)
  current=os.stat(mount_name,dir_fd=runtime_descriptor,follow_symlinks=False)
  if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino) or opened.st_uid!=0 or opened.st_gid!=0 or stat.S_IMODE(opened.st_mode)!=0o700:
   raise RuntimeError("stage mount bridge identity differs")
  source=f"/proc/{{os.getpid()}}/fd/{{candidate_descriptor}}"
  if libc.mount(source.encode(),mount_path.encode(),None,MS_BIND,None)!=0:
   error=ctypes.get_errno()
   raise OSError(error,os.strerror(error))
  return bridge
 except BaseException as error:
  os.close(bridge)
  try:
   os.rmdir(mount_name,dir_fd=runtime_descriptor)
  except BaseException as cleanup_error:
   error.add_note("stage mount bridge directory cleanup failed: "+str(cleanup_error))
  raise

def cleanup_mount_bridge(runtime_descriptor,bridge_descriptor,mounted):
 if mounted:
  if libc.umount2(mount_path.encode(),0)!=0:
   error=ctypes.get_errno()
   current=os.stat(mount_name,dir_fd=runtime_descriptor,follow_symlinks=False)
   opened=os.fstat(bridge_descriptor)
   if error!=errno.EINVAL or (current.st_dev,current.st_ino)!=(opened.st_dev,opened.st_ino):
    raise OSError(error,os.strerror(error))
 current=os.stat(mount_name,dir_fd=runtime_descriptor,follow_symlinks=False)
 opened=os.fstat(bridge_descriptor)
 if (current.st_dev,current.st_ino)!=(opened.st_dev,opened.st_ino) or opened.st_uid!=0 or opened.st_gid!=0 or stat.S_IMODE(opened.st_mode)!=0o700:
  raise RuntimeError("stage mount bridge changed before removal")
 os.rmdir(mount_name,dir_fd=runtime_descriptor)

def cleanup_candidate(root_descriptor,candidate_descriptor):
 current=os.stat(candidate_name,dir_fd=root_descriptor,follow_symlinks=False)
 if (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):
  raise RuntimeError("stage cleanup path no longer identifies the candidate")
 directory=os.open(".",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=candidate_descriptor)
 try:
  if identity(directory)!=(wanted_dev,wanted_ino):
   raise RuntimeError("stage cleanup candidate identity differs")
  entries=set(os.listdir(directory))
  if not entries<=allowed:
   raise RuntimeError("stage cleanup candidate has an unexpected entry")
  for name in ("manifest.json","trainer-reference.json"):
   if name in entries:
    child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
    try:
     if not stat.S_ISREG(os.fstat(child).st_mode):
      raise RuntimeError("stage cleanup child is not regular")
    finally:
     os.close(child)
    os.unlink(name,dir_fd=directory)
  if "trainer-router-trace" in entries:
   trace=os.open("trainer-router-trace",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
   try:
    trace_entries=set(os.listdir(trace))
    if not trace_entries<={{"session.json"}}:
     raise RuntimeError("stage cleanup trace has an unexpected entry")
    if "session.json" in trace_entries:
     child=os.open("session.json",os.O_RDONLY|os.O_NOFOLLOW,dir_fd=trace)
     try:
      if not stat.S_ISREG(os.fstat(child).st_mode):
       raise RuntimeError("stage cleanup session is not regular")
     finally:
      os.close(child)
     os.unlink("session.json",dir_fd=trace)
   finally:
    os.close(trace)
   os.rmdir("trainer-router-trace",dir_fd=directory)
 finally:
  os.close(directory)
 current=os.stat(candidate_name,dir_fd=root_descriptor,follow_symlinks=False)
 if (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):
  raise RuntimeError("stage cleanup path changed before directory removal")
 os.rmdir(candidate_name,dir_fd=root_descriptor)

runtime=os.open("/run",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
root=os.open(candidate_root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
candidate=os.open(candidate_name,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root)
bridge=-1
bridge_mounted=False
bridge_cleanup_ok=True
evidence_cleanup_ok=True
container_cleanup_required=False
try:
 runtime_opened=os.fstat(runtime)
 runtime_current=os.stat("/run",follow_symlinks=False)
 if (runtime_opened.st_dev,runtime_opened.st_ino)!=(runtime_current.st_dev,runtime_current.st_ino) or runtime_opened.st_uid!=0 or runtime_opened.st_gid!=0 or stat.S_IMODE(runtime_opened.st_mode)&0o022:
  raise RuntimeError("stage runtime root identity or mode differs")
 root_opened=os.fstat(root)
 root_current=os.stat(candidate_root_path,follow_symlinks=False)
 if (root_opened.st_dev,root_opened.st_ino)!=(root_current.st_dev,root_current.st_ino):
  raise RuntimeError("stage candidate root identity changed")
 if root_opened.st_uid!=0 or root_opened.st_gid!=0 or stat.S_IMODE(root_opened.st_mode)!=0o755:
  raise RuntimeError("stage candidate root owner or mode differs")
 candidate_opened=os.fstat(candidate)
 current=os.stat(candidate_name,dir_fd=root,follow_symlinks=False)
 if (candidate_opened.st_dev,candidate_opened.st_ino)!=(wanted_dev,wanted_ino) or (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):
  raise RuntimeError("stage candidate identity differs before build")
 if candidate_opened.st_uid!=1000 or candidate_opened.st_gid!=1000 or stat.S_IMODE(candidate_opened.st_mode)!=0o750:
  raise RuntimeError("stage candidate owner or mode differs before build")
 check=os.open(".",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=candidate)
 try:
  if os.listdir(check):
   raise RuntimeError("stage candidate is not empty before build")
 finally:
  os.close(check)
 try:
  try:
   existing=inspect_container(container)
  except BaseException:
   evidence_cleanup_ok=False
   raise
  if existing is not None:
   evidence_cleanup_ok=False
   raise RuntimeError("stage container name is already in use")
  try:
   prepare_cidfile()
  except BaseException:
   evidence_cleanup_ok=False
   raise
  try:
   bridge=prepare_mount_bridge(runtime,candidate)
  except FileExistsError as error:
   bridge_cleanup_ok=False
   raise RuntimeError("stage mount bridge path already exists") from error
  bridge_mounted=True
  current=os.stat(mount_name,dir_fd=runtime,follow_symlinks=False)
  if (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):
   raise RuntimeError("stage mount bridge does not identify the candidate")
  command=[
   "docker","run","--rm","--name",container,"--cidfile",cidfile,
   "-v","/data:/data:ro","-v",f"{{mount_path}}:/stage",
   "-v",f"{{repo}}:/workspace:ro","-w","/workspace",
   "--entrypoint","/app/.venv/bin/python",image,
   "scripts/glm52_policy_parity.py","stage-trainer",
   "--fixture",fixture,"--h200-capture",capture,
   "--host-output",host_output,"/stage",
  ]
  container_cleanup_required=True
  subprocess.run(command,timeout={glm_parity_stage_docker_timeout},check=True)
  if inspect_container(container) is not None:
   raise RuntimeError("stage container remained after successful build")
  container_cleanup_required=False
  cleanup_mount_bridge(runtime,bridge,bridge_mounted)
  bridge_mounted=False
  os.close(bridge)
  bridge=-1
  root_current=os.stat(candidate_root_path,follow_symlinks=False)
  current=os.stat(candidate_name,dir_fd=root,follow_symlinks=False)
  if (root_opened.st_dev,root_opened.st_ino)!=(root_current.st_dev,root_current.st_ino) or (current.st_dev,current.st_ino)!=(wanted_dev,wanted_ino):
   raise RuntimeError("stage candidate path identity changed during build")
  remove_cidfile()
 except BaseException as error:
  container_cleanup_ok=True
  if container_cleanup_required:
   try:
    cleanup_container()
   except BaseException as cleanup_error:
    container_cleanup_ok=False
    error.add_note("stage container cleanup failed: "+str(cleanup_error))
  if bridge>=0:
   try:
    cleanup_mount_bridge(runtime,bridge,bridge_mounted)
    bridge_mounted=False
    os.close(bridge)
    bridge=-1
   except BaseException as cleanup_error:
    bridge_cleanup_ok=False
    error.add_note("stage mount bridge cleanup failed: "+str(cleanup_error))
  if container_cleanup_ok and bridge_cleanup_ok and evidence_cleanup_ok:
   try:
    cleanup_candidate(root,candidate)
   except BaseException as cleanup_error:
    error.add_note("stage candidate cleanup failed: "+str(cleanup_error))
   try:
    remove_cidfile()
   except BaseException as cleanup_error:
    error.add_note("stage container ID file cleanup failed: "+str(cleanup_error))
  else:
   error.add_note("stage candidate and container ID evidence retained")
  raise
finally:
 if bridge>=0:
  os.close(bridge)
 os.close(candidate)
 os.close(root)
 os.close(runtime)
print("built")
PY"""


def _glm_parity_stage_install_command(
    stage: str,
    candidate: str,
    manifest_sha256: str,
    reference_sha256: str,
    session_sha256: str,
) -> str:
    root = PurePosixPath(glm_parity_root)
    stage_path = PurePosixPath(stage)
    candidate_path = PurePosixPath(candidate)
    if stage_path != root / "trainer-stage-final":
        raise Refused("GLM parity install target is not the exact stage path")
    if (
        candidate_path.parent != PurePosixPath(glm_parity_stage_candidate_root)
        or re.fullmatch(r"trainer-stage-candidate-[0-9a-f]{16,64}", candidate_path.name)
        is None
    ):
        raise Refused("GLM parity candidate is not an exact stage candidate path")
    for name, value in (
        ("manifest", manifest_sha256),
        ("reference", reference_sha256),
        ("session", session_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise Refused(f"GLM parity stage {name} checksum is invalid")
    return f"""sudo python3 - {shlex.quote(str(root))} {shlex.quote(glm_parity_stage_candidate_root)} {shlex.quote(stage_path.name)} {shlex.quote(candidate_path.name)} {shlex.quote(manifest_sha256)} {shlex.quote(reference_sha256)} {shlex.quote(session_sha256)} <<'PY'
import ctypes,fcntl,hashlib,json,os,stat,sys
root_path,candidate_root_path,stage,candidate,wanted,wanted_reference,wanted_session=sys.argv[1:]
expected_top={{"manifest.json","trainer-reference.json","trainer-router-trace"}}
expected_files={{"trainer-reference.json","trainer-router-trace/session.json"}}

def identity(fd):
 st=os.fstat(fd)
 return st.st_dev,st.st_ino

def open_dir_at(parent,name):
 fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
 current=os.stat(name,dir_fd=parent,follow_symlinks=False)
 if not stat.S_ISDIR(current.st_mode) or identity(fd)!=(current.st_dev,current.st_ino):
  os.close(fd)
  raise RuntimeError("directory identity changed: "+name)
 return fd

def read_file_at(parent,name):
 fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
 try:
  opened=os.fstat(fd)
  current=os.stat(name,dir_fd=parent,follow_symlinks=False)
  if not stat.S_ISREG(opened.st_mode) or identity(fd)!=(current.st_dev,current.st_ino):
   raise RuntimeError("file identity changed: "+name)
  if opened.st_uid!=1000 or opened.st_gid!=1000 or stat.S_IMODE(opened.st_mode)!=0o644 or opened.st_nlink!=1:
   raise RuntimeError("file owner or mode differs: "+name)
  digest=hashlib.sha256()
  data=bytearray()
  while True:
   block=os.read(fd,8*1024*1024)
   if not block:
    break
   digest.update(block)
   data.extend(block)
  current=os.stat(name,dir_fd=parent,follow_symlinks=False)
  if identity(fd)!=(current.st_dev,current.st_ino):
   raise RuntimeError("file identity changed while reading: "+name)
  return bytes(data),digest.hexdigest()
 finally:
  os.close(fd)

def validate(parent,name,required=None):
 directory=open_dir_at(parent,name)
 try:
  opened=os.fstat(directory)
  if opened.st_uid!=1000 or opened.st_gid!=1000 or stat.S_IMODE(opened.st_mode)!=0o750:
   raise RuntimeError("stage owner or mode differs: "+name)
  if set(os.listdir(directory))!=expected_top:
   raise RuntimeError("stage entries differ: "+name)
  manifest_data,manifest_sha=read_file_at(directory,"manifest.json")
  reference_data,reference_sha=read_file_at(directory,"trainer-reference.json")
  del reference_data
  trace=open_dir_at(directory,"trainer-router-trace")
  try:
   trace_opened=os.fstat(trace)
   if trace_opened.st_uid!=1000 or trace_opened.st_gid!=1000 or stat.S_IMODE(trace_opened.st_mode)!=0o755:
    raise RuntimeError("stage trace owner or mode differs: "+name)
   if set(os.listdir(trace))!={{"session.json"}}:
    raise RuntimeError("stage trace entries differ: "+name)
   session_data,session_sha=read_file_at(trace,"session.json")
   del session_data
   trace_current=os.stat("trainer-router-trace",dir_fd=directory,follow_symlinks=False)
   if identity(trace)!=(trace_current.st_dev,trace_current.st_ino) or set(os.listdir(trace))!={{"session.json"}}:
    raise RuntimeError("stage trace identity changed: "+name)
  finally:
   os.close(trace)
  manifest=json.loads(manifest_data)
  if manifest.get("schema_version")!=1 or manifest.get("kind")!="glm52_policy_parity_trainer_stage":
   raise RuntimeError("stage manifest kind differs: "+name)
  if set(manifest.get("files",{{}}))!=expected_files:
   raise RuntimeError("stage manifest files differ: "+name)
  if manifest["files"]["trainer-reference.json"]!=reference_sha or manifest["files"]["trainer-router-trace/session.json"]!=session_sha:
   raise RuntimeError("stage artifact checksum differs: "+name)
  hashes=(manifest_sha,reference_sha,session_sha)
  if required is not None and hashes!=required:
   raise RuntimeError("stage required checksum differs: "+name)
  current=os.stat(name,dir_fd=parent,follow_symlinks=False)
  if identity(directory)!=(current.st_dev,current.st_ino) or set(os.listdir(directory))!=expected_top:
   raise RuntimeError("stage identity changed: "+name)
  return identity(directory),hashes
 finally:
  os.close(directory)

def remove_exact(parent,name,wanted_identity):
 directory=open_dir_at(parent,name)
 try:
  if identity(directory)!=wanted_identity:
   raise RuntimeError("cleanup identity differs: "+name)
  validate(parent,name)
  os.unlink("trainer-reference.json",dir_fd=directory)
  trace=open_dir_at(directory,"trainer-router-trace")
  try:
   os.unlink("session.json",dir_fd=trace)
  finally:
   os.close(trace)
  os.rmdir("trainer-router-trace",dir_fd=directory)
  os.unlink("manifest.json",dir_fd=directory)
  current=os.stat(name,dir_fd=parent,follow_symlinks=False)
  if identity(directory)!=(current.st_dev,current.st_ino):
   raise RuntimeError("cleanup target identity changed: "+name)
 finally:
  os.close(directory)
 os.rmdir(name,dir_fd=parent)

def exchange():
 libc=ctypes.CDLL(None,use_errno=True)
 result=libc.renameat2(root,stage.encode(),candidates,candidate.encode(),2)
 if result!=0:
  error=ctypes.get_errno()
  raise OSError(error,os.strerror(error))

def replace(stage_identity,candidate_identity,required):
 exchange()
 try:
  installed_identity,_=validate(root,stage,required)
  if installed_identity!=candidate_identity:
   raise RuntimeError("installed stage identity differs")
 except BaseException:
  current_stage=os.stat(stage,dir_fd=root,follow_symlinks=False)
  current_candidate=os.stat(candidate,dir_fd=candidates,follow_symlinks=False)
  if candidate_identity!=(current_stage.st_dev,current_stage.st_ino) or stage_identity!=(current_candidate.st_dev,current_candidate.st_ino):
   raise RuntimeError("cannot roll back exchanged stage identities")
  exchange()
  raise
 retained=os.stat(candidate,dir_fd=candidates,follow_symlinks=False)
 if stage_identity!=(retained.st_dev,retained.st_ino):
  raise RuntimeError("retained stage identity differs")
 print("replaced-retained:"+candidate)

root=os.open(root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 root_opened=os.fstat(root)
 root_current=os.stat(root_path,follow_symlinks=False)
 if not stat.S_ISDIR(root_current.st_mode) or identity(root)!=(root_current.st_dev,root_current.st_ino):
  raise RuntimeError("qualification root identity changed")
 candidates=os.open(candidate_root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 candidate_root_current=os.stat(candidate_root_path,follow_symlinks=False)
 candidate_root_opened=os.fstat(candidates)
 if identity(candidates)!=(candidate_root_current.st_dev,candidate_root_current.st_ino):
  raise RuntimeError("candidate root identity changed")
 if candidate_root_opened.st_uid!=0 or candidate_root_opened.st_gid!=0 or stat.S_IMODE(candidate_root_opened.st_mode)!=0o755:
  raise RuntimeError("candidate root owner or mode differs")
 if root_opened.st_dev!=candidate_root_opened.st_dev:
  raise RuntimeError("candidate root is on another filesystem")
 lock=os.open(".trainer-stage.lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=root)
 candidate_lock=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX)
  candidate_lock=open_dir_at(candidates,candidate)
  fcntl.flock(candidate_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  required=(wanted,wanted_reference,wanted_session)
  candidate_identity,_=validate(candidates,candidate,required)
  try:
   stage_identity,stage_hashes=validate(root,stage)
  except FileNotFoundError:
   os.rename(candidate,stage,src_dir_fd=candidates,dst_dir_fd=root)
   try:
    installed_identity,installed_hashes=validate(root,stage,required)
    if installed_identity!=candidate_identity or installed_hashes!=required:
     raise RuntimeError("installed stage identity differs")
   except BaseException:
    current_stage=os.stat(stage,dir_fd=root,follow_symlinks=False)
    if candidate_identity!=(current_stage.st_dev,current_stage.st_ino):
     raise RuntimeError("cannot roll back installed stage identity")
    try:
     os.stat(candidate,dir_fd=candidates,follow_symlinks=False)
    except FileNotFoundError:
     pass
    else:
     raise RuntimeError("cannot roll back because candidate path reappeared")
    os.rename(stage,candidate,src_dir_fd=root,dst_dir_fd=candidates)
    raise
   print("installed")
  except Exception as error:
   stage_fd=open_dir_at(root,stage)
   try:
    stage_identity=identity(stage_fd)
   finally:
    os.close(stage_fd)
   del error
   replace(stage_identity,candidate_identity,required)
  else:
   if stage_hashes==required:
    remove_exact(candidates,candidate,candidate_identity)
    print("current")
   else:
    replace(stage_identity,candidate_identity,required)
 finally:
  if candidate_lock is not None:
   os.close(candidate_lock)
  os.close(lock)
  os.close(candidates)
finally:
 os.close(root)
PY"""


def _glm_parity_stage_candidate_cleanup_command(
    candidate: str, candidate_identity: str
) -> str:
    root = PurePosixPath(glm_parity_root)
    candidate_path = PurePosixPath(candidate)
    if (
        candidate_path.parent != PurePosixPath(glm_parity_stage_candidate_root)
        or re.fullmatch(r"trainer-stage-candidate-[0-9a-f]{16,64}", candidate_path.name)
        is None
    ):
        raise Refused("GLM parity cleanup target is not an exact stage candidate")
    if re.fullmatch(r"[0-9]+:[0-9]+", candidate_identity) is None:
        raise Refused("GLM parity cleanup candidate identity is invalid")
    return f"""python3 - {shlex.quote(str(root))} {shlex.quote(glm_parity_stage_candidate_root)} {shlex.quote(candidate_path.name)} {shlex.quote(candidate_identity)} <<'PY'
import fcntl,os,stat,sys
root_path,candidate_root_path,candidate,wanted=sys.argv[1:]
wanted_identity=tuple(int(value) for value in wanted.split(":"))
allowed={{"manifest.json","trainer-reference.json","trainer-router-trace"}}
root=os.open(root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 opened=os.fstat(root)
 current=os.stat(root_path,follow_symlinks=False)
 if not stat.S_ISDIR(current.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
  raise RuntimeError("qualification root identity changed")
 candidates=os.open(candidate_root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 candidate_root_stat=os.fstat(candidates)
 candidate_root_current=os.stat(candidate_root_path,follow_symlinks=False)
 if (candidate_root_stat.st_dev,candidate_root_stat.st_ino)!=(candidate_root_current.st_dev,candidate_root_current.st_ino):
  raise RuntimeError("candidate root identity changed")
 if candidate_root_stat.st_uid!=0 or candidate_root_stat.st_gid!=0 or stat.S_IMODE(candidate_root_stat.st_mode)!=0o755:
  raise RuntimeError("candidate root owner or mode differs")
 lock=os.open(".trainer-stage.lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=root)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   directory=os.open(candidate,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=candidates)
  except FileNotFoundError:
   print("absent")
   raise SystemExit(0)
  try:
   directory_stat=os.fstat(directory)
   current=os.stat(candidate,dir_fd=candidates,follow_symlinks=False)
   directory_identity=(directory_stat.st_dev,directory_stat.st_ino)
   if directory_identity!=(current.st_dev,current.st_ino):
    raise RuntimeError("candidate identity changed")
   if directory_identity!=wanted_identity:
    raise RuntimeError("candidate identity differs from the created directory")
   fcntl.flock(directory,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if directory_stat.st_uid!=1000 or directory_stat.st_gid!=1000 or stat.S_IMODE(directory_stat.st_mode)!=0o750:
    raise RuntimeError("candidate owner or mode differs")
   entries=set(os.listdir(directory))
   if not entries<=allowed:
    raise RuntimeError("candidate contains an unexpected entry")
   for name in ("manifest.json","trainer-reference.json"):
    if name not in entries:
     continue
    descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
    try:
     child=os.fstat(descriptor)
     current=os.stat(name,dir_fd=directory,follow_symlinks=False)
     if not stat.S_ISREG(child.st_mode) or (child.st_dev,child.st_ino)!=(current.st_dev,current.st_ino):
      raise RuntimeError("candidate file identity differs: "+name)
    finally:
     os.close(descriptor)
    os.unlink(name,dir_fd=directory)
   if "trainer-router-trace" in entries:
    trace=os.open("trainer-router-trace",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
    try:
     trace_stat=os.fstat(trace)
     current=os.stat("trainer-router-trace",dir_fd=directory,follow_symlinks=False)
     if (trace_stat.st_dev,trace_stat.st_ino)!=(current.st_dev,current.st_ino):
      raise RuntimeError("candidate trace identity differs")
     trace_entries=set(os.listdir(trace))
     if not trace_entries<={{"session.json"}}:
      raise RuntimeError("candidate trace contains an unexpected entry")
     if "session.json" in trace_entries:
      descriptor=os.open("session.json",os.O_RDONLY|os.O_NOFOLLOW,dir_fd=trace)
      try:
       child=os.fstat(descriptor)
       current=os.stat("session.json",dir_fd=trace,follow_symlinks=False)
       if not stat.S_ISREG(child.st_mode) or (child.st_dev,child.st_ino)!=(current.st_dev,current.st_ino):
        raise RuntimeError("candidate session identity differs")
      finally:
       os.close(descriptor)
      os.unlink("session.json",dir_fd=trace)
    finally:
     os.close(trace)
    os.rmdir("trainer-router-trace",dir_fd=directory)
   current=os.stat(candidate,dir_fd=candidates,follow_symlinks=False)
   if directory_identity!=(current.st_dev,current.st_ino):
    raise RuntimeError("candidate identity changed before removal")
  finally:
   os.close(directory)
  os.rmdir(candidate,dir_fd=candidates)
  print("removed")
 finally:
  os.close(lock)
  os.close(candidates)
finally:
 os.close(root)
PY"""


def _glm_parity_stage_candidate_identity_command(candidate: str) -> str:
    candidate_path = PurePosixPath(candidate)
    if (
        candidate_path.parent != PurePosixPath(glm_parity_stage_candidate_root)
        or re.fullmatch(
            r"trainer-stage-candidate-[0-9a-f]{16,64}", candidate_path.name
        )
        is None
    ):
        raise Refused("GLM parity identity target is not an exact stage candidate")
    return f"""python3 - {shlex.quote(glm_parity_stage_candidate_root)} {shlex.quote(candidate_path.name)} <<'PY'
import os,stat,sys
root_path,name=sys.argv[1:]
root=os.open(root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 opened=os.fstat(root)
 current=os.stat(root_path,follow_symlinks=False)
 if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino) or opened.st_uid!=0 or opened.st_gid!=0 or stat.S_IMODE(opened.st_mode)!=0o755:
  raise RuntimeError("candidate root identity differs")
 candidate=os.open(name,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root)
 try:
  details=os.fstat(candidate)
  current=os.stat(name,dir_fd=root,follow_symlinks=False)
  if (details.st_dev,details.st_ino)!=(current.st_dev,current.st_ino) or details.st_uid!=1000 or details.st_gid!=1000 or stat.S_IMODE(details.st_mode)!=0o750:
   raise RuntimeError("candidate identity differs")
  print(f"{{details.st_dev}}:{{details.st_ino}}")
 finally:
  os.close(candidate)
finally:
 os.close(root)
PY"""


def _recover_glm_parity_stage_candidate_identity(
    host: str, candidate: str
) -> str | None:
    code, detail = _ssh(
        host, _glm_parity_stage_candidate_identity_command(candidate)
    )
    value = detail.strip()
    if code or re.fullmatch(r"[0-9]+:[0-9]+", value) is None:
        return None
    return value


def _glm_parity_stage_current_command(
    stage: str,
    manifest_sha256: str,
    reference_sha256: str,
    session_sha256: str,
) -> str:
    root = PurePosixPath(glm_parity_root)
    stage_path = PurePosixPath(stage)
    if stage_path != root / "trainer-stage-final":
        raise Refused("GLM parity current check target is not the exact stage")
    for value in (manifest_sha256, reference_sha256, session_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise Refused("GLM parity current check checksum is invalid")
    return f"""python3 - {shlex.quote(str(root))} {shlex.quote(stage_path.name)} {shlex.quote(manifest_sha256)} {shlex.quote(reference_sha256)} {shlex.quote(session_sha256)} <<'PY'
import fcntl,hashlib,os,stat,sys
root_path,stage,wanted_manifest,wanted_reference,wanted_session=sys.argv[1:]
root=os.open(root_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 opened=os.fstat(root)
 current=os.stat(root_path,follow_symlinks=False)
 if not stat.S_ISDIR(current.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
  raise RuntimeError("qualification root identity changed")
 lock=os.open(".trainer-stage.lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=root)
 try:
  fcntl.flock(lock,fcntl.LOCK_SH)
  try:
   directory=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root)
  except FileNotFoundError:
   print("stale")
   raise SystemExit(0)
  try:
   directory_stat=os.fstat(directory)
   current=os.stat(stage,dir_fd=root,follow_symlinks=False)
   if (directory_stat.st_dev,directory_stat.st_ino)!=(current.st_dev,current.st_ino):
    raise RuntimeError("stage identity changed")
   if directory_stat.st_uid!=1000 or directory_stat.st_gid!=1000 or stat.S_IMODE(directory_stat.st_mode)!=0o750:
    print("stale")
    raise SystemExit(0)
   if set(os.listdir(directory))!={{"manifest.json","trainer-reference.json","trainer-router-trace"}}:
    print("stale")
    raise SystemExit(0)
   def digest(parent,name):
    try:
     descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
    except OSError:
     print("stale")
     raise SystemExit(0)
    try:
     opened=os.fstat(descriptor)
     current=os.stat(name,dir_fd=parent,follow_symlinks=False)
     if not stat.S_ISREG(opened.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
      raise RuntimeError("stage file identity changed")
     if opened.st_uid!=1000 or opened.st_gid!=1000 or stat.S_IMODE(opened.st_mode)!=0o644 or opened.st_nlink!=1:
      print("stale")
      raise SystemExit(0)
     value=hashlib.sha256()
     while True:
      block=os.read(descriptor,8*1024*1024)
      if not block:
       break
      value.update(block)
     current=os.stat(name,dir_fd=parent,follow_symlinks=False)
     if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
      raise RuntimeError("stage file identity changed while reading")
     return value.hexdigest()
    finally:
     os.close(descriptor)
   manifest=digest(directory,"manifest.json")
   reference=digest(directory,"trainer-reference.json")
   try:
    trace=os.open("trainer-router-trace",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
   except OSError:
    print("stale")
    raise SystemExit(0)
   try:
    trace_stat=os.fstat(trace)
    trace_current=os.stat("trainer-router-trace",dir_fd=directory,follow_symlinks=False)
    if (trace_stat.st_dev,trace_stat.st_ino)!=(trace_current.st_dev,trace_current.st_ino):
     raise RuntimeError("stage trace identity changed")
    if trace_stat.st_uid!=1000 or trace_stat.st_gid!=1000 or stat.S_IMODE(trace_stat.st_mode)!=0o755:
     print("stale")
     raise SystemExit(0)
    if set(os.listdir(trace))!={{"session.json"}}:
     print("stale")
     raise SystemExit(0)
    session=digest(trace,"session.json")
    trace_current=os.stat("trainer-router-trace",dir_fd=directory,follow_symlinks=False)
    if (trace_stat.st_dev,trace_stat.st_ino)!=(trace_current.st_dev,trace_current.st_ino):
     raise RuntimeError("stage trace identity changed while reading")
   finally:
    os.close(trace)
   current=os.stat(stage,dir_fd=root,follow_symlinks=False)
   if (directory_stat.st_dev,directory_stat.st_ino)!=(current.st_dev,current.st_ino):
    raise RuntimeError("stage identity changed while reading")
   print("current" if (manifest,reference,session)==(wanted_manifest,wanted_reference,wanted_session) else "stale")
  finally:
   os.close(directory)
 finally:
  os.close(lock)
finally:
 os.close(root)
PY"""


def _cleanup_glm_parity_stage_candidate(
    host: str, candidate: str, candidate_identity: str
) -> None:
    code, detail = _ssh(
        host,
        _glm_parity_stage_candidate_cleanup_command(
            candidate, candidate_identity
        ),
    )
    state = detail.strip()
    if code or state not in {"absent", "removed"}:
        raise RuntimeError(
            "GLM parity candidate cleanup incomplete: "
            f"{state[:160] or f'exit {code}'}"
        )


def _copy_glm_parity_stage(host: str, stage: str, local_parent: Path) -> Path:
    try:
        copied = subprocess.run(
            [
                "scp",
                "-C",
                "-q",
                "-o",
                "BatchMode=yes",
                "-r",
                f"{host}:{stage}",
                str(local_parent),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("GLM parity stage copy timed out") from error
    if copied.returncode:
        raise RuntimeError(
            f"cannot copy GLM parity stage: {copied.stderr.strip()[:160]}"
        )
    return local_parent / PurePosixPath(stage).name


def _glm_parity_transfer_command(
    source: str, target: str, candidate: str, identity: str
) -> list[str]:
    path = PurePosixPath(candidate)
    if (
        target not in {trainer_head, *trainer_internal_addresses.values()}
        or path.parent != PurePosixPath(glm_parity_stage_candidate_root)
        or re.fullmatch(r"trainer-stage-candidate-[0-9a-f]{16}", path.name) is None
        or re.fullmatch(r"[0-9]+:[0-9]+", identity) is None
    ):
        raise Refused("GLM parity transfer target is invalid")
    locked_rsync = (
        'test ! -L "$1" && test -d "$1" && '
        'test "$(stat -c \'%d:%i\' "$1")" = "$2" && '
        'shift 2 && exec rsync "$@"'
    )
    remote = (
        f"test ! -L {shlex.quote(candidate)} && test -d {shlex.quote(candidate)} && "
        f"test \"$(stat -c '%d:%i' {shlex.quote(candidate)})\" = {shlex.quote(identity)} "
        f"&& exec timeout --signal=TERM --kill-after=5s 500s "
        f"flock -- {shlex.quote(candidate)} sh -c {shlex.quote(locked_rsync)} "
        f"sh {shlex.quote(candidate)} {shlex.quote(identity)}"
    )
    return [
        "timeout", "--signal=TERM", "--kill-after=5s", "540s", "rsync",
        "--archive", "--compress", "--partial", "--append-verify",
        "--timeout=120", "--quiet", "-e", shlex.join(trainer_ssh_command),
        f"--rsync-path={remote}", f"{source}/.", f"{target}:{candidate}/",
    ]


def _glm_parity_internal_transfer_command(
    source: str, target: str, candidate: str, identity: str
) -> str:
    path = PurePosixPath(candidate)
    if (
        source != glm_parity_paths["trainer_stage"]
        or target not in trainer_internal_addresses.values()
        or path.parent != PurePosixPath(glm_parity_stage_candidate_root)
        or re.fullmatch(r"trainer-stage-candidate-[0-9a-f]{16}", path.name) is None
        or re.fullmatch(r"[0-9]+:[0-9]+", identity) is None
    ):
        raise Refused("GLM parity internal transfer route is invalid")
    receiver = """import fcntl,os,stat,sys
path,wanted=sys.argv[1:]
descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
opened=os.fstat(descriptor)
if (opened.st_uid,opened.st_gid,stat.S_IMODE(opened.st_mode))!=(1000,1000,0o750):
 raise RuntimeError("candidate owner or mode differs")
if f"{opened.st_dev}:{opened.st_ino}"!=wanted:
 raise RuntimeError("candidate identity differs")
fcntl.flock(descriptor,fcntl.LOCK_EX)
current=os.stat(path,follow_symlinks=False)
if (current.st_dev,current.st_ino)!=(opened.st_dev,opened.st_ino):
 raise RuntimeError("candidate identity changed after lock")
os.set_inheritable(descriptor,True)
os.umask(0o022)
os.execvp("tar",["tar","--extract","--gzip","--file","-","--directory",f"/proc/self/fd/{descriptor}","--no-same-owner","--no-same-permissions"])
"""
    receive = shlex.join(
        [
            "timeout", "--signal=TERM", "--kill-after=5s", "500s",
            "python3", "-c", receiver, candidate, identity,
        ]
    )
    archive = shlex.join(
        [
            "tar", "--create", "--gzip", "--file", "-", "--directory", source,
            "manifest.json", "trainer-reference.json", "trainer-router-trace",
        ]
    )
    pipeline = f"{archive} | {shlex.join([*trainer_ssh_command, target, receive])}"
    return shlex.join(
        [
            "timeout", "--signal=TERM", "--kill-after=5s", "540s",
            "bash", "-o", "pipefail", "-c", pipeline,
        ]
    )


def _glm_parity_transfer_preflight(
    target: str, source_host: str | None = None
) -> str | None:
    if source_host is None:
        check = "command -v rsync && command -v timeout && command -v flock"
        local = subprocess.run(
            ["sh", "-c", check], capture_output=True, text=True, timeout=10
        )
        if local.returncode:
            return "local rsync, timeout, or flock is unavailable"
        code, detail = _ssh(target, check)
    else:
        if source_host != trainer_head or target not in trainer_internal_addresses.values():
            raise Refused("GLM parity internal transfer route is invalid")
        source_check = (
            "command -v tar && command -v gzip && command -v timeout && "
            "command -v bash && command -v ssh"
        )
        target_check = (
            "command -v tar && command -v gzip && command -v timeout && "
            "command -v python3"
        )
        code, detail = _ssh(
            source_host,
            f"{source_check} && "
            f"{shlex.join([*trainer_ssh_command, target, target_check])}",
        )
    return None if code == 0 else (detail.strip()[:120] or f"exit {code}")


def _transfer_glm_parity_stage(
    source: str, target: str, candidate: str, identity: str,
    source_host: str | None = None,
) -> tuple[bool, str]:
    command = (
        _glm_parity_transfer_command(source, target, candidate, identity)
        if source_host is None
        else _glm_parity_internal_transfer_command(
            source, target, candidate, identity
        )
    )
    failures = []
    for attempt in range(1, 4):
        try:
            if source_host is None:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=580
                )
                code, detail = result.returncode, result.stderr or result.stdout
            else:
                code, detail = _ssh(source_host, command, timeout=580)
                if code == 124 and detail.startswith("TIMEOUT"):
                    return False, f"attempt {attempt}: controller timeout"
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"attempt {attempt}: process state uncertain: {error}"
        if code == 0:
            return True, "ok"
        detail = " ".join((detail or "failed").strip().splitlines())[:120]
        failures.append(f"attempt {attempt}: exit {code}: {detail}")
    return False, "; ".join(failures)


def _validate_glm_parity_stage_live_identity(
    local_stage: Path, fixture: str, capture: str
) -> str:
    manifest_sha256 = _validate_glm_parity_stage(local_stage)
    stage_manifest = json.loads((local_stage / "manifest.json").read_text())
    command = f"""python3 - {shlex.quote(fixture + '/manifest.json')} {shlex.quote(capture + '/manifest.json')} <<'PY'
import hashlib,json,os,stat,sys
fixture_path,capture_path=sys.argv[1:]
def read(path):
 descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(descriptor)
  current=os.stat(path,follow_symlinks=False)
  if not stat.S_ISREG(opened.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
   raise RuntimeError("manifest identity changed: "+path)
  data=bytearray()
  while True:
   block=os.read(descriptor,1024*1024)
   if not block:
    break
   data.extend(block)
  current=os.stat(path,follow_symlinks=False)
  if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):
   raise RuntimeError("manifest identity changed while reading: "+path)
  return bytes(data)
 finally:
  os.close(descriptor)
fixture_data=read(fixture_path)
capture_data=read(capture_path)
fixture=json.loads(fixture_data)
capture=json.loads(capture_data)
if fixture.get("schema_version")!=1 or fixture.get("kind")!="glm52_policy_parity_fixture":
 raise RuntimeError("live fixture manifest kind differs")
if capture.get("schema_version")!=1 or capture.get("kind")!="glm52_policy_parity_h200_capture":
 raise RuntimeError("live H200 capture manifest kind differs")
canonical_fixture=(json.dumps(fixture,sort_keys=True,separators=(",",":"))+"\\n").encode()
result={{
 "fixture_sha256":hashlib.sha256(canonical_fixture).hexdigest(),
 "capture_manifest_sha256":hashlib.sha256(capture_data).hexdigest(),
 "capture_fixture_sha256":capture.get("fixture_sha256"),
 "profile_sha256":capture.get("profile_sha256"),
 "model_revision":capture.get("model_revision"),
 "adapter_sha256":capture.get("adapter_sha256"),
 "h200_image":capture.get("image"),
 "capture_router_ids_sha256":capture.get("arrays",{{}}).get("router_ids",{{}}).get("sha256"),
 "capture_full_logprobs_sha256":capture.get("arrays",{{}}).get("full_logprobs",{{}}).get("sha256"),
}}
print(json.dumps(result,sort_keys=True,separators=(",",":")))
PY"""
    live_output = _remote_ok(
        orchestrator_host,
        command,
        "GLM parity live stage identity",
    )
    try:
        live = json.loads(live_output)
    except json.JSONDecodeError as error:
        raise RuntimeError("GLM parity live identity output is invalid") from error
    identity = stage_manifest["identity"]
    if stage_manifest.get("fixture_sha256") != live.get("fixture_sha256"):
        raise RuntimeError("GLM parity stage does not match the live fixture")
    if stage_manifest.get("h200_capture_manifest_sha256") != live.get(
        "capture_manifest_sha256"
    ):
        raise RuntimeError("GLM parity stage does not match the live H200 capture")
    if (
        live.get("capture_fixture_sha256") != live.get("fixture_sha256")
        or live.get("profile_sha256") != stage_manifest["profile_sha256"]
        or live.get("model_revision") != identity["model_revision"]
        or live.get("adapter_sha256") != identity["adapter_sha256"]
        or live.get("h200_image") != identity["h200_image"]
        or live.get("capture_router_ids_sha256")
        != stage_manifest["reference_schema"]["sha256"]
        or live.get("capture_full_logprobs_sha256")
        != stage_manifest["reference_schema"]["full_logprobs_sha256"]
    ):
        raise RuntimeError("GLM parity stage capture identity differs")
    return manifest_sha256


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
    if (
        trainer_head not in nodes
        or len(nodes) != len(set(nodes))
        or any(host not in trainer_hosts for host in nodes)
    ):
        raise Refused("GLM parity staging requires configured trainers and the head")
    fixture = glm_parity_paths["fixture"]
    capture = glm_parity_paths["h200_capture"]
    stage = glm_parity_paths["trainer_stage"]
    qualification_root = f"{glm_parity_root}/"
    if not all(path.startswith(qualification_root) for path in (fixture, capture, stage)):
        raise Refused("GLM parity paths must stay under the qualification directory")
    _prepare_glm_parity_capture_directory()
    _deploy_glm_parity_staging_sources()
    stage_state = _prepare_glm_parity_stage_directory()
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-stage-") as temporary:
        temporary_root = Path(temporary)
        current_parent = temporary_root / "current"
        current_parent.mkdir()
        local_stage: Path | None = None
        manifest_sha256: str | None = None
        if stage_state == "existing":
            current_stage = _copy_glm_parity_stage(
                orchestrator_host, stage, current_parent
            )
            try:
                manifest_sha256 = _validate_glm_parity_stage_live_identity(
                    current_stage, fixture, capture
                )
            except RuntimeError as error:
                print(f"GLM parity stage is stale: {error}")
            else:
                local_stage = current_stage
        if local_stage is None:
            candidate = (
                f"{glm_parity_stage_candidate_root}/trainer-stage-candidate-"
                f"{secrets.token_hex(8)}"
            )
            candidate_exists = True
            candidate_identity: str | None = None
            try:
                candidate_identity = _remote_ok(
                    orchestrator_host,
                    f"test ! -e {shlex.quote(candidate)} && "
                    f"sudo install -d -o 1000 -g 1000 -m 0750 "
                    f"{shlex.quote(candidate)} && "
                    f"stat -c '%d:%i' {shlex.quote(candidate)}",
                    "GLM parity candidate directory",
                ).strip()
                if re.fullmatch(r"[0-9]+:[0-9]+", candidate_identity) is None:
                    raise RuntimeError("GLM parity candidate identity is invalid")
                command = _glm_parity_stage_build_command(
                    candidate,
                    candidate_identity,
                    fixture,
                    capture,
                )
                candidate_exists = False
                _remote_ok(
                    orchestrator_host,
                    command,
                    "GLM parity trainer staging",
                    timeout=glm_parity_stage_outer_timeout,
                )
                candidate_exists = True
                _remote_ok(
                    orchestrator_host,
                    f"test \"$(stat -c '%d:%i' {shlex.quote(candidate)})\" = "
                    f"{shlex.quote(candidate_identity)}",
                    "GLM parity candidate identity after build",
                )
                candidate_parent = temporary_root / "candidate"
                candidate_parent.mkdir()
                candidate_stage = _copy_glm_parity_stage(
                    orchestrator_host, candidate, candidate_parent
                )
                manifest_sha256 = _validate_glm_parity_stage_live_identity(
                    candidate_stage, fixture, capture
                )
                stage_manifest = json.loads(
                    (candidate_stage / "manifest.json").read_text()
                )
                install_command = _glm_parity_stage_install_command(
                    stage,
                    candidate,
                    manifest_sha256,
                    stage_manifest["files"]["trainer-reference.json"],
                    stage_manifest["files"]["trainer-router-trace/session.json"],
                )
                install_state = _remote_ok(
                    orchestrator_host,
                    install_command,
                    "GLM parity stage replacement",
                ).strip()
                retained_prefix = "replaced-retained:"
                if install_state not in {
                    "current",
                    "installed",
                    "replaced",
                } and not install_state.startswith(retained_prefix):
                    raise RuntimeError(
                        "GLM parity stage replacement state is invalid"
                    )
                if install_state.startswith(retained_prefix):
                    print(
                        "WARNING: GLM parity stage replacement committed; "
                        f"old exact stage retained: {install_state[len(retained_prefix):]}"
                    )
                candidate_exists = False
                local_stage = candidate_stage
            except BaseException as error:
                if candidate_exists:
                    candidate_identity = candidate_identity or (
                        _recover_glm_parity_stage_candidate_identity(
                            orchestrator_host, candidate
                        )
                    )
                    if candidate_identity is None:
                        error.add_note(
                            "GLM parity exact candidate identity could not be "
                            "recovered; candidate was retained"
                        )
                    else:
                        try:
                            _cleanup_glm_parity_stage_candidate(
                                orchestrator_host,
                                candidate,
                                candidate_identity,
                            )
                        except BaseException as cleanup_error:
                            error.add_note(
                                "GLM parity exact candidate cleanup is incomplete: "
                                f"{cleanup_error}"
                            )
                raise
        if manifest_sha256 is None:
            raise RuntimeError("GLM parity stage manifest checksum is unavailable")
        stage_manifest = json.loads((local_stage / "manifest.json").read_text())
        reference_sha256 = stage_manifest["files"]["trainer-reference.json"]
        session_sha256 = stage_manifest["files"][
            "trainer-router-trace/session.json"
        ]
        fanout = False

        def install(host: str) -> str:
            current_code, current_state = _ssh(
                host,
                _glm_parity_stage_current_command(
                    stage,
                    manifest_sha256,
                    reference_sha256,
                    session_sha256,
                ),
            )
            if current_code:
                return f"current stage check failed: {current_state.strip()[:120]}"
            if current_state.strip() == "current":
                return "ok"
            if current_state.strip() != "stale":
                return "current stage check returned an invalid state"
            source_host = trainer_head if fanout else None
            source = stage if fanout else str(local_stage)
            target = trainer_internal_addresses[host] if fanout else host
            preflight = _glm_parity_transfer_preflight(target, source_host)
            if preflight is not None:
                return f"rsync preflight failed: {preflight}"
            remote_candidate = (
                f"{glm_parity_stage_candidate_root}/trainer-stage-candidate-"
                f"{secrets.token_hex(8)}"
            )
            candidate_root_code, candidate_root_detail = _ssh(
                host, _glm_parity_stage_candidate_root_prepare_command()
            )
            if (
                candidate_root_code
                or candidate_root_detail.strip() != "ready"
            ):
                return (
                    "candidate root refused: "
                    f"{candidate_root_detail.strip()[:120]}"
                )
            absent_code, absent_detail = _ssh(
                host, f"test ! -e {shlex.quote(remote_candidate)}"
            )
            if absent_code:
                return f"candidate path refused: {absent_detail.strip()[:120]}"
            candidate_exists = True
            remote_candidate_identity: str | None = None
            transfer_started = False
            result = "ok"
            try:
                create_code, create_detail = _ssh(
                    host,
                    f"sudo install -d -o 1000 -g 1000 -m 0750 "
                    f"{shlex.quote(remote_candidate)} && "
                    f"stat -c '%d:%i' {shlex.quote(remote_candidate)}",
                )
                if create_code:
                    raise RuntimeError(
                        "candidate create failed: "
                        f"{create_detail.strip()[:120]}"
                    )
                remote_candidate_identity = create_detail.strip()
                if re.fullmatch(
                    r"[0-9]+:[0-9]+", remote_candidate_identity
                ) is None:
                    raise RuntimeError("candidate create identity is invalid")
                transfer_started = True
                copied, copy_detail = _transfer_glm_parity_stage(
                    source,
                    target,
                    remote_candidate,
                    remote_candidate_identity,
                    source_host,
                )
                if not copied:
                    result = f"transfer failed: {copy_detail}"
                else:
                    command = _glm_parity_stage_install_command(
                        stage,
                        remote_candidate,
                        manifest_sha256,
                        reference_sha256,
                        session_sha256,
                    )
                    rc, detail = _ssh(host, command)
                    state = detail.strip()
                    if rc:
                        error_line = state.splitlines()[-1] if state else f"exit {rc}"
                        result = f"install failed: {error_line[:120]}"
                    elif state.startswith("replaced-retained:"):
                        candidate_exists = False
                        result = f"ok-retained:{state.removeprefix('replaced-retained:')}"
                    elif state in {"current", "installed", "replaced"}:
                        candidate_exists = False
                    else:
                        result = "install returned an invalid state"
            except BaseException as error:
                if candidate_exists and transfer_started:
                    error.add_note(
                        "GLM parity transfer started; exact candidate was retained"
                    )
                elif candidate_exists:
                    remote_candidate_identity = remote_candidate_identity or (
                        _recover_glm_parity_stage_candidate_identity(
                            host, remote_candidate
                        )
                    )
                    if remote_candidate_identity is None:
                        error.add_note(
                            "GLM parity exact candidate identity could not be "
                            "recovered; candidate was retained"
                        )
                    else:
                        try:
                            _cleanup_glm_parity_stage_candidate(
                                host,
                                remote_candidate,
                                remote_candidate_identity,
                            )
                        except BaseException as cleanup_error:
                            error.add_note(
                                "GLM parity exact candidate cleanup is incomplete: "
                                f"{cleanup_error}"
                            )
                raise
            else:
                if candidate_exists:
                    if transfer_started:
                        result = f"{result}; exact candidate retained"
                    elif remote_candidate_identity is None:
                        result = (
                            f"{result}; exact candidate identity unavailable; "
                            "candidate retained"
                        )
                    else:
                        try:
                            _cleanup_glm_parity_stage_candidate(
                                host,
                                remote_candidate,
                                remote_candidate_identity,
                            )
                        except Exception as cleanup_error:
                            result = (
                                f"{result}; exact candidate cleanup incomplete: "
                                f"{str(cleanup_error)[:100]}"
                            )
            return result

        seed = install(trainer_head)
        if seed != "ok" and not seed.startswith("ok-retained:"):
            raise RuntimeError(f"GLM parity seed transfer failed: {seed[:240]}")
        current_command = _glm_parity_stage_current_command(
            stage, manifest_sha256, reference_sha256, session_sha256
        )
        if _ssh(trainer_head, current_command) != (0, "current\n"):
            raise RuntimeError("GLM parity trainer-head source validation failed")
        fanout = True
        results = {trainer_head: seed}
        results.update(_fan([host for host in nodes if host != trainer_head], install))
        if _ssh(trainer_head, current_command) != (0, "current\n"):
            raise RuntimeError("GLM parity trainer-head source changed during fanout")
    failed = [
        host
        for host, result in results.items()
        if result != "ok" and not result.startswith("ok-retained:")
    ]
    if failed:
        reasons = "; ".join(f"{host}: {results[host][:240]}" for host in failed)
        raise RuntimeError(f"GLM parity stage failed: {reasons}")
    for host, result in results.items():
        if result.startswith("ok-retained:"):
            print(
                "WARNING: GLM parity stage replacement committed on "
                f"{host}; old exact stage retained: {result.removeprefix('ok-retained:')}"
            )
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
    tunnel_manager = f"{remote_repo}/scripts/h200/glm_adapter_tunnels.py"
    deployed_operations = f"{remote_repo}/configs/operations.toml"
    _remote_ok(
        orchestrator_host,
        f"if test -f {shlex.quote(tunnel_manager)} && "
        f"test -f {shlex.quote(deployed_operations)}; then "
        f"python3 {shlex.quote(tunnel_manager)} --operations "
        f"{shlex.quote(deployed_operations)} stop; "
        f"elif test -e {shlex.quote(glm_adapter_tunnel_state)}; then "
        "echo 'GLM adapter tunnel state exists without its exact manager or inventory' >&2; "
        "exit 44; fi",
        "GLM adapter tunnels stop",
    )
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


def _qualify_glm52_step20_parity_action(arguments: list[str]):
    if arguments:
        raise Refused("qualify-glm52-step20-parity takes no arguments")
    adapter = glm_step20_h200_gate["trainer_adapter"]
    return qualify(
        "glm52-policy-parity",
        resume_adapter=adapter,
        resume_step=0,
    )


def _stage_glm52_step20_adapter_action(arguments: list[str]):
    if arguments:
        raise Refused("stage-glm52-step20-adapter takes no arguments")
    stability, result_sha256 = _step20_stability_evidence()
    path = _stage_step20_adapter_on_h200(
        stability["adapter"], result_sha256
    )
    return {
        "adapter": path,
        "adapter_sha256": stability["adapter"]["sha256"],
        "qualification_result": glm_step20_h200_gate["qualification_result"],
        "qualification_result_sha256": result_sha256,
    }


def _prepare_glm52_adapter_server_action(arguments: list[str]):
    if arguments:
        raise Refused("prepare-glm52-adapter-server takes no arguments")
    stability, _result_sha256 = _step20_stability_evidence()
    adapter = stability.get("adapter")
    if not isinstance(adapter, dict) or not isinstance(adapter.get("path"), str):
        raise Refused("step-20 adapter server source is missing")
    return _ensure_glm_adapter_range_server(adapter["path"])


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
    "qualify-glm52-step20-parity": _qualify_glm52_step20_parity_action,
    "prepare-glm52-adapter-server": _prepare_glm52_adapter_server_action,
    "stage-glm52-step20-adapter": _stage_glm52_step20_adapter_action,
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
  qualify-glm52-step20-parity  score fixed parity tokens with the step-20 adapter
  stage-glm52-step20-adapter   copy the exact step-20 adapter to H200
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
