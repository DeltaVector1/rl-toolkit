"""Fail-closed validation and remote script for one adapter cache entry."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shlex
import shutil
import stat
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


SOURCE_SUFFIX = "/broadcasts/step_0"
RECEIPT_KEYS = frozenset(
    {
        "source_url",
        "target_sha256",
        "target_size",
        "config_sha256",
        "config_size",
        "lora_name",
        "server_instance",
    }
)
REQUIRED_FILES = frozenset(
    {
        "READY",
        "manifest.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.safetensors.sha256",
    }
)
OPTIONAL_FILES = frozenset(
    {
        "transport.json",
        "adapter_manifest.json",
        "relay_manifest.json",
        "transport-manifest.json",
        "transport-chunks",
    }
)
TRANSPORT_MANIFEST_KEYS = frozenset(
    {
        "base2_config_sha256",
        "base2_config_size",
        "base2_sha256",
        "base2_size",
        "base2_url",
        "base_config_sha256",
        "base_config_size",
        "base_sha256",
        "base_size",
        "base_url",
        "chunks",
        "config_sha256",
        "config_size",
        "data_offset",
        "mode",
        "resume_identity_version",
        "source_url",
        "target_sha256",
        "target_size",
        "transport",
        "version",
    }
)
TRANSPORT_CHUNK_KEYS = frozenset(
    {
        "codec",
        "data_offset",
        "file",
        "offset",
        "raw_sha256",
        "raw_size",
        "wire_sha256",
        "wire_size",
    }
)
TRANSPORT_CHUNK_COUNT = 986
TRANSPORT_CHUNK_PREFIX = "adapter_model.safetensors.transport-full-zstd-"
TRANSPORT_WIRE_SIZE = 6382335838


def exact_source_url(source_url: str) -> str:
    """Return the exact source object URL used to derive the cache key."""
    parsed = _validated_loopback_url(source_url)
    if parsed.path.endswith(SOURCE_SUFFIX):
        return source_url
    if "/broadcasts/" in parsed.path:
        raise ValueError("adapter cache source URL must name the exact step_0 object")
    return source_url + SOURCE_SUFFIX


def _validated_loopback_url(source_url: str):
    if (
        not isinstance(source_url, str)
        or not source_url
        or source_url != source_url.strip()
        or "\n" in source_url
        or "\r" in source_url
    ):
        raise ValueError("adapter cache source URL is invalid")
    parsed = urlsplit(source_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or parsed.path.endswith("/")
    ):
        raise ValueError("adapter cache source URL must be an exact loopback URL")
    return parsed


def configured_source_url(
    source_url: str,
    *,
    run: str,
    service_name: str,
) -> str:
    """Derive the exact nested runtime source URL for one configured run."""
    if not isinstance(run, str) or re.fullmatch(r"[A-Za-z0-9_-]+", run) is None:
        raise ValueError("adapter cache run name is invalid")
    if service_name != f"run_{run}" or re.fullmatch(r"[A-Za-z0-9_-]+", service_name) is None:
        raise ValueError("adapter cache service identity is invalid")
    if not isinstance(source_url, str) or not source_url:
        raise ValueError("adapter cache source URL is invalid")
    base = source_url.rstrip("/")
    parsed = _validated_loopback_url(base)
    if parsed.path != f"/{run}":
        raise ValueError("adapter cache configured source URL does not match the run")
    return f"{base}/{service_name}{SOURCE_SUFFIX}"


def canonical_receipt_source_url(source_url: str, *, target_source_url: str) -> str:
    """Map safe config and legacy aliases to the exact nested target URL."""
    target = exact_source_url(target_source_url)
    target_parts = _validated_loopback_url(target)
    if not target_parts.path.endswith(SOURCE_SUFFIX):
        raise ValueError("adapter cache target source URL is invalid")
    service_path = target_parts.path[: -len(SOURCE_SUFFIX)]
    if not service_path.startswith("/") or not service_path.strip("/"):
        raise ValueError("adapter cache target source URL is invalid")
    if service_path.count("/") >= 2:
        base_path, service_name = service_path.rsplit("/", 1)
    else:
        base_path, service_name = service_path, ""
    parsed = _validated_loopback_url(source_url)
    if parsed.scheme != target_parts.scheme or parsed.netloc != target_parts.netloc:
        raise ValueError("adapter cache source URL does not match the target")
    allowed_paths = {
        target_parts.path,
        base_path,
        base_path + SOURCE_SUFFIX,
    }
    if service_name:
        if re.fullmatch(r"run_[A-Za-z0-9_-]+", service_name) is None:
            raise ValueError("adapter cache target source URL is invalid")
        allowed_paths.add(service_path)
    if parsed.path in allowed_paths:
        return target
    if parsed.path.endswith(SOURCE_SUFFIX):
        return source_url
    if "/broadcasts/" in parsed.path:
        raise ValueError("adapter cache source URL must name the exact step_0 object")
    return source_url + SOURCE_SUFFIX


def validate_metrics(metrics: str) -> None:
    values: dict[str, list[float]] = {"running": [], "waiting": []}
    pattern = re.compile(
        r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+"
        r"([0-9]+(?:\.[0-9]+)?)$"
    )
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)].append(float(match.group(2)))
    if any(not samples for samples in values.values()):
        raise ValueError("adapter cache cleanup request metrics are unknown")
    if any(any(value != 0 for value in samples) for samples in values.values()):
        raise ValueError("adapter cache cleanup requires no running or waiting requests")


def validate_status(
    status: object,
    *,
    source_url: str,
    lora_name: str,
    cache_path: str,
    cache_root: str,
) -> None:
    """Reject active work, loaded receipts, and unknown receipt contents."""
    if not isinstance(status, dict):
        raise ValueError("adapter cache cleanup staging status is not an object")
    state = status.get("status")
    if state not in {"idle", "ready", "failed", "staging"}:
        raise ValueError("adapter cache cleanup staging status is unknown")
    receipts = status.get("loaded_receipts")
    if not isinstance(receipts, dict):
        raise ValueError("adapter cache cleanup loaded receipts are unknown")
    if state == "staging":
        raise ValueError("adapter cache cleanup refused: an adapter stage task is active")
    instances: set[str] = set()
    for name, receipt in receipts.items():
        if not isinstance(name, str) or not isinstance(receipt, dict):
            raise ValueError("adapter cache cleanup loaded receipt is unknown")
        if set(receipt) != RECEIPT_KEYS:
            raise ValueError("adapter cache cleanup loaded receipt contents are unknown")
        if receipt.get("lora_name") != name:
            raise ValueError("adapter cache cleanup loaded receipt name is invalid")
        receipt_source = receipt.get("source_url")
        if not isinstance(receipt_source, str):
            raise ValueError("adapter cache cleanup loaded receipt source is invalid")
        try:
            receipt_source = canonical_receipt_source_url(
                receipt_source,
                target_source_url=source_url,
            )
        except ValueError as error:
            raise ValueError("adapter cache cleanup loaded receipt source is invalid") from error
        for field in ("target_sha256", "config_sha256"):
            if not isinstance(receipt.get(field), str) or re.fullmatch(
                r"[0-9a-f]{64}", receipt[field]
            ) is None:
                raise ValueError("adapter cache cleanup loaded receipt key is invalid")
        if (
            type(receipt.get("target_size")) is not int
            or receipt["target_size"] <= 0
            or type(receipt.get("config_size")) is not int
            or receipt["config_size"] < 0
        ):
            raise ValueError("adapter cache cleanup loaded receipt size is invalid")
        instance = receipt.get("server_instance")
        if not isinstance(instance, str) or re.fullmatch(r"[0-9a-f]{32}", instance) is None:
            raise ValueError("adapter cache cleanup server identity is invalid")
        instances.add(instance)
        receipt_key = hashlib.sha256(receipt_source.encode()).hexdigest()
        if name == lora_name or receipt_key == Path(cache_path).name:
            raise ValueError("adapter cache cleanup refused: exact adapter receipt is still loaded")
    if len(instances) > 1:
        raise ValueError("adapter cache cleanup server identity is inconsistent")
    stage_path = status.get("path")
    stage_source = status.get("source_url")
    if state in {"ready", "failed"} and stage_path is not None:
        if not isinstance(stage_path, str) or not isinstance(stage_source, str):
            raise ValueError("adapter cache cleanup stage identity is unknown")
        if stage_path == cache_path and stage_source != source_url:
            raise ValueError("adapter cache cleanup stage path and source URL differ")


def _validate_transport_manifest(
    target: Path,
    *,
    source_url: str,
    files: dict[str, object],
) -> None:
    """Validate the measured full-transport manifest without its chunks."""
    path = target / "transport-manifest.json"
    try:
        info = path.lstat()
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("adapter cache transport manifest is invalid") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("adapter cache transport manifest permissions are invalid")
    if not isinstance(value, dict) or set(value) != TRANSPORT_MANIFEST_KEYS:
        raise ValueError("adapter cache transport manifest schema is invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("adapter cache transport manifest version is invalid")
    if (
        type(value["resume_identity_version"]) is not int
        or value["resume_identity_version"] != 1
        or value["mode"] != "full"
        or value["transport"] != "chunked-full-zstd"
        or value["source_url"] != source_url
    ):
        raise ValueError("adapter cache transport manifest identity is invalid")
    for name in (
        "base2_config_sha256",
        "base2_config_size",
        "base2_sha256",
        "base2_size",
        "base2_url",
        "base_config_sha256",
        "base_config_size",
        "base_sha256",
        "base_size",
        "base_url",
        "data_offset",
    ):
        if value[name] is not None:
            raise ValueError("adapter cache transport manifest legacy fields are invalid")
    config = files["adapter_config.json"]
    model = files["adapter_model.safetensors"]
    if not isinstance(config, dict) or not isinstance(model, dict):
        raise ValueError("adapter cache transport manifest file identity is invalid")
    for name, record in (("config", config), ("target", model)):
        digest = value[f"{name}_sha256"]
        size = value[f"{name}_size"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
            or digest != record["sha256"]
            or size != record["size"]
        ):
            raise ValueError("adapter cache transport manifest file identity is invalid")
    chunks = value["chunks"]
    if not isinstance(chunks, list) or len(chunks) != TRANSPORT_CHUNK_COUNT:
        raise ValueError("adapter cache transport manifest chunks are invalid")
    expected_offset = 0
    expected_names = {
        f"{TRANSPORT_CHUNK_PREFIX}{index:06d}" for index in range(TRANSPORT_CHUNK_COUNT)
    }
    seen_names: set[str] = set()
    wire_size = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != TRANSPORT_CHUNK_KEYS:
            raise ValueError("adapter cache transport manifest chunk schema is invalid")
        name = chunk["file"]
        if (
            not isinstance(name, str)
            or name != f"{TRANSPORT_CHUNK_PREFIX}{index:06d}"
            or name in seen_names
            or "/" in name
            or "\\" in name
            or not name
        ):
            raise ValueError("adapter cache transport manifest chunk file is invalid")
        seen_names.add(name)
        if (
            chunk["codec"] is not None
            or chunk["data_offset"] is not None
            or type(chunk["offset"]) is not int
            or chunk["offset"] != expected_offset
            or type(chunk["raw_size"]) is not int
            or chunk["raw_size"] <= 0
            or type(chunk["wire_size"]) is not int
            or chunk["wire_size"] <= 0
            or not isinstance(chunk["raw_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", chunk["raw_sha256"]) is None
            or not isinstance(chunk["wire_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", chunk["wire_sha256"]) is None
        ):
            raise ValueError("adapter cache transport manifest chunk identity is invalid")
        wire_size += chunk["wire_size"]
        expected_offset += chunk["raw_size"]
    if type(value["target_size"]) is not int or value["target_size"] <= 0:
        raise ValueError("adapter cache transport manifest target size is invalid")
    if (
        expected_offset != value["target_size"]
        or seen_names != expected_names
        or wire_size != TRANSPORT_WIRE_SIZE
    ):
        raise ValueError("adapter cache transport manifest chunk coverage is invalid")


def validate_directory(
    target: Path,
    *,
    source_url: str,
    expected_key: str,
) -> tuple[int, int]:
    """Validate path, symlink, manifest, key, and contents before deletion.

    The exact source URL hash and manifest bind this directory to one cache
    entry. The sidecar and file sizes bind the expected adapter shape. This
    cleanup never reuses the bytes, so hashing potentially multi-gigabyte
    adapter files would only extend the readers-lock window; any metadata
    disagreement still refuses deletion.
    """
    root = target.parent
    try:
        root_info = root.lstat()
        target_info = target.lstat()
    except OSError as error:
        raise ValueError("adapter cache target is missing") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("adapter cache root is not a real directory")
    if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISDIR(target_info.st_mode):
        raise ValueError("adapter cache target path is not a real directory")
    if target.name != expected_key or hashlib.sha256(source_url.encode()).hexdigest() != expected_key:
        raise ValueError("adapter cache path or key mismatch")
    try:
        if root.resolve(strict=True) != root or target.resolve(strict=True).parent != root:
            raise ValueError("adapter cache target resolved path differs")
    except OSError as error:
        raise ValueError("adapter cache target resolved path is unavailable") from error
    try:
        entries = {entry.name: entry for entry in target.iterdir()}
    except OSError as error:
        raise ValueError("adapter cache target contents are unknown") from error
    if set(entries) - REQUIRED_FILES - OPTIONAL_FILES or not REQUIRED_FILES <= set(entries):
        raise ValueError("adapter cache target contains unknown contents")
    for name, entry in entries.items():
        if entry.is_symlink():
            raise ValueError(f"adapter cache target contains a symbolic link: {name}")
        if name == "transport-chunks":
            if not entry.is_dir():
                raise ValueError("adapter cache transport chunks are invalid")
            for child in entry.rglob("*"):
                if child.is_symlink() or not child.is_file():
                    raise ValueError("adapter cache transport chunks contain unknown contents")
        elif not entry.is_file():
            raise ValueError(f"adapter cache target entry is not regular: {name}")
    try:
        if (target / "READY").read_text(encoding="ascii") != "ready\n":
            raise ValueError("adapter cache ready marker is invalid")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("adapter cache manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("source_url") != source_url:
        raise ValueError("adapter cache manifest source identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("adapter cache manifest file list is invalid")
    for name, record in files.items():
        if not isinstance(record, dict) or type(record.get("size")) is not int or record["size"] < 0 or not isinstance(record.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise ValueError("adapter cache manifest file identity is invalid")
        if name == "adapter_model.safetensors" and record["size"] <= 0:
            raise ValueError("adapter cache weights are empty")
        if (target / name).stat().st_size != record["size"]:
            raise ValueError("adapter cache file size differs from manifest")
    try:
        sidecar = (target / "adapter_model.safetensors.sha256").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("adapter cache key sidecar is invalid") from error
    if sidecar != files["adapter_model.safetensors"]["sha256"]:
        raise ValueError("adapter cache key sidecar differs from manifest")
    for name in ("transport.json", "relay_manifest.json", "adapter_manifest.json"):
        path = target / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"adapter cache {name} is invalid") from error
        if isinstance(value, dict) and "source_url" in value and value["source_url"] != source_url:
            raise ValueError(f"adapter cache {name} source identity differs")
    if "transport-manifest.json" in entries:
        _validate_transport_manifest(target, source_url=source_url, files=files)
    return target_info.st_dev, target_info.st_ino


_REMOTE_PYTHON = r'''import fcntl
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

cache_root, cache_path, source_url, lora_name, container, identity_before, *port_values = sys.argv[1:]
ports = [int(value) for value in port_values]
source_suffix = "/broadcasts/step_0"
receipt_keys = {"source_url", "target_sha256", "target_size", "config_sha256", "config_size", "lora_name", "server_instance"}
required_files = {"READY", "manifest.json", "adapter_config.json", "adapter_model.safetensors", "adapter_model.safetensors.sha256"}
optional_files = {"transport.json", "adapter_manifest.json", "relay_manifest.json", "transport-manifest.json", "transport-chunks"}
transport_manifest_keys = {
    "base2_config_sha256", "base2_config_size", "base2_sha256", "base2_size", "base2_url",
    "base_config_sha256", "base_config_size", "base_sha256", "base_size", "base_url",
    "chunks", "config_sha256", "config_size", "data_offset", "mode",
    "resume_identity_version", "source_url", "target_sha256", "target_size", "transport", "version",
}
transport_chunk_keys = {"codec", "data_offset", "file", "offset", "raw_sha256", "raw_size", "wire_sha256", "wire_size"}
transport_chunk_count = 986
transport_chunk_prefix = "adapter_model.safetensors.transport-full-zstd-"
transport_wire_size = 6382335838
expected_key = hashlib.sha256(source_url.encode()).hexdigest()
target = Path(cache_path)

def exact_source_url(value):
    parsed = _validated_loopback_url(value)
    if parsed.path.endswith(source_suffix):
        return value
    if "/broadcasts/" in parsed.path:
        raise ValueError("adapter cache source URL must name the exact step_0 object")
    return value + source_suffix

def _validated_loopback_url(value):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("adapter cache source URL is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or parsed.path.endswith("/")
    ):
        raise ValueError("adapter cache source URL must be an exact loopback URL")
    return parsed

def canonical_receipt_source_url(value):
    target = exact_source_url(source_url)
    target_parts = _validated_loopback_url(target)
    if not target_parts.path.endswith(source_suffix):
        raise ValueError("adapter cache target source URL is invalid")
    service_path = target_parts.path[:-len(source_suffix)]
    if not service_path.startswith("/") or not service_path.strip("/"):
        raise ValueError("adapter cache target source URL is invalid")
    if service_path.count("/") >= 2:
        base_path, service_name = service_path.rsplit("/", 1)
    else:
        base_path, service_name = service_path, ""
    parsed = _validated_loopback_url(value)
    if parsed.scheme != target_parts.scheme or parsed.netloc != target_parts.netloc:
        raise ValueError("adapter cache source URL does not match the target")
    allowed_paths = {target_parts.path, base_path, base_path + source_suffix}
    if service_name:
        if re.fullmatch(r"run_[A-Za-z0-9_-]+", service_name) is None:
            raise ValueError("adapter cache target source URL is invalid")
        allowed_paths.add(service_path)
    if parsed.path in allowed_paths:
        return target
    if parsed.path.endswith(source_suffix):
        return value
    if "/broadcasts/" in parsed.path:
        raise ValueError("adapter cache source URL must name the exact step_0 object")
    return value + source_suffix

def fetch_json(port, suffix):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{suffix}", timeout=30) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"inference response {suffix} is not an object")
    return value

def validate_status(status):
    state = status.get("status")
    if state not in {"idle", "ready", "failed", "staging"}:
        raise RuntimeError("adapter cache cleanup staging status is unknown")
    receipts = status.get("loaded_receipts")
    if not isinstance(receipts, dict):
        raise RuntimeError("adapter cache cleanup loaded receipts are unknown")
    if state == "staging":
        raise RuntimeError("adapter cache cleanup refused: an adapter stage task is active")
    instances = set()
    for name, receipt in receipts.items():
        if not isinstance(name, str) or not isinstance(receipt, dict) or set(receipt) != receipt_keys:
            raise RuntimeError("adapter cache cleanup loaded receipt contents are unknown")
        if receipt.get("lora_name") != name:
            raise RuntimeError("adapter cache cleanup loaded receipt name is invalid")
        receipt_source = receipt.get("source_url")
        try:
            receipt_source = canonical_receipt_source_url(receipt_source)
        except ValueError as error:
            raise RuntimeError("adapter cache cleanup loaded receipt source is invalid") from error
        if not re.fullmatch(r"[0-9a-f]{64}", receipt.get("target_sha256", "")) or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("config_sha256", "")):
            raise RuntimeError("adapter cache cleanup loaded receipt key is invalid")
        if type(receipt.get("target_size")) is not int or receipt["target_size"] <= 0 or type(receipt.get("config_size")) is not int or receipt["config_size"] < 0:
            raise RuntimeError("adapter cache cleanup loaded receipt size is invalid")
        instance = receipt.get("server_instance")
        if not isinstance(instance, str) or not re.fullmatch(r"[0-9a-f]{32}", instance):
            raise RuntimeError("adapter cache cleanup server identity is invalid")
        instances.add(instance)
        if name == lora_name or hashlib.sha256(receipt_source.encode()).hexdigest() == expected_key:
            raise RuntimeError("adapter cache cleanup refused: exact adapter receipt is still loaded")
    if len(instances) > 1:
        raise RuntimeError("adapter cache cleanup server identity is inconsistent")
    stage_path = status.get("path")
    stage_source = status.get("source_url")
    if state in {"ready", "failed"} and stage_path is not None:
        if not isinstance(stage_path, str) or not isinstance(stage_source, str):
            raise RuntimeError("adapter cache cleanup stage identity is unknown")
        if stage_path == cache_path and stage_source != source_url:
            raise RuntimeError("adapter cache cleanup stage path and source URL differ")

def validate_metrics(metrics):
    values = {"running": [], "waiting": []}
    pattern = re.compile(r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9]+(?:\.[0-9]+)?)$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)].append(float(match.group(2)))
    if any(not samples for samples in values.values()):
        raise RuntimeError("adapter cache cleanup request metrics are unknown")
    if any(any(value != 0 for value in samples) for samples in values.values()):
        raise RuntimeError("adapter cache cleanup requires no running or waiting requests")

def validate_transport_manifest(files):
    path = target / "transport-manifest.json"
    try:
        info = path.lstat()
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("adapter cache transport manifest is invalid") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("adapter cache transport manifest permissions are invalid")
    if not isinstance(value, dict) or set(value) != transport_manifest_keys:
        raise RuntimeError("adapter cache transport manifest schema is invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise RuntimeError("adapter cache transport manifest version is invalid")
    if (
        type(value["resume_identity_version"]) is not int
        or value["resume_identity_version"] != 1
        or value["mode"] != "full"
        or value["transport"] != "chunked-full-zstd"
        or value["source_url"] != source_url
    ):
        raise RuntimeError("adapter cache transport manifest identity is invalid")
    for name in (
        "base2_config_sha256", "base2_config_size", "base2_sha256", "base2_size", "base2_url",
        "base_config_sha256", "base_config_size", "base_sha256", "base_size", "base_url", "data_offset",
    ):
        if value[name] is not None:
            raise RuntimeError("adapter cache transport manifest legacy fields are invalid")
    config = files["adapter_config.json"]
    model = files["adapter_model.safetensors"]
    if not isinstance(config, dict) or not isinstance(model, dict):
        raise RuntimeError("adapter cache transport manifest file identity is invalid")
    for name, record in (("config", config), ("target", model)):
        digest = value[f"{name}_sha256"]
        size = value[f"{name}_size"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
            or digest != record["sha256"]
            or size != record["size"]
        ):
            raise RuntimeError("adapter cache transport manifest file identity is invalid")
    chunks = value["chunks"]
    if not isinstance(chunks, list) or len(chunks) != transport_chunk_count:
        raise RuntimeError("adapter cache transport manifest chunks are invalid")
    expected_offset = 0
    expected_names = {f"{transport_chunk_prefix}{index:06d}" for index in range(transport_chunk_count)}
    seen_names = set()
    wire_size = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != transport_chunk_keys:
            raise RuntimeError("adapter cache transport manifest chunk schema is invalid")
        name = chunk["file"]
        if (
            not isinstance(name, str)
            or name != f"{transport_chunk_prefix}{index:06d}"
            or name in seen_names
            or "/" in name
            or "\\" in name
            or not name
        ):
            raise RuntimeError("adapter cache transport manifest chunk file is invalid")
        if (
            chunk["codec"] is not None
            or chunk["data_offset"] is not None
            or type(chunk["offset"]) is not int
            or chunk["offset"] != expected_offset
            or type(chunk["raw_size"]) is not int
            or chunk["raw_size"] <= 0
            or type(chunk["wire_size"]) is not int
            or chunk["wire_size"] <= 0
            or not isinstance(chunk["raw_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", chunk["raw_sha256"]) is None
            or not isinstance(chunk["wire_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", chunk["wire_sha256"]) is None
        ):
            raise RuntimeError("adapter cache transport manifest chunk identity is invalid")
        wire_size += chunk["wire_size"]
        seen_names.add(name)
        expected_offset += chunk["raw_size"]
    if type(value["target_size"]) is not int or value["target_size"] <= 0:
        raise RuntimeError("adapter cache transport manifest target size is invalid")
    if expected_offset != value["target_size"] or seen_names != expected_names or wire_size != transport_wire_size:
        raise RuntimeError("adapter cache transport manifest chunk coverage is invalid")

def observe():
    observed = []
    for port in ports:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as response:
            if response.status != 200:
                raise RuntimeError("adapter cache cleanup server health is invalid")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
            validate_metrics(response.read().decode("utf-8"))
        observed.append(fetch_json(port, "/staging_status"))
    first = json.dumps(observed[0], sort_keys=True, separators=(",", ":"))
    if any(json.dumps(value, sort_keys=True, separators=(",", ":")) != first for value in observed[1:]):
        raise RuntimeError("adapter cache cleanup inference endpoints disagree")
    validate_status(observed[0])

def validate_cache():
    root = Path(cache_root)
    root_info = root.lstat()
    target_info = target.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError("adapter cache root is not a real directory")
    if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISDIR(target_info.st_mode):
        raise RuntimeError("adapter cache target path is not a real directory")
    if target.name != expected_key or target.resolve(strict=True).parent != root.resolve(strict=True):
        raise RuntimeError("adapter cache path or key mismatch")
    entries = {item.name: item for item in target.iterdir()}
    if set(entries) - required_files - optional_files or not required_files <= set(entries):
        raise RuntimeError("adapter cache target contains unknown contents")
    for name, item in entries.items():
        if item.is_symlink():
            raise RuntimeError(f"adapter cache target contains a symbolic link: {name}")
        if name == "transport-chunks":
            if not item.is_dir():
                raise RuntimeError("adapter cache transport chunks are invalid")
            for child in item.rglob("*"):
                if child.is_symlink() or not child.is_file():
                    raise RuntimeError("adapter cache transport chunks contain unknown contents")
        elif not item.is_file():
            raise RuntimeError(f"adapter cache target entry is not regular: {name}")
    if (target / "READY").read_text(encoding="ascii") != "ready\n":
        raise RuntimeError("adapter cache ready marker is invalid")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("source_url") != source_url:
        raise RuntimeError("adapter cache manifest source identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise RuntimeError("adapter cache manifest file list is invalid")
    # The exact source hash, manifest, sidecar, and sizes identify this cache
    # entry. Do not hash multi-gigabyte weights while holding the readers lock:
    # this operation never reuses the bytes, and metadata disagreement refuses.
    for name, record in files.items():
        if not isinstance(record, dict) or type(record.get("size")) is not int or record["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise RuntimeError("adapter cache manifest file identity is invalid")
        if name == "adapter_model.safetensors" and record["size"] <= 0:
            raise RuntimeError("adapter cache weights are empty")
        if (target / name).stat().st_size != record["size"]:
            raise RuntimeError("adapter cache file size differs from manifest")
    if (target / "adapter_model.safetensors.sha256").read_text(encoding="ascii").strip() != files["adapter_model.safetensors"]["sha256"]:
        raise RuntimeError("adapter cache key sidecar differs from manifest")
    for name in ("transport.json", "relay_manifest.json", "adapter_manifest.json"):
        path = target / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "source_url" in value and value["source_url"] != source_url:
                raise RuntimeError(f"adapter cache {name} source identity differs")
    if "transport-manifest.json" in entries:
        validate_transport_manifest(files)
    return (target_info.st_dev, target_info.st_ino)

def check_identity():
    try:
        result = subprocess.run(
            [
                "sudo",
                "docker",
                "inspect",
                "--format",
                "{{.Name}}|{{.Id}}|{{.State.StartedAt}}|{{.State.Running}}|{{.State.Pid}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("adapter cache cleanup server identity changed") from error
    if result.returncode != 0 or result.stdout.strip() != identity_before:
        raise RuntimeError("adapter cache cleanup server identity changed")

observe()
before = validate_cache()
lock_path = Path(cache_root) / f".{expected_key}.readers.lock"
if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
    raise RuntimeError("adapter cache readers lock is invalid")
with lock_path.open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("adapter cache readers lock is held") from error
    observe()
    if validate_cache() != before:
        raise RuntimeError("adapter cache target changed during validation")
    check_identity()
    shutil.rmtree(target)
    if target.exists() or target.is_symlink():
        raise RuntimeError("adapter cache target was not removed")
print(json.dumps({"status": "removed", "cache_key": expected_key, "cache_path": str(target), "source_url": source_url}))
'''


def build_script(plan: dict[str, object]) -> str:
    """Build the exact one-entry remote removal script."""
    values = [
        plan.get("cache_root"),
        plan.get("cache_path"),
        plan.get("source_url"),
        plan.get("lora_name"),
        plan.get("engine_container"),
    ]
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("adapter cache cleanup plan is invalid")
    ports = plan.get("ports")
    if not isinstance(ports, list) or not ports or not all(isinstance(port, int) and 1 <= port <= 65535 for port in ports):
        raise ValueError("adapter cache cleanup ports are invalid")
    cache_root, cache_path, source_url, lora_name, container = values
    arguments = " ".join(shlex.quote(value) for value in values)
    script = f'''set -euo pipefail
container={shlex.quote(container)}
identity_before=$(sudo docker inspect --format '{{{{.Name}}}}|{{{{.Id}}}}|{{{{.State.StartedAt}}}}|{{{{.State.Running}}}}|{{{{.State.Pid}}}}' "$container" 2>/dev/null) || {{ echo "adapter cache cleanup server identity is unknown" >&2; exit 71; }}
[[ "$identity_before" =~ ^/[-A-Za-z0-9_.]+\\|[0-9a-f]{{64}}\\|[^|]+\\|true\\|[1-9][0-9]*$ ]] || {{ echo "adapter cache cleanup server identity is invalid" >&2; exit 72; }}
[ "$(printf '%s' "$identity_before" | cut -d'|' -f1)" = "/$container" ] || {{ echo "adapter cache cleanup server identity is invalid" >&2; exit 73; }}
container_pid=$(printf '%s' "$identity_before" | awk -F'|' '{{print $5}}')
for port in {" ".join(str(port) for port in ports)}; do
  owner=$(sudo ss -H -ltnp "sport = :$port" 2>/dev/null || true)
  pids=$(printf '%s\\n' "$owner" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
  [ "$(printf '%s\\n' "$pids" | sed '/^$/d' | wc -l)" = 1 ] || {{ echo "adapter cache cleanup listener identity is invalid" >&2; exit 76; }}
  [ "$pids" = "$container_pid" ] || {{ echo "adapter cache cleanup listener identity is invalid" >&2; exit 77; }}
done
sudo python3 - {arguments} "$identity_before" {" ".join(shlex.quote(str(port)) for port in ports)} <<'PY'
{_REMOTE_PYTHON}
PY
identity_after=$(sudo docker inspect --format '{{{{.Name}}}}|{{{{.Id}}}}|{{{{.State.StartedAt}}}}|{{{{.State.Running}}}}|{{{{.State.Pid}}}}' "$container" 2>/dev/null) || {{ echo "adapter cache cleanup server identity changed" >&2; exit 74; }}
[ "$identity_after" = "$identity_before" ] || {{ echo "adapter cache cleanup server identity changed" >&2; exit 75; }}
'''
    return script
