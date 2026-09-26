"""Guarded correction of legacy policy-0 adapter metadata.

This module is side-effect free until the caller explicitly invokes apply.
It only copies an adapter and its published transport artifacts within the
configured trainer storage root.
"""
from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import PurePosixPath
from urllib.parse import quote


LEGACY_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "experts",
)
CANONICAL_TARGETS = tuple(sorted((*LEGACY_TARGETS, "shared_expert")))


class Policy0SourceError(RuntimeError):
    """A policy-0 source correction failed closed."""


def _canonical_step(path_value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(path_value)
    if (
        not path.is_absolute()
        or path.as_posix() != path_value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.name != "step_0"
    ):
        raise Policy0SourceError(f"{label} must be one canonical absolute step_0 path")
    return path


def build_plan(config_dir, source_adapter_dir, destination_adapter_dir, *, repo_root, resolve_config, trainer_head):
    selected = resolve_config(config_dir, require_orchestrator=True)
    source = _canonical_step(source_adapter_dir, "source adapter directory")
    destination = _canonical_step(destination_adapter_dir, "destination adapter directory")
    if source == destination:
        raise Policy0SourceError("source and destination adapter directories must differ")
    try:
        with (repo_root / selected["orchestrator"]).open("rb") as stream:
            orchestrator = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Policy0SourceError(f"cannot read policy-0 source config: {error}") from error
    broadcast = orchestrator.get("weight_broadcast")
    source_url = broadcast.get("source_url") if isinstance(broadcast, dict) else None
    configured_root = broadcast.get("source_dir") if isinstance(broadcast, dict) else None
    if (
        not isinstance(source_url, str)
        or not source_url.startswith(("http://", "https://"))
        or not isinstance(configured_root, str)
    ):
        raise Policy0SourceError("weight_broadcast source URL and source_dir are required")
    source_root = PurePosixPath(configured_root)
    if (
        not source_root.is_absolute()
        or source_root.as_posix() != configured_root
        or any(part in {"", ".", ".."} for part in source_root.parts)
    ):
        raise Policy0SourceError("configured weight_broadcast.source_dir is unsafe")
    try:
        source_relative = source.relative_to(source_root)
        destination_relative = destination.relative_to(source_root)
    except ValueError as error:
        raise Policy0SourceError("source and destination must be below weight_broadcast.source_dir") from error
    if not source_relative.parts or not destination_relative.parts:
        raise Policy0SourceError("source and destination must be strictly below weight_broadcast.source_dir")
    if "broadcasts" not in source.parts[:-1]:
        raise Policy0SourceError("source adapter directory must be the preserved broadcasts step_0")
    if not any("qualification" in part.lower() for part in destination_relative.parts[:-1]):
        raise Policy0SourceError("destination must be a qualification-only path below source_dir")
    source_url = source_url.rstrip("/")
    if not source_url.endswith("/" + source_root.name):
        raise Policy0SourceError("configured source URL does not name its source_dir")
    encoded_source = quote(source_relative.as_posix(), safe="/")
    source_url = source_url + "/" + encoded_source
    destination_url = source_url[: -len(encoded_source)] + quote(
        destination_relative.as_posix(), safe="/"
    )
    plan = {
        "config_dir": config_dir,
        "trainer_head": trainer_head,
        "source_dir": source.as_posix(),
        "source_adapter": (source / "adapter_model.safetensors").as_posix(),
        "source_config": (source / "adapter_config.json").as_posix(),
        "destination_dir": destination.as_posix(),
        "destination_adapter": (destination / "adapter_model.safetensors").as_posix(),
        "destination_config": (destination / "adapter_config.json").as_posix(),
        "configured_source_dir": configured_root,
        "source_url": source_url,
        "destination_url": destination_url,
        "legacy_targets": list(LEGACY_TARGETS),
        "canonical_lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "target_modules": list(CANONICAL_TARGETS),
        },
    }
    plan["apply_script"] = build_apply_script(plan)
    return plan


_SOURCE_APPLY_PYTHON = r'''
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


source_dir, source_adapter, source_config, destination, source_root, destination_url, legacy_json, canonical_json = sys.argv[1:]
source_dir = Path(source_dir)
source_adapter = Path(source_adapter)
source_config = Path(source_config)
destination = Path(destination)
source_root = Path(source_root)
legacy_targets = tuple(json.loads(legacy_json))
canonical_targets = tuple(json.loads(canonical_json))


def fail(message):
    raise SystemExit(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def identity(path):
    stat = path.stat()
    return {
        "size": stat.st_size,
        "sha256": digest(path),
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def regular(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file")
    return identity(path)


def no_symlink_path(root, path, label):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        fail(f"{label} escapes source_dir")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            fail(f"{label} contains a symlink")


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def load_json(path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        fail(f"{label} is invalid: {error}")
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value


def validate_legacy_config(path):
    value = load_json(path, "source adapter_config.json")
    allowed = {
        "peft_type", "task_type", "base_model_name_or_path", "r",
        "lora_alpha", "lora_dropout", "bias", "target_modules",
        "modules_to_save",
    }
    if set(value) - allowed:
        fail("source adapter config has unexpected fields")
    if value.get("peft_type", "LORA") != "LORA" or value.get("task_type", "CAUSAL_LM") != "CAUSAL_LM" or value.get("bias", "none") != "none":
        fail("source adapter config has unexpected PEFT settings")
    if type(value.get("r")) is not int or value["r"] != 32:
        fail("source adapter config rank is not 32")
    alpha = value.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) != 64:
        fail("source adapter config alpha is not numeric 64")
    dropout = value.get("lora_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(float(dropout)) or float(dropout) != 0:
        fail("source adapter config dropout is not zero")
    targets = value.get("target_modules")
    if not isinstance(targets, list) or len(targets) != len(legacy_targets) or set(targets) != set(legacy_targets):
        fail("source adapter config is not the recognized eight-target legacy form")
    if "shared_expert" in targets:
        fail("source adapter config is already canonical")
    return value


def validate_chunk_records(manifest, target_size):
    transport = manifest.get("transport")
    mode = manifest.get("mode")
    raw_chunks = manifest.get("chunks")
    if mode == "full" and transport in (None, "full"):
        if raw_chunks not in (None, []):
            fail("full adapter manifest has unexpected transport chunks")
        return []
    if mode != "full" or transport != "chunked-full-zstd":
        fail("source adapter manifest is not a full step-0 transport")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        fail("source adapter manifest has no transport chunks")
    chunks = []
    seen = set()
    for item in raw_chunks:
        if not isinstance(item, dict):
            fail("source transport chunk record is invalid")
        name = item.get("file")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen
            or name in {"adapter_config.json", "adapter_model.safetensors", "adapter_manifest.json"}
            or type(item.get("offset")) is not int
            or item["offset"] < 0
            or type(item.get("raw_size")) is not int
            or item["raw_size"] <= 0
            or type(item.get("wire_size")) is not int
            or item["wire_size"] <= 0
            or item["offset"] + item["raw_size"] > target_size
            or item.get("codec") != "zstd"
            or not valid_digest(item.get("wire_sha256"))
            or not valid_digest(item.get("raw_sha256"))
        ):
            fail("source transport chunk record is invalid")
        seen.add(name)
        chunks.append(item)
    chunks.sort(key=lambda item: item["offset"])
    cursor = 0
    for item in chunks:
        if item["offset"] != cursor:
            fail("source transport chunks do not cover the adapter exactly")
        cursor += item["raw_size"]
    if cursor != target_size or manifest.get("chunk_count") != len(chunks):
        fail("source transport chunks do not cover the adapter exactly")
    return chunks


def validate_source():
    if not source_root.is_absolute() or not source_dir.is_absolute() or not destination.is_absolute():
        fail("source paths must be absolute")
    no_symlink_path(source_root, source_dir, "source")
    no_symlink_path(source_root, destination.parent, "destination")
    if source_dir.resolve() != source_dir:
        fail("source directory resolves through a symlink")
    if not source_dir.is_dir() or source_dir.is_symlink():
        fail("source adapter directory is not a regular directory")
    if destination.exists() and destination.is_symlink():
        fail("destination adapter directory is a symlink")
    stable = regular(source_dir / "STABLE", "source STABLE")
    for marker in ("WRITING", ".writing", ".lock"):
        if (source_dir / marker).exists():
            fail("source adapter has an active writer marker")
    if any(path.is_file() for path in source_dir.glob("*.partial")):
        fail("source adapter has a partial writer artifact")
    config_value = validate_legacy_config(source_config)
    config_identity = regular(source_config, "source adapter_config.json")
    weight_identity = regular(source_adapter, "source adapter_model.safetensors")
    sidecar_path = source_dir / "adapter_model.safetensors.sha256"
    sidecar_identity = regular(sidecar_path, "source adapter checksum sidecar")
    if sidecar_path.read_text(encoding="ascii").strip() != weight_identity["sha256"]:
        fail("source adapter checksum sidecar differs from the weight")
    manifest_path = source_dir / "adapter_manifest.json"
    manifest_identity = regular(manifest_path, "source adapter manifest")
    manifest = load_json(manifest_path, "source adapter manifest")
    if manifest.get("version") not in (1, 2) or manifest.get("kind") != "lora-adapter" or manifest.get("mode") != "full" or manifest.get("target_step") != 0 or manifest.get("base_step") is not None:
        fail("source adapter manifest is not a full step-0 manifest")
    if manifest.get("full_size") != weight_identity["size"] or manifest.get("full_sha256") != weight_identity["sha256"] or manifest.get("target_size") != weight_identity["size"] or manifest.get("target_sha256") != weight_identity["sha256"]:
        fail("source adapter manifest weight identity differs")
    if not valid_digest(manifest.get("config_sha256")) or manifest.get("config_size") != config_identity["size"] or manifest.get("config_sha256") != config_identity["sha256"]:
        fail("source adapter manifest config identity differs")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        fail("source adapter manifest file list is invalid")
    records = {}
    for item in files:
        if not isinstance(item, dict) or item.get("name") in records:
            fail("source adapter manifest file list is invalid")
        name = item.get("name")
        if name not in {"adapter_config.json", "adapter_model.safetensors"} or type(item.get("size")) is not int or not valid_digest(item.get("sha256")):
            fail("source adapter manifest file record is invalid")
        records[name] = item
    if set(records) != {"adapter_config.json", "adapter_model.safetensors"}:
        fail("source adapter manifest file list is incomplete")
    if records["adapter_config.json"]["size"] != config_identity["size"] or records["adapter_config.json"]["sha256"] != config_identity["sha256"] or records["adapter_model.safetensors"]["size"] != weight_identity["size"] or records["adapter_model.safetensors"]["sha256"] != weight_identity["sha256"]:
        fail("source adapter manifest file identity differs")
    chunks = validate_chunk_records(manifest, weight_identity["size"])
    chunk_identities = {}
    chunk_root = source_dir / "transport-chunks"
    if chunks:
        if chunk_root.is_symlink() or not chunk_root.is_dir():
            fail("source transport chunk directory is invalid")
        for item in chunks:
            path = chunk_root / item["file"]
            record = regular(path, "source transport chunk")
            if record["size"] != item["wire_size"] or record["sha256"] != item["wire_sha256"]:
                fail("source transport chunk identity differs")
            chunk_identities[item["file"]] = record
        actual_chunks = {path.name for path in chunk_root.iterdir()}
        if actual_chunks != set(chunk_identities):
            fail("source transport chunk directory contains unexpected files")
    elif chunk_root.exists():
        if chunk_root.is_symlink() or not chunk_root.is_dir() or any(chunk_root.iterdir()):
            fail("source adapter has unexpected transport chunks")
    expected_entries = {"STABLE", "adapter_config.json", "adapter_model.safetensors", "adapter_model.safetensors.sha256", "adapter_manifest.json"}
    if chunks:
        expected_entries.add("transport-chunks")
    actual_entries = {path.name for path in source_dir.iterdir()}
    if actual_entries != expected_entries:
        fail("source adapter contains unexpected files")
    snapshot = {
        "STABLE": stable,
        "adapter_config.json": config_identity,
        "adapter_model.safetensors": weight_identity,
        "adapter_model.safetensors.sha256": sidecar_identity,
        "adapter_manifest.json": manifest_identity,
        **{f"transport-chunks/{name}": record for name, record in chunk_identities.items()},
    }
    return config_value, manifest, chunks, snapshot


def corrected_config_bytes(source_value):
    value = dict(source_value)
    value["lora_alpha"] = 64
    value["target_modules"] = sorted(canonical_targets)
    return json.dumps(value, indent=2).encode("utf-8")


def corrected_manifest_bytes(source_manifest, config_bytes):
    value = json.loads(json.dumps(source_manifest))
    value["target_url"] = destination_url
    value["config_size"] = len(config_bytes)
    value["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    for item in value["files"]:
        if item["name"] == "adapter_config.json":
            item["size"] = len(config_bytes)
            item["sha256"] = value["config_sha256"]
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


def expected_artifacts(config_value, source_manifest):
    config_bytes = corrected_config_bytes(config_value)
    manifest_bytes = corrected_manifest_bytes(source_manifest, config_bytes)
    return config_bytes, manifest_bytes


def validate_tree(root, config_bytes, manifest_bytes, *, stable):
    if root.is_symlink() or not root.is_dir():
        fail("published adapter root is invalid")
    config_path = root / "adapter_config.json"
    weight_path = root / "adapter_model.safetensors"
    sidecar_path = root / "adapter_model.safetensors.sha256"
    manifest_path = root / "adapter_manifest.json"
    for path, label in ((config_path, "published config"), (weight_path, "published weights"), (sidecar_path, "published checksum sidecar"), (manifest_path, "published manifest")):
        regular(path, label)
    if config_path.read_bytes() != config_bytes or manifest_path.read_bytes() != manifest_bytes:
        fail("published adapter metadata differs from the expected correction")
    published_manifest = load_json(manifest_path, "published adapter manifest")
    if published_manifest.get("target_url") != destination_url:
        fail("published adapter target_url differs")
    if sidecar_path.read_text(encoding="ascii").strip() != published_manifest.get("target_sha256"):
        fail("published adapter checksum sidecar differs")
    if weight_path.stat().st_size != published_manifest.get("target_size") or digest(weight_path) != published_manifest.get("target_sha256"):
        fail("published adapter weight identity differs")
    chunks = validate_chunk_records(published_manifest, weight_path.stat().st_size)
    chunk_root = root / "transport-chunks"
    if chunks:
        if chunk_root.is_symlink() or not chunk_root.is_dir():
            fail("published transport chunk directory is invalid")
        names = set()
        for item in chunks:
            names.add(item["file"])
            path = chunk_root / item["file"]
            record = regular(path, "published transport chunk")
            if record["size"] != item["wire_size"] or record["sha256"] != item["wire_sha256"]:
                fail("published transport chunk identity differs")
        if {path.name for path in chunk_root.iterdir()} != names:
            fail("published transport chunk directory contains unexpected files")
    elif chunk_root.exists():
        fail("published adapter has unexpected transport chunks")
    expected = {"adapter_config.json", "adapter_model.safetensors.sha256", "adapter_model.safetensors", "adapter_manifest.json"}
    if stable:
        expected.add("STABLE")
    if chunks:
        expected.add("transport-chunks")
    if {path.name for path in root.iterdir()} != expected:
        fail("published adapter contains unexpected files")
    if stable:
        regular(root / "STABLE", "published STABLE")
    elif (root / "STABLE").exists():
        fail("temporary adapter published STABLE too early")


def copy_reflink(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "--reflink=auto", "--", str(source), str(target)], check=True)


def rename_noreplace(source, target):
    RENAME_NOREPLACE = 1
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        fail("atomic no-replace directory publish is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(target), RENAME_NOREPLACE)
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            fail("destination appeared during atomic publish")
        fail(f"atomic no-replace directory publish failed: errno {error}")


config_value, source_manifest, chunks, source_snapshot = validate_source()
config_bytes, manifest_bytes = expected_artifacts(config_value, source_manifest)

if destination.exists():
    if destination.is_symlink() or not destination.is_dir():
        fail("destination adapter path conflicts")
    try:
        validate_tree(destination, config_bytes, manifest_bytes, stable=True)
    except SystemExit as error:
        fail(f"destination conflicts: {error}")
    print("SOURCE_RESULT idempotent", source_snapshot["adapter_model.safetensors"]["sha256"], source_snapshot["adapter_model.safetensors"]["size"], hashlib.sha256(config_bytes).hexdigest(), len(config_bytes))
    raise SystemExit(0)

if not destination.parent.exists():
    destination.parent.mkdir(parents=True, exist_ok=True)
no_symlink_path(source_root, destination.parent, "destination")
if destination.exists():
    fail("destination appeared before staging")
source_after = validate_source()[3]
if source_after != source_snapshot:
    fail("source changed during validation")

temporary = Path(tempfile.mkdtemp(prefix=".policy0-source.", dir=destination.parent))
try:
    copy_reflink(source_adapter, temporary / "adapter_model.safetensors")
    copy_reflink(source_dir / "adapter_model.safetensors.sha256", temporary / "adapter_model.safetensors.sha256")
    for item in chunks:
        copy_reflink(source_dir / "transport-chunks" / item["file"], temporary / "transport-chunks" / item["file"])
    (temporary / "adapter_config.json").write_bytes(config_bytes)
    (temporary / "adapter_manifest.json").write_bytes(manifest_bytes)
    validate_tree(temporary, config_bytes, manifest_bytes, stable=False)
    if validate_source()[3] != source_snapshot:
        fail("source changed while creating corrected adapter")
    stable_path = temporary / "STABLE"
    with stable_path.open("wb") as stream:
        stream.write(b"")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    validate_tree(temporary, config_bytes, manifest_bytes, stable=True)
    rename_noreplace(temporary, destination)
    temporary = None
    validate_tree(destination, config_bytes, manifest_bytes, stable=True)
except BaseException:
    if temporary is not None:
        shutil.rmtree(temporary, ignore_errors=True)
    raise

print("SOURCE_RESULT created", source_snapshot["adapter_model.safetensors"]["sha256"], source_snapshot["adapter_model.safetensors"]["size"], hashlib.sha256(config_bytes).hexdigest(), len(config_bytes))
'''


def build_apply_script(plan):
    arguments = [
        str(plan["source_dir"]),
        str(plan["source_adapter"]),
        str(plan["source_config"]),
        str(plan["destination_dir"]),
        str(plan["configured_source_dir"]),
        str(plan["destination_url"]),
        json.dumps(list(LEGACY_TARGETS), separators=(",", ":")),
        json.dumps(list(CANONICAL_TARGETS), separators=(",", ":")),
    ]
    return (
        "set -euo pipefail\n"
        "python3 - "
        + " ".join(shlex.quote(value) for value in arguments)
        + " <<'PY'\n"
        + _SOURCE_APPLY_PYTHON
        + "\nPY\n"
    )


def parse_result(output, plan):
    records = [line.split() for line in output.splitlines() if line.startswith("SOURCE_RESULT ")]
    if len(records) != 1 or len(records[0]) != 6 or records[0][1] not in {"created", "idempotent"}:
        raise Policy0SourceError("policy-0 source preparation returned no valid identity")
    _, status, adapter_sha, adapter_size, config_sha, config_size = records[0]
    if (
        not re.fullmatch(r"[0-9a-f]{64}", adapter_sha)
        or not re.fullmatch(r"[0-9a-f]{64}", config_sha)
        or not adapter_size.isdigit()
        or not config_size.isdigit()
    ):
        raise Policy0SourceError("policy-0 source preparation returned invalid identities")
    return {
        "status": status,
        "source_url": plan["source_url"],
        "destination_url": plan["destination_url"],
        "adapter_identity": {"sha256": adapter_sha, "size": int(adapter_size)},
        "config_identity": {"sha256": config_sha, "size": int(config_size)},
    }
